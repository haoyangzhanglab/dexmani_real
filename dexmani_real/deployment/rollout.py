"""单段策略运行；持有结果与 recorder，由 Session 编排资源收尾。"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Literal, cast

import numpy as np

from dexmani_real.control.ik import IKReferenceError
from dexmani_real.data.raw import episode_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.inference import POLICY_STARTUP_TIMEOUT_S
from dexmani_real.deployment.schedule import (
    ActionChunk,
    Prediction,
    control_slot,
)
from dexmani_real.geometry.pose import EEF_POSE_DIM, Pose, rot6d_to_quat_wxyz
from dexmani_real.ipc.sensors import read_camera_frame
from dexmani_real.robot.commands import (
    COMMAND_LEASE_NS,
    FeedbackUnavailable,
    HandCommandUnconfirmed,
    HandTargetRejected,
    OperationCancelled,
    RobotCommand,
    arm_reference_from_feedback,
)
from dexmani_real.robot.model import ARM_DOF
from dexmani_real.robot.xhand.feedback import expire_tactile
from dexmani_real.runtime.ik_reference import IKReference, prepare_reference
from dexmani_real.runtime.observation import (
    RAW_RECORDING_NEEDS,
    Observation,
    check_device_feedback,
    check_hoi_status,
    check_observation,
    is_recoverable_hoi_failure,
    read_observation,
)
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S, RunEndReason
from dexmani_real.utils.log import error_detail

logger = logging.getLogger(__name__)

HOI_HOLD_REFRESH_S = 1.0
HOI_REQUEST_LOCK_TIMEOUT_S = 0.10


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EpisodeResult:
    index: int
    started_utc: str | None = None
    duration_s: float = 0.0
    termination_reason: str | None = None
    termination_details: str | None = None
    errors: list[dict] = field(default_factory=list)
    raw: dict | None = None
    skipped_slots: int = 0
    discard_requested: bool = False


class Rollout:
    def __init__(
        self, runtime_state, runtime, policy_info, *, episode_cancel_epoch: int, robot,
        observations,
        sensors, action_mapper, model, max_duration_s: float | None,
        index, output_dir, record_raw, effective_parameters, camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.policy_info = policy_info
        self.execution = runtime.execution
        self.robot, self.sensors = robot, sensors
        model_needs = observations.needs
        self.observations = observations
        # 派生 Ring 只提供模型图像，Raw RGB-D 由相机 Ring 提供。
        self.read_raw_camera = record_raw and (model_needs.hoi or model_needs.point_cloud)
        self.read_needs = RAW_RECORDING_NEEDS if record_raw and not self.read_raw_camera else model_needs
        self.action_mapper, self.model = action_mapper, model
        self.recorder: EpisodeRecorder | None = None
        self.output_dir, self.record_raw = output_dir, record_raw
        self.result = EpisodeResult(index)
        self.camera_snapshot = camera_snapshot
        self.effective_parameters = effective_parameters
        self.max_duration_ns = int(max_duration_s * 1e9) if max_duration_s is not None else None
        self.episode_cancel_epoch = episode_cancel_epoch
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=policy_info.n_obs_steps)
        self.plan: ActionChunk | None = None
        self.query: Prediction | None = None
        self.cursor = 0
        self.preparation_issue: str | None = None
        self.last_slot = -1
        self.dt_ns = round(self.execution.control_dt_s * 1e9)
        self.started_ns = 0
        self.ik_reference: IKReference | None = None
        self.reset_pending = True
        self.prev_action: np.ndarray | None = None
        self.state: Literal["RUNNING", "HOI_HOLD", "RESUMING"] = "RUNNING"
        self.next_hoi_hold_ns = 0

    @property
    def next_step_ns(self):
        return self.started_ns + (self.last_slot + 1) * self.dt_ns

    @property
    def finished(self):
        return self.result.termination_reason is not None

    def recording_issue(self, row):
        issues = check_observation(
            row, self.runtime, RAW_RECORDING_NEEDS, now_ns=row.observation_timestamp_ns,
        )
        return "; ".join(issues) if issues else None

    def _request_hoi(self, episode_id: int) -> None:
        channels = self.sensors.channels
        lock = channels.hoi_request_lock
        # 子进程被强行终止时可能遗留发布锁，控制和收尾只能有界等待。
        if not lock.acquire(timeout=HOI_REQUEST_LOCK_TIMEOUT_S):
            raise TimeoutError("HOI request lock timed out")
        try:
            channels.hoi_request.write(
                episode_id=episode_id, after_ns=time.monotonic_ns(),
                after_sequence=channels.camera_ring.latest_sequence,
            )
        finally:
            lock.release()

    def stop(self, detail, *, run_end_reason=RunEndReason.CANCELLED):
        if self.finished:
            return
        ended_ns = time.monotonic_ns()
        self.runtime_state.stop_signal.set()
        self.result.duration_s = (ended_ns - self.started_ns) / 1e9 if self.started_ns else 0.0
        self.result.termination_reason = (
            run_end_reason.name.lower() if self.started_ns
            else "preparation_failed" if self.preparation_issue or run_end_reason is RunEndReason.FAILED
            else "start_cancelled"
        )
        self.result.termination_details = detail
        self._clear_policy_context()
        if self.observations.needs.hoi:
            self._request_hoi(0)

    def snapshot_result(self) -> EpisodeResult:
        """活动 Recorder 提供当前快照，释放后保留最后的完整结果。"""
        if self.recorder is not None:
            self.result.raw = self.recorder.result()
        return self.result

    def finish_recording(self, *, wait=True) -> bool:
        """True 仅表示收尾完成；录制成败保留在本段 result 中。"""
        result = self.result
        if self.recorder is not None:
            recorder = self.recorder
            if recorder.finish(
                save=not result.discard_requested, reason=result.termination_reason, wait=wait,
            ) is None:
                return False
            self.snapshot_result()
            # 线程实际退出才释放 owner；失败结果及 staging 路径保留在 result。
            self.recorder = None
            if recorder.error is not None:
                self.fail("recording", recorder.error)
        return True

    def fail(self, stage, error):
        if isinstance(error, KeyboardInterrupt):
            self.runtime_state.request_emergency_stop()
        self.result.errors.append(error_detail(stage, error))
        self.stop(None, run_end_reason=RunEndReason.FAILED)

    def run(self, *, poll_operator):
        """运行结束先停设备；段间及最终资源收尾由 Session 编排。"""
        try:
            try:
                self.prepare(poll_operator)
                while not self.finished:
                    due_ns = min(self.next_step_ns, self.robot.feedback_next_ns)
                    timeout = min(0.005, max(0, due_ns - time.monotonic_ns()) / 1e9)
                    poll_operator(timeout=timeout)
                    self.robot.service_feedback()
                    self.step()
            except RecordingError:
                self.stop("recording_failed", run_end_reason=RunEndReason.FAILED)
                self.robot.check()
                poll_operator()
            except OperationCancelled as exc:
                # 取消不能掩盖同时发生的设备或 worker 故障。
                self.robot.check()
                poll_operator()
                if (not isinstance(exc, FeedbackUnavailable) and not self.started_ns
                        and self.runtime_state.cancel_epoch == self.episode_cancel_epoch
                        and self.runtime_state.requests_allow_motion()):
                    raise
                self.stop(str(exc))
        except BaseException as exc:
            self.fail("control" if self.started_ns else "preparation", exc)
            raise
        finally:
            primary_error = sys.exc_info()[1]
            try:
                self.robot.stop()
            except BaseException as exc:
                self.fail("stop", exc)
                if primary_error is None:
                    raise
                logger.exception("device stop also failed")
        return self.result

    def preparation_valid(self):
        valid = (
            self.runtime_state.cancel_epoch == self.episode_cancel_epoch
            and self.runtime_state.requests_allow_motion()
        )
        if not valid:
            self.robot.check()
        return valid

    def read_observations(
        self, *, include_visual_payloads=True, include_raw_payloads=True,
    ) -> tuple[Observation, Observation]:
        """共享关节采样；模型使用派生观测时，Raw 独立读取相机。"""
        row = read_observation(
            self.sensors.channels,
            self.robot,
            needs=self.read_needs,
            include_visual_payloads=include_visual_payloads,
        )
        raw = row
        if self.read_raw_camera:
            camera_fields = (
                RAW_RECORDING_NEEDS.camera_fields if include_visual_payloads and include_raw_payloads else ()
            )
            camera = read_camera_frame(self.sensors.channels, fields=camera_fields)
            observed_ns = time.monotonic_ns()
            hand = expire_tactile(row.hand, max_age_s=self.runtime.hand.tactile_max_age_s, now_ns=observed_ns)
            raw = replace(
                row, hand=hand, camera=camera, point_cloud=None, pointcloud_timestamp_ns=None,
                hoi=None, observation_timestamp_ns=observed_ns,
            )
        return row, raw

    def start_observation(self, poll_operator, *, after_ns=None, initial_hoi=False, warmup_payloads=False):
        timeout_s = 120.0 if initial_hoi else INITIAL_STATE_TIMEOUT_S
        until = time.monotonic_ns() + int(timeout_s * 1e9)
        include_visual_payloads = False
        while True:
            poll_operator()
            if not self.preparation_valid():
                raise OperationCancelled("initial observation cancelled")
            self.robot.service_feedback()
            row, raw = self.read_observations(
                include_visual_payloads=include_visual_payloads, include_raw_payloads=False,
            )
            devices = check_device_feedback(row, self.runtime)
            model_issues = check_observation(row, self.runtime, self.observations.needs)
            issue = ("device feedback: " + "; ".join(devices)) if devices else (
                "model_input_" + model_issues[0] if model_issues else None
            )
            if self.record_raw:
                recording_issue = self.recording_issue(raw)
                issue = issue or recording_issue
            joints_ready = after_ns is None or (
                int(row.arm["timestamp_ns"][0]) >= after_ns
                and row.hand is not None and int(row.hand["timestamp_ns"][0]) >= after_ns
            )
            ready = issue is None and joints_ready and not self.robot.hand_feedback_pending
            if time.monotonic_ns() >= until:
                self.preparation_issue = issue or (
                    "initial_observation_timeout" if ready else "initial_joint_feedback_unavailable"
                )
                logger.warning("本轮未开始：%s；等待停止与收尾。", self.preparation_issue)
                return None
            if ready:
                if warmup_payloads and not include_visual_payloads:
                    # 预热要用模型载荷，重读后仍按本轮截止时间与来源检查准入。
                    include_visual_payloads = True
                    continue
                self.preparation_issue = None
                break
            include_visual_payloads = False
            time.sleep(0.005)
        if self.policy_info.action_mode == "eef":
            try:
                self.action_mapper.solver.measured_reference(row.arm["qpos"][0])
            except IKReferenceError as exc:
                self.preparation_issue = str(exc)
                logger.warning("本轮未开始：%s；等待停止与收尾后检查起始姿态。", exc)
                return None
        else:
            arm_reference_from_feedback(
                row.arm["qpos"][0], self.runtime.arm.joint_limit_lower, self.runtime.arm.joint_limit_upper,
            )
        return row

    def prepare(self, poll_operator):
        logger.info("Policy preparing")
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before preparation")
        self.robot.stop()
        if self.observations.needs.hoi:
            self._request_hoi(self.result.index)
        # 先检查各来源就绪；只有 HOI 预热会消费这次模型载荷。
        initial_hoi = self.observations.needs.hoi
        warmup_needed = initial_hoi and not self.model.hoi_warmed_up
        row = self.start_observation(poll_operator, initial_hoi=initial_hoi, warmup_payloads=warmup_needed)
        if row is None:
            self.stop(self.preparation_issue)
            return
        if warmup_needed:
            kwargs = ({"rtc_delay": self.execution.prefetch_steps, "rtc_guidance_cap": self.execution.rtc_guidance_cap}
                      if self.execution.mode == "rtc" else {})
            self.model.warmup(
                (self.observations.project(row),) * self.policy_info.n_obs_steps,
                **kwargs,
            )
            warmup_deadline = time.monotonic() + POLICY_STARTUP_TIMEOUT_S
            while True:
                if time.monotonic() >= warmup_deadline:
                    raise TimeoutError("HOI policy warmup timeout")
                poll_operator()
                self.robot.service_feedback()
                if not self.preparation_valid():
                    raise OperationCancelled("HOI warmup cancelled")
                future = self.model.poll()
                if future is not None:
                    future.result()
                    break
                time.sleep(0.005)
            poll_operator()
            if not self.preparation_valid():
                raise OperationCancelled("HOI warmup cancelled")
            if time.monotonic() >= warmup_deadline:
                raise TimeoutError("HOI policy warmup timeout")
            self.model.hoi_warmed_up = True
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before recorder start")
        if self.record_raw:
            self.recorder = EpisodeRecorder(
                self.output_dir, control_hz=1 / self.execution.control_dt_s,
                read_xhand_crc=self.robot.hand.crc_counts.copy,
            )
            metadata = episode_recording_metadata(
                self.camera_snapshot, self.effective_parameters,
                tactile_reset=self.robot.hand.tactile_reset_metadata,
                tactile_calibration=self.robot.hand.tactile_calibration_metadata,
            )
            self.recorder.start(
                task_label=self.policy_info.task_name,
                episode_index=self.result.index, collection_source="policy_rollout", metadata=metadata,
            )
            # writer 初始化期间设备保持停止，准备运动前重新等候新反馈。
            if self.start_observation(poll_operator, after_ns=time.monotonic_ns()) is None:
                self.stop(self.preparation_issue)
                return
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before streaming preparation")
        self.robot.prepare_streaming(self.episode_cancel_epoch, poll_operator=poll_operator)
        if self.start_observation(poll_operator) is None:
            self.stop(self.preparation_issue)
            return
        with self.runtime_state.motion_lock:
            started_ns = (
                time.monotonic_ns()
                if self.runtime_state.cancel_epoch == self.episode_cancel_epoch
                and self.runtime_state.requests_allow_motion()
                and not self.runtime_state.stop_signal.is_set()
                else None
            )
        if started_ns is None:
            self.robot.check()
            raise OperationCancelled("final admission declined")
        self.started_ns = started_ns
        self.result.started_utc = utc_now()
        logger.info("Policy control loop started; waiting for the first prediction")

    def poll_model(self) -> None:
        future = self.model.poll()
        if future is None:
            return
        actions = future.result()
        # 已清除的请求仍需传播推理异常。
        if self.query is not None:
            self.query.result = actions

    def _submit_prediction(self, slot: int, *, handoff_slot=None, prefix=None) -> bool:
        kwargs = {} if prefix is None else {
            "rtc_prefix": prefix,
            "delay_steps": min(self.execution.prefetch_steps, len(prefix)),
            "rtc_guidance_cap": self.execution.rtc_guidance_cap,
        }
        if self.policy_info.requires_prev_action:
            kwargs["prev_action"] = self.prev_action
        rows = tuple(self.history)
        # sync 和无前缀 RTC 的首动作在下一槽。
        first_action_offset = int(self.execution.mode != "async" and prefix is None)
        deadline_ns = self.started_ns + (
            slot + first_action_offset + self.policy_info.future_steps
        ) * self.dt_ns
        self.check_motion()
        if time.monotonic_ns() >= deadline_ns:
            return False
        self.model.predict(rows, reset_episode=self.reset_pending, **kwargs)
        self.reset_pending = False
        self.query = Prediction(
            slot=slot,
            handoff_slot=slot + first_action_offset if handoff_slot is None else handoff_slot,
            deadline_ns=deadline_ns,
            request_cursor=self.cursor,
        )
        return True

    def _adopt_prediction(self, slot: int, now_ns: int) -> None:
        query = self.query
        if (
            query is None or query.result is None
            or slot < query.handoff_slot or now_ns >= query.deadline_ns
        ):
            return
        mode, plan = self.execution.mode, self.plan
        if mode == "async":
            offset = slot - query.slot
        elif mode == "rtc":
            if (plan is not None and now_ns < plan.deadline_ns
                    and self.cursor < min(self.execution.action_steps, len(plan.actions))):
                return
            offset = self.cursor - query.request_cursor
        else:
            offset = 0
        self.plan = ActionChunk(query.result[offset:], slot, query.deadline_ns)
        self.query = None
        self.cursor = 0

    def _expire_plan(self, now_ns: int) -> bool:
        plan = self.plan
        if plan is None or now_ns < plan.deadline_ns:
            return False
        # 保留仍有效的接续，并让下一控制槽选动作。
        self.poll_model()
        self._adopt_prediction((now_ns - self.started_ns) // self.dt_ns, now_ns)
        if self.plan is plan:
            self.stop("prediction_expired")
        return True

    def select_action(self, slot: int) -> np.ndarray | None:
        now = time.monotonic_ns()
        self._adopt_prediction(slot, now)
        if self._expire_plan(now):
            return None
        plan = self.plan
        query = self.query
        if plan is None:
            expired = not self._submit_prediction(slot) if query is None else now >= query.deadline_ns
            if expired:
                self.stop("prediction_expired")
            return None
        mode = self.execution.mode
        index = slot - plan.anchor_slot if mode == "async" else self.cursor
        limit = min(self.execution.action_steps, len(plan.actions))
        if index >= limit:
            if query is None:
                self.stop("prediction exhausted")
            return None
        if query is None and mode != "sync" and index >= limit - self.execution.prefetch_steps:
            # 预取未入队时，仍由当前计划决定本槽动作。
            if mode == "rtc":
                self._submit_prediction(slot, prefix=plan.actions[index:].copy())
            else:
                self._submit_prediction(slot, handoff_slot=plan.anchor_slot + limit)
        return plan.actions[index]

    def check_motion(self):
        if (self.runtime_state.cancel_epoch != self.episode_cancel_epoch
                or not self.runtime_state.requests_allow_motion()
                or self.runtime_state.stop_signal.is_set()):
            self.robot.check()
            raise OperationCancelled("operation cancelled")

    def sample_observation(self, *, include_visual_payloads=True) -> tuple[Observation, Observation]:
        row, raw = self.read_observations(include_visual_payloads=include_visual_payloads)
        devices = check_device_feedback(row, self.runtime)
        if devices:
            raise FeedbackUnavailable("device feedback: " + "; ".join(devices))
        return row, raw

    def update_history(self, row):
        """收集有效控制采样，历史不足时重复段首。"""
        issues = check_observation(row, self.runtime, self.observations.needs)
        if issues:
            self._observation_failure(issues, "model_input")
            return False
        projected = self.observations.project(row)
        if not self.history:
            self.history.extend([projected] * self.policy_info.n_obs_steps)
        else:
            self.history.append(projected)
        return True

    def duration_expired(self):
        if (
            self.started_ns
            and not self.finished
            and self.max_duration_ns is not None
            and time.monotonic_ns() >= self.started_ns + self.max_duration_ns
        ):
            self.stop("duration_timeout", run_end_reason=RunEndReason.DURATION_COMPLETE)
            return True
        return False

    def prepare_command(self, action, row) -> tuple[RobotCommand, np.ndarray | None] | None:
        anchor = None
        try:
            if self.policy_info.action_mode == "joint":
                decoded = self.action_mapper.map_joint(action[:ARM_DOF], action[ARM_DOF:])
            else:
                pose = Pose(action[:3], rot6d_to_quat_wxyz(action[3:EEF_POSE_DIM]))
                reference = prepare_reference(
                    self.action_mapper.solver, self.ik_reference, row.arm["qpos"][0],
                    control_ns=time.monotonic_ns(),
                    period_s=self.execution.control_dt_s,
                )
                _, _, anchor, _ = reference
                decoded = self.action_mapper.map_eef(pose, action[EEF_POSE_DIM:], reference)
        except (IKReferenceError, HandTargetRejected) as exc:
            self.stop(str(exc))
            return None
        if decoded.arm_qpos is None:
            self.stop(f"action_rejected: {decoded.rejection_reason}")
            return None
        command = RobotCommand(decoded.arm_qpos, decoded.hand_qpos)
        self.check_motion()
        if self.duration_expired():
            return None
        issues = check_device_feedback(row, self.runtime)
        if issues:
            raise FeedbackUnavailable("device feedback: " + "; ".join(issues))
        issues = check_observation(row, self.runtime, self.observations.needs)
        if issues:
            self._observation_failure(issues, "model_input before submission")
            return None
        if self.observations.needs.hoi:
            issues = check_hoi_status(self.sensors.channels, self.runtime)
            if issues:
                self._enter_hoi_hold()
                return None
        return command, anchor

    def _observation_failure(self, issues, context):
        if is_recoverable_hoi_failure(issues):
            self._enter_hoi_hold()
        else:
            self.stop(context + ": " + "; ".join(issues))

    def _clear_policy_context(self):
        """清除控制侧上下文，等在途推理回收后再重置模型。"""
        self.history.clear()
        self.plan = self.query = None
        self.cursor = 0
        self.prev_action = None
        self.ik_reference = None
        self.reset_pending = True

    def _enter_hoi_hold(self):
        if self.state == "HOI_HOLD":
            return
        # 首条恢复动作提交前的再次遮挡仍属于同一个恢复区间。
        if self.state == "RUNNING" and self.recorder is not None:
            self.recorder.mark_gap()
        self.state = "HOI_HOLD"
        self._clear_policy_context()
        self.next_hoi_hold_ns = 0
        self._refresh_hoi_hold()

    def _refresh_hoi_hold(self):
        if time.monotonic_ns() < self.next_hoi_hold_ns:
            return
        self.check_motion()
        state = self.robot.read_state()
        now = time.monotonic_ns()
        issues = check_device_feedback(state, self.runtime, now_ns=now)
        if issues:
            raise FeedbackUnavailable("device feedback: " + "; ".join(issues))
        self.robot.send_action(RobotCommand(arm_hold=True), expires_ns=now + COMMAND_LEASE_NS)
        if self.robot.hand_feedback_pending:
            return
        if not self.robot.hold_hand(check_abort=self.check_motion):
            if self.robot.hand_feedback_pending:
                return
            raise FeedbackUnavailable("HOI HOLD XHand position hold not confirmed")
        self.next_hoi_hold_ns = now + int(HOI_HOLD_REFRESH_S * 1e9)

    def step(self):
        """由 run 在反馈服务与健康检查成功后调用。"""
        if self.finished:
            return
        self.check_motion()
        if self.recorder is not None:
            self.recorder.check_error()
        if self.duration_expired():
            return
        if self.observations.needs.hoi and self.state != "HOI_HOLD":
            issues = check_hoi_status(self.sensors.channels, self.runtime)
            if issues:
                self._enter_hoi_hold()
        if self.state != "RUNNING" and not self.robot.hand_feedback_pending:
            self._refresh_hoi_hold()
        if self.robot.hand_feedback_pending:
            slot, _ = self._advance_control_slot()
            if slot is not None:
                self._clear_policy_context()
                if self.recorder is not None:
                    self.recorder.mark_gap()
            self.poll_model()
            return
        if self.state == "HOI_HOLD":
            # 即使旧推理未完成，也不能掩盖设备或其他输入故障。
            # 暂停不构建模型历史或追加 Raw，仅读取视觉来源元数据检查时效。
            row, raw = self.sample_observation(include_visual_payloads=False)
            quality_issue = self.recording_issue(raw) if self.recorder is not None else None
            if quality_issue is not None:
                self.stop(quality_issue)
                return
            issues = check_observation(row, self.runtime, self.observations.needs)
            if issues and not is_recoverable_hoi_failure(issues):
                self.stop("model_input: " + "; ".join(issues))
                return
            self.poll_model()
            self._advance_control_slot()
            if not issues and self.model.future is None:
                self.state = "RESUMING"
            return
        self.step_control()

    def _advance_control_slot(self):
        """暂停时也推进原调度，恢复后不补跑历史控制槽。"""
        slot, skipped = control_slot(
            now_ns=time.monotonic_ns(), next_step_ns=self.next_step_ns,
            last_slot=self.last_slot, dt_ns=self.dt_ns,
        )
        self.result.skipped_slots += skipped
        if slot is not None:
            self.last_slot = slot
        return slot, skipped

    def step_control(self):
        slot, skipped = self._advance_control_slot()
        if slot is None:
            self.poll_model()
            return
        was_running = self.state == "RUNNING"
        ode_gap = skipped and self.policy_info.requires_prev_action
        if ode_gap:
            self._clear_policy_context()
            if was_running and self.recorder is not None:
                self.recorder.mark_gap()
        if self.reset_pending and self.model.future is not None:
            self.poll_model()
            if self.model.future is not None:
                # 新段从首个可提交查询的观测开始，与离线起点一致。
                return
        row, raw = self.sample_observation()
        quality_issue = self.recording_issue(raw) if self.recorder is not None else None
        submitted_command = None
        try:
            if quality_issue is not None:
                self.stop(quality_issue)
                return
            if not self.update_history(row):
                return
            self.check_motion()
            if self.duration_expired():
                return
            self.poll_model()
            # 本 tick 固定选中的目标，映射后重查同一预测窗口。
            action = self.select_action(slot)
            if action is None:
                return
            plan = cast(ActionChunk, self.plan)
            prepared = self.prepare_command(action, row)
            if prepared is None:
                return
            command, anchor = prepared
            now = time.monotonic_ns()
            if self._expire_plan(now):
                return
            arm_published_ns = self.robot.send_action(command, expires_ns=now + COMMAND_LEASE_NS)
            submitted_command = command
            if self.duration_expired():
                return
            if self.policy_info.requires_prev_action:
                self.prev_action = np.concatenate(
                    (cast(np.ndarray, command.arm_qpos), cast(np.ndarray, command.hand_qpos)),
                )
            if anchor is not None:
                self.ik_reference = IKReference(
                    cast(np.ndarray, command.arm_qpos), anchor, cast(int, arm_published_ns),
                )
            if self.execution.mode in {"sync", "rtc"}:
                self.cursor += 1
                if self.execution.mode == "sync" and self.cursor == min(len(plan.actions), self.execution.action_steps):
                    self.plan = None
            if self.state == "RESUMING":
                self.state = "RUNNING"
        except HandCommandUnconfirmed as exc:
            self.stop(str(exc))
        finally:
            primary_error = sys.exc_info()[1]
            if primary_error is not None:
                self.runtime_state.stop_signal.set()
            still_running = (
                self.state == "RUNNING" and not self.finished and not self.runtime_state.stop_signal.is_set()
            )
            if self.recorder is not None and (
                submitted_command is not None or quality_issue is not None or still_running
            ):
                try:
                    if skipped and not ode_gap and was_running:
                        self.recorder.mark_gap()
                    issue = self.recorder.append(raw, submitted_command, quality_issue=quality_issue)
                    if issue is not None:
                        self.stop(issue)
                except BaseException as exc:
                    if primary_error is None:
                        raise
                    if exc is not primary_error:
                        self.result.errors.append(error_detail("recording_append", exc))
                        if isinstance(exc, KeyboardInterrupt):
                            self.runtime_state.request_emergency_stop()
                        logger.exception("recording also failed during control failure")
