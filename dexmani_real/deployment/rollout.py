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
from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.schedule import (
    ActionChunk,
    Prediction,
    control_slot,
    handoff_plan,
    plan_exhausted,
    prefetch_handoff,
)
from dexmani_real.geometry.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.ipc.sensors import read_camera_frame
from dexmani_real.robot.commands import (
    COMMAND_LEASE_NS,
    FeedbackUnavailable,
    OperationCancelled,
    RobotCommand,
    arm_reference_from_feedback,
)
from dexmani_real.runtime.ik_reference import IKReference, prepare_reference
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_device_feedback,
    check_hoi_status,
    check_observation,
    is_recoverable_hoi_failure,
    read_observation,
)
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S, RunEndReason
from dexmani_real.utils.log import error_detail

logger = logging.getLogger(__name__)

HOI_RECOVERY_TIMEOUT_S = 5.0
HOI_HOLD_REFRESH_S = 1.0


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


class Rollout:
    def __init__(
        self, runtime_state, runtime, policy_info, *, cancel_epoch, robot,
        observations, read_needs: ObservationNeeds, record_needs: ObservationNeeds,
        sensors, action_mapper, model, max_running_s,
        index, output_dir, record_raw, camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.policy_info = policy_info
        self.execution = runtime.execution
        self.robot, self.sensors = robot, sensors
        if observations.needs.hoi_joint_devices:
            robot.enable_joint_history(
                self.execution.hoi_max_age_s + self.execution.hoi_max_joint_gap_s,
                devices=observations.needs.hoi_joint_devices,
            )
        self.observations = observations
        self.read_needs = read_needs
        self.record_needs = record_needs
        self.action_mapper, self.model = action_mapper, model
        self.recorder: EpisodeRecorder | None = None
        self.output_dir, self.record_raw = output_dir, record_raw
        self.result = EpisodeResult(index)
        self.camera_snapshot = camera_snapshot
        self.max_duration_ns = int(max_running_s * 1e9) if max_running_s is not None else None
        self.cancel_epoch = cancel_epoch
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=policy_info.n_obs_steps)
        self.plan: ActionChunk | None = None
        self.query: Prediction | None = None
        self.cursor = 0
        self.preparation_issue: str | None = None
        self.run_deadline_ns: int | None = None
        self.last_slot = -1
        self.dt_ns = round(self.execution.control_dt_s * 1e9)
        self.started_ns = self.next_step_ns = 0
        self.has_submitted = False
        self.ik_reference: IKReference | None = None
        self.first_prediction = True
        self.prev_action: np.ndarray | None = None
        self.discard_requested = False
        self.state: Literal["RUNNING", "HOI_HOLD", "RESUMING"] = "RUNNING"
        self.hoi_recovery_started_ns: int | None = None
        self.next_hoi_hold_ns = 0

    @property
    def finished(self):
        return self.result.termination_reason is not None

    def recording_issue(self, row):
        issues = check_observation(row, self.runtime, self.record_needs,
                                   now_ns=row.observation_timestamp_ns)
        return "; ".join(issues) if issues else None

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
        self.run_deadline_ns = None
        if self.observations.needs.hoi:
            channels = self.sensors.channels
            channels.hoi_request.write(episode_id=0, after_ns=ended_ns,
                                       after_sequence=channels.camera_ring.latest_sequence)

    def finish_recording(self, *, wait=True) -> bool:
        """True 仅表示收尾完成；录制成败保留在本段 result 中。"""
        result = self.result
        if self.recorder is not None:
            recorder = self.recorder
            if wait:
                recorder.finish(save=not self.discard_requested, reason=result.termination_reason)
            else:
                recorder.request_finish(save=not self.discard_requested, reason=result.termination_reason)
                if not recorder.poll_finish():
                    return False
            result.raw = recorder.result()
            # 线程实际退出才释放 owner；失败结果及 staging 路径保留在 result。
            self.recorder = None
            if recorder.error is not None:
                self.fail("recording", recorder.error)
        return True

    def fail(self, stage, error):
        if isinstance(error, KeyboardInterrupt):
            self.runtime_state.request_emergency_stop()
        self.stop(None, run_end_reason=RunEndReason.FAILED)
        # 停止与收尾仍属于本段；不能保留它们失败前的正常结束状态。
        self.result.termination_reason = "failed" if self.started_ns else "preparation_failed"
        self.result.termination_details = None
        self.result.errors.append(error_detail(stage, error))

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
            except FeedbackUnavailable as exc:
                self.robot.check()
                poll_operator()
                self.stop(str(exc))
            except OperationCancelled as exc:
                # 取消不能掩盖同时发生的设备或 worker 故障。
                self.robot.check()
                poll_operator()
                if (not self.started_ns and self.runtime_state.cancel_epoch == self.cancel_epoch
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
            self.runtime_state.cancel_epoch == self.cancel_epoch
            and self.runtime_state.requests_allow_motion()
        )
        if not valid:
            self.robot.check()
        return valid

    def read_observations(self) -> tuple[Observation, Observation]:
        """共享关节采样；派生模型图像同源，Raw 独立取最新相机帧。"""
        row = read_observation(
            self.sensors.channels,
            self.robot,
            needs=self.read_needs,
            joint_gap_s=self.execution.hoi_max_joint_gap_s,
        )
        raw = row
        if self.record_raw and (self.observations.needs.hoi or self.observations.needs.point_cloud):
            camera = read_camera_frame(self.sensors.channels, fields=self.record_needs.camera_fields)
            raw = replace(row, camera=camera, hoi=None, observation_timestamp_ns=time.monotonic_ns())
        return row, raw

    def start_observation(self, poll_operator, *, after_ns=None, initial_hoi=False):
        timeout_s = 120.0 if initial_hoi else INITIAL_STATE_TIMEOUT_S
        until = time.monotonic_ns() + int(timeout_s * 1e9)
        while True:
            poll_operator()
            if not self.preparation_valid():
                raise OperationCancelled("initial observation cancelled")
            self.robot.service_feedback()
            row, raw = self.read_observations()
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
                and (not self.runtime.hand.enabled or row.hand is not None
                     and int(row.hand["timestamp_ns"][0]) >= after_ns)
            )
            if issue is None and joints_ready:
                self.preparation_issue = None
                break
            if time.monotonic_ns() >= until:
                self.preparation_issue = issue or "initial_joint_feedback_unavailable"
                logger.warning("本轮未开始：%s；等待停止与收尾。", self.preparation_issue)
                return None
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
            channels = self.sensors.channels
            channels.hoi_request.write(episode_id=self.result.index, after_ns=time.monotonic_ns(),
                                       after_sequence=channels.camera_ring.latest_sequence)
        # 全模态初值先于 writer；等待与资源初始化期间设备保持停止。
        row = self.start_observation(poll_operator, initial_hoi=self.observations.needs.hoi)
        if row is None:
            self.stop(self.preparation_issue)
            return
        if self.observations.needs.hoi:
            kwargs = ({"rtc_delay": self.execution.prefetch_steps, "rtc_guidance_cap": self.execution.rtc_guidance_cap}
                      if self.execution.mode == "rtc" else {})
            self.model.warmup((self.observations.project(row),) * self.policy_info.n_obs_steps, **kwargs)
            warmup_deadline = time.monotonic() + 120
            while self.model.future is not None:
                if time.monotonic() >= warmup_deadline:
                    raise TimeoutError("HOI policy warmup timeout")
                poll_operator()
                self.robot.service_feedback()
                if not self.preparation_valid():
                    raise OperationCancelled("HOI warmup cancelled")
                future = self.model.poll()
                if future is not None:
                    future.result()
                time.sleep(0.005)
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before recorder start")
        if self.record_raw:
            self.recorder = EpisodeRecorder(self.output_dir, control_hz=1 / self.execution.control_dt_s)
            metadata = snapshot_recording_metadata(
                self.camera_snapshot,
                tactile_reset_metadata=self.robot.hand.tactile_reset_metadata,
            )
            self.recorder.start(
                task_label=self.policy_info.task_name,
                episode_name=f"episode_{self.result.index:03d}", collection_source="policy_rollout", metadata=metadata,
            )
            # writer 初始化期间设备保持停止，准备运动前重新等候新反馈。
            if self.start_observation(poll_operator, after_ns=time.monotonic_ns()) is None:
                self.stop(self.preparation_issue)
                return
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before streaming preparation")
        self.robot.prepare_streaming(self.cancel_epoch, poll_operator=poll_operator)
        if self.start_observation(poll_operator) is None:
            self.stop(self.preparation_issue)
            return
        with self.runtime_state.motion_lock:
            started_ns = (
                time.monotonic_ns()
                if self.runtime_state.cancel_epoch == self.cancel_epoch
                and self.runtime_state.requests_allow_motion()
                and not self.runtime_state.stop_signal.is_set()
                else None
            )
        if started_ns is None:
            self.robot.check()
            raise OperationCancelled("final admission declined")
        self.started_ns = started_ns
        self.next_step_ns = self.started_ns
        self.run_deadline_ns = (
            self.started_ns + self.max_duration_ns if self.max_duration_ns is not None else None
        )
        self.result.started_utc = utc_now()
        logger.info("Policy active; first valid sample fills %d history frames", self.history.maxlen)

    def poll_model(self):
        future = self.model.poll()
        if future is not None:
            actions = future.result()
            if self.state == "HOI_HOLD":
                # result() 的异常仍须传播，旧预测不能进入恢复后的计划。
                return
            # sync 直接接收动作；async/RTC 将结果关联到原预测请求。
            if self.execution.mode == "sync":
                self.plan = ActionChunk(actions, 0)
                self.cursor = 0
            else:
                cast(Prediction, self.query).result = actions

    def submit(self, slot, *, handoff_slot=None, prefix=None):
        kwargs = {} if prefix is None else {
            "rtc_prefix": prefix,
            "delay_steps": min(self.execution.prefetch_steps, len(prefix)),
            "rtc_guidance_cap": self.execution.rtc_guidance_cap,
        }
        if self.policy_info.requires_prev_action:
            kwargs["prev_action"] = self.prev_action
        rows = tuple(self.history)
        self.check_motion()
        self.model.predict(rows, reset_episode=self.first_prediction, **kwargs)
        self.first_prediction = False
        if self.execution.mode != "sync":
            self.query = Prediction(slot, handoff_slot, request_cursor=self.cursor)

    def map_command(self, action, row):
        mode = self.policy_info.action_mode
        if mode == "joint":
            decoded = self.action_mapper.map_joint(action[:7], action[7:])
        else:
            current = row.arm["qpos"][0]
            pose = Pose(action[:3], rot6d_to_quat_wxyz(action[3:9]))
            reference = prepare_reference(
                self.action_mapper.solver, self.ik_reference, current,
                control_ns=time.monotonic_ns(),
                period_s=self.execution.control_dt_s,
            )
            decoded = self.action_mapper.map_eef(pose, action[9:], reference)
        if decoded.rejection_reason is not None or decoded.arm_qpos is None:
            return None, f"action_rejected: {decoded.rejection_reason}"
        return RobotCommand(decoded.arm_qpos, decoded.hand_qpos), None

    def select_action(self, slot):
        if self.execution.mode == "sync":
            if self.plan is None:
                if self.model.future is None:
                    self.submit(None)
                return None
            return self.plan.actions[self.cursor]
        rtc = self.execution.mode == "rtc"
        if rtc:
            query = self.query
            window = min(self.execution.action_steps, len(self.plan.actions)) if self.plan is not None else 0
            if query is not None and query.result is not None and self.cursor >= window:
                consumed = self.cursor - query.request_cursor if self.plan is not None else 0
                actions = query.result[consumed:]
                self.query = None
                if not len(actions):
                    self.stop("prediction exhausted")
                    return None
                self.plan = ActionChunk(actions, 0)
                self.cursor = 0
        else:
            new_plan = handoff_plan(self.query, slot=slot)
            if new_plan is not None:
                self.plan, self.query = new_plan, None
        if self.plan is None:
            # 启动或恢复后的首预测允许无旧计划等待；执行中耗尽仍结束。
            if self.first_prediction:
                self.submit(slot)
            return None
        exhausted = self.cursor >= len(self.plan.actions) if rtc else plan_exhausted(self.plan, slot)
        if exhausted:
            self.stop("prediction exhausted")
            return None
        if self.query is None:
            if rtc:
                window = min(self.execution.action_steps, len(self.plan.actions))
                if self.cursor >= max(0, window - self.execution.prefetch_steps):
                    self.submit(slot, prefix=self.plan.actions[self.cursor:].copy())
            else:
                handoff = prefetch_handoff(self.plan, slot=slot,
                    action_steps=self.execution.action_steps, prefetch_steps=self.execution.prefetch_steps)
                if handoff is not None:
                    self.submit(slot, handoff_slot=handoff)
        return self.plan.actions[self.cursor if rtc else slot - self.plan.anchor_slot]

    def check_motion(self):
        if (self.runtime_state.cancel_epoch != self.cancel_epoch
                or not self.runtime_state.requests_allow_motion()
                or self.runtime_state.stop_signal.is_set()):
            self.robot.check()
            raise OperationCancelled("operation cancelled")

    def sample_observation(self) -> tuple[Observation, Observation]:
        row, raw = self.read_observations()
        devices = check_device_feedback(row, self.runtime)
        if devices:
            raise FeedbackUnavailable("device feedback: " + "; ".join(devices))
        return row, raw

    def update_history(self, row):
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
        if self.run_deadline_ns is not None and time.monotonic_ns() >= self.run_deadline_ns:
            self.stop("duration_timeout", run_end_reason=RunEndReason.DURATION_COMPLETE)
            return True
        return False

    def prepare_command(self, action, row):
        try:
            command, issue = self.map_command(action, row)
        except IKReferenceError as exc:
            self.stop(str(exc))
            return None
        if issue is not None:
            self.stop(issue)
            return None
        self.check_motion()
        if self.duration_expired() or self._recovery_expired():
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
        return command

    def record_row(self, row, attempted_command, *, quality_issue):
        primary_error = sys.exc_info()[1]
        if primary_error is not None:
            self.runtime_state.stop_signal.set()
        if self.recorder is not None:
            try:
                issue = self.recorder.append(row, attempted_command, quality_issue=quality_issue)
                if issue is not None:
                    self.stop(issue)
            except BaseException as exc:
                if primary_error is None:
                    raise
                if exc is primary_error:
                    return
                self.result.errors.append(error_detail("recording_append", exc))
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                logger.exception("recording also failed during control failure")

    def _observation_failure(self, issues, context):
        if is_recoverable_hoi_failure(issues):
            self._enter_hoi_hold()
        else:
            self.stop(context + ": " + "; ".join(issues))

    def _clear_policy_context(self):
        """清除控制侧上下文；在途推理仍由 poll_model 或 Session 回收。"""
        self.history.clear()
        self.plan = self.query = None
        self.cursor = 0
        self.prev_action = None
        self.ik_reference = None
        self.first_prediction = True
        self.has_submitted = False

    def _enter_hoi_hold(self):
        if self.state == "HOI_HOLD":
            return
        # 首条恢复动作提交前的再次遮挡仍属于同一个恢复区间。
        if self.state == "RUNNING":
            self.hoi_recovery_started_ns = time.monotonic_ns()
            if self.recorder is not None:
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
        hand = state.hand["qpos"][0] if self.runtime.hand.enabled else None
        self.robot.send_action(
            RobotCommand(arm_hold=True, hand_qpos=hand),
            expires_ns=now + COMMAND_LEASE_NS,
        )
        self.next_hoi_hold_ns = now + int(HOI_HOLD_REFRESH_S * 1e9)

    def _recovery_expired(self):
        if (self.hoi_recovery_started_ns is not None
                and time.monotonic_ns() - self.hoi_recovery_started_ns
                >= int(HOI_RECOVERY_TIMEOUT_S * 1e9)):
            self.stop("hoi_recovery_timeout")
            return True
        return False

    def step(self):
        """由 run 在反馈服务与健康检查成功后调用。"""
        if self.finished:
            return
        self.check_motion()
        if self.recorder is not None:
            self.recorder.check_error()
        if self.duration_expired() or self._recovery_expired():
            return
        if self.observations.needs.hoi:
            issues = check_hoi_status(self.sensors.channels, self.runtime)
            if issues:
                self._enter_hoi_hold()
        if self.state != "RUNNING":
            self._refresh_hoi_hold()
        if self.state == "HOI_HOLD":
            # 即使旧推理未完成，也不能掩盖设备或其他输入故障。
            row, raw = self.sample_observation()
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

    def commit_arm(self, command, row):
        submitted_ns = time.monotonic_ns()
        if self.policy_info.action_mode == "eef":
            anchor = (arm_reference_from_feedback(
                row.arm["qpos"][0], *self.action_mapper.solver.limits.T,
            ) if self.ik_reference is None else self.ik_reference.anchor)
            self.ik_reference = IKReference(command.arm_qpos, anchor, submitted_ns)
        self.has_submitted = True

    def _advance_control_slot(self):
        """暂停时也推进原调度，恢复后不补跑历史控制槽。"""
        slot, self.next_step_ns, skipped = control_slot(
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
        row, raw = self.sample_observation()
        quality_issue = self.recording_issue(raw) if self.recorder is not None else None
        attempted = None
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
            action = self.select_action(slot)
            if action is None:
                return
            plan = cast(ActionChunk, self.plan)
            command = self.prepare_command(action, row)
            if command is None:
                return
            now = time.monotonic_ns()
            if (self.execution.mode == "async"
                    and now >= self.started_ns + (plan.anchor_slot + len(plan.actions)) * self.dt_ns):
                self.poll_model()
                current_slot = (time.monotonic_ns() - self.started_ns) // self.dt_ns
                replacement = handoff_plan(self.query, slot=current_slot)
                if replacement is not None and not plan_exhausted(replacement, current_slot):
                    self.plan, self.query = replacement, None
                    return
                self.stop("prediction window expired during mapping")
                return
            attempted = command
            self.robot.send_action(command, expires_ns=now + COMMAND_LEASE_NS)
            if self.state == "RESUMING" and self._recovery_expired():
                return
            if self.policy_info.requires_prev_action:
                self.prev_action = np.concatenate((command.arm_qpos, command.hand_qpos))
            if not self.has_submitted:
                logger.info("Policy first target submitted")
            self.commit_arm(command, row)
            if self.execution.mode in {"sync", "rtc"}:
                self.cursor += 1
                if self.execution.mode == "sync" and self.cursor == min(len(plan.actions), self.execution.action_steps):
                    self.plan = None
            if self.state == "RESUMING":
                self.state = "RUNNING"
                self.hoi_recovery_started_ns = None
        finally:
            # 恢复首条动作可以录制；其间断已在进入 HOLD 时标记。
            if self.state == "RUNNING":
                if skipped and was_running and self.recorder is not None:
                    self.recorder.mark_gap()
                self.record_row(raw, attempted, quality_issue=quality_issue)
