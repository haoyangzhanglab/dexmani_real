"""单段策略运行；持有结果与 recorder，由 Session 编排资源收尾。"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import cast

import numpy as np

from dexmani_real.control.ik import IKReferenceError
from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder
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
    OperationCancelled,
    RobotCommand,
    arm_reference_from_feedback,
)
from dexmani_real.runtime.ik_reference import IKReference, prepare_reference
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_device_feedback,
    check_observation,
    missing_tactile_channels,
    read_observation,
)
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S, RunEndReason
from dexmani_real.utils.log import error_detail
from dexmani_real.utils.timing import next_tick_ns

logger = logging.getLogger(__name__)


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
        observations, record_needs, sensors, action_mapper, model, max_running_s,
        index, output_dir, record_raw, camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.policy_info = policy_info
        self.execution = runtime.execution
        self.robot, self.sensors = robot, sensors
        self.observations = observations
        self.model_needs = observations.needs
        self.capture_needs = self.model_needs.union(record_needs)
        if self.model_needs.hoi:
            self.capture_needs = replace(self.capture_needs, camera_fields=self.model_needs.camera_fields)
        self.record_needs = record_needs
        self.action_mapper, self.model = action_mapper, model
        self.recorder = None
        self.output_dir, self.record_raw = output_dir, record_raw
        self.result = EpisodeResult(index)
        self.camera_snapshot = camera_snapshot
        self.max_duration_ns = int(max_running_s * 1e9) if max_running_s is not None else None
        self.cancel_epoch = cancel_epoch
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=policy_info.n_obs_steps)
        self.model_input_issues: tuple[str, ...] = ()
        self.plan: ActionChunk | None = None
        self.query: Prediction | None = None
        self.cursor = 0
        self.preparation_issue = None
        self.run_deadline_ns = None
        self.last_slot = -1
        self.dt_ns = round(self.execution.control_dt_s * 1e9)
        self.started_ns = self.next_step_ns = 0
        self.has_submitted = False
        self.ik_reference = None
        self.first_prediction = True
        self.discard_requested = False

    @property
    def finished(self):
        return self.result.termination_reason is not None

    def recording_issue(self, row):
        if not self.sensors.check(names=("camera",)):
            raise RuntimeError("recording camera worker exited")
        issues = check_observation(row, self.runtime, self.record_needs,
                                   now_ns=row.observation_timestamp_ns, allow_tactile_hold=True)
        return "; ".join(issues) if issues else None

    def stop(self, detail, *, run_end_reason=RunEndReason.CANCELLED, errors=()):
        if self.finished:
            return
        ended_ns = time.monotonic_ns()
        self.runtime_state.stop_signal.set()
        self.result.duration_s = (ended_ns - self.started_ns) / 1e9 if self.started_ns else 0.0
        self.result.termination_reason = (
            run_end_reason.name.lower() if self.started_ns
            else "preparation_failed" if self.preparation_issue or errors else "start_cancelled"
        )
        self.result.termination_details = detail if not errors else None
        self.result.errors.extend(errors)
        self.plan = self.query = None
        self.history.clear()
        self.has_submitted = False
        self.run_deadline_ns = None
        self.ik_reference = None
        if self.model_needs.hoi:
            channels = self.sensors.channels
            channels.hoi_request.write(episode_id=0, after_ns=ended_ns,
                                       after_sequence=channels.camera_ring.latest_sequence)

    def finish_recording(self, *, wait=True) -> bool:
        result = self.result
        if self.recorder is not None:
            recorder = self.recorder
            finish_kwargs = {
                "save": not self.discard_requested, "reason": result.termination_reason,
                "details": result.errors + ([{"stage": "termination", "reason": result.termination_details}]
                                            if result.termination_details is not None else []),
            }
            if wait:
                recorder.finish(**finish_kwargs)
            else:
                recorder.request_finish(**finish_kwargs)
                if not recorder.poll_finish():
                    return False
            result.raw = recorder.result()
            # 只在真正完成后释放 owner；失败/仍存活时保留 recorder。
            self.recorder = None
            if not recorder.finish_request.save:
                logger.info("Raw 已丢弃。")
        return True

    def run(self, *, poll_operator):
        """运行结束先停设备；段间及最终资源收尾由 Session 编排。"""
        try:
            try:
                self.prepare(poll_operator)
                while not self.finished:
                    poll_operator()
                    self.robot.service_feedback()
                    self.step()
                    if not self.finished:
                        time.sleep(0.001)
            except OperationCancelled as exc:
                # 取消不能掩盖同时发生的设备或 worker 故障。
                self.robot.check()
                poll_operator()
                if (not self.started_ns and self.runtime_state.cancel_epoch == self.cancel_epoch
                        and self.runtime_state.requests_allow_motion()):
                    raise
                self.stop(str(exc))
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            if not self.finished:
                stage = "control" if self.started_ns else "preparation"
                self.stop(str(exc), run_end_reason=RunEndReason.FAILED,
                          errors=[error_detail(stage, exc)])
            else:
                self.result.errors.append(error_detail("control", exc))
            raise
        finally:
            primary_error = sys.exc_info()[1]
            try:
                self.robot.stop()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                self.result.errors.append(error_detail("stop", exc))
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
        """共享关节采样；HOI 模型图像同源，Raw 独立取最新相机帧。"""
        row = read_observation(self.sensors.channels, self.robot, needs=self.capture_needs)
        raw = row
        if self.record_raw and self.model_needs.hoi:
            camera = read_camera_frame(self.sensors.channels, fields=self.record_needs.camera_fields)
            raw = replace(row, camera=camera, hoi=None, observation_timestamp_ns=time.monotonic_ns())
        return row, raw

    def start_observation(self, poll_operator=lambda: None, *, after_ns=None, initial_hoi=False):
        timeout_s = 120.0 if initial_hoi else INITIAL_STATE_TIMEOUT_S
        until = time.monotonic_ns() + int(timeout_s * 1e9)
        while True:
            poll_operator()
            if not self.preparation_valid():
                raise OperationCancelled("initial observation cancelled")
            self.robot.service_feedback()
            row, raw = self.read_observations()
            devices = check_device_feedback(row, self.runtime)
            model_issues = check_observation(row, self.runtime, self.model_needs)
            if "hoi_pose_invalid" in model_issues:
                self.preparation_issue = "model_input_hoi_pose_invalid"
                logger.warning("本轮 HOI 初始化/位姿失效；详见 Worker 日志，可再次按 B。")
                return None
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
                missing = missing_tactile_channels(row, self.capture_needs.required_tactile)
                if missing:
                    self.preparation_issue = "initial_tactile_unavailable: " + ", ".join(missing)
                else:
                    self.preparation_issue = issue or "initial_joint_feedback_unavailable"
                logger.warning("本轮未开始：%s；可稍后再次按 B。", self.preparation_issue)
                return None
            time.sleep(0.005)
        if self.policy_info.action_mode == "eef":
            try:
                self.action_mapper.solver.measured_reference(row.arm["qpos"][0])
            except IKReferenceError as exc:
                self.preparation_issue = str(exc)
                logger.warning("本轮未开始：%s；请就位后再次按 B。", exc)
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
        if self.model_needs.hoi:
            channels = self.sensors.channels
            channels.hoi_request.write(episode_id=self.result.index, after_ns=time.monotonic_ns(),
                                       after_sequence=channels.camera_ring.latest_sequence)
        # 全模态初值先于 writer；等待与资源初始化期间设备保持停止。
        row = self.start_observation(poll_operator, initial_hoi=self.model_needs.hoi)
        if row is None:
            self.stop(self.preparation_issue)
            return
        if self.model_needs.hoi:
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
            # writer 可能阻塞超过反馈 TTL；仍停止采样，允许一次 CRC 后恢复。
            if self.start_observation(poll_operator, after_ns=time.monotonic_ns()) is None:
                self.stop(self.preparation_issue)
                return
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before streaming preparation")
        self.robot.prepare_streaming(self.cancel_epoch)
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
            # 当前 rollout 独占消费预测结果；只有时间窗口模式需要 query 锚点。
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
        rows = tuple(self.history)
        self.check_motion()
        self.model.predict(rows, reset_episode=self.first_prediction, **kwargs)
        self.first_prediction = False
        if self.execution.mode != "sync":
            self.query = Prediction(slot, handoff_slot, request_cursor=self.cursor)

    def map_command(self, action, row):
        mode = self.policy_info.action_mode
        current = row.arm["qpos"][0]
        if mode == "joint":
            decoded = self.action_mapper.map_joint(action[:7], action[7:], current)
        else:
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

    def select_async_action(self, slot, model_input_valid):
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
            # 仅首预测允许无旧计划等待；运行中耗尽直接结束。
            if self.first_prediction and model_input_valid:
                self.submit(slot)
            return None
        exhausted = self.cursor >= len(self.plan.actions) if rtc else plan_exhausted(self.plan, slot)
        if exhausted:
            self.stop("prediction exhausted")
            return None
        if self.query is None and model_input_valid:
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

    def sample_observation(self) -> tuple[Observation, Observation] | None:
        row, raw = self.read_observations()
        devices = check_device_feedback(row, self.runtime)
        if devices:
            raise RuntimeError("device feedback: " + "; ".join(devices))
        if self.recorder is not None and (quality := self.recording_issue(raw)) is not None:
            self.recorder.append(raw, quality_issue=quality)
            self.stop(quality)
            return None
        return row, raw

    def update_history(self, row):
        issues = tuple(check_observation(row, self.runtime, self.model_needs))
        if issues and issues != self.model_input_issues:
            logger.warning("策略暂缓新推理：%s", "; ".join(issues))
        self.model_input_issues = issues
        if issues:
            self.history.clear()
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
        if self.duration_expired():
            return None
        issues = check_device_feedback(row, self.runtime)
        if issues:
            raise RuntimeError("device feedback: " + "; ".join(issues))
        if self.model_needs.hoi:
            needs = ObservationNeeds(hoi=True)
            latest = read_observation(self.sensors.channels, self.robot, needs=needs)
            issues = check_observation(latest, self.runtime, needs)
            if issues:
                self.stop("HOI before submission: " + "; ".join(issues))
                return None
        return command

    def record_row(self, row, attempted_command):
        primary_error = sys.exc_info()[1]
        if primary_error is not None:
            self.runtime_state.stop_signal.set()
        if row is not None and self.recorder is not None:
            try:
                issue = self.recorder.append(row, attempted_command)
                if issue is not None:
                    self.stop(issue)
            except BaseException as exc:
                if primary_error is None:
                    raise
                self.result.errors.append(error_detail("recording_append", exc))
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                logger.exception("recording also failed during control failure")

    def step(self):
        if self.finished:
            return
        self.robot.check()
        if (self.runtime_state.cancel_epoch != self.cancel_epoch
                or not self.runtime_state.requests_allow_motion()):
            raise OperationCancelled("operation cancelled")
        if self.recorder is not None:
            self.recorder.check_error()
        if self.duration_expired():
            return
        if self.runtime_state.stop_signal.is_set():
            raise OperationCancelled("device stopped")
        if self.model_needs.hoi:
            needs = ObservationNeeds(hoi=True)
            row = read_observation(self.sensors.channels, self.robot, needs=needs)
            issues = check_observation(row, self.runtime, needs)
            if issues:
                self.stop("HOI admission: " + "; ".join(issues))
                return
        if self.execution.mode == "sync":
            self.step_sync()
        else:
            self.step_async()

    def commit_arm(self, command, row):
        submitted_ns = time.monotonic_ns()
        if self.policy_info.action_mode == "eef":
            anchor = (arm_reference_from_feedback(
                row.arm["qpos"][0], *self.action_mapper.solver.limits.T,
            ) if self.ik_reference is None else self.ik_reference.anchor)
            self.ik_reference = IKReference(command.arm_qpos, anchor, submitted_ns)
        self.has_submitted = True

    def step_sync(self):
        now = time.monotonic_ns()
        if now < self.next_step_ns:
            self.poll_model()
            return
        self.next_step_ns = next_tick_ns(self.next_step_ns, self.dt_ns, now)
        row = raw = attempted = None
        try:
            sample = self.sample_observation()
            if sample is None:
                return
            row, raw = sample
            valid = self.update_history(row)
            self.check_motion()
            if self.duration_expired():
                return
            self.poll_model()
            if self.plan is None:
                if valid and self.model.future is None:
                    self.submit(None)
                return
            command = self.prepare_command(self.plan.actions[self.cursor], row)
            if command is None:
                return
            attempted = command
            self.robot.send_action(command, expires_ns=time.monotonic_ns() + COMMAND_LEASE_NS)
            if not self.has_submitted:
                logger.info("Policy first target submitted")
            self.commit_arm(command, row)
            self.cursor += 1
            if self.cursor == min(len(self.plan.actions), self.execution.action_steps):
                self.plan = None
        finally:
            self.record_row(raw, attempted)

    def step_async(self):
        slot, self.next_step_ns, skipped = control_slot(
            now_ns=time.monotonic_ns(), next_step_ns=self.next_step_ns,
            last_slot=self.last_slot, dt_ns=self.dt_ns,
        )
        if slot is None:
            self.poll_model()
            return
        self.result.skipped_slots += skipped
        self.last_slot = slot
        row = raw = attempted = None
        pending = self.model.future
        try:
            sample = self.sample_observation()
            if sample is None:
                return
            row, raw = sample
            valid = self.update_history(row)
            self.check_motion()
            if self.duration_expired():
                return
            self.poll_model()
            action = self.select_async_action(slot, valid)
            if action is None:
                return
            command = self.prepare_command(action, row)
            if command is None:
                return
            now = time.monotonic_ns()
            if self.execution.mode == "async":
                plan = cast(ActionChunk, self.plan)
                if now >= self.started_ns + (plan.anchor_slot + len(plan.actions)) * self.dt_ns:
                    # 只收取进入本次操作前已有的任务，不轮询刚提交的新请求。
                    if pending is not None and self.model.future is pending:
                        self.poll_model()
                    current_slot = (now - self.started_ns) // self.dt_ns
                    replacement = handoff_plan(self.query, slot=current_slot)
                    if replacement is not None and not plan_exhausted(replacement, current_slot):
                        self.plan, self.query = replacement, None
                        return
                    self.stop("prediction window expired during mapping")
                    return
            attempted = command
            self.robot.send_action(command, expires_ns=now + COMMAND_LEASE_NS)
            if not self.has_submitted:
                logger.info("Policy first target submitted")
            self.commit_arm(command, row)
            if self.execution.mode == "rtc":
                self.cursor += 1
        finally:
            self.record_row(raw, attempted)
