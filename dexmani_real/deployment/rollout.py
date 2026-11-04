"""单轮策略执行：准备、预测、控制与录制收尾。"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import cast

import numpy as np

from dexmani_real.config.deployment import COMMAND_LEASE_NS
from dexmani_real.config.xhand import INITIAL_STATE_TIMEOUT_S
from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder
from dexmani_real.deployment.inference import MODEL_SHUTDOWN_TIMEOUT_S
from dexmani_real.deployment.schedule import (
    ActionChunk,
    Prediction,
    control_slot,
    handoff_plan,
    plan_rejection,
    prefetch_handoff,
)
from dexmani_real.geometry.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.runtime.observation import (
    check_observation,
    missing_tactile_channels,
    read_observation,
    sources_available,
)
from dexmani_real.runtime.state import RunEndReason
from dexmani_real.utils.log import error_detail

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

    @property
    def normal_termination(self):
        return not self.errors and self.termination_reason in {
            "operator", "quit", "duration_complete",
        }


class Rollout:
    def __init__(
        self, runtime_state, runtime, policy_info, *, cancel_epoch, robot,
        observations, control_needs, record_needs, sensors, action_mapper, model, max_running_s,
        index, output_dir, record_raw, audio, camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.policy_info = policy_info
        self.execution = runtime.execution
        self.robot, self.sensors = robot, sensors
        self.observations = observations
        self.capture_needs = control_needs.union(record_needs)
        self.control_needs = control_needs
        self.record_needs = record_needs
        self.action_mapper, self.model = action_mapper, model
        self.audio = audio
        self.recorder = None
        self.output_dir, self.record_raw = output_dir, record_raw
        self.result = EpisodeResult(index)
        self.camera_snapshot = camera_snapshot
        self.max_duration_ns = int(max_running_s * 1e9) if max_running_s is not None else None
        self.cancel_epoch = cancel_epoch
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=policy_info.n_obs_steps)
        self.sync_actions = None
        self.sync_cursor = 0
        self.plan: ActionChunk | None = None
        self.query: Prediction | None = None
        self.preparation_issue = None
        self.run_deadline_ns = None
        self.last_slot = -1
        self.dt_ns = round(policy_info.control_dt_s * 1e9)
        self.started_ns = self.next_step_ns = 0
        self.previous_arm = None
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

    def read_observation(self):
        row = read_observation(self.sensors.channels, self.robot, needs=self.capture_needs)
        issues = check_observation(row, self.runtime, self.control_needs)
        return row, "control_input_" + issues[0] if issues else None

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
        self.plan = self.query = self.sync_actions = None
        self.history.clear()
        self.previous_arm = self.run_deadline_ns = None
        self.robot.stop()
        if self.runtime_state.estop_request:
            self.audio.play("emergency")

    def finish_recording(self):
        result = self.result
        if self.recorder is not None:
            save = not self.discard_requested
            try:
                published = self.recorder.finish(
                    save=save, reason=result.termination_reason,
                    details=result.errors + ([dict(stage="termination", reason=result.termination_details)]
                                             if result.termination_details is not None else []),
                )
            finally:
                result.raw = self.recorder.result()
            # 只在真正完成后释放 owner；失败/仍存活时保留 recorder。
            self.recorder = None
            if not result.errors and not self.runtime_state.estop_request:
                if published is not None and result.normal_termination and result.raw["recording_truncated_reason"] is None:
                    self.audio.play("save")
                elif not save:
                    self.audio.play("discard")

    def run(self, *, poll_operator):
        """正常结束回收段内任务；退出或故障时由 session 先关设备再收尾。"""
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
                if not self.started_ns:
                    if (self.runtime_state.cancel_epoch == self.cancel_epoch
                            and self.runtime_state.requests_allow_motion()):
                        raise
                    if not self.robot.arm.is_connected:
                        self.runtime_state.request_quit()
                self.stop(str(exc))
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            if not self.finished:
                stage = "control" if self.started_ns else "preparation"
                try:
                    self.stop(
                        str(exc), run_end_reason=RunEndReason.FAILED,
                        errors=[error_detail(stage, exc)],
                    )
                except BaseException as stop_error:
                    if isinstance(stop_error, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    self.result.errors.append(error_detail("stop", stop_error))
                    logger.exception("episode stop also failed")
            else:
                self.result.errors.append(error_detail("stop", exc))
            self.audio.play("emergency")
            raise
        # 异常直接交给 session：先关闭设备，再处理仍由 episode 持有的 writer。
        # 正常返回待命才在此回收段内资源；任何失败都阻止下一轮。
        if not self.runtime_state.quit_requested and not self.runtime_state.estop_request:
            for stage, finish in (
                ("recorder_finalize", self.finish_recording),
                ("retired_model", lambda: self.model.discard_pending(MODEL_SHUTDOWN_TIMEOUT_S)),
            ):
                try:
                    finish()
                except BaseException as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    self.result.errors.append(error_detail(stage, exc))
                    raise
        return self.result

    def preparation_valid(self):
        valid = (
            self.runtime_state.cancel_epoch == self.cancel_epoch
            and self.runtime_state.requests_allow_motion()
        )
        if not valid:
            self.robot.check()
        return valid

    def start_observation(self, poll_operator=lambda: None, *, after_ns=None):
        until = time.monotonic_ns() + int(INITIAL_STATE_TIMEOUT_S * 1e9)
        while True:
            poll_operator()
            if not self.preparation_valid():
                raise OperationCancelled("initial observation cancelled")
            self.robot.service_feedback()
            row, issue = self.read_observation()
            if self.record_raw:
                recording_issue = self.recording_issue(row)
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
        current = row.arm["qpos"][0]
        if (np.any(current < self.runtime.arm.joint_limit_lower)
                or np.any(current > self.runtime.arm.joint_limit_upper)):
            raise ValueError("initial arm posture violates operational joint limits")
        return row

    def prepare(self, poll_operator):
        logger.info("Policy preparing")
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before preparation")
        self.robot.stop()
        # 全模态初值先于 writer；等待与资源初始化期间设备保持停止。
        if self.start_observation(poll_operator) is None:
            self.stop(self.preparation_issue)
            return
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before recorder start")
        if self.record_raw:
            self.recorder = EpisodeRecorder(self.output_dir, control_hz=1 / self.policy_info.control_dt_s)
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
        logger.info("Policy active; collecting %d observation frames before first prediction",
                    self.history.maxlen)

    def poll_model(self):
        future = self.model.poll()
        if future is not None:
            result = future.result()
            actions = np.asarray(result, dtype=np.float64)
            expected = (self.policy_info.future_steps, self.policy_info.control_action_dim)
            if actions.shape != expected or not np.isfinite(actions).all():
                raise ValueError(f"policy prediction requires finite shape {expected}")
            # 当前 rollout 独占消费预测结果；只有时间窗口模式需要 query 锚点。
            if self.execution.mode == "sync":
                self.sync_actions = actions
                self.sync_cursor = 0
            else:
                cast(Prediction, self.query).result = actions
            return True

    def submit(self, slot, *, handoff_slot=None, prefix=None):
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=handoff_slot - slot,
                 rtc_guidance_cap=self.execution.rtc_guidance_cap) if prefix is not None else {}
        )
        rows = tuple(self.history)
        self.model.predict(rows, reset_episode=self.first_prediction, **kwargs)
        self.first_prediction = False
        if self.execution.mode != "sync":
            self.query = Prediction(slot, handoff_slot)

    def map_command(self, action, row):
        mode = self.policy_info.action_mode
        current = row.arm["qpos"][0]
        if mode == "joint":
            decoded = self.action_mapper.map_joint(action[:7], action[7:], current)
        else:
            pose = Pose(action[:3], rot6d_to_quat_wxyz(action[3:9]))
            decoded = self.action_mapper.map_eef(pose, action[9:], current, self.previous_arm)
        if decoded.rejection_reason is not None or decoded.arm_qpos is None:
            return None, f"action_rejected: {decoded.rejection_reason}"
        return RobotCommand(decoded.arm_qpos, decoded.hand_qpos), None

    def submit_prefetch(self, plan: ActionChunk, slot):
        handoff_slot, issue = prefetch_handoff(
            plan, slot=slot, mode=self.execution.mode,
            action_steps=self.execution.action_steps,
            prefetch_steps=self.execution.prefetch_steps,
            query_pending=self.query is not None,
        )
        if issue is not None or handoff_slot is None:
            return issue
        # 与执行使用同一规范化策略动作；不提前映射未来 RobotCommand。
        prefix = (
            plan.actions[slot - plan.anchor_slot:].copy()
            if self.execution.mode == "rtc" else None
        )
        self.submit(slot, handoff_slot=handoff_slot, prefix=prefix)
        return None

    def step(self):
        row = attempted_command = None
        try:
            if self.finished:
                return
            motion_allowed = (
                self.runtime_state.cancel_epoch == self.cancel_epoch
                and self.runtime_state.requests_allow_motion()
                and not self.runtime_state.stop_signal.is_set()
            )
            if not motion_allowed:
                self.robot.check()
                if (self.runtime_state.cancel_epoch != self.cancel_epoch
                        or not self.runtime_state.requests_allow_motion()):
                    raise OperationCancelled("operation cancelled")
            prediction_ready = self.poll_model()
            if self.recorder is not None:
                self.recorder.check_error()
            if self.run_deadline_ns is not None and time.monotonic_ns() >= self.run_deadline_ns:
                self.stop("duration_timeout", run_end_reason=RunEndReason.DURATION_COMPLETE)
                return
            if not motion_allowed:
                raise OperationCancelled("device stopped")

            now = time.monotonic_ns()
            synchronous = self.execution.mode == "sync"
            waiting_sync = synchronous and self.model.future is not None
            if synchronous:
                if now < self.next_step_ns and not prediction_ready:
                    return
                self.next_step_ns = now + self.dt_ns
            else:
                slot, self.next_step_ns, skipped = control_slot(
                    now_ns=now, next_step_ns=self.next_step_ns, last_slot=self.last_slot,
                    dt_ns=self.dt_ns,
                )
                if slot is None:
                    return
                self.result.skipped_slots += skipped
                if skipped and self.execution.mode == "rtc" and self.plan is not None:
                    self.stop(f"missed_slot: expected={self.last_slot + 1}, actual={slot}")
                    return
                self.last_slot = slot
            observation, issue = self.read_observation()
            if self.recorder is not None and (quality := self.recording_issue(observation)) is not None:
                self.recorder.append(observation, quality_issue=quality)
                self.stop(quality)
                return
            row = observation
            if not sources_available(row, self.control_needs):
                row = None
                if self.execution.mode == "rtc" and self.plan is not None:
                    self.stop("committed_slot_observation_unavailable")
                return
            if issue is not None:
                self.stop(issue)
                return
            if waiting_sync:
                # 当前输入持续裁决；等待行是评估事实，不重复插入模型历史。
                return
            if not (synchronous and prediction_ready):
                # 最近 N 次已接纳的实际采样；模型等待与调度抖动可使间隔不均。
                self.history.append(self.observations.project(row))

            if synchronous:
                actions = self.sync_actions
                if actions is None:
                    if len(self.history) == self.policy_info.n_obs_steps:
                        self.submit(None)
                    return
                action_count = min(len(actions), self.execution.action_steps)
                action = actions[self.sync_cursor]
            else:
                new_plan, issue = handoff_plan(self.query, slot=slot, mode=self.execution.mode)
                if issue == "handoff_prediction_exhausted" and self.execution.mode == "async":
                    self.query = None
                    issue = None
                if issue is not None:
                    self.stop(issue)
                    return
                if new_plan is not None:
                    if self.execution.mode == "rtc":
                        new_plan.actions = self.action_mapper.normalize_policy_actions(
                            new_plan.actions, self.policy_info.action_mode,
                        )
                    self.plan, self.query = new_plan, None
                if self.plan is not None:
                    issue = plan_rejection(self.plan, slot, self.execution.mode, self.execution.action_steps)
                    if issue is not None:
                        if self.execution.mode == "async":
                            self.plan = None
                        else:
                            self.stop(issue)
                            return
                if self.plan is None:
                    if len(self.history) == self.policy_info.n_obs_steps and self.query is None:
                        self.submit(slot)
                    return

                issue = self.submit_prefetch(self.plan, slot)
                if issue is not None:
                    self.stop(issue)
                    return
                action = self.plan.actions[slot - self.plan.anchor_slot]
            command, issue = self.map_command(action, row)
            if issue is not None:
                self.stop(issue)
                return
            # 长计算后重查控制来源；设备最终只收到目标租期。
            if self.run_deadline_ns is not None and time.monotonic_ns() >= self.run_deadline_ns:
                self.stop("duration_timeout", run_end_reason=RunEndReason.DURATION_COMPLETE)
                return
            source_issues = check_observation(row, self.runtime, self.control_needs)
            if source_issues:
                self.stop("control_sources: " + "; ".join(source_issues))
                return
            now = time.monotonic_ns()
            if self.execution.mode == "rtc" and now >= self.next_step_ns:
                self.stop("committed_slot_submission_missed")
                return
            if self.execution.mode == "async":
                plan = cast(ActionChunk, self.plan)
                # 裁剪前缀后的 anchor + 长度仍是原预测截止；A 只决定预取。
                if now >= self.started_ns + (plan.anchor_slot + len(plan.actions)) * self.dt_ns:
                    self.plan = None
                    return
            expires_ns = now + COMMAND_LEASE_NS
            attempted_command = command
            self.robot.send_action(command, expires_ns=expires_ns)
            if self.previous_arm is None:
                logger.info("Policy first target submitted")
                self.audio.play("begin")
            self.previous_arm = command.arm_qpos
            if synchronous:
                self.sync_cursor += 1
                if self.sync_cursor == action_count:
                    self.sync_actions = None
        finally:
            # 每个真实采样与至多一次提交尝试只记一行；部分失败不改写为成功。
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
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    logger.exception("recording also failed during control failure")
