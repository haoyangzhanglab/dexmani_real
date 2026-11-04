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

from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder
from dexmani_real.deployment.inference import MODEL_SHUTDOWN_TIMEOUT_S
from dexmani_real.deployment.schedule import (
    ActionChunk,
    Prediction,
    control_slot,
    handoff_plan,
    plan_exhausted,
    prefetch_handoff,
)
from dexmani_real.geometry.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.robot.commands import (
    COMMAND_LEASE_NS,
    OperationCancelled,
    RobotCommand,
    arm_reference_from_feedback,
)
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_observation,
    missing_tactile_channels,
    read_observation,
)
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S, RunEndReason
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
        self.record_needs = record_needs
        self.action_mapper, self.model = action_mapper, model
        self.recorder = None
        self.output_dir, self.record_raw = output_dir, record_raw
        self.result = EpisodeResult(index)
        self.camera_snapshot = camera_snapshot
        self.max_duration_ns = int(max_running_s * 1e9) if max_running_s is not None else None
        self.cancel_epoch = cancel_epoch
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=policy_info.n_obs_steps)
        self.sync_actions = None
        self.plan: ActionChunk | None = None
        self.query: Prediction | None = None
        self.cursor = 0
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
        issues = check_observation(row, self.runtime, self.model_needs)
        return row, "model_input_" + issues[0] if issues else None

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

    def finish_recording(self):
        result = self.result
        if self.recorder is not None:
            save = not self.discard_requested
            try:
                self.recorder.finish(
                    save=save, reason=result.termination_reason,
                    details=result.errors + ([dict(stage="termination", reason=result.termination_details)]
                                             if result.termination_details is not None else []),
                )
            finally:
                result.raw = self.recorder.result()
            # 只在真正完成后释放 owner；失败/仍存活时保留 recorder。
            self.recorder = None
            if not save:
                logger.info("Raw 已丢弃。")

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
                self.stop(str(exc), run_end_reason=RunEndReason.FAILED,
                          errors=[error_detail(stage, exc)])
            else:
                self.result.errors.append(error_detail("control", exc))
            raise
        finally:
            try:
                self.robot.stop()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                self.result.errors.append(error_detail("stop", exc))
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
        logger.info("Policy active; first valid sample fills %d history frames", self.history.maxlen)

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
                self.cursor = 0
            else:
                cast(Prediction, self.query).result = actions

    def submit(self, slot, *, handoff_slot=None, prefix=None):
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=min(self.execution.prefetch_steps, len(prefix)),
                 rtc_guidance_cap=self.execution.rtc_guidance_cap) if prefix is not None else {}
        )
        rows = tuple(self.history)
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
            decoded = self.action_mapper.map_eef(pose, action[9:], current, self.previous_arm)
        if decoded.rejection_reason is not None or decoded.arm_qpos is None:
            return None, f"action_rejected: {decoded.rejection_reason}"
        return RobotCommand(decoded.arm_qpos, decoded.hand_qpos), None

    def step_sync(self, model_input_valid):
        if self.sync_actions is None:
            if model_input_valid and self.model.future is None:
                self.submit(None)
            return None
        return self.sync_actions[self.cursor]

    def step_async(self, slot, model_input_valid):
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
                actions = self.action_mapper.normalize_policy_actions(actions, self.policy_info.action_mode)
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
            if self.recorder is not None:
                self.recorder.check_error()
            if self.run_deadline_ns is not None and time.monotonic_ns() >= self.run_deadline_ns:
                self.poll_model()
                self.stop("duration_timeout", run_end_reason=RunEndReason.DURATION_COMPLETE)
                return
            if not motion_allowed:
                self.poll_model()
                raise OperationCancelled("device stopped")

            now = time.monotonic_ns()
            synchronous = self.execution.mode == "sync"
            if synchronous:
                if now < self.next_step_ns:
                    self.poll_model()
                    return
                self.next_step_ns = now + self.dt_ns
            else:
                slot, self.next_step_ns, skipped = control_slot(
                    now_ns=now, next_step_ns=self.next_step_ns, last_slot=self.last_slot,
                    dt_ns=self.dt_ns,
                )
                if slot is None:
                    self.poll_model()
                    return
                self.result.skipped_slots += skipped
                self.last_slot = slot
            observation, issue = self.read_observation()
            device_issues = check_observation(observation, self.runtime, ObservationNeeds())
            if device_issues:
                raise RuntimeError("device feedback: " + "; ".join(device_issues))
            if self.recorder is not None and (quality := self.recording_issue(observation)) is not None:
                self.recorder.append(observation, quality_issue=quality)
                self.stop(quality)
                return
            row = observation
            model_input_valid = issue is None
            if model_input_valid:
                projected = self.observations.project(row)
                if not self.history:
                    self.history.extend([projected] * self.policy_info.n_obs_steps)
                else:
                    self.history.append(projected)
            # 采样/投影期间取消时，旧任务由收尾回收，不再接收或新发请求。
            self.check_motion()
            self.poll_model()
            prior_query = self.query
            action = (self.step_sync(model_input_valid) if synchronous
                      else self.step_async(slot, model_input_valid))
            if action is None:
                return
            actions = self.sync_actions
            command, issue = self.map_command(action, row)
            if issue is not None:
                self.stop(issue)
                return
            # 长计算后只复核设备反馈；模型模态不裁决已有动作。
            self.check_motion()
            if self.run_deadline_ns is not None and time.monotonic_ns() >= self.run_deadline_ns:
                self.stop("duration_timeout", run_end_reason=RunEndReason.DURATION_COMPLETE)
                return
            source_issues = check_observation(row, self.runtime, ObservationNeeds())
            if source_issues:
                raise RuntimeError("device feedback: " + "; ".join(source_issues))
            now = time.monotonic_ns()
            if self.execution.mode == "async":
                plan = cast(ActionChunk, self.plan)
                # 裁剪前缀后的 anchor + 长度仍是原预测截止；A 只决定预取。
                if now >= self.started_ns + (plan.anchor_slot + len(plan.actions)) * self.dt_ns:
                    # 映射期间原窗口可能耗尽；先收取此前请求的可用结果。
                    if self.query is not None and self.query is prior_query:
                        self.poll_model()
                        current_slot = (now - self.started_ns) // self.dt_ns
                        replacement = handoff_plan(self.query, slot=current_slot)
                        if replacement is not None and not plan_exhausted(replacement, current_slot):
                            self.plan, self.query = replacement, None
                            return
                    self.stop("prediction window expired during mapping")
                    return
            expires_ns = now + COMMAND_LEASE_NS
            attempted_command = command
            self.robot.send_action(command, expires_ns=expires_ns)
            if self.previous_arm is None:
                logger.info("Policy first target submitted")
            self.previous_arm = command.arm_qpos
            if synchronous:
                self.cursor += 1
                if self.cursor == min(len(cast(np.ndarray, actions)), self.execution.action_steps):
                    self.sync_actions = None
            elif self.execution.mode == "rtc":
                self.cursor += 1
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
