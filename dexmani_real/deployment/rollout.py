"""单轮策略执行：准备、预测、控制与录制收尾。"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.config.deployment import (
    COMMAND_LEASE_NS,
    PREDICTION_WAIT_TIMEOUT_NS,
    START_ARM_TOLERANCE_DEG,
)
from dexmani_real.config.xhand import INITIAL_STATE_TIMEOUT_S
from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
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
    read_observation,
    sources_available,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState
from dexmani_real.utils.log import error_detail

if TYPE_CHECKING:
    from dexmani_real.deployment.observation import PolicyObservationRow

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
        index, output_dir, record_raw, audio, keyboard, camera_snapshot=None,
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
        self.keyboard = keyboard
        self.recorder = None
        self.output_dir, self.record_raw = output_dir, record_raw
        self.result = EpisodeResult(index)
        self.camera_snapshot = camera_snapshot
        self.max_duration_ns = int(max_running_s * 1e9) if max_running_s is not None else None
        self.cancel_epoch = cancel_epoch
        self.history: deque[PolicyObservationRow] = deque(maxlen=policy_info.n_obs_steps)
        self.last_history_timestamp_ns = None
        self.sync_actions = None
        self.sync_cursor = 0
        self.plan: ActionChunk | None = None
        self.query: Prediction | None = None
        self.preparation_issue = None
        self.run_deadline_ns = self.wait_deadline_ns = None
        self.last_slot = -1
        self.dt_ns = round(policy_info.control_dt_s * 1e9)
        self.started_ns = self.next_step_ns = 0
        self.previous_arm = None
        self.first_prediction = True
        self.target_expires_ns = None
        self.discard_requested = False

    @property
    def finished(self):
        return self.result.termination_reason is not None

    def recording_issue(self, row):
        if self.recorder is None or not self.recorder.is_recording:
            return None
        if not self.sensors.check(names=("camera",)):
            return "recording_camera_exited"
        issues = check_observation(row, self.runtime, self.record_needs, now_ns=row.observation_timestamp_ns)
        return "; ".join(issues) if issues else None

    def read_observation(self):
        needs = (self.capture_needs if self.recorder is not None and self.recorder.is_recording
                 else self.control_needs)
        row = read_observation(self.sensors.channels, self.robot, needs=needs)
        issues = check_observation(row, self.runtime, self.control_needs)
        return row, "control_input_" + issues[0] if issues else None

    def budget_limit(self):
        limits = [(deadline, reason) for deadline, reason in (
            (self.run_deadline_ns, "duration"),
            (self.wait_deadline_ns, "wait"),
            (self.target_expires_ns, "wait"),
        ) if deadline is not None]
        return min(limits, key=lambda item: item[0]) if limits else None

    def expired_budget(self):
        limit = self.budget_limit()
        return limit if limit is not None and time.monotonic_ns() >= limit[0] else None

    def stop(self, detail, *, run_end_reason=RunEndReason.CANCELLED, errors=()):
        if self.finished:
            return
        ended_ns = time.monotonic_ns()
        self.runtime_state.revoke_motion()
        self.result.duration_s = (ended_ns - self.started_ns) / 1e9 if self.started_ns else 0.0
        self.result.termination_reason = (
            run_end_reason.name.lower() if self.started_ns
            else "preparation_failed" if self.preparation_issue or errors else "start_cancelled"
        )
        self.result.termination_details = detail if not errors else None
        self.result.errors.extend(errors)
        self.plan = self.query = self.sync_actions = None
        self.history.clear()
        self.previous_arm = self.wait_deadline_ns = self.run_deadline_ns = None
        self.robot.stop()
        if self.runtime_state.estop_request:
            self.audio.play("emergency")

    def finish_recording(self):
        result = self.result
        if self.recorder is not None:
            # 与键盘回调共同锁定首次 finish 的保存意图，等待 writer 时不持锁。
            with self.keyboard.condition:
                self.discard_requested |= any(event.key == "d" for event in self.keyboard.events)
                self.keyboard.events.clear()
                save = not self.discard_requested
            try:
                published = self.recorder.finish(
                    save=save, reason=result.termination_reason,
                    details=result.errors + ([dict(stage="termination", reason=result.termination_details)]
                                             if result.termination_details is not None else []),
                )
            finally:
                result.raw = self.recorder.result()
            if not result.errors and not self.runtime_state.estop_request:
                if published is not None and result.normal_termination and self.recorder.recording_truncated_reason is None:
                    self.audio.play("save")
                elif not save:
                    self.audio.play("discard")

    @property
    def writer_pending(self):
        return self.recorder is not None and self.recorder.writer_pending

    def run(self, *, poll_operator):
        """一段控制及其收尾顺序完成；返回后不再依赖外层 tick 回收任务。"""
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
        finally:
            primary_failure = sys.exc_info()[1]
            failure = primary_failure
            if failure is not None or self.runtime_state.quit_requested or self.runtime_state.estop_request:
                try:
                    self.robot.close()
                except BaseException as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    self.result.errors.append(error_detail("robot_close", exc))
                    if failure is None:
                        failure = exc
            # stop() 已撤权并请求设备停止；writer 独占自己的完整收尾预算。
            # Raw 记录执行结束状态，之后的模型回收错误仅进入 episode/session 结果。
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
                    if failure is not None:
                        logger.exception("episode %s also failed", stage)
                    self.audio.play("emergency")
                    if failure is None:
                        failure = exc
                        try:
                            self.robot.close()
                        except BaseException as close_error:
                            if isinstance(close_error, KeyboardInterrupt):
                                self.runtime_state.request_emergency_stop()
                            self.result.errors.append(error_detail("robot_close", close_error))
            if failure is not None and primary_failure is None:
                raise failure
        return self.result

    def preparation_valid(self):
        valid = self.runtime_state.can_move(
            cancel_epoch=self.cancel_epoch,
            required_safety_state=SafetyState.ARMED,
        )
        if not valid:
            self.robot.check()
        return valid

    def start_observation(self):
        until = time.monotonic_ns() + int(INITIAL_STATE_TIMEOUT_S * 1e9)
        while True:
            if not self.preparation_valid():
                raise OperationCancelled("initial observation cancelled")
            self.robot.service_feedback()
            row, self.preparation_issue = self.read_observation()
            if sources_available(row, self.capture_needs):
                break
            if time.monotonic_ns() >= until:
                raise TimeoutError("initial observation unavailable")
            time.sleep(0.005)
        if self.preparation_issue is not None:
            logger.warning("本轮未开始：%s", self.preparation_issue)
            return None
        if np.max(np.abs(row.arm["qpos"][0] - self.runtime.arm.home_qpos)) > np.deg2rad(
            START_ARM_TOLERANCE_DEG
        ) or np.max(
            np.abs(row.hand["qpos"][0] - np.deg2rad(self.runtime.hand.home_qpos_deg))
        ) > np.deg2rad(
            self.runtime.hand.home_tolerance_deg
        ):
            self.preparation_issue = "start_requires_home_pose"
            logger.warning("本轮未开始：机械臂和手需在初始位，请先按 H。")
            return None
        return row

    def prepare(self, poll_operator):
        logger.info("Policy preparing")
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled before preparation")
        self.robot.prepare_streaming(self.cancel_epoch)
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled during streaming preparation")
        if self.record_raw:
            self.recorder = EpisodeRecorder(self.output_dir, control_hz=1 / self.policy_info.control_dt_s)
            metadata = snapshot_recording_metadata(
                self.camera_snapshot,
                tactile_reset_metadata=self.robot.hand.get_tactile_reset_metadata(),
                tactile_hold_max_age_s=self.runtime.hand.feedback_max_age_s,
            )
            self.recorder.start(
                task_label=self.policy_info.task_name,
                episode_name=f"episode_{self.result.index:03d}", collection_source="policy_rollout", metadata=metadata,
            )
        poll_operator()
        if not self.preparation_valid():
            raise OperationCancelled("cancelled during recorder start")
        # 设备准备和 writer 启动后重新取得最终快照。
        row = self.start_observation()
        if row is None:
            self.stop(self.preparation_issue)
            return
        if self.recorder is not None:
            if (issue := self.recording_issue(row)) is not None:
                raise RecordingError("recording startup: " + issue)
        started_ns = self.runtime_state.begin_motion(cancel_epoch=self.cancel_epoch)
        if started_ns is None:
            self.robot.check()
            raise OperationCancelled("final admission declined")
        self.started_ns = started_ns
        self.next_step_ns = self.started_ns
        self.wait_deadline_ns = (
            self.started_ns + (self.policy_info.n_obs_steps - 1) * self.dt_ns + PREDICTION_WAIT_TIMEOUT_NS
        )
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
        # 不能用新等待窗口续期已经耗尽的 bootstrap/预测预算。
        if self.expired_budget() is not None:
            return "prediction_budget_exhausted"
        submitted_ns = time.monotonic_ns()
        if self.plan is None and self.wait_deadline_ns is None:
            limits = [submitted_ns + PREDICTION_WAIT_TIMEOUT_NS]
            if self.target_expires_ns is not None:
                limits.append(self.target_expires_ns)
            self.wait_deadline_ns = min(limits)
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=handoff_slot - slot,
                 rtc_guidance_cap=self.execution.rtc_guidance_cap) if prefix is not None else {}
        )
        rows = tuple(self.history)
        if self.expired_budget() is not None:
            return "prediction_budget_exhausted"
        self.model.predict(rows, reset_episode=self.first_prediction, **kwargs)
        self.first_prediction = False
        if self.execution.mode != "sync":
            self.query = Prediction(slot, handoff_slot)
        return None

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
        if len(self.history) < self.policy_info.n_obs_steps:
            return "prefetch_history_incomplete" if self.execution.mode == "rtc" else None
        # 与执行使用同一规范化策略动作；不提前映射未来 RobotCommand。
        prefix = (
            plan.actions[slot - plan.anchor_slot:].copy()
            if self.execution.mode == "rtc" else None
        )
        return self.submit(slot, handoff_slot=handoff_slot, prefix=prefix)

    def stop_rejection(self, detail):
        expired = self.expired_budget()
        if expired is not None:
            self.stop_budget(expired)
        else:
            self.stop(detail)

    def stop_budget(self, limit):
        if limit[1] == "wait":
            raise TimeoutError("prediction wait budget expired")
        self.stop("duration_timeout", run_end_reason=RunEndReason.DURATION_COMPLETE)

    def step(self):
        row = attempted_command = None
        try:
            if self.finished:
                return
            motion_allowed = self.runtime_state.can_move(cancel_epoch=self.cancel_epoch)
            if not motion_allowed:
                self.robot.check()
                if (self.runtime_state.cancel_epoch != self.cancel_epoch
                        or not self.runtime_state.requests_allow_motion()):
                    raise OperationCancelled("owner_authority_revoked")
            prediction_ready = self.poll_model()
            if self.recorder is not None:
                self.recorder.check_error()
            if (limit := self.expired_budget()) is not None:
                self.stop_budget(limit)
                return
            if not motion_allowed:
                raise OperationCancelled("owner_authority_revoked")

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
            row, issue = self.read_observation()
            if not sources_available(row, self.control_needs):
                row = None
                return
            if issue is not None:
                self.stop(issue)
                return
            if waiting_sync:
                # 当前输入持续裁决；等待行是评估事实，不重复插入模型历史。
                return
            if not (synchronous and prediction_ready):
                if self.policy_info.n_obs_steps > 1:
                    # sync 等待预测时不追加样本；只比较真正进入 history 的采样时刻。
                    # 半周期余量容忍小幅抖动，明确漏槽则直接重建连续窗口。
                    gap = (
                        row.control_timestamp_ns - self.last_history_timestamp_ns > 1.5 * self.dt_ns
                        if self.last_history_timestamp_ns is not None else False
                    )
                    if gap:
                        self.history.clear()
                    self.last_history_timestamp_ns = row.control_timestamp_ns
                self.history.append(self.observations.project(row))

            if synchronous:
                actions = self.sync_actions
                if actions is None:
                    if len(self.history) == self.policy_info.n_obs_steps:
                        issue = self.submit(None)
                        if issue is not None:
                            self.stop_rejection(issue)
                    return
                action_count = min(len(actions), self.execution.action_steps)
                action = actions[self.sync_cursor]
            else:
                new_plan, issue = handoff_plan(self.query, slot=slot, mode=self.execution.mode)
                if issue == "handoff_prediction_exhausted" and self.execution.mode == "async":
                    self.query = None
                    if self.plan is not None and plan_rejection(
                        self.plan, slot, self.execution.mode, self.execution.action_steps,
                    ) is not None:
                        self.plan = None
                    issue = None
                if issue is not None:
                    self.stop_rejection(issue)
                    return
                if new_plan is not None:
                    if self.execution.mode == "rtc":
                        new_plan.actions = self.action_mapper.normalize_policy_actions(
                            new_plan.actions, self.policy_info.action_mode,
                        )
                    self.plan, self.query = new_plan, None
                if self.plan is None:
                    if len(self.history) == self.policy_info.n_obs_steps and self.query is None:
                        issue = self.submit(slot)
                        if issue is not None:
                            self.stop_rejection(issue)
                    return

                issue = plan_rejection(self.plan, slot, self.execution.mode, self.execution.action_steps)
                if issue is None:
                    issue = self.submit_prefetch(self.plan, slot)
                if issue is not None:
                    self.stop_rejection(issue)
                    return
                action = self.plan.actions[slot - self.plan.anchor_slot]
            command, issue = self.map_command(action, row)
            if issue is not None:
                self.stop_rejection(issue)
                return
            # 长计算后重查控制来源；设备最终只收到目标租期。
            now = time.monotonic_ns()
            if (limit := self.expired_budget()) is not None:
                self.stop_budget(limit)
                return
            source_issues = check_observation(row, self.runtime, self.control_needs)
            if source_issues:
                self.stop("control_sources: " + "; ".join(source_issues))
                return
            if not synchronous and now >= self.next_step_ns:
                if self.execution.mode == "rtc":
                    self.stop("committed_slot_submission_missed")
                # async 下一 tick 消费仍有效的时间窗口。
                return
            expires_ns = now + COMMAND_LEASE_NS
            attempted_command = command
            self.robot.send_action(command, expires_ns=expires_ns)
            if self.previous_arm is None:
                logger.info("Policy first target submitted")
                self.audio.play("begin")
            self.previous_arm = command.arm_qpos
            self.target_expires_ns = expires_ns
            self.wait_deadline_ns = None
            if synchronous:
                self.sync_cursor += 1
                if self.sync_cursor == action_count:
                    self.sync_actions = None
        finally:
            # 每个真实采样与至多一次提交尝试只记一行；部分失败不改写为成功。
            primary_error = sys.exc_info()[1]
            if primary_error is not None:
                self.runtime_state.revoke_motion()
            if row is not None and self.recorder is not None:
                try:
                    self.recorder.append(row, attempted_command, quality_issue=self.recording_issue(row))
                except BaseException as exc:
                    if primary_error is None:
                        raise
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    self.recorder.store_error(exc)
