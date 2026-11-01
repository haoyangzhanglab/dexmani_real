"""单轮策略执行：准备、预测、控制与录制收尾。"""

import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from dexmani_real.config.deployment import (
    ACTION_TIMEOUT_NS,
    START_ARM_TOLERANCE_DEG,
)
from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.schedule import (
    Prediction,
    control_slot,
    handoff_plan,
    plan_rejection,
    prefetch_handoff,
    window_end,
)
from dexmani_real.geometry.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.runtime.lifecycle import SHUTDOWN_TIMEOUT_S
from dexmani_real.runtime.observation import (
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState, device_failure_reason
from dexmani_real.utils.log import error_detail

logger = logging.getLogger(__name__)


def failure_reason(error):
    if isinstance(error, RecordingError):
        return RunEndReason.RECORDING_FAILURE
    return device_failure_reason(error) or RunEndReason.POLICY_FAILURE


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
        self, runtime_state, runtime, policy_info, *, epoch, robot,
        observations, needs, sensors, action_mapper, model, max_running_s,
        capture_ids, index, output_dir, record_raw, camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.policy_info = policy_info
        self.execution = runtime.execution
        self.robot, self.sensors = robot, sensors
        self.observations = observations
        self.needs = needs
        self.action_mapper, self.model = action_mapper, model
        self.recorder = (
            EpisodeRecorder(output_dir, control_hz=1.0 / policy_info.control_dt_s)
            if record_raw else None
        )
        self.result = EpisodeResult(index)
        self.camera_snapshot = camera_snapshot
        self.max_duration_ns = int(max_running_s * 1e9) if max_running_s is not None else None
        self.capture_ids = capture_ids
        self.epoch = epoch
        self.history = deque(maxlen=policy_info.n_obs_steps)
        self.plan = self.query = None
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

    def read_observation(self):
        row = read_observation(self.sensors.channels, self.robot, needs=self.needs)
        issues = check_observation(row, self.runtime, self.needs)
        return row, "control_input_" + issues[0] if issues else None

    def budget_limit(self):
        if self.wait_deadline_ns is not None and (
            self.run_deadline_ns is None or self.wait_deadline_ns <= self.run_deadline_ns
        ):
            return self.wait_deadline_ns, "wait"
        if self.run_deadline_ns is not None:
            return self.run_deadline_ns, "duration"
        return None

    def expired_budget(self):
        limit = self.budget_limit()
        return limit if limit is not None and time.monotonic_ns() >= limit[0] else None

    def stop(self, detail, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, errors=()):
        if self.finished:
            return
        ended_ns = time.monotonic_ns()
        self.runtime_state.revoke_motion(
            reason=run_end_reason,
            expected_run_id=self.epoch,
        )
        with self.runtime_state.motion_lock:
            reason = (
                self.runtime_state.run_ended_reason
                if self.started_ns and self.runtime_state.run_ended_id == self.epoch else run_end_reason
            )
        self.result.duration_s = (ended_ns - self.started_ns) / 1e9 if self.started_ns else 0.0
        self.result.termination_reason = (
            reason.name.lower() if self.started_ns
            else "preparation_failed" if self.preparation_issue or errors else "start_cancelled"
        )
        self.result.termination_details = detail
        self.result.errors.extend(errors)
        self.plan = None
        if not self.model.has_pending_task:
            self.query = None
        self.history.clear()
        self.previous_arm = self.wait_deadline_ns = self.run_deadline_ns = None
        if self.started_ns or self.robot.stop_required:
            self.robot.stop()

    def finish_recording(self, *, deadline):
        result = self.result
        if self.recorder is not None and self.recorder.start_attempted:
            try:
                self.recorder.finish(
                    save=not self.discard_requested, reason=result.termination_reason,
                    details=[dict(stage="termination", reason=result.termination_details), *result.errors],
                    technical_status="normal" if result.normal_termination else "failed",
                    deadline=deadline,
                )
            except BaseException as exc:
                result.errors.append(error_detail("recorder_finalize", exc))
                raise
            finally:
                result.raw = self.recorder.result()

    @property
    def writer_pending(self):
        return self.recorder is not None and self.recorder.writer_pending

    def run(self, *, keyboard, poll_operator):
        """一段控制及其收尾顺序完成；返回后不再依赖外层 tick 回收任务。"""
        try:
            try:
                poll_operator()
                if self.preparation_valid():
                    self.prepare(poll_operator)
                else:
                    self.stop("cancelled before preparation")
                while not self.finished:
                    poll_operator()
                    self.robot.service_feedback()
                    self.step()
                    if not self.finished:
                        time.sleep(0.001)
            except OperationCancelled as exc:
                # 取消不能掩盖同时发生的设备或 worker 故障。
                self.robot.check(operation_error=exc)
                self.stop(str(exc))
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            if not self.finished:
                stage = "control" if self.started_ns else "preparation"
                try:
                    self.stop(
                        str(exc), run_end_reason=failure_reason(exc),
                        errors=[error_detail(stage, exc)],
                    )
                except BaseException as stop_error:
                    if isinstance(stop_error, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    self.result.errors.append(error_detail("stop", stop_error))
                    logger.exception("episode stop also failed")
            else:
                self.result.errors.append(error_detail("stop", exc))
            raise
        finally:
            primary_failure = sys.exc_info()[0] is not None
            cleanup_failure = None
            deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
            # 模型回收失败必须先进入本段结果，Raw 才能保存正确的结束状态。
            # stop() 已撤权；两项收尾共用 deadline，不持 motion_lock。
            try:
                self.model.discard_pending(deadline - time.monotonic())
            except BaseException as exc:
                logger.exception("retired episode model task failed")
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                self.result.errors.append(error_detail("retired_model", exc))
                cleanup_failure = exc
            finally:
                try:
                    self.finish_recording(deadline=deadline)
                except BaseException as exc:
                    logger.exception("episode recording finalization failed")
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    cleanup_failure = cleanup_failure or exc
            if not self.model.has_pending_task:
                self.query = None
            keyboard.discard_events("b", "h", "t")
            if cleanup_failure is not None and not primary_failure:
                raise cleanup_failure
        return self.result

    def preparation_valid(self):
        valid = self.runtime_state.can_move(
            run_id=self.epoch,
            required_safety_state=SafetyState.ARMED,
        )
        if not valid:
            self.robot.check()
        return valid

    def start_observation(self):
        self.robot.service_feedback()
        row, self.preparation_issue = self.read_observation()
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
        if self.recorder is not None:
            metadata = snapshot_recording_metadata(
                self.camera_snapshot,
                tactile_reset_metadata=self.robot.hand.get_tactile_reset_metadata(),
                tactile_hold_max_age_s=self.runtime.hand.feedback_max_age_s,
            )
            capture_id = next(self.capture_ids)
            self.recorder.start(
                task_label=self.policy_info.task_name,
                episode_name=f"episode_{capture_id:03d}", collection_source="policy_rollout", metadata=metadata,
            )
        # writer 与模式恢复之后只读一次有效快照，不要求来源晚于 prepare。
        poll_operator()
        if not self.preparation_valid():
            self.stop("cancelled during recorder start")
            return
        self.robot.prepare_streaming(self.epoch)
        poll_operator()
        if not self.preparation_valid():
            self.stop("cancelled during streaming preparation")
            return
        if self.start_observation() is None:
            self.stop(self.preparation_issue)
            return
        with self.runtime_state.motion_lock:
            started_ns = self.runtime_state.begin_motion() if self.epoch == self.runtime_state.run_id else None
        if started_ns is None:
            self.robot.check()
            self.stop("final admission declined")
            return
        self.started_ns = started_ns
        self.next_step_ns = self.started_ns
        self.wait_deadline_ns = (
            self.started_ns + (self.policy_info.n_obs_steps - 1) * self.dt_ns + ACTION_TIMEOUT_NS
        )
        self.run_deadline_ns = (
            self.started_ns + self.max_duration_ns if self.max_duration_ns is not None else None
        )
        self.result.started_utc = utc_now()

    def poll_model(self):
        item = self.model.poll()
        if item is not None:
            _, result = item
            actions = np.asarray(result, dtype=np.float64)
            expected = (self.policy_info.future_steps, self.policy_info.control_action_dim)
            if actions.shape != expected or not np.isfinite(actions).all():
                raise ValueError(f"policy prediction requires finite shape {expected}")
            self.query.result = actions

    def submit(self, slot, *, handoff_slot=None, prefix=None):
        submitted_ns = time.monotonic_ns()
        if self.plan is None:
            limits = [submitted_ns + ACTION_TIMEOUT_NS]
            if self.target_expires_ns is not None:
                limits.append(self.target_expires_ns)
            if self.run_deadline_ns is not None:
                limits.append(self.run_deadline_ns)
            self.wait_deadline_ns = min(limits)
            if submitted_ns >= self.wait_deadline_ns:
                return "prediction_wait_exhausted"
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=handoff_slot - slot,
                 rtc_guidance_cap=self.execution.rtc_guidance_cap) if prefix is not None else {}
        )
        self.model.predict(tuple(self.history), reset_episode=self.first_prediction, **kwargs)
        self.first_prediction = False
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
        return RobotCommand(self.epoch, decoded.arm_qpos, decoded.hand_qpos), None

    def submit_prefetch(self, slot):
        handoff_slot, issue = prefetch_handoff(
            self.plan, slot=slot, mode=self.execution.mode,
            action_steps=self.execution.action_steps,
            prefetch_steps=self.execution.prefetch_steps,
            query_pending=self.query is not None,
        )
        if issue is not None or handoff_slot is None:
            return issue
        # 与执行使用同一规范化策略动作；不提前映射未来 RobotCommand。
        prefix = (
            self.plan.actions[slot - self.plan.anchor_slot:].copy()
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
        self.stop(
            f"{limit[1]}_timeout",
            run_end_reason=(
                RunEndReason.DURATION_COMPLETE if limit[1] == "duration"
                else RunEndReason.PREDICTION_TIMEOUT
            ),
        )

    def step(self):
        row = attempted_command = None
        try:
            if self.finished:
                return
            if not self.runtime_state.can_move(run_id=self.epoch):
                self.robot.check()
                self.stop("owner_authority_revoked")
                return
            self.poll_model()
            if self.recorder is not None:
                self.recorder.check_error()
            if (limit := self.expired_budget()) is not None:
                self.stop_budget(limit)
                return

            now = time.monotonic_ns()
            synchronous = self.execution.mode == "sync"
            waiting = synchronous and self.plan is None and self.query is not None
            if waiting and self.query.result is not None:
                # 结果到达时读取执行快照，给予一个完整提交机会；不再进入模型 history。
                slot = self.last_slot + 1
                self.next_step_ns = now + self.dt_ns
                skipped = 0
            else:
                slot, self.next_step_ns, skipped = control_slot(
                    now_ns=now, next_step_ns=self.next_step_ns, last_slot=self.last_slot,
                    dt_ns=self.dt_ns, synchronous=synchronous,
                )
                if slot is None:
                    return
            self.result.skipped_slots += skipped
            if skipped and self.execution.mode == "rtc" and self.plan is not None:
                self.stop(f"missed_slot: expected={self.last_slot + 1}, actual={slot}")
                return
            if not synchronous:
                self.last_slot = slot
            row, issue = self.read_observation()
            if issue is not None:
                self.stop(issue)
                return
            if waiting and self.query.result is None:
                # 当前输入持续裁决；等待行是评估事实，不重复插入模型历史。
                return
            if not waiting:
                self.history.append(self.observations.project(row))

            new_plan, issue = handoff_plan(self.query, slot=slot, mode=self.execution.mode)
            if issue is not None:
                self.stop_rejection(issue)
                return
            if new_plan is not None:
                if self.execution.mode == "rtc":
                    new_plan.actions = self.action_mapper.normalize_policy_actions(
                        new_plan.actions, self.policy_info.action_mode,
                    )
                self.plan, self.query = new_plan, None
                self.wait_deadline_ns = None
            if self.plan is None:
                if len(self.history) == self.history.maxlen and self.query is None:
                    issue = self.submit(slot)
                    if issue is not None:
                        self.stop_rejection(issue)
                return

            issue = plan_rejection(self.plan, slot, self.execution.mode, self.execution.action_steps)
            if issue is None:
                issue = self.submit_prefetch(slot)
            if issue is not None:
                self.stop_rejection(issue)
                return
            command, issue = self.map_command(self.plan.actions[slot - self.plan.anchor_slot], row)
            if issue is not None:
                self.stop_rejection(issue)
                return
            # 读取时的年龄只裁决一次；tick 机会由主线程检查，设备只收到唯一租期。
            now = time.monotonic_ns()
            if (limit := self.expired_budget()) is not None:
                self.stop_budget(limit)
                return
            if now >= self.next_step_ns:
                if self.execution.mode == "rtc":
                    self.stop("committed_slot_submission_missed")
                # sync 保留动作游标，async 下一 tick 消费仍有效的时间窗口。
                return
            expires_ns = now + ACTION_TIMEOUT_NS
            attempted_command = command
            self.robot.send_action(command, expires_ns=expires_ns)
            self.previous_arm = command.arm_qpos
            self.target_expires_ns = expires_ns
            self.wait_deadline_ns = None
            if synchronous:
                self.last_slot = slot
                if slot + 1 == window_end(self.plan, self.execution.action_steps):
                    self.plan = None
        finally:
            # 每个真实采样与至多一次提交尝试只记一行；部分失败不改写为成功。
            primary_error = sys.exc_info()[1]
            if primary_error is not None:
                self.runtime_state.revoke_motion(
                    reason=failure_reason(primary_error), expected_run_id=self.epoch,
                )
            try:
                if row is not None and self.recorder is not None:
                    self.recorder.append(row, attempted_command)
            except BaseException as exc:
                if primary_error is None:
                    raise
                self.result.errors.append(error_detail("recording", exc))
                logger.exception("episode frame recording also failed")
