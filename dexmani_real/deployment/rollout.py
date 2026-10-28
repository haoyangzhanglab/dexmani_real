"""单轮策略执行：准备、预测、控制与录制收尾。"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np

from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.config import (
    PREDICTION_TIMEOUT_NS,
    START_ARM_TOLERANCE_DEG,
)
from dexmani_real.deployment.schedule import (
    Prediction,
    control_slot,
    handoff_plan,
    plan_rejection,
    prefetch_handoff,
)
from dexmani_real.geometry.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.ipc.sensors import read_camera_frame
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState
from dexmani_real.utils.log import error_detail
from dexmani_real.utils.timing import next_tick_ns

logger = logging.getLogger(__name__)


def failure_reason(error):
    if isinstance(error, KeyboardInterrupt):
        return RunEndReason.ESTOP
    if isinstance(error, RecordingError):
        return RunEndReason.RECORDING_FAILURE
    if isinstance(error, RobotError):
        return RunEndReason.HARDWARE_FAULT
    return RunEndReason.POLICY_FAILURE


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
    metrics: dict = field(default_factory=lambda: dict(
        inference_count=0, inference_mean_s=None, inference_max_s=None, skipped_slots=0,
    ))

    def record_prediction(self, duration_ns):
        metrics = self.metrics
        count = metrics["inference_count"]
        mean = metrics["inference_mean_s"] or 0.0
        duration = duration_ns / 1e9
        metrics.update(
            inference_count=count + 1,
            inference_mean_s=mean + (duration - mean) / (count + 1),
            inference_max_s=max(metrics["inference_max_s"] or 0.0, duration),
        )


class Rollout:
    def __init__(
        self, runtime_state, runtime, policy_info, *, preparing_epoch, robot,
        observations, record_needs, needs, sensors, action_mapper, model, max_running_s,
        capture_ids, index, output_dir, record_raw, camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.policy_info = policy_info
        self.execution = runtime.execution
        self.robot, self.sensors = robot, sensors
        self.observations = observations
        self.control_needs, self.record_needs = observations.needs, record_needs
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
        self.preparing_epoch = preparing_epoch
        self.run_id = None
        self.history = deque(maxlen=policy_info.n_obs_steps)
        self.hand_history = (
            deque(maxlen=policy_info.hand_history_steps)
            if policy_info.hand_history_steps is not None else None
        )
        self.plan = self.query = None
        self.preparation_issue = None
        self.run_deadline_ns = self.wait_deadline_ns = None
        self.last_slot = -1
        self.dt_ns = round(policy_info.control_dt_s * 1e9)
        self.max_decision_age_ns = int(self.execution.max_decision_age_s * 1e9)
        self.started_ns = self.next_step_ns = 0
        self.previous_arm = None
        # sync 推理期间持有本步观测，结果到达后只提交/录制一次。
        self.sync_row = None

    @property
    def finished(self):
        return self.result.termination_reason is not None

    def read_observation(self):
        row = read_observation(
            self.sensors.channels, self.robot, needs=self.needs,
        )
        issues, feedback_until_ns = check_observation(row, self.runtime, self.control_needs)
        if not issues:
            if self.recorder is not None:
                record_issues, _ = check_observation(row, self.runtime, self.record_needs)
                if record_issues:
                    raise RecordingError("recording sources: " + "; ".join(record_issues))
        return row, "control_input_" + issues[0] if issues else None, feedback_until_ns

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

    def has_motion_authority(self):
        return self.runtime_state.can_move(run_id=self.run_id)

    def stop(self, detail, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, errors=()):
        if self.finished:
            return
        epoch = self.run_id
        ended_ns = time.monotonic_ns()
        self.runtime_state.revoke_motion(
            reason=run_end_reason,
            expected_run_id=epoch if epoch is not None else self.preparing_epoch,
        )
        with self.runtime_state.motion_lock:
            stopped_epoch = self.runtime_state.run_id
            reason = (
                self.runtime_state.run_ended_reason
                if epoch is not None and self.runtime_state.run_ended_id == epoch else run_end_reason
            )
        self.result.duration_s = (ended_ns - self.started_ns) / 1e9 if epoch is not None else 0.0
        self.result.termination_reason = (
            reason.name.lower() if epoch is not None
            else "preparation_failed" if self.preparation_issue or errors else "start_cancelled"
        )
        self.result.termination_details = detail
        self.result.errors.extend(errors)
        self.run_id = self.plan = None
        if not self.model.has_pending_task:
            self.query = self.preparing_epoch = None
        self.history.clear()
        if self.hand_history is not None:
            self.hand_history.clear()
        self.previous_arm = self.wait_deadline_ns = self.run_deadline_ns = None
        try:
            if epoch is not None or self.robot.stop_required:
                self.robot.stop()
            self.runtime_state.acknowledge_stop(stopped_epoch)
        except BaseException as exc:
            logger.exception("robot stop failed")
            self.result.errors.append(error_detail("stop", exc))
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            if epoch is None:
                self.result.termination_reason = "preparation_failed"
            raise

    def finish(self):
        result = self.result
        normal = result.termination_reason in {"operator", "quit"} or (
            result.termination_reason == "timeout" and result.termination_details == "duration_timeout"
        )
        if self.recorder is not None and self.recorder.start_attempted:
            failure = None
            if self.sync_row is not None:
                row, self.sync_row = self.sync_row, None
                try:
                    self.recorder.append(row)
                except BaseException as exc:
                    result.errors.append(error_detail("recording", exc))
                    failure = exc
            try:
                self.recorder.finish(
                    save=True, reason=result.termination_reason,
                    details=[dict(stage="termination", reason=result.termination_details), *result.errors],
                    technical_status="normal" if normal and not result.errors else "failed",
                )
            except BaseException as exc:
                result.errors.append(error_detail("recorder_finalize", exc))
                if failure is not None:
                    logger.exception("recording finalization also failed")
                failure = failure or exc
            finally:
                result.raw = self.recorder.result()
            if failure is not None:
                raise failure

    def close(self):
        if self.recorder is not None:
            try:
                self.recorder.close()
            finally:
                self.refresh_raw()

    def refresh_raw(self):
        if self.recorder is not None:
            self.result.raw = self.recorder.result()

    @property
    def writer_pending(self):
        return self.recorder is not None and self.recorder.writer_pending

    def preparation_valid(self):
        return self.runtime_state.can_move(
            run_id=self.preparing_epoch,
            required_safety_state=SafetyState.ARMED,
        )

    def start_observation(self):
        self.robot.service_feedback()
        row, self.preparation_issue, _ = self.read_observation()
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

    def prepare(self):
        try:
            self.robot.reset_hand_feedback()
            self.robot.read_initial_state(cancel_requested=lambda: not self.preparation_valid())
            if not self.preparation_valid():
                raise OperationCancelled("cancelled during initial feedback acquisition")
            row = self.start_observation()
        except OperationCancelled as exc:
            self.stop(str(exc))
            return
        if row is None:
            self.stop(self.preparation_issue)
            return
        logger.info("正在准备本轮模型…")
        self.model.reset()

    def admit(self):
        if not self.preparation_valid():
            self.stop("cancelled during model reset")
            return
        epoch = self.preparing_epoch
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
        # 录制启动和设备模式恢复可能阻塞，之后分别读取新反馈。
        if self.start_observation() is None:
            self.stop(self.preparation_issue)
            return
        self.robot.prepare_streaming(epoch, service=self.robot.service_feedback)
        if self.start_observation() is None:
            self.stop(self.preparation_issue)
            return
        with self.runtime_state.motion_lock:
            admitted = self.runtime_state.begin_motion() if epoch == self.runtime_state.run_id else None
        if admitted is None:
            self.stop("final admission declined")
            return
        self.run_id, self.started_ns = admitted
        self.preparing_epoch = None
        self.next_step_ns = self.started_ns
        self.wait_deadline_ns = self.started_ns + PREDICTION_TIMEOUT_NS
        self.run_deadline_ns = (
            self.started_ns + self.max_duration_ns if self.max_duration_ns is not None else None
        )
        self.result.started_utc = utc_now()

    def poll_model(self):
        try:
            item = self.model.poll()
        except BaseException as exc:
            if self.finished:
                self.result.errors.append(error_detail("retired_model", exc))
            raise
        if item is None:
            return
        operation, result = item
        if self.finished:
            self.query = self.preparing_epoch = None
            return
        if operation == "reset":
            if self.preparation_valid():
                self.admit()
            else:
                self.stop("cancelled during model reset")
            return
        if not self.has_motion_authority():
            self.stop("owner_authority_revoked")
            self.query = None
            return
        self.result.record_prediction(result.duration_ns)
        self.query.result = result.actions

    def submit(self, slot, *, handoff_slot=None, prefix=None, commands=None):
        now = time.monotonic_ns()
        authorized, expired = self.has_motion_authority(), self.expired_budget()
        if expired is not None:
            return f"{expired[1]}_timeout"
        if not authorized:
            return "query_authority_revoked"
        rows = tuple(self.history)
        source_ns = rows[-1].source_timestamp_ns
        # 来源时间已校验；+1 保持 age <= budget 的整数纳秒边界。
        expires_at_ns = source_ns + self.max_decision_age_ns + 1
        if now >= expires_at_ns:
            return "query_sources_invalid_or_expired"
        if prefix is not None and now >= self.plan.expires_at_ns:
            return "prefix_decision_expired"
        query = Prediction(self.run_id, slot, expires_at_ns, handoff_slot)
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=handoff_slot - slot,
                 rtc_guidance_cap=self.execution.rtc_guidance_cap) if prefix is not None else {}
        )
        initial_hand_history = None
        if self.hand_history is not None:
            if not self.hand_history:
                # 与训练时一致，初始历史使用 float32 精度的实测手姿。
                hand = rows[-1].values["hand_qpos"].astype(np.float32).astype(np.float64)
                initial_hand_history = tuple(
                    hand.copy() for _ in range(self.hand_history.maxlen)
                )
            # 推理持有独立历史快照，不受后续提交或重置影响。
            kwargs["action_context"] = {
                "hand_history": np.stack(
                    self.hand_history if initial_hand_history is None else initial_hand_history
                )
            }
        self.model.predict(rows, **kwargs)
        if initial_hand_history is not None:
            self.hand_history.extend(initial_hand_history)
        self.query = query
        if commands is not None:
            self.plan.frozen.update(commands)
        return None

    def prepare_prefix(self, row, slot):
        """先计算引导尾部与冻结命令，预测成功提交后再提交冻结状态。"""
        prefix = self.plan.actions[slot - self.plan.anchor_slot:].copy()
        slots = range(slot, slot + self.execution.prefetch_steps)
        commands, previous = {}, self.previous_arm
        for index, command_slot in enumerate(slots):
            self.robot.check()
            if not self.has_motion_authority():
                return None, None, "prefix_authority_revoked"
            expired = self.expired_budget()
            if expired is not None:
                return None, None, f"{expired[1]}_timeout"
            command, issue = self.map_command(
                prefix[index], row, previous, context="prefix_action"
            )
            if issue is not None:
                return None, None, issue
            commands[command_slot] = command
            previous = command.arm_qpos
            prefix[index] = self.action_mapper.control_from_command(
                command, self.policy_info.action_mode
            )
        return prefix, commands, None

    def map_command(self, action, row, previous, *, context="action"):
        action = np.asarray(action, dtype=np.float64)
        mode = self.policy_info.action_mode
        current = row.arm["qpos"][0]
        if mode == "joint":
            decoded = self.action_mapper.map_joint(action[:7], action[7:], current)
        else:
            pose = Pose(action[:3], rot6d_to_quat_wxyz(action[3:9]))
            decoded = self.action_mapper.map_eef(pose, action[9:], current, previous)
        if decoded.rejection_reason is not None or decoded.arm_qpos is None:
            return None, f"{context}_rejected: {decoded.rejection_reason}"
        return RobotCommand(self.run_id, decoded.arm_qpos, decoded.hand_qpos), None

    def command_for_slot(self, row, slot):
        plan = self.plan
        command = plan.frozen.get(slot)
        if command is not None:
            rejection = self.action_mapper.frozen_rejection(
                command, row.arm["qpos"][0], self.policy_info.action_mode
            )
            if rejection is not None:
                return None, f"frozen_target_rejected: {rejection}"
        else:
            return self.map_command(
                plan.actions[slot - plan.anchor_slot],
                row,
                self.previous_arm,
            )
        return command, None

    def submit_prefetch(self, row, slot):
        handoff_slot, issue = prefetch_handoff(
            self.plan, slot=slot, mode=self.execution.mode,
            prefetch_steps=self.execution.prefetch_steps,
            query_pending=self.query is not None,
        )
        if issue is not None or handoff_slot is None:
            return issue
        prefix = commands = None
        if self.execution.mode == "rtc":
            prefix, commands, issue = self.prepare_prefix(row, slot)
            if issue is not None:
                return issue
        return self.submit(
            slot, handoff_slot=handoff_slot, prefix=prefix, commands=commands
        )

    def stop_rejection(self, detail):
        expired = self.expired_budget()
        if expired is not None:
            self.stop(f"{expired[1]}_timeout", run_end_reason=RunEndReason.TIMEOUT)
        else:
            self.stop(detail)

    def step(self):
        row = attempted_command = None
        failure = None
        try:
            if self.finished:
                self.poll_model()
                return
            if self.run_id is None and not self.model.has_pending_task:
                self.prepare()
                return
            if self.run_id is None and not self.preparation_valid():
                self.stop("cancelled during model reset")
                return
            if self.run_id is not None and not self.has_motion_authority():
                self.stop("owner_authority_revoked")
                return
            self.poll_model()
            if self.finished or self.run_id is None:
                return
            if self.recorder is not None:
                self.recorder.check_error()
            if (limit := self.expired_budget()) is not None:
                self.stop(f"{limit[1]}_timeout", run_end_reason=RunEndReason.TIMEOUT)
                return

            now = time.monotonic_ns()
            synchronous = self.execution.mode == "sync"
            if synchronous and self.sync_row is not None:
                if self.query.result is None and now < self.next_step_ns:
                    return
                # 仅复核当前反馈和录制来源健康；不采集策略观测或替换待录制行。
                execution_state = self.robot.read_state()
                issues, feedback_until_ns = check_observation(
                    execution_state, self.runtime, ObservationNeeds(),
                )
                if issues:
                    self.stop("control_input_" + issues[0])
                    return
                if self.recorder is not None:
                    camera = (
                        read_camera_frame(self.sensors.channels, fields=())
                        if self.record_needs.camera_fields else None
                    )
                    record_state = replace(
                        self.sync_row, arm=execution_state.arm, hand=execution_state.hand, camera=camera,
                    )
                    record_issues, _ = check_observation(record_state, self.runtime, self.record_needs)
                    if record_issues:
                        raise RecordingError("recording sources: " + "; ".join(record_issues))
                if now >= self.next_step_ns:
                    self.result.metrics["skipped_slots"] += 1 + (now - self.next_step_ns) // self.dt_ns
                    self.next_step_ns = next_tick_ns(self.next_step_ns, self.dt_ns, now)
                if self.query.result is None:
                    return
                row, self.sync_row = self.sync_row, None
                slot = self.last_slot
            else:
                slot, self.next_step_ns, skipped = control_slot(
                    now_ns=now, next_step_ns=self.next_step_ns, last_slot=self.last_slot,
                    dt_ns=self.dt_ns, synchronous=synchronous,
                )
                if slot is None:
                    return
                self.result.metrics["skipped_slots"] += skipped
                if skipped and self.execution.mode == "rtc":
                    self.stop(f"missed_slot: expected={self.last_slot + 1}, actual={slot}")
                    return
                self.last_slot = slot
                row, issue, feedback_until_ns = self.read_observation()
                self.poll_model()
                if not self.has_motion_authority():
                    self.stop("observation_authority_revoked")
                    return
                if issue is not None:
                    self.stop(issue)
                    return
                self.history.append(self.observations.project(row))
                execution_state = row

            new_plan, issue = handoff_plan(
                self.query, run_id=self.run_id, slot=slot, mode=self.execution.mode,
                action_steps=self.execution.action_steps, now_ns=time.monotonic_ns(),
                authorized=self.has_motion_authority(),
            )
            if issue is not None:
                self.stop_rejection(issue)
                return
            if new_plan is not None:
                self.plan, self.query = new_plan, None
            if self.plan is None:
                if self.wait_deadline_ns is None:
                    self.wait_deadline_ns = now + PREDICTION_TIMEOUT_NS
                    self.robot.arm.limit_execution_until(self.wait_deadline_ns)
                if len(self.history) == self.history.maxlen and self.query is None:
                    issue = self.submit(slot)
                    if issue is not None:
                        self.stop_rejection(issue)
                    elif synchronous:
                        self.sync_row, row = row, None
                return

            issue = plan_rejection(self.plan, slot, self.execution.mode, time.monotonic_ns())
            if issue is None:
                issue = self.submit_prefetch(row, slot)
            if issue is not None:
                self.stop_rejection(issue)
                return
            command, issue = self.command_for_slot(execution_state, slot)
            if issue is not None:
                self.stop_rejection(issue)
                return
            if not self.has_motion_authority():
                self.stop("before_submission_revoked")
                return
            limit = self.budget_limit()
            if limit is not None and time.monotonic_ns() >= limit[0]:
                self.stop(f"{limit[1]}_timeout", run_end_reason=RunEndReason.TIMEOUT)
                return
            action_deadline_ns = min(self.next_step_ns, feedback_until_ns, self.plan.expires_at_ns)
            valid_until_ns = min(action_deadline_ns, limit[0]) if limit else action_deadline_ns
            if time.monotonic_ns() >= valid_until_ns:
                self.stop("submission_deadline_expired")
                return
            execute_until_ns = self.plan.expires_at_ns
            if self.run_deadline_ns is not None:
                execute_until_ns = min(execute_until_ns, self.run_deadline_ns)

            attempted_command = command
            try:
                self.robot.send_action(command, valid_until_ns=valid_until_ns, execute_until_ns=execute_until_ns)
            except OperationCancelled as exc:
                self.stop(str(exc))
                return
            except TimeoutError as exc:
                self.stop(str(exc), errors=[error_detail("submission", exc)])
                return
            now = time.monotonic_ns()
            if not self.has_motion_authority():
                self.stop("motion authority revoked during submission")
                return
            if now >= action_deadline_ns:
                error = TimeoutError("submission returned after action deadline")
                self.stop(str(error), errors=[error_detail("submission", error)])
                return
            if limit is not None and now >= limit[0]:
                self.stop(f"{limit[1]}_timeout", run_end_reason=RunEndReason.TIMEOUT)
                return
            self.previous_arm = command.arm_qpos
            if self.hand_history is not None:
                # 只在提交正常返回后推进动作历史。
                self.hand_history.append(command.hand_qpos.copy())
            self.wait_deadline_ns = None
            if slot + 1 == self.plan.segment_end and self.execution.mode == "sync":
                self.plan = None
        except BaseException as exc:
            preparing = self.run_id is None and not self.finished
            cancelled = preparing and isinstance(exc, OperationCancelled)
            failure = None if cancelled else exc
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            if not self.finished:
                stage = "preparation" if preparing else "control"
                try:
                    self.stop(
                        f"{stage}_failed: {type(exc).__name__}: {exc}", run_end_reason=failure_reason(exc),
                        errors=[] if cancelled or isinstance(exc, KeyboardInterrupt) else [error_detail(stage, exc)],
                    )
                except BaseException as stop_error:
                    failure = failure or stop_error
        finally:
            # 本步只尝试入队一次；终止时已先撤权和请求停止。
            if row is None and self.finished:
                row, self.sync_row = self.sync_row, None
            try:
                if row is not None and self.recorder is not None:
                    self.recorder.append(row, attempted_command)
            except BaseException as exc:
                failure = failure or exc
                self.result.errors.append(error_detail("recording", exc))
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                try:
                    self.stop(str(exc), run_end_reason=failure_reason(exc))
                except BaseException as stop_error:
                    failure = failure or stop_error
            try:
                if (limit := self.expired_budget()) is not None:
                    self.stop(f"{limit[1]}_timeout", run_end_reason=RunEndReason.TIMEOUT)
            except BaseException as exc:
                failure = failure or exc
            if failure is not None:
                raise failure
