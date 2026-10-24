"""One Policy attempt owns preparation, control, capture and its first termination."""

import logging
import time
from collections import deque
from concurrent.futures import CancelledError
from dataclasses import dataclass

import numpy as np

from dexmani_real.data.recorder import (
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.data.results import error_detail
from dexmani_real.deployment.observation import (
    project_policy_row,
)
from dexmani_real.deployment.schedule import (
    ControlRejection,
    PolicyQuery,
    control_slot,
    decision_expires_at_ns,
    handoff_plan,
    plan_rejection,
    prefetch_handoff,
    rtc_prefix_request,
    run_budget,
)
from dexmani_real.planning.action import ActionIntent
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.planning.kinematics.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.robot.commands import OperationCancelled, RobotError, RobotCommand
from dexmani_real.runtime.observation import (
    feedback_deadline_ns, read_observation, observation_issues, control_deadline_ns, service_feedback,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState

logger = logging.getLogger(__name__)


def failure_reason(error):
    if isinstance(error, KeyboardInterrupt):
        return RunEndReason.ESTOP
    if isinstance(error, RecordingError):
        return RunEndReason.RECORDING_FAILURE
    if isinstance(error, RobotError):
        return RunEndReason.HARDWARE_FAULT
    return RunEndReason.POLICY_FAILURE


@dataclass(frozen=True)
class EpisodeEnd:
    run_id: int | None
    reason: str
    detail: str
    duration_s: float


class PolicyEpisode:
    def __init__(
        self,
        runtime_state,
        runtime,
        policy_info,
        *,
        preparing_epoch,
        robot,
        control_needs,
        record_needs,
        sensors,
        action_mapper,
        model_runtime,
        max_running_s,
        query_ids,
        capture_ids,
        recorder=None,
        camera_snapshot=None,
        results=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.policy_info = policy_info
        self.execution = runtime.execution
        self.robot = robot
        self.sensors = sensors
        self.control_needs, self.record_needs = control_needs, record_needs
        self.needs = control_needs.union(record_needs)
        self.action_mapper = action_mapper
        self.model = model_runtime
        self.recorder = recorder
        self.results = results
        self.camera_snapshot = camera_snapshot
        self.max_running_s = max_running_s
        self.query_ids = query_ids
        self.capture_ids = capture_ids
        self.preparing_epoch = preparing_epoch
        self.run_id = None
        self.termination = None
        self.history = deque(maxlen=policy_info.n_obs_steps)
        self.hand_history = (
            deque(maxlen=policy_info.hand_history_steps)
            if policy_info.hand_history_steps is not None else None
        )
        self.plan = self.query = None
        self.preparation_issue = None
        self.observation_read_issues = []
        self.wait_started_ns = None
        self.last_slot = -1
        self.dt_ns = int(policy_info.control_dt_s * 1e9)
        self.max_decision_age_ns = int(self.execution.max_decision_age_s * 1e9)
        self.started_ns = self.next_step_ns = 0
        self.previous_arm = None
        # Cleared before enqueue: a failed/partial writer submission must not be retried.
        self.pending_frame = None

    def read_observation(self, *, wait_for_hand=False):
        row = read_observation(
            self.sensors.channels, self.robot,
            needs=self.needs, wait_for_hand=wait_for_hand,
            cancel_requested=lambda: not self.preparation_valid(),
        )
        self.observation_read_issues = observation_issues(row, self.runtime, self.control_needs)
        if not self.observation_read_issues:
            if self.recorder is not None:
                record_issues = observation_issues(row, self.runtime, self.record_needs)
                if record_issues:
                    raise RecordingError("recording sources: " + "; ".join(record_issues))
            self.robot.arm.update_health(
                health_until_ns=control_deadline_ns(row, self.runtime, self.control_needs),
            )
        return row

    def budget_limit(self):
        return run_budget(
            run_id=self.run_id, started_ns=self.started_ns, max_running_s=self.max_running_s,
            wait_started_ns=self.wait_started_ns, max_wait_s=self.execution.max_wait_s,
        )

    def expired_budget(self):
        limit = self.budget_limit()
        return limit if limit is not None and time.monotonic_ns() >= limit[0] else None

    def finish_budget(self, limit):
        self.finish(f"{limit[1]}_timeout", run_end_reason=RunEndReason.TIMEOUT)

    def has_motion_authority(self):
        return self.runtime_state.command_may_cross_sdk(run_id=self.run_id)

    def finish(self, detail, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, errors=None):
        if self.termination is not None:
            return
        epoch = self.run_id
        ended_ns = time.monotonic_ns()
        duration_s = (ended_ns - self.started_ns) / 1e9 if epoch is not None else 0.0
        self.runtime_state.revoke_motion(
            reason=run_end_reason,
            expected_run_id=epoch if epoch is not None else self.preparing_epoch,
        )
        with self.runtime_state.motion_lock:
            stopped_epoch = self.runtime_state.run_id
            reason = (
                self.runtime_state.run_ended_reason
                if epoch is not None and self.runtime_state.run_ended_id == epoch
                else run_end_reason
            )
        if epoch is None:
            reason_name = "preparation_failed" if self.preparation_issue or errors else "start_cancelled"
        else:
            reason_name = reason.name.lower()
        # The first end owns all business finalization, even if cleanup later raises.
        self.termination = EpisodeEnd(epoch, reason_name, detail, duration_s)
        self.run_id = None
        self.plan = None
        if self.model.future is None:
            self.query = self.preparing_epoch = None
        self.history.clear()
        if self.hand_history is not None:
            self.hand_history.clear()
        self.previous_arm = self.wait_started_ns = None
        errors, failure = list(errors or ()), None
        capture_failed = bool(errors) or not (
            reason_name in {"operator", "quit"}
            or (reason_name == "timeout" and detail == "duration_timeout")
        )
        try:
            if epoch is not None or self.robot.stop_required:
                self.robot.stop()
            self.runtime_state.acknowledge_stop(stopped_epoch)
        except BaseException as exc:
            failure = exc
            errors.append(error_detail("stop", exc))
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            if epoch is None:
                reason_name = "preparation_failed"
                self.termination = EpisodeEnd(None, reason_name, detail, duration_s)
        try:
            self.record_pending()
        except BaseException as exc:
            failure = failure or exc
            errors.append(error_detail("recording", exc))
            capture_failed = True
        try:
            self.finalize_outputs(
                reason_name, detail, errors, duration_s=duration_s, capture_failed=capture_failed,
            )
        except BaseException as exc:
            failure = failure or exc
        if failure is not None:
            raise failure

    def finalize_outputs(self, reason, detail, errors, *, duration_s, capture_failed):
        failure, raw = None, None
        if self.recorder is not None and self.recorder.start_attempted:
            saved = None
            try:
                saved = self.recorder.save_episode(
                    reason=reason,
                    details=[dict(stage="termination", reason=detail), *errors],
                    technical_status="failed" if capture_failed else "normal",
                )
                status = "published" if saved is not None else "not_published"
            except BaseException as exc:
                failure, status = exc, "failed"
                errors.append(error_detail("recorder_finalize", exc))
            raw = dict(
                status=status,
                path=str(saved) if saved is not None else None,
                staging_path=(
                    str(self.recorder.staging_path)
                    if self.recorder.staging_path is not None and saved is None
                    else None
                ),
                row_count=self.recorder.written_frames,
            )
        if self.results is not None:
            try:
                self.results.finish_episode(
                    reason, details=detail, errors=errors, raw=raw, duration_s=duration_s
                )
            except BaseException as exc:
                failure = failure or exc
                self.results.record_errors([error_detail("evaluation_save", exc)])
        if failure is not None:
            raise failure

    def observation_issue(self):
        return (
            "control_input_" + self.observation_read_issues[0]
            if self.observation_read_issues else None
        )

    def preparation_valid(self):
        return self.runtime_state.command_may_cross_sdk(
            run_id=self.preparing_epoch,
            required_safety_state=SafetyState.ARMED,
        )

    def start_observation(self, *, wait_for_hand=False):
        row = self.read_observation(wait_for_hand=wait_for_hand)
        self.preparation_issue = self.observation_issue()
        if self.preparation_issue is not None:
            logger.warning("本轮未开始：%s", self.preparation_issue)
            return None
        if np.max(np.abs(row.arm["qpos"][0] - self.runtime.arm.home_qpos)) > np.deg2rad(
            self.execution.start_arm_home_tolerance_deg
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
        if self.results is not None:
            self.results.begin_episode()
        try:
            self.robot.reset_hand_feedback()
            row = self.start_observation(wait_for_hand=True)
            if not self.preparation_valid():
                raise CancelledError("cancelled during initial feedback acquisition")
        except CancelledError as exc:
            self.finish(str(exc))
            return
        if row is None:
            self.finish(self.preparation_issue)
            return
        logger.info("正在准备本轮模型…")
        self.model.submit("reset_episode", query_context=dict(preparing_epoch=self.preparing_epoch))

    def admit(self):
        if not self.preparation_valid():
            self.finish("cancelled during model reset")
            return
        epoch = self.preparing_epoch
        if self.recorder is not None:
            metadata = snapshot_recording_metadata(
                self.camera_snapshot, hand=self.robot.hand,
            )
            capture_id = next(self.capture_ids)
            self.recorder.start_episode(
                task_label=self.policy_info.task_name,
                episode_name=f"episode_{capture_id:03d}", collection_source="policy_rollout", metadata=metadata,
            )
        # START and mode restoration can block; each boundary requires fresh feedback.
        if self.start_observation() is None:
            self.finish(self.preparation_issue)
            return
        self.robot.prepare_streaming(epoch, service=lambda: service_feedback(self.robot, self.runtime))
        if self.start_observation() is None:
            self.finish(self.preparation_issue)
            return
        with self.runtime_state.motion_lock:
            admitted = self.runtime_state.begin_motion() if epoch == self.runtime_state.run_id else None
        if admitted is None:
            self.finish("final admission declined")
            return
        self.run_id, self.started_ns = admitted
        self.preparing_epoch = None
        self.next_step_ns = self.wait_started_ns = self.started_ns
        if self.results is not None:
            self.results.entered(self.run_id)

    def poll_model(self):
        item = self.model.poll()
        if item is None:
            return
        operation, result = item
        if operation == "reset_episode":
            if result.query_context != dict(preparing_epoch=self.preparing_epoch):
                raise RuntimeError("model reset/preparation ownership mismatch")
            retired = self.termination is not None or not self.preparation_valid()
            failure = (
                self.retired_model_error(operation, result)
                if retired and result.error is not None else None
            )
            if self.termination is None and retired:
                self.finish("cancelled during model reset")
            if retired:
                self.preparing_epoch = None
                if failure is not None:
                    raise failure from result.error
                return
            if result.error is not None:
                raise result.error
            self.admit()
            return
        query = self.query
        if query is None:
            raise RuntimeError("Model result has no owned query")
        if result.query_context != dict(run_id=query.run_id, query_id=query.query_id):
            raise RuntimeError("model result/query ownership mismatch")
        if self.results is not None and query.run_id == self.run_id:
            self.results.record_inference(result)
        if query.run_id != self.run_id or not self.has_motion_authority():
            failure = self.retired_model_error(operation, result) if result.error is not None else None
            if self.termination is None:
                self.finish("owner_authority_revoked")
            self.query = None
            if failure is not None:
                raise failure from result.error
            return
        if result.error is not None:
            raise result.error
        query.result = result.value

    def retired_model_error(self, operation, result):
        # An ended episode remains immutable; an unknown model state fences the session.
        self.runtime_state.mark_error()
        detail = dict(
            error_detail("retired_model", result.error), operation=operation,
            query_context=dict(result.query_context),
            started_ns=result.started_ns, completed_ns=result.completed_ns,
        )
        logger.error("retired model task failed: %s", detail)
        if self.results is not None:
            self.results.record_errors([detail])
        return RuntimeError(
            f"retired model {operation} failed: context={result.query_context}; "
            f"{type(result.error).__name__}: {result.error}"
        )

    def submit(self, rows, slot, *, handoff_slot=None, prefix=None, commands=None):
        now = time.monotonic_ns()
        authorized, expired = self.has_motion_authority(), self.expired_budget()
        if expired is not None:
            return ControlRejection(f"{expired[1]}_timeout", RunEndReason.TIMEOUT)
        if not authorized:
            return ControlRejection("query_authority_revoked")
        sources = rows[-1].source_times
        expires_at_ns = decision_expires_at_ns(sources, now, self.max_decision_age_ns)
        if self.results is not None:
            self.results.record_duration("source_age", now - min(sources))
        if expires_at_ns is None:
            return ControlRejection("query_sources_invalid_or_expired")
        if prefix is not None and now >= self.plan.expires_at_ns:
            return ControlRejection("prefix_decision_expired")
        query = PolicyQuery(next(self.query_ids), self.run_id, slot, expires_at_ns, handoff_slot)
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=handoff_slot - slot,
                 rtc_guidance_cap=self.execution.rtc_guidance_cap) if prefix is not None else {}
        )
        initial_hand_history = None
        if self.hand_history is not None:
            if not self.hand_history:
                # Match training's episode-initial measured pose, before the first query action.
                hand = rows[-1].values["hand_qpos"].astype(np.float32).astype(np.float64)
                if hand.shape != (12,) or not np.isfinite(hand).all():
                    return ControlRejection("invalid_hand_history_bootstrap")
                initial_hand_history = tuple(
                    hand.copy() for _ in range(self.hand_history.maxlen)
                )
            # The model thread owns this snapshot; later appends/reset cannot change it.
            kwargs["action_context"] = {
                "hand_history": np.stack(
                    self.hand_history if initial_hand_history is None else initial_hand_history
                )
            }
        self.model.submit(
            "predict",
            tuple(rows),
            query_context=dict(run_id=self.run_id, query_id=query.query_id),
            **kwargs,
        )
        if initial_hand_history is not None:
            self.hand_history.extend(initial_hand_history)
        self.query = query
        if commands is not None:
            self.plan.frozen.update(commands)
        return None

    def prepare_prefix(self, row, slot):
        """Prepare the whole guidance tail and d physical commands without committing either."""
        prefix, slots = rtc_prefix_request(self.plan, slot, self.execution.prefetch_steps)
        commands, previous = {}, self.previous_arm
        for index, command_slot in enumerate(slots):
            self.robot.check()
            if not self.has_motion_authority():
                return None, None, ControlRejection("prefix_authority_revoked")
            expired = self.expired_budget()
            if expired is not None:
                return None, None, ControlRejection(f"{expired[1]}_timeout", RunEndReason.TIMEOUT)
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

    def record_pending(self):
        frame, self.pending_frame = self.pending_frame, None
        if self.recorder is not None and frame is not None:
            self.recorder.add_frame(*frame)

    def map_command(self, action, row, previous, *, context="action"):
        """Map one policy action to a checked joint command."""
        action = np.asarray(action, dtype=np.float64)
        mode = self.policy_info.action_mode
        intent = (
            ActionIntent(mode, action[:7], action[7:])
            if mode == "joint"
            else ActionIntent(mode, Pose(action[:3], rot6d_to_quat_wxyz(action[3:9])), action[9:])
        )
        decoded = self.action_mapper.map(intent, row.arm["qpos"][0], previous)
        if decoded.arm_qpos is None:
            if (
                decoded.ik_result is not None
                and decoded.ik_result.failure_kind == IKFailureKind.INVALID_OUTPUT
            ):
                raise RuntimeError(f"online IK technical failure: {decoded.ik_result.reason}")
            return None, ControlRejection(f"{context}_rejected: {decoded.rejection_reason}")
        return RobotCommand(self.run_id, decoded.arm_qpos, decoded.hand_qpos), None

    def command_for_slot(self, row, slot):
        plan = self.plan
        command = plan.frozen.get(slot)
        if command is not None:
            rejection = self.action_mapper.frozen_rejection(
                command, row.arm["qpos"][0], self.previous_arm, self.policy_info.action_mode
            )
            if rejection is not None:
                return None, ControlRejection(f"frozen_target_rejected: {rejection}")
        else:
            return self.map_command(
                plan.actions[slot - plan.anchor_slot],
                row,
                self.previous_arm,
            )
        return command, None

    def execute_slot(self, row, slot):
        plan = self.plan
        mapping_started = time.monotonic_ns()
        command, issue = self.command_for_slot(row, slot)
        if self.results is not None:
            self.results.record_duration("mapping", time.monotonic_ns() - mapping_started)
        if issue is not None:
            self.finish(issue.detail, run_end_reason=issue.run_end_reason)
            return
        if not self.has_motion_authority():
            self.finish("before_submission_revoked")
            return
        if (expired := self.expired_budget()) is not None:
            self.finish_budget(expired)
            return
        budget_limit = self.budget_limit()
        decision_deadline_ns = plan.expires_at_ns
        action_deadline_ns = min(
            self.next_step_ns,
            feedback_deadline_ns(row, self.runtime, needs=self.control_needs),
            decision_deadline_ns,
        )
        valid_until_ns = (
            min(action_deadline_ns, budget_limit[0]) if budget_limit else action_deadline_ns
        )
        if time.monotonic_ns() >= valid_until_ns:
            self.finish("submission_deadline_expired")
            return
        execute_until_ns = decision_deadline_ns
        if self.max_running_s is not None:
            execute_until_ns = min(execute_until_ns, self.started_ns + int(self.max_running_s * 1e9))
        started_ns = time.monotonic_ns()
        self.pending_frame = (row, command)
        try:
            try:
                self.robot.send_action(
                    command, valid_until_ns=valid_until_ns, execute_until_ns=execute_until_ns,
                )
            finally:
                if self.results is not None:
                    self.results.record_duration("submission", time.monotonic_ns() - started_ns)
        except (OperationCancelled, TimeoutError) as exc:
            self.finish(
                str(exc), run_end_reason=RunEndReason.EXECUTOR_BOUNDARY,
                errors=[error_detail("submission", exc)],
            )
            return
        now = time.monotonic_ns()
        if budget_limit is not None and now >= budget_limit[0]:
            self.finish_budget(budget_limit)
            return
        if now >= action_deadline_ns or not self.has_motion_authority():
            self.finish("submission_return_expired_or_cancelled")
            return
        self.previous_arm = command.arm_qpos
        if self.hand_history is not None:
            # Normal submission updates action context; it does not prove delivery.
            self.hand_history.append(command.hand_qpos.copy())
        self.wait_started_ns = None
        self.record_pending()
        if slot + 1 == plan.segment_end and self.execution.execution_mode == "sync":
            self.plan = None

    def submit_prefetch(self, row, rows, slot):
        handoff_slot, issue = prefetch_handoff(
            self.plan, slot=slot, mode=self.execution.execution_mode,
            prefetch_steps=self.execution.prefetch_steps,
            query_pending=self.query is not None, model_pending=self.model.future is not None,
            history_ready=bool(rows),
        )
        if issue is not None or handoff_slot is None:
            return issue
        prefix = commands = None
        if self.execution.execution_mode == "rtc":
            prefix, commands, issue = self.prepare_prefix(row, slot)
            if issue is not None:
                return issue
        return self.submit(
            rows, slot, handoff_slot=handoff_slot, prefix=prefix, commands=commands
        )

    def step(self):
        try:
            self.advance()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            if self.termination is not None:
                raise
            preparing = self.run_id is None
            cancelled = preparing and isinstance(exc, (OperationCancelled, CancelledError))
            stage = "preparation" if preparing else "control"
            detail = f"{stage}_failed: {type(exc).__name__}: {exc}"
            try:
                self.finish(
                    detail,
                    run_end_reason=failure_reason(exc),
                    errors=[] if cancelled or (preparing and isinstance(exc, KeyboardInterrupt))
                    else [error_detail(stage, exc)],
                )
            except BaseException:
                if cancelled:
                    raise
                logger.exception("episode failure cleanup also failed")
            if not cancelled:
                raise

    def advance(self):
        if self.termination is not None:
            self.poll_model()
            return
        if self.run_id is None and self.model.future is None:
            self.prepare()
            return
        if self.run_id is None and not self.preparation_valid():
            self.finish("cancelled during model reset")
            return
        if self.run_id is not None and not self.has_motion_authority():
            self.finish("owner_authority_revoked")
            return
        self.poll_model()
        if self.termination is not None:
            return
        if self.recorder is not None:
            self.recorder.check_error()
        if self.run_id is None:
            return
        if (expired := self.expired_budget()) is not None:
            self.finish_budget(expired)
            return
        now = time.monotonic_ns()
        slot, self.next_step_ns, skipped = control_slot(
            now_ns=now, next_step_ns=self.next_step_ns, last_slot=self.last_slot,
            dt_ns=self.dt_ns, synchronous=self.execution.execution_mode == "sync",
            resume_ready=(
                self.plan is None and self.query is not None and self.query.result is not None
            ),
        )
        if slot is None:
            return
        if skipped:
            if self.results is not None:
                self.results.record_skipped_slots(skipped)
            self.finish(f"missed_slot: expected={slot}, actual={slot + skipped}")
            return
        self.last_slot = slot
        row = self.read_observation()
        self.pending_frame = (row, None)
        self.poll_model()
        if not self.has_motion_authority():
            self.finish("observation_authority_revoked")
            return
        issue = self.observation_issue()
        if issue is not None:
            self.finish(issue)
            return
        self.history.append(project_policy_row(row, self.policy_info.observation_fields))
        rows = tuple(self.history) if len(self.history) == self.history.maxlen else ()
        new_plan, issue = handoff_plan(
            self.query, run_id=self.run_id, slot=slot,
            mode=self.execution.execution_mode, action_steps=self.execution.action_steps,
            now_ns=time.monotonic_ns(), authorized=self.has_motion_authority(),
        )
        if issue is not None:
            self.finish(issue.detail, run_end_reason=issue.run_end_reason)
            return
        if new_plan is not None:
            self.plan, self.query = new_plan, None
        if self.plan is None:
            self.robot.end_hand_send_interval()
            if self.wait_started_ns is None:
                self.wait_started_ns = now
                self.robot.arm.limit_execution_until(now + int(self.execution.max_wait_s * 1e9))
            if rows and self.query is None and self.model.future is None:
                issue = self.submit(rows, slot)
                if issue is not None:
                    self.finish(
                        issue.detail, run_end_reason=issue.run_end_reason
                    )
                    return
            self.record_pending()
        else:
            # Exhausted sync/RTC segments cannot prefetch; async also checks its live tail/age.
            issue = plan_rejection(
                self.plan, slot, self.execution.execution_mode, time.monotonic_ns()
            )
            if issue is not None:
                self.finish(issue.detail)
                return
            issue = self.submit_prefetch(row, rows, slot)
            if issue is not None:
                self.finish(issue.detail, run_end_reason=issue.run_end_reason)
                return
            # Pending prefetch never suspends a still-valid current plan.
            self.execute_slot(row, slot)
        if (expired := self.expired_budget()) is not None:
            self.finish_budget(expired)
