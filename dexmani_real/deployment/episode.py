"""One Policy attempt owns preparation, control, capture and termination evidence."""

import logging
import time
from collections import deque
from dataclasses import dataclass
from concurrent.futures import CancelledError

import numpy as np

from dexmani_real.data.raw.schema import observation_timing
from dexmani_real.data.recorder import (
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.data.results import error_detail
from dexmani_real.deployment.observation import (
    build_policy_observation,
    policy_observation_issue,
    policy_sources,
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
from dexmani_real.robot.commands import (
    DispatchError,
    DispatchInterrupted,
    DispatchStatus,
    RobotCommand,
)
from dexmani_real.runtime.observation import (
    feedback_deadline_ns,
    read_observation,
    upstream_deadline_ns,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState

logger = logging.getLogger(__name__)


def failure_reason(error):
    if isinstance(error, KeyboardInterrupt):
        return RunEndReason.ESTOP
    if isinstance(error, RecordingError):
        return RunEndReason.RECORDING_FAILURE
    if isinstance(error, DispatchError):
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
        self, runtime_state, runtime, policy_info, *, preparing_epoch,
        robot, needs, sensors, action_mapper, model_runtime, kinematics,
        max_running_s, query_ids, capture_ids, recorder=None,
        camera_calibration=None, camera_parameters=None, results=None,
    ):
        self.runtime_state, self.runtime = runtime_state, runtime
        self.policy_info, self.execution = policy_info, runtime.execution
        self.robot, self.sensors, self.needs = robot, sensors, needs
        self.action_mapper, self.kinematics = action_mapper, kinematics
        self.model, self.recorder, self.results = model_runtime, recorder, results
        self.camera_calibration, self.camera_parameters = camera_calibration, camera_parameters
        self.max_running_s = max_running_s
        self.query_ids, self.capture_ids = query_ids, capture_ids
        self.preparing_epoch = preparing_epoch
        self.run_id = None
        self.termination = None
        self.history = deque(maxlen=policy_info.n_obs_steps)
        self.hand_history = (
            deque(maxlen=policy_info.hand_history_steps)
            if policy_info.hand_history_steps is not None else None
        )
        # A START attempt needs finalization even when the writer never becomes ready.
        self.capture_attempted = False
        self.plan = self.query = None
        self.slot_timing = {}
        self.termination_timing = None
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
        self.observation_read_issues.clear()
        row = read_observation(
            self.sensors.channels,
            self.runtime,
            self.robot,
            issues=self.observation_read_issues,
            require_hand=True,
            needs=self.needs,
            wait_for_hand=wait_for_hand,
            cancel_requested=lambda: not self.preparation_valid(),
        )
        if row is not None:
            self.robot.arm.update_health(upstream_until_ns=upstream_deadline_ns(row, self.runtime))
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
        if epoch is not None and self.recorder is not None:
            self.termination_timing = dict(
                run_id=epoch, started_ns=self.started_ns, slot=self.last_slot,
                ended_ns=ended_ns, reason=detail,
                query=dict(self.query.evidence) if self.query is not None else None,
                action_query_id=self.slot_timing.get("query_id"),
                prediction_index=self.slot_timing.get("prediction_index"),
            )
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
        try:
            self.finalize_outputs(reason_name, detail, errors, duration_s=duration_s)
        except BaseException as exc:
            failure = failure or exc
        if failure is not None:
            raise failure

    def finalize_outputs(self, reason, detail, errors=None, *, duration_s=0.0):
        errors = list(errors or ())
        failure, raw = None, None
        try:
            if self.recorder is not None and self.capture_attempted:
                saved = None
                try:
                    saved = self.recorder.save_episode(
                        reason=reason,
                        details=[dict(stage="termination", reason=detail), *errors],
                        timing=self.termination_timing,
                        arm_execution=dict(
                            self.robot.arm.execution_snapshot(),
                            robot_stopped=not self.robot.stop_required,
                        ),
                    )
                    status = "published" if saved is not None else "empty"
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
        finally:
            self.capture_attempted = False

    def observation_issue(self, row):
        if row is None and self.observation_read_issues:
            issue = self.observation_read_issues[0]
            if (
                self.recorder is not None
                and issue.startswith("camera_")
                and not (
                    set(self.policy_info.observation_fields)
                    & {"rgb", "camera_intrinsic", "camera_extrinsic"}
                )
            ):
                return f"recording_source_{issue}"
            return f"policy_input_{issue}"
        return policy_observation_issue(row, self.policy_info.observation_fields)

    def preparation_valid(self):
        return self.runtime_state.command_may_cross_sdk(
            run_id=self.preparing_epoch,
            required_safety_state=SafetyState.ARMED,
        )

    def start_observation(self, *, wait_for_hand=False):
        row = self.read_observation(wait_for_hand=wait_for_hand)
        self.preparation_issue = self.observation_issue(row)
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
                self.sensors.channels, camera_calibration=self.camera_calibration, hand=self.robot.hand,
            )
            self.capture_attempted = True
            capture_id = next(self.capture_ids)
            self.recorder.start_episode(
                task_label=self.policy_info.task_name,
                arm_execution=self.robot.arm.execution_snapshot(),
                arm_control_metadata=self.robot.arm.execution_metadata(1 / self.recorder.control_hz),
                episode_name=f"episode_{capture_id:03d}", collection_source="policy_rollout", metadata=metadata,
            )
        # START and mode restoration can block; each boundary requires fresh feedback.
        if self.start_observation() is None:
            self.finish(self.preparation_issue)
            return
        self.robot.prepare_streaming(epoch)
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
            if result.error is not None:
                raise result.error
            if self.termination is None:
                self.admit()
            else:
                self.preparing_epoch = None
            return
        query = self.query
        if query is None:
            raise RuntimeError("Model result has no owned query")
        if result.query_context != dict(run_id=query.run_id, query_id=query.query_id):
            raise RuntimeError("model result/query ownership mismatch")
        if self.recorder is not None and query.run_id == self.run_id:
            query.evidence.update(
                started_ns=result.started_ns,
                completed_ns=result.completed_ns,
                status="failed" if result.error is not None else "completed",
                error=str(result.error) if result.error is not None else None,
            )
            self.record_query(query)
        if self.results is not None and query.run_id == self.run_id:
            self.results.record_inference(result.completed_ns - result.started_ns)
        if query.run_id != self.run_id or not self.has_motion_authority():
            logger.debug(
                "retired query %s from run %s reclaimed; error=%s",
                query.query_id,
                query.run_id,
                result.error,
            )
            self.query = None
            return
        if result.error is not None:
            raise result.error
        query.result = result.value

    def submit(self, rows, slot, *, handoff_slot=None, prefix=None, commands=None):
        observation = build_policy_observation(
            rows,
            self.policy_info,
            kinematics=self.kinematics,
            camera_parameters=self.camera_parameters,
        )
        now = time.monotonic_ns()
        authorized, expired = self.has_motion_authority(), self.expired_budget()
        if expired is not None:
            return ControlRejection(f"{expired[1]}_timeout", RunEndReason.TIMEOUT)
        if not authorized:
            return ControlRejection("query_authority_revoked")
        sources = policy_sources(rows[-1], self.policy_info.observation_fields)
        expires_at_ns = decision_expires_at_ns(sources, now, self.max_decision_age_ns)
        if expires_at_ns is None:
            return ControlRejection("query_sources_invalid_or_expired")
        if prefix is not None and now >= self.plan.expires_at_ns:
            return ControlRejection("prefix_decision_expired")
        query = PolicyQuery(next(self.query_ids), self.run_id, slot, expires_at_ns, handoff_slot)
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=handoff_slot - slot) if prefix is not None else {}
        )
        initial_hand_history = None
        if self.hand_history is not None:
            if not self.hand_history:
                # Match training's episode-initial measured pose, before the first query action.
                hand = np.asarray(observation["joint_state"][-1, -12:], dtype=np.float64)
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
        if self.recorder is not None:
            query.evidence = dict(
                query_id=query.query_id,
                run_id=self.run_id,
                slot=int(slot),
                handoff_slot=handoff_slot,
                planned_handoff_ns=(
                    self.started_ns + handoff_slot * self.dt_ns
                    if handoff_slot is not None else None
                ),
                decision_expires_at_ns=expires_at_ns,
                actual_handoff_slot=None,
                actual_handoff_ns=None,
                deadline_ns=(
                    (self.started_ns + (handoff_slot + 1) * self.dt_ns)
                    if handoff_slot is not None and self.execution.execution_mode == "rtc"
                    else None
                ),
                submitted_ns=time.monotonic_ns(),
                started_ns=None,
                completed_ns=None,
                status="submitted",
                bootstrap=handoff_slot is None,
                observations=[
                    dict(
                        observation_timing(row),
                        policy_sources=policy_sources(row, self.policy_info.observation_fields),
                    )
                    for row in rows
                ],
            )
        self.model.submit(
            "predict",
            observation,
            query_context=dict(run_id=self.run_id, query_id=query.query_id),
            **kwargs,
        )
        if initial_hand_history is not None:
            self.hand_history.extend(initial_hand_history)
        self.query = query
        if commands is not None:
            self.plan.frozen.update(commands)
        self.record_query(query)
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

    def record_query(self, query):
        if self.recorder is not None and self.capture_attempted:
            self.recorder.record_query(query.evidence)

    def record_pending(self):
        frame, self.pending_frame = self.pending_frame, None
        if self.recorder is not None and frame is not None and frame[0] is not None:
            self.recorder.add_frame(*frame, timing=self.slot_timing)

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
        self.slot_timing.update(query_id=plan.query_id, prediction_index=slot - plan.anchor_slot)
        command, issue = self.command_for_slot(row, slot)
        if issue is not None:
            self.finish(issue.detail, run_end_reason=issue.run_end_reason)
            return
        if not self.has_motion_authority():
            self.finish("before_dispatch_revoked")
            return
        if (expired := self.expired_budget()) is not None:
            self.finish_budget(expired)
            return
        budget_limit = self.budget_limit()
        decision_deadline_ns = plan.expires_at_ns
        action_deadline_ns = min(
            self.next_step_ns,
            feedback_deadline_ns(row, self.runtime),
            decision_deadline_ns,
        )
        valid_until_ns = (
            min(action_deadline_ns, budget_limit[0]) if budget_limit else action_deadline_ns
        )
        if time.monotonic_ns() >= valid_until_ns:
            self.finish("dispatch_deadline_expired")
            return
        failure = None
        try:
            self.slot_timing["submit_started_ns"] = time.monotonic_ns()
            execute_until_ns = decision_deadline_ns
            if self.max_running_s is not None:
                execute_until_ns = min(
                    execute_until_ns, self.started_ns + int(self.max_running_s * 1e9)
                )
            result = self.robot.send_action(
                command, valid_until_ns=valid_until_ns, execute_until_ns=execute_until_ns
            )
            if result.arm != DispatchStatus.SUBMITTED or result.hand not in (
                DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED
            ):
                raise DispatchError("incomplete action dispatch", result, cause="dispatch_status")
        except (DispatchError, DispatchInterrupted) as exc:
            result, failure = exc.result, exc
        finally:
            self.slot_timing["submit_completed_ns"] = time.monotonic_ns()
        self.pending_frame = (row, command, result)
        if self.results is not None and (
            result.arm != DispatchStatus.NOT_CALLED or result.hand != DispatchStatus.NOT_CALLED
        ):
            self.results.record_dispatch()
        now = time.monotonic_ns()
        expired = budget_limit if budget_limit is not None and now >= budget_limit[0] else None
        late = now >= action_deadline_ns
        cancelled = isinstance(failure, DispatchInterrupted)
        cause = failure.cause if isinstance(failure, DispatchError) else None
        if failure is not None or expired is not None or late or not self.has_motion_authority():
            if cancelled:
                self.runtime_state.request_emergency_stop()
            hardware_failure = (
                not cancelled
                and failure is not None
                and (not failure.revoked or cause not in {"authority_revoked", "deadline_expired"})
            )
            budget_timeout = expired is not None and budget_limit[0] < action_deadline_ns
            if cancelled:
                reason, detail = RunEndReason.ESTOP, "dispatch_cancelled"
            elif hardware_failure:
                reason, detail = (
                    RunEndReason.HARDWARE_FAULT,
                    f"dispatch_failed: {cause or str(failure)}",
                )
            elif budget_timeout:
                reason = RunEndReason.TIMEOUT
                detail = (
                    f"dispatch_failed: {cause or str(failure)}"
                    if failure is not None
                    else f"dispatch_return_{expired[1]}_timeout"
                )
            else:
                reason = RunEndReason.EXECUTOR_BOUNDARY
                if failure is not None:
                    detail = f"dispatch_failed: {cause or str(failure)}"
                elif late:
                    detail = "dispatch_return_deadline_expired"
                else:
                    detail = "dispatch_authority_revoked"
            errors = [error_detail("dispatch", failure)] if failure is not None else []
            try:
                self.finish(
                    detail,
                    run_end_reason=reason,
                    errors=errors,
                )
            except Exception as exc:
                if hardware_failure or cancelled:
                    raise failure from exc
                raise
            if hardware_failure or cancelled:
                raise failure
            return
        self.previous_arm = command.arm_qpos
        if self.hand_history is not None:
            # Ordinary-send CRC is admitted but unconfirmed, just as in canonical actions.
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
            preparing = self.run_id is None and self.termination is None
            cancelled = (
                preparing and isinstance(exc, DispatchError)
                and exc.revoked and exc.cause == "authority_revoked"
            )
            try:
                self.finish(
                    str(exc) if preparing else f"control_failed: {type(exc).__name__}: {exc}",
                    run_end_reason=failure_reason(exc),
                    errors=[] if cancelled or isinstance(exc, KeyboardInterrupt)
                    else [error_detail("preparation" if preparing else "control", exc)],
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
        self.slot_timing = dict(run_id=self.run_id, slot=int(slot), control_ns=now)
        self.last_slot = slot
        row = self.read_observation()
        self.pending_frame = (row, None, None)
        self.poll_model()
        if not self.has_motion_authority():
            self.finish("observation_authority_revoked")
            return
        issue = self.observation_issue(row)
        if issue is not None:
            self.finish(issue)
            return
        self.history.append(row)
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
            if self.recorder is not None:
                if self.query.handoff_slot is None:
                    self.query.evidence.update(handoff_slot=int(slot), bootstrap=True)
                self.query.evidence.update(
                    actual_handoff_slot=int(slot), actual_handoff_ns=time.monotonic_ns()
                )
                self.record_query(self.query)
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
            if self.execution.execution_mode == "async":
                issue = plan_rejection(self.plan, slot, "async", time.monotonic_ns())
                if issue is not None:
                    self.finish(issue.detail)
                    return
            issue = self.submit_prefetch(row, rows, slot)
            if issue is not None:
                self.finish(issue.detail, run_end_reason=issue.run_end_reason)
                return
            if self.execution.execution_mode != "async":
                issue = plan_rejection(
                    self.plan, slot, self.execution.execution_mode, time.monotonic_ns()
                )
                if issue is not None:
                    self.finish(issue.detail)
                    return
            # Pending prefetch never suspends a still-valid current plan.
            self.execute_slot(row, slot)
        if (expired := self.expired_budget()) is not None:
            self.finish_budget(expired)
