"""Device owner on one control grid, with one serial model worker in all modes."""

import logging
import time
from dataclasses import dataclass
from collections import deque

import numpy as np

from dexmani_real.deployment.action import (
    physical_action_dim,
    policy_action_intent,
)
from dexmani_real.deployment.observation import (
    build_policy_observation,
    decision_is_fresh,
    policy_sources,
)
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.recording.recorder import (
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.utils.episode_results import error_detail
from dexmani_real.robot.commands import DispatchStatus, RobotCommand
from dexmani_real.robot.robot import DispatchError, DispatchInterrupted
from dexmani_real.runtime.observation import (
    feedback_deadline_ns,
    read_observation,
)
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    StopRequest,
    _begin_motion_locked,
    revoke_motion_if_run_id,
)


logger = logging.getLogger(__name__)


@dataclass
class Query:
    """slot anchors the observation; handoff_slot is the first consumable slot."""

    query_id: int
    run_id: int
    slot: int
    sources: dict[str, tuple[int, ...]]
    handoff_slot: int | None
    valid: bool = True
    result: np.ndarray | None = None
    bootstrap: bool = False


@dataclass
class Plan:
    """actions index from anchor_slot; execution ends at exclusive segment_end."""

    actions: np.ndarray
    anchor_slot: int
    segment_end: int
    sources: dict[str, tuple[int, ...]]
    frozen: dict[int, RobotCommand]


class PolicyRunner:
    def __init__(
        self,
        shared,
        runtime,
        policy_info,
        *,
        robot,
        realizer,
        recorder=None,
        poll_operator=None,
        model_runtime,
        kinematics,
        execute,
        max_running_s,
        num_episodes=1,
        task_label=None,
        camera_calibration=None,
        results=None,
    ):
        self.results = results
        self.shared = shared
        self.robot = robot
        self.poll_operator = poll_operator
        self.runtime = runtime
        self.policy_info = policy_info
        self.model = model_runtime
        self.execution = runtime.execution
        self.kinematics = kinematics
        self.execute = execute
        self.max_running_s = max_running_s
        self.num_episodes = num_episodes
        self.task_label = task_label
        self.camera_calibration = camera_calibration
        self.recorder = recorder
        self.history = deque(maxlen=policy_info.n_obs_steps)
        self.recording_started = False
        self.plan = None
        self.query = None
        self.query_id = 0
        self.wait_started_ns = None
        self.preparing_epoch = None
        self.reset_ready = False
        self.last_slot = -1
        self.dt_ns = int(policy_info.control_dt_s * 1e9)
        self.run_id = None
        self.started_ns = 0
        self.next_step_ns = 0
        self.previous_arm = None
        self.completed = 0
        self.realizer = realizer
        fields = set(self.policy_info.observation_fields)
        requires_rgb = "rgb" in fields
        self.requires_cloud = "point_cloud" in fields
        self.requires_camera_payload = requires_rgb or self.recorder is not None
        self.requires_rgb_cloud_identity = self.requires_camera_payload and self.requires_cloud

    def _read_observation(self):
        return read_observation(
            self.shared,
            self.runtime,
            self.robot,
            require_hand=True,
            require_camera=self.requires_camera_payload,
            require_pointcloud=self.requires_cloud,
            require_rgb_cloud_identity=self.requires_rgb_cloud_identity,
        )

    def _budget_limit(self):
        deadlines = []
        if self.run_id is not None:
            if self.max_running_s is not None:
                deadlines.append((self.started_ns + int(self.max_running_s * 1e9), "duration"))
            if self.wait_started_ns is not None:
                deadlines.append(
                    (self.wait_started_ns + int(self.execution.max_wait_s * 1e9), "wait")
                )
        # WAIT wins a tie: no action was supplied before its strict deadline.
        return min(deadlines, key=lambda item: (item[0], item[1] != "wait")) if deadlines else None

    def _expired_budget(self):
        limit = self._budget_limit()
        return limit if limit is not None and time.monotonic_ns() >= limit[0] else None

    def _within_budget(self):
        return self._expired_budget() is None

    def _finish_budget(self, limit):
        self._finish_episode(f"{limit[1]}_timeout", run_end_reason=RunEndReason.TIMEOUT)
        if limit[1] == "wait":
            raise TimeoutError("Policy WAIT budget exhausted without successful consumption")

    def _dispatch_deadline_ns(self, row, slot):
        deadlines = [
            feedback_deadline_ns(row, self.runtime),
            self.started_ns + slot * self.dt_ns + int(self.execution.max_tick_lateness_s * 1e9) + 1,
        ]
        deadlines.extend(
            stamp + int(self.execution.max_decision_age_s * 1e9) + 1
            for times in self.plan.sources.values()
            for stamp in times
        )
        return min(deadlines)

    def _has_motion_authority(self):
        return (
            self.run_id is not None
            and self.shared.sensors.is_running.value
            and not self.shared.error_state
            and not self.shared.estop_request
            and int(self.shared.run_id) == self.run_id
            and int(self.shared.safety_state) == int(SafetyState.RUNNING)
        )

    def _invalidate(self):
        self.plan = None
        if self.query is not None:
            self.query.valid = False
            if self.model.future is None:
                self.query = None
        if self.run_id is not None and self.wait_started_ns is None:
            self.wait_started_ns = time.monotonic_ns()

    def _finish_episode(
        self,
        detail,
        *,
        run_end_reason=RunEndReason.EXECUTOR_BOUNDARY,
        stop_motion=True,
        errors=None,
    ):
        if self.run_id is None:
            return
        epoch = self.run_id
        duration_s = (time.monotonic_ns() - self.started_ns) / 1e9
        revoke_motion_if_run_id(self.shared, epoch, reason=run_end_reason)
        with self.shared.motion_lock:
            reason = (
                RunEndReason(int(self.shared.run_ended_reason))
                if int(self.shared.run_ended_id) == epoch
                else run_end_reason
            )
        self._invalidate()
        self.history.clear()
        self.previous_arm = None
        self.run_id = None
        self.completed += 1
        self.shared.stop_request = int(StopRequest.NONE)
        errors, failure = list(errors or ()), None
        try:
            if stop_motion:
                self.robot.stop()
        except Exception as exc:
            failure = exc
            errors.append(error_detail("stop", exc))
        try:
            self._finalize_outputs(reason.name.lower(), detail, errors, duration_s=duration_s)
        except Exception as exc:
            failure = failure or exc
        finally:
            logger.info("policy episode %d ended: %s (%s)", self.completed, reason.name, detail)
            if self.completed >= self.num_episodes:
                self.shared.quit_requested = True
        if failure is not None:
            raise failure

    def _finalize_outputs(self, reason, detail, errors=None, *, duration_s=0.0):
        errors = list(errors or ())
        failure, raw = None, None
        try:
            if self.recorder is not None and self.recording_started:
                saved, status = None, "empty"
                try:
                    saved = self.recorder.save_episode(reason=reason, details=errors)
                    status = "published" if saved is not None else "empty"
                except BaseException as exc:
                    failure, status = exc, "failed"
                    errors.append(error_detail("recorder_finalize", exc))
                raw = dict(status=status, path=str(saved) if saved is not None else None,
                           staging_path=str(self.recorder.staging_path)
                           if self.recorder.staging_path is not None and saved is None else None,
                           row_count=self.recorder.written_frames)
            if self.results is not None:
                try:
                    self.results.finish_episode(reason, details=detail, errors=errors,
                                                raw=raw, duration_s=duration_s)
                except BaseException as exc:
                    failure = failure or exc
                    self.results.session["errors"].append(error_detail("evaluation_save", exc))
            if failure is not None:
                raise failure
        finally:
            self.recording_started = False

    def _start_observation(self):
        row = self._read_observation()
        if row is None:
            logger.warning("policy B rejected: required observation unavailable")
            return None
        if self.execute and (
            np.max(np.abs(row.arm["qpos"][0] - self.runtime.arm.home_qpos))
            > self.runtime.arm.homing.convergence_rad
            or np.max(np.abs(row.hand["qpos"][0] - np.deg2rad(self.runtime.hand.home_qpos_deg)))
            > np.deg2rad(self.runtime.hand.home_tolerance_deg)
        ):
            logger.warning("policy B requires current arm + hand home pose")
            return None
        return row

    def _begin_episode(self):
        if self.preparing_epoch is None:
            with self.shared.motion_lock:
                if not self.shared.start_request or self.model.future is not None:
                    return
                self.shared.start_request = False
                if (
                    self.shared.stop_request
                    or self.shared.quit_requested
                    or not self.shared.sensors.is_running.value
                    or self.shared.error_state
                    or self.shared.estop_request
                    or int(self.shared.safety_state) != int(SafetyState.ARMED)
                ):
                    return
                epoch = int(self.shared.run_id)
            if self._start_observation() is None:
                return
            self.recording_started = False
            if self.results is not None:
                self.results.begin_episode()
            self.preparing_epoch = epoch
            self.reset_ready = False
            self.model.submit("reset_episode")
            return
        if not self.reset_ready:
            return
        epoch, self.preparing_epoch = self.preparing_epoch, None
        if (
            epoch != int(self.shared.run_id)
            or self.shared.quit_requested
            or self.shared.stop_request
        ):
            self._finalize_outputs("start_cancelled", "cancelled during model reset")
            return
        committed = False
        start_failure = None
        try:
            self.realizer.reset_episode()
            if self.recorder is not None:
                metadata = snapshot_recording_metadata(
                    self.shared,
                    collection_source="policy_rollout",
                    camera_calibration=self.camera_calibration,
                )
                self.recording_started = True
                self.recorder.start_episode(
                    task_label=self.task_label,
                    episode_name=f"episode_{self.completed + 1:03d}",
                    **metadata,
                )
            # Recorder startup can block; use a new row before granting motion.
            if self._start_observation() is None:
                return
            with self.shared.motion_lock:
                admitted = (
                    _begin_motion_locked(self.shared)
                    if epoch == int(self.shared.run_id)
                    and not self.shared.quit_requested
                    and not self.shared.stop_request
                    else None
                )
            if admitted is None:
                return
            self.run_id, self.started_ns = admitted
            if self.results is not None:
                self.results.entered(self.run_id)
            committed = True
            self.history.clear()
            self.plan = self.query = None
            self.previous_arm = None
            self.last_slot = -1
            self.next_step_ns = self.started_ns
            self.wait_started_ns = self.started_ns
        except BaseException as exc:
            start_failure = exc
            raise
        finally:
            if not committed:
                try:
                    self._finalize_outputs(
                        "start_cancelled",
                        str(start_failure) if start_failure else "final admission declined",
                    )
                except Exception:
                    if start_failure is None:
                        raise
                    logger.exception("start finalization also failed")

    def _poll_model(self):
        item = self.model.poll()
        if item is None:
            return
        operation, result = item
        if operation == "reset_episode":
            if result.error is not None:
                raise result.error
            self.reset_ready = True
            return
        if operation != "predict":
            if result.error is not None:
                raise result.error
            return
        query = self.query
        if query is None:
            raise RuntimeError("Model result has no owned query")
        if self.results is not None and query.run_id == self.run_id:
            self.results.record_inference(result.completed_ns - result.started_ns)
        if result.error is None:
            future = np.asarray(result.value)
            finite = bool(np.isfinite(future).all()) if future.dtype.kind in "fiubc" else None
            expected = (
                self.policy_info.horizon - self.policy_info.n_obs_steps + 1,
                physical_action_dim(self.policy_info.action_mode),
            )
            shaped_future = future.shape == expected and future.dtype.kind == "f"
            valid_future = shaped_future and finite
        if not query.valid or query.run_id != self.run_id or not self._has_motion_authority():
            logger.info(
                "retired query %s from run %s reclaimed; error=%s",
                query.query_id,
                query.run_id,
                result.error,
            )
            self.query = None
            return
        if result.error is not None:
            raise result.error
        now = time.monotonic_ns()
        if not decision_is_fresh(query.sources, now, self.execution.max_decision_age_s):
            self._invalidate()
            return
        if not valid_future:
            raise ValueError(f"Policy future requires finite floating point {expected}")
        query.result = future
        if query.handoff_slot is None:
            # Bootstrap was not executed during inference; preserve its entire head.
            query.handoff_slot = max(self.last_slot + 1, (now - self.started_ns) // self.dt_ns + 1)
            query.bootstrap = True

    def _submit(self, rows, slot, *, prefix=None, commands=None):
        observation = build_policy_observation(rows, self.policy_info, kinematics=self.kinematics)
        if observation is None:
            return False
        now = time.monotonic_ns()
        if not self._has_motion_authority() or not self._within_budget():
            return False
        if now > self.started_ns + slot * self.dt_ns + int(
            self.execution.max_tick_lateness_s * 1e9
        ):
            self._invalidate()
            return False
        sources = policy_sources(rows[-1], self.policy_info.observation_fields)
        if not decision_is_fresh(sources, now, self.execution.max_decision_age_s):
            return False
        self.query_id += 1
        delay = self.execution.prefetch_steps if prefix is not None else 0
        self.query = Query(
            self.query_id, self.run_id, slot, sources, slot + delay if prefix is not None else None
        )
        # build_policy_observation stacks owned arrays; the worker only reads them.
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=delay)
            if prefix is not None and self.execution.execution_mode == "rtc"
            else {}
        )
        self.model.query_context = dict(
            run_id=self.run_id,
            query_id=self.query_id,
        )
        self.model.submit("predict", observation, **kwargs)
        if commands is not None:
            self.plan.frozen.update(commands)
        return True

    def _prepare_prefix(self, row, slot):
        start = slot - self.plan.anchor_slot
        prefix = self.plan.actions[start:]
        rtc = self.execution.execution_mode == "rtc"
        if rtc:
            prefix = prefix.copy()
        commands = {}
        previous = self.previous_arm
        for i in range(self.execution.prefetch_steps):
            if self.poll_operator is not None:
                self.poll_operator()
            if (
                not self._has_motion_authority()
                or not self._within_budget()
                or time.monotonic_ns()
                > self.started_ns
                + slot * self.dt_ns
                + int(self.execution.max_tick_lateness_s * 1e9)
            ):
                return None, None
            decoded = self.realizer.realize(
                policy_action_intent(prefix[i], self.policy_info.action_mode),
                row.arm["qpos"][0],
                previous,
            )
            if decoded.arm_qpos is None:
                if (
                    decoded.ik_result is not None
                    and decoded.ik_result.failure_kind == IKFailureKind.INVALID_OUTPUT
                ):
                    raise RuntimeError(f"online IK technical failure: {decoded.ik_result.reason}")
                return None, None
            command = RobotCommand(self.run_id, decoded.arm_qpos, decoded.hand_qpos)
            commands[slot + i] = command
            previous = command.arm_qpos
            if rtc:
                prefix[i] = self.realizer.control_from_command(
                    command, self.policy_info.action_mode
                )
        return prefix, commands

    def _record(self, row, command=None, result=None):
        if self.recorder is not None and row is not None:
            self.recorder.add_frame(row, command, result)

    def _command_for_slot(self, row, slot):
        plan = self.plan
        command = plan.frozen.get(slot)
        if command is not None:
            if not self.realizer.frozen_is_valid(
                command, row.arm["qpos"][0], self.previous_arm, self.policy_info.action_mode
            ):
                self._invalidate()
                return
        else:
            decoded = self.realizer.realize(
                policy_action_intent(
                    plan.actions[slot - plan.anchor_slot], self.policy_info.action_mode
                ),
                row.arm["qpos"][0],
                self.previous_arm,
            )
            if decoded.arm_qpos is None:
                if (
                    decoded.ik_result is not None
                    and decoded.ik_result.failure_kind == IKFailureKind.INVALID_OUTPUT
                ):
                    raise RuntimeError(f"online IK technical failure: {decoded.ik_result.reason}")
                self._invalidate()
                return
            command = RobotCommand(self.run_id, decoded.arm_qpos, decoded.hand_qpos)
        return command

    def _execute_slot(self, row, slot):
        plan = self.plan
        command = self._command_for_slot(row, slot)
        if command is None:
            self._record(row)
            return
        now = time.monotonic_ns()
        if not self._has_motion_authority() or self.shared.quit_requested:
            self._finish_episode("before_dispatch_revoked")
            return
        if (expired := self._expired_budget()) is not None:
            self._finish_budget(expired)
            return
        if now > self.started_ns + slot * self.dt_ns + int(
            self.execution.max_tick_lateness_s * 1e9
        ) or not decision_is_fresh(plan.sources, now, self.execution.max_decision_age_s):
            self._invalidate()
            self._record(row)
            return
        budget_limit = self._budget_limit()
        action_deadline_ns = self._dispatch_deadline_ns(row, slot)
        valid_until_ns = (
            min(action_deadline_ns, budget_limit[0]) if budget_limit else action_deadline_ns
        )
        if time.monotonic_ns() >= valid_until_ns:
            self._record(row)
            if (expired := self._expired_budget()) is not None:
                self._finish_budget(expired)
            else:
                self._invalidate()
            return
        result, failure = None, None
        if self.execute:
            try:
                result = self.robot.send_action(command, valid_until_ns=valid_until_ns)
            except (DispatchError, DispatchInterrupted) as exc:
                result, failure = exc.result, exc
        if (self.results is not None and self.results.current is not None
                and result is not None
                and (result.arm != DispatchStatus.NOT_CALLED or result.hand != DispatchStatus.NOT_CALLED)):
            self.results.current["metrics"]["dispatch_count"] += 1
        # Admission deadlines only govern entry into each SDK. Total owner budgets
        # also govern completion; an accepted late return remains ACCEPTED evidence.
        expired = (
            budget_limit
            if budget_limit is not None and time.monotonic_ns() >= budget_limit[0]
            else None
        )
        timed_out = expired is not None
        cancelled = isinstance(failure, DispatchInterrupted)
        cause = failure.cause if isinstance(failure, DispatchError) else None
        # A total budget explains a refusal only when it constrained this call.
        # A coincident action deadline remains an independent technical failure.
        budget_rejection = (
            cause == "deadline_expired"
            and expired is not None
            and budget_limit[0] < action_deadline_ns
        )
        with self.shared.motion_lock:
            operator_revocation = (
                cause == "authority_revoked"
                and int(self.shared.run_ended_id) == self.run_id
                and int(self.shared.run_ended_reason)
                in (int(RunEndReason.OPERATOR), int(RunEndReason.QUIT))
                and not self.shared.estop_request
                and not self.shared.error_state
            )
        clean_partial = result is not None and all(
            status in (DispatchStatus.ACCEPTED, DispatchStatus.NOT_CALLED)
            for status in (result.arm, result.hand)
        )
        accepted = (
            result is not None
            and result.arm == DispatchStatus.ACCEPTED
            and result.hand == DispatchStatus.ACCEPTED
        )
        failed = (
            timed_out
            or failure is not None
            or (self.execute and self.execution.execution_mode != "sync" and not accepted)
        )
        stop_error = record_error = None
        try:
            if failed:
                if cancelled:
                    self.shared.estop_request = True
                reason = (
                    RunEndReason.ESTOP
                    if cancelled
                    else RunEndReason.TIMEOUT
                    if timed_out and (failure is None or budget_rejection)
                    else RunEndReason.EXECUTOR_BOUNDARY
                    if failure is not None and failure.revoked
                    else RunEndReason.HARDWARE_FAULT
                )
                revoke_motion_if_run_id(self.shared, self.run_id, reason=reason)
                try:
                    self.robot.stop()
                except Exception as exc:
                    stop_error = exc
                    if failure is not None:
                        logger.exception("stop after dispatch also failed")
            elif not self.execute or (result is not None and result.continued):
                self.previous_arm = command.arm_qpos
                self.wait_started_ns = None
        finally:
            try:
                self._record(row, command, result)
            except Exception as exc:
                record_error = exc
                logger.exception("recording after dispatch failed")
        if failed:
            normal_boundary = clean_partial and (operator_revocation or budget_rejection)
            error = failure if failure is not None and not normal_boundary else None
            if error is None and expired is not None and expired[1] == "wait":
                error = TimeoutError("Policy WAIT budget exhausted during dispatch")
            if (
                error is None
                and not normal_boundary
                and self.execute
                and self.execution.execution_mode != "sync"
                and not accepted
            ):
                error = DispatchError("Strict chunk dispatch was not confirmed", result)
            error = error or stop_error or record_error
            try:
                self._finish_episode(
                    "dispatch_cancelled"
                    if cancelled
                    else f"dispatch_return_{expired[1]}_timeout"
                    if timed_out and (failure is None or budget_rejection)
                    else "dispatch_unconfirmed_or_failed",
                    run_end_reason=reason,
                    stop_motion=False,
                    errors=[
                        error_detail(stage, exc)
                        for stage, exc in (("stop", stop_error), ("recording", record_error))
                        if exc is not None
                    ],
                )
            except Exception as exc:
                if error is not None:
                    raise error from exc
                raise
            if error is not None:
                raise error
            return
        if record_error is not None:
            raise record_error
        if slot + 1 == plan.segment_end and self.execution.execution_mode == "sync":
            self.plan = None

    def _handoff(self, slot):
        query = self.query
        if (
            query is not None
            and query.valid
            and query.handoff_slot is not None
            and slot >= query.handoff_slot
        ):
            if (
                slot != query.handoff_slot
                or query.result is None
                or time.monotonic_ns()
                > self.next_step_ns - self.dt_ns + int(self.execution.max_tick_lateness_s * 1e9)
                or not decision_is_fresh(
                    query.sources, time.monotonic_ns(), self.execution.max_decision_age_s
                )
            ):
                self._invalidate()
            else:
                anchor = slot if query.bootstrap else query.slot
                self.plan = Plan(
                    actions=query.result,
                    anchor_slot=anchor,
                    segment_end=slot + self.policy_info.n_action_steps,
                    sources=query.sources,
                    frozen={},
                )
                self.query = None

    def _prefetch(self, row, rows, slot):
        if (
            self.execution.execution_mode != "sync"
            and slot == self.plan.segment_end - self.execution.prefetch_steps
        ):
            if not rows or self.query is not None or self.model.future is not None:
                self._invalidate()
            else:
                prefix, commands = self._prepare_prefix(row, slot)
                if prefix is None or not self._submit(rows, slot, prefix=prefix, commands=commands):
                    self._invalidate()

    def step(self):
        self.robot.check()
        if self.run_id is not None and (
            not self._has_motion_authority() or self.shared.quit_requested
        ):
            self._finish_episode("owner_authority_revoked")
        self._poll_model()
        if self.recorder is not None:
            self.recorder.check_error()
        if self.run_id is None:
            if self.preparing_epoch is None:
                self.shared.stop_request = int(StopRequest.NONE)
            if self.preparing_epoch is not None or self.shared.start_request:
                self._begin_episode()
            if self.run_id is None:
                return
        now = time.monotonic_ns()
        if (expired := self._expired_budget()) is not None:
            self._finish_budget(expired)
            return
        if now < self.next_step_ns:
            return
        slot = (now - self.started_ns) // self.dt_ns
        tick_failed = False
        try:
            missed = slot != self.last_slot + 1
            if self.results is not None and self.results.current is not None:
                self.results.current["metrics"]["skipped_slots"] += max(0, slot - self.last_slot - 1)
            self.last_slot = slot
            self.next_step_ns = self.started_ns + (slot + 1) * self.dt_ns
            row = self._read_observation()
            self._poll_model()
            if (
                missed
                or time.monotonic_ns() - (self.next_step_ns - self.dt_ns)
                > int(self.execution.max_tick_lateness_s * 1e9)
                or row is None
            ):
                self.history.clear()
                self._invalidate()
                self._record(row)
                return
            self.history.append(row)
            rows = tuple(self.history) if len(self.history) == self.history.maxlen else ()
            self._handoff(slot)
            if self.plan is None:
                if self.wait_started_ns is None:
                    self.wait_started_ns = now
                if rows and self.query is None and self.model.future is None:
                    self._submit(rows, slot)
                self._record(row)
            else:
                self._prefetch(row, rows, slot)
                if self.plan is not None:
                    if slot >= self.plan.segment_end:
                        self._invalidate()
                        self._record(row)
                    else:
                        self._execute_slot(row, slot)
                else:
                    self._record(row)
        except BaseException:
            tick_failed = True
            raise
        finally:
            # Input/prefix construction may cross the total deadline without dispatch.
            if not tick_failed and (expired := self._expired_budget()) is not None:
                self._finish_budget(expired)

    def run(self):
        failure = None
        try:
            while self.shared.sensors.is_running.value and not self.shared.quit_requested:
                if self.poll_operator is not None:
                    self.poll_operator()
                if self.shared.estop_request or self.shared.error_state:
                    break
                if self.run_id is None and self.preparing_epoch is None:
                    self.robot.service_idle()
                self.step()
                time.sleep(0.001 if self.run_id is not None else 0.005)
        except KeyboardInterrupt as exc:
            self.shared.estop_request = True
            failure = exc
        except Exception as exc:
            failure = exc
            self.shared.error_state = True
            self.shared.sensors.is_running.value = False
        finally:
            reason = (
                RunEndReason.RECORDING_FAILURE
                if isinstance(failure, RecordingError)
                else RunEndReason.HARDWARE_FAULT
                if isinstance(failure, DispatchError)
                else RunEndReason.POLICY_FAILURE
                if failure is not None
                else RunEndReason.RUNTIME_SHUTDOWN
            )
            try:
                self._finish_episode(str(failure) if failure else "shutdown", run_end_reason=reason)
            except Exception as exc:
                failure = failure or exc
                logger.exception("episode cleanup failed")
            try:
                if self.robot.stop_required:
                    self.robot.stop()
            except Exception as exc:
                failure = failure or exc
                logger.exception(
                    "policy shutdown stop failed before remaining artifact finalization"
                )
            try:
                if self.recording_started or (self.results is not None and self.results.current is not None):
                    self._finalize_outputs(
                        "start_cancelled", str(failure) if failure else "shutdown"
                    )
            except Exception as exc:
                failure = failure or exc
            if self.recorder is not None:
                try:
                    self.recorder.close()
                except Exception as exc:
                    failure = failure or exc
                    logger.exception("policy recorder close failed after episode finalization")
            try:
                self.model.close()
            except Exception as exc:
                failure = failure or exc
                logger.exception("policy model worker close failed")
        if failure is not None:
            raise failure
