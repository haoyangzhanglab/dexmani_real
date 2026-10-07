"""Device owner on one control grid, with one serial model worker in all modes."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.recording.recorder import (
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.recording.results import error_detail, finalize_policy_attempt
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.robot import DispatchError, DispatchInterrupted, DispatchStatus
from dexmani_real.runtime.observation import (
    ObservationHistory,
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
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


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
    query_id: int
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
        execution_config,
        kinematics,
        execute,
        max_running_s,
        num_episodes=1,
        recording_config=None,
        camera_calibration=None,
        results=None,
    ):
        if recorder is not None:
            if results is None or recording_config is None:
                raise ValueError("policy recording requires recorder, recording_config and results")
            if not (
                recorder.data_dir.resolve()
                == results.directory.resolve()
                == Path(recording_config.data_dir).resolve()
            ):
                raise ValueError(
                    "recorder, recording_config and results must share one session directory"
                )
        elif recording_config is not None:
            raise ValueError("recording_config requires an assembled recorder")
        self.results = results
        self.query_arrays = {}
        self.shared = shared
        self.robot = robot
        self.poll_operator = poll_operator
        self.runtime = runtime
        self.policy_info = policy_info
        self.model = model_runtime
        self.execution = execution_config.validate(policy_info)
        self.kinematics = kinematics
        self.execute = execute
        self.max_running_s = max_running_s
        self.num_episodes = num_episodes
        self.recording_config = recording_config
        self.camera_calibration = camera_calibration
        self.recorder = recorder
        self.history = ObservationHistory(policy_info.n_obs_steps)
        self.recording_started = False
        self.plan = None
        self.query = None
        self.query_id = 0
        self.wait_started_ns = None
        self.preparing_epoch = None
        self.reset_ready = False
        self.events = []
        self._active_tick = None
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

    def _budget_deadline_ns(self):
        deadlines = []
        if self.run_id is not None:
            if self.max_running_s is not None:
                deadlines.append(self.started_ns + int(self.max_running_s * 1e9))
            if self.wait_started_ns is not None:
                deadlines.append(self.wait_started_ns + int(self.execution.max_wait_s * 1e9))
        return min(deadlines) if deadlines else None

    def _within_budget(self):
        deadline = self._budget_deadline_ns()
        return deadline is None or time.monotonic_ns() < deadline

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
        budget_deadline = self._budget_deadline_ns()
        if budget_deadline is not None:
            deadlines.append(budget_deadline)
        return min(deadlines)

    def _has_motion_authority(self):
        return (
            self.run_id is not None
            and self.shared.is_running.value
            and not self.shared.error_state.value
            and not self.shared.estop_request.value
            and int(self.shared.run_id.value) == self.run_id
            and int(self.shared.safety_state.value) == int(SafetyState.RUNNING)
        )

    def _event(self, name, **details):
        self.events.append(
            {"event": name, "time_ns": time.monotonic_ns(), "run_id": self.run_id, **details}
        )

    def _end_current_tick(self):
        tick, self._active_tick = self._active_tick, None
        if tick is None:
            return
        epoch, slot, scheduled_ns, started_ns = tick
        try:
            self._event(
                "owner_tick",
                run_id=epoch,
                slot=slot,
                scheduled_ns=scheduled_ns,
                started_ns=started_ns,
                duration_ns=time.monotonic_ns() - started_ns,
                lateness_ns=started_ns - scheduled_ns,
            )
        except Exception:
            # Telemetry must not replace a control failure or prevent cleanup.
            logger.exception("owner tick trace failed")

    def _invalidate(self, reason):
        self.plan = None
        if self.query is not None:
            self.query.valid = False
            if self.model.future is None:
                self.query = None
        self._event("invalidate", reason=reason)
        if self.run_id is not None and self.wait_started_ns is None:
            self.wait_started_ns = time.monotonic_ns()

    def _finish_episode(
        self, detail, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, stop_motion=True
    ):
        if self.run_id is None:
            return
        epoch = self.run_id
        revoke_motion_if_run_id(self.shared, epoch, reason=run_end_reason)
        with self.shared.motion_lock:
            reason = (
                RunEndReason(int(self.shared.run_ended_reason.value))
                if int(self.shared.run_ended_id.value) == epoch
                else run_end_reason
            )
        self._invalidate(detail)
        self._event("end", reason=reason.name.lower(), detail=detail)
        self.history.clear()
        self.previous_arm = None
        self.run_id = None
        self.completed += 1
        self.shared.stop_request.value = int(StopRequest.NONE)
        errors, failure = [], None
        try:
            if stop_motion:
                self.robot.stop()
        except Exception as exc:
            failure = exc
            errors.append(error_detail("stop", exc))
            self.events.append(dict(event="stop_failure", run_id=epoch, detail=str(exc)))
        try:
            self._finalize_attempt(reason.name.lower(), detail, errors)
        except Exception as exc:
            failure = failure or exc
        finally:
            logger.info("policy episode %d ended: %s (%s)", self.completed, reason.name, detail)
            if self.completed >= self.num_episodes:
                self.shared.quit_requested.value = True
        if failure is not None:
            raise failure

    def _finalize_attempt(self, reason, detail, errors=None):
        # Nested termination closes the tick after stop/record, before the snapshot.
        self._end_current_tick()
        try:
            finalize_policy_attempt(
                self.results,
                self.recorder,
                recording_started=self.recording_started,
                reason=reason,
                detail=detail,
                events=self.events,
                query_arrays=self.query_arrays,
                errors=errors,
            )
        finally:
            self.query_arrays = {}
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
                if not self.shared.start_request.value or self.model.future is not None:
                    return
                self.shared.start_request.value = False
                if (
                    self.shared.stop_request.value
                    or self.shared.quit_requested.value
                    or not self.shared.is_running.value
                    or self.shared.error_state.value
                    or self.shared.estop_request.value
                    or int(self.shared.safety_state.value) != int(SafetyState.ARMED)
                ):
                    return
                epoch = int(self.shared.run_id.value)
            if self._start_observation() is None:
                return
            self.recording_started = False
            if self.results is not None:
                self.results.prepare(recording=self.recorder is not None)
            self.events = []
            self.query_arrays = {}
            self.preparing_epoch = epoch
            self.reset_ready = False
            self.model.submit("reset_episode")
            return
        if not self.reset_ready:
            return
        epoch, self.preparing_epoch = self.preparing_epoch, None
        if (
            epoch != int(self.shared.run_id.value)
            or self.shared.quit_requested.value
            or self.shared.stop_request.value
        ):
            self._finalize_attempt("start_cancelled", "cancelled during model reset")
            return
        committed = False
        start_failure = None
        try:
            self.realizer.reset_episode()
            if self.recorder is not None:
                cfg = self.recording_config
                metadata = snapshot_recording_metadata(
                    self.shared,
                    self.runtime,
                    collection_source="policy_rollout",
                    camera_calibration=self.camera_calibration,
                )
                self.recording_started = True
                if not self.recorder.start_episode(
                    task_label=cfg.task_label,
                    episode_name=f"episode_{self.completed + 1:03d}",
                    **metadata,
                ):
                    raise RecordingError("policy recorder refused START")
            # Recorder startup can block; use a new row before granting motion.
            if self._start_observation() is None:
                return
            with self.shared.motion_lock:
                admitted = (
                    _begin_motion_locked(self.shared)
                    if epoch == int(self.shared.run_id.value)
                    and not self.shared.quit_requested.value
                    and not self.shared.stop_request.value
                    else None
                )
            if admitted is None:
                return
            self.run_id, self.started_ns = admitted
            if self.results is not None:
                self.results.entered(self.run_id)
            committed = True
            self.events = []
            self.history.clear()
            self.plan = self.query = None
            self.previous_arm = None
            self.last_slot = -1
            self.next_step_ns = self.started_ns
            self.wait_started_ns = self.started_ns
            self._event(
                "begin",
                mode=self.execution.execution_mode,
                execute=self.execute,
                execution=asdict(self.execution),
            )
        except BaseException as exc:
            start_failure = exc
            raise
        finally:
            if not committed:
                try:
                    self._finalize_attempt(
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
        self._event(
            "query_complete",
            run_id=query.run_id,
            query_id=query.query_id,
            started_ns=result.started_ns,
            completed_ns=result.completed_ns,
            error=str(result.error) if result.error is not None else None,
        )
        if result.error is None:
            future = np.asarray(result.value)
            finite = bool(np.isfinite(future).all()) if future.dtype.kind in "fiubc" else None
            expected = (
                self.policy_info.horizon - self.policy_info.n_obs_steps + 1,
                physical_action_dim(self.policy_info.action_mode),
            )
            archivable_future = future.shape == expected and future.dtype.kind == "f"
            valid_future = archivable_future and finite
            self._event(
                "query_result",
                run_id=query.run_id,
                query_id=query.query_id,
                shape=list(future.shape),
                dtype=str(future.dtype),
                finite=finite,
            )
            if (
                archivable_future
                and self.results is not None
                and self.results.attempt is not None
                and query.run_id == self.run_id
            ):
                # Rejected NaN/Inf remain evidence in NPZ, never executable targets or JSON.
                self.query_arrays[f"query_{query.query_id}"] = future.copy()
        if not query.valid or query.run_id != self.run_id or not self._has_motion_authority():
            logger.info(
                "retired query %s from run %s reclaimed; error=%s",
                query.query_id,
                query.run_id,
                result.error,
            )
            if self.results is not None:
                self.results.session["retired_queries"].append(
                    dict(
                        run_id=query.run_id,
                        query_id=query.query_id,
                        started_ns=result.started_ns,
                        completed_ns=result.completed_ns,
                        reason="retired",
                        error=str(result.error) if result.error else None,
                    )
                )
            self.query = None
            return
        if result.error is not None:
            raise result.error
        now = time.monotonic_ns()
        if not decision_is_fresh(query.sources, now, self.execution.max_decision_age_s):
            self._invalidate("result_decision_age")
            return
        if not valid_future:
            raise ValueError(f"Policy future requires finite floating point {expected}")
        query.result = future
        if query.handoff_slot is None:
            # Bootstrap was not executed during inference; preserve its entire head.
            query.handoff_slot = max(self.last_slot + 1, (now - self.started_ns) // self.dt_ns + 1)
            query.bootstrap = True

    def _submit(self, rows, slot, *, prefix=None, commands=None):
        build_started_ns = time.monotonic_ns()
        observation = build_policy_observation(rows, self.policy_info, kinematics=self.kinematics)
        if observation is None:
            return False
        now = time.monotonic_ns()
        self._event("observation_build", duration_ns=now - build_started_ns)
        if not self._has_motion_authority() or not self._within_budget():
            return False
        if now > self.started_ns + slot * self.dt_ns + int(
            self.execution.max_tick_lateness_s * 1e9
        ):
            self._invalidate("prefix_or_input_preparation_overrun")
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
            attempt_id=self.results.attempt["attempt_id"]
            if self.results is not None and self.results.attempt
            else None,
        )
        self.model.submit("predict", observation, **kwargs)
        if commands is not None:
            self.plan.frozen.update(commands)
        camera = rows[-1].camera
        self._event(
            "query_submit",
            query_id=self.query_id,
            slot=slot,
            sources=sources,
            submitted_ns=now,
            handoff_slot=self.query.handoff_slot,
            delay_steps=delay,
            observation_slots=list(range(slot - len(rows) + 1, slot + 1)),
            observation_sources=[
                policy_sources(row, self.policy_info.observation_fields) for row in rows
            ],
            camera_frame=int(camera["color_frame_number"]) if camera is not None else None,
            cloud_camera_sequence=rows[-1].pointcloud_camera_sequence,
        )
        return True

    def _prepare_prefix(self, row, slot):
        start = slot - self.plan.anchor_slot
        prefix = self.plan.actions[start:]
        rtc = self.execution.execution_mode == "rtc"
        if rtc:
            prefix = prefix.copy()
        commands = {}
        previous = self.previous_arm
        begin = time.monotonic_ns()
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
                self._event(
                    "prefix_rejected", slot=slot, detail="authorization_or_preparation_deadline"
                )
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
                self._event(
                    "prefix_rejected",
                    slot=slot,
                    query_id=self.plan.query_id,
                    action_index=start + i,
                    detail=decoded.rejection_reason,
                )
                return None, None
            command = RobotCommand(self.run_id, decoded.arm_qpos, decoded.hand_qpos)
            commands[slot + i] = command
            previous = command.arm_qpos
            if rtc:
                prefix[i] = self.realizer.control_from_command(
                    command, self.policy_info.action_mode
                )
        self._event("prefix_prepared", slot=slot, duration_ns=time.monotonic_ns() - begin)
        return prefix, commands

    def _record(self, row, command=None, result=None):
        if self.recorder is not None and row is not None:
            index = self.recorder.add_frame(build_episode_frame(row, command, result))
            if index is not None:
                self._event("record_submitted", slot=self.last_slot, raw_row_index=index)

    def _command_for_slot(self, row, slot):
        plan = self.plan
        self._event(
            "action_attempt",
            query_id=plan.query_id,
            action_index=slot - plan.anchor_slot,
            slot=slot,
        )
        command = plan.frozen.get(slot)
        if command is not None:
            if not self.realizer.frozen_is_valid(
                command, row.arm["qpos"][0], self.previous_arm, self.policy_info.action_mode
            ):
                self._invalidate(self.realizer.rejection_reason)
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
                self._invalidate(decoded.rejection_reason or "action_realization_rejected")
                return
            command = RobotCommand(self.run_id, decoded.arm_qpos, decoded.hand_qpos)
        self._event(
            "action_selected",
            query_id=plan.query_id,
            action_index=slot - plan.anchor_slot,
            slot=slot,
            arm_qpos=command.arm_qpos.tolist(),
            hand_qpos=command.hand_qpos.tolist() if command.hand_qpos is not None else None,
        )
        return command

    def _execute_slot(self, row, slot):
        plan = self.plan
        realization_start = time.monotonic_ns()
        command = self._command_for_slot(row, slot)
        self._event(
            "realization",
            query_id=plan.query_id,
            slot=slot,
            duration_ns=time.monotonic_ns() - realization_start,
        )
        if command is None:
            self._record(row)
            return
        now = time.monotonic_ns()
        if not self._has_motion_authority() or self.shared.quit_requested.value:
            self._finish_episode("before_dispatch_revoked")
            return
        if not self._within_budget():
            self._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
            return
        if now > self.started_ns + slot * self.dt_ns + int(
            self.execution.max_tick_lateness_s * 1e9
        ) or not decision_is_fresh(plan.sources, now, self.execution.max_decision_age_s):
            self._invalidate("dispatch_slot_or_decision_age")
            self._record(row)
            return
        valid_until_ns = self._dispatch_deadline_ns(row, slot)
        if time.monotonic_ns() >= valid_until_ns:
            self._record(row)
            if not self._within_budget():
                self._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
            else:
                self._invalidate("dispatch_feedback_or_budget")
            return
        result, failure = None, None
        sdk_started_ns = time.monotonic_ns()
        if self.execute:
            try:
                result = self.robot.send_action(command, valid_until_ns=valid_until_ns)
            except (DispatchError, DispatchInterrupted) as exc:
                result, failure = exc.result, exc
        # Admission deadlines only govern entry into each SDK. Total owner budgets
        # also govern completion; an accepted late return remains ACCEPTED evidence.
        timed_out = not self._within_budget()
        cancelled = isinstance(failure, DispatchInterrupted)
        accepted = (
            result is not None
            and result.arm == DispatchStatus.ACCEPTED
            and result.hand == DispatchStatus.ACCEPTED
        )
        self._event(
            "dispatch",
            slot=slot,
            query_id=plan.query_id,
            start_ns=sdk_started_ns,
            duration_ns=time.monotonic_ns() - sdk_started_ns,
            action_index=slot - plan.anchor_slot,
            arm=int(result.arm) if result else 0,
            hand=int(result.hand) if result else 0,
            logical_consume=not self.execute,
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
                    self.shared.estop_request.value = True
                reason = (
                    RunEndReason.ESTOP
                    if cancelled
                    else RunEndReason.TIMEOUT
                    if timed_out
                    else RunEndReason.EXECUTOR_BOUNDARY
                    if failure is not None and failure.revoked
                    else RunEndReason.HARDWARE_FAULT
                )
                revoke_motion_if_run_id(self.shared, self.run_id, reason=reason)
                try:
                    self.robot.stop()
                except Exception as exc:
                    stop_error = exc
                    self._event("stop_failure", exception_type=type(exc).__name__, detail=str(exc))
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
                self._event("recording_failure", exception_type=type(exc).__name__, detail=str(exc))
                logger.exception("recording after dispatch failed")
        if failed:
            error = (
                failure
                if cancelled or (failure is not None and not failure.revoked)
                else stop_error or record_error
            )
            try:
                self._finish_episode(
                    "dispatch_cancelled"
                    if cancelled
                    else "dispatch_return_timeout"
                    if timed_out
                    else "dispatch_unconfirmed_or_failed",
                    run_end_reason=reason,
                    stop_motion=False,
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
                self._invalidate("handoff_miss")
            else:
                anchor = slot if query.bootstrap else query.slot
                self.plan = Plan(
                    actions=query.result,
                    anchor_slot=anchor,
                    segment_end=slot + self.policy_info.n_action_steps,
                    query_id=query.query_id,
                    sources=query.sources,
                    frozen={},
                )
                self.query = None
                self._event(
                    "bootstrap_reanchor" if query.bootstrap else "handoff",
                    query_id=query.query_id,
                    slot=slot,
                    query_slot=query.slot,
                )

    def _prefetch(self, row, rows, slot):
        if (
            self.execution.execution_mode != "sync"
            and slot == self.plan.segment_end - self.execution.prefetch_steps
        ):
            if not rows or self.query is not None or self.model.future is not None:
                self._invalidate("prefetch_unavailable")
            else:
                prefix, commands = self._prepare_prefix(row, slot)
                if prefix is None or not self._submit(rows, slot, prefix=prefix, commands=commands):
                    self._invalidate("prefetch_rejected")

    def step(self):
        self.robot.check()
        if self.run_id is not None and (
            not self._has_motion_authority() or self.shared.quit_requested.value
        ):
            self._finish_episode("owner_authority_revoked")
        self._poll_model()
        if self.recorder is not None:
            self.recorder.check_error()
        if self.run_id is None:
            if self.preparing_epoch is None:
                self.shared.stop_request.value = int(StopRequest.NONE)
            if self.preparing_epoch is not None or self.shared.start_request.value:
                self._begin_episode()
            if self.run_id is None:
                return
        now = time.monotonic_ns()
        if not self._within_budget():
            self._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
            return
        if now < self.next_step_ns:
            return
        slot = (now - self.started_ns) // self.dt_ns
        self._active_tick = (self.run_id, slot, self.started_ns + slot * self.dt_ns, now)
        try:
            missed = slot != self.last_slot + 1
            self.last_slot = slot
            self.next_step_ns = self.started_ns + (slot + 1) * self.dt_ns
            observation_start = time.monotonic_ns()
            row = self._read_observation()
            self._event(
                "observation_read", slot=slot, duration_ns=time.monotonic_ns() - observation_start
            )
            self._poll_model()
            if (
                missed
                or time.monotonic_ns() - (self.next_step_ns - self.dt_ns)
                > int(self.execution.max_tick_lateness_s * 1e9)
                or row is None
            ):
                self.history.clear()
                self._invalidate("missed_slot_or_observation")
                self._record(row)
                return
            self.history.append(row)
            rows = self.history.ready_rows()
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
                        self._invalidate("segment_exhausted")
                        self._record(row)
                    else:
                        self._execute_slot(row, slot)
                else:
                    self._record(row)
        finally:
            self._end_current_tick()

    def run(self):
        failure = None
        try:
            while self.shared.is_running.value and not self.shared.quit_requested.value:
                if self.poll_operator is not None:
                    self.poll_operator()
                if self.shared.estop_request.value or self.shared.error_state.value:
                    break
                if self.run_id is None and self.preparing_epoch is None:
                    self.robot.service_idle()
                self.step()
                time.sleep(0.001 if self.run_id is not None else 0.005)
        except KeyboardInterrupt as exc:
            self.shared.estop_request.value = True
            failure = exc
        except Exception as exc:
            failure = exc
            self.shared.error_state.value = True
            self.shared.is_running.value = False
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
                if (
                    self.results is not None
                    and self.results.attempt is not None
                    and not self.results.failed
                ):
                    self._finalize_attempt(
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
            if (
                self.results is not None
                and self.query is not None
                and self.model.future is not None
            ):
                self.results.session["retired_queries"].append(
                    dict(
                        run_id=self.query.run_id,
                        query_id=self.query.query_id,
                        reason="pending_at_shutdown",
                    )
                )
            try:
                self.model.close()
            except Exception as exc:
                failure = failure or exc
                logger.exception("policy model worker close failed")
        if failure is not None:
            raise failure
