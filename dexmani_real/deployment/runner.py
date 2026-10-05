"""Device owner on one control grid, with one serial model worker in all modes."""

import time
from dataclasses import asdict, dataclass

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
    AsyncEpisodeRecorder,
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.robot.action import ActionRealizer
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.robot import DispatchError, DispatchStatus
from dexmani_real.runtime.observation import ObservationHistory, read_observation, sample_is_fresh
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    StopRequest,
    _begin_motion_locked,
    revoke_motion_if_run_id,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class PolicyRunner:
    def __init__(
        self,
        shared,
        runtime,
        policy_info,
        *,
        robot,
        poll_operator=None,
        model_runtime,
        execution_config,
        fingertip_runtime,
        execute,
        max_running_s,
        num_episodes=1,
        recording_config=None,
    ):
        self.shared = shared
        self.robot = robot
        self.poll_operator = poll_operator
        self.robot.before_send = self._dispatch_allowed
        self.runtime = runtime
        self.policy_info = policy_info
        self.model = model_runtime
        self.execution = execution_config.validate(policy_info)
        self.fingertip_runtime = fingertip_runtime
        self.execute = execute
        self.max_running_s = max_running_s
        self.num_episodes = num_episodes
        self.recording_config = recording_config
        self.recorder = (
            AsyncEpisodeRecorder(
                recording_config.data_dir,
                control_hz=1.0 / policy_info.control_dt_s,
                rgb_shape=(runtime.camera.height, runtime.camera.width, 3),
                execution_path=f"worker_grid_{self.execution.execution_mode}_v1",
            )
            if recording_config is not None
            else None
        )
        self.history = ObservationHistory(policy_info.n_obs_steps)
        self.plan = None
        self.query = None
        self.query_id = 0
        self.wait_started_ns = None
        self.preparing_epoch = None
        self.reset_ready = False
        self.events = []
        self.dispatch_row = None
        self.last_slot = -1
        self.dt_ns = int(policy_info.control_dt_s * 1e9)
        self.run_id = None
        self.started_ns = 0
        self.next_step_ns = 0
        self.previous_arm = None
        self.completed = 0
        self.realizer = ActionRealizer.for_mode(runtime, policy_info.action_mode)
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

    def _within_budget(self):
        return (
            self.run_id is None
            or self.max_running_s is None
            or time.monotonic_ns() - self.started_ns < int(self.max_running_s * 1e9)
        )

    def _dispatch_allowed(self):
        if not self._within_budget():
            return False
        if self.run_id is None:
            return True
        now = time.monotonic_ns()
        row = self.dispatch_row
        return (
            self.plan is not None
            and row is not None
            and decision_is_fresh(self.plan.sources, now, self.execution.max_decision_age_s)
            and sample_is_fresh(
                row.arm["timestamp_ns"][0], self.runtime.arm.feedback_max_age_s, now
            )
            and sample_is_fresh(
                row.hand["timestamp_ns"][0], self.runtime.hand.feedback_max_age_s, now
            )
            and now
            <= self.started_ns
            + self.last_slot * self.dt_ns
            + int(self.execution.max_tick_lateness_s * 1e9)
            and (
                self.wait_started_ns is None
                or now - self.wait_started_ns < int(self.execution.max_wait_s * 1e9)
            )
        )

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
        error = None
        try:
            if stop_motion:
                self.robot.stop()
        except Exception as exc:
            error = exc
            self.events.append(dict(event="stop_failure", run_id=epoch, detail=str(exc)))
        try:
            if self.recorder is not None:
                self.recorder.policy_trace = {"execute": self.execute, "events": self.events}
                self.recorder.save_episode(reason=reason.name.lower())
        finally:
            logger.info("policy episode %d ended: %s (%s)", self.completed, reason.name, detail)
            if self.completed >= self.num_episodes:
                self.shared.quit_requested.value = True
        if error is not None:
            raise error

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

    def set_execution_mode(self, execution_config):
        if (
            self.run_id is not None
            or self.preparing_epoch is not None
            or self.model.future is not None
        ):
            raise RuntimeError("Mode changes require an idle owner and a reclaimed model Future")
        execution_config.validate(self.policy_info)
        self.execution = execution_config
        self.plan = self.query = None
        self.model.submit(
            "configure_execution",
            execution_config.execution_mode,
            execution_config.rtc_guidance_cap,
            warmup=True,
            rgb_hw=(self.runtime.camera.height, self.runtime.camera.width),
        )
        if self.recorder is not None:
            self.recorder.execution_path = f"worker_grid_{execution_config.execution_mode}_v1"

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
            return
        committed = False
        try:
            self.realizer.reset_episode()
            if self.recorder is not None:
                cfg = self.recording_config
                if not self.recorder.start_episode(
                    task_label=cfg.task_label,
                    episode_name=f"episode_{self.completed + 1:03d}",
                    **snapshot_recording_metadata(
                        self.shared, self.runtime, collection_source="policy_rollout"
                    ),
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
        finally:
            if not committed and self.recorder is not None and self.recorder.is_recording:
                self.recorder.save_episode(reason="start_cancelled")

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
            self._invalidate("result_decision_age")
            return
        future = np.asarray(result.value)
        expected = (
            self.policy_info.horizon - self.policy_info.n_obs_steps + 1,
            physical_action_dim(self.policy_info.action_mode),
        )
        if future.shape != expected or future.dtype.kind != "f" or not np.isfinite(future).all():
            raise ValueError(f"Policy future requires finite floating point {expected}")
        query.result = future
        if query.handoff_slot is None:
            # Bootstrap was not executed during inference; preserve its entire head.
            query.handoff_slot = max(self.last_slot + 1, (now - self.started_ns) // self.dt_ns + 1)
            query.bootstrap = True

    def _submit(self, rows, slot, *, prefix=None, commands=None):
        observation = build_policy_observation(
            rows, self.policy_info, fingertip_runtime=self.fingertip_runtime
        )
        if observation is None:
            return False
        now = time.monotonic_ns()
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
                self._event("prefix_rejected", slot=slot, detail=decoded.ik_result.reason)
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
            self.recorder.add_frame(build_episode_frame(row, command, result))

    def _send(self, row, slot):
        plan = self.plan
        command = plan.frozen.get(slot)
        if command is not None:
            if not self.realizer.frozen_is_valid(
                command, row.arm["qpos"][0], self.previous_arm, self.policy_info.action_mode
            ):
                self._invalidate("frozen_dynamic_rejection")
                self._record(row)
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
                if decoded.ik_result.failure_kind == IKFailureKind.INVALID_OUTPUT:
                    raise RuntimeError(f"online IK technical failure: {decoded.ik_result.reason}")
                self._invalidate("action_realization_rejected")
                self._record(row)
                return
            command = RobotCommand(self.run_id, decoded.arm_qpos, decoded.hand_qpos)
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
        self.dispatch_row = row
        if not self._dispatch_allowed():
            self._invalidate("dispatch_feedback_or_budget")
            self._record(row)
            return
        result, failure = None, None
        if self.execute:
            try:
                result = self.robot.send_action(command)
            except DispatchError as exc:
                result, failure = exc.result, exc
        accepted = (
            result is not None
            and result.arm == DispatchStatus.ACCEPTED
            and result.hand == DispatchStatus.ACCEPTED
        )
        self._event(
            "dispatch",
            slot=slot,
            query_id=plan.query_id,
            start_ns=now,
            arm=int(result.arm) if result else 0,
            hand=int(result.hand) if result else 0,
            logical_consume=not self.execute,
        )
        if failure is not None or (
            self.execute and self.execution.execution_mode != "sync" and not accepted
        ):
            # Revoke/stop before the recorder may block; retain the attempted row.
            reason = (
                (
                    RunEndReason.TIMEOUT
                    if not self._within_budget()
                    else RunEndReason.EXECUTOR_BOUNDARY
                )
                if failure is not None and failure.revoked
                else RunEndReason.HARDWARE_FAULT
            )
            revoke_motion_if_run_id(self.shared, self.run_id, reason=reason)
            stop_error = None
            try:
                self.robot.stop()
            except Exception as exc:
                stop_error = exc
                self._event("stop_failure", detail=str(exc))
            self._record(row, command, result)
            self._finish_episode(
                "dispatch_unconfirmed_or_failed", run_end_reason=reason, stop_motion=False
            )
            if stop_error is not None:
                raise stop_error
            if failure is not None and not failure.revoked:
                raise failure
            return
        if not self.execute or (result is not None and result.continued):
            self.previous_arm = command.arm_qpos
            self.wait_started_ns = None
        self._record(row, command, result)
        if slot + 1 == plan.segment_end and self.execution.execution_mode == "sync":
            self.plan = None

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
        if self.wait_started_ns is not None and now - self.wait_started_ns >= int(
            self.execution.max_wait_s * 1e9
        ):
            self._finish_episode("wait_timeout", run_end_reason=RunEndReason.TIMEOUT)
            return
        if now < self.next_step_ns:
            return
        tick_start = now
        slot = (now - self.started_ns) // self.dt_ns
        missed = slot != self.last_slot + 1
        self.last_slot = slot
        self.next_step_ns = self.started_ns + (slot + 1) * self.dt_ns
        row = self._read_observation()
        if (
            missed
            or time.monotonic_ns() - (self.next_step_ns - self.dt_ns)
            > int(self.execution.max_tick_lateness_s * 1e9)
            or row is None
        ):
            self.history.clear()
            self._invalidate("missed_slot_or_observation")
            self._event(
                "owner_tick",
                slot=slot,
                duration_ns=time.monotonic_ns() - tick_start,
                lateness_ns=time.monotonic_ns() - (self.started_ns + slot * self.dt_ns),
            )
            self._record(row)
            return
        self.history.append(row)
        rows = self.history.ready_rows()
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
                    query.result,
                    anchor,
                    slot + self.policy_info.n_action_steps,
                    query.query_id,
                    query.sources,
                    {},
                )
                self.query = None
                self._event(
                    "bootstrap_reanchor" if query.bootstrap else "handoff",
                    query_id=query.query_id,
                    slot=slot,
                    query_slot=query.slot,
                )
        if self.plan is None:
            if self.wait_started_ns is None:
                self.wait_started_ns = now
            if rows and self.query is None and self.model.future is None:
                self._submit(rows, slot)
            self._record(row)
        else:
            if (
                self.execution.execution_mode != "sync"
                and slot == self.plan.segment_end - self.execution.prefetch_steps
            ):
                if not rows or self.query is not None or self.model.future is not None:
                    self._invalidate("prefetch_unavailable")
                else:
                    prefix, commands = self._prepare_prefix(row, slot)
                    if prefix is None or not self._submit(
                        rows, slot, prefix=prefix, commands=commands
                    ):
                        self._invalidate("prefetch_rejected")
            if self.plan is not None:
                if slot >= self.plan.segment_end:
                    self._invalidate("segment_exhausted")
                    self._record(row)
                else:
                    self._send(row, slot)
            else:
                self._record(row)
        self._event(
            "owner_tick",
            slot=slot,
            duration_ns=time.monotonic_ns() - tick_start,
            lateness_ns=tick_start - (self.started_ns + slot * self.dt_ns),
        )

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
            self.robot.before_send = None
            try:
                if self.robot._motion_active or self.robot._hand_stop_pending:
                    self.robot.stop()
            except Exception as exc:
                failure = failure or exc
            if self.recorder is not None:
                try:
                    self.recorder.close()
                except Exception as exc:
                    failure = failure or exc
            self.model.close()
        if failure is not None:
            raise failure


@dataclass
class Query:
    query_id: int
    run_id: int
    slot: int
    sources: dict
    handoff_slot: int | None
    valid: bool = True
    result: object = None
    bootstrap: bool = False


@dataclass
class Plan:
    actions: np.ndarray
    anchor_slot: int
    segment_end: int
    query_id: int
    sources: dict
    frozen: dict
