"""Synchronous policy inference, local control-row history and local action chunks."""

import time
from collections import deque

import numpy as np

from dexmani_real.deployment.action import (
    physical_action_dim,
    policy_action_intent,
)
from dexmani_real.deployment.observation import build_policy_observation
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.recording.recorder import (
    AsyncEpisodeRecorder,
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.robot.action import ActionRealizer
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.robot import DispatchError
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
        fingertip_runtime,
        execute,
        max_running_s,
        num_episodes=1,
        recording_config=None,
    ):
        self.shared = shared
        self.robot = robot
        self.poll_operator = poll_operator
        self.robot.before_send = self._within_budget
        self.runtime = runtime
        self.policy_info = policy_info
        self.model = model_runtime
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
            )
            if recording_config is not None
            else None
        )
        self.history = ObservationHistory(policy_info.n_obs_steps, policy_info.control_dt_s)
        self.action_queue = deque()
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
        self.requires_rgb_cloud_identity = requires_rgb and self.requires_cloud

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

    def _has_motion_authority(self):
        return (
            self.run_id is not None
            and self.shared.is_running.value
            and not self.shared.error_state.value
            and not self.shared.estop_request.value
            and int(self.shared.run_id.value) == self.run_id
            and int(self.shared.safety_state.value) == int(SafetyState.RUNNING)
        )

    def _finish_episode(
        self, reason, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, stop_motion=True
    ):
        if self.run_id is None:
            return
        revoke_motion_if_run_id(self.shared, self.run_id, reason=run_end_reason)
        self.shared.physical_home_completed.value = False
        stop_error = None
        try:
            if stop_motion:
                self.robot.stop()
        except Exception as exc:
            stop_error = exc
            logger.exception("episode stop failed")
            reason = f"{reason}:stop_failure"
        self.action_queue.clear()
        self.history.clear()
        self.previous_arm = None
        self.completed += 1
        self.run_id = None
        self.shared.stop_request.value = int(StopRequest.NONE)
        try:
            if self.recorder is not None:
                self.recorder.save_episode(reason=reason)
        finally:
            logger.info("policy episode %d ended: %s", self.completed, reason)
            if self.completed >= self.num_episodes:
                self.shared.quit_requested.value = True
        if stop_error is not None:
            raise stop_error

    def _start_observation(self):
        row = self._read_observation()
        if row is None:
            logger.warning("policy B rejected: required observation unavailable")
            return None
        if self.execute and (
            not self.shared.physical_home_completed.value
            or np.max(np.abs(row.arm["qpos"][0] - self.runtime.arm.home_qpos))
            > self.runtime.arm.homing.convergence_rad
            or np.max(np.abs(row.hand["qpos"][0] - np.deg2rad(self.runtime.hand.home_qpos_deg)))
            > np.deg2rad(self.runtime.hand.home_tolerance_deg)
        ):
            logger.warning("policy B requires completed HOME and current arm + hand home pose")
            return None
        try:
            observation = build_policy_observation(
                (row,) * self.policy_info.n_obs_steps,
                self.policy_info,
                fingertip_runtime=self.fingertip_runtime,
            )
        except Exception:
            # No motion or recording has begun; a fresh B may retry failed derivation.
            logger.warning(
                "policy B rejected: initial observation construction failed", exc_info=True
            )
            return None
        if observation is None:
            logger.warning("policy B rejected: required policy modality unavailable")
            return None
        return row, observation

    def _begin_episode(self):
        with self.shared.motion_lock:
            requested = bool(self.shared.start_request.value)
            self.shared.start_request.value = False
            if (
                not requested
                or self.shared.stop_request.value
                or self.shared.quit_requested.value
                or not self.shared.is_running.value
                or self.shared.error_state.value
                or self.shared.estop_request.value
                or int(self.shared.safety_state.value) != int(SafetyState.ARMED)
            ):
                return None
            preparation_epoch = int(self.shared.run_id.value)
        committed = False
        failure = None
        try:
            if self._start_observation() is None:
                return None
            if self.recorder is not None:
                cfg = self.recording_config
                if not self.recorder.start_episode(
                    task_label=cfg.task_label,
                    episode_name=f"episode_{self.completed + 1:03d}",
                    **snapshot_recording_metadata(
                        self.shared, self.runtime, collection_source="policy_rollout"
                    ),
                ):
                    raise RuntimeError("policy recorder refused START")
            self.model.reset_episode()
            self.realizer.reset_episode()
            # START and model reset may block: only a new row can authorize motion.
            initial = self._start_observation()
            if initial is None:
                return None
            with self.shared.motion_lock:
                self.shared.start_request.value = False
                epoch = (
                    _begin_motion_locked(self.shared)
                    if preparation_epoch == int(self.shared.run_id.value)
                    and not self.shared.quit_requested.value
                    and not self.shared.stop_request.value
                    and (not self.execute or self.shared.physical_home_completed.value)
                    else None
                )
            if epoch is None:
                return None
            self.run_id, self.started_ns = epoch
            committed = True
            self.shared.physical_home_completed.value = False
            self.history.clear()
            self.history.append(initial[0])
            self.action_queue.clear()
            self.previous_arm = None
            self.next_step_ns = 0
            return initial
        except BaseException as exc:
            failure = exc
            if isinstance(exc, KeyboardInterrupt):
                self.shared.estop_request.value = True
            raise
        finally:
            if not committed:
                try:
                    if self.recorder is not None and self.recorder.is_recording:
                        self.recorder.discard_episode(reason="start_cancelled")
                except Exception:
                    if failure is None:
                        raise
                    logger.exception("cancelled START recording cleanup also failed")
                finally:
                    with self.shared.motion_lock:
                        self.shared.start_request.value = False

    def step(self):
        self.robot.check()
        if self.recorder is not None:
            self.recorder.check_error()
        initial = None
        if self.run_id is None:
            with self.shared.motion_lock:
                if self.shared.stop_request.value:
                    self.shared.stop_request.value = int(StopRequest.NONE)
            if self.shared.start_request.value:
                initial = self._begin_episode()
            if initial is None:
                return
        if not self._has_motion_authority() or self.shared.quit_requested.value:
            reason = RunEndReason(int(self.shared.run_ended_reason.value))
            self._finish_episode(
                reason.name.lower(),
            )
            return
        now = time.monotonic_ns()
        if self.max_running_s is not None and now - self.started_ns >= int(
            self.max_running_s * 1e9
        ):
            self._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
            return
        if now < self.next_step_ns:
            return
        # Observation, synchronous inference and dispatch share this tick's budget.
        tick_start_ns = now
        self.next_step_ns = tick_start_ns + int(self.policy_info.control_dt_s * 1e9)
        row = initial[0] if initial is not None else self._read_observation()
        policy_row = row
        if row is None:
            self._finish_episode("required_observation_stale")
            return
        if initial is None:
            self.history.append(row)
        execution_state = policy_row
        if not self.action_queue:
            observation = (
                initial[1]
                if initial is not None
                else build_policy_observation(
                    self.history.padded(),
                    self.policy_info,
                    fingertip_runtime=self.fingertip_runtime,
                )
            )
            if observation is None:
                self._finish_episode("required_tactile_unavailable")
                return
            epoch = self.run_id
            prediction = self.model.predict(observation)
            self.robot.check()
            if (
                not self._has_motion_authority()
                or self.shared.quit_requested.value
                or epoch != int(self.shared.run_id.value)
            ):
                self._finish_episode("motion_revoked")
                return
            if self.recorder is not None:
                self.recorder.check_error()
            if self.max_running_s is not None and time.monotonic_ns() - self.started_ns >= int(
                self.max_running_s * 1e9
            ):
                self._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
                return
            prediction = np.asarray(prediction, dtype=np.float64)
            if (
                prediction.shape
                != (
                    self.policy_info.n_action_steps,
                    physical_action_dim(self.policy_info.action_mode),
                )
                or not np.isfinite(prediction).all()
            ):
                raise ValueError("policy prediction violates shape/finite contract")
            self.action_queue.extend(prediction)
            # Execution feedback after a blocking query is telemetry, not a synthetic history row.
            execution_state = self.robot.read_state()
            if (
                execution_state is None
                or execution_state.arm is None
                or execution_state.hand is None
                or not sample_is_fresh(
                    execution_state.arm["timestamp_ns"][0], self.runtime.arm.feedback_max_age_s
                )
                or not sample_is_fresh(
                    execution_state.hand["timestamp_ns"][0], self.runtime.hand.feedback_max_age_s
                )
            ):
                self._finish_episode("required_observation_stale")
                return
        action = self.action_queue.popleft()
        decoded = self.realizer.realize(
            policy_action_intent(action, self.policy_info.action_mode),
            execution_state.arm["qpos"][0],
            self.previous_arm,
        )
        arm, prepared_hand = decoded.arm_qpos, decoded.hand_qpos
        if arm is None:
            self.action_queue.clear()
            kind = decoded.ik_result.failure_kind
            if kind == IKFailureKind.INVALID_OUTPUT:
                raise RuntimeError(f"online IK technical failure: {decoded.ik_result.reason}")
            if self.recorder:
                self.recorder.add_frame(
                    build_episode_frame(policy_row),
                    step_timestamp_ns=time.monotonic_ns(),
                )
            return
        prepared_arm = arm
        if self.recorder is not None:
            self.recorder.check_error()
        command = RobotCommand(self.run_id, prepared_arm, prepared_hand)
        result = None
        dispatch_error = None
        stop_error = None
        if self.execute:
            try:
                result = self.robot.send_action(command)
            except DispatchError as exc:
                result, dispatch_error = exc.result, exc
                run_end_reason = (
                    RunEndReason.HARDWARE_FAULT
                    if not exc.revoked
                    else RunEndReason.TIMEOUT
                    if not self._within_budget()
                    else RunEndReason.EXECUTOR_BOUNDARY
                )
                revoke_motion_if_run_id(self.shared, self.run_id, reason=run_end_reason)
                try:
                    self.robot.stop()
                except Exception as stop_exc:
                    stop_error = stop_exc
                    logger.exception("stop after dispatch failure also failed")
        if result is not None and result.continued and dispatch_error is None:
            self.previous_arm = prepared_arm
        elif not self.execute:
            self.previous_arm = prepared_arm
        if self.recorder:
            self.recorder.add_frame(
                build_episode_frame(policy_row, command, result),
                step_timestamp_ns=result.timestamp_ns if result else time.monotonic_ns(),
            )
        if dispatch_error is not None:
            try:
                reason = (
                    "dispatch_failure"
                    if not dispatch_error.revoked
                    else "timeout"
                    if not self._within_budget()
                    else "motion_revoked"
                )
                self._finish_episode(
                    f"{reason}:stop_failure" if stop_error is not None else reason,
                    run_end_reason=run_end_reason,
                    stop_motion=False,
                )
            except Exception:
                logger.exception("dispatch failure finalization also failed")
                if not dispatch_error.revoked:
                    raise dispatch_error
                raise
            if not dispatch_error.revoked:
                raise dispatch_error
            if stop_error is not None:
                raise stop_error
        elif not self._within_budget():
            self._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)

    def run(self):
        failure = None
        failure_reason = RunEndReason.RUNTIME_SHUTDOWN
        try:
            while self.shared.is_running.value and not self.shared.quit_requested.value:
                if self.poll_operator is not None:
                    self.poll_operator()
                if self.shared.estop_request.value or self.shared.error_state.value:
                    break
                if self.run_id is None:
                    self.robot.service_idle()
                self.step()
                remaining = (self.next_step_ns - time.monotonic_ns()) / 1e9
                if self.run_id is None:
                    time.sleep(0.005)
                elif remaining > 0:
                    time.sleep(min(0.005, remaining))
        except KeyboardInterrupt:
            self.shared.estop_request.value = True
            raise
        except Exception as exc:
            failure = exc
            failure_reason = (
                RunEndReason.RECORDING_FAILURE
                if isinstance(exc, RecordingError)
                else RunEndReason.HARDWARE_FAULT
                if isinstance(exc, DispatchError)
                else RunEndReason.POLICY_FAILURE
            )
            self.shared.error_state.value = True
            self.shared.is_running.value = False
        finally:
            if self.run_id is not None:
                revoke_motion_if_run_id(
                    self.shared,
                    self.run_id,
                    reason=failure_reason,
                )
            self.robot.before_send = None
            try:
                reason = RunEndReason(int(self.shared.run_ended_reason.value))
                normal_stop = (
                    failure is None
                    and self.shared.is_running.value
                    and not self.shared.error_state.value
                    and not self.shared.estop_request.value
                    and reason in (RunEndReason.OPERATOR, RunEndReason.QUIT)
                )
                self._finish_episode(
                    reason.name.lower()
                    if normal_stop or reason in (RunEndReason.ESTOP, RunEndReason.HARDWARE_FAULT)
                    else failure_reason.name.lower()
                    if failure is not None
                    else "shutdown",
                    run_end_reason=failure_reason,
                )
                if self.recorder is not None:
                    if self.recorder.is_recording:
                        self.recorder.save_episode(reason="interrupted")
            except Exception as exc:
                if failure is None:
                    failure = exc
                logger.exception("policy recording cleanup failed")
            finally:
                # Retry unresolved stops, including motion outside a policy episode.
                try:
                    if self.robot._motion_active or self.robot._hand_stop_pending:
                        self.robot.stop()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    logger.exception("policy stop failed")
                if self.recorder is not None:
                    try:
                        self.recorder.close()
                    except Exception as exc:
                        if failure is None:
                            failure = exc
                        logger.exception("policy recorder close failed")
        if failure is not None:
            raise failure
