"""Synchronous policy inference, local control-row history and local action chunks."""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.deployment.action import (
    decode_policy_action,
    make_action_planner,
    physical_action_dim,
)
from dexmani_real.deployment.observation import build_fingertip_runtime, build_policy_observation
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.recording.client import RecorderClient
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.robot.projection import project_arm_command
from dexmani_real.runtime.observation import ObservationHistory, read_observation
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    StopRequest,
    _begin_requested_motion_locked,
    revoke_motion_if_run_id,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class RolloutStats:
    """Session-wide diagnostics; interval anchoring resets at each episode."""

    inference_ms: list[float] = field(default_factory=list)
    action_step_intervals_ms: list[float] = field(default_factory=list)
    previous_step_ns: int | None = None
    arm_clip_count: int = 0
    workspace_clip_count: int = 0
    hand_clip_count: int = 0
    max_arm_clip_rad: float = 0.0
    max_workspace_clip_m: float = 0.0
    max_hand_clip_rad: float = 0.0
    ik_failure_counts: dict[str, int] = field(default_factory=dict)
    publications: int = 0


class PolicyRunner:
    def __init__(
        self,
        shared,
        runtime,
        policy_spec,
        *,
        model_runtime,
        fingertip_runtime,
        execute,
        max_running_s,
        num_episodes=1,
        recording_config=None,
    ):
        self.shared = shared
        self.runtime = runtime
        self.policy_spec = policy_spec
        self.model = model_runtime
        self.fingertip_runtime = fingertip_runtime
        self.execute = execute
        self.max_running_s = max_running_s
        self.num_episodes = num_episodes
        self.recording_config = recording_config
        self.recorder = (
            RecorderClient(shared, control_hz=1.0 / policy_spec.control_dt_s)
            if recording_config is not None
            else None
        )
        self.history = ObservationHistory(policy_spec.n_obs_steps, policy_spec.control_dt_s)
        self.action_queue = deque()
        self.run_id = None
        self.started_ns = 0
        self.next_step_ns = 0
        self.previous_arm = None
        self.completed = 0
        self.stats = RolloutStats()
        self.planner = make_action_planner(policy_spec.action_mode, runtime)
        fields = {f.name for f in self.policy_spec.observation_fields}
        requires_rgb = "rgb" in fields
        self.requires_cloud = "point_cloud" in fields
        self.requires_camera_payload = requires_rgb or self.recorder is not None
        self.requires_rgb_cloud_identity = requires_rgb and self.requires_cloud

    def _read_observation(self):
        return read_observation(
            self.shared,
            self.runtime,
            require_hand=True,
            require_camera=self.requires_camera_payload,
            require_pointcloud=self.requires_cloud,
            require_rgb_cloud_identity=self.requires_rgb_cloud_identity,
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
        self, reason, abnormal=False, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY
    ):
        if self.run_id is None:
            return
        revoke_motion_if_run_id(self.shared, self.run_id, reason=run_end_reason)
        self.shared.physical_home_completed.value = False
        self.action_queue.clear()
        self.history.clear()
        self.previous_arm = None
        self.completed += 1
        self.run_id = None
        self.shared.stop_request.value = int(StopRequest.NONE)
        try:
            if self.recorder is not None:
                if abnormal:
                    self.recorder.invalidate_episode()
                self.recorder.stop_episode(save=True, reason=reason)
                self.recorder.join_stop()
        finally:
            logger.info("policy episode %d ended: %s", self.completed, reason)
            if self.completed >= self.num_episodes:
                self.shared.quit_requested.value = True

    def _begin_episode(self):
        row = self._read_observation()
        if row is None:
            return
        if self.execute and (
            not self.shared.physical_home_completed.value
            or np.max(np.abs(row.arm["qpos"][0] - self.runtime.arm.home_qpos))
            > self.runtime.arm.homing.convergence_rad
        ):
            self.shared.start_request.value = False
            logger.warning("policy B requires home at the training start pose")
            return
        preparation_epoch = int(self.shared.run_id.value)
        if self.recorder is not None:
            cfg = self.recording_config
            if not self.recorder.start_episode(
                task_label=cfg.task_label,
                episode_name=f"episode_{self.completed + 1:03d}",
            ):
                raise RuntimeError("policy recorder refused START")
        with self.shared.motion_lock:
            epoch = (
                _begin_requested_motion_locked(self.shared)
                if preparation_epoch == int(self.shared.run_id.value)
                and not self.shared.quit_requested.value
                else None
            )
        if epoch is None:
            if self.recorder is not None:
                self.recorder.stop_episode(save=False, reason="start_cancelled")
                self.recorder.join_stop()
            return
        self.run_id, self.started_ns = epoch
        self.stats.previous_step_ns = None
        self.shared.physical_home_completed.value = False
        self.history.clear()
        self.action_queue.clear()
        self.next_step_ns = 0
        self.model.reset_episode()

    def step(self):
        if self.run_id is None:
            if self.shared.stop_request.value:
                self.shared.stop_request.value = int(StopRequest.NONE)
            if self.shared.start_request.value:
                self._begin_episode()
            return
        if not self._has_motion_authority() or self.shared.quit_requested.value:
            reason = RunEndReason(int(self.shared.run_ended_reason.value))
            self._finish_episode(
                reason.name.lower(),
                abnormal=bool(
                    not self.shared.is_running.value
                    or self.shared.error_state.value
                    or self.shared.estop_request.value
                    or reason not in (RunEndReason.OPERATOR, RunEndReason.QUIT)
                ),
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
        row = self._read_observation()
        if row is None:
            self._finish_episode("required_observation_stale", abnormal=True)
            return
        self.history.append(row)
        if not self.action_queue:
            observation = build_policy_observation(
                self.history.padded(), self.policy_spec, fingertip_runtime=self.fingertip_runtime
            )
            if observation is None:
                self._finish_episode("required_tactile_unavailable", abnormal=True)
                return
            epoch = self.run_id
            start = time.monotonic_ns()
            prediction = self.model.predict(observation)
            self.stats.inference_ms.append((time.monotonic_ns() - start) / 1e6)
            if not self._has_motion_authority() or epoch != int(self.shared.run_id.value):
                self.action_queue.clear()
                return
            if self.max_running_s is not None and time.monotonic_ns() - self.started_ns >= int(
                self.max_running_s * 1e9
            ):
                self._finish_episode("timeout", run_end_reason=RunEndReason.TIMEOUT)
                return
            prediction = np.asarray(prediction, dtype=np.float64)
            if (
                prediction.shape
                != (
                    self.policy_spec.n_action_steps,
                    physical_action_dim(self.policy_spec.action_mode),
                )
                or not np.isfinite(prediction).all()
            ):
                raise ValueError("policy prediction violates shape/finite contract")
            self.action_queue.extend(prediction)
            # Execution feedback after a blocking query is telemetry, not a synthetic history row.
            row = self._read_observation()
            if row is None:
                self._finish_episode("required_observation_stale", abnormal=True)
                return
        action = self.action_queue.popleft()
        decoded = decode_policy_action(
            action,
            self.policy_spec.action_mode,
            row.arm["qpos"][0],
            previous_arm_command_qpos=self.previous_arm,
            planner=self.planner,
            workspace=self.runtime.policy.workspace.as_array(),
            hand_qpos_min_rad=self.runtime.hand.qpos_min_rad,
            hand_qpos_max_rad=self.runtime.hand.qpos_max_rad,
        )
        arm, prepared_hand = decoded.arm_qpos, decoded.hand_qpos
        self.stats.workspace_clip_count += int(decoded.workspace_clip_m > 1e-9)
        self.stats.max_workspace_clip_m = max(
            self.stats.max_workspace_clip_m, decoded.workspace_clip_m
        )
        self.stats.hand_clip_count += int(decoded.hand_clip_rad > 1e-9)
        self.stats.max_hand_clip_rad = max(self.stats.max_hand_clip_rad, decoded.hand_clip_rad)
        if arm is None:
            self.action_queue.clear()
            kind = decoded.ik_result.failure_kind
            self.stats.ik_failure_counts[kind.value] = (
                self.stats.ik_failure_counts.get(kind.value, 0) + 1
            )
            if kind == IKFailureKind.INVALID_OUTPUT:
                raise RuntimeError(f"online IK technical failure: {decoded.ik_result.reason}")
            if self.recorder:
                self.recorder.add_frame(
                    build_episode_frame(row, frame_valid=False),
                    observation_timestamp_ns=row.observation_timestamp_ns,
                    publication_timestamp_ns=0,
                )
            self.next_step_ns = time.monotonic_ns() + int(self.policy_spec.control_dt_s * 1e9)
            return
        if self.policy_spec.action_mode == "joint":
            prepared_arm = project_arm_command(
                arm,
                row.arm["qpos"][0],
                joint_lower_rad=self.runtime.arm.joint_limit_lower,
                joint_upper_rad=self.runtime.arm.joint_limit_upper,
            )
            arm_change = prepared_arm - arm
            equivalent = (
                np.asarray(self.runtime.arm.joint_limit_upper)
                - np.asarray(self.runtime.arm.joint_limit_lower)
                >= 2 * np.pi
            )
            arm_change[equivalent] = (arm_change[equivalent] + np.pi) % (2 * np.pi) - np.pi
            arm_clip = float(np.max(np.abs(arm_change)))
            self.stats.arm_clip_count += int(arm_clip > 1e-9)
            self.stats.max_arm_clip_rad = max(self.stats.max_arm_clip_rad, arm_clip)
        else:
            prepared_arm = arm
        command = RobotCommand(self.run_id, prepared_arm, prepared_hand)
        stamp = publish_command(self.shared, command) if self.execute else time.monotonic_ns()
        if not stamp:
            if self.recorder is not None:
                self.recorder.note_publication_rejected()
            return
        if self.stats.previous_step_ns is not None:
            self.stats.action_step_intervals_ms.append((stamp - self.stats.previous_step_ns) / 1e6)
        self.stats.previous_step_ns = stamp
        self.previous_arm = prepared_arm
        self.stats.publications += 1
        self.next_step_ns = stamp + int(self.policy_spec.control_dt_s * 1e9)
        if self.recorder:
            self.recorder.add_frame(
                build_episode_frame(row, command, frame_valid=True),
                observation_timestamp_ns=row.observation_timestamp_ns,
                publication_timestamp_ns=stamp,
            )

    def run(self):
        failure = None
        try:
            while self.shared.is_running.value and not self.shared.quit_requested.value:
                self.step()
                time.sleep(0.001)
        except Exception as exc:
            failure = exc
            self.shared.is_running.value = False
        finally:
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
                    reason.name.lower() if normal_stop else "shutdown",
                    abnormal=not normal_stop,
                    run_end_reason=RunEndReason.POLICY_FAILURE
                    if failure is not None
                    else RunEndReason.RUNTIME_SHUTDOWN,
                )
                if self.recorder is not None:
                    if self.recorder.is_recording:
                        self.recorder.invalidate_episode()
                        self.recorder.stop_episode(save=True, reason="interrupted")
                    if self.recorder.stop_pending:
                        self.recorder.join_stop()
            except Exception as exc:
                if failure is None:
                    failure = exc
                logger.exception("policy recording cleanup failed")
            self._log_summary()
        if failure is not None:
            raise failure

    def _log_summary(self):
        def statistics(samples):
            if not samples:
                return "unavailable"
            return "mean=%.2f p95=%.2f max=%.2f" % (
                np.mean(samples),
                np.percentile(samples, 95),
                np.max(samples),
            )

        mean_interval_ms = (
            float(np.mean(self.stats.action_step_intervals_ms))
            if self.stats.action_step_intervals_ms
            else 0.0
        )
        effective_hz = f"{1000 / mean_interval_ms:.2f}" if mean_interval_ms > 0 else "unavailable"
        logger.info(
            "policy summary: execute=%s steps=%d arm_clipped=%d max_arm_clip_rad=%.5f "
            "workspace_clipped=%d max_workspace_clip_m=%.5f hand_clipped=%d max_hand_clip_rad=%.5f "
            "ik_failures=%s configured_action_hz=%.2f "
            "inference_ms[n=%d %s] action_step_interval_ms[n=%d %s] effective_action_step_hz=%s",
            self.execute,
            self.stats.publications,
            self.stats.arm_clip_count,
            self.stats.max_arm_clip_rad,
            self.stats.workspace_clip_count,
            self.stats.max_workspace_clip_m,
            self.stats.hand_clip_count,
            self.stats.max_hand_clip_rad,
            self.stats.ik_failure_counts,
            1 / self.policy_spec.control_dt_s,
            len(self.stats.inference_ms),
            statistics(self.stats.inference_ms),
            len(self.stats.action_step_intervals_ms),
            statistics(self.stats.action_step_intervals_ms),
            effective_hz,
        )


def run_policy_worker(
    shared,
    runtime,
    config,
    execute,
    max_running_s=None,
    num_episodes=1,
    recording_config=None,
    fingertip_config=None,
):
    from dexmani_policy.deployment import load_experiment

    model = None
    failure = None
    try:
        model = load_experiment(
            config.experiment,
            device=config.device,
            seed=config.seed,
            artifact=config.artifact,
            inference_steps=config.inference_steps,
        )
        if model.spec != config.spec:
            raise ValueError("PolicySpec changed between inspect and load")
        fingertip_runtime = build_fingertip_runtime(config.spec, fingertip_config)
        model.warmup(samples=5)
        runner = PolicyRunner(
            shared,
            runtime,
            config.spec,
            model_runtime=model,
            fingertip_runtime=fingertip_runtime,
            execute=execute,
            max_running_s=max_running_s,
            num_episodes=num_episodes,
            recording_config=recording_config,
        )
        shared.policy_ready.set()
        runner.run()
    except Exception as exc:
        failure = exc
        logger.exception("policy worker failed")
        raise
    finally:
        shared.policy_ready.clear()
        if model is not None:
            try:
                model.close()
            except Exception:
                if failure is None:
                    raise
                logger.exception("policy model cleanup failed")
