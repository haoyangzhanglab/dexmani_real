"""Synchronous policy inference, local control-row history and local action chunks."""

import time
from collections import deque

import numpy as np

from dexmani_real.deployment.observation import build_fingertip_runtime, build_policy_observation
from dexmani_real.planning import OnlineIKConfig, Pose, XArm7MotionPlanner
from dexmani_real.planning.kinematics.pose import rot6d_to_quat_wxyz
from dexmani_real.recording.client import RecorderClient
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.recording.storage.schema import FRAME_IK_FAIL
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.robot.projection import project_arm_command, project_hand_command
from dexmani_real.runtime.observation import ObservationHistory, read_observation
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    StopRequest,
    begin_requested_motion,
    revoke_motion_if_run_id,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def decode_policy_action(
    action,
    policy_spec,
    current_arm_qpos,
    *,
    previous_arm_command_qpos,
    planner,
    workspace,
    hand_qpos_min_rad,
    hand_qpos_max_rad,
):
    raw_hand = action[7:19] if policy_spec.action_key == "action" else action[9:21]
    hand = project_hand_command(
        raw_hand, qpos_min_rad=hand_qpos_min_rad, qpos_max_rad=hand_qpos_max_rad
    )
    if policy_spec.action_key == "action":
        return action[:7], hand, None
    planner.set_hand_qpos(hand)
    position = np.clip(action[:3], workspace[:, 0], workspace[:, 1])
    intent = np.concatenate((position, action[3:9]))
    result = planner.solve_teleop_ik(
        Pose(p=position, q=rot6d_to_quat_wxyz(action[3:9])),
        current_arm_qpos,
        current_arm_qpos if previous_arm_command_qpos is None else previous_arm_command_qpos,
    )
    return result.qpos if result.success else None, hand, intent


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
        self.shared, self.runtime, self.spec = (
            shared,
            runtime,
            policy_spec,
        )
        self.model, self.fk = model_runtime, fingertip_runtime
        self.execute, self.max_running_s, self.num_episodes = (
            execute,
            max_running_s,
            num_episodes,
        )
        self.recording_config = recording_config
        self.recorder = RecorderClient(shared) if recording_config is not None else None
        self.history = ObservationHistory(policy_spec.n_obs_steps, policy_spec.control_dt_s)
        self.actions = deque()
        self.run_id = None
        self.started_ns = 0
        self.next_step_ns = 0
        self.previous_arm = None
        self.completed = 0
        self.inference_ms = []
        self.action_step_intervals_ms = []
        self.previous_step_ns = None
        self.clip_count = 0
        self.max_clip_rad = 0.0
        self.publications = 0
        self.planner = (
            XArm7MotionPlanner.create_default(
                teleop_profile=OnlineIKConfig(
                    max_pose_error_pos_m=runtime.policy.ik_max_pose_error_pos_m,
                    max_pose_error_rot_rad=runtime.policy.ik_max_pose_error_rot_rad,
                )
            )
            if policy_spec.action_key == "action_ee"
            else None
        )
        fields = {f.name for f in self.spec.observation_fields}
        self.requires_rgb = "rgb" in fields
        self.requires_cloud = "point_cloud" in fields
        self.requires_recording_camera = self.recorder is not None
        self.requires_camera_payload = self.requires_rgb or self.requires_recording_camera
        self.requires_rgb_cloud_identity = self.requires_rgb and self.requires_cloud

    def _row(self):
        return read_observation(
            self.shared,
            self.runtime,
            require_hand=True,
            require_camera=self.requires_camera_payload,
            require_pointcloud=self.requires_cloud,
            require_rgb_cloud_identity=self.requires_rgb_cloud_identity,
        )

    def _live(self):
        return (
            self.run_id is not None
            and self.shared.is_running.value
            and not self.shared.error_state.value
            and not self.shared.estop_request.value
            and int(self.shared.run_id.value) == self.run_id
            and int(self.shared.safety_state.value) == int(SafetyState.RUNNING)
        )

    def _finish(self, reason, incomplete=False):
        if self.run_id is None:
            return
        revoke_motion_if_run_id(self.shared, self.run_id)
        self.shared.physical_home_completed.value = False
        self.actions.clear()
        self.history.clear()
        self.previous_arm = None
        if self.recorder is not None:
            if incomplete:
                self.recorder.technical_status = "invalid"
            self.recorder.stop_episode(save=True, reason=reason, retain_partial=incomplete)
            result = self.recorder.join_stop()
            if result.error:
                self.shared.workflow_failed.value = True
            self.shared.is_recording.value = False
        self.completed += 1
        self.run_id = None
        self.shared.stop_request.value = int(StopRequest.NONE)
        logger.info("policy episode %d ended: %s", self.completed, reason)
        if self.completed >= self.num_episodes:
            self.shared.quit_requested.value = True

    def _begin(self):
        row = self._row()
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
                operator=cfg.operator,
                episode_name=f"episode_{self.completed + 1:03d}",
            ):
                self.shared.workflow_failed.value = True
                return
            self.shared.is_recording.value = True
        with self.shared.motion_lock:
            epoch = (
                begin_requested_motion(self.shared)
                if preparation_epoch == int(self.shared.run_id.value)
                and not self.shared.quit_requested.value
                else None
            )
        if epoch is None:
            if self.recorder is not None:
                self.recorder.stop_episode(save=False, reason="start_cancelled")
                self.recorder.join_stop()
                self.shared.is_recording.value = False
            return
        self.run_id, self.started_ns = epoch
        self.previous_step_ns = None
        self.shared.physical_home_completed.value = False
        self.history.clear()
        self.actions.clear()
        self.next_step_ns = 0
        self.model.reset_episode()

    def step(self):
        if self.run_id is None:
            if self.shared.stop_request.value:
                self.shared.stop_request.value = int(StopRequest.NONE)
            if self.shared.start_request.value and not self.shared.workflow_failed.value:
                self._begin()
            return
        if not self._live() or self.shared.quit_requested.value:
            self._finish(
                RunEndReason(int(self.shared.run_ended_reason.value)).name.lower(),
                incomplete=bool(
                    self.shared.error_state.value
                    or self.shared.estop_request.value
                    or self.shared.workflow_failed.value
                ),
            )
            return
        if self.shared.workflow_failed.value:
            self._finish("workflow_failure", incomplete=True)
            return
        now = time.monotonic_ns()
        if self.max_running_s is not None and now - self.started_ns >= int(
            self.max_running_s * 1e9
        ):
            self._finish("timeout")
            return
        if now < self.next_step_ns:
            return
        row = self._row()
        if row is None:
            self._finish("required_observation_stale", incomplete=True)
            return
        self.history.append(row)
        if not self.actions:
            observation = build_policy_observation(
                self.history.padded(), self.spec, fingertip_runtime=self.fk
            )
            if observation is None:
                self._finish("required_tactile_unavailable", incomplete=True)
                return
            epoch = self.run_id
            start = time.monotonic_ns()
            prediction = self.model.predict(observation)
            self.inference_ms.append((time.monotonic_ns() - start) / 1e6)
            if not self._live() or epoch != int(self.shared.run_id.value):
                self.actions.clear()
                return
            if self.max_running_s is not None and time.monotonic_ns() - self.started_ns >= int(
                self.max_running_s * 1e9
            ):
                self._finish("timeout")
                return
            prediction = np.asarray(prediction, dtype=np.float64)
            if (
                prediction.shape
                != (
                    self.spec.n_action_steps,
                    self.spec.control_action_dim,
                )
                or not np.isfinite(prediction).all()
            ):
                raise ValueError("policy prediction violates shape/finite contract")
            self.actions.extend(prediction)
            # Execution feedback after a blocking query is telemetry, not a synthetic history row.
            row = self._row()
            if row is None:
                self._finish("required_observation_stale", incomplete=True)
                return
        action = self.actions.popleft()
        arm, prepared_hand, intent = decode_policy_action(
            action,
            self.spec,
            row.arm["qpos"][0],
            previous_arm_command_qpos=self.previous_arm,
            planner=self.planner,
            workspace=self.runtime.policy.workspace.as_array(),
            hand_qpos_min_rad=self.runtime.hand.qpos_min_rad,
            hand_qpos_max_rad=self.runtime.hand.qpos_max_rad,
        )
        if arm is None:
            self.actions.clear()
            if self.recorder:
                self.recorder.add_frame(
                    build_episode_frame(row, frame_status=FRAME_IK_FAIL, arm_eef_intent=intent)
                )
            self.next_step_ns = time.monotonic_ns() + int(self.spec.control_dt_s * 1e9)
            return
        prepared_arm = project_arm_command(
            arm,
            row.arm["qpos"][0],
            joint_lower_rad=self.runtime.arm.joint_limit_lower,
            joint_upper_rad=self.runtime.arm.joint_limit_upper,
        )
        raw_hand = action[7:19] if self.spec.action_key == "action" else action[9:21]
        arm_change = prepared_arm - arm
        equivalent = (
            np.asarray(self.runtime.arm.joint_limit_upper)
            - np.asarray(self.runtime.arm.joint_limit_lower)
            >= 2 * np.pi
        )
        arm_change[equivalent] = (arm_change[equivalent] + np.pi) % (2 * np.pi) - np.pi
        clip = max(
            float(np.max(np.abs(arm_change))), float(np.max(np.abs(prepared_hand - raw_hand)))
        )
        self.clip_count += int(clip > 1e-9)
        self.max_clip_rad = max(self.max_clip_rad, clip)
        command = RobotCommand(self.run_id, prepared_arm, prepared_hand)
        stamp = publish_command(self.shared, command) if self.execute else time.monotonic_ns()
        if not stamp:
            return
        if self.previous_step_ns is not None:
            self.action_step_intervals_ms.append((stamp - self.previous_step_ns) / 1e6)
        self.previous_step_ns = stamp
        self.previous_arm = prepared_arm
        self.publications += 1
        self.next_step_ns = stamp + int(self.spec.control_dt_s * 1e9)
        if self.recorder:
            self.recorder.add_frame(
                build_episode_frame(row, command, action_timestamp_ns=stamp, arm_eef_intent=intent)
            )

    def run(self):
        try:
            while self.shared.is_running.value and not self.shared.quit_requested.value:
                self.step()
                time.sleep(0.001)
        except Exception:
            self.shared.workflow_failed.value = True
            raise
        finally:
            self._finish(
                "shutdown",
                incomplete=bool(
                    self.shared.workflow_failed.value
                    or self.shared.error_state.value
                    or self.shared.estop_request.value
                ),
            )
            self._log_summary()

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
            float(np.mean(self.action_step_intervals_ms)) if self.action_step_intervals_ms else 0.0
        )
        effective_hz = f"{1000 / mean_interval_ms:.2f}" if mean_interval_ms > 0 else "unavailable"
        logger.info(
            "policy summary: execute=%s steps=%d clipped=%d max_clip_rad=%.5f configured_action_hz=%.2f "
            "inference_ms[n=%d %s] action_step_interval_ms[n=%d %s] effective_action_step_hz=%s",
            self.execute,
            self.publications,
            self.clip_count,
            self.max_clip_rad,
            1 / self.spec.control_dt_s,
            len(self.inference_ms),
            statistics(self.inference_ms),
            len(self.action_step_intervals_ms),
            statistics(self.action_step_intervals_ms),
            effective_hz,
        )


def policy_runner_loop(
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
    runner = None
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
        fk = build_fingertip_runtime(config.spec, fingertip_config)
        model.warmup(samples=5)
        runner = PolicyRunner(
            shared,
            runtime,
            config.spec,
            model_runtime=model,
            fingertip_runtime=fk,
            execute=execute,
            max_running_s=max_running_s,
            num_episodes=num_episodes,
            recording_config=recording_config,
        )
        shared.policy_ready.set()
        runner.run()
    except Exception:
        shared.workflow_failed.value = True
        if runner is not None:
            runner._finish("policy_failure", incomplete=True)
        logger.exception("policy worker failed")
        raise
    finally:
        if model is not None:
            model.close()
