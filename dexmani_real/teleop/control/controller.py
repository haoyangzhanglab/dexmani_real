"""Current-row teleop mapping, absolute target publication and raw recording."""

import numpy as np

from dexmani_real.planning import Pose
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.robot.projection import project_hand_command
from dexmani_real.teleop.control.action_proposal import compute_target_eef_pose
from dexmani_real.teleop.control.hand_retargeting import (
    HandRetargetObservationCache,
    compute_hand_command,
    reset_hand_retargeter,
)


class TeleopController:
    def __init__(self, planner, arm_mapper, runtime, hand_retargeter):
        self.planner, self.arm_mapper, self.runtime = (
            planner,
            arm_mapper,
            runtime,
        )
        self.hand_retargeter = hand_retargeter
        self.arm_fk = make_arm_fk()
        self.hand_observation_cache = HandRetargetObservationCache()
        self.previous_arm_command = None
        if not runtime.policy.hand_enabled:
            self.planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))
        self.clear_reference()

    def clear_reference(self):
        self.arm_mapper.clear()
        self.smoothed_eef_position = self.smoothed_eef_quaternion = None
        self.hand_observation_cache.reset()
        reset_hand_retargeter(self.hand_retargeter)

    def reset_reference(self, row):
        self.clear_reference()
        self.previous_arm_command = row.arm["qpos"][0].copy()
        pos, rot = self.arm_fk.compute(self.previous_arm_command)
        self.arm_mapper.reset(
            wrist_pos=row.vr["wrist_pos"],
            wrist_quat_wxyz=row.vr["wrist_quat_wxyz"],
            eef_pos=pos,
            eef_quat_wxyz=rot6d_to_quat_wxyz(rot),
        )
        if row.hand is not None:
            reset_hand_retargeter(self.hand_retargeter, row.hand["qpos"][0])
        return self.arm_mapper.is_ready()

    def compute_command(self, row, run_id):
        cfg = self.runtime
        mapped = self.arm_mapper.map(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        if mapped is None:
            return None, False, None
        target = compute_target_eef_pose(
            mapped["pos"],
            mapped["quat_wxyz"],
            previous_position_world_m=self.smoothed_eef_position,
            previous_quat_world_wxyz=self.smoothed_eef_quaternion,
            workspace_bounds_world_m=cfg.policy.workspace.as_array(),
            ema_alpha_position=cfg.policy.ema.alpha_pos,
            ema_alpha_rotation=cfg.policy.ema.alpha_rot,
        )
        intent = np.concatenate(
            (target.position_world_m, quat_wxyz_to_rot6d(target.quat_world_wxyz))
        )
        hand = None
        if cfg.policy.hand_enabled:
            try:
                proposal = compute_hand_command(
                    self.hand_retargeter, row.vr, self.hand_observation_cache
                )
                if proposal is None:
                    return None, False, intent
                hand = project_hand_command(
                    proposal,
                    qpos_min_rad=cfg.hand.qpos_min_rad,
                    qpos_max_rad=cfg.hand.qpos_max_rad,
                )
            except (ValueError, RuntimeError):
                return None, False, intent
            self.planner.set_hand_qpos(hand)
        solution = self.planner.solve_online_ik(
            Pose(p=target.position_world_m, q=target.quat_world_wxyz),
            row.arm["qpos"][0],
            self.previous_arm_command,
        )
        if solution.failure_kind == IKFailureKind.INVALID_OUTPUT:
            raise RuntimeError(f"online IK technical failure: {solution.reason}")
        if not solution.success:
            return None, False, intent
        arm = solution.qpos
        return RobotCommand(run_id, arm, hand), True, intent


def execute_control_step(controller, shared, row, recorder=None):
    epoch = int(shared.run_id.value)
    target, control_ok, intent = controller.compute_command(row, epoch)
    stamp = publish_command(shared, target) if target is not None else 0
    if int(shared.run_id.value) != epoch or (target is not None and not stamp):
        if recorder is not None:
            recorder.note_publication_rejected()
        return control_ok
    if stamp:
        controller.previous_arm_command = target.arm_qpos.copy()
        controller.smoothed_eef_position = intent[:3].copy()
        controller.smoothed_eef_quaternion = rot6d_to_quat_wxyz(intent[3:])
    if recorder is not None and recorder.is_recording:
        recorder.add_frame(
            build_episode_frame(
                row, target, frame_valid=control_ok and target is not None and bool(stamp)
            ),
            observation_timestamp_ns=row.observation_timestamp_ns,
            publication_timestamp_ns=stamp,
        )
    return control_ok
