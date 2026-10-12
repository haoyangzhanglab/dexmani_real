"""Current-row teleop mapping and commit of dispatched absolute targets."""


import numpy as np

from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.robot.action import ActionIntent
from dexmani_real.teleop.control.action_proposal import compute_target_eef_pose
from dexmani_real.teleop.control.hand_retargeting import (
    HandRetargetObservationCache,
    compute_hand_command,
    reset_hand_retargeter,
)


class TeleopController:
    def __init__(self, realizer, arm_mapper, runtime, hand_retargeter):
        self.realizer, self.arm_mapper, self.runtime = (
            realizer,
            arm_mapper,
            runtime,
        )
        self.hand_retargeter = hand_retargeter
        self.arm_fk = make_arm_fk()
        self.hand_observation_cache = HandRetargetObservationCache()
        self.clear_reference()

    def clear_reference(self):
        self.previous_arm_command = None
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

    def compute_target(self, row):
        cfg = self.runtime
        mapped = self.arm_mapper.map(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        if mapped is None:
            return None
        target = compute_target_eef_pose(
            mapped["pos"],
            mapped["quat_wxyz"],
            previous_position_xarm_base_m=self.smoothed_eef_position,
            previous_quat_xarm_base_wxyz=self.smoothed_eef_quaternion,
            ema_alpha_position=cfg.policy.ema.alpha_pos,
            ema_alpha_rotation=cfg.policy.ema.alpha_rot,
        )
        intent = np.concatenate(
            (target.position_xarm_base_m, quat_wxyz_to_rot6d(target.quat_xarm_base_wxyz))
        )
        hand = None
        if cfg.policy.hand_enabled:
            proposal = compute_hand_command(
                self.hand_retargeter, row.vr, self.hand_observation_cache
            )
            if proposal is None:
                return None
            hand = proposal
        realized = self.realizer.realize(
            ActionIntent("eef", intent, hand), row.arm["qpos"][0], self.previous_arm_command
        )
        solution = realized.ik_result
        if solution is not None and solution.failure_kind == IKFailureKind.INVALID_OUTPUT:
            raise RuntimeError(f"online IK technical failure: {solution.reason}")
        return realized

    def commit_dispatched_target(self, realization):
        self.previous_arm_command = realization.arm_qpos.copy()
        self.smoothed_eef_position = realization.eef_pose[:3].copy()
        self.smoothed_eef_quaternion = rot6d_to_quat_wxyz(realization.eef_pose[3:])
