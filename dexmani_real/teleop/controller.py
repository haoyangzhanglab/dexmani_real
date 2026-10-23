"""Current-row teleop mapping and commit of dispatched absolute targets."""

import numpy as np

from dexmani_real.planning.action import ActionIntent
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.planning.kinematics.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.teleop.vr_mapping import ema_smooth_pose


class TeleopController:
    def __init__(self, action_mapper, arm_mapper, runtime, hand_retargeter):
        self.action_mapper, self.arm_mapper, self.runtime = (
            action_mapper,
            arm_mapper,
            runtime,
        )
        self.hand_retargeter = hand_retargeter
        self.arm_fk = make_arm_fk()
        self.clear_reference()

    def clear_reference(self):
        self.previous_arm_command = None
        self.arm_mapper.clear()
        self.smoothed_eef_pose = None
        self.hand_observation_id = self.hand_target = None
        if self.hand_retargeter is not None:
            self.hand_retargeter.reset()

    def reset_reference(self, row):
        self.clear_reference()
        prepared = False
        try:
            arm_qpos = row.arm["qpos"][0].copy()
            pos, rot = self.arm_fk.compute(arm_qpos)
            quat = rot6d_to_quat_wxyz(rot)
            self.arm_mapper.reset(
                wrist_pos=row.vr["wrist_pos"],
                wrist_quat_wxyz=row.vr["wrist_quat_wxyz"],
                eef_pos=pos,
                eef_quat_wxyz=quat,
            )
            if not self.arm_mapper.is_ready():
                return False
            if self.hand_retargeter is not None and row.hand is not None:
                self.hand_retargeter.reset(initial_qpos=row.hand["qpos"][0])
            # The first EMA step starts at the same measured FK pose as the VR anchor.
            self.previous_arm_command = arm_qpos
            self.smoothed_eef_pose = Pose(pos.copy(), quat.copy())
            prepared = True
            return True
        finally:
            if not prepared:
                self.clear_reference()

    def compute_target(self, row):
        cfg = self.runtime
        mapped = self.arm_mapper.map(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        if mapped is None:
            return None
        target = ema_smooth_pose(
            mapped,
            self.smoothed_eef_pose,
            cfg.control.ema.alpha_pos,
            cfg.control.ema.alpha_rot,
        )
        hand = None
        if cfg.control.hand_enabled:
            observation_id = row.vr["ring_sequence"]
            if self.hand_observation_id != observation_id:
                # Claim each sample before solving; a failed solve is not retried on later ticks.
                self.hand_observation_id = observation_id
                self.hand_target = None
                proposal = self.hand_retargeter.retarget(row.vr["landmarks"])
                if proposal is not None:
                    self.hand_target = np.asarray(proposal, dtype=np.float64).copy()
            if self.hand_target is None:
                return None
            hand = self.hand_target
        mapped = self.action_mapper.map(
            ActionIntent("eef", target, hand), row.arm["qpos"][0], self.previous_arm_command
        )
        solution = mapped.ik_result
        if solution is not None and solution.failure_kind == IKFailureKind.INVALID_OUTPUT:
            raise RuntimeError(f"online IK technical failure: {solution.reason}")
        return mapped

    def commit_dispatched_target(self, action_result):
        self.previous_arm_command = action_result.arm_qpos.copy()
        self.smoothed_eef_pose = action_result.eef_pose.copy()
