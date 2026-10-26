"""映射当前遥操作观测，并在正常提交后更新绝对目标参考。"""

import numpy as np

from dexmani_real.control.action_mapping import ActionIntent
from dexmani_real.geometry.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.teleop.vr_mapping import ema_smooth_pose


class TeleopController:
    def __init__(
        self, action_mapper, arm_mapper, hand_retargeter, *, ema_alpha_pos, ema_alpha_rot,
    ):
        self.action_mapper, self.arm_mapper = action_mapper, arm_mapper
        self.ema_alpha_pos, self.ema_alpha_rot = ema_alpha_pos, ema_alpha_rot
        self.hand_retargeter = hand_retargeter
        self.clear_reference()

    def clear_reference(self):
        self.previous_arm_command = None
        self.arm_mapper.clear()
        self.smoothed_eef_pose = None
        self.hand_observation_id = self.hand_target = None
        if self.hand_retargeter is not None:
            self.hand_retargeter.reset()

    def reset_reference(self, row) -> None:
        self.clear_reference()
        try:
            arm_qpos = row.arm["qpos"][0].copy()
            pos, rot = self.action_mapper.target_checker.target_fk.compute(arm_qpos)
            quat = rot6d_to_quat_wxyz(rot)
            self.arm_mapper.reset(
                wrist_pos=row.vr["wrist_pos"],
                wrist_quat_wxyz=row.vr["wrist_quat_wxyz"],
                eef_pos=pos,
                eef_quat_wxyz=quat,
            )
            if self.hand_retargeter is not None and row.hand is not None:
                self.hand_retargeter.reset(initial_qpos=row.hand["qpos"][0])
            # The first EMA step starts at the same measured FK pose as the VR anchor.
            self.previous_arm_command = arm_qpos
            self.smoothed_eef_pose = Pose(pos.copy(), quat.copy())
        except BaseException:
            self.clear_reference()
            raise

    def compute_target(self, row):
        mapped = self.arm_mapper.map(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        target = ema_smooth_pose(
            mapped,
            self.smoothed_eef_pose,
            self.ema_alpha_pos,
            self.ema_alpha_rot,
        )
        hand = None
        if self.hand_retargeter is not None:
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
        return self.action_mapper.map(
            ActionIntent("eef", target, hand), row.arm["qpos"][0], self.previous_arm_command
        )

    def commit_submitted_target(self, action_result):
        self.previous_arm_command = action_result.arm_qpos.copy()
        self.smoothed_eef_pose = action_result.eef_pose.copy()
