"""VR 锚点映射、手部重定向与提交后的位姿平滑。"""

from scipy.spatial.transform import Rotation

from dexmani_real.geometry.pose import (
    WXYZ_TO_XYZW,
    XYZW_TO_WXYZ,
    Pose,
    rot6d_to_quat_wxyz,
)


class VRController:
    def __init__(self, action_mapper, hand_retargeter, config, vr_to_robot_rot):
        self.action_mapper = action_mapper
        self.hand_retargeter = hand_retargeter
        self.config = config
        self.rotation = Rotation.from_matrix(vr_to_robot_rot)
        self.clear()

    def clear(self):
        self.anchor = None
        self.previous_arm_command = None
        self.previous_eef_target = None
        self.hand_observation_id = self.hand_target = None
        self.target_timestamp_ns = None
        self.rejection_reason = None

    def reset(self, row):
        self.clear()
        arm_qpos = row.arm["qpos"][0].copy()
        position, rot6d = self.action_mapper.constraints.target_fk.forward(arm_qpos)
        pose = Pose(position, rot6d_to_quat_wxyz(rot6d))
        wrist_rotation = Rotation.from_quat(row.vr["wrist_quat_wxyz"][WXYZ_TO_XYZW])
        # R_target = A W (A W0)^-1 E0，初始腕姿态映射到实测 EEF 姿态。
        rotation_right = (self.rotation * wrist_rotation).inv() * Rotation.from_quat(pose.q[WXYZ_TO_XYZW])
        if self.hand_retargeter is not None:
            self.hand_retargeter.reset(
                initial_qpos=row.hand["qpos"][0], timestamp_ns=row.observation_timestamp_ns,
            )

        # 所有初始化成功后才发布新锚点；首帧 EMA 从同一个实测 FK 位姿开始。
        self.anchor = row.vr["wrist_pos"].copy(), pose.p, rotation_right
        self.previous_arm_command = arm_qpos
        self.previous_eef_target = pose

    def _wrist_target(self, wrist_pos, wrist_quat_wxyz):
        """锚点相对映射与最短旋转 EMA；平滑参考只在提交后更新。"""
        previous = self.previous_eef_target
        if self.anchor is None or previous is None:
            raise RuntimeError("VR wrist mapping requires a reset anchor")
        wrist_origin, eef_origin, rotation_right = self.anchor
        position = eef_origin + self.config.vr_pos_scale * self.rotation.apply(wrist_pos - wrist_origin)
        rotation = self.rotation * Rotation.from_quat(wrist_quat_wxyz[WXYZ_TO_XYZW]) * rotation_right
        previous_rotation = Rotation.from_quat(previous.q[WXYZ_TO_XYZW])
        relative = previous_rotation.inv() * rotation
        smoothed = previous_rotation * Rotation.from_rotvec(self.config.ema_alpha_rot * relative.as_rotvec())
        alpha_pos = self.config.ema_alpha_pos
        return Pose(alpha_pos * position + (1.0 - alpha_pos) * previous.p, smoothed.as_quat()[XYZW_TO_WXYZ])

    def compute_target(self, row):
        self.target_timestamp_ns = row.observation_timestamp_ns
        self.rejection_reason = None
        target = self._wrist_target(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        hand = None
        if self.hand_retargeter is not None:
            observation_id = row.vr["ring_sequence"]
            if self.hand_observation_id != observation_id:
                # 先标记输入，求解失败后也不重复消费同一帧。
                self.hand_observation_id = observation_id
                self.hand_target = None
                self.hand_target = self.hand_retargeter.retarget(
                    row.vr["landmarks"], timestamp_ns=row.observation_timestamp_ns,
                )
            if self.hand_target is None:
                self.rejection_reason = self.hand_retargeter.rejection_reason
                return None
            hand = self.hand_target
        return self.action_mapper.map_eef(
            target, hand, row.arm["qpos"][0], self.previous_arm_command
        )

    def commit(self, command, *, eef_pose):
        if self.hand_retargeter is not None:
            self.hand_retargeter.commit(command.hand_qpos, timestamp_ns=self.target_timestamp_ns)
        self.previous_arm_command = command.arm_qpos.copy()
        self.previous_eef_target = eef_pose.copy()
