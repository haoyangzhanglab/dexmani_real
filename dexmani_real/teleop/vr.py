"""VR 锚点映射、手部重定向与提交后的位姿平滑。"""

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.pose import Pose, quat_wxyz_to_rotmat, rot6d_to_quat_wxyz


def ema_smooth_pose(target: Pose, previous: Pose, alpha_pos: float, alpha_rot: float) -> Pose:
    """平移做 EMA，旋转沿最短相对旋转插值。"""
    position = alpha_pos * target.p + (1.0 - alpha_pos) * previous.p
    rotation = Rotation.from_quat(previous.q[[1, 2, 3, 0]])
    relative = rotation.inv() * Rotation.from_quat(target.q[[1, 2, 3, 0]])
    smoothed = rotation * Rotation.from_rotvec(alpha_rot * relative.as_rotvec())
    return Pose(position, smoothed.as_quat()[[3, 0, 1, 2]])


class VRController:
    def __init__(self, action_mapper, hand_retargeter, config, vr_to_robot_rot):
        self.action_mapper = action_mapper
        self.hand_retargeter = hand_retargeter
        self.config = config
        self.rotation = np.array(vr_to_robot_rot, dtype=np.float64, copy=True)
        self.clear()

    def clear(self):
        self.anchor = None
        self.previous_arm_command = None
        self.smoothed_eef_pose = None
        self.hand_observation_id = self.hand_target = None
        self.target_timestamp_ns = None
        self.rejection_reason = None

    def reset(self, row):
        self.clear()
        arm_qpos = row.arm["qpos"][0].copy()
        position, rot6d = self.action_mapper.constraints.target_fk.forward(arm_qpos)
        pose = Pose(position.copy(), rot6d_to_quat_wxyz(rot6d))
        wrist_rotation = quat_wxyz_to_rotmat(row.vr["wrist_quat_wxyz"])
        rotation_right = wrist_rotation.T @ self.rotation.T @ quat_wxyz_to_rotmat(pose.q)
        if self.hand_retargeter is not None:
            self.hand_retargeter.reset(
                initial_qpos=row.hand["qpos"][0], timestamp_ns=row.observation_timestamp_ns,
            )

        # 所有初始化成功后才发布新锚点；首帧 EMA 从同一个实测 FK 位姿开始。
        self.anchor = row.vr["wrist_pos"].copy(), pose.p, rotation_right
        self.previous_arm_command = arm_qpos
        self.smoothed_eef_pose = pose

    def map_wrist(self, wrist_pos, wrist_quat_wxyz):
        if self.anchor is None:
            raise RuntimeError("VR wrist mapping requires a reset anchor")
        wrist_origin, eef_origin, rotation_right = self.anchor
        position = eef_origin + self.config.vr_pos_scale * (
            self.rotation @ (wrist_pos - wrist_origin)
        )
        rotation = self.rotation @ quat_wxyz_to_rotmat(wrist_quat_wxyz) @ rotation_right
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        return Pose(position, quaternion if quaternion[0] >= 0 else -quaternion)

    def compute_target(self, row):
        self.target_timestamp_ns = row.observation_timestamp_ns
        self.rejection_reason = None
        mapped = self.map_wrist(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        target = ema_smooth_pose(
            mapped,
            self.smoothed_eef_pose,
            self.config.ema_alpha_pos,
            self.config.ema_alpha_rot,
        )
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

    def commit(self, result):
        if self.hand_retargeter is not None:
            self.hand_retargeter.commit(result.hand_qpos, timestamp_ns=self.target_timestamp_ns)
        self.previous_arm_command = result.arm_qpos.copy()
        self.smoothed_eef_pose = result.eef_pose.copy()
