"""xArm 基座系的机械臂 FK，以及手基座系的 XHand link FK。"""

from collections.abc import Sequence

import numpy as np

from dexmani_real.robot.model import (
    HAND_SDK_TO_URDF_IDX,
    XARM7_XHAND_COLLISION_URDF_PATH,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_RIGHT_URDF_PATH,
    XHAND_URDF_JOINT_NAMES,
)

SDK_TO_URDF_IDX = np.asarray(HAND_SDK_TO_URDF_IDX, dtype=np.intp)


class ArmFK:
    """URDF EEF 与 xArm 固件 EEF 的坐标定义不同。"""

    def __init__(self):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(XARM7_XHAND_COLLISION_URDF_PATH))
        self.data = self.model.createData()
        self.eef_frame_id = self.model.getFrameId("custom_eef_link")
        if self.eef_frame_id >= self.model.nframes:
            raise ValueError("arm URDF is missing custom_eef_link")

    def forward(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.pin.forwardKinematics(self.model, self.data, np.asarray(qpos, dtype=np.float64))
        pose = self.pin.updateFramePlacement(self.model, self.data, self.eef_frame_id)
        rotation = pose.rotation
        return pose.translation.copy(), np.concatenate((rotation[:, 0], rotation[:, 1]))


class HandFK:
    def __init__(self, fingertip_link_names: Sequence[str] = XHAND_FINGERTIP_LINK_NAMES):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(XHAND_RIGHT_URDF_PATH))
        self.data = self.model.createData()
        active = sorted(
            (joint.idx_q, self.model.names[i])
            for i, joint in enumerate(self.model.joints)
            if i and joint.nq
        )
        if self.model.nq != 12 or tuple(name for _, name in active) != XHAND_URDF_JOINT_NAMES:
            raise ValueError("hand FK active q order must match XHand URDF joints")
        if len(fingertip_link_names) != 5 or len(set(fingertip_link_names)) != 5:
            raise ValueError("hand FK requires five distinct fingertip frames")
        self.fingertip_frame_ids = tuple(self.model.getFrameId(n) for n in fingertip_link_names)
        if any(fid >= self.model.nframes for fid in self.fingertip_frame_ids):
            raise ValueError("fingertip frame is missing from the hand URDF")

    def forward(self, qpos: np.ndarray, frame_ids) -> np.ndarray:
        q = np.asarray(qpos, dtype=np.float64)[SDK_TO_URDF_IDX]
        self.pin.forwardKinematics(self.model, self.data, q)
        # stack 复制 placement，避免后续 FK 覆盖结果。
        return np.stack([
            self.pin.updateFramePlacement(self.model, self.data, int(fid)).homogeneous
            for fid in frame_ids
        ])


def eef_poses(arm_qpos: np.ndarray, *, arm_fk: ArmFK | None = None) -> np.ndarray:
    """返回 float64[T,9]，依次为基座系 XYZ 和旋转矩阵前两列。"""
    fk = ArmFK() if arm_fk is None else arm_fk
    poses = np.empty((len(arm_qpos), 9), dtype=np.float64)
    for index, qpos in enumerate(arm_qpos):
        poses[index, :3], poses[index, 3:] = fk.forward(qpos)
    return poses
