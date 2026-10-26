"""以重置锚点映射 VR 手腕位姿，并平滑已提交的遥操作目标。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.pose import (
    Pose,
    normalize_quat_wxyz,
    quat_multiply,
    quat_wxyz_to_rotmat,
)


def matrix_quat_wxyz(matrix):
    quaternion = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    return quaternion if quaternion[0] >= 0 else -quaternion


class VRWristMapper:
    """Reset-relative wrist mapper in xarm_base, using one VR calibration."""

    def __init__(
        self,
        *,
        pos_scale: float,
        vr_to_robot_rot: np.ndarray,
    ) -> None:
        self.pos_scale = pos_scale
        self.vr_to_robot_rot = np.array(vr_to_robot_rot, dtype=np.float64, copy=True)

        self.anchor: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def reset(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
        eef_pos: np.ndarray,
        eef_quat_wxyz: np.ndarray,
    ) -> None:
        # A failed reset must not leave an older anchor available.
        self.clear()
        wrist = Pose(wrist_pos, wrist_quat_wxyz)
        eef = Pose(eef_pos, eef_quat_wxyz)
        self.anchor = (
            wrist.p.copy(), quat_wxyz_to_rotmat(wrist.q),
            eef.p.copy(), quat_wxyz_to_rotmat(eef.q),
        )

    def map(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
    ) -> Pose:
        if self.anchor is None:
            raise RuntimeError("VR wrist mapping requires a reset anchor")
        wrist_pos0, wrist_rot0, eef_pos0, eef_rot0 = self.anchor
        wrist = Pose(wrist_pos, wrist_quat_wxyz)
        delta_pos_base = self.pos_scale * (self.vr_to_robot_rot @ (wrist.p - wrist_pos0))

        # C maps VR axes into xarm_base; conjugation expresses the anchored rotation there.
        delta_rot_vr = quat_wxyz_to_rotmat(wrist.q) @ wrist_rot0.T
        delta_rot_base = self.vr_to_robot_rot @ delta_rot_vr @ self.vr_to_robot_rot.T
        return Pose(eef_pos0 + delta_pos_base, matrix_quat_wxyz(delta_rot_base @ eef_rot0))

    def clear(self) -> None:
        self.anchor = None


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert a normalized WXYZ quaternion to a shortest-arc rotation vector."""
    sign = q if q[0] >= 0 else -q
    w, x, y, z = np.clip(sign[0], -1.0, 1.0), sign[1], sign[2], sign[3]
    sin_half = np.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, w)
    return angle * np.array([x, y, z], dtype=np.float64) / sin_half


def ema_smooth_pose(target: Pose, previous: Pose, alpha_pos: float, alpha_rot: float) -> Pose:
    """Smooth base-frame meters and shortest relative rotations with startup-validated alphas."""
    pos = alpha_pos * target.p + (1.0 - alpha_pos) * previous.p
    prev_quat = previous.q
    target_quat = target.q
    if float(np.dot(prev_quat, target_quat)) < 0.0:
        target_quat = -target_quat
    prev_conjugate = prev_quat * np.array([1.0, -1.0, -1.0, -1.0])
    relative_quat = normalize_quat_wxyz(quat_multiply(prev_conjugate, target_quat), name="relative")
    rv = alpha_rot * quat_to_rotvec(relative_quat)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        quat = prev_quat.copy()
    else:
        axis = rv / angle
        half = angle / 2.0
        delta_quat = np.array(
            [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
            dtype=np.float64,
        )
        quat = quat_multiply(prev_quat, delta_quat)
    return Pose(pos, quat)
