from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.planning.kinematics.pose import Pose, quat_multiply, quat_wxyz_to_rotmat
from dexmani_real.utils.geometry import normalize_quat_wxyz, validate_rotation_matrix

logger = logging.getLogger(__name__)


def finite_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a finite float64 copy with the exact expected shape."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


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
        self.vr_to_robot_rot = validate_rotation_matrix(vr_to_robot_rot, name="vr_to_robot_rot")

        self.wrist_pos0: np.ndarray | None = None
        self.wrist_rot0: np.ndarray | None = None
        self.eef_pos0: np.ndarray | None = None
        self.eef_rot0: np.ndarray | None = None
        self.last_quat_wxyz: np.ndarray | None = None

    def reset(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
        eef_pos: np.ndarray,
        eef_quat_wxyz: np.ndarray,
    ) -> None:
        # Validate before committing the new anchor; failed resets clear stale state.
        try:
            next_wrist_pos0 = finite_vector(wrist_pos, (3,), "wrist_pos")
            next_wrist_rot0 = quat_wxyz_to_rotmat(wrist_quat_wxyz)
            next_eef_pos0 = finite_vector(eef_pos, (3,), "eef_pos")
            next_eef_rot0 = quat_wxyz_to_rotmat(eef_quat_wxyz)
        except (TypeError, ValueError):
            self.clear()
            logger.warning(
                "VRWristMapper.reset: invalid pose input — mapper cleared",
                exc_info=True,
            )
            return
        self.wrist_pos0 = next_wrist_pos0
        self.wrist_rot0 = next_wrist_rot0
        self.eef_pos0 = next_eef_pos0
        self.eef_rot0 = next_eef_rot0
        self.last_quat_wxyz = matrix_quat_wxyz(self.eef_rot0)

    def map(
        self,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
    ) -> Pose | None:
        if not self.is_ready():
            return None

        try:
            current_wrist_pos = finite_vector(wrist_pos, (3,), "wrist_pos")
            wrist_rot = quat_wxyz_to_rotmat(wrist_quat_wxyz)
        except (TypeError, ValueError):
            logger.warning("VRWristMapper.map: invalid wrist pose — no target", exc_info=True)
            return None

        delta_pos_vr = current_wrist_pos - self.wrist_pos0
        delta_pos_base = self.pos_scale * (self.vr_to_robot_rot @ delta_pos_vr)

        # C maps VR axes into xarm_base; conjugation expresses the anchored rotation there.
        delta_rot_vr = wrist_rot @ self.wrist_rot0.T
        delta_rot_base = self.vr_to_robot_rot @ delta_rot_vr @ self.vr_to_robot_rot.T
        target_pos = self.eef_pos0 + delta_pos_base
        if not np.all(np.isfinite(target_pos)):
            logger.warning("VRWristMapper.map: non-finite mapped pose — no target")
            return None

        target_rot = delta_rot_base @ self.eef_rot0
        pose = Pose(target_pos, matrix_quat_wxyz(target_rot))
        if np.dot(pose.q, self.last_quat_wxyz) < 0:
            pose.q = -pose.q
        self.last_quat_wxyz = pose.q.copy()
        return pose

    def clear(self) -> None:
        self.wrist_pos0 = None
        self.wrist_rot0 = None
        self.eef_pos0 = None
        self.eef_rot0 = None
        self.last_quat_wxyz = None

    def is_ready(self) -> bool:
        return self.wrist_pos0 is not None


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
