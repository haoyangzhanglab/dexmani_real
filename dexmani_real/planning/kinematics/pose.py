"""Pose composition, error, quaternion, and rotation-6D operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.utils.geometry import normalize_quat_wxyz


@dataclass
class Pose:
    """Pose with position and wxyz quaternion."""

    p: np.ndarray
    q: np.ndarray

    def __init__(self, p: Any, q: Any) -> None:
        self.p = np.asarray(p, dtype=np.float64).reshape(3)
        self.q = normalize_quat_wxyz(q)
        if not np.isfinite(self.p).all():
            raise ValueError("Pose position must be finite")

    @classmethod
    def identity(cls) -> "Pose":
        return cls(p=np.zeros(3, dtype=np.float64), q=np.array([1.0, 0.0, 0.0, 0.0]))

    def copy(self) -> "Pose":
        return Pose(p=self.p.copy(), q=self.q.copy())


WXYZ_TO_XYZW = np.array([1, 2, 3, 0], dtype=np.intp)
XYZW_TO_WXYZ = np.array([3, 0, 1, 2], dtype=np.intp)
ROT6D_MIN_BASIS_NORM = 1e-4
ROT6D_MIN_ORTHOGONAL_RATIO = 1e-4

__all__ = [
    "compose_pose",
    "compute_pose_error",
    "forward_from_quat_wxyz",
    "invert_pose",
    "quat_multiply",
    "quat_wxyz_to_rotmat",
    "rot6d_to_quat_wxyz",
    "rot6d_to_rotmat",
]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q[WXYZ_TO_XYZW]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q[XYZW_TO_WXYZ]


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a wxyz quaternion."""
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    qv_rot = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return qv_rot[1:]


def compose_pose(parent: Pose, child: Pose) -> Pose:
    p = parent.p + quat_rotate_vector(parent.q, child.p)
    q = quat_multiply(parent.q, child.q)
    return Pose(p=p, q=q)


def invert_pose(pose: Pose) -> Pose:
    q_inv = quat_conjugate(pose.q)
    p = quat_rotate_vector(q_inv, -pose.p)
    return Pose(p=p, q=q_inv)


def compute_pose_error(target: Pose, actual: Pose) -> tuple[float, float]:
    # Return infinite errors for non-finite poses so gates reject them.
    if not np.all(np.isfinite(actual.p)) or not np.all(np.isfinite(actual.q)):
        return (float("inf"), float("inf"))
    if not np.all(np.isfinite(target.p)) or not np.all(np.isfinite(target.q)):
        return (float("inf"), float("inf"))
    position_error = float(np.linalg.norm(target.p - actual.p))
    q_dot = abs(float(np.dot(target.q, actual.q)))
    rotation_error = 2.0 * np.arccos(min(1.0, q_dot))
    return position_error, rotation_error


def rot6d_to_quat_wxyz(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt [c1x,c1y,c1z,c2x,c2y,c2z] to WXYZ; reject nonfinite or near-collinear bases."""
    quat_xyzw = Rotation.from_matrix(rot6d_to_rotmat(r6)).as_quat()
    return xyzw_to_wxyz(quat_xyzw)


def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt rot6d columns to a right-handed rotation matrix."""
    values = np.asarray(r6, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("rot6d must be a finite (6,) vector")
    first, second = values[:3], values[3:]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if (
        not np.isfinite([first_norm, second_norm]).all()
        or min(first_norm, second_norm) < ROT6D_MIN_BASIS_NORM
    ):
        raise ValueError("rot6d contains a zero or unrepresentable basis")

    unit_first = first / first_norm
    orthogonal_second = second - np.dot(second, unit_first) * unit_first
    orthogonal_norm = float(np.linalg.norm(orthogonal_second))
    if (
        not np.isfinite(orthogonal_norm)
        or orthogonal_norm / second_norm < ROT6D_MIN_ORTHOGONAL_RATIO
    ):
        raise ValueError("rot6d contains a near-collinear basis")
    unit_second = orthogonal_second / orthogonal_norm
    return np.column_stack((unit_first, unit_second, np.cross(unit_first, unit_second)))


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = wxyz_to_xyzw(normalize_quat_wxyz(q_wxyz))
    return Rotation.from_quat(quat_xyzw).as_matrix()


def forward_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Return the rotated FLU +X forward vector for a WXYZ quaternion."""
    quat_xyzw = wxyz_to_xyzw(normalize_quat_wxyz(q_wxyz))
    return Rotation.from_quat(quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))
