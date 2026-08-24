"""Pose composition, error, quaternion, and rotation-6D operations."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .types import Pose

_WXYZ_TO_XYZW = np.array([1, 2, 3, 0], dtype=np.intp)
_XYZW_TO_WXYZ = np.array([3, 0, 1, 2], dtype=np.intp)

__all__ = [
    "compose_pose",
    "compute_pose_error",
    "forward_from_quat_wxyz",
    "invert_pose",
    "normalize_quat_wxyz",
    "quat_multiply",
    "quat_wxyz_to_rot6d",
    "quat_wxyz_to_rotmat",
    "rot6d_to_quat_wxyz",
]


def ensure_qpos(qpos: np.ndarray, dof: int, name: str) -> np.ndarray:
    if isinstance(qpos, np.ndarray) and qpos.ndim == 1 and qpos.shape[0] == dof and qpos.dtype == np.float64:
        return qpos
    array = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if array.shape[0] != dof:
        raise ValueError(f"{name} must have length {dof}, got {array.shape[0]}.")
    return array.copy()


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q[_WXYZ_TO_XYZW]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q[_XYZW_TO_WXYZ]


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


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a wxyz quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a wxyz quaternion."""
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    qv_rot = quat_multiply(quat_multiply(q, qv), _quat_conjugate(q))
    return qv_rot[1:]


def compose_pose(parent: Pose, child: Pose) -> Pose:
    p = parent.p + _quat_rotate_vector(parent.q, child.p)
    q = quat_multiply(parent.q, child.q)
    return Pose(p=p, q=q)


def invert_pose(pose: Pose) -> Pose:
    q_inv = _quat_conjugate(pose.q)
    p = _quat_rotate_vector(q_inv, -pose.p)
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


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Normalize a wxyz quaternion, returning identity on degenerate input."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def rot6d_to_quat_wxyz(r6: np.ndarray) -> np.ndarray:
    """6D rotation → WXYZ quaternion via Gram-Schmidt orthonormalization.

    r6 = [c1x, c1y, c1z, c2x, c2y, c2z]  (first two columns of a 3×3 rotation matrix).

    Robust against degenerate inputs: near-zero c1, collinear c1/c2, and
    negative-determinant matrices — all of which can arise from policy predictions.
    """
    r6 = np.asarray(r6, dtype=np.float64).reshape(6)
    c1 = r6[:3].copy()
    c2 = r6[3:].copy()

    c1_norm = float(np.linalg.norm(c1))
    if c1_norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    u1 = c1 / c1_norm
    c2_orth = c2 - np.dot(c2, u1) * u1
    c2_orth_norm = float(np.linalg.norm(c2_orth))

    if c2_orth_norm < 1e-12:
        idx = int(np.argmin(np.abs(u1)))
        e = np.zeros(3, dtype=np.float64)
        e[idx] = 1.0
        u2 = np.cross(u1, e)
        if float(np.linalg.norm(u2)) < 1e-12:
            e = np.zeros(3, dtype=np.float64)
            e[(idx + 1) % 3] = 1.0
            u2 = np.cross(u1, e)
        u2 = u2 / np.linalg.norm(u2)
    else:
        u2 = c2_orth / c2_orth_norm

    u3 = np.cross(u1, u2)
    R = np.column_stack([u1, u2, u3])

    if np.linalg.det(R) < 0.0:
        R[:, 2] = -u3

    quat_xyzw = Rotation.from_matrix(R).as_quat()
    return xyzw_to_wxyz(quat_xyzw)


def quat_wxyz_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    """WXYZ quaternion → 6D rotation (first two columns of the 3×3 rotation matrix)."""
    quat_xyzw = wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float64).reshape(4))
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    return np.concatenate([R[:, 0], R[:, 1]])


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """WXYZ quaternion → 3×3 rotation matrix.

    Uses scipy.spatial.transform.Rotation for numerical stability.
    """
    quat_xyzw = wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float64).reshape(4))
    return Rotation.from_quat(quat_xyzw).as_matrix()


def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """6D rotation → 3×3 rotation matrix (via WXYZ quaternion)."""
    return quat_wxyz_to_rotmat(rot6d_to_quat_wxyz(r6))


def forward_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """FLU +X forward direction from a WXYZ quaternion.

    Applies the quaternion rotation to the canonical forward vector [1, 0, 0],
    returning the resulting direction in world coordinates.
    """
    quat_xyzw = wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float64).reshape(4))
    return Rotation.from_quat(quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))
