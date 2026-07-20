"""Pose math utilities — composition, error, quaternion/rot6d conversions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .types import Pose

_WXYZ_TO_XYZW = np.array([1, 2, 3, 0], dtype=np.intp)
_XYZW_TO_WXYZ = np.array([3, 0, 1, 2], dtype=np.intp)

__all__ = [
    "angular_dist_rad",
    "compose_pose",
    "compute_pose_error",
    "invert_pose",
    "normalize_quat_wxyz",
    "pose_error_vector",
    "quat_multiply",
    "quat_wxyz_to_rot6d",
    "quat_wxyz_to_rotmat",
    "random_quat_full_so3",
    "random_quat_multi_axis",
    "random_quat_within_angle",
    "rot6d_to_quat_wxyz",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
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


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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
    qv_rot = _quat_multiply(_quat_multiply(q, qv), _quat_conjugate(q))
    return qv_rot[1:]


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to rotation vector (axis * angle).

    Handles quaternion double cover: q and -q represent the same rotation.
    Forcing w ≥ 0 ensures the rotation vector angle is always ≤ π.
    """
    sign = np.asarray(q, dtype=np.float64)
    if sign[0] < 0:
        sign = -sign
    w, x, y, z = sign[0], sign[1], sign[2], sign[3]
    sin_half = np.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, w)
    return angle * np.array([x, y, z], dtype=np.float64) / sin_half


def compose_pose(parent: Pose, child: Pose) -> Pose:
    p = parent.p + _quat_rotate_vector(parent.q, child.p)
    q = _quat_multiply(parent.q, child.q)
    return Pose(p=p, q=q)


def invert_pose(pose: Pose) -> Pose:
    q_inv = _quat_conjugate(pose.q)
    p = _quat_rotate_vector(q_inv, -pose.p)
    return Pose(p=p, q=q_inv)


def compute_pose_error(target: Pose, actual: Pose) -> tuple[float, float]:
    position_error = float(np.linalg.norm(target.p - actual.p))
    q_dot = abs(float(np.dot(target.q, actual.q)))
    rotation_error = 2.0 * np.arccos(min(1.0, q_dot))
    return position_error, rotation_error


def pose_error_vector(target: Pose, actual: Pose, max_pos_step: float, max_rot_step: float) -> np.ndarray:
    position_error = target.p - actual.p
    position_norm = np.linalg.norm(position_error)
    if position_norm > max_pos_step > 0:
        position_error = position_error / position_norm * max_pos_step

    q_error = _quat_multiply(target.q, _quat_conjugate(actual.q))
    rotation_vector = quat_to_rotvec(q_error)
    rotation_norm = np.linalg.norm(rotation_vector)
    if rotation_norm > max_rot_step > 0:
        rotation_vector = rotation_vector / rotation_norm * max_rot_step

    return np.concatenate([position_error, rotation_vector])


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
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """WXYZ quaternion → 3×3 rotation matrix.

    Uses scipy.spatial.transform.Rotation for numerical stability.
    """
    quat_xyzw = wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float64).reshape(4))
    return Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Public quaternion helpers (aliases / new functions)
# ═══════════════════════════════════════════════════════════════════════


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions: q1 ⊗ q2."""
    return _quat_multiply(q1, q2)


def angular_dist_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance between two wxyz quaternions (radians)."""
    return float(2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)))


def random_quat_within_angle(rng: np.random.RandomState, max_deg: float) -> np.ndarray:
    """Uniform random rotation quaternion (wxyz) with angle ≤ max_deg.

    Args:
        rng: numpy RandomState for reproducibility.
        max_deg: maximum rotation angle in degrees.

    Returns:
        (4,) wxyz quaternion.
    """
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, np.deg2rad(max_deg))
    half = angle / 2
    return np.array([np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)])


def random_quat_full_so3(rng: np.random.RandomState) -> np.ndarray:
    """Uniformly sample SO(3) full-space random quaternion (wxyz).

    Uses the Marsaglia method (uniform distribution on the unit sphere in S³).
    """
    u = rng.uniform(0, 1, 3)
    q = np.array(
        [
            np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
            np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
            np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
            np.sqrt(u[0]) * np.cos(2 * np.pi * u[2]),
        ]
    )
    q /= np.linalg.norm(q)
    return q


def random_quat_multi_axis(
    rng: np.random.RandomState,
    max_deg1: float = 45.0,
    max_deg2: float = 30.0,
) -> np.ndarray:
    """Two successive rotations around independent random axes.

    Produces richer attitude distribution than single-axis rotation.
    Composite rotation = R₂ * R₁ (applied in that order).

    Args:
        rng: numpy RandomState for reproducibility.
        max_deg1: maximum angle for the first rotation (degrees).
        max_deg2: maximum angle for the second rotation (degrees).

    Returns:
        (4,) wxyz quaternion.
    """
    # Axis 1: random direction
    a1 = rng.randn(3)
    a1 /= np.linalg.norm(a1)
    angle1 = rng.uniform(0, np.deg2rad(max_deg1))
    half1 = angle1 / 2
    q1 = np.array([np.cos(half1), a1[0] * np.sin(half1), a1[1] * np.sin(half1), a1[2] * np.sin(half1)])

    # Axis 2: orthogonal to axis 1
    a2 = rng.randn(3)
    a2 -= a1 * np.dot(a2, a1)
    norm = np.linalg.norm(a2)
    if norm < 1e-10:
        a2 = np.array([-a1[1], a1[0], 0.0]) if abs(a1[0]) > 1e-10 else np.array([1.0, 0.0, 0.0])
        a2 -= a1 * np.dot(a2, a1)
    a2 /= np.linalg.norm(a2)
    angle2 = rng.uniform(0, np.deg2rad(max_deg2))
    half2 = angle2 / 2
    q2 = np.array([np.cos(half2), a2[0] * np.sin(half2), a2[1] * np.sin(half2), a2[2] * np.sin(half2)])

    return quat_multiply(q2, q1)  # R₂ * R₁
