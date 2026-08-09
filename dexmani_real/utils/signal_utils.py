"""Signal processing utilities: Cartesian-space pose EMA."""

from __future__ import annotations

__all__ = ["ema_smooth_pose"]

import numpy as np


def _normalize_quat(q: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite (4,) quaternion")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError(f"{name} quaternion norm is too small")
    return value / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert a normalized wxyz quaternion to a shortest-arc rotation vector."""
    sign = np.asarray(q, dtype=np.float64)
    if sign[0] < 0:
        sign = -sign
    w, x, y, z = np.clip(sign[0], -1.0, 1.0), sign[1], sign[2], sign[3]
    sin_half = np.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, w)
    return angle * np.array([x, y, z], dtype=np.float64) / sin_half


def ema_smooth_pose(
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    prev_pos: np.ndarray,
    prev_quat_wxyz: np.ndarray,
    alpha_pos: float,
    alpha_rot: float,
) -> tuple[np.ndarray, np.ndarray]:
    """EMA in Cartesian space: position R³ + rotation vector so(3).

    Smooths a 6-DOF EEF target pose before IK.  Position uses standard
    Euclidean EMA; orientation converts the quaternion to a rotation
    vector (axis * angle), applies EMA in so(3), then converts back.

    Rotation-vector EMA naturally takes the short geodesic path on S³
    (magnitude = angle ∈ [0, π]) without the overhead of scipy Slerp.

    Position and rotation are smoothed with independent factors because
    they have different noise profiles and human motion bandwidths:
    position benefits from higher α (lower latency), rotation from lower
    α (stronger filtering of orientation jitter).

    Args:
        target_pos: (3,) target EEF position in meters.
        target_quat_wxyz: (4,) target EEF orientation quaternion (w, x, y, z).
        prev_pos: (3,) previous smoothed position.
        prev_quat_wxyz: (4,) previous smoothed orientation quaternion.
        alpha_pos: Smoothing factor for position in [0, 1].  1.0 = no smoothing.
        alpha_rot: Smoothing factor for rotation in [0, 1].  1.0 = no smoothing.

    Returns:
        ``(pos_smoothed, quat_wxyz_smoothed)`` — both float64 copies.
    """
    alpha_pos = float(np.clip(alpha_pos, 0.0, 1.0))
    alpha_rot = float(np.clip(alpha_rot, 0.0, 1.0))

    # Position: standard EMA in R³
    pos = alpha_pos * np.asarray(target_pos, dtype=np.float64) + (1.0 - alpha_pos) * np.asarray(
        prev_pos, dtype=np.float64
    )

    # Orientation: interpolate the relative rotation, not two absolute
    # rotation vectors.  Choose the target quaternion sign from the adjacent
    # dot product so the ±π/S³ antipode boundary always follows the shortest
    # arc, then apply an exp/log SO(3) step from the previous orientation.
    prev_quat = _normalize_quat(prev_quat_wxyz, name="prev_quat_wxyz")
    target_quat = _normalize_quat(target_quat_wxyz, name="target_quat_wxyz")
    if float(np.dot(prev_quat, target_quat)) < 0.0:
        target_quat = -target_quat
    prev_conjugate = prev_quat * np.array([1.0, -1.0, -1.0, -1.0])
    relative_quat = _normalize_quat(_quat_multiply(prev_conjugate, target_quat), name="relative")
    rv = alpha_rot * _quat_to_rotvec(relative_quat)

    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        quat = prev_quat.copy()
    else:
        axis = rv / angle
        half = angle / 2.0
        delta_quat = np.array(
            [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)], dtype=np.float64
        )
        quat = _normalize_quat(_quat_multiply(prev_quat, delta_quat), name="smoothed")

    return pos, quat
