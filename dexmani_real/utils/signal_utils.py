"""Signal processing utilities: EMA smoothing, jerk limiting."""

from __future__ import annotations

__all__ = ["ema_smooth", "ema_smooth_pose", "limit_jerk"]

import numpy as np


def ema_smooth(
    new_val: np.ndarray, prev_val: np.ndarray | None, alpha: float
) -> np.ndarray:
    """Exponential moving average smooth.

    Args:
        new_val: New sample value.
        prev_val: Previous smoothed value (None means no history).
        alpha: Smoothing factor in [0, 1]. 1.0 = no smoothing.

    Returns:
        Smoothed value.
    """
    if prev_val is None:
        return np.asarray(new_val, dtype=np.float64).copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * np.asarray(new_val, dtype=np.float64) + (1.0 - alpha) * prev_val


def limit_jerk(
    cmd_vel: np.ndarray,
    prev_vel: np.ndarray | None,
    prev_accel: np.ndarray | None,
    dt: float,
    max_jerk: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint jerk clamp via proportional scaling.

    When |(accel - prev_accel)/dt| > max_jerk on any joint, all joints are
    scaled proportionally to preserve trajectory shape.

    Ref: LeFranX Ruckig jerk limiting.

    Args:
        cmd_vel: Desired joint velocity (rad/s).
        prev_vel: Previous velocity command (None → no limiting).
        prev_accel: Previous acceleration (None → no limiting).
        dt: Time step in seconds.
        max_jerk: Maximum jerk per joint (rad/s³).

    Returns:
        (clamped_vel, new_accel) — caller must save new_accel for next frame.
    """
    vel = np.asarray(cmd_vel, dtype=np.float64).copy()

    if prev_vel is None or prev_accel is None or dt <= 0:
        return vel, np.zeros_like(vel)

    # Current acceleration
    accel = (vel - prev_vel) / dt

    # Jerk = (accel - prev_accel) / dt
    jerk = (accel - prev_accel) / dt

    jerk_abs = np.abs(jerk)
    max_ratio = np.max(jerk_abs) / max_jerk

    if max_ratio > 1.0:
        # Scale jerk proportionally
        jerk = jerk / max_ratio
        # Reconstruct accel from clamped jerk
        accel = prev_accel + jerk * dt
        # Reconstruct vel from clamped accel
        vel = prev_vel + accel * dt

    return vel, accel.copy()


# ═══════════════════════════════════════════════════════════════════════════
# Cartesian-space pose EMA — position R³ + rotation vector so(3)
# ═══════════════════════════════════════════════════════════════════════════


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
    pos = alpha_pos * np.asarray(target_pos, dtype=np.float64) + (1.0 - alpha_pos) * np.asarray(prev_pos, dtype=np.float64)

    # Orientation: quat → rotvec → EMA → quat
    def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
        """Convert wxyz quaternion to rotation vector (always shortest geodesic path).

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

    target_rv = _quat_to_rotvec(np.asarray(target_quat_wxyz, dtype=np.float64))
    prev_rv = _quat_to_rotvec(np.asarray(prev_quat_wxyz, dtype=np.float64))
    rv = alpha_rot * target_rv + (1.0 - alpha_rot) * prev_rv

    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = rv / angle
        half = angle / 2.0
        quat = np.array([np.cos(half), axis[0] * np.sin(half),
                         axis[1] * np.sin(half), axis[2] * np.sin(half)], dtype=np.float64)

    return pos, quat


