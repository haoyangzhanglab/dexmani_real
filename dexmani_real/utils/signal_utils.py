"""Signal processing utilities: EMA smoothing, jerk limiting."""

from __future__ import annotations

__all__ = ["ema_smooth", "limit_jerk"]

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
