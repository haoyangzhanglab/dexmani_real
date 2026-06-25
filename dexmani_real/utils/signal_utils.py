"""Signal processing utilities: EMA smoothing, robust EMA, jerk limiting."""

from __future__ import annotations

__all__ = ["ema_smooth", "limit_jerk", "robust_ema"]

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


def robust_ema(
    new_val: np.ndarray,
    prev_val: np.ndarray | None,
    prev_raw: np.ndarray | None,
    alpha_normal: float = 0.95,
    alpha_anomaly: float = 0.3,
    anomaly_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust EMA with anomaly-adaptive alpha.

    Normal frames use alpha_normal (e.g. 0.95, ~1-frame TC at 50Hz).
    When the frame-to-frame jump exceeds anomaly_threshold (rad), alpha drops
    to alpha_anomaly (e.g. 0.3, ~10ms lag) for that single frame to suppress
    IK jitter spikes without adding persistent latency.

    Ref: LeFranX + ManiUniCon adaptive smoothing.

    Args:
        new_val: New raw value (e.g. IK result).
        prev_val: Previous smoothed output (None → no smoothing).
        prev_raw: Previous raw input for anomaly detection (None → skip).
        alpha_normal: Smoothing factor for normal frames.
        alpha_anomaly: Smoothing factor when anomaly detected.
        anomaly_threshold: Max per-joint frame-to-frame delta (rad) before
                           classifying as anomaly.

    Returns:
        (smoothed, new_raw) — caller must save new_raw for the next frame.
    """
    new_raw = np.asarray(new_val, dtype=np.float64).copy()
    if prev_val is None:
        return new_raw.copy(), new_raw

    # Detect anomaly: any joint jump exceeds threshold
    if prev_raw is not None and np.all(np.isfinite(prev_raw)):
        jump = np.max(np.abs(new_raw - prev_raw))
        alpha = alpha_anomaly if jump > anomaly_threshold else alpha_normal
    else:
        alpha = alpha_normal

    alpha = float(np.clip(alpha, 0.0, 1.0))
    smoothed = alpha * new_raw + (1.0 - alpha) * prev_val
    return smoothed, new_raw


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
