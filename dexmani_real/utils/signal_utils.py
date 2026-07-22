"""Signal processing utilities: Cartesian-space pose EMA."""

from __future__ import annotations

__all__ = ["alpha_from_tau", "ema_smooth_pose", "tau_from_alpha", "EMA_ALPHA_POS", "EMA_ALPHA_ROT"]

import numpy as np

from dexmani_real.planning.pose_utils import quat_to_rotvec

# ── Canonical EMA alpha values (tuned at 16Hz, dt=62.5ms) ──
# 2026-07-22 调整: 配合 joint_max_acc 500→900°/s²，双通道加重平滑，
# 降低腕部(J5/J6/J7)跟踪误差。位置/姿态突变是腕关节跟踪滞后的主因 —
# EMA 平缓 IK 输入 → 关节目标步长缩小 → 固件加速度瓶颈冲击减轻。
# 旧值: POS=0.8(τ≈39ms) ROT=0.4(τ≈122ms)
# 新值: POS=0.65(τ≈60ms) ROT=0.3(τ≈175ms)
EMA_ALPHA_POS = 0.65  # Cartesian position: moderate smoothing
EMA_ALPHA_ROT = 0.3   # Cartesian rotation: heavier smoothing


def alpha_from_tau(tau_s: float, dt: float) -> float:
    """Discrete EMA coefficient preserving a continuous time constant.

    For the new-sample-weighted EMA ``y += alpha * (x - y)`` running at
    period ``dt``, ``alpha = 1 - exp(-dt / tau_s)`` yields the same
    smoothing time constant ``tau_s`` regardless of loop rate.  Use this
    when changing the control rate so filters keep identical dynamics
    (e.g. alpha=0.6 @ 50Hz ↔ tau=21.8ms ↔ alpha=0.94 @ 16Hz).

    Args:
        tau_s: Filter time constant in seconds (<= 0 → no smoothing).
        dt: Loop period in seconds.

    Returns:
        alpha in (0, 1] for ``y += alpha * (x - y)``.
    """
    if tau_s <= 0:
        return 1.0
    return float(1.0 - np.exp(-dt / tau_s))


def tau_from_alpha(alpha: float, dt: float) -> float:
    """Inverse of :func:`alpha_from_tau`: time constant of an EMA coefficient.

    Use to convert a filter constant tuned at one loop rate to another:
    ``alpha_from_tau(tau_from_alpha(0.6, 1/50), 1/16) ≈ 0.94``.

    Args:
        alpha: EMA coefficient in [0, 1] for ``y += alpha * (x - y)``.
        dt: Loop period in seconds the coefficient was tuned at.

    Returns:
        Time constant in seconds (0.0 for alpha >= 1, inf for alpha <= 0).
    """
    if alpha >= 1.0:
        return 0.0
    if alpha <= 0.0:
        return float("inf")
    return float(-dt / np.log(1.0 - alpha))


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
    pos = alpha_pos * np.asarray(target_pos, dtype=np.float64) + (1.0 - alpha_pos) * np.asarray(
        prev_pos, dtype=np.float64
    )

    # Orientation: quat → rotvec → EMA → quat
    target_rv = quat_to_rotvec(np.asarray(target_quat_wxyz, dtype=np.float64))
    prev_rv = quat_to_rotvec(np.asarray(prev_quat_wxyz, dtype=np.float64))
    rv = alpha_rot * target_rv + (1.0 - alpha_rot) * prev_rv

    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = rv / angle
        half = angle / 2.0
        quat = np.array(
            [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)], dtype=np.float64
        )

    return pos, quat
