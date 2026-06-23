"""Per-frame safety checks. Stateless, shared by teleop and deploy.

Functions accept base numpy types (not RobotState) for flexible use
by both controller and pipeline layers.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.types import _ARM_TORQUE_LIMIT_NM

# Default thresholds (teleop). Deploy may use stricter values.
# _ARM_TORQUE_LIMIT_NM is defined in robot/types.py (hardware property, not teleop policy).
# _HAND_CURRENT_LIMIT_MA: software safety backup for XHand hardware torque-current limit.
# Verified 2026-06-23: XHand hardware tor_max=400mA (xhand.py:120) trips first as the
# primary current-limiting layer. This 500mA software threshold is intentionally higher
# — it only triggers if the hardware limit fails, serving as a layered-defense backup.
# The 500mA value is verified safe against XHand motor specs (peak handling ~600mA).
_HAND_CURRENT_LIMIT_MA = 500.0
_HAND_TEMP_LIMIT_C = 70.0
# Retarget validity range aligned with XHand hardware joint limits (2026-06-22).
# Previous hardcoded [-0.5, 2.5] was too narrow on the low end (-0.5 > XHand
# thumb min -0.698 rad, causing false negatives) and too wide on the high end
# (2.5 > XHand max 1.92 rad, risking false positives).
# New bounds match XHand qpos_min.min() ≈ -0.698 rad and qpos_max.max() ≈ 1.92 rad,
# with a small safety margin (±0.05 rad) to allow for numerical noise at IK boundaries.
_RETARGET_VALID_MIN = -0.75  # rad, ~43°, XHand thumb min=-40° (-0.698 rad)
_RETARGET_VALID_MAX = 2.0  # rad, ~115°, XHand max=110° (1.92 rad)


def check_arm_torque(
    arm_tau: np.ndarray,
    torque_limit_nm: np.ndarray | float = _ARM_TORQUE_LIMIT_NM,
) -> bool:
    tau = np.asarray(arm_tau, dtype=np.float64)
    if not np.all(np.isfinite(tau)):
        return False
    if isinstance(torque_limit_nm, np.ndarray):
        if len(tau) != len(torque_limit_nm):
            return False
        return not np.any(np.abs(tau) >= torque_limit_nm)
    return float(np.max(np.abs(tau))) < torque_limit_nm


def check_hand_current(
    hand_current: np.ndarray,
    current_limit_ma: float = _HAND_CURRENT_LIMIT_MA,
) -> bool:
    cur = np.asarray(hand_current, dtype=np.float64)
    if not np.all(np.isfinite(cur)):
        return False
    return float(np.max(cur)) < current_limit_ma


def check_hand_temperature(
    hand_temp: np.ndarray,
    temp_limit_c: float = _HAND_TEMP_LIMIT_C,
) -> bool:
    temp = np.asarray(hand_temp, dtype=np.float64)
    if not np.all(np.isfinite(temp)):
        return False
    return float(np.max(temp)) < temp_limit_c


def check_hand_comm(hand_error: bool) -> bool:
    return not hand_error


def check_retarget_valid(
    hand_qpos: np.ndarray,
    physio_min: float = _RETARGET_VALID_MIN,
    physio_max: float = _RETARGET_VALID_MAX,
) -> bool:
    """Check retargeted hand_qpos is within hardware-aligned range.

    Default bounds match XHand joint limits with safety margin (±0.05 rad).
    Callers can override for other hand hardware or tighter safety policies.
    """
    if not np.all(np.isfinite(hand_qpos)):
        return False
    if np.any(hand_qpos < physio_min) or np.any(hand_qpos > physio_max):
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# Sliding-window cyclic limit monitoring (P3.3)
# ═══════════════════════════════════════════════════════════════════════


class SlidingWindowMonitor:
    """Tracks a signal over a sliding window and warns on persistent elevation.

    Complements single-tick threshold checks (check_hand_current, etc.) by
    detecting gradual degradation that stays below the hard limit but
    consistently exceeds a warning threshold — e.g. a motor running warm
    (55°C < 70°C limit) for several seconds, or current creeping toward the
    limit under sustained load.

    Usage:
        temp_monitor = SlidingWindowMonitor(window_size=50, warn_threshold=55.0)
        # In _tick():
        over_limit, should_warn = temp_monitor.update(hand_temp)
        if should_warn:
            logger.warning("Hand temp elevated: mean=%.1f°C over %d ticks",
                           temp_monitor.window_mean, temp_monitor.window_size)
    """

    def __init__(
        self,
        window_size: int = 50,  # ~1 second at 50 Hz
        warn_threshold: float = 0.0,
        warn_fraction: float = 0.6,  # fraction of window exceeding threshold to trigger
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.warn_threshold = float(warn_threshold)
        self.warn_fraction = float(warn_fraction)
        self._buffer: np.ndarray | None = None
        self._idx: int = 0
        self._count: int = 0

    def update(self, value: float | np.ndarray) -> tuple[bool, bool]:
        """Push a new value and return (over_hard_limit, should_warn).

        Args:
            value: Scalar or array (max is taken for arrays).

        Returns:
            (over_limit, should_warn) — over_limit=True when the current value
            exceeds the hard limit (separate from this monitor's warn threshold);
            should_warn=True when the fraction of window exceeding warn_threshold
            reaches warn_fraction.
        """
        if isinstance(value, np.ndarray):
            scalar = float(np.max(np.abs(value)))
        else:
            scalar = float(value)

        if self._buffer is None:
            self._buffer = np.full(self.window_size, np.nan, dtype=np.float64)

        self._buffer[self._idx] = scalar
        self._idx = (self._idx + 1) % self.window_size
        if self._count < self.window_size:
            self._count += 1

        # Check if current value exceeds warn threshold
        over_warn = scalar > self.warn_threshold if self.warn_threshold > 0 else False

        # Compute fraction of window exceeding warn threshold
        if self.warn_threshold > 0 and self._count >= self.window_size // 2:
            valid = self._buffer[: self._count]
            valid = valid[~np.isnan(valid)]
            if len(valid) > 0:
                exceed_frac = np.mean(valid > self.warn_threshold)
                should_warn = exceed_frac >= self.warn_fraction
                return over_warn, should_warn

        return over_warn, False

    @property
    def window_mean(self) -> float:
        if self._buffer is None or self._count == 0:
            return 0.0
        valid = self._buffer[: self._count]
        valid = valid[~np.isnan(valid)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    @property
    def window_max(self) -> float:
        if self._buffer is None or self._count == 0:
            return 0.0
        valid = self._buffer[: self._count]
        valid = valid[~np.isnan(valid)]
        return float(np.max(valid)) if len(valid) > 0 else 0.0

    @property
    def is_ready(self) -> bool:
        """True when buffer has enough data for meaningful statistics."""
        return self._count >= self.window_size // 2

    def reset(self) -> None:
        self._buffer = None
        self._idx = 0
        self._count = 0
