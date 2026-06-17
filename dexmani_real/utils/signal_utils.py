"""Signal processing utilities: EMA smoothing, interpolation, etc."""

from __future__ import annotations

__all__ = ["ema_smooth"]

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
