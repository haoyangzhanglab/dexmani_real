"""Path processing utilities — densification, interpolation, smoothing."""

from __future__ import annotations

import numpy as np


def interpolate_waypoints(path: np.ndarray, max_step: float) -> np.ndarray:
    """Linearly densify a sparse joint path so each step ≤ max_step rad.

    Args:
        path:     (W, J) array of joint-space waypoints.
        max_step: maximum joint change per step (radians).

    Returns:
        (D, J) dense path (D ≥ W).
    """
    if len(path) <= 1:
        return path.astype(np.float64)
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)
