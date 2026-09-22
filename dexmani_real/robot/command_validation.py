"""Final physical target guards for arm and hand hardware workers.

Workers own only the HARD boundary at the SDK fence: finiteness and the
physical joint limits. The soft command-jump bound is owned once by
``robot/projection.py`` at production time; a worker never re-rejects the
same soft threshold (no duplicate validation of one constraint).
"""

from __future__ import annotations

import numpy as np


def check_worker_arm_target(
    target_qpos_rad: np.ndarray,
    *,
    joint_limit_lower_rad: np.ndarray,
    joint_limit_upper_rad: np.ndarray,
) -> str | None:
    """Return why an arm target cannot cross the SDK boundary (hard limits)."""
    target = np.asarray(target_qpos_rad, dtype=np.float64)
    if target.shape != (7,):
        return "invalid target shape"
    if not np.all(np.isfinite(target)):
        return "non-finite target"
    if np.any(target < joint_limit_lower_rad) or np.any(target > joint_limit_upper_rad):
        return "joint limit violation"
    return None


def check_worker_hand_target(
    target_qpos_rad: np.ndarray,
    *,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
) -> str | None:
    """Return why a hand target cannot cross the SDK boundary."""
    target = np.asarray(target_qpos_rad, dtype=np.float64)
    if target.shape != (12,):
        return "invalid target shape"
    if not np.all(np.isfinite(target)):
        return "non-finite target"
    if np.any(target < mechanical_lower_rad - 1e-12) or np.any(
        target > mechanical_upper_rad + 1e-12
    ):
        return "mechanical joint limit violation"
    return None
