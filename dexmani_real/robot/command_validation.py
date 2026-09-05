"""Final physical target guards for arm and hand hardware workers."""

from __future__ import annotations

import numpy as np


def check_worker_arm_target(
    target_qpos_rad: np.ndarray,
    *,
    previous_target_qpos_rad: np.ndarray,
    joint_limit_lower_rad: np.ndarray,
    joint_limit_upper_rad: np.ndarray,
    max_command_jump_rad: float,
) -> str | None:
    """Return why an arm target cannot cross the SDK boundary."""
    target = np.asarray(target_qpos_rad, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        return "non-finite target"
    if np.any(target < joint_limit_lower_rad) or np.any(target > joint_limit_upper_rad):
        return "joint limit violation"
    if np.any(
        np.abs(target - previous_target_qpos_rad) > float(max_command_jump_rad)
    ):
        return "command jump limit violation"
    return None


def check_worker_hand_target(
    target_qpos_rad: np.ndarray,
    *,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
) -> str | None:
    """Return why a hand target cannot cross the SDK boundary."""
    target = np.asarray(target_qpos_rad, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        return "non-finite target"
    if np.any(target < mechanical_lower_rad - 1e-12) or np.any(
        target > mechanical_upper_rad + 1e-12
    ):
        return "mechanical joint limit violation"
    return None
