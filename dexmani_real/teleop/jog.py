"""Pure Cartesian jog mapping from held operator keys."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.planning.kinematics.pose import Pose, quat_multiply

CARTESIAN_JOG_KEYS = frozenset(
    {"w", "s", "a", "d", "up", "down", "left", "right", "i", "k", "j", "l"}
)


def any_jog_key_held(active_keys: Iterable[str]) -> bool:
    """Check physical jog keys even when opposite increments cancel out."""
    return not CARTESIAN_JOG_KEYS.isdisjoint(active_keys)


def propose_cartesian_jog_pose(
    command_pose: Pose,
    dx: np.ndarray,
    drpy: np.ndarray,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
) -> tuple[Pose, np.ndarray, bool]:
    """Project a jog proposal without advancing the dispatched target reference."""
    desired_position = command_pose.p + dx
    position = np.clip(desired_position, position_lower, position_upper)
    rotating = bool(np.any(drpy != 0.0))
    changed = rotating or not np.array_equal(position, command_pose.p)
    quaternion = command_pose.q.copy()
    if rotating:
        delta_quaternion = Rotation.from_euler("xyz", drpy).as_quat()[[3, 0, 1, 2]]
        quaternion = quat_multiply(delta_quaternion, quaternion)
    return Pose(p=position, q=quaternion), desired_position, changed


def compute_cartesian_jog_delta(
    pressed_keys: Iterable[str],
    delta_pos: float,
    delta_rpy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map held keys to EEF position and rotation increments."""
    keys = frozenset(pressed_keys)
    dx = np.zeros(3, dtype=np.float64)
    if "w" in keys:
        dx[0] += delta_pos
    if "s" in keys:
        dx[0] -= delta_pos
    if "a" in keys:
        dx[1] -= delta_pos
    if "d" in keys:
        dx[1] += delta_pos
    if "up" in keys:
        dx[2] += delta_pos
    if "down" in keys:
        dx[2] -= delta_pos
    dx_norm = float(np.linalg.norm(dx))
    if dx_norm > delta_pos:
        dx *= delta_pos / dx_norm

    drpy = np.zeros(3, dtype=np.float64)
    if "left" in keys:
        drpy[0] += delta_rpy
    if "right" in keys:
        drpy[0] -= delta_rpy
    if "i" in keys:
        drpy[1] += delta_rpy
    if "k" in keys:
        drpy[1] -= delta_rpy
    if "j" in keys:
        drpy[2] -= delta_rpy
    if "l" in keys:
        drpy[2] += delta_rpy
    drpy_norm = float(np.linalg.norm(drpy))
    if drpy_norm > delta_rpy:
        drpy *= delta_rpy / drpy_norm

    return dx, drpy
