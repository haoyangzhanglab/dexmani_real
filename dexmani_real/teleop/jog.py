"""Pure Cartesian jog mapping from held operator keys."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from dexmani_real.runtime.operator_input import KeyboardInput

CARTESIAN_JOG_KEYS = frozenset(
    {"w", "s", "a", "d", "up", "down", "left", "right", "i", "k", "j", "l"}
)


def any_jog_key_held(active_keys: Iterable[str]) -> bool:
    """Check physical jog keys even when opposite increments cancel out."""
    return not CARTESIAN_JOG_KEYS.isdisjoint(active_keys)


def limit_cartesian_pose_lead(
    measured_pos: np.ndarray,
    measured_quat_wxyz: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    *,
    max_position_lead_m: float,
    max_rotation_lead_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound world-frame position and shortest rotation lead independently.

    Inputs are finite poses with wxyz quaternions and positive lead caps.
    Workspace clipping belongs to the caller and precedes this limit.
    """
    bounded_pos = np.asarray(target_pos, dtype=np.float64).copy()
    position_lead = bounded_pos - measured_pos
    position_lead_norm = float(np.linalg.norm(position_lead))
    if position_lead_norm > max_position_lead_m:
        bounded_pos = measured_pos + position_lead * (max_position_lead_m / position_lead_norm)

    measured_rotation = Rotation.from_quat(measured_quat_wxyz, scalar_first=True)
    target_rotation = Rotation.from_quat(target_quat_wxyz, scalar_first=True)
    rotation_lead = (target_rotation * measured_rotation.inv()).as_rotvec()
    rotation_lead_norm = float(np.linalg.norm(rotation_lead))
    if rotation_lead_norm > max_rotation_lead_rad:
        target_rotation = (
            Rotation.from_rotvec(rotation_lead * (max_rotation_lead_rad / rotation_lead_norm))
            * measured_rotation
        )
    return bounded_pos, target_rotation.as_quat(scalar_first=True)


def compute_cartesian_jog_delta(
    keys: KeyboardInput,
    delta_pos: float,
    delta_rpy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map held keys to EEF position and rotation increments."""
    dx = np.zeros(3, dtype=np.float64)
    if keys.is_pressed("w"):
        dx[0] += delta_pos
    if keys.is_pressed("s"):
        dx[0] -= delta_pos
    if keys.is_pressed("a"):
        dx[1] -= delta_pos
    if keys.is_pressed("d"):
        dx[1] += delta_pos
    if keys.is_pressed("up"):
        dx[2] += delta_pos
    if keys.is_pressed("down"):
        dx[2] -= delta_pos
    dx_norm = float(np.linalg.norm(dx))
    if dx_norm > delta_pos:
        dx *= delta_pos / dx_norm

    drpy = np.zeros(3, dtype=np.float64)
    if keys.is_pressed("left"):
        drpy[0] += delta_rpy
    if keys.is_pressed("right"):
        drpy[0] -= delta_rpy
    if keys.is_pressed("i"):
        drpy[1] += delta_rpy
    if keys.is_pressed("k"):
        drpy[1] -= delta_rpy
    if keys.is_pressed("j"):
        drpy[2] -= delta_rpy
    if keys.is_pressed("l"):
        drpy[2] += delta_rpy
    drpy_norm = float(np.linalg.norm(drpy))
    if drpy_norm > delta_rpy:
        drpy *= delta_rpy / drpy_norm

    return dx, drpy
