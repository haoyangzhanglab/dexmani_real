"""Pure Cartesian jog mapping from held operator keys."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dexmani_real.runtime.operator_input import KeyboardInput

CARTESIAN_JOG_KEYS = frozenset(
    {"w", "s", "a", "d", "up", "down", "left", "right", "i", "k", "j", "l"}
)


def any_jog_key_held(active_keys: Iterable[str]) -> bool:
    """Check physical jog keys even when opposite increments cancel out."""
    return not CARTESIAN_JOG_KEYS.isdisjoint(active_keys)


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
