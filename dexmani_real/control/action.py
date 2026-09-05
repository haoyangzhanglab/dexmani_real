"""Backend-neutral, immutable action contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE


def _readonly_array(
    value: Any,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    name: str,
    allow_nan: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    invalid = np.isinf(array) if allow_nan else ~np.isfinite(array)
    if np.any(invalid):
        raise ValueError(f"{name} contains NaN/Inf")
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ActionCandidate:
    """One current-tick joint target proposed by a control source.

    The controller assigns a globally monotonic action ID while building the
    candidate; publication confirms it still belongs to the active
    ``run_generation``. ``scheduled_target_monotonic_ns`` preserves the policy
    grid endpoint for provenance. ``target_monotonic_ns`` is the worker delivery
    target chosen at publication, and ``valid_until_monotonic_ns`` is its hard
    expiry. An overdue policy endpoint is never relabeled as a fresh endpoint.
    """

    observation_id: int
    run_generation: int
    created_monotonic_ns: int
    scheduled_target_monotonic_ns: int
    target_monotonic_ns: int
    valid_until_monotonic_ns: int
    action_id: int = 0
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    is_hold: bool = False

    def __post_init__(self) -> None:
        if (
            min(
                self.observation_id,
                self.created_monotonic_ns,
                self.scheduled_target_monotonic_ns,
                self.target_monotonic_ns,
                self.valid_until_monotonic_ns,
            )
            <= 0
        ):
            raise ValueError("action identifiers/timestamps must be positive")
        if self.created_monotonic_ns > self.target_monotonic_ns:
            raise ValueError("delivery target precedes creation")
        if self.scheduled_target_monotonic_ns > self.created_monotonic_ns:
            raise ValueError("scheduled policy endpoint is not due yet")
        if self.target_monotonic_ns > self.valid_until_monotonic_ns:
            raise ValueError("action validity ends before target")
        if self.run_generation < 0 or self.action_id < 0:
            raise ValueError("action generation and ID must be non-negative")
        if self.arm_qpos is None and self.hand_qpos is None:
            raise ValueError("action candidate controls no actuator")
        if self.arm_qpos is not None:
            object.__setattr__(
                self,
                "arm_qpos",
                _readonly_array(
                    self.arm_qpos, ARM_JOINT_SHAPE, np.float64, name="arm_qpos"
                ),
            )
        if self.hand_qpos is not None:
            object.__setattr__(
                self,
                "hand_qpos",
                _readonly_array(
                    self.hand_qpos, HAND_JOINT_SHAPE, np.float64, name="hand_qpos"
                ),
            )
