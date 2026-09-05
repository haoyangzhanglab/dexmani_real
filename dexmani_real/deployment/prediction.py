"""Immutable inference output transported to the policy executor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.ipc.schema import MAX_POLICY_ACTION_DIM, MAX_PREDICTION_STEPS


@dataclass(frozen=True)
class Prediction:
    """One immutable, flat policy prediction transported to execution.

    ``actions`` remains in the representation selected by the validated
    ``PolicySpec``. The inference process never slices it; the executor is its
    sole decoder.
    """

    run_generation: int
    source_monotonic_ns: int
    logical_step_monotonic_ns: int
    actions: np.ndarray

    def __post_init__(self) -> None:
        if type(self.run_generation) is not int or self.run_generation < 0:
            raise ValueError("run_generation must be a non-negative integer")
        for name in ("source_monotonic_ns", "logical_step_monotonic_ns"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if not (
            0 < self.source_monotonic_ns <= self.logical_step_monotonic_ns
        ):
            raise ValueError("prediction timestamps must satisfy source <= logical step")
        if max(
            self.run_generation,
            self.source_monotonic_ns,
            self.logical_step_monotonic_ns,
        ) > int(np.iinfo(np.uint64).max):
            raise ValueError("prediction metadata exceeds uint64")

        raw = np.asarray(self.actions)
        if raw.dtype != np.dtype(np.float64):
            raise TypeError(f"actions must have dtype float64, got {raw.dtype}")
        if raw.ndim != 2:
            raise ValueError(f"actions must be [N, D], got {raw.shape}")
        if not 0 < raw.shape[0] <= MAX_PREDICTION_STEPS:
            raise ValueError(
                f"actions N must be in [1, {MAX_PREDICTION_STEPS}], got {raw.shape[0]}"
            )
        if not 0 < raw.shape[1] <= MAX_POLICY_ACTION_DIM:
            raise ValueError(
                f"actions D must be in [1, {MAX_POLICY_ACTION_DIM}], got {raw.shape[1]}"
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("actions contain NaN/Inf")
        actions = np.array(raw, dtype=np.float64, copy=True, order="C")
        actions.flags.writeable = False
        object.__setattr__(self, "actions", actions)

    @property
    def num_steps(self) -> int:
        """Return the fixed prediction horizon carried by this instance."""
        return int(self.actions.shape[0])
