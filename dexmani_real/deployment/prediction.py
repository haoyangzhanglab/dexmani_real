"""Process-local inference result; the inference boundary validates actions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """One full action chunk with its scheduling timestamps and diagnostics."""

    run_generation: int
    source_monotonic_ns: int
    logical_step_monotonic_ns: int
    actions: np.ndarray
    inference_latency_ms: float
    observation_age_ms: float
    observation_skew_ms: float

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[0])
