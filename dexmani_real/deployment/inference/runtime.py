"""Typed model runtime contract owned by the inference child."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from dexmani_real.deployment.inference.observation import PolicyObservation


class PolicyRuntime(Protocol):
    """Model boundary owned by the inference process."""

    def warmup(self, *, samples: int) -> tuple[float, ...]: ...

    def reset_episode(self) -> None: ...

    def predict(self, observation: PolicyObservation) -> np.ndarray: ...

    def close(self) -> None: ...
