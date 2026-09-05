"""NumPy adapter from Real observations to the public DexMani Policy runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.deployment.observation import PolicyObservation


class DexManiPolicyRuntime:
    """Adapt one already-loaded public Policy runtime to Real's typed contract.

    This class owns no checkpoint, Hydra, EMA, normalizer, Torch, or device
    behavior. The inference worker loads those through the Policy public API.
    """

    def __init__(self, loaded_policy: Any, expected_spec: Any) -> None:
        if loaded_policy.spec != expected_spec:
            raise RuntimeError("PolicySpec changed between inspect and load")
        self._policy = loaded_policy
        self.spec = expected_spec

    def warmup(self, *, samples: int) -> tuple[float, ...]:
        return self._policy.warmup(samples=samples)

    def reset_episode(self) -> None:
        self._policy.reset_episode()

    def predict(self, observation: PolicyObservation) -> np.ndarray:
        return self._policy.predict(observation.arrays)

    def close(self) -> None:
        self._policy.close()
