"""NumPy adapter from Real observations to the public DexMani Policy runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.deployment.inference.observation import PolicyObservation


class DexManiPolicyAdapter:
    """Adapt one already-loaded public Policy runtime to Real's typed contract.

    This class owns no checkpoint, Hydra, EMA, normalizer, Torch, or device
    behavior. The policy runner loads those through the Policy public API.
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
        actions = np.asarray(self._policy.predict(observation.arrays), dtype=np.float64)
        expected = (self.spec.n_action_steps, self.spec.control_action_dim)
        if actions.shape != expected:
            raise ValueError(f"Policy action shape {actions.shape} conflicts with {expected}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Policy actions contain NaN/Inf")
        return actions

    def close(self) -> None:
        self._policy.close()
