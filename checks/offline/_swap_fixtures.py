"""A second deterministic backend for the backend-swap verification (P12).

``ZeroTargetPolicyBackend`` is deliberately distinct from
``deployment.fake.FakePolicyBackend`` (which holds-at-current plus an offset):
this one ignores the observation and emits a fixed hold-at-zero chunk with no
hand command. Swapping one for the other is a pure ``backend_target`` config
change — the deployment core (loader -> worker -> coordinator) runs both with
zero code changes (§100). Underscore-prefixed so ``run_all.py`` never treats
this as a check.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_HORIZON = 4


class ZeroTargetPolicyBackend:
    """Deterministic hold-at-zero policy: emits ``zeros([N,7])``, no hand."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def load(self) -> None:
        return None

    def reset(self, *, run_generation: int) -> None:
        return None

    def infer(self, model_input: Any) -> dict[str, np.ndarray | None]:
        return {
            "arm_qpos": np.zeros((_HORIZON, 7), dtype=np.float64),
            "hand_qpos": None,
        }

    def close(self) -> None:
        return None
