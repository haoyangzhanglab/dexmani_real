"""Minimal ``module:symbol`` targets for the lazy-loader offline check.

Structural Protocol stand-ins only — no torch, no hardware, no SharedStorage.
The functional deterministic fake lives in ``deployment/fake.py`` (P6); these
exist so :mod:`dexmani_real.deployment.loader` can be exercised against a real
``importlib`` module before that lands.  Underscore-prefixed so ``run_all.py``
never treats this as a check.
"""

from __future__ import annotations

from typing import Any


class FakePolicyBackend:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def load(self) -> None:
        return None

    def reset(self, *, run_generation: int) -> None:
        return None

    def infer(self, model_input: Any) -> Any:
        return model_input

    def close(self) -> None:
        return None


class FakeObservationAdapter:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def encode(self, observation: Any) -> Any:
        return observation


class FakeActionAdapter:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def decode(self, raw_output: Any, *, context: Any) -> Any:
        return raw_output


def build_fake_backend(config: Any = None) -> FakePolicyBackend:
    """A function-style factory (mirrors ``package.module:build_backend``)."""
    return FakePolicyBackend(config)


class NotABackend:
    """Missing ``reset``/``infer``/``close`` -> must fail the Protocol check."""

    def load(self) -> None:
        return None
