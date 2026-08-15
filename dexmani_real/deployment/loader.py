"""Lazy backend/adapter loader (execution doc §58–§59).

A ``module:symbol`` target is imported and instantiated only when the inference
child calls these functions, so the parent process never imports torch or
initializes CUDA, and a checkpoint/model object never crosses ``spawn``.  The
factory is called with the resolved ``DeploymentConfig`` as a keyword so the
backend can receive ``checkpoint`` / ``model_config_path`` / ``device``; every
failure raises (fail closed) rather than falling back to a dummy safe mode.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, TypeVar

from dexmani_real.deployment.contracts import (
    ActionAdapter,
    ObservationAdapter,
    PolicyBackend,
)

if TYPE_CHECKING:
    from dexmani_real.deployment.config import DeploymentConfig

_ProtocolT = TypeVar("_ProtocolT")


def _split_target(target: str) -> tuple[str, str]:
    if not isinstance(target, str) or ":" not in target:
        raise ValueError(f"loader target must be 'module:symbol', got {target!r}")
    module_name, _, symbol = target.rpartition(":")
    if not module_name.strip() or not symbol.strip():
        raise ValueError(f"loader target must be 'module:symbol', got {target!r}")
    return module_name.strip(), symbol.strip()


def _load(
    target: str,
    protocol: type[_ProtocolT],
    kind: str,
    config: DeploymentConfig | None,
) -> _ProtocolT:
    module_name, symbol = _split_target(target)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - fail closed on any import error
        raise ImportError(f"failed to import {kind} module {module_name!r}: {exc}") from exc

    factory = getattr(module, symbol, None)
    if factory is None:
        raise ImportError(f"{kind} module {module_name!r} has no symbol {symbol!r}")
    try:
        instance = factory(config=config) if config is not None else factory()
    except Exception as exc:  # noqa: BLE001 - fail closed on any construction error
        raise TypeError(f"failed to instantiate {kind} {target!r}: {exc}") from exc

    if not isinstance(instance, protocol):
        raise TypeError(f"{kind} {target!r} does not satisfy {protocol.__name__}")
    return instance


def load_backend(target: str, *, config: DeploymentConfig | None = None) -> PolicyBackend:
    """Load a :class:`PolicyBackend` from ``module:symbol``."""
    return _load(target, PolicyBackend, "backend", config)


def load_observation_adapter(
    target: str, *, config: DeploymentConfig | None = None
) -> ObservationAdapter:
    """Load an :class:`ObservationAdapter` from ``module:symbol``."""
    return _load(target, ObservationAdapter, "observation adapter", config)


def load_action_adapter(
    target: str, *, config: DeploymentConfig | None = None
) -> ActionAdapter:
    """Load an :class:`ActionAdapter` from ``module:symbol``."""
    return _load(target, ActionAdapter, "action adapter", config)
