"""P4: deployment config resolution + lazy backend/adapter loader.

Locks the frozen ``DeploymentConfig`` validation, the ``CLI > data > defaults``
precedence and stable SHA-256 identity of ``resolve_deployment_config``, and the
fail-closed ``module:symbol`` loader against structural Protocol stand-ins.
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.deployment.config import DeploymentConfig, resolve_deployment_config
from dexmani_real.deployment.contracts import (
    ActionAdapter,
    ObservationAdapter,
    PolicyBackend,
)
from dexmani_real.deployment.loader import (
    load_action_adapter,
    load_backend,
    load_observation_adapter,
)


def main() -> int:
    # ── DeploymentConfig validation ──
    cfg = DeploymentConfig()
    assert cfg.max_chunk_steps == 32
    assert cfg.hand_enabled is False
    assert cfg.device == "cpu"

    for bad in (
        {"observation_horizon": 0},
        {"max_chunk_steps": 0},
        {"inference_hz": 0.0},
        {"max_plan_age_s": -1.0},
    ):
        try:
            DeploymentConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"DeploymentConfig(**{bad}) must raise ValueError")

    # ── resolve: CLI > data > defaults, stable SHA-256 ──
    resolved = resolve_deployment_config(
        data={"backend_target": "pkg.mod:build_backend"},
        cli_overrides={"device": "cuda:0", "inference_hz": 20},
    )
    assert resolved.deployment.backend_target == "pkg.mod:build_backend"
    assert resolved.deployment.device == "cuda:0", "CLI must override file/data"
    assert resolved.deployment.inference_hz == 20.0
    assert resolved.deployment.hand_enabled is False, "untouched default preserved"
    assert len(resolved.sha256) == 64

    again = resolve_deployment_config(
        data={"backend_target": "pkg.mod:build_backend"},
        cli_overrides={"device": "cuda:0", "inference_hz": 20},
    )
    assert again.sha256 == resolved.sha256, "same input must yield the same digest"
    assert again.canonical_json == resolved.canonical_json

    # CLI None values mean "not supplied" and never mask file/data defaults.
    none_resolved = resolve_deployment_config(
        data={"backend_target": "pkg.mod:build_backend", "device": "cuda:1"},
        cli_overrides={"device": None},
    )
    assert none_resolved.deployment.device == "cuda:1", "None CLI must not mask the file value"

    # missing backend_target / unknown field fail closed
    try:
        resolve_deployment_config(data={})
    except ValueError:
        pass
    else:
        raise AssertionError("missing backend_target must raise ValueError")
    try:
        resolve_deployment_config(data={"backend_target": "a:b", "nope": 1})
    except TypeError:
        pass
    else:
        raise AssertionError("unknown field must raise TypeError")

    # ── loader: valid class + function factories ──
    backend = load_backend("_loader_fixtures:FakePolicyBackend", config=resolved.deployment)
    assert isinstance(backend, PolicyBackend)
    obs = load_observation_adapter("_loader_fixtures:FakeObservationAdapter")
    assert isinstance(obs, ObservationAdapter)
    act = load_action_adapter("_loader_fixtures:FakeActionAdapter", config=resolved.deployment)
    assert isinstance(act, ActionAdapter)

    backend2 = load_backend("_loader_fixtures:build_fake_backend", config=resolved.deployment)
    assert isinstance(backend2, PolicyBackend)
    assert backend2.config is resolved.deployment, "config must reach the factory"

    # ── loader: fail closed ──
    try:
        load_backend("_loader_fixtures:NotABackend")
    except TypeError:
        pass
    else:
        raise AssertionError("non-conforming backend must raise TypeError")
    try:
        load_backend("_loader_fixtures:NoSuchSymbol")
    except ImportError:
        pass
    else:
        raise AssertionError("missing symbol must raise ImportError")
    try:
        load_backend("_loader_fixtures")  # no colon
    except ValueError:
        pass
    else:
        raise AssertionError("target without colon must raise ValueError")
    try:
        load_backend("no_such_module_xyz:Foo")
    except ImportError:
        pass
    else:
        raise AssertionError("missing module must raise ImportError")

    print("check_deployment_loader: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
