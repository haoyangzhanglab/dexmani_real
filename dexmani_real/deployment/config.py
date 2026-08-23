"""Immutable learned-policy deployment configuration.

Mirrors :func:`dexmani_real.config.runtime.resolve_runtime_config` for the
deployment namespace: a frozen :class:`DeploymentConfig` template plus a
``CLI > file/data > defaults`` resolver that stamps a stable canonical-JSON
SHA-256 identity for provenance.  Model-internal parameters (transformer depth,
hidden dim, diffusion schedule, etc.) never appear here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from dexmani_real.deployment.observation import parse_observation_fields
from dexmani_real.utils.schema import SUPPORTED_POINT_CLOUD_COUNTS

_POSITIVE_FLOAT_FIELDS = (
    "inference_hz",
    "max_observation_age_s",
    "max_plan_age_s",
    "max_command_silence_s",
    "action_validity_s",
)


@dataclass(frozen=True)
class DeploymentConfig:
    """Frozen learned-policy deployment parameters.

    ``runtime_target`` names the ``module:symbol`` :class:`PolicyRuntime`
    factory resolved by :mod:`dexmani_real.deployment.worker`; ``checkpoint`` /
    ``model_config_path`` / ``device`` and the explicit ``observation_fields``
    contract are the model-facing values that cross the deployment boundary
    (everything else model-internal stays in the model repository).
    """

    runtime_target: str = ""
    checkpoint: str | None = None
    model_config_path: str | None = None
    device: str = "cpu"
    inference_hz: float = 10.0
    observation_horizon: int = 4
    max_observation_age_s: float = 0.5
    max_plan_age_s: float = 1.0
    max_command_silence_s: float = 2.0
    action_validity_s: float = 0.5
    hand_enabled: bool = False
    # Comma-separated observation keys; the default preserves the joint-only contract.
    observation_fields: str = "arm_qpos,hand_qpos"
    pointcloud_num_points: int = 1024

    def __post_init__(self) -> None:
        if self.observation_horizon <= 0:
            raise ValueError("observation_horizon must be positive")
        for name in _POSITIVE_FLOAT_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        parse_observation_fields(self.observation_fields)
        if (
            isinstance(self.pointcloud_num_points, bool)
            or self.pointcloud_num_points not in SUPPORTED_POINT_CLOUD_COUNTS
        ):
            raise ValueError(
                "pointcloud_num_points must be one of "
                f"{sorted(SUPPORTED_POINT_CLOUD_COUNTS)}"
            )


DEFAULT_DEPLOYMENT_CONFIG = DeploymentConfig()


def _coerce_value(template: Any, raw: Any, path: str) -> Any:
    """Coerce one override to the template field's scalar type."""
    if template is None:
        if raw is not None and not isinstance(raw, str):
            raise TypeError(
                f"deployment config field {path!r} must be a string or null"
            )
        return raw
    if isinstance(template, bool):
        if not isinstance(raw, bool):
            raise TypeError(f"deployment config field {path!r} must be a boolean")
        return raw
    if isinstance(template, int):
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"deployment config field {path!r} must be an integer")
        return int(raw)
    if isinstance(template, float):
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise TypeError(f"deployment config field {path!r} must be a finite number")
        return float(raw)
    if isinstance(template, str):
        if not isinstance(raw, str):
            raise TypeError(f"deployment config field {path!r} must be a string")
        return raw
    raise TypeError(
        f"unsupported deployment config field type {type(template).__name__}"
    )


def _merge(base: DeploymentConfig, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay *overrides* onto *base*, rejecting unknown fields."""
    known = {field.name for field in fields(DeploymentConfig)}
    unknown = {str(key) for key in overrides} - known
    if unknown:
        raise TypeError(f"unknown deployment config field(s): {sorted(unknown)}")
    result: dict[str, Any] = {}
    for field in fields(DeploymentConfig):
        name = field.name
        if name in overrides:
            result[name] = _coerce_value(getattr(base, name), overrides[name], name)
        else:
            result[name] = getattr(base, name)
    return result


@dataclass(frozen=True)
class ResolvedDeploymentConfig:
    """A validated deployment snapshot plus its canonical identity."""

    deployment: DeploymentConfig
    canonical_json: str
    sha256: str


def resolve_deployment_config(
    *,
    yaml_path: str | Path | None = None,
    data: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ResolvedDeploymentConfig:
    """Resolve ``CLI > file/data > defaults`` without mutating the template.

    Accepts either a flat mapping of deployment fields or a YAML document with a
    top-level ``deployment:`` section.  Raises when ``runtime_target`` is
    absent; it is the required entry point (the worker cannot run without it).
    """
    sources = [yaml_path is not None, data is not None]
    if sum(sources) > 1:
        raise ValueError("provide at most one of yaml_path or data")

    file_overrides: Mapping[str, Any] = {}
    if yaml_path is not None:
        config_path = Path(yaml_path)
        if config_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("deployment config path must use a .yaml or .yml suffix")
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise TypeError("deployment config root must be a mapping")
        file_overrides = loaded
    elif data is not None:
        file_overrides = data

    if isinstance(file_overrides.get("deployment"), Mapping):
        file_overrides = file_overrides["deployment"]

    merged = _merge(DEFAULT_DEPLOYMENT_CONFIG, file_overrides)
    # Unset CLI values do not override file or data defaults.
    cli = {
        str(key): value
        for key, value in (cli_overrides or {}).items()
        if value is not None
    }
    merged = _merge(DeploymentConfig(**merged), cli)

    config = DeploymentConfig(**merged)
    if not config.runtime_target.strip():
        raise ValueError("deployment runtime_target must be provided")

    canonical = {
        field.name: getattr(config, field.name) for field in fields(DeploymentConfig)
    }
    canonical_json = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return ResolvedDeploymentConfig(
        deployment=config, canonical_json=canonical_json, sha256=digest
    )
