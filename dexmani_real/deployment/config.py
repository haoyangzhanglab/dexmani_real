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
from typing import Any, cast

import yaml

from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.config.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.deployment.artifact import (
    PolicyArtifactContract,
    ResolvedPolicyArtifact,
)
from dexmani_real.deployment.observation import parse_observation_fields
from dexmani_real.ipc.schema import SUPPORTED_POINT_CLOUD_COUNTS

_POSITIVE_FLOAT_FIELDS = (
    "inference_hz",
    "max_input_age_s",
    "max_observation_skew_s",
    "max_grid_lag_s",
    "max_plan_age_s",
    "max_source_to_command_age_s",
    "command_lead_s",
    "max_command_silence_s",
    "action_validity_s",
    "first_command_timeout_s",
)

FIXED_POLICY_RUNTIME_TARGET = (
    "dexmani_real.integrations.dexmani_policy:DexManiPolicyRuntime"
)

# The artifact is authoritative for model-facing fields.  A deployment YAML
# may state one as an expectation, but may never change it.
_ARTIFACT_OWNED_FIELDS = frozenset(
    {
        "runtime_target",
        "checkpoint",
        "task_name",
        "action_key",
        "observation_horizon",
        "hand_enabled",
        "observation_fields",
        "pointcloud_num_points",
    }
)
_REAL_OWNED_DEPLOYMENT_FIELDS = frozenset(
    {
        "inference_seed",
        "inference_hz",
        "max_input_age_s",
        "max_observation_skew_s",
        "max_grid_lag_s",
        "max_plan_age_s",
        "max_source_to_command_age_s",
        "command_lead_s",
        "max_command_silence_s",
        "action_validity_s",
        "first_command_timeout_s",
    }
)


@dataclass(frozen=True)
class DeploymentConfig:
    """Frozen learned-policy deployment parameters.

    ``runtime_target`` names the ``module:symbol`` :class:`PolicyRuntime`
    factory resolved by :mod:`dexmani_real.deployment.worker`; ``checkpoint`` /
    ``device`` and the explicit ``observation_fields``
    contract are the model-facing values that cross the deployment boundary
    (everything else model-internal stays in the model repository).
    """

    runtime_target: str = ""
    checkpoint: str | None = None
    device: str = "cpu"
    # Operator-declared semantic task identity; the DexMani Policy adapter
    # requires an exact match with the checkpoint training-data contract.
    task_name: str = ""
    # Operator-declared action contract; validated against the checkpoint's
    # train_params at load time (mismatch fails closed, never coerces).
    action_key: str = "action"
    # Seeds model construction and the subsequent diffusion RNG stream once in
    # the inference worker, matching the Policy evaluation convention.
    inference_seed: int = 0
    inference_hz: float = 10.0
    observation_horizon: int = 2
    max_input_age_s: float = 0.15
    max_observation_skew_s: float = 0.10
    max_grid_lag_s: float = 0.08
    max_plan_age_s: float = 1.0
    max_source_to_command_age_s: float = 0.75
    command_lead_s: float = 0.01
    max_command_silence_s: float = 2.0
    action_validity_s: float = 0.5
    # Abort a RUNNING run that never produced its first command within this
    # window (the command-to-command silence watchdog is exempt until then).
    first_command_timeout_s: float = 5.0
    hand_enabled: bool = False
    # Comma-separated observation keys; the runtime supports point-cloud policies.
    observation_fields: str = "arm_qpos,hand_qpos,point_cloud"
    pointcloud_num_points: int = 1024

    def __post_init__(self) -> None:
        if (
            isinstance(self.inference_seed, bool)
            or not isinstance(self.inference_seed, int)
            or not 0 <= self.inference_seed <= 2**31 - 1
        ):
            raise ValueError("inference_seed must be an integer in [0, 2**31 - 1]")
        if (
            isinstance(self.observation_horizon, bool)
            or not isinstance(self.observation_horizon, int)
            or self.observation_horizon <= 0
        ):
            raise ValueError("observation_horizon must be a positive integer")
        if self.action_key not in ("action", "action_ee"):
            raise ValueError("action_key must be 'action' or 'action_ee'")
        if not isinstance(self.task_name, str):
            raise TypeError("task_name must be a string")
        if self.task_name != self.task_name.strip():
            raise ValueError("task_name must not have leading or trailing whitespace")
        for name in _POSITIVE_FLOAT_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.command_lead_s >= self.max_source_to_command_age_s:
            raise ValueError(
                "command_lead_s must be smaller than max_source_to_command_age_s"
            )
        if not isinstance(self.hand_enabled, bool):
            raise TypeError("hand_enabled must be a boolean")
        parse_observation_fields(self.observation_fields)
        if (
            isinstance(self.pointcloud_num_points, bool)
            or self.pointcloud_num_points not in SUPPORTED_POINT_CLOUD_COUNTS
        ):
            raise ValueError(
                "pointcloud_num_points must be one of "
                f"{sorted(SUPPORTED_POINT_CLOUD_COUNTS)}"
            )


@dataclass(frozen=True)
class H4ExecuteBounds:
    """Explicit, one-publication physical-execution bounds for the H4 gate.

    This is operator-owned runtime intent, never a model artifact or YAML
    default.  H4 deliberately supports exactly one coupled publication; a
    later level must introduce and review a different contract rather than
    silently widening this bound.
    """

    max_published_endpoints: int
    acknowledgement_timeout_s: float
    max_running_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_published_endpoints, bool)
            or not isinstance(self.max_published_endpoints, int)
            or self.max_published_endpoints != 1
        ):
            raise ValueError(
                "H4 execute requires max_published_endpoints exactly equal to 1"
            )
        for name in ("acknowledgement_timeout_s", "max_running_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.acknowledgement_timeout_s > self.max_running_s:
            raise ValueError("acknowledgement_timeout_s must not exceed max_running_s")


@dataclass(frozen=True)
class TaskExecuteBounds:
    """Explicit bounds for one independently reviewed full policy episode.

    This profile is intentionally separate from H4: it permits more than one
    acknowledged coupled endpoint, but remains bounded by both endpoint count
    and B-relative wall time.
    """

    max_published_endpoints: int
    acknowledgement_timeout_s: float
    max_running_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_published_endpoints, bool)
            or not isinstance(self.max_published_endpoints, int)
            or self.max_published_endpoints <= 1
        ):
            raise ValueError(
                "task execute requires max_published_endpoints greater than 1"
            )
        for name in ("acknowledgement_timeout_s", "max_running_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.acknowledgement_timeout_s > self.max_running_s:
            raise ValueError("acknowledgement_timeout_s must not exceed max_running_s")


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """Deployment config plus resolved sensor semantics visible to the model.

    The deployment fields remain available through attribute forwarding so
    existing ``PolicyRuntime`` factories keep a narrow, read-only config
    surface. Sensor identity is populated by the lifecycle from the exact
    config passed to the realtime point-cloud worker, never by user YAML.
    """

    deployment: DeploymentConfig
    control_dt_s: float = 0.0
    point_cloud_frame: str = ""
    point_cloud_color_source: str = ""
    point_cloud_policy_id: str = ""
    point_cloud_config_sha256: str = ""
    point_cloud_table_plane_abcd_json: str = ""
    point_cloud_sampling: str = ""
    point_cloud_transform: str = ""
    arm_max_delta_rad_per_tick: float | None = None
    hand_max_delta_rad_per_tick: float = 0.3
    endpoint_delta_tolerance_rad: float = policy_defaults.endpoint_delta_tolerance_rad
    artifact: ResolvedPolicyArtifact | None = None
    execution_mode: str = "shadow"
    hand_acknowledged: bool = False
    h4_execute_bounds: H4ExecuteBounds | None = None
    task_execute_bounds: TaskExecuteBounds | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.deployment, DeploymentConfig):
            raise TypeError("deployment must be a DeploymentConfig")
        if self.execution_mode not in {"shadow", "execute", "task"}:
            raise ValueError("execution_mode must be 'shadow', 'execute', or 'task'")
        if not isinstance(self.hand_acknowledged, bool):
            raise TypeError("hand_acknowledged must be a boolean")
        if self.execution_mode == "execute" and self.h4_execute_bounds is None:
            raise ValueError("execute requires explicit H4 execute bounds")
        if self.h4_execute_bounds is not None and not isinstance(
            self.h4_execute_bounds, H4ExecuteBounds
        ):
            raise TypeError("h4_execute_bounds must be H4ExecuteBounds or None")
        if self.execution_mode == "shadow" and self.h4_execute_bounds is not None:
            raise ValueError("shadow must not carry H4 execute bounds")
        if self.execution_mode != "execute" and self.h4_execute_bounds is not None:
            raise ValueError("only H4 execute may carry H4 execute bounds")
        if self.execution_mode == "task" and self.task_execute_bounds is None:
            raise ValueError("task execute requires explicit task execute bounds")
        if self.task_execute_bounds is not None and not isinstance(
            self.task_execute_bounds, TaskExecuteBounds
        ):
            raise TypeError("task_execute_bounds must be TaskExecuteBounds or None")
        if self.execution_mode != "task" and self.task_execute_bounds is not None:
            raise ValueError("only task execute may carry task execute bounds")
        if (
            self.execution_mode in {"execute", "task"}
            and not self.deployment.hand_enabled
        ):
            raise ValueError("physical execute requires a hand-enabled artifact")
        if self.execution_mode in {"execute", "task"} and not self.hand_acknowledged:
            raise ValueError(
                "physical execute requires explicit --hand acknowledgement"
            )
        if self.artifact is not None:
            if not isinstance(self.artifact, ResolvedPolicyArtifact):
                raise TypeError("artifact must be a ResolvedPolicyArtifact")
            allocation = self.artifact.allocation_contract
            expected = {
                "runtime_target": FIXED_POLICY_RUNTIME_TARGET,
                "checkpoint": str(self.artifact.checkpoint_path),
                "task_name": allocation.task_name,
                "action_key": allocation.action_key,
                "observation_horizon": allocation.n_obs_steps,
                "hand_enabled": allocation.requires_hand,
                "observation_fields": ",".join(allocation.observation_fields),
                "pointcloud_num_points": allocation.point_cloud_num_points,
            }
            for name, value in expected.items():
                if getattr(self.deployment, name) != value:
                    raise ValueError(
                        f"artifact-owned deployment field {name!r} does not match "
                        "the resolved artifact"
                    )
        if not math.isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and positive")
        if self.arm_max_delta_rad_per_tick is not None and (
            not math.isfinite(self.arm_max_delta_rad_per_tick)
            or self.arm_max_delta_rad_per_tick <= 0.0
        ):
            raise ValueError(
                "arm_max_delta_rad_per_tick must be finite and positive or None"
            )
        if (
            not math.isfinite(self.hand_max_delta_rad_per_tick)
            or self.hand_max_delta_rad_per_tick <= 0.0
        ):
            raise ValueError("hand_max_delta_rad_per_tick must be finite and positive")
        if (
            isinstance(self.endpoint_delta_tolerance_rad, bool)
            or not math.isfinite(self.endpoint_delta_tolerance_rad)
            or self.endpoint_delta_tolerance_rad < 0.0
        ):
            raise ValueError(
                "endpoint_delta_tolerance_rad must be finite and non-negative"
            )
        if "point_cloud" in parse_observation_fields(
            self.deployment.observation_fields
        ):
            names = (
                "point_cloud_frame",
                "point_cloud_color_source",
                "point_cloud_policy_id",
                "point_cloud_config_sha256",
                "point_cloud_table_plane_abcd_json",
                "point_cloud_sampling",
                "point_cloud_transform",
            )
            if any(not getattr(self, name) for name in names):
                raise ValueError("point-cloud runtime semantics must be complete")

    def __getattr__(self, name: str) -> Any:
        try:
            deployment = object.__getattribute__(self, "deployment")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(deployment, name)

    @property
    def physical_execute_bounds(self) -> H4ExecuteBounds | TaskExecuteBounds | None:
        if self.execution_mode == "execute":
            return self.h4_execute_bounds
        if self.execution_mode == "task":
            return self.task_execute_bounds
        return None


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


@dataclass(frozen=True)
class ResolvedPolicyRuntimeConfig:
    """Pickle-safe artifact/Real projection for isolated model preflight."""

    runtime: PolicyRuntimeConfig
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
        sibling_keys = {str(key) for key in file_overrides} - {"deployment"}
        if sibling_keys:
            raise TypeError(
                "deployment config wrapper has unknown sibling field(s): "
                f"{sorted(sibling_keys)}"
            )
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


def resolve_policy_runtime_config(
    *,
    artifact: ResolvedPolicyArtifact,
    runtime_config: Any,
    yaml_path: str | Path | None = None,
    data: Mapping[str, Any] | None = None,
    device: str | None = None,
    inference_seed: int | None = None,
    execution_mode: str = "shadow",
    hand_acknowledged: bool = False,
    h4_execute_bounds: H4ExecuteBounds | None = None,
    task_execute_bounds: TaskExecuteBounds | None = None,
) -> ResolvedPolicyRuntimeConfig:
    """Project one immutable preflight config from an artifact and Real config.

    Model-facing fields come only from the sidecar receipt.  The optional
    deployment YAML can set Real timing values or repeat artifact values
    exactly as an auditable expectation.  It cannot redirect a checkpoint or
    runtime implementation.
    """
    sources = [yaml_path is not None, data is not None]
    if sum(sources) > 1:
        raise ValueError("provide at most one of yaml_path or data")
    raw: Mapping[str, Any] = {}
    if yaml_path is not None:
        path = Path(yaml_path)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("deployment config path must use a .yaml or .yml suffix")
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise TypeError("deployment config root must be a mapping")
        raw = loaded
    elif data is not None:
        raw = data
    if isinstance(raw.get("deployment"), Mapping):
        sibling_keys = {str(key) for key in raw} - {"deployment"}
        if sibling_keys:
            raise TypeError(
                "deployment config wrapper has unknown sibling field(s): "
                f"{sorted(sibling_keys)}"
            )
        raw = raw["deployment"]
    if not isinstance(raw, Mapping):
        raise TypeError("deployment config root must be a mapping")

    allocation = artifact.allocation_contract
    expected_artifact = {
        "runtime_target": FIXED_POLICY_RUNTIME_TARGET,
        "checkpoint": str(artifact.checkpoint_path),
        "task_name": allocation.task_name,
        "action_key": allocation.action_key,
        "observation_horizon": allocation.n_obs_steps,
        "hand_enabled": allocation.requires_hand,
        "observation_fields": ",".join(allocation.observation_fields),
        "pointcloud_num_points": allocation.point_cloud_num_points,
    }
    allowed = _ARTIFACT_OWNED_FIELDS | _REAL_OWNED_DEPLOYMENT_FIELDS
    unknown = {str(key) for key in raw} - allowed
    if unknown:
        raise TypeError(
            "deployment config may contain only artifact expectations or "
            f"Real-owned fields; unknown={sorted(unknown)}"
        )
    base = DeploymentConfig(**cast(dict[str, Any], expected_artifact))
    real_overrides = {
        key: value
        for key, value in raw.items()
        if str(key) in _REAL_OWNED_DEPLOYMENT_FIELDS
    }
    merged = _merge(base, real_overrides)
    for name, expected in expected_artifact.items():
        if name not in raw:
            continue
        actual = _coerce_value(getattr(base, name), raw[name], name)
        if actual != expected:
            raise ValueError(
                f"artifact-owned deployment field {name!r} is an expectation "
                f"and must equal {expected!r}"
            )
    if device is not None:
        merged = _merge(DeploymentConfig(**merged), {"device": device})
    if inference_seed is not None:
        merged = _merge(DeploymentConfig(**merged), {"inference_seed": inference_seed})
    deployment = DeploymentConfig(**merged)

    control_dt_s = 1.0 / float(runtime_config.policy.control_hz)
    if not math.isclose(
        control_dt_s, allocation.control_dt_s, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("artifact control_dt_s does not match Real policy.control_hz")
    if runtime_config.pointcloud.num_points != allocation.point_cloud_num_points:
        raise ValueError(
            "artifact point_cloud_num_points does not match Real pointcloud config"
        )
    table = runtime_config.environment.table
    if table.enabled:
        plane = tuple(float(value) for value in table.plane_abcd)
        point_cloud_table_plane_abcd_json = json.dumps(
            plane, separators=(",", ":"), allow_nan=False
        )
    else:
        point_cloud_table_plane_abcd_json = "null"
    runtime = PolicyRuntimeConfig(
        deployment=deployment,
        control_dt_s=control_dt_s,
        point_cloud_frame="xarm_base",
        point_cloud_color_source=POINT_CLOUD_COLOR_SOURCE,
        point_cloud_policy_id=POINT_CLOUD_POLICY_ID,
        point_cloud_config_sha256=runtime_config.pointcloud.sha256,
        point_cloud_table_plane_abcd_json=point_cloud_table_plane_abcd_json,
        point_cloud_sampling=POINT_CLOUD_SAMPLING,
        point_cloud_transform=POINT_CLOUD_TRANSFORM,
        arm_max_delta_rad_per_tick=runtime_config.policy.arm_max_delta_rad_per_tick,
        hand_max_delta_rad_per_tick=float(
            runtime_config.hand.hand_max_delta_rad_per_tick
        ),
        endpoint_delta_tolerance_rad=float(
            runtime_config.policy.endpoint_delta_tolerance_rad
        ),
        artifact=artifact,
        execution_mode=execution_mode,
        hand_acknowledged=hand_acknowledged,
        h4_execute_bounds=h4_execute_bounds,
        task_execute_bounds=task_execute_bounds,
    )
    canonical = {
        "artifact": {
            "checkpoint": artifact.checkpoint_path.name,
            "checkpoint_size_bytes": artifact.checkpoint_size_bytes,
            "checkpoint_sha256_from_index": artifact.checkpoint_sha256_from_index,
            "embedded_contract_sha256": artifact.embedded_contract_sha256,
            "index_sha256": artifact.index_sha256,
            "producer": {
                "repository": artifact.producer.repository,
                "commit": artifact.producer.commit,
                "metadata_provenance": artifact.producer.metadata_provenance,
            },
            "allocation": {
                field.name: getattr(allocation, field.name)
                for field in fields(PolicyArtifactContract)
            },
        },
        "deployment": {
            field.name: getattr(deployment, field.name)
            for field in fields(DeploymentConfig)
        },
        "point_cloud": {
            "frame": runtime.point_cloud_frame,
            "color_source": runtime.point_cloud_color_source,
            "policy_id": runtime.point_cloud_policy_id,
            "config_sha256": runtime.point_cloud_config_sha256,
            "table_plane_abcd_json": runtime.point_cloud_table_plane_abcd_json,
            "sampling": runtime.point_cloud_sampling,
            "transform": runtime.point_cloud_transform,
        },
        "control_dt_s": runtime.control_dt_s,
        "execution_mode": runtime.execution_mode,
        "hand_acknowledged": runtime.hand_acknowledged,
        "h4_execute_bounds": (
            None
            if runtime.h4_execute_bounds is None
            else {
                "max_published_endpoints": runtime.h4_execute_bounds.max_published_endpoints,
                "acknowledgement_timeout_s": runtime.h4_execute_bounds.acknowledgement_timeout_s,
                "max_running_s": runtime.h4_execute_bounds.max_running_s,
            }
        ),
        "task_execute_bounds": (
            None
            if runtime.task_execute_bounds is None
            else {
                "max_published_endpoints": runtime.task_execute_bounds.max_published_endpoints,
                "acknowledgement_timeout_s": runtime.task_execute_bounds.acknowledgement_timeout_s,
                "max_running_s": runtime.task_execute_bounds.max_running_s,
            }
        ),
    }
    canonical_json = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return ResolvedPolicyRuntimeConfig(
        runtime=runtime,
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    )
