"""Immutable Real-owned learned-policy runtime configuration.

Policy owns experiment discovery and every model/checkpoint detail.  This
module projects a public ``PolicySpec`` onto Real's observation, timing, IPC,
and execution-safety contracts without reading model artifacts.
"""

from __future__ import annotations

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
from dexmani_real.deployment.observation import parse_observation_fields
from dexmani_real.ipc.schema import (
    MAX_POLICY_CHUNK_STEPS,
    POINT_CLOUD_FEATURE_DIM,
    SUPPORTED_POINT_CLOUD_COUNTS,
)

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

# PolicySpec is authoritative for model-facing fields.  A deployment YAML may
# repeat one as an expectation, but may never change it.
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


def _policy_deployment_values(
    *, experiment: str, task_name: str, policy_spec: Any
) -> dict[str, Any]:
    """Validate a public PolicySpec and return its Real runtime projection."""
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment must be a non-empty Policy selector")
    if experiment != experiment.strip():
        raise ValueError("experiment must not have surrounding whitespace")
    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a non-empty string")
    required = (
        "action_key",
        "action_dim",
        "control_action_dim",
        "horizon",
        "n_obs_steps",
        "n_action_steps",
        "sensor_modalities",
        "point_cloud_num_points",
        "point_cloud_feature_dim",
        "control_dt_s",
        "requires_hand",
    )
    missing = [name for name in required if not hasattr(policy_spec, name)]
    if missing:
        raise TypeError(f"PolicySpec is missing required field(s): {missing}")
    modalities = tuple(policy_spec.sensor_modalities)
    if set(modalities) != {"joint_state", "point_cloud"} or len(modalities) != 2:
        raise ValueError(
            "Real deployment currently supports exactly joint_state + point_cloud"
        )
    if policy_spec.requires_hand is not True:
        raise ValueError(
            "Real deployment requires hand actions because its control schema is arm7 + hand12"
        )
    if policy_spec.point_cloud_feature_dim != POINT_CLOUD_FEATURE_DIM:
        raise ValueError(
            "Policy point_cloud_feature_dim does not match Real's xyzrgb contract"
        )
    if policy_spec.point_cloud_num_points not in SUPPORTED_POINT_CLOUD_COUNTS:
        raise ValueError(
            "Policy point_cloud_num_points is unsupported by Real shared memory"
        )
    if (
        type(policy_spec.n_action_steps) is not int
        or policy_spec.n_action_steps < 1
        or policy_spec.n_action_steps > MAX_POLICY_CHUNK_STEPS
    ):
        raise ValueError(
            f"Policy n_action_steps exceeds Real IPC capacity {MAX_POLICY_CHUNK_STEPS}"
        )
    expected_control_dim = 21 if policy_spec.action_key == "action_ee" else 19
    if policy_spec.action_key not in {"action", "action_ee"}:
        raise ValueError("Policy action_key is unsupported by Real")
    if policy_spec.control_action_dim != expected_control_dim:
        raise ValueError("Policy control_action_dim conflicts with its action_key")
    values: dict[str, Any] = {
        "experiment": experiment,
        "task_name": task_name,
        "action_key": policy_spec.action_key,
        "action_dim": policy_spec.action_dim,
        "control_action_dim": policy_spec.control_action_dim,
        "horizon": policy_spec.horizon,
        "observation_horizon": policy_spec.n_obs_steps,
        "n_action_steps": policy_spec.n_action_steps,
        "control_dt_s": float(policy_spec.control_dt_s),
        "hand_enabled": True,
        "observation_fields": "arm_qpos,hand_qpos,point_cloud",
        "pointcloud_num_points": policy_spec.point_cloud_num_points,
        "pointcloud_feature_dim": policy_spec.point_cloud_feature_dim,
    }
    return values


@dataclass(frozen=True)
class DeploymentConfig:
    """Frozen learned-policy deployment parameters.

    ``experiment`` selects the Policy-owned runtime.  Shape and modality fields
    are a validated projection of ``PolicySpec`` used by Real's fixed IPC
    allocation; users cannot override them.
    """

    experiment: str = ""
    # Policy inference is GPU-first.  CPU remains an explicit operator choice;
    # the inference child validates the selected torch device before restore.
    device: str = "cuda:0"
    # Read-only semantic task identity reported by the Policy experiment.
    task_name: str = ""
    # Read-only action contract projected from PolicySpec.
    action_key: str = "action"
    action_dim: int = 19
    control_action_dim: int = 19
    horizon: int = 16
    # Seeds model construction and subsequent process RNG streams once in the
    # inference worker, matching the Policy evaluation convention.
    inference_seed: int = 0
    inference_hz: float = 10.0
    observation_horizon: int = 2
    n_action_steps: int = 8
    control_dt_s: float = 0.0625
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
    pointcloud_feature_dim: int = POINT_CLOUD_FEATURE_DIM

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, str) or not self.experiment.strip():
            raise ValueError("experiment must be a non-empty Policy selector")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")
        if self.device != self.device.strip():
            raise ValueError("device must not have leading or trailing whitespace")
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
        expected_control_dim = 21 if self.action_key == "action_ee" else 19
        if self.control_action_dim != expected_control_dim:
            raise ValueError("control_action_dim does not match action_key")
        for name in ("action_dim", "horizon", "n_action_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_dim < self.control_action_dim:
            raise ValueError("action_dim must not be smaller than control_action_dim")
        if self.observation_horizon - 1 + self.n_action_steps > self.horizon:
            raise ValueError("observation/action window exceeds Policy horizon")
        if self.n_action_steps > MAX_POLICY_CHUNK_STEPS:
            raise ValueError(
                f"n_action_steps exceeds Real IPC capacity {MAX_POLICY_CHUNK_STEPS}"
            )
        if not math.isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and positive")
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
        if self.pointcloud_feature_dim != POINT_CLOUD_FEATURE_DIM:
            raise ValueError(
                f"pointcloud_feature_dim must be {POINT_CLOUD_FEATURE_DIM}"
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

    Deployment fields remain explicitly owned by ``deployment``. Sensor
    identity is populated by the lifecycle from the exact config passed to the
    realtime point-cloud worker, never by user YAML.
    """

    deployment: DeploymentConfig
    control_dt_s: float = 0.0
    point_cloud_frame: str = ""
    point_cloud_color_source: str = ""
    point_cloud_policy_id: str = ""
    point_cloud_table_plane_abcd_json: str = ""
    point_cloud_sampling: str = ""
    point_cloud_transform: str = ""
    arm_max_delta_rad_per_tick: float | None = None
    hand_max_delta_rad_per_tick: float = 0.3
    endpoint_delta_tolerance_rad: float = policy_defaults.endpoint_delta_tolerance_rad
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
            raise ValueError("physical execute requires a hand-enabled PolicySpec")
        if self.execution_mode in {"execute", "task"} and not self.hand_acknowledged:
            raise ValueError(
                "physical execute requires explicit --hand acknowledgement"
            )
        if not math.isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and positive")
        if not math.isclose(
            self.control_dt_s,
            self.deployment.control_dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Policy control_dt_s does not match Real control rate")
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
                "point_cloud_table_plane_abcd_json",
                "point_cloud_sampling",
                "point_cloud_transform",
            )
            if any(not getattr(self, name) for name in names):
                raise ValueError("point-cloud runtime semantics must be complete")

    @property
    def physical_execute_bounds(self) -> H4ExecuteBounds | TaskExecuteBounds | None:
        if self.execution_mode == "execute":
            return self.h4_execute_bounds
        if self.execution_mode == "task":
            return self.task_execute_bounds
        return None


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
class ResolvedPolicyRuntimeConfig:
    """Pickle-safe PolicySpec/Real projection for the deployment lifecycle."""

    runtime: PolicyRuntimeConfig


def resolve_policy_runtime_config(
    *,
    experiment: str,
    task_name: str,
    policy_spec: Any,
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
    """Project one immutable runtime config from PolicySpec and Real config.

    Model-facing fields come only from the public Policy contract.  The optional
    deployment YAML can set Real timing values or repeat Policy values exactly
    as a compatibility expectation.  It cannot redirect the experiment.
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

    expected_policy = _policy_deployment_values(
        experiment=experiment,
        task_name=task_name,
        policy_spec=policy_spec,
    )
    # Policy-owned fields are accepted only as exact compatibility expectations.
    allowed = frozenset(expected_policy) | _REAL_OWNED_DEPLOYMENT_FIELDS
    unknown = {str(key) for key in raw} - allowed
    if unknown:
        raise TypeError(
            "deployment config may contain only Policy expectations or "
            f"Real-owned fields; unknown={sorted(unknown)}"
        )
    base = DeploymentConfig(**cast(dict[str, Any], expected_policy))
    real_overrides = {
        key: value
        for key, value in raw.items()
        if str(key) in _REAL_OWNED_DEPLOYMENT_FIELDS
    }
    merged = _merge(base, real_overrides)
    for name, expected in expected_policy.items():
        if name not in raw:
            continue
        actual = _coerce_value(getattr(base, name), raw[name], name)
        if actual != expected:
            raise ValueError(
                f"Policy-owned deployment field {name!r} is an expectation "
                f"and must equal {expected!r}"
            )
    if device is not None:
        merged = _merge(DeploymentConfig(**merged), {"device": device})
    if inference_seed is not None:
        merged = _merge(DeploymentConfig(**merged), {"inference_seed": inference_seed})
    deployment = DeploymentConfig(**merged)

    control_dt_s = 1.0 / float(runtime_config.policy.control_hz)
    if not math.isclose(
        control_dt_s, deployment.control_dt_s, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Policy control_dt_s does not match Real policy.control_hz")
    if runtime_config.pointcloud.num_points != deployment.pointcloud_num_points:
        raise ValueError(
            "Policy point_cloud_num_points does not match Real pointcloud config"
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
        execution_mode=execution_mode,
        hand_acknowledged=hand_acknowledged,
        h4_execute_bounds=h4_execute_bounds,
        task_execute_bounds=task_execute_bounds,
    )
    return ResolvedPolicyRuntimeConfig(runtime=runtime)
