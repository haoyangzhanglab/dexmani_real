"""PolicySpec compatibility and narrow inference-worker configuration.

Policy owns model shape, modality, and horizon. Real owns all control timing in
``ResolvedRuntimeConfig.policy``. This module only validates their boundary and
carries pickle-safe experiment identity into the spawned worker; it never
imports Policy or Torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dexmani_real.ipc.schema import (
    MAX_POLICY_CHUNK_STEPS,
    POINT_CLOUD_FEATURE_DIM,
    SUPPORTED_POINT_CLOUD_COUNTS,
)

FIXED_POLICY_RUNTIME_TARGET = (
    "dexmani_real.integrations.dexmani_policy:DexManiPolicyRuntime"
)

_REQUIRED_POLICY_SPEC_FIELDS = (
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


def validate_policy_spec(policy_spec: Any) -> None:
    """Validate the public model contract supported by the fixed Real setup."""
    missing = [
        name for name in _REQUIRED_POLICY_SPEC_FIELDS if not hasattr(policy_spec, name)
    ]
    if missing:
        raise TypeError(f"PolicySpec is missing required field(s): {missing}")

    modalities = tuple(policy_spec.sensor_modalities)
    if set(modalities) != {"joint_state", "point_cloud"} or len(modalities) != 2:
        raise ValueError(
            "Real deployment currently supports exactly joint_state + point_cloud"
        )
    if policy_spec.requires_hand is not True:
        raise ValueError(
            "Real deployment requires hand actions because its control schema is "
            "arm7 + hand12"
        )
    if policy_spec.point_cloud_feature_dim != POINT_CLOUD_FEATURE_DIM:
        raise ValueError(
            "Policy point_cloud_feature_dim does not match Real's xyzrgb contract"
        )
    if (
        isinstance(policy_spec.point_cloud_num_points, bool)
        or policy_spec.point_cloud_num_points not in SUPPORTED_POINT_CLOUD_COUNTS
    ):
        raise ValueError(
            "Policy point_cloud_num_points is unsupported by Real shared memory"
        )

    integer_fields = (
        "action_dim",
        "control_action_dim",
        "horizon",
        "n_obs_steps",
        "n_action_steps",
    )
    for name in integer_fields:
        value = getattr(policy_spec, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Policy {name} must be a positive integer")
    if policy_spec.n_action_steps > MAX_POLICY_CHUNK_STEPS:
        raise ValueError(
            f"Policy n_action_steps exceeds Real IPC capacity {MAX_POLICY_CHUNK_STEPS}"
        )
    if policy_spec.action_dim < policy_spec.control_action_dim:
        raise ValueError(
            "Policy action_dim must not be smaller than control_action_dim"
        )
    if policy_spec.n_obs_steps - 1 + policy_spec.n_action_steps > policy_spec.horizon:
        raise ValueError("Policy observation/action window exceeds its horizon")

    if policy_spec.action_key not in {"action", "action_ee"}:
        raise ValueError("Policy action_key is unsupported by Real")
    expected_control_dim = 21 if policy_spec.action_key == "action_ee" else 19
    if policy_spec.control_action_dim != expected_control_dim:
        raise ValueError("Policy control_action_dim conflicts with its action_key")
    if (
        isinstance(policy_spec.control_dt_s, bool)
        or not isinstance(policy_spec.control_dt_s, (int, float))
        or not math.isfinite(float(policy_spec.control_dt_s))
        or float(policy_spec.control_dt_s) <= 0.0
    ):
        raise ValueError("Policy control_dt_s must be finite and positive")


def validate_policy_runtime_compatibility(policy_spec: Any, runtime: Any) -> None:
    """Validate Policy-owned structure against Real-owned runtime semantics."""
    validate_policy_spec(policy_spec)
    control_dt_s = 1.0 / float(runtime.policy.control_hz)
    if not math.isclose(
        control_dt_s,
        float(policy_spec.control_dt_s),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Policy control_dt_s does not match Real policy.control_hz")
    if runtime.pointcloud.num_points != policy_spec.point_cloud_num_points:
        raise ValueError(
            "Policy point_cloud_num_points does not match Real pointcloud config"
        )


@dataclass(frozen=True)
class PolicyWorkerConfig:
    """Narrow Policy-owned inputs required by the inference child."""

    experiment: str
    device: str
    spec: Any
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, str) or not self.experiment.strip():
            raise ValueError("experiment must be a non-empty Policy selector")
        if self.experiment != self.experiment.strip():
            raise ValueError("experiment must not have surrounding whitespace")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")
        if self.device != self.device.strip():
            raise ValueError("device must not have leading or trailing whitespace")
        if type(self.seed) is not int or self.seed != 0:
            raise ValueError("policy inference seed is fixed to 0")
        validate_policy_spec(self.spec)


__all__ = [
    "FIXED_POLICY_RUNTIME_TARGET",
    "PolicyWorkerConfig",
    "validate_policy_runtime_compatibility",
    "validate_policy_spec",
]
