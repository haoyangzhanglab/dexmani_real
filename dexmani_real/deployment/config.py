"""PolicySpec compatibility and narrow inference-worker configuration.

Policy owns model shape, modality, horizon, and action-grid spacing. Real
validates that spacing against its control frequency and owns safety timing.
This module only validates their boundary and carries pickle-safe experiment
identity into the spawned worker; it never imports Policy or Torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dexmani_real.ipc.schema import (
    MAX_PREDICTION_STEPS,
    POINT_CLOUD_FEATURE_DIM,
    SUPPORTED_POINT_CLOUD_COUNTS,
)
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH

FIXED_POLICY_RUNTIME_TARGET = (
    "dexmani_real.deployment.inference.dexmani_policy:DexManiPolicyAdapter"
)

_DEPLOYMENT_DEFAULT_MODE = "sync"
_DEPLOYMENT_MODES = frozenset({"sync", "async"})

_SUPPORTED_OBSERVATION_FIELDS = frozenset(
    {"joint_state", "point_cloud", "rgb", "contact_force", "fingertip_points"}
)


def validate_max_running_s(max_running_s: float | None) -> float | None:
    """Validate the executor-owned B-relative episode duration limit."""
    if max_running_s is None:
        return None
    if isinstance(max_running_s, bool):
        raise TypeError("max_running_s must be a finite positive number or None")
    value = float(max_running_s)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("max_running_s must be finite and positive")
    return value


def _validate_real_observation_capability(policy_spec: Any) -> tuple[Any, ...]:
    """Validate the Policy observation projection that Real can produce."""
    fields = policy_spec.observation_fields
    names = tuple(field.name for field in fields)
    if not set(names) <= _SUPPORTED_OBSERVATION_FIELDS or "joint_state" not in names:
        raise ValueError(
            "Real supports only configured modalities including joint_state"
        )

    fixed_fields = {
        "joint_state": ((19,), "float32"),
        "contact_force": ((5, 3), "float32"),
        "fingertip_points": ((5, 3), "float32"),
    }
    for field, name in zip(fields, names, strict=True):
        shape = field.shape
        dtype = field.dtype
        if name in fixed_fields:
            if (shape, dtype) != fixed_fields[name]:
                raise ValueError(
                    f"Policy observation field {name!r} does not match Real's raw tensor"
                )
        elif name == "point_cloud":
            if (
                len(shape) != 2
                or shape[1] != POINT_CLOUD_FEATURE_DIM
                or dtype != "float32"
                or shape[0] not in SUPPORTED_POINT_CLOUD_COUNTS
            ):
                raise ValueError(
                    "Policy point_cloud must be float32 [N, 6] with a supported N"
                )
        elif name == "rgb" and (
            len(shape) != 3
            or shape[2] != 3
            or dtype != "uint8"
            or shape[0] <= 0
            or shape[1] <= 0
        ):
            raise ValueError(
                "Policy rgb must be uint8 [H, W, 3] with positive H and W"
            )
    return fields


def validate_policy_runtime_compatibility(policy_spec: Any, runtime: Any) -> None:
    """Validate only whether Real can run the Policy-owned public contract."""
    fields = _validate_real_observation_capability(policy_spec)
    if policy_spec.requires_hand is not True:
        raise ValueError(
            "Real deployment requires hand actions because its control schema is "
            "arm7 + hand12"
        )
    if policy_spec.n_action_steps > MAX_PREDICTION_STEPS:
        raise ValueError(
            f"Policy n_action_steps exceeds Real IPC capacity {MAX_PREDICTION_STEPS}"
        )
    if policy_spec.action_key not in {"action", "action_ee"}:
        raise ValueError("Policy action_key is unsupported by Real")
    expected_control_dim = 21 if policy_spec.action_key == "action_ee" else 19
    if policy_spec.control_action_dim != expected_control_dim:
        raise ValueError("Policy control_action_dim conflicts with its action_key")
    control_dt_s = 1.0 / float(runtime.policy.control_hz)
    if not math.isclose(
        control_dt_s,
        float(policy_spec.control_dt_s),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Policy control_dt_s does not match Real policy.control_hz")
    fields_by_name = {field.name: field for field in fields}
    point_cloud = fields_by_name.get("point_cloud")
    if (
        point_cloud is not None
        and runtime.pointcloud.num_points != point_cloud.shape[0]
    ):
        raise ValueError(
            "Policy point_cloud shape does not match Real pointcloud config"
        )


@dataclass(frozen=True)
class PolicyDeploymentConfig:
    """Small operator-owned policy deployment configuration.

    Model structure and chunk length remain owned by ``PolicySpec``.  Real
    runtime and safety parameters remain owned by ``ExperimentConfig``;
    this object only carries the explicit scheduler mode and episode horizon
    requested by the deployment workflow.
    """

    inference_mode: str = _DEPLOYMENT_DEFAULT_MODE
    max_action_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inference_mode, str) or (
            self.inference_mode not in _DEPLOYMENT_MODES
        ):
            raise ValueError("inference_mode must be one of 'sync' or 'async'")
        if self.max_action_steps is not None and (
            type(self.max_action_steps) is not int or self.max_action_steps <= 0
        ):
            raise ValueError("max_action_steps must be a positive integer or null")


@dataclass(frozen=True)
class InferenceWorkerConfig:
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


@dataclass(frozen=True)
class FingertipAssemblerConfig:
    """Explicit pickle-safe geometry inputs for deployment-local FK."""

    hand_urdf_path: str
    fingertip_link_names: tuple[str, ...]
    handbase_position_eef_m: tuple[float, float, float]
    handbase_quat_eef_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_runtime(cls, runtime: Any) -> "FingertipAssemblerConfig":
        hand = runtime.hand
        return cls(
            hand_urdf_path=str(XHAND_RIGHT_URDF_PATH),
            fingertip_link_names=tuple(hand.fingertip_link_names),
            handbase_position_eef_m=tuple(hand.T_eef_handbase_pos_xyz),
            handbase_quat_eef_wxyz=tuple(hand.T_eef_handbase_quat_wxyz),
        )

    def __post_init__(self) -> None:
        if not self.hand_urdf_path or len(self.fingertip_link_names) != 5:
            raise ValueError("fingertip FK requires one URDF and five link names")
        values = (*self.handbase_position_eef_m, *self.handbase_quat_eef_wxyz)
        if (
            len(self.handbase_position_eef_m) != 3
            or len(self.handbase_quat_eef_wxyz) != 4
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise ValueError("fingertip mount transform is malformed")


__all__ = [
    "FIXED_POLICY_RUNTIME_TARGET",
    "FingertipAssemblerConfig",
    "PolicyDeploymentConfig",
    "InferenceWorkerConfig",
    "validate_policy_runtime_compatibility",
    "validate_max_running_s",
]
