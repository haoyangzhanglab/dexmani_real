"""PolicySpec compatibility and narrow inference-worker configuration.

Policy owns model shape, modality, horizon, and action-grid spacing. Real
validates that spacing against its control frequency and owns safety timing.
This module only validates their boundary and carries pickle-safe experiment
identity into the spawned worker; it never imports Policy or Torch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dexmani_real.config.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.ipc.schema import (
    MAX_PREDICTION_STEPS,
    POINT_CLOUD_FEATURE_DIM,
    SUPPORTED_POINT_CLOUD_COUNTS,
)
from dexmani_real.planning.kinematics.arm_fk import (
    EEF_POSE_ALGORITHM_ID,
    EEF_POSE_DERIVATION,
)
from dexmani_real.planning.kinematics.fingertip import (
    FINGERTIP_POINTS_DERIVATION,
    FINGERTIP_POLICY_ID,
)
from dexmani_real.robot.model import HAND_FINGER_ORDER_ID, XHAND_RIGHT_URDF_PATH

FIXED_POLICY_RUNTIME_TARGET = (
    "dexmani_real.deployment.inference.dexmani_policy:DexManiPolicyAdapter"
)


_SUPPORTED_OBSERVATION_FIELDS = frozenset(
    {
        "joint_state",
        "point_cloud",
        "rgb",
        "contact_force",
        "fingertip_points",
        "eef_pose",
        "tactile_force",
    }
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
        "eef_pose": ((9,), "float32"),
        "tactile_force": ((5, 120, 3), "float32"),
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
            raise ValueError("Policy rgb must be uint8 [H, W, 3] with positive H and W")
    return fields


def _validate_field_semantics(
    field: Any,
    *,
    field_name: str,
    expected: Mapping[str, object],
) -> None:
    semantics = getattr(field, "semantics", None)
    if not isinstance(semantics, Mapping):
        raise ValueError(f"{field_name} semantics mismatch")
    for key, value in expected.items():
        actual = semantics.get(key)
        if actual != value or (isinstance(value, bool) and type(actual) is not bool):
            raise ValueError(f"{field_name} {key} mismatch")


def _expected_pointcloud_semantics(runtime: Any) -> dict[str, str]:
    table = runtime.environment.table
    table_plane = table.plane_abcd if table.enabled else None
    return {
        "representation": "xyzrgb",
        "frame": "xarm_base",
        "position_units": "m",
        "color_order": "rgb",
        "color_source": POINT_CLOUD_COLOR_SOURCE,
        "policy_id": POINT_CLOUD_POLICY_ID,
        "table_plane_abcd_json": json.dumps(
            table_plane,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "sampling": POINT_CLOUD_SAMPLING,
        "transform": POINT_CLOUD_TRANSFORM,
    }


def _expected_fingertip_semantics(runtime: Any) -> dict[str, str]:
    return {
        "representation": "point_xyz",
        "frame": "xarm_base",
        "units": "m",
        "finger_order": HAND_FINGER_ORDER_ID,
        "derivation": FINGERTIP_POINTS_DERIVATION,
        "policy_id": FINGERTIP_POLICY_ID,
    }


def _expected_tactile_force_semantics() -> dict[str, object]:
    return {
        "representation": "xhand_sdk_raw_force_fx_fy_fz",
        "finger_order": HAND_FINGER_ORDER_ID,
        "sensor_order": "xhand_sdk_sensor_data_order",
        "point_order": "xhand_sdk_sensor_data_raw_force_order",
        "axis_labels": "fx_fy_fz",
        "unit": "sdk_scaled_unknown_si",
        "si_verified": False,
        "spatial_geometry_verified": False,
    }


def _expected_eef_pose_semantics() -> dict[str, str]:
    return {
        "representation": "position_m_rot6d",
        "frame": "xarm_base",
        "position_units": "m",
        "rotation_representation": "rot6d",
        "derivation": EEF_POSE_DERIVATION,
        "algorithm_id": EEF_POSE_ALGORITHM_ID,
    }


def validate_policy_runtime_compatibility(policy_spec: Any, runtime: Any) -> None:
    """Validate only whether Real can run the Policy-owned public contract."""
    fields = _validate_real_observation_capability(policy_spec)
    if policy_spec.requires_hand is not True:
        raise ValueError(
            "Real deployment requires hand actions because its control schema is "
            "arm7 + hand12"
        )
    if policy_spec.chunk_size > MAX_PREDICTION_STEPS:
        raise ValueError(
            f"Policy chunk_size exceeds Real IPC capacity {MAX_PREDICTION_STEPS}"
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
    tactile_force = fields_by_name.get("tactile_force")
    if tactile_force is not None:
        _validate_field_semantics(
            tactile_force,
            field_name="tactile_force",
            expected=_expected_tactile_force_semantics(),
        )
    point_cloud = fields_by_name.get("point_cloud")
    if (
        point_cloud is not None
        and runtime.pointcloud.num_points != point_cloud.shape[0]
    ):
        raise ValueError(
            "Policy point_cloud shape does not match Real pointcloud config"
        )
    if point_cloud is not None:
        _validate_field_semantics(
            point_cloud,
            field_name="point_cloud",
            expected=_expected_pointcloud_semantics(runtime),
        )
    fingertip_points = fields_by_name.get("fingertip_points")
    if fingertip_points is not None:
        _validate_field_semantics(
            fingertip_points,
            field_name="fingertip_points",
            expected=_expected_fingertip_semantics(runtime),
        )
    eef_pose = fields_by_name.get("eef_pose")
    if eef_pose is not None:
        _validate_field_semantics(
            eef_pose,
            field_name="eef_pose",
            expected=_expected_eef_pose_semantics(),
        )


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
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("policy inference seed must be a non-negative integer")


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
    "InferenceWorkerConfig",
    "validate_policy_runtime_compatibility",
    "validate_max_running_s",
]
