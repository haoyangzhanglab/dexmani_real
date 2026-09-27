"""Canonical multimodal representations and their numerical recipes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.hardware import HandParams
from dexmani_real.config.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
    PointCloudConfig,
)
from dexmani_real.robot.model import (
    CONTACT_FORCE_REPRESENTATION,
    HAND_FINGER_NAMES,
    ROBOT_JOINT_NAMES,
    TACTILE_FORCE_POINT_ORDER,
    TACTILE_FORCE_REPRESENTATION,
    XHAND_SDK_JOINT_NAMES,
    XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
    XHAND_SENSOR_NATIVE_AXES_FRAME,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
)


def validate_task_name(value: str) -> str:
    """Validate the shared policy task identity, not a path component."""
    if not isinstance(value, str):
        raise TypeError("task_name must be a string")
    if not value or value == "unknown" or value != value.strip():
        raise ValueError("task_name must be non-empty, trimmed, and not 'unknown'")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("task_name must not contain control characters")
    return value


@dataclass(frozen=True)
class ProcessingConfig:
    """Point-cloud, fingertip and table transforms; output is always multimodal.

    Every included episode retains joint targets, RGB-D, geometry, contact,
    arm velocity/effort, hand current and derived geometry.
    """

    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = None
    fingertip_link_names: tuple[str, ...] = HandParams.fingertip_link_names

    @classmethod
    def from_runtime(cls, runtime: object, **overrides: Any) -> "ProcessingConfig":
        hand_config = getattr(runtime, "hand")
        values = {
            "pointcloud": getattr(runtime, "pointcloud"),
            "table_plane_abcd": None,
            "fingertip_link_names": tuple(hand_config.fingertip_link_names),
        }
        unknown = set(overrides) - {item.name for item in dataclasses.fields(cls)}
        if unknown:
            raise TypeError(f"unknown ProcessingConfig override(s): {sorted(unknown)}")
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        if not isinstance(self.pointcloud, PointCloudConfig):
            raise TypeError("pointcloud must be a PointCloudConfig")
        if (
            len(self.fingertip_link_names) != 5
            or len(set(self.fingertip_link_names)) != 5
            or any(
                not isinstance(name, str) or not name.strip() for name in self.fingertip_link_names
            )
        ):
            raise ValueError("fingertip geometry requires five distinct link names")
        if not self.pointcloud.remove_table:
            object.__setattr__(self, "table_plane_abcd", None)
        elif self.table_plane_abcd is None:
            raise ValueError("remove_table=True requires an explicit table_plane_abcd")
        if self.table_plane_abcd is not None:
            plane = tuple(float(v) for v in self.table_plane_abcd)
            if len(plane) != 4 or not np.all(np.isfinite(plane)) or plane[2] <= 0:
                raise ValueError("table_plane_abcd must be finite with upward normal")
            object.__setattr__(self, "table_plane_abcd", plane)

    def to_dict(self) -> dict[str, Any]:
        values = dataclasses.asdict(self)
        values["pointcloud"] = self.pointcloud.to_dict()
        return values


_CORE_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "joint_state": ((19,), np.dtype(np.float32)),
    "arm_qvel": ((7,), np.dtype(np.float32)),
    "arm_effort": ((7,), np.dtype(np.float32)),
    "hand_current": ((12,), np.dtype(np.float32)),
    "action": ((19,), np.dtype(np.float32)),
    "action_ee": ((21,), np.dtype(np.float32)),
    "contact_force": ((5, 3), np.dtype(np.float32)),
    "tactile_force": ((5, 120, 3), np.dtype(np.float32)),
    "fingertip_points": ((5, 3), np.dtype(np.float32)),
    "eef_pose": ((9,), np.dtype(np.float32)),
}


def canonical_array_specs(
    length: int,
    num_points: int,
    rgb_height: int,
    rgb_width: int,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Canonical array shapes and dtypes for one aligned RGB-D episode."""
    specs = {
        name: ((length, *tail_shape), dtype)
        for name, (tail_shape, dtype) in _CORE_DATASET_SPECS.items()
    }
    specs.update(
        {
            "rgb": ((length, rgb_height, rgb_width, 3), np.dtype(np.uint8)),
            "depth": ((length, rgb_height, rgb_width), np.dtype(np.uint16)),
            "point_cloud": ((length, num_points, 6), np.dtype(np.float32)),
        }
    )
    return specs


CANONICAL_FORMAT = "dexmani.real.canonical"
OPTIONAL_TELEMETRY = frozenset(
    {"arm_qvel", "arm_effort", "hand_current", "contact_force", "tactile_force"}
)
FINGERTIP_KINEMATIC_MODEL = "xarm7_xhand_right"

# These facts cannot be inferred from array shape/dtype. Field identities keep
# their meaning; derived recipes may vary and are recorded separately below.
CANONICAL_MODALITY_SEMANTICS = {
    "joint_state": {
        "semantic_id": "dexmani.joint_state",
        "unit": "rad",
        "joint_order": list(ROBOT_JOINT_NAMES),
    },
    "arm_qvel": {
        "semantic_id": "dexmani.xarm.joint_velocity",
        "unit": "rad/s",
        "joint_order": list(ROBOT_JOINT_NAMES[:7]),
        "missing": "nan",
    },
    "arm_effort": {
        "semantic_id": "dexmani.xarm.effort.native",
        "unit": "sdk_native_unverified",
        "joint_order": list(ROBOT_JOINT_NAMES[:7]),
        "missing": "nan",
    },
    "hand_current": {
        "semantic_id": "dexmani.xhand.current",
        "unit": "mA",
        "joint_order": list(XHAND_SDK_JOINT_NAMES),
        "missing": "nan",
    },
    "eef_pose": {
        "semantic_id": "dexmani.eef_pose.xarm_base",
        "frame": "xarm_base",
        "components": "position_m(3)+rot6d(6)",
        "rotation": "first_two_columns_column_major",
        "recipe": {"kinematic_model": "xarm7", "eef_link": "custom_eef_link"},
    },
    "fingertip_points": {
        "semantic_id": "dexmani.fingertip_points.xarm_base",
        "frame": "xarm_base",
        "unit": "m",
        "finger_order": list(HAND_FINGER_NAMES),
    },
    "contact_force": {
        "semantic_id": "dexmani.xhand.tactile_aggregate.native",
        "representation": CONTACT_FORCE_REPRESENTATION,
        "unit": XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
        "frame": XHAND_SENSOR_NATIVE_AXES_FRAME,
        "finger_order": list(HAND_FINGER_NAMES),
        "sensor_order": list(XHAND_TACTILE_SENSOR_FINGER_IDS),
        "axis_order": ["fx", "fy", "fz"],
        "missing": "nan",
    },
    "tactile_force": {
        "semantic_id": "dexmani.xhand.tactile_dense.native",
        "representation": TACTILE_FORCE_REPRESENTATION,
        "unit": XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT,
        "frame": XHAND_SENSOR_NATIVE_AXES_FRAME,
        "finger_order": list(HAND_FINGER_NAMES),
        "sensor_order": list(XHAND_TACTILE_SENSOR_FINGER_IDS),
        "point_order": TACTILE_FORCE_POINT_ORDER,
        "axis_order": ["fx", "fy", "fz"],
        "missing": "nan",
    },
    "rgb": {
        "semantic_id": "dexmani.rgb.aligned_color",
        "channels": ["r", "g", "b"],
        "range": [0, 255],
    },
    "depth": {
        "semantic_id": "dexmani.depth.aligned_z16",
        "unit": "camera_depth_unit",
        "invalid_value": 0,
        "grid": "aligned_color",
    },
    "point_cloud": {
        "semantic_id": "dexmani.point_cloud.xyzrgb.xarm_base",
        "frame": "xarm_base",
        "features": ["x", "y", "z", "r", "g", "b"],
        "xyz_unit": "m",
        "rgb_range": [0, 1],
        "derivation": {
            "transform": POINT_CLOUD_TRANSFORM,
            "color_source": POINT_CLOUD_COLOR_SOURCE,
            "sampling": POINT_CLOUD_SAMPLING,
        },
    },
    "action": {
        "semantic_id": "dexmani.action.joint_absolute_published",
        "unit": "rad",
        "joint_order": list(ROBOT_JOINT_NAMES),
    },
    "action_ee": {
        "semantic_id": "dexmani.action.ee_target",
        "frame": "xarm_base",
        "components": "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)",
        "rotation": "first_two_columns_column_major",
        "hand_joint_order": list(XHAND_SDK_JOINT_NAMES),
        "recipe": {
            "kinematic_model": "xarm7",
            "eef_link": "custom_eef_link",
            "source": "final_published_joint_target",
        },
    },
}
for _name, _contract in CANONICAL_MODALITY_SEMANTICS.items():
    _contract["alignment"] = (
        "published_target_for_current_control_step"
        if _name in {"action", "action_ee"}
        else "control_step_latest_causal_before_action"
    )


def canonical_modality_contracts(reader, config: ProcessingConfig):
    contracts = {name: dict(attrs) for name, attrs in CANONICAL_MODALITY_SEMANTICS.items()}
    contracts["depth"]["scale_m_per_unit"] = float(reader.meta["depth_scale"])
    contracts["fingertip_points"]["recipe"] = {
        "kinematic_model": FINGERTIP_KINEMATIC_MODEL,
        "fingertip_link_names": list(config.fingertip_link_names),
        "mount_source": "raw_episode",
    }
    contracts["point_cloud"]["recipe"] = config.pointcloud.to_dict()
    # Export provenance is not a deployment calibration: live Real uses today's plane.
    contracts["point_cloud"]["export_provenance"] = {
        "table_plane_abcd": list(config.table_plane_abcd)
        if config.table_plane_abcd is not None
        else None,
    }
    return contracts
