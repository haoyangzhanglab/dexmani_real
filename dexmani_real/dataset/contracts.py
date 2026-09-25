"""Contracts for raw-to-policy numerical transforms."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.experiment import resolve_table_plane
from dexmani_real.config.hardware import HandParams
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH


def canonical_json(value: Any) -> str:
    """Serialize static metadata deterministically; reject NaN and infinity."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
    hand_urdf_path: str = str(XHAND_RIGHT_URDF_PATH)
    fingertip_link_names: tuple[str, ...] = HandParams.fingertip_link_names
    handbase_position_eef_m: tuple[float, float, float] = HandParams.T_eef_handbase_pos_xyz
    handbase_quat_eef_wxyz: tuple[float, float, float, float] = HandParams.T_eef_handbase_quat_wxyz

    @classmethod
    def from_runtime(cls, runtime: object, **overrides: Any) -> "ProcessingConfig":
        hand_config = getattr(runtime, "hand")
        table = getattr(runtime, "environment").table
        values = {
            "pointcloud": getattr(runtime, "pointcloud"),
            "table_plane_abcd": None,
            "fingertip_link_names": tuple(hand_config.fingertip_link_names),
            "handbase_position_eef_m": tuple(hand_config.T_eef_handbase_pos_xyz),
            "handbase_quat_eef_wxyz": tuple(hand_config.T_eef_handbase_quat_wxyz),
        }
        unknown = set(overrides) - {item.name for item in dataclasses.fields(cls)}
        if unknown:
            raise TypeError(f"unknown ProcessingConfig override(s): {sorted(unknown)}")
        values.update(overrides)
        if "table_plane_abcd" not in overrides and values["pointcloud"].remove_table:
            values["table_plane_abcd"] = resolve_table_plane(table)
        return cls(**values)

    def __post_init__(self) -> None:
        if not isinstance(self.pointcloud, PointCloudConfig):
            raise TypeError("pointcloud must be a PointCloudConfig")
        if not self.hand_urdf_path or len(self.fingertip_link_names) != 5:
            raise ValueError("fingertip geometry requires a URDF and five link names")
        if (
            len(self.handbase_position_eef_m) != 3
            or len(self.handbase_quat_eef_wxyz) != 4
            or not np.all(
                np.isfinite((*self.handbase_position_eef_m, *self.handbase_quat_eef_wxyz))
            )
            or np.linalg.norm(self.handbase_quat_eef_wxyz) <= 0
        ):
            raise ValueError("fingertip hand-mount transform is invalid")
        if not self.pointcloud.remove_table:
            object.__setattr__(self, "table_plane_abcd", None)
        elif self.table_plane_abcd is None:
            raise ValueError("remove_table=True requires a current calibrated table plane")
        if self.table_plane_abcd is not None:
            plane = tuple(float(v) for v in self.table_plane_abcd)
            if len(plane) != 4 or not np.all(np.isfinite(plane)) or plane[2] <= 0:
                raise ValueError("table_plane_abcd must be finite with upward normal")
            object.__setattr__(self, "table_plane_abcd", plane)

    def to_dict(self) -> dict[str, Any]:
        values = dataclasses.asdict(self)
        values["pointcloud"] = self.pointcloud.to_dict()
        return values


@dataclass(frozen=True)
class EpisodeAnnotation:
    """An operator may include/exclude an entire episode and set its task."""

    include: bool = True
    task_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.include, bool):
            raise TypeError("episode include must be boolean")
        if self.task_name is not None:
            validate_task_name(self.task_name)


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


def policy_array_specs(
    length: int,
    num_points: int,
    rgb_height: int,
    rgb_width: int,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Policy array shapes and dtypes for one native-RGB-D episode."""
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
