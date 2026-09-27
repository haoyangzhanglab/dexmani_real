"""Canonical multimodal representations and their numerical recipes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.hardware import HandParams
from dexmani_real.config.pointcloud import PointCloudConfig


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
