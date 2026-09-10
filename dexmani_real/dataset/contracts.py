"""Contracts for row-preserving offline episode processing."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.config.defaults import environment, hand
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH


def validate_processed_task_name(value: str) -> str:
    """Validate the shared processed/Policy task identity, not a path component."""
    if not isinstance(value, str):
        raise TypeError("processed task_name must be a string")
    if not value or value == "unknown" or value != value.strip():
        raise ValueError(
            "processed task_name must be non-empty, trimmed, and not 'unknown'"
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("processed task_name must not contain control characters")
    return value


@dataclass(frozen=True)
class ProcessingConfig:
    """Point-cloud, fingertip and table transforms; output is always multimodal.

    The canonical processed artifact is one full multimodal episode (joint,
    action, RGB-D, camera geometry, point cloud, aggregate contact, dense
    tactile, fingertip, and timing/validity), never a modality-specific profile.
    """

    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = (
        environment.table.plane_abcd
    )
    gzip_level: int = 4
    hand_urdf_path: str = str(XHAND_RIGHT_URDF_PATH)
    fingertip_link_names: tuple[str, ...] = hand.fingertip_link_names
    handbase_position_eef_m: tuple[float, float, float] = hand.T_eef_handbase_pos_xyz
    handbase_quat_eef_wxyz: tuple[float, float, float, float] = (
        hand.T_eef_handbase_quat_wxyz
    )

    @classmethod
    def from_runtime(cls, runtime: object, **overrides: Any) -> "ProcessingConfig":
        hand_config = getattr(runtime, "hand")
        table = getattr(runtime, "environment").table
        values = {
            "pointcloud": getattr(runtime, "pointcloud"),
            "table_plane_abcd": table.plane_abcd if table.enabled else None,
            "fingertip_link_names": tuple(hand_config.fingertip_link_names),
            "handbase_position_eef_m": tuple(hand_config.T_eef_handbase_pos_xyz),
            "handbase_quat_eef_wxyz": tuple(hand_config.T_eef_handbase_quat_wxyz),
        }
        unknown = set(overrides) - {item.name for item in dataclasses.fields(cls)}
        if unknown:
            raise TypeError(f"unknown ProcessingConfig override(s): {sorted(unknown)}")
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        if not isinstance(self.pointcloud, PointCloudConfig):
            raise TypeError("pointcloud must be a PointCloudConfig")
        if not 0 <= self.gzip_level <= 9:
            raise ValueError("gzip_level must be in [0, 9]")
        if not self.hand_urdf_path or len(self.fingertip_link_names) != 5:
            raise ValueError("fingertip geometry requires a URDF and five link names")
        if (
            len(self.handbase_position_eef_m) != 3
            or len(self.handbase_quat_eef_wxyz) != 4
            or not np.all(
                np.isfinite(
                    (*self.handbase_position_eef_m, *self.handbase_quat_eef_wxyz)
                )
            )
            or np.linalg.norm(self.handbase_quat_eef_wxyz) <= 0
        ):
            raise ValueError("fingertip hand-mount transform is invalid")
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
            validate_processed_task_name(self.task_name)


@dataclass(frozen=True)
class EpisodeDecision:
    """One whole-episode admission result; accepted rows are never selected."""

    source_path: Path
    source_frames: int
    rejected_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected_reason is None and self.source_frames > 0

    @property
    def processed_frames(self) -> int:
        return self.source_frames if self.accepted else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_episode": self.source_path.name,
            "accepted": self.accepted,
            "rejected_reason": self.rejected_reason,
            "source_frames": self.source_frames,
            "processed_frames": self.processed_frames,
        }
