"""Typed contracts for offline episode cleaning and mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class OutputProfile(str, Enum):
    """Uniform top-level dataset profiles accepted by downstream consumers."""

    JOINT = "joint"
    RGB = "rgb"
    POINTCLOUD = "pointcloud"
    RGB_PC = "rgb_pc"

    @property
    def needs_rgb(self) -> bool:
        return self in (OutputProfile.RGB, OutputProfile.RGB_PC)

    @property
    def needs_pointcloud(self) -> bool:
        return self in (OutputProfile.POINTCLOUD, OutputProfile.RGB_PC)

    @property
    def dataset_keys(self) -> tuple[str, ...]:
        keys = ["joint_state", "action"]
        if self.needs_rgb:
            keys.extend(("rgb", "camera_intrinsic"))
        if self.needs_pointcloud:
            keys.append("point_cloud")
        return tuple(keys)


@dataclass(frozen=True)
class ProcessingConfig:
    """Resolved, immutable processing policy.

    Soft quality metrics never remove individual rows.  ``horizon`` and
    ``min_full_windows`` only admit complete contiguous segments after hard
    validity and user range selection have been applied.
    """

    profile: OutputProfile
    horizon: int = 16
    min_full_windows: int = 1
    target_rgb_height: int = 240
    target_rgb_width: int = 320
    target_point_count: int = 1024
    max_camera_age_s: float = 0.25
    grid_dt_relative_tolerance: float = 0.05
    joint_limit_tolerance_rad: float = 1e-6
    hand_state_limit_tolerance_rad: float = float(np.deg2rad(3.0))
    gzip_level: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.profile, OutputProfile):
            raise TypeError("profile must be an OutputProfile")
        positive_ints = (
            self.horizon,
            self.min_full_windows,
            self.target_rgb_height,
            self.target_rgb_width,
            self.target_point_count,
        )
        if any(not isinstance(value, int) or value <= 0 for value in positive_ints):
            raise ValueError("horizon, window count, target image size, and point count must be positive integers")
        finite_positive = (
            self.max_camera_age_s,
            self.grid_dt_relative_tolerance,
            self.joint_limit_tolerance_rad,
            self.hand_state_limit_tolerance_rad,
        )
        if not all(np.isfinite(value) and value > 0 for value in finite_positive):
            raise ValueError("camera age, grid tolerance, and joint tolerance must be finite and positive")
        if not 0 <= self.gzip_level <= 9:
            raise ValueError("gzip_level must be in [0, 9]")

    @property
    def min_segment_frames(self) -> int:
        return self.horizon + self.min_full_windows - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "horizon": self.horizon,
            "min_full_windows": self.min_full_windows,
            "min_segment_frames": self.min_segment_frames,
            "target_rgb_height": self.target_rgb_height,
            "target_rgb_width": self.target_rgb_width,
            "target_point_count": self.target_point_count,
            "max_camera_age_s": self.max_camera_age_s,
            "grid_dt_relative_tolerance": self.grid_dt_relative_tolerance,
            "joint_limit_tolerance_rad": self.joint_limit_tolerance_rad,
            "hand_state_limit_tolerance_rad": self.hand_state_limit_tolerance_rad,
            "gzip_level": self.gzip_level,
        }


@dataclass(frozen=True)
class EpisodeAnnotation:
    """Optional human-authored selection and task metadata for one source."""

    include: bool = True
    task_name: str | None = None
    task_outcome: str = "unknown"
    include_ranges: tuple[tuple[int, int], ...] = ()
    exclude_ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.task_name is not None and not self.task_name.strip():
            raise ValueError("task_name must be non-empty when provided")
        if self.task_outcome not in ("success", "failure", "unknown"):
            raise ValueError("task_outcome must be success, failure, or unknown")
        for label, ranges in (
            ("include_ranges", self.include_ranges),
            ("exclude_ranges", self.exclude_ranges),
        ):
            for start, end in ranges:
                if start < 0 or end <= start:
                    raise ValueError(f"{label} entries must be non-negative half-open [start, end) ranges")


@dataclass(frozen=True)
class SegmentDecision:
    """One admitted, contiguous half-open range in source-grid coordinates."""

    start: int
    end: int
    full_window_count: int
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame_start": self.start,
            "source_frame_end_exclusive": self.end,
            "length": self.length,
            "full_window_count": self.full_window_count,
            "quality": self.quality,
        }


@dataclass(frozen=True)
class EpisodeDecision:
    """Complete auditable cleaning decision for one source episode."""

    source_path: Path
    source_frames: int
    profile: OutputProfile
    segments: tuple[SegmentDecision, ...]
    hard_reason_counts: dict[str, int]
    boundary_counts: dict[str, int]
    dropped_short_segment_frames: int
    selected_frames: int
    quality: dict[str, Any]
    warnings: tuple[str, ...] = ()
    rejected_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected_reason is None and bool(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_episode": self.source_path.name,
            "source_path": str(self.source_path),
            "profile": self.profile.value,
            "accepted": self.accepted,
            "rejected_reason": self.rejected_reason,
            "source_frames": self.source_frames,
            "selected_frames": self.selected_frames,
            "retention_ratio": (self.selected_frames / self.source_frames if self.source_frames else 0.0),
            "hard_reason_counts": self.hard_reason_counts,
            "boundary_counts": self.boundary_counts,
            "dropped_short_segment_frames": self.dropped_short_segment_frames,
            "segments": [segment.to_dict() for segment in self.segments],
            "quality": self.quality,
            "warnings": list(self.warnings),
        }
