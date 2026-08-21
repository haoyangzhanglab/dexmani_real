"""Typed contracts for offline episode cleaning and mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, hand, policy as policy_defaults


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
        # Robot and tactile modalities form the processed core of every raw episode.
        keys = [
            "joint_state",
            "action",
            "action_ee",
            "contact_force",
            "fingertip_points",
        ]
        if self.needs_rgb:
            keys.extend(("rgb", "depth", "camera_intrinsic", "camera_extrinsic"))
        if self.needs_pointcloud:
            keys.append("point_cloud")
        return tuple(keys)


class QualityPolicy(str, Enum):
    """How temporal quality findings affect otherwise valid source rows."""

    HARD_ONLY = "hard_only"
    AUDIT = "audit"
    STRICT = "strict"


class BridgePolicy(str, Enum):
    """Handle a risky compacted transition without splitting the demonstration."""

    AUDIT = "audit"
    REJECT = "reject"


@dataclass(frozen=True)
class TemporalQualityConfig:
    """Resolved thresholds for conservative temporal anomaly detection.

    Abrupt steps and persistent tracking error are audit-only evidence.  Strict
    processing excludes only reversible one-frame command impulses and
    high-confidence arm command/feedback stalls.
    """

    policy: QualityPolicy = QualityPolicy.AUDIT
    abrupt_arm_step_rad: float = float(np.deg2rad(8.0))
    abrupt_hand_step_rad: float = policy_defaults.hand_max_delta_rad_per_tick
    impulse_arm_min_rad: float = 0.08
    impulse_hand_min_rad: float = 0.12
    impulse_min_return_ratio: float = 0.5
    tracking_persistence_frames: int = 4
    stall_window_frames: int = 8
    stall_arm_command_delta_rad: float = 0.15
    stall_arm_state_delta_rad: float = 0.02
    stall_max_applied_command_advance: int = 1
    strict_guard_before_frames: int = 0
    strict_guard_after_frames: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.policy, QualityPolicy):
            raise TypeError("quality policy must be a QualityPolicy")
        positive_floats = (
            self.abrupt_arm_step_rad,
            self.abrupt_hand_step_rad,
            self.impulse_arm_min_rad,
            self.impulse_hand_min_rad,
            self.stall_arm_command_delta_rad,
            self.stall_arm_state_delta_rad,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError(
                "temporal quality radian thresholds must be finite and positive"
            )
        if (
            not np.isfinite(self.impulse_min_return_ratio)
            or not 0.0 < self.impulse_min_return_ratio <= 1.0
        ):
            raise ValueError("impulse_min_return_ratio must be in (0, 1]")
        positive_ints = (self.tracking_persistence_frames, self.stall_window_frames)
        if any(not isinstance(value, int) or value <= 0 for value in positive_ints):
            raise ValueError("temporal quality window sizes must be positive integers")
        nonnegative_ints = (
            self.stall_max_applied_command_advance,
            self.strict_guard_before_frames,
            self.strict_guard_after_frames,
        )
        if any(not isinstance(value, int) or value < 0 for value in nonnegative_ints):
            raise ValueError(
                "temporal quality counts and guards must be non-negative integers"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "abrupt_arm_step_rad": self.abrupt_arm_step_rad,
            "abrupt_hand_step_rad": self.abrupt_hand_step_rad,
            "impulse_arm_min_rad": self.impulse_arm_min_rad,
            "impulse_hand_min_rad": self.impulse_hand_min_rad,
            "impulse_min_return_ratio": self.impulse_min_return_ratio,
            "tracking_persistence_frames": self.tracking_persistence_frames,
            "stall_window_frames": self.stall_window_frames,
            "stall_arm_command_delta_rad": self.stall_arm_command_delta_rad,
            "stall_arm_state_delta_rad": self.stall_arm_state_delta_rad,
            "stall_max_applied_command_advance": self.stall_max_applied_command_advance,
            "strict_guard_before_frames": self.strict_guard_before_frames,
            "strict_guard_after_frames": self.strict_guard_after_frames,
        }


@dataclass(frozen=True)
class ProcessingConfig:
    """Resolved, immutable processing policy.

    Soft quality metrics never remove individual rows.  An explicit strict
    temporal policy may exclude only its high-confidence findings before
    ``horizon`` and ``min_full_windows`` admit the compact episode.
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
    arm_joint_limit_lower_rad: tuple[float, ...] = arm.joint_limit_lower
    arm_joint_limit_upper_rad: tuple[float, ...] = arm.joint_limit_upper
    hand_state_limit_lower_rad: tuple[float, ...] = hand.mechanical_qpos_min_rad
    hand_state_limit_upper_rad: tuple[float, ...] = hand.mechanical_qpos_max_rad
    hand_action_limit_lower_rad: tuple[float, ...] = hand.qpos_min_rad
    hand_action_limit_upper_rad: tuple[float, ...] = hand.qpos_max_rad
    tracking_error_warn_rad: float = arm.tracking_error_warn_rad
    temporal_quality: TemporalQualityConfig = field(
        default_factory=TemporalQualityConfig
    )
    bridge_policy: BridgePolicy = BridgePolicy.REJECT

    def __post_init__(self) -> None:
        if not isinstance(self.profile, OutputProfile):
            raise TypeError("profile must be an OutputProfile")
        if not isinstance(self.temporal_quality, TemporalQualityConfig):
            raise TypeError("temporal_quality must be a TemporalQualityConfig")
        if not isinstance(self.bridge_policy, BridgePolicy):
            raise TypeError("bridge_policy must be a BridgePolicy")
        positive_ints = (
            self.horizon,
            self.min_full_windows,
            self.target_rgb_height,
            self.target_rgb_width,
            self.target_point_count,
        )
        if any(not isinstance(value, int) or value <= 0 for value in positive_ints):
            raise ValueError(
                "horizon, window count, target image size, and point count must be positive integers"
            )
        finite_positive = (
            self.max_camera_age_s,
            self.grid_dt_relative_tolerance,
            self.joint_limit_tolerance_rad,
            self.hand_state_limit_tolerance_rad,
        )
        if not all(np.isfinite(value) and value > 0 for value in finite_positive):
            raise ValueError(
                "camera age, grid tolerance, and joint tolerance must be finite and positive"
            )
        if not 0 <= self.gzip_level <= 9:
            raise ValueError("gzip_level must be in [0, 9]")
        limit_pairs = (
            (
                "arm_joint_limit",
                "arm_joint_limit_lower_rad",
                "arm_joint_limit_upper_rad",
                7,
            ),
            (
                "hand_state_limit",
                "hand_state_limit_lower_rad",
                "hand_state_limit_upper_rad",
                12,
            ),
            (
                "hand_action_limit",
                "hand_action_limit_lower_rad",
                "hand_action_limit_upper_rad",
                12,
            ),
        )
        for label, lower_name, upper_name, size in limit_pairs:
            lower = tuple(float(value) for value in getattr(self, lower_name))
            upper = tuple(float(value) for value in getattr(self, upper_name))
            object.__setattr__(self, lower_name, lower)
            object.__setattr__(self, upper_name, upper)
            lower_array = np.asarray(lower, dtype=np.float64)
            upper_array = np.asarray(upper, dtype=np.float64)
            if lower_array.shape != (size,) or upper_array.shape != (size,):
                raise ValueError(f"{label} bounds must each have shape ({size},)")
            if not np.all(np.isfinite(lower_array)) or not np.all(
                np.isfinite(upper_array)
            ):
                raise ValueError(f"{label} bounds must be finite")
            if np.any(lower_array >= upper_array):
                raise ValueError(
                    f"{label} lower bounds must be strictly below upper bounds"
                )
        if (
            not np.isfinite(self.tracking_error_warn_rad)
            or self.tracking_error_warn_rad <= 0.0
        ):
            raise ValueError("tracking_error_warn_rad must be finite and positive")

    @property
    def min_episode_frames(self) -> int:
        return self.horizon + self.min_full_windows - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "horizon": self.horizon,
            "min_full_windows": self.min_full_windows,
            "min_episode_frames": self.min_episode_frames,
            "target_rgb_height": self.target_rgb_height,
            "target_rgb_width": self.target_rgb_width,
            "target_point_count": self.target_point_count,
            "max_camera_age_s": self.max_camera_age_s,
            "grid_dt_relative_tolerance": self.grid_dt_relative_tolerance,
            "joint_limit_tolerance_rad": self.joint_limit_tolerance_rad,
            "hand_state_limit_tolerance_rad": self.hand_state_limit_tolerance_rad,
            "gzip_level": self.gzip_level,
            "arm_joint_limit_lower_rad": list(self.arm_joint_limit_lower_rad),
            "arm_joint_limit_upper_rad": list(self.arm_joint_limit_upper_rad),
            "hand_state_limit_lower_rad": list(self.hand_state_limit_lower_rad),
            "hand_state_limit_upper_rad": list(self.hand_state_limit_upper_rad),
            "hand_action_limit_lower_rad": list(self.hand_action_limit_lower_rad),
            "hand_action_limit_upper_rad": list(self.hand_action_limit_upper_rad),
            "tracking_error_warn_rad": self.tracking_error_warn_rad,
            "temporal_quality": self.temporal_quality.to_dict(),
            "bridge_policy": self.bridge_policy.value,
        }


@dataclass(frozen=True)
class EpisodeAnnotation:
    """Optional row selection/task metadata; task outcome is intentionally absent."""

    include: bool = True
    task_name: str | None = None
    include_ranges: tuple[tuple[int, int], ...] = ()
    exclude_ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.task_name is not None and not self.task_name.strip():
            raise ValueError("task_name must be non-empty when provided")
        for label, ranges in (
            ("include_ranges", self.include_ranges),
            ("exclude_ranges", self.exclude_ranges),
        ):
            for start, end in ranges:
                if start < 0 or end <= start:
                    raise ValueError(
                        f"{label} entries must be non-negative half-open [start, end) ranges"
                    )


@dataclass(frozen=True)
class EpisodeDecision:
    """Complete auditable cleaning decision for one source episode."""

    source_path: Path
    source_frames: int
    profile: OutputProfile
    selected_indices: np.ndarray
    keep_mask: np.ndarray
    drop_reason_bits: np.ndarray
    drop_reason_names: tuple[str, ...]
    hard_reason_counts: dict[str, int]
    boundary_counts: dict[str, int]
    selected_frames: int
    quality: dict[str, Any]
    bridge_findings: tuple[dict[str, Any], ...] = ()
    temporal_quality: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    rejected_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected_reason is None and self.selected_frames > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_episode": self.source_path.name,
            "source_path": str(self.source_path),
            "profile": self.profile.value,
            "accepted": self.accepted,
            "rejected_reason": self.rejected_reason,
            "source_frames": self.source_frames,
            "selected_frames": self.selected_frames,
            "retention_ratio": (
                self.selected_frames / self.source_frames if self.source_frames else 0.0
            ),
            "hard_reason_counts": self.hard_reason_counts,
            "boundary_counts": self.boundary_counts,
            "dropped_frames": self.source_frames - self.selected_frames,
            "full_window_count": self.quality.get("full_window_count", 0),
            "selected_source_ranges": _indices_to_ranges(self.selected_indices),
            "bridge_findings": list(self.bridge_findings),
            "quality": self.quality,
            "temporal_quality": self.temporal_quality,
            "warnings": list(self.warnings),
        }


def _indices_to_ranges(indices: np.ndarray) -> list[list[int]]:
    """Compact sorted source-row indices into half-open ranges for reports."""

    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 1) + 1]
    ends = np.r_[starts[1:], len(values)]
    return [
        [int(values[start]), int(values[end - 1] + 1)]
        for start, end in zip(starts, ends)
    ]
