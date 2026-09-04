"""Typed contracts for offline episode cleaning and mapping."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, environment, hand, policy
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.robot_spec import XHAND_RIGHT_URDF_PATH


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


@dataclass(frozen=True)
class TemporalQualityConfig:
    """Resolved thresholds for conservative temporal anomaly detection.

    Abrupt steps and persistent tracking error are audit-only evidence.  Strict
    processing excludes only reversible one-frame command impulses and
    high-confidence arm command/feedback stalls.
    """

    policy: QualityPolicy = QualityPolicy.AUDIT
    abrupt_arm_step_rad: float = float(np.deg2rad(8.0))
    # This is a raw-episode sampling-quality threshold, not the hand worker's
    # servo-tick limit. It must remain independent when those rates differ.
    abrupt_hand_step_rad: float = 0.25
    impulse_arm_min_rad: float = 0.08
    impulse_hand_min_rad: float = 0.12
    impulse_min_return_ratio: float = 0.5
    tracking_persistence_frames: int = 4
    # Number of samples included in each feedback/command stall window. The
    # first and last sample are compared, so a valid window requires >= 2.
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
        if (
            isinstance(self.tracking_persistence_frames, bool)
            or not isinstance(self.tracking_persistence_frames, int)
            or self.tracking_persistence_frames <= 0
        ):
            raise ValueError("tracking_persistence_frames must be a positive integer")
        if (
            isinstance(self.stall_window_frames, bool)
            or not isinstance(self.stall_window_frames, int)
            or self.stall_window_frames < 2
        ):
            raise ValueError("stall_window_frames must be an integer >= 2")
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
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = (
        environment.table.plane_abcd
    )
    max_camera_age_s: float = 0.25
    # Must match the learned-policy deployment observation-pairing budget.
    # It is persisted in each processed artifact so a checkpoint cannot silently
    # mix a looser recording policy with a stricter deployment policy.
    max_observation_skew_s: float = 0.10
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
    hand_urdf_path: str = str(XHAND_RIGHT_URDF_PATH)
    fingertip_link_names: tuple[str, ...] = hand.fingertip_link_names
    handbase_position_eef_m: tuple[float, float, float] = hand.T_eef_handbase_pos_xyz
    handbase_quat_eef_wxyz: tuple[float, float, float, float] = (
        hand.T_eef_handbase_quat_wxyz
    )
    # Learned-policy endpoints are rejected, never clipped, by the deployment
    # coordinator.  Offline processing therefore applies the same per-grid
    # contract before an episode can enter a deployment training set.
    arm_max_delta_rad_per_tick: float | None = policy.arm_max_delta_rad_per_tick
    hand_max_delta_rad_per_tick: float = hand.hand_max_delta_rad_per_tick
    # Shared numerical slack for the reject-only endpoint-delta predicate.
    endpoint_delta_tolerance_rad: float = policy.endpoint_delta_tolerance_rad
    tracking_error_warn_rad: float = arm.tracking_error_warn_rad
    temporal_quality: TemporalQualityConfig = field(
        default_factory=TemporalQualityConfig
    )

    @classmethod
    def from_runtime(
        cls,
        runtime: object,
        *,
        profile: OutputProfile,
        **overrides: Any,
    ) -> "ProcessingConfig":
        """Project one resolved runtime snapshot into offline processing policy."""
        arm_config = getattr(runtime, "arm")
        hand_config = getattr(runtime, "hand")
        environment_config = getattr(runtime, "environment")
        table = environment_config.table
        values: dict[str, Any] = {
            "profile": profile,
            "pointcloud": getattr(runtime, "pointcloud"),
            "table_plane_abcd": table.plane_abcd if table.enabled else None,
            "arm_joint_limit_lower_rad": tuple(arm_config.joint_limit_lower),
            "arm_joint_limit_upper_rad": tuple(arm_config.joint_limit_upper),
            "hand_state_limit_lower_rad": tuple(hand_config.mechanical_qpos_min_rad),
            "hand_state_limit_upper_rad": tuple(hand_config.mechanical_qpos_max_rad),
            "hand_action_limit_lower_rad": tuple(hand_config.qpos_min_rad),
            "hand_action_limit_upper_rad": tuple(hand_config.qpos_max_rad),
            "hand_urdf_path": str(XHAND_RIGHT_URDF_PATH),
            "fingertip_link_names": tuple(hand_config.fingertip_link_names),
            "handbase_position_eef_m": tuple(hand_config.T_eef_handbase_pos_xyz),
            "handbase_quat_eef_wxyz": tuple(hand_config.T_eef_handbase_quat_wxyz),
            "arm_max_delta_rad_per_tick": getattr(
                runtime, "policy"
            ).arm_max_delta_rad_per_tick,
            "hand_max_delta_rad_per_tick": float(
                hand_config.hand_max_delta_rad_per_tick
            ),
            "endpoint_delta_tolerance_rad": float(
                getattr(runtime, "policy").endpoint_delta_tolerance_rad
            ),
            "tracking_error_warn_rad": float(arm_config.tracking_error_warn_rad),
        }
        unknown = set(overrides) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise TypeError(f"unknown ProcessingConfig override(s): {sorted(unknown)}")
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, OutputProfile):
            raise TypeError("profile must be an OutputProfile")
        if not isinstance(self.temporal_quality, TemporalQualityConfig):
            raise TypeError("temporal_quality must be a TemporalQualityConfig")
        if not isinstance(self.pointcloud, PointCloudConfig):
            raise TypeError("pointcloud must be a PointCloudConfig")
        if not self.hand_urdf_path or len(self.fingertip_link_names) != 5:
            raise ValueError("fingertip geometry requires a URDF and five link names")
        mount_values = (
            *self.handbase_position_eef_m,
            *self.handbase_quat_eef_wxyz,
        )
        if (
            len(self.handbase_position_eef_m) != 3
            or len(self.handbase_quat_eef_wxyz) != 4
            or not np.all(np.isfinite(mount_values))
        ):
            raise ValueError("fingertip hand-mount transform is invalid")
        positive_ints = (
            self.horizon,
            self.min_full_windows,
            self.target_rgb_height,
            self.target_rgb_width,
        )
        if any(not isinstance(value, int) or value <= 0 for value in positive_ints):
            raise ValueError(
                "horizon, window count, and target image size must be positive integers"
            )
        if self.table_plane_abcd is not None:
            table_plane = tuple(float(value) for value in self.table_plane_abcd)
            if len(table_plane) != 4 or not np.all(np.isfinite(table_plane)):
                raise ValueError("table_plane_abcd must contain four finite values")
            norm = np.linalg.norm(table_plane[:3])
            if norm <= 0.0 or table_plane[2] / norm <= 0.0:
                raise ValueError("table_plane_abcd normal must point upward")
            object.__setattr__(self, "table_plane_abcd", table_plane)
        finite_positive = (
            self.max_camera_age_s,
            self.max_observation_skew_s,
            self.grid_dt_relative_tolerance,
            self.joint_limit_tolerance_rad,
            self.hand_state_limit_tolerance_rad,
        )
        if not all(np.isfinite(value) and value > 0 for value in finite_positive):
            raise ValueError(
                "camera age, grid tolerance, and joint tolerance must be finite and positive"
            )
        if self.arm_max_delta_rad_per_tick is not None and (
            not np.isfinite(self.arm_max_delta_rad_per_tick)
            or self.arm_max_delta_rad_per_tick <= 0.0
        ):
            raise ValueError(
                "arm_max_delta_rad_per_tick must be finite and positive or None"
            )
        if (
            not np.isfinite(self.hand_max_delta_rad_per_tick)
            or self.hand_max_delta_rad_per_tick <= 0.0
        ):
            raise ValueError("hand_max_delta_rad_per_tick must be finite and positive")
        if (
            isinstance(self.endpoint_delta_tolerance_rad, bool)
            or not np.isfinite(self.endpoint_delta_tolerance_rad)
            or self.endpoint_delta_tolerance_rad < 0.0
        ):
            raise ValueError(
                "endpoint_delta_tolerance_rad must be finite and non-negative"
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
            "pointcloud": self.pointcloud.to_dict(),
            "table_plane_abcd": (
                None if self.table_plane_abcd is None else list(self.table_plane_abcd)
            ),
            "max_camera_age_s": self.max_camera_age_s,
            "max_observation_skew_s": self.max_observation_skew_s,
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
            "hand_urdf_path": self.hand_urdf_path,
            "fingertip_link_names": list(self.fingertip_link_names),
            "handbase_position_eef_m": list(self.handbase_position_eef_m),
            "handbase_quat_eef_wxyz": list(self.handbase_quat_eef_wxyz),
            "arm_max_delta_rad_per_tick": self.arm_max_delta_rad_per_tick,
            "hand_max_delta_rad_per_tick": self.hand_max_delta_rad_per_tick,
            "endpoint_delta_tolerance_rad": self.endpoint_delta_tolerance_rad,
            "tracking_error_warn_rad": self.tracking_error_warn_rad,
            "temporal_quality": self.temporal_quality.to_dict(),
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
    source_gap_findings: tuple[dict[str, Any], ...] = ()
    temporal_quality: dict[str, Any] = field(default_factory=dict)
    hard_invalid_reason_names: tuple[str, ...] = ()
    audit_reason_counts: dict[str, int] = field(default_factory=dict)
    repair_reason_counts: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    rejected_reason: str | None = None

    @property
    def segment_ends(self) -> np.ndarray:
        """Exclusive compact-row ends for source-contiguous policy episodes."""
        return build_source_segment_ends(
            self.selected_indices, self.source_gap_findings
        )

    @property
    def accepted(self) -> bool:
        return self.rejected_reason is None and self.selected_frames > 0

    def to_dict(self) -> dict[str, Any]:
        hard_invalid_ranges, hard_invalid_reasons = self._hard_invalid_report()
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
            "hard_invalid_reason_names": list(self.hard_invalid_reason_names),
            "hard_invalid_frame_count": sum(
                end - start for start, end in hard_invalid_ranges
            ),
            "hard_invalid_ranges": hard_invalid_ranges,
            "hard_invalid_reasons": hard_invalid_reasons,
            "audit_reason_counts": self.audit_reason_counts,
            "repair_reason_counts": self.repair_reason_counts,
            "boundary_counts": self.boundary_counts,
            "dropped_frames": self.source_frames - self.selected_frames,
            "full_window_count": self.quality.get("full_window_count", 0),
            "selected_source_ranges": _indices_to_ranges(self.selected_indices),
            "selected_segment_ends": self.segment_ends.tolist(),
            "source_gap_findings": list(self.source_gap_findings),
            "quality": self.quality,
            "temporal_quality": self.temporal_quality,
            "warnings": list(self.warnings),
        }

    def _hard_invalid_report(self) -> tuple[list[list[int]], list[dict[str, Any]]]:
        """Summarize only hard-invalid source rows from provenance reason bits."""

        reason_masks = {
            name: (self.drop_reason_bits & (np.uint64(1) << np.uint64(bit))) != 0
            for bit, name in enumerate(self.drop_reason_names)
            if name in self.hard_invalid_reason_names
        }
        hard_invalid = np.zeros(self.drop_reason_bits.shape, dtype=bool)
        for mask in reason_masks.values():
            hard_invalid |= mask
        reasons = [
            {
                "reason": name,
                "frame_count": int(np.count_nonzero(mask)),
                "ranges": _indices_to_ranges(np.flatnonzero(mask)),
            }
            for name, mask in reason_masks.items()
            if np.any(mask)
        ]
        return _indices_to_ranges(np.flatnonzero(hard_invalid)), reasons


def build_source_segment_ends(
    selected_indices: np.ndarray,
    source_gap_findings: tuple[dict[str, Any], ...],
) -> np.ndarray:
    """Map source-row discontinuities to compact-array exclusive ends."""
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    boundary_rows = {int(item["source_row_after"]) for item in source_gap_findings}
    ends = [
        compact_index
        for compact_index, source_row in enumerate(selected)
        if compact_index > 0 and int(source_row) in boundary_rows
    ]
    ends.append(int(selected.size))
    return np.asarray(ends, dtype=np.int64)


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
