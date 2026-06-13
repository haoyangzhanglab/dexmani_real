from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


def build_brief_dict(report: dict[str, Any], keys: tuple[tuple[str, str], ...], max_items: int) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for source_key, display_key in keys:
        value = report.get(source_key)
        if value is None or (isinstance(value, str) and value == ""):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            value = round(value, 6)
        compact[display_key] = value
        if len(compact) >= max_items:
            break
    return compact


def format_brief(
    label_prefix: str, success: bool, reason: str, compact: dict[str, Any], max_reason_lines: int = 3
) -> str:
    label = f"{label_prefix} success" if success else f"{label_prefix} failure"
    reason_str = str(reason or "").strip() or ("ok" if success else "unknown")
    lines = [line.strip() for line in reason_str.splitlines() if line.strip()]
    reason_text = " | ".join(lines[:max_reason_lines]) or reason_str
    if compact:
        return f"{label}: {reason_text}\n{label_prefix} summary: {compact}"
    return f"{label}: {reason_text}"


IK_BRIEF_KEYS: tuple[tuple[str, str], ...] = (
    ("held", "held"),
    ("teleop_ik_method", "method"),
    ("fallback_method", "fallback"),
    ("differential_ik_status", "diff_status"),
    ("teleop_ik_success_count", "ik_success"),
    ("teleop_ik_rejected_success_count", "ik_rejected"),
    ("max_raw_delta_deg", "max_raw_delta_deg"),
    ("cmd_tracking_error_pos_m", "tracking_pos_m"),
    ("raw_pose_error_pos_m", "raw_pos_err_m"),
)


PATH_BRIEF_KEYS: tuple[tuple[str, str], ...] = (
    ("source", "source"),
    ("num_waypoints", "waypoints"),
    ("joint_path_length", "joint_path_len"),
    ("max_waypoint_delta_rad", "max_step_rad"),
    ("terminal_pos_error_m", "terminal_pos_err_m"),
    ("terminal_rot_error_rad", "terminal_rot_err_rad"),
    ("num_planning_attempts", "attempts"),
    ("num_valid_plans", "valid_plans"),
    ("max_limit_violation_deg", "max_limit_violation_deg"),
)


@dataclass
class Pose:
    """Pose with position and wxyz quaternion."""

    p: np.ndarray
    q: np.ndarray

    def __init__(self, p: Any, q: Any) -> None:
        self.p = np.asarray(p, dtype=np.float64).reshape(3)
        self.q = np.asarray(q, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(self.q)
        if norm <= 1e-12:
            raise ValueError("Pose quaternion norm is too small.")
        self.q = self.q / norm

    @classmethod
    def identity(cls) -> "Pose":
        return cls(p=np.zeros(3, dtype=np.float64), q=np.array([1.0, 0.0, 0.0, 0.0]))

    def copy(self) -> "Pose":
        return Pose(p=self.p.copy(), q=self.q.copy())


@dataclass(kw_only=True)
class IKResult:
    success: bool
    qpos: np.ndarray | None
    reason: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    held: bool = False

    def brief_dict(self, max_items: int = 8) -> dict[str, Any]:
        return build_brief_dict(self.report or {}, IK_BRIEF_KEYS, max_items)

    def brief(self, max_reason_lines: int = 3, max_items: int = 8) -> str:
        return format_brief("IK", self.success, self.reason, self.brief_dict(max_items=max_items), max_reason_lines)


@dataclass(kw_only=True)
class PathResult:
    success: bool
    qpos_path: np.ndarray | None
    reason: str = ""
    source: str = ""
    report: dict[str, Any] = field(default_factory=dict)

    def brief_dict(self, max_items: int = 10) -> dict[str, Any]:
        report = {**self.report, "source": self.source}
        return build_brief_dict(report, PATH_BRIEF_KEYS, max_items)

    def brief(self, max_reason_lines: int = 3, max_items: int = 10) -> str:
        return format_brief("Path", self.success, self.reason, self.brief_dict(max_items=max_items), max_reason_lines)


@dataclass(kw_only=True)
class XArm7PlannerConfig:
    urdf_path: str
    srdf_path: str
    eef_link_name: str = "custom_eef_link"
    base_pose_world: Pose = field(default_factory=Pose.identity)
    use_convex: bool = False
    joint_vel_limits_deg: tuple[float, ...] = (60, 60, 60, 60, 90, 90, 120)
    joint_acc_scale: float = 2.0


@dataclass(kw_only=True)
class PlanningProfile:
    """Offline path planning configuration."""

    path_dt: float = 1 / 15
    planning_limits_deg: tuple[tuple[float, float], ...] | None = None
    max_ik_delta_deg: tuple[float, ...] = (120, 135, 120, 120, 180, 150, 180)
    max_waypoint_delta_deg: float = 8.0
    max_pose_error_pos_m: float = 0.005
    max_pose_error_rot_rad: float = 0.05

    rrt_time_limit: float = 2.0
    rrt_range_options: tuple[float, ...] = (0.05, 0.08, 0.12)
    num_rrt_attempts: int = 4
    simplify_path: bool = True
    screw_qpos_step: float = 0.02

    ik_seed_offsets_deg: tuple[float, ...] = (15.0, 8.0, 15.0, 3.0, 15.0, 8.0, 15.0)
    num_random_ik_seeds: int = 15
    num_ik_candidates: int = 6
    n_init_qpos: int = 3
    random_seed: int | None = None
    check_self_collision: bool = True

    neutral_qpos: np.ndarray | None = None
    ik_score_manipulability_weight: float = 1.0
    ik_score_neutral_weight: float = 0.5
    ik_score_joint_delta_weight: float = 1.0
    ik_score_pose_error_weight: float = 0.2
    ik_score_joint_limit_weight: float = 0.2


@dataclass(kw_only=True)
class TeleopProfile:
    """Online teleoperation IK/servo configuration."""

    teleop_dt: float = 0.04
    max_qpos_cmd_speed_deg: tuple[float, ...] = (90, 90, 90, 90, 120, 120, 150)
    max_ik_jump_deg: tuple[float, ...] = (90, 90, 90, 90, 120, 120, 180)
    max_pose_error_pos_m: float = 0.008
    max_pose_error_rot_rad: float = 0.08
    hold_on_failure: bool = True

    use_position_ik: bool = True
    use_differential_ik_fallback: bool = True
    differential_ik_gain: float = 0.6
    differential_ik_damping: float = 0.05
    differential_ik_max_pos_step_m: float = 0.02
    differential_ik_max_rot_step_rad: float = np.deg2rad(5.0)
    debug: bool = False

