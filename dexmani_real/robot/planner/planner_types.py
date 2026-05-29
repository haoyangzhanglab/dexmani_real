from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(kw_only=True)
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

    def brief_dict(self, max_items: int = 8) -> dict[str, Any]:
        report = self.report or {}
        keys = (
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
        compact: dict[str, Any] = {}
        for source_key, display_key in keys:
            value = report.get(source_key)
            if value is None:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float):
                value = round(value, 6)
            compact[display_key] = value
            if len(compact) >= max_items:
                break
        return compact

    def brief(self, max_reason_lines: int = 3, max_items: int = 8) -> str:
        label = "IK success" if self.success else "IK failure"
        reason = str(self.reason or "").strip() or ("ok" if self.success else "unknown")
        lines = [line.strip() for line in reason.splitlines() if line.strip()]
        reason_text = " | ".join(lines[:max_reason_lines]) or reason
        compact = self.brief_dict(max_items=max_items)
        if compact:
            return f"{label}: {reason_text}\nIK summary: {compact}"
        return f"{label}: {reason_text}"


@dataclass(kw_only=True)
class PathResult:
    success: bool
    qpos_path: np.ndarray | None
    reason: str = ""
    source: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    qvel_path: np.ndarray | None = None
    qacc_path: np.ndarray | None = None
    time_path: np.ndarray | None = None
    duration: float | None = None

    def brief_dict(self, max_items: int = 10) -> dict[str, Any]:
        report = self.report or {}
        keys = (
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
        compact: dict[str, Any] = {}
        for source_key, display_key in keys:
            value = self.source if source_key == "source" else report.get(source_key)
            if value is None or value == "":
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float):
                value = round(value, 6)
            compact[display_key] = value
            if len(compact) >= max_items:
                break
        return compact

    def brief(self, max_reason_lines: int = 3, max_items: int = 10) -> str:
        label = "Path success" if self.success else "Path failure"
        reason = str(self.reason or "").strip() or ("ok" if self.success else "unknown")
        lines = [line.strip() for line in reason.splitlines() if line.strip()]
        reason_text = " | ".join(lines[:max_reason_lines]) or reason
        compact = self.brief_dict(max_items=max_items)
        if compact:
            return f"{label}: {reason_text}\nPath summary: {compact}"
        return f"{label}: {reason_text}"


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

    path_dt: float = 0.05
    planning_limits_deg: tuple[tuple[float, float], ...] | None = None
    max_ik_delta_deg: tuple[float, ...] = (120, 135, 120, 120, 180, 150, 180)
    max_waypoint_delta_deg: float = 8.0
    max_pose_error_pos_m: float = 0.005
    max_pose_error_rot_rad: float = 0.05

    use_joint_interpolation: bool = True
    use_screw: bool = True
    use_rrt: bool = True
    rrt_time_limit: float = 2.0
    rrt_range_options: tuple[float, ...] = (0.05, 0.08, 0.12)
    num_rrt_attempts: int = 4
    simplify_path: bool = True
    screw_qpos_step: float = 0.02

    ik_seed_offsets_deg: tuple[float, ...] = (3.0, 8.0, 15.0)
    num_random_ik_seeds: int = 8
    num_ik_candidates: int = 6
    random_seed: int | None = 0

    enable_self_collision_check: bool = False
    enable_env_collision_check: bool = False
    debug: bool = False


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


@dataclass(kw_only=True)
class HandPlanningProfile:
    hand_dt: float = 0.05
    max_hand_qpos_speed: float = np.pi / 2.0
    max_hand_waypoint_delta: float | None = None
    sync_mode: str = "post"
