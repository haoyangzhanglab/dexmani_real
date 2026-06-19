"""Core planning types — Pose, IKResult, PathResult, config/profile dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .collision_config import CollisionConfig


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


@dataclass(kw_only=True)
class PathResult:
    success: bool
    qpos_path: np.ndarray | None
    reason: str = ""
    source: str = ""
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class XArm7PlannerConfig:
    urdf_path: str
    srdf_path: str
    eef_link_name: str = "custom_eef_link"
    base_pose_world: Pose = field(default_factory=Pose.identity)
    use_convex: bool = False
    joint_vel_limits_deg: tuple[float, ...] = (60, 60, 60, 60, 90, 90, 120)
    joint_acc_scale: float = 2.0
    # Cartesian workspace bounds (world frame). (3,2) [[x_min,x_max],[y_min,y_max],[z_min,z_max]].
    # None disables the check (backward compatible).
    workspace_bounds: np.ndarray | None = None

    # Unified collision configuration (desk safety, hand margins, fingertip FK).
    # None disables geometric FK desk safety checks (backward compatible).
    collision: CollisionConfig | None = None


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
    check_env_collision: bool = True

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
    max_ik_jump_deg: tuple[float, ...] = (30, 30, 30, 30, 45, 45, 60)
    # Speed limiting is handled exclusively by XArm7._limit_joint_step()
    # (hardware driver layer, per BunnyVisionPro architecture).
    # max_qpos_cmd_speed_deg was removed — IK/planning layer should not clip speed.
    max_pose_error_pos_m: float = 0.008
    max_pose_error_rot_rad: float = 0.08
    check_self_collision: bool = True  # checked in teleop IK hot path; holds on collision

    use_position_ik: bool = True
    use_differential_ik_fallback: bool = True
    differential_ik_gain: float = 1.0       # full tracking, bottleneck limit handles speed
    differential_ik_damping: float = 0.02
    # NOTE: damping=0.02 (λ²=0.0004) is a "single-step DLS" design choice.
    # Single-step DLS avoids the ~100 iteration cost of iterative DLS
    # (BunnyVisionPro uses damping=1e-5 with 100 iterations) at the cost of
    # ~1-2 mm extra IK error in non-singular regions — the λ² term in the
    # damped pseudo-inverse (J·Jᵀ + λ²I)⁻¹ biases the solution away from the
    # optimal least-squares direction even when far from singularities.
    # Trade-off: 1 ms latency vs ~1-2 mm precision. Acceptable for teleop
    # where human hand tremor (~2-3 mm) dominates. For precision tasks,
    # consider an adaptive damping schedule (lower λ² in non-singular regions).
    differential_ik_max_pos_step_m: float = 0.02
    differential_ik_max_rot_step_rad: float = np.deg2rad(5.0)

