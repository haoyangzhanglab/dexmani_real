"""Core planning types — Pose, IKResult, PathResult, config/profile dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.utils.serialization import from_dict_helper

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

    @classmethod
    def from_dict(cls, d: dict) -> "PlanningProfile":
        kw = from_dict_helper(cls, d)
        return cls(**kw)


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

    # Fast-accept threshold for position IK fallback (ref: ssik seed_tolerance).
    # A candidate is accepted immediately without trying additional seeds when
    # max single-joint delta from current hardware position is below this value.
    # 15° default: accept quickly if the solution is close to where we already are.
    position_ik_fast_accept_rad: float = np.deg2rad(15.0)

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

    # ── Adaptive damping (NEW) ──
    # When True, damping scales with manipulability:
    #   high manipulability (far from singularity) → min_damping (~0.001)
    #   low manipulability (near singularity) → max_damping (~0.05)
    # Enabled by default (2026-06-22): the redundant Jacobian computation
    # in the adaptive path has been eliminated (ik.py reuses the pre-computed
    # Jacobian via manipulability_from_jacobian), so there is no performance
    # cost.  Set to False to use fixed differential_ik_damping.
    adaptive_damping: bool = True

    # Min damping in non-singular regions (near-zero to minimize tracking bias).
    differential_ik_min_damping: float = 0.001

    # Max damping near singularities (prevents unsafe joint velocities).
    differential_ik_max_damping: float = 0.05

    # Manipulability threshold below which damping begins to ramp up.
    # Typical XArm7 values: ~0.01 (far from singularity), ~0.001 (near elbow singularity).
    manipulability_threshold: float = 0.005

    # Minimum manipulability for IK acceptance.
    # qpos solutions with manipulability below this value are rejected.
    # Retries with heavier damping (singularity_damping_scale × damping)
    # before falling through to position IK.
    # 0.001 is near-elbow-singularity for XArm7 (far-from-singularity ≈ 0.01).
    # Set to 0.0 to disable (backward compatible).
    # Ref: LeFranX weighted_ik.cpp:71-76 Yoshikawa manipulability scoring.
    min_manipulability: float = 0.001

    # Damping multiplier when manipulability falls below min_manipulability.
    singularity_damping_scale: float = 10.0

    # ── Cartesian pose interpolation (NEW) ──
    # When True, CartPoseInterpolator runs between VR input and IK,
    # linearly interpolating position and SLERP-interpolating rotation.
    # Eliminates stale VR frame re-use when VR rate < control rate.
    # Ref: ManiUniCon PoseTrajectoryInterpolator.
    use_cartesian_interpolation: bool = False

    # Speed limits for interpolator (prevent sudden jumps).
    interpolation_max_pos_speed: float = 0.25   # m/s
    interpolation_max_rot_speed: float = 0.5    # rad/s

    @classmethod
    def from_dict(cls, d: dict) -> "TeleopProfile":
        kw = from_dict_helper(cls, d)
        return cls(**kw)

