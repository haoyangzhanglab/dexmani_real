"""Core planning types — Pose, IKResult, PathResult, config/profile dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from dexmani_real.utils.serialization import FromDictMixin, from_dict_helper

if TYPE_CHECKING:
    from .collision_config import CollisionConfig


@dataclass(frozen=True, slots=True)
class CollisionPair:
    """A single self-collision contact between two links.

    Extracted from MPlib ``WorldCollisionResult`` into a lightweight,
    hashable, pickle-safe dataclass suitable for diagnostic dicts and logs.
    """

    link_name1: str
    link_name2: str
    object_name1: str
    object_name2: str
    collision_type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "link1": self.link_name1,
            "link2": self.link_name2,
            "obj1": self.object_name1,
            "obj2": self.object_name2,
            "type": self.collision_type,
        }


@dataclass
class CollisionInfo:
    """Structured self-collision diagnostic result.

    **Hot-path safety:** When no collision is detected, ``no_collision()``
    returns a module-level cached singleton — zero allocation beyond the
    underlying MPlib C++ ``check_for_self_collision`` call.

    When a collision exists, the ``WorldCollisionResult`` list is already
    computed by MPlib; wrapping it here only iterates the result strings
    (no second collision query).

    ``bool(collision_info)`` returns ``in_collision``, so existing code
    that checks ``if check_self_collision(q):`` continues to work.
    """

    in_collision: bool
    collision_pairs: tuple[CollisionPair, ...] = ()
    num_contacts: int = 0

    # Module-level cached singleton — set after the class body.
    _NO_COLLISION: ClassVar[CollisionInfo]

    @classmethod
    def no_collision(cls) -> CollisionInfo:
        """Return the cached no-collision singleton (zero allocation)."""
        return cls._NO_COLLISION

    @classmethod
    def from_mplib_results(cls, results: list[Any]) -> CollisionInfo:
        """Build ``CollisionInfo`` from a list of MPlib ``WorldCollisionResult``.

        ``results`` is the raw list returned by
        ``mp_planner.check_for_self_collision(qpos)``.
        """
        pairs = tuple(
            CollisionPair(
                link_name1=str(r.link_name1),
                link_name2=str(r.link_name2),
                object_name1=str(r.object_name1),
                object_name2=str(r.object_name2),
                collision_type=str(r.collision_type),
            )
            for r in results
        )
        return cls(in_collision=True, collision_pairs=pairs, num_contacts=len(pairs))

    def __bool__(self) -> bool:
        return self.in_collision

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict for ``IKResult.report`` integration."""
        if not self.in_collision:
            return {"in_collision": False}
        return {
            "in_collision": True,
            "num_contacts": self.num_contacts,
            "collision_pairs": [p.to_dict() for p in self.collision_pairs],
        }

    @property
    def summary(self) -> str:
        """One-line human-readable summary for log messages."""
        if not self.in_collision:
            return "no collision"
        pairs = [f"{p.link_name1}↔{p.link_name2}" for p in self.collision_pairs]
        return f"{self.num_contacts} contact(s): " + ", ".join(pairs)


CollisionInfo._NO_COLLISION = CollisionInfo(in_collision=False)


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
    joint_vel_limits_deg: tuple[float, ...] = (180, 180, 180, 180, 180, 180, 180)
    joint_acc_scale: float = 2.0
    # Cartesian workspace bounds (world frame). (3,2) [[x_min,x_max],[y_min,y_max],[z_min,z_max]].
    # None disables the check (backward compatible).
    #
    # NOTE: robot/types.py RobotInterfaceConfig also has workspace_bounds with a hardcoded
    # default. These are independent config paths — keep them in sync when tuning workspace.
    workspace_bounds: np.ndarray | None = None

    # Unified collision configuration (desk safety, hand margins, fingertip FK).
    # None disables geometric FK desk safety checks (backward compatible).
    collision: CollisionConfig | None = None


@dataclass(kw_only=True)
class PlanningProfile(FromDictMixin):
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
class TeleopProfile(FromDictMixin):
    """Online teleoperation IK/servo configuration."""

    teleop_dt: float = 0.04
    max_ik_jump_deg: tuple[float, ...] = (90, 90, 90, 90, 90, 90, 90)
    # Speed limiting is handled exclusively by XArm7._limit_joint_step()
    # (hardware driver layer, per BunnyVisionPro architecture).
    # max_qpos_cmd_speed_deg was removed — IK/planning layer should not clip speed.
    max_pose_error_pos_m: float = 0.008
    max_pose_error_rot_rad: float = 0.08
    check_self_collision: bool = True  # checked in teleop IK hot path; holds on collision
    check_env_collision: bool = True   # env (table/obstacle) collision gate; holds on contact

    # Fast-accept threshold for position IK fallback (ref: ssik seed_tolerance).
    # A candidate is accepted immediately without trying additional seeds when
    # max single-joint delta from current hardware position is below this value.
    # 15° default: accept quickly if the solution is close to where we already are.
    position_ik_fast_accept_rad: float = np.deg2rad(15.0)

    use_position_ik: bool = True
    use_differential_ik_fallback: bool = True

    # ── Iterative DLS (ref: BunnyVisionPro xarm7_ability.py:136-159 compute_ik) ──
    # Each iteration: FK → Jacobian → DLS solve → integrate.  Converges when
    # ||error|| < convergence_threshold or max_iterations reached.
    differential_ik_gain: float = 0.05  # step size per iteration (matches BVP v*0.05)
    differential_ik_damping: float = 0.003162  # λ = √(1e-5), matches BVP λ²=1e-5
    differential_ik_max_iterations: int = 100  # matches BVP for k in range(100)
    differential_ik_convergence_threshold: float = 1e-3  # matches BVP norm(err) < 1e-3

    # Step limits applied to the FINAL iteration only (not internal iterations).
    # These cap the per-frame Cartesian delta at the solver output, preventing
    # large joint jumps from unconverged DLS.  Set to inf for no limit.
    differential_ik_max_pos_step_m: float = 0.05
    differential_ik_max_rot_step_rad: float = np.deg2rad(5.0)

    # ── Adaptive damping (disabled by default — aligned with BVP fixed damping) ──
    adaptive_damping: bool = False

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

    # ── Joint-specific IK scoring weights (ref: LeFranX weighted_ik.cpp:62-69) ──
    # Higher weight → solver penalises moving that joint away from its current
    # position, i.e. "expensive" joints stay stable while "cheap" joints do the
    # tracking work.  Applied via weighted, range-normalised L2 distance.
    #
    # xArm7 joint semantics and tuning rationale:
    #   joint1 (base rotation, ±360°):     3.0 — huge range makes raw radians
    #                                             cheap; high weight keeps the base
    #                                             stable and avoids large arm swings
    #                                             for small VR hand motions.
    #   joint2 (shoulder lift, -118~120°): 1.2 — small range (238°) already
    #                                             provides natural normalisation
    #                                             penalty; moderate extra weight
    #                                             because shoulder is the heaviest
    #                                             joint yet essential for vertical
    #                                             VR tracking.
    #   joint3 (elbow, ±360°):             1.0 — neutral; elbow contributes to
    #                                             both reach and orientation and
    #                                             should move without bias.
    #   joint4 (wrist pitch, -11~225°):    0.5 — small range provides natural
    #                                             penalty; low weight lets wrist
    #                                             pitch track VR orientation freely
    #                                             within its asymmetric limits.
    #   joint5 (wrist roll, ±360°):        0.5 — low weight; wrist roll is the
    #                                             primary manipulation axis and
    #                                             has ample range.
    #   joint6 (wrist yaw, -97~180°):      0.8 — moderate range (277°); slight
    #                                             penalty relative to roll to
    #                                             prefer roll (joint5) over yaw
    #                                             for hand orientation changes.
    #   joint7 (tool flange, ±360°):       0.3 — lowest weight; tool rotation
    #                                             has negligible effect on arm
    #                                             configuration and should move
    #                                             most freely.
    #
    # Effective cost per radian (weight ÷ joint_range):
    #   J2(0.289) > J1(0.239) > J6(0.166) > J4(0.121)
    #   > J3(0.080) > J5(0.040) > J7(0.024)
    joint_weights: tuple[float, ...] = (3.0, 1.2, 1.0, 0.5, 0.5, 0.8, 0.3)

    # ── Cartesian pose interpolation (REMOVED 2026-06-24) ──
    # CartPoseInterpolator (linear + SLERP) was removed because:
    #   1. VR is native 50 Hz = control loop rate → no frequency decoupling needed
    #   2. EMA smoothing (ema_alpha_arm) already handles frame-to-frame filtering
    #   3. Velocity-limited step provides per-frame delta capping
    #   4. BunnyVisionPro / LeFranX / T-Rex all operate without Cartesian interpolation
    # The interpolator added ~20ms latency without measurable smoothness benefit.
    use_cartesian_interpolation: bool = False

    # Speed limits for interpolator (DEPRECATED — no longer used).
    # Kept for backward-compatible config deserialization.
    interpolation_max_pos_speed: float = 0.4  # m/s
    interpolation_max_rot_speed: float = 0.8  # rad/s

    # ── Null-space optimization ──
    # Post-IK null-space projection that adjusts the redundant DOF to repel
    # joints from their limits without altering the EEF pose (J · dq_null = 0).
    # Enabled by default: the ~130 us overhead is negligible at 50 Hz, all
    # safety gates (collision, pre-send, step-limit) run after this step,
    # and it cannot degrade EEF tracking by construction.
    enable_nullspace_optimization: bool = True
    nullspace_step_size_deg: float = 1.0       # max null-space joint step per frame [deg]
    nullspace_joint_limit_margin_deg: float = 15.0  # repulsion margin from limits [deg]
