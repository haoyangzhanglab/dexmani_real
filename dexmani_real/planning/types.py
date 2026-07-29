"""Core planning types — Pose, IKResult, PathResult, config/profile dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from dexmani_real.utils.serialization import FromDictMixin

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


@dataclass
class IKStats:
    """Aggregate IK test statistics (motion planning benchmarks)."""

    ok: int
    total: int = 0
    pos_errs_mm: list[float] = field(default_factory=list)
    rot_errs_deg: list[float] = field(default_factory=list)
    max_dq_deg: list[float] = field(default_factory=list)


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

    # Unified collision configuration (FCL table obstacle, tier margins).
    # None disables environment collision detection (backward compatible).
    collision: CollisionConfig | None = None


@dataclass(kw_only=True)
class PlanningProfile(FromDictMixin):
    """Offline path planning configuration."""

    path_dt: float = 1 / 15
    planning_limits_deg: tuple[tuple[float, float], ...] | None = None
    max_ik_delta_deg: tuple[float, ...] = (120, 135, 120, 120, 180, 150, 180)
    max_waypoint_delta_deg: float = 15.0  # relaxed from 8° — execution always interpolates at 1° resolution anyway
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

    max_ik_jump_deg: tuple[float, ...] = (90, 90, 90, 90, 90, 90, 90)
    # Speed limiting is handled by ArmInnerLoop (30 Hz, Mode 6): per-step joint
    # delta clamp + firmware online trajectory planning (joint_max_speed/acc).
    max_pose_error_pos_m: float = 0.008
    max_pose_error_rot_rad: float = 0.08
    check_self_collision: bool = True  # checked in teleop IK hot path; holds on collision
    check_env_collision: bool = True  # env (table/obstacle) collision gate; holds on contact

    # Fast-accept threshold for position IK fallback (ref: ssik seed_tolerance).
    # A candidate is accepted immediately without trying additional seeds when
    # max single-joint delta from current hardware position is below this value.
    # 15° default: accept quickly if the solution is close to where we already are.
    position_ik_fast_accept_rad: float = np.deg2rad(15.0)

    use_position_ik: bool = True

    # ── Multi-candidate scoring (Phase 2, dexterous manipulation) ──
    # When the fast-accept path (prev_cmd seed within 15°) doesn't trigger,
    # multiple seeds are tried and candidates are scored by:
    #   score = weighted_joint_distance - manipulability_weight * μ + limit_penalty_weight * penalty
    # Lower score = better.  μ is the Yoshikawa manipulability measure.
    position_ik_num_random_seeds: int = 3  # extra random seeds around prev_cmd
    position_ik_seed_offset_deg: float = 5.0  # ±offset per joint for random seeds
    teleop_ik_seed: int | None = 42  # deterministic RNG seed (set None for non-det legacy behavior)
    position_ik_manipulability_weight: float = 0.05  # higher → prefer dexterous configs
    position_ik_limit_penalty_weight: float = 0.01  # higher → prefer configs farther from limits
    # Soft velocity tiebreaker: prefer candidates close to the previous command.
    # This is NOT a hard ceiling (the 90° jump guard already handles that) — it is a
    # lightweight preference that breaks ties between equally-valid IK solutions by
    # penalising frame-to-frame oscillation.  Set to 0.0 to disable.
    position_ik_velocity_weight: float = 0.03
    # EEF pose accuracy in multi-candidate scoring.  Within the acceptable range
    # (≤ max_pose_error_pos_m / max_pose_error_rot_rad), some candidates match
    # the target pose more precisely than others.  This weight adds pos_err + rot_err
    # (both normalised to their respective max thresholds) to the score.
    # Set to 0.0 to disable (backward compatible).
    position_ik_pose_accuracy_weight: float = 0.05
    # Minimum Yoshikawa manipulability μ = sqrt(det(J·Jᵀ)) for IK candidates.
    # Candidates below this threshold are rejected in validation (before scoring),
    # preventing large joint motions near kinematic singularities.
    # 0.0 = disabled (backward compatible).  For xArm7, μ ≈ 0.02–0.15 in normal
    # operation; values below 0.002 indicate near-singular configurations.
    position_ik_min_manipulability: float = 0.0
    # Per-joint weights for the velocity term.  Unlike joint_weights (which penalise
    # static displacement from hardware position), velocity weights penalise frame-to-frame
    # command changes and are tuned for joint INERTIA and RESPONSIVENESS rather than range:
    #
    #   J1 base (high inertia):       5.0 — heavy, resist oscillation aggressively
    #   J2 shoulder (highest inertia): 1.5 — heaviest joint, moderate extra damping
    #   J3 elbow:                      0.8 — neutral, slight relaxation vs position
    #   J4 wrist pitch:                0.5 — small range provides natural penalty
    #   J5 wrist roll (dexterity):     0.2 — primary manipulation axis, high freedom
    #   J6 wrist yaw (dexterity):      0.3 — secondary orientation axis, high freedom
    #   J7 tool flange (negligible):   0.1 — negligible inertia, near-zero resistance
    #
    # Effective velocity cost per radian (weight ÷ joint_range):
    #   J1(0.398) > J2(0.361) > J4(0.121) > J3(0.064)
    #   > J6(0.062) > J5(0.016) > J7(0.005)
    #
    # Contrast with position cost: J1 rises from #2→#1 (now most damped),
    # J6 drops #3→#5 (2.7× freer), J5/J7 get 3-5× more freedom.
    # Set to None to fall back to joint_weights (backward compatible).
    velocity_joint_weights: tuple[float, ...] | None = (5.0, 1.5, 0.8, 0.5, 0.2, 0.3, 0.1)

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

    # ── Null-space optimization ──
    # Post-IK null-space projection that adjusts the redundant DOF to repel
    # joints from their limits without altering the EEF pose (J · dq_null = 0).
    # Enabled by default: the ~130 us overhead is negligible at 30 Hz, all
    # safety gates (collision, pre-send, step-limit) run after this step,
    # and it cannot degrade EEF tracking by construction.
    enable_nullspace_optimization: bool = True
    nullspace_step_size_deg: float = 1.0  # max null-space joint step per frame [deg]
    nullspace_joint_limit_margin_deg: float = 15.0  # repulsion margin from limits [deg]
