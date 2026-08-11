"""Core planning types — Pose, IKResult, PathResult, config/profile dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from dexmani_real.utils.serialization import from_dict_helper


@dataclass(frozen=True, slots=True)
class CollisionPair:
    """A single self-collision contact between two links.

    Extracted from Pinocchio/hpp-fcl collision results into a lightweight,
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
    underlying Pinocchio/hpp-fcl collision query.

    When a collision exists, the Pinocchio collision result vector is already
    populated; wrapping it here only extracts structured diagnostics.

    ``bool(collision_info)`` returns ``in_collision``, so existing code
    that checks ``if check_self_collision(q):`` continues to work.
    """

    in_collision: bool
    collision_pairs: tuple[CollisionPair, ...] = ()
    num_contacts: int = 0
    sample_qpos_rad: tuple[float, ...] | None = None

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
        result: dict[str, Any] = {
            "in_collision": True,
            "num_contacts": self.num_contacts,
            "collision_pairs": [p.to_dict() for p in self.collision_pairs],
        }
        if self.sample_qpos_rad is not None:
            result["sample_qpos_rad"] = list(self.sample_qpos_rad)
        return result

    @property
    def summary(self) -> str:
        """One-line human-readable summary for log messages."""
        if not self.in_collision:
            return "no collision"
        pairs = [f"{p.link_name1}↔{p.link_name2}" for p in self.collision_pairs]
        if not pairs:
            return f"{self.num_contacts} contact(s), pair details unavailable"
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
    # Optional world-frame EEF bounds, shape (3, 2). Paths are validated at
    # every returned waypoint; None keeps generic offline use unrestricted.
    workspace_bounds: np.ndarray | None = None


@dataclass(kw_only=True)
class PlanningProfile:
    """Offline path planning configuration."""

    path_dt: float = 1 / 15
    planning_limits_deg: tuple[tuple[float, float], ...] | None = None
    max_ik_delta_deg: tuple[float, ...] = (120, 135, 120, 120, 180, 150, 180)
    # Validation bound between returned planner waypoints. Firmware, not the
    # application, owns execution-time trajectory generation.
    max_waypoint_delta_deg: float = 15.0
    max_pose_error_pos_m: float = 0.005
    max_pose_error_rot_rad: float = 0.05

    rrt_time_limit: float = 2.0
    rrt_range_options: tuple[float, ...] = (0.05, 0.12)
    num_rrt_attempts: int = 2
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlanningProfile":
        """Reconstruct from a serialized dict."""
        return cls(**from_dict_helper(cls, d))  # type: ignore[arg-type]


@dataclass(kw_only=True)
class TeleopProfile:
    """Online teleoperation IK/servo configuration."""

    max_ik_jump_deg: tuple[float, ...] = (90, 90, 90, 90, 90, 90, 90)
    # max_ik_jump_deg is a discontinuity guard, not a velocity clamp. arm_loop
    # passes speed/acceleration limits to Mode 6; firmware owns smoothing.
    max_pose_error_pos_m: float = 0.008
    max_pose_error_rot_rad: float = 0.08
    check_self_collision: bool = True  # checked in teleop IK hot path; holds on collision

    # Skip extra seeds when the first valid solution is near measured state.
    position_ik_fast_accept_rad: float = np.deg2rad(15.0)

    # Multi-seed scoring balances motion, manipulability, limits, and pose accuracy.
    position_ik_num_random_seeds: int = 3
    position_ik_seed_offset_deg: float = 5.0
    teleop_ik_seed: int | None = 42
    position_ik_manipulability_weight: float = 0.05
    position_ik_limit_penalty_weight: float = 0.01
    position_ik_velocity_weight: float = 0.03
    position_ik_pose_accuracy_weight: float = 0.05
    # Zero disables hard rejection by Yoshikawa manipulability.
    position_ik_min_manipulability: float = 0.0
    # Dampen the heavy base/shoulder while leaving wrist joints responsive.
    velocity_joint_weights: tuple[float, ...] | None = (5.0, 1.5, 0.8, 0.5, 0.2, 0.3, 0.1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TeleopProfile":
        """Reconstruct from a serialized dict."""
        return cls(**from_dict_helper(cls, d))  # type: ignore[arg-type]

    # Range-normalized weights keep the base stable and favor wrist motion.
    joint_weights: tuple[float, ...] = (3.0, 1.2, 1.0, 0.5, 0.5, 0.8, 0.3)

    # Post-IK null-space projection that adjusts the redundant DOF to repel
    # joints from their limits while preserving the EEF pose to first order
    # (J · dq_null = 0). Final nonlinear pose and collision gates run after it.
    enable_nullspace_optimization: bool = True
    nullspace_step_size_deg: float = 1.0  # max null-space joint step per frame [deg]
    nullspace_joint_limit_margin_deg: float = 15.0  # repulsion margin from limits [deg]
