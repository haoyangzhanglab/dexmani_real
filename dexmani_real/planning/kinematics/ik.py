"""Continuation-first Cartesian CLIK with bounded predictor/corrector redundancy search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from .pose import Pose, compute_pose_error

if TYPE_CHECKING:
    from .arm_fk import XArm7Kinematics
    from ..collision import CollisionModel


# Relative rank diagnosis is a search guard, never an execution constraint.
_SVD_RANK_RTOL = 1e-4
_NUMERIC_EPS = 1e-12
_MODEL_LIMIT_TOL_RAD = 1e-5


DEFAULT_JUMP_DEG = (30, 30, 30, 35, 40, 40, 40)


def joint_target_rejection(q, current, previous, limits, jump_limit):
    """Absolute hardware target differences, never periodic shortest angles."""
    q, current, previous = (np.asarray(v, dtype=np.float64) for v in (q, current, previous))
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in (q, current, previous)):
        return "invalid_target"
    lo, hi = np.asarray(limits).T
    if np.any(q < lo) or np.any(q > hi):
        return "operational_limits"
    if np.any(np.abs(q - previous) > jump_limit):
        return "jump"
    if np.max(np.abs(q - current)) > np.deg2rad(150):
        return "hw_dist"
    return None


@dataclass(kw_only=True)
class IKResult:
    success: bool
    qpos: np.ndarray | None
    reason: str = ""
    failure_kind: IKFailureKind | None = None


class IKFailureKind(str, Enum):
    NO_SOLUTION_FOUND = "no_solution_found"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    INVALID_OUTPUT = "invalid_output"


@dataclass(kw_only=True)
class OnlineIKConfig:
    """Bounded online search; all joint radii are maximum component displacements."""

    max_ik_jump_deg: tuple[float, ...] = DEFAULT_JUMP_DEG
    max_pose_error_pos_m: float = 0.008
    max_pose_error_rot_rad: float = 0.08
    fast_accept_max_delta_deg: float = 8.0
    operational_joint_lower_rad: tuple[float, ...] | None = None
    operational_joint_upper_rad: tuple[float, ...] | None = None
    redundancy_seed_step_deg: float = 5.0
    joint_limit_search_margin_deg: float = 15.0
    previous_command_distance_weight: float = 0.25
    joint_limit_penalty_weight: float = 0.01
    pose_accuracy_weight: float = 0.1
    pose_rotation_weight: float = 0.5
    singularity_margin_weight: float = 0.02
    joint_weights: tuple[float, ...] = (4.0, 1.8, 1.2, 0.6, 0.6, 0.9, 0.35)
    previous_command_joint_weights: tuple[float, ...] | None = (10, 4, 2, 1, 0.5, 0.6, 0.1)


def make_online_ik_config(
    runtime,
    *,
    max_pose_error_pos_m: float | None = None,
    max_pose_error_rot_rad: float | None = None,
) -> OnlineIKConfig:
    return OnlineIKConfig(
        max_pose_error_pos_m=(
            runtime.control.ik_max_pose_error_pos_m
            if max_pose_error_pos_m is None
            else max_pose_error_pos_m
        ),
        max_pose_error_rot_rad=(
            runtime.control.ik_max_pose_error_rot_rad
            if max_pose_error_rot_rad is None
            else max_pose_error_rot_rad
        ),
        operational_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
        operational_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
    )


@dataclass
class IKCandidate:
    qpos: np.ndarray
    source: str
    pos_err_m: float
    rot_err_rad: float
    max_physical_delta: float
    base_score: float
    near_limit: bool
    collision_free: bool | None = None
    singularity_margin: float | None = None
    score: float = 0.0


def make_online_ik_solver(runtime, *, kin=None, collision_model=None, **tolerances):
    from .arm_fk import make_arm_kinematics
    from ..collision import CollisionModel

    kin = make_arm_kinematics() if kin is None else kin
    collision_model = (
        CollisionModel(static_boxes=runtime.environment.static_boxes)
        if collision_model is None
        else collision_model
    )
    return OnlineIKSolver(kin, collision_model, make_online_ik_config(runtime, **tolerances))


class OnlineIKSolver:
    """Exact CLIK candidates, measured-state representation and published-target continuity."""

    def __init__(
        self,
        kin: XArm7Kinematics,
        collision_model: CollisionModel,
        online_ik_profile: OnlineIKConfig,
    ) -> None:
        self.kin, self.collision_model, self.profile = kin, collision_model, online_ik_profile
        model = kin.joint_limits
        if kin.dof != 7 or model.shape != (7, 2) or not np.isfinite(model).all():
            raise ValueError("online IK requires finite (7, 2) model limits")
        if np.any(model[:, 0] >= model[:, 1]):
            raise ValueError("online IK model limits must be ordered")
        lower, upper = (
            online_ik_profile.operational_joint_lower_rad,
            online_ik_profile.operational_joint_upper_rad,
        )
        if lower is None and upper is None:
            self.operational_limits = model.copy()
        else:
            lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
            if any(a.shape != (7,) or not np.isfinite(a).all() for a in (lower, upper)):
                raise ValueError("operational lower/upper limits must both be finite (7,) arrays")
            if np.any(lower >= upper):
                raise ValueError("operational lower limits must be less than upper limits")
            if np.any(lower < model[:, 0] - _MODEL_LIMIT_TOL_RAD) or np.any(
                upper > model[:, 1] + _MODEL_LIMIT_TOL_RAD
            ):
                raise ValueError("runtime operational limits exceed the loaded model/URDF limits")
            self.operational_limits = np.column_stack((lower, upper))
        self._jump_limit = np.deg2rad(self.finite_q(online_ik_profile.max_ik_jump_deg))
        self._weights = self.finite_q(online_ik_profile.joint_weights)
        self._previous_weights = self.finite_q(
            (
                online_ik_profile.previous_command_joint_weights
                if online_ik_profile.previous_command_joint_weights is not None
                else online_ik_profile.joint_weights
            ),
        )
        self._joint_ranges = np.maximum(model[:, 1] - model[:, 0], 1e-6)
        self._limit_center = 0.5 * (self.operational_limits[:, 0] + self.operational_limits[:, 1])
        self._limit_radius = np.maximum(
            0.5 * (self.operational_limits[:, 1] - self.operational_limits[:, 0]), 1e-6
        )
        positive = (
            online_ik_profile.max_pose_error_pos_m,
            online_ik_profile.max_pose_error_rot_rad,
            online_ik_profile.redundancy_seed_step_deg,
        )
        nonnegative = (
            online_ik_profile.fast_accept_max_delta_deg,
            online_ik_profile.joint_limit_search_margin_deg,
            online_ik_profile.previous_command_distance_weight,
            online_ik_profile.joint_limit_penalty_weight,
            online_ik_profile.pose_accuracy_weight,
            online_ik_profile.pose_rotation_weight,
            online_ik_profile.singularity_margin_weight,
        )
        if (
            any(not np.isfinite(v) or v <= 0 for v in positive)
            or any(not np.isfinite(v) or v < 0 for v in nonnegative)
            or any(np.any(a < 0) for a in (self._jump_limit, self._weights, self._previous_weights))
        ):
            raise ValueError("online IK tolerances, radii and weights must be finite and valid")

    def solve(
        self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray
    ) -> IKResult:
        try:
            current = self.finite_q(current_qpos)
            previous = self.finite_q(previous_qpos_cmd)
            if (
                not np.isfinite(target_eef_pose_world.p).all()
                or not np.isfinite(target_eef_pose_world.q).all()
            ):
                raise ValueError("non-finite target pose")
            selected, converged, rejection = self.search(target_eef_pose_world, current, previous)
            if selected is not None:
                result = IKResult(success=True, qpos=selected.qpos)
            else:
                kind = (
                    IKFailureKind.NO_VALID_CANDIDATE
                    if converged
                    else IKFailureKind.NO_SOLUTION_FOUND
                )
                reason = (
                    f"no_valid_candidate: {rejection}" if converged else "solver_no_convergence"
                )
                result = IKResult(success=False, qpos=None, reason=reason, failure_kind=kind)
        except (
            ValueError,
            RuntimeError,
            TypeError,
            FloatingPointError,
        ) as exc:
            # Model/numerical contract failures must not masquerade as local solver rejection.
            result = IKResult(
                success=False,
                qpos=None,
                reason=str(exc),
                failure_kind=IKFailureKind.INVALID_OUTPUT,
            )
        return result

    @staticmethod
    def finite_q(value: np.ndarray) -> np.ndarray:
        q = np.asarray(value, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("IK joint vectors must be finite shape (7,)")
        return q

    def search(self, target, current, previous):
        target_base = self.kin.to_mplib_pose(self.kin.world_to_base_pose(target))
        pool: list[IKCandidate] = []
        tried_seeds = []
        converged = False
        rejection = "solver_no_convergence"

        def try_seed(name, seed):
            nonlocal converged, rejection
            # Exact equality only; even a nearby distinct seed can reach another IK branch.
            if any(np.array_equal(seed, tried) for tried in tried_seeds):
                return None
            tried_seeds.append(seed.copy())
            candidate, issue = self.try_clik(
                name, seed, target_base, target, current, previous, pool
            )
            if issue != "solver_no_convergence":
                converged = True
                if issue:
                    rejection = issue
            return candidate

        for name, seed in (("prev_cmd", previous), ("current_qpos", current)):
            candidate = try_seed(name, seed)
            if candidate is None:
                continue
            pool.append(candidate)
            if (
                candidate.source == "prev_cmd"
                and not candidate.near_limit
                and candidate.max_physical_delta
                <= np.deg2rad(self.profile.fast_accept_max_delta_deg)
                and candidate.pos_err_m <= 0.5 * self.profile.max_pose_error_pos_m
                and candidate.rot_err_rad <= 0.5 * self.profile.max_pose_error_rot_rad
                and self.collision_free(candidate)
            ):
                return candidate, converged, ""

        base = self.select_collision_free(sorted(pool, key=lambda c: c.base_score))
        if base is not None and not base.near_limit:
            return base, converged, ""
        anchor = min(pool, key=lambda c: c.base_score) if pool else None
        if anchor is not None:
            direction = self.scaled_jacobian_svd(anchor)
            if direction is not None:
                for name, seed in self.make_null_seeds(anchor.qpos, direction):
                    candidate = try_seed(name, seed)
                    if candidate is not None:
                        pool.append(candidate)
            # Known collisions can anchor search, but cannot participate in final selection.
            viable = [c for c in pool if c.collision_free is not False]
            weight = self.profile.singularity_margin_weight
            for candidate in viable:
                # Dexterity ∈ [0,1], so its weight bounds the base-score gap it can reverse.
                if weight <= 0 or not any(
                    other is not candidate
                    and abs(other.base_score - candidate.base_score) <= weight
                    for other in viable
                ):
                    continue
                if candidate.singularity_margin is None:
                    self.scaled_jacobian_svd(candidate)
                candidate.score = candidate.base_score - weight * candidate.singularity_margin
            selected = self.select_collision_free(sorted(viable, key=lambda c: c.score))
            if selected is not None:
                return selected, converged, ""

        return None, converged, "self_collision" if pool else rejection

    def try_clik(self, name, seed, target_base, target, current, previous, pool):
        model = self.kin.mp_planner
        raw, converged, _ = model.pinocchio_model.compute_IK_CLIK(
            model.link_name_2_idx[model.move_group],
            target_base,
            seed.copy(),
            [],
        )
        if raw is not None:
            raw = self.finite_q(raw)
        if not converged:
            return None, "solver_no_convergence"
        if raw is None:
            raise ValueError("CLIK reported convergence without a joint vector")
        return self.prepare_candidate(raw, name, target, current, previous, pool)

    def dynamic_rejection(self, q, current, previous):
        """Recheck a frozen solution against new feedback without changing its branch."""
        return joint_target_rejection(
            q, current, previous, self.operational_limits, self._jump_limit
        )

    def prepare_candidate(self, raw, source, target, current, previous, pool):
        # Mechanical equivalence survives narrowing the operational interval below 2*pi.
        q = self.kin.canonicalize_qpos(raw, current, self.operational_limits)
        lo, hi = self.operational_limits.T
        if not np.isfinite(q).all():
            raise ValueError("canonical IK output is non-finite")
        if np.any(q < lo) or np.any(q > hi):
            return None, "operational_limits"
        if any(np.max(np.abs(q - c.qpos)) < 1e-4 for c in pool):
            return None, "duplicate"
        rejection = self.dynamic_rejection(q, current, previous)
        if rejection is not None:
            return None, rejection
        delta_prev = self.kin.compute_qpos_delta(q, previous)
        delta = self.kin.compute_qpos_delta(q, current)
        distance = float(np.max(np.abs(delta)))
        pos_err, rot_err = compute_pose_error(target, self.kin.compute_eef_pose_world(q))
        if not np.isfinite((pos_err, rot_err)).all():
            raise ValueError("IK FK/pose error is non-finite")
        p = self.profile
        if pos_err > p.max_pose_error_pos_m or rot_err > p.max_pose_error_rot_rad:
            return None, "pose_error"
        score = (
            self.weighted_joint_distance(delta, self._weights)
            + p.previous_command_distance_weight
            * self.weighted_joint_distance(delta_prev, self._previous_weights)
            + p.joint_limit_penalty_weight * self.joint_limit_penalty(q)
            + p.pose_accuracy_weight
            * (
                pos_err / p.max_pose_error_pos_m
                + p.pose_rotation_weight * rot_err / p.max_pose_error_rot_rad
            )
        )
        if not np.isfinite(score):
            raise ValueError("IK candidate score is non-finite")
        return (
            IKCandidate(
                q,
                source,
                pos_err,
                rot_err,
                distance,
                score,
                bool(
                    np.min(np.minimum(q - lo, hi - q)) < np.deg2rad(p.joint_limit_search_margin_deg)
                ),
                score=score,
            ),
            "",
        )

    def weighted_joint_distance(self, delta, weights):
        """LeFranX ranking: weighted squared joint deltas normalized by model ranges."""
        return float(np.sqrt(np.sum(weights * (delta / self._joint_ranges) ** 2)))

    def joint_limit_penalty(self, qpos):
        return float(np.sum(((qpos - self._limit_center) / self._limit_radius) ** 2))

    def collision_free(self, candidate):
        if candidate.collision_free is None:
            candidate.collision_free = not self.collision_model.check_self_collision(candidate.qpos)
        return candidate.collision_free

    def select_collision_free(self, candidates):
        for candidate in candidates:
            if self.collision_free(candidate):
                return candidate
        return None

    def scaled_jacobian_svd(self, candidate):
        jacobian = self.kin.compute_eef_jacobian_world(candidate.qpos)
        if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
            raise ValueError("IK Jacobian must be finite shape (6, 7)")
        norms = np.array([np.linalg.norm(jacobian[:3]), np.linalg.norm(jacobian[3:])])
        if not np.isfinite(norms).all():
            raise ValueError("IK Jacobian block norms are non-finite")
        if np.any(norms <= _NUMERIC_EPS):
            candidate.singularity_margin = 0.0
            return None
        conditioned = jacobian / np.repeat(norms, 3)[:, None]
        _, sigma, vh = np.linalg.svd(conditioned, full_matrices=True)
        if not np.isfinite(sigma).all() or not np.isfinite(vh).all():
            raise ValueError("conditioned Jacobian SVD is non-finite")
        rank = int(np.count_nonzero(sigma > _SVD_RANK_RTOL * sigma[0]))
        candidate.singularity_margin = float(sigma[-1] / max(sigma[0], _NUMERIC_EPS))
        return vh[-1] if rank == 6 else None

    def make_null_seeds(self, anchor, direction):
        desired = (
            np.deg2rad(self.profile.redundancy_seed_step_deg)
            * direction
            / np.max(np.abs(direction))
        )
        seeds = []
        for sign in (1, -1):
            delta = sign * desired
            moving = np.abs(delta) > _NUMERIC_EPS
            room = np.where(
                delta > 0,
                self.operational_limits[:, 1] - anchor,
                anchor - self.operational_limits[:, 0],
            )
            fraction = min(1.0, float(np.min(room[moving] / np.abs(delta[moving]))))
            if fraction < 1.0:
                fraction = np.nextafter(fraction, 0.0)
            seed = anchor + fraction * delta
            if np.max(np.abs(seed - anchor)) <= 1e-8:
                continue
            if np.any(seed < self.operational_limits[:, 0]) or np.any(
                seed > self.operational_limits[:, 1]
            ):
                continue
            if not any(np.array_equal(seed, old) for old in seeds):
                seeds.append(seed)
        seeds.sort(key=lambda q: self.joint_limit_penalty(q))
        return list(zip(("null_preferred", "null_opposite"), seeds))
