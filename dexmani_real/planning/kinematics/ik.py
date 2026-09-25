"""Continuation-first Cartesian CLIK with bounded predictor/corrector redundancy search."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.utils.log import get_logger

from .pose import Pose, compute_pose_error

if TYPE_CHECKING:
    from .arm_fk import XArm7Kinematics
    from .ik_geometry import IKGeometry

logger = get_logger(__name__)

# Relative rank diagnosis is a search guard, never an execution constraint.
_SVD_RANK_RTOL = 1e-4
_NUMERIC_EPS = 1e-12
_MODEL_LIMIT_TOL_RAD = 1e-5


@dataclass(kw_only=True)
class IKResult:
    success: bool
    qpos: np.ndarray | None
    reason: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    failure_kind: IKFailureKind | None = None


class IKFailureKind(str, Enum):
    NO_SOLUTION_FOUND = "no_solution_found"
    NO_VALID_CANDIDATE = "no_valid_candidate"
    INVALID_OUTPUT = "invalid_output"


@dataclass(kw_only=True)
class OnlineIKConfig:
    """Bounded online search; all joint radii are maximum component displacements."""

    max_ik_jump_deg: tuple[float, ...] = (30, 30, 30, 35, 40, 40, 40)
    max_pose_error_pos_m: float = 0.008
    max_pose_error_rot_rad: float = 0.08
    fast_accept_max_delta_deg: float = 8.0
    operational_joint_lower_rad: tuple[float, ...] | None = None
    operational_joint_upper_rad: tuple[float, ...] | None = None
    redundancy_seed_step_deg: float = 5.0
    joint_limit_search_margin_deg: float = 15.0
    enable_random_fallback: bool = False
    random_fallback_step_deg: float = 5.0
    random_seed: int | None = 42
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
    enable_random_fallback: bool = False,
) -> OnlineIKConfig:
    return OnlineIKConfig(
        max_pose_error_pos_m=(
            runtime.policy.ik_max_pose_error_pos_m
            if max_pose_error_pos_m is None
            else max_pose_error_pos_m
        ),
        max_pose_error_rot_rad=(
            runtime.policy.ik_max_pose_error_rot_rad
            if max_pose_error_rot_rad is None
            else max_pose_error_rot_rad
        ),
        operational_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
        operational_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
        enable_random_fallback=enable_random_fallback,
    )


@dataclass
class _Candidate:
    qpos: np.ndarray
    attempt: dict[str, Any]
    pos_err_m: float
    rot_err_rad: float
    max_physical_delta: float
    distance_current: float
    base_score: float
    near_limit: bool
    collision_free: bool | None = None
    singularity_margin: float | None = None
    score: float = 0.0


class OnlineIKSolver:
    """Exact CLIK candidates, measured-state representation and published-target continuity."""

    _ELBOW_FLIP_NEG_THRESH_RAD = np.deg2rad(-5.0)
    _ELBOW_FLIP_POS_THRESH_RAD = np.deg2rad(15.0)
    _ELBOW_FLIP_MIN_DELTA_RAD = np.deg2rad(40.0)

    def __init__(
        self,
        kin: XArm7Kinematics,
        ik_geometry: IKGeometry,
        online_ik_profile: OnlineIKConfig,
        elbow_joint_index: int = 3,
    ) -> None:
        self.kin, self.ik_geometry, self.profile = kin, ik_geometry, online_ik_profile
        self._elbow_joint_index = elbow_joint_index
        self._rng = np.random.default_rng(online_ik_profile.random_seed)
        self._failure_start: float | None = None
        self._failure_warned = False
        model = np.asarray(ik_geometry.joint_limits, dtype=np.float64)
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
        self._jump_limit = np.deg2rad(
            ik_geometry.profile_array(online_ik_profile.max_ik_jump_deg, "max_ik_jump_deg")
        )
        self._weights = ik_geometry.profile_array(online_ik_profile.joint_weights, "joint_weights")
        self._previous_weights = ik_geometry.profile_array(
            online_ik_profile.previous_command_joint_weights
            if online_ik_profile.previous_command_joint_weights is not None
            else online_ik_profile.joint_weights,
            "previous_command_joint_weights",
        )
        positive = (
            online_ik_profile.max_pose_error_pos_m,
            online_ik_profile.max_pose_error_rot_rad,
            online_ik_profile.redundancy_seed_step_deg,
            online_ik_profile.random_fallback_step_deg,
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
            or any(
                not np.isfinite(a).all() or np.any(a < 0)
                for a in (self._jump_limit, self._weights, self._previous_weights)
            )
        ):
            raise ValueError("online IK tolerances, radii and weights must be finite and valid")

    def solve(
        self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray
    ) -> IKResult:
        start = time.perf_counter()
        report: dict[str, Any] = {
            "attempts": [],
            "clik_calls": 0,
            "collision_checks": 0,
            "jacobian_calls": 0,
            "funnel": dict.fromkeys(
                (
                    "attempted",
                    "clik_converged",
                    "operational_limits_valid",
                    "continuity_valid",
                    "hardware_valid",
                    "pose_valid",
                    "collision_free",
                ),
                0,
            ),
        }
        try:
            current = self._finite_q(current_qpos)
            previous = self._finite_q(previous_qpos_cmd)
            if (
                not np.isfinite(target_eef_pose_world.p).all()
                or not np.isfinite(target_eef_pose_world.q).all()
            ):
                raise ValueError("non-finite target pose")
            selected, mode = self._search(target_eef_pose_world, current, previous, report)
            if selected is not None:
                report.update(
                    seed=selected.attempt["seed"],
                    mode=mode,
                    best_score=selected.score,
                    cmd_tracking_error_pos_m=selected.pos_err_m,
                    cmd_tracking_error_rot_rad=selected.rot_err_rad,
                    qpos_distance_to_current=selected.distance_current,
                    max_qpos_cmd_delta_deg=float(np.rad2deg(selected.max_physical_delta)),
                )
                if selected.singularity_margin is not None:
                    report["singularity_margin"] = selected.singularity_margin
                result = IKResult(success=True, qpos=selected.qpos, report=report)
            else:
                kind = (
                    IKFailureKind.NO_VALID_CANDIDATE
                    if report["funnel"]["clik_converged"]
                    else IKFailureKind.NO_SOLUTION_FOUND
                )
                reason = (
                    "no_valid_candidate: "
                    + ", ".join(dict.fromkeys(a["result"] for a in report["attempts"]))
                    if kind == IKFailureKind.NO_VALID_CANDIDATE
                    else "solver_no_convergence"
                )
                result = IKResult(
                    success=False, qpos=None, reason=reason, failure_kind=kind, report=report
                )
        except (
            ValueError,
            RuntimeError,
            TypeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            # Model/numerical contract failures must not masquerade as local solver rejection.
            report["invalid_output"] = str(exc)
            result = IKResult(
                success=False,
                qpos=None,
                reason=str(exc),
                failure_kind=IKFailureKind.INVALID_OUTPUT,
                report=report,
            )
        report["ik_timing_ms"] = (time.perf_counter() - start) * 1000.0
        if result.success:
            self._failure_start, self._failure_warned = None, False
        elif self._failure_start is None:
            self._failure_start = time.monotonic()
        elif time.monotonic() - self._failure_start > 2.0 and not self._failure_warned:
            logger.warning(
                "IK failed for %.1fs — no new target (reason: %s)",
                time.monotonic() - self._failure_start,
                result.reason,
            )
            self._failure_warned = True
        return result

    @staticmethod
    def _finite_q(value: np.ndarray) -> np.ndarray:
        q = np.asarray(value, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("IK joint vectors must be finite shape (7,)")
        return q

    def _search(self, target, current, previous, report):
        target_base = self.kin.to_mplib_pose(self.kin.world_to_base_pose(target))
        pool: list[_Candidate] = []
        for name, seed in (("prev_cmd", previous), ("current_qpos", current)):
            candidate = self._try_clik(
                name, seed, target_base, target, current, previous, pool, report
            )
            if candidate is None:
                continue
            pool.append(candidate)
            if (
                name == "prev_cmd"
                and not candidate.near_limit
                and candidate.max_physical_delta
                <= np.deg2rad(self.profile.fast_accept_max_delta_deg)
                and candidate.pos_err_m <= 0.5 * self.profile.max_pose_error_pos_m
                and candidate.rot_err_rad <= 0.5 * self.profile.max_pose_error_rot_rad
                and self._collision_free(candidate, report)
            ):
                return candidate, "fast"

        base = self._select_collision_free(sorted(pool, key=lambda c: c.base_score), report)
        if base is not None and not base.near_limit:
            return base, "base"
        anchor = min(pool, key=lambda c: c.base_score) if pool else None
        if anchor is not None:
            direction = self._scaled_jacobian_svd(anchor, report)
            if direction is not None:
                for name, seed in self._make_null_seeds(anchor.qpos, direction):
                    candidate = self._try_clik(
                        name, seed, target_base, target, current, previous, pool, report
                    )
                    if candidate is not None:
                        pool.append(candidate)
            # Known collisions can anchor search, but cannot participate in final selection.
            viable = [c for c in pool if c.collision_free is not False]
            weight = self.profile.singularity_margin_weight
            for candidate in viable:
                # Dexterity lies in [0, 1], so it cannot reverse a base-score gap
                # larger than its weight, even if cheaper candidates later collide.
                if weight <= 0 or not any(
                    other is not candidate
                    and abs(other.base_score - candidate.base_score) <= weight
                    for other in viable
                ):
                    continue
                if candidate.singularity_margin is None:
                    self._scaled_jacobian_svd(candidate, report)
                candidate.score = candidate.base_score - weight * candidate.singularity_margin
                candidate.attempt["score"] = candidate.score
            selected = self._select_collision_free(sorted(viable, key=lambda c: c.score), report)
            if selected is not None:
                return selected, "null" if selected.attempt["seed"].startswith("null_") else "base"

        if self.profile.enable_random_fallback:
            reference = anchor.qpos if anchor is not None else previous
            radius = np.deg2rad(self.profile.random_fallback_step_deg)
            lower = np.maximum(self.operational_limits[:, 0], reference - radius)
            upper = np.minimum(self.operational_limits[:, 1], reference + radius)
            if np.all(lower <= upper):
                seed = self._rng.uniform(lower, upper)
                candidate = self._try_clik(
                    "random", seed, target_base, target, current, previous, pool, report
                )
                if candidate is not None and self._collision_free(candidate, report):
                    return candidate, "random"
        return None, "failed"

    def _try_clik(self, name, seed, target_base, target, current, previous, pool, report):
        attempt = {"seed": name, "result": "invalid_output", "solve_ms": 0.0}
        report["attempts"].append(attempt)
        report["clik_calls"] += 1
        report["funnel"]["attempted"] += 1
        model = self.ik_geometry.mp_planner
        start = time.perf_counter()
        try:
            raw, converged, _ = model.pinocchio_model.compute_IK_CLIK(
                model.link_name_2_idx[model.move_group],
                target_base,
                seed.copy(),
                [],
            )
        finally:
            attempt["solve_ms"] = (time.perf_counter() - start) * 1000.0
        if raw is not None:
            raw = self._finite_q(raw)
        if not converged:
            attempt["result"] = "solver_no_convergence"
            return None
        report["funnel"]["clik_converged"] += 1
        if raw is None:
            raise ValueError("CLIK reported convergence without a joint vector")
        return self._prepare_candidate(raw, attempt, target, current, previous, pool, report)

    def _prepare_candidate(self, raw, attempt, target, current, previous, pool, report):
        # Mechanical equivalence survives narrowing the operational interval below 2*pi.
        q = raw.copy()
        mask = self.ik_geometry.equivalent_joint_mask
        lo, hi = self.operational_limits.T
        period = 2 * np.pi
        k_min, k_max = (
            np.ceil((lo[mask] - q[mask]) / period),
            np.floor((hi[mask] - q[mask]) / period),
        )
        k = np.minimum(np.maximum(np.rint((current[mask] - q[mask]) / period), k_min), k_max)
        q[mask] = np.where(k_min <= k_max, q[mask] + period * k, q[mask])
        if not np.isfinite(q).all():
            raise ValueError("canonical IK output is non-finite")
        if np.any(q < lo) or np.any(q > hi):
            attempt["result"] = "operational_limits"
            return None
        report["funnel"]["operational_limits_valid"] += 1
        if any(np.max(np.abs(q - c.qpos)) < 1e-4 for c in pool):
            attempt["result"] = "duplicate"
            return None
        delta_prev = self.ik_geometry.compute_qpos_delta(q, previous)
        if np.any(np.abs(delta_prev) > self._jump_limit):
            attempt["result"] = "jump"
            return None
        if self._has_elbow_flip(q, previous):
            attempt["result"] = "elbow_flip"
            return None
        if np.linalg.norm(delta_prev) > np.deg2rad(120):
            attempt["result"] = "branch_jump_l2"
            return None
        report["funnel"]["continuity_valid"] += 1
        delta = self.ik_geometry.compute_qpos_delta(q, current)
        distance = float(np.max(np.abs(delta)))
        if np.max(np.abs(q - current)) - distance > np.deg2rad(90):
            attempt["result"] = "band_switch"
            return None
        if distance > np.deg2rad(150):
            attempt["result"] = "hw_dist"
            return None
        report["funnel"]["hardware_valid"] += 1
        pos_err, rot_err = compute_pose_error(target, self.kin.compute_eef_pose_world(q))
        if not np.isfinite((pos_err, rot_err)).all():
            raise ValueError("IK FK/pose error is non-finite")
        attempt.update(pos_err_m=pos_err, rot_err_rad=rot_err)
        p = self.profile
        if pos_err > p.max_pose_error_pos_m or rot_err > p.max_pose_error_rot_rad:
            attempt["result"] = "pose_error"
            return None
        report["funnel"]["pose_valid"] += 1
        score = (
            self.ik_geometry.weighted_joint_distance(q, current, self._weights, delta=delta)
            + p.previous_command_distance_weight
            * self.ik_geometry.weighted_joint_distance(
                q, previous, self._previous_weights, delta=delta_prev
            )
            + p.joint_limit_penalty_weight
            * self.ik_geometry.joint_limit_penalty(q, self.operational_limits)
            + p.pose_accuracy_weight
            * (
                pos_err / p.max_pose_error_pos_m
                + p.pose_rotation_weight * rot_err / p.max_pose_error_rot_rad
            )
        )
        if not np.isfinite(score):
            raise ValueError("IK candidate score is non-finite")
        attempt.update(result="pose_valid", score=score)
        return _Candidate(
            q,
            attempt,
            pos_err,
            rot_err,
            distance,
            float(np.linalg.norm(delta)),
            score,
            bool(np.min(np.minimum(q - lo, hi - q)) < np.deg2rad(p.joint_limit_search_margin_deg)),
            score=score,
        )

    def _collision_free(self, candidate, report):
        if candidate.collision_free is None:
            report["collision_checks"] += 1
            candidate.collision_free = not self.ik_geometry.has_self_collision(candidate.qpos)
            candidate.attempt["result"] = "ok" if candidate.collision_free else "self_collision"
            report["funnel"]["collision_free"] += int(candidate.collision_free)
        return candidate.collision_free

    def _select_collision_free(self, candidates, report):
        for candidate in candidates:
            if self._collision_free(candidate, report):
                return candidate
        return None

    def _scaled_jacobian_svd(self, candidate, report):
        report["jacobian_calls"] += 1
        jacobian = self.kin.compute_eef_jacobian_world(candidate.qpos)
        if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
            raise ValueError("IK Jacobian must be finite shape (6, 7)")
        norms = np.array([np.linalg.norm(jacobian[:3]), np.linalg.norm(jacobian[3:])])
        if not np.isfinite(norms).all():
            raise ValueError("IK Jacobian block norms are non-finite")
        if np.any(norms <= _NUMERIC_EPS):
            candidate.singularity_margin = 0.0
            candidate.attempt.update(
                effective_rank=0, singularity_margin=0.0, jacobian_diagnostic="degenerate_block"
            )
            return None
        conditioned = jacobian / np.repeat(norms, 3)[:, None]
        _, sigma, vh = np.linalg.svd(conditioned, full_matrices=True)
        if not np.isfinite(sigma).all() or not np.isfinite(vh).all():
            raise ValueError("conditioned Jacobian SVD is non-finite")
        rank = int(np.count_nonzero(sigma > _SVD_RANK_RTOL * sigma[0]))
        candidate.singularity_margin = float(sigma[-1] / max(sigma[0], _NUMERIC_EPS))
        candidate.attempt.update(
            effective_rank=rank,
            singular_values=sigma.tolist(),
            singularity_margin=candidate.singularity_margin,
        )
        return vh[-1] if rank == 6 else None

    def _make_null_seeds(self, anchor, direction):
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
            if not any(np.allclose(seed, old, atol=1e-8, rtol=0) for old in seeds):
                seeds.append(seed)
        seeds.sort(key=lambda q: self.ik_geometry.joint_limit_penalty(q, self.operational_limits))
        return list(zip(("null_preferred", "null_opposite"), seeds))

    def _has_elbow_flip(self, candidate_qpos, previous_qpos_cmd):
        prev = float(previous_qpos_cmd[self._elbow_joint_index])
        cand = float(candidate_qpos[self._elbow_joint_index])
        crosses = (
            prev < self._ELBOW_FLIP_NEG_THRESH_RAD and cand > self._ELBOW_FLIP_POS_THRESH_RAD
        ) or (cand < self._ELBOW_FLIP_NEG_THRESH_RAD and prev > self._ELBOW_FLIP_POS_THRESH_RAD)
        return crosses and abs(cand - prev) > self._ELBOW_FLIP_MIN_DELTA_RAD
