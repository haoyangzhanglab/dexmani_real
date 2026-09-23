"""Online IK solver — position IK with deterministic MPlib seeding."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from .arm_fk import XArm7Kinematics
    from .ik_geometry import IKGeometry

from .pose import Pose, compute_pose_error, ensure_qpos

logger = get_logger(__name__)


@dataclass(kw_only=True)
class IKResult:
    success: bool
    qpos: np.ndarray | None
    reason: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    failure_kind: "IKFailureKind | None" = None


class IKFailureKind(str, Enum):
    """Machine-readable reason for a failed IK solve."""

    NO_SOLUTION = "no_solution"
    GEOMETRY_REJECTED = "geometry_rejected"
    INVALID_OUTPUT = "invalid_output"


@dataclass(kw_only=True)
class OnlineIKConfig:
    """Online IK/servo configuration."""

    max_ik_jump_deg: tuple[float, ...] = (30, 30, 30, 35, 40, 40, 40)
    max_pose_error_pos_m: float = 0.008
    max_pose_error_rot_rad: float = 0.08
    position_ik_fast_accept_rad: float = np.deg2rad(8.0)
    position_ik_num_random_seeds: int = 3
    position_ik_seed_offset_deg: float = 5.0
    teleop_ik_seed: int | None = 42
    position_ik_manipulability_weight: float = 0.02
    position_ik_limit_penalty_weight: float = 0.01
    position_ik_velocity_weight: float = 0.25
    position_ik_pose_accuracy_weight: float = 0.1
    position_ik_pose_rot_weight: float = 0.5
    position_ik_min_manipulability: float = 0.0
    velocity_joint_weights: tuple[float, ...] | None = (
        10.0,
        4.0,
        2.0,
        1.0,
        0.5,
        0.6,
        0.1,
    )
    joint_weights: tuple[float, ...] = (4.0, 1.8, 1.2, 0.6, 0.6, 0.9, 0.35)
    enable_nullspace_optimization: bool = True
    nullspace_step_size_deg: float = 1.0
    nullspace_joint_limit_margin_deg: float = 15.0


def make_online_ik_config(runtime, *, control_dt_s: float) -> OnlineIKConfig:
    if not np.isfinite(control_dt_s) or control_dt_s <= 0:
        raise ValueError("online IK control_dt_s must be finite and positive")
    return OnlineIKConfig(
        max_pose_error_pos_m=runtime.policy.ik_max_pose_error_pos_m,
        max_pose_error_rot_rad=runtime.policy.ik_max_pose_error_rot_rad,
        nullspace_step_size_deg=(runtime.policy.ik_nullspace_step_rate_deg_s * control_dt_s),
    )


class OnlineIKSolver:
    """MPlib position IK with prev_cmd seeding and fast-accept.

    Priority: prev_cmd seed (position IK) → multi-seed fallback → failure.
    """

    # Elbow-branch selection thresholds.
    _ELBOW_FLIP_NEG_THRESH_RAD: float = np.deg2rad(-5.0)
    _ELBOW_FLIP_POS_THRESH_RAD: float = np.deg2rad(15.0)
    _ELBOW_FLIP_MIN_DELTA_RAD: float = np.deg2rad(40.0)

    def __init__(
        self,
        kin: XArm7Kinematics,
        ik_mgr: IKGeometry,
        teleop_profile: OnlineIKConfig,
        elbow_joint_index: int = 3,
    ) -> None:
        self.kin = kin
        self.ik_mgr = ik_mgr
        self.profile = teleop_profile
        self._elbow_joint_index = elbow_joint_index
        self._nullspace_warn_last_s: float = 0.0
        self._failure_start: float | None = None
        self._failure_warned: bool = False
        self._rng = np.random.default_rng(teleop_profile.teleop_ik_seed)

    def solve(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
    ) -> IKResult:
        """Run teleop IK — prev_cmd seed, multi-seed fallback, explicit failure."""
        t_start = time.perf_counter()

        profile = self.profile
        current_qpos = ensure_qpos(current_qpos, self.kin.dof, "current_qpos")
        previous_qpos_cmd = ensure_qpos(previous_qpos_cmd, self.kin.dof, "previous_qpos_cmd")

        qpos, report = self._solve_position_ik(
            target_eef_pose_world, current_qpos, previous_qpos_cmd, profile
        )

        if qpos is not None:
            result = self._command_from_target_qpos(
                target_eef_pose_world=target_eef_pose_world,
                current_qpos=current_qpos,
                target_qpos=qpos,
                profile=profile,
                report=report,
            )
            result.report["ik_timing_ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)
        else:
            dt_total_ms = (time.perf_counter() - t_start) * 1000
            diagnostic = self._build_diagnostic(report)
            result = IKResult(
                success=False,
                qpos=None,
                reason=diagnostic["summary"],
                failure_kind=report.get("failure_kind", IKFailureKind.NO_SOLUTION),
                report={
                    **report,
                    "diagnostic": diagnostic,
                    "ik_timing_ms": round(dt_total_ms, 1),
                },
            )

        if not result.success:
            if self._failure_start is None:
                self._failure_start = time.monotonic()
            elif time.monotonic() - self._failure_start > 2.0 and not self._failure_warned:
                logger.warning(
                    "IK failed for %.1fs — no new target (reason: %s)",
                    time.monotonic() - self._failure_start,
                    result.reason,
                )
                self._failure_warned = True
        else:
            self._failure_start = None
            self._failure_warned = False

        return result

    def _solve_position_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: OnlineIKConfig,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Position IK: prev_cmd seed with fast-accept, multi-seed fallback, scoring."""
        target_pose_base = self.kin.world_to_base_pose(target_eef_pose_world)
        jump_limit = np.deg2rad(
            self.ik_mgr.profile_array(profile.max_ik_jump_deg, "max_ik_jump_deg")
        )
        fast_accept_rad = profile.position_ik_fast_accept_rad
        weights = self.ik_mgr.profile_array(profile.joint_weights, "joint_weights")

        attempts: list[str] = []

        # Normalize manipulability against the measured posture for scoring.
        try:
            mu_current = self.kin.compute_manipulability(current_qpos)
        except (ValueError, RuntimeError):
            return None, {
                "method": "position_ik",
                "failure_reason": "measured manipulability evaluation failed",
                "failure_kind": IKFailureKind.INVALID_OUTPUT,
                "attempts": attempts,
            }
        if not np.isfinite(mu_current):
            return None, {
                "method": "position_ik",
                "failure_reason": "measured manipulability is non-finite",
                "failure_kind": IKFailureKind.INVALID_OUTPUT,
                "attempts": attempts,
            }

        seeds = self._make_teleop_seeds(previous_qpos_cmd, current_qpos, profile)

        candidates: list[
            tuple[np.ndarray, str, float, float]
        ] = []  # (qpos, seed_name, score, manipulability)
        seen_qpos: list[np.ndarray] = []

        for seed_name, seed in seeds:
            _tik0 = time.perf_counter()
            model = self.ik_mgr.mp_planner
            raw_qpos, success, _ = model.pinocchio_model.compute_IK_CLIK(
                model.link_name_2_idx[model.move_group],
                self.kin.to_mplib_pose(target_pose_base),
                seed,
                [],
            )
            _solve_ms = (time.perf_counter() - _tik0) * 1000.0
            if not success or raw_qpos is None:
                attempts.append(f"{seed_name}:mplib_failed({_solve_ms:.1f}ms)")
                continue

            raw_qpos = np.asarray(raw_qpos, dtype=np.float64)
            # Numerical corruption is not an ordinary no-solution outcome.
            if not np.all(np.isfinite(raw_qpos)):
                attempts.append(f"{seed_name}:nan_qpos({_solve_ms:.1f}ms)")
                return None, {
                    "method": "position_ik",
                    "failure_reason": "solver returned non-finite qpos",
                    "failure_kind": IKFailureKind.INVALID_OUTPUT,
                    "attempts": attempts,
                }
            # Canonicalize against physical encoder position to avoid long rotations.
            qpos = np.asarray(
                self.ik_mgr.canonicalize_qpos(raw_qpos, current_qpos),
                dtype=np.float64,
            )
            if not np.all(np.isfinite(qpos)):
                attempts.append(f"{seed_name}:canonical_nan({_solve_ms:.1f}ms)")
                return None, {
                    "method": "position_ik",
                    "failure_reason": "canonical IK qpos is non-finite",
                    "failure_kind": IKFailureKind.INVALID_OUTPUT,
                    "attempts": attempts,
                }
            outside, _ = self.ik_mgr.limit_violation(qpos, self.ik_mgr.joint_limits)
            if np.any(outside):
                attempts.append(f"{seed_name}:limits({_solve_ms:.1f}ms)")
                continue
            duplicate = False
            for seen in seen_qpos:
                seen_delta = self.ik_mgr.compute_qpos_delta(qpos, seen)
                if not np.all(np.isfinite(seen_delta)):
                    return None, {
                        "method": "position_ik",
                        "failure_reason": "IK qpos delta is non-finite",
                        "failure_kind": IKFailureKind.INVALID_OUTPUT,
                        "attempts": attempts,
                    }
                duplicate = duplicate or np.max(np.abs(seen_delta)) < 1e-4
            if duplicate:
                attempts.append(f"{seed_name}:duplicate({_solve_ms:.1f}ms)")
                continue
            seen_qpos.append(qpos.copy())

            jacobian, eef_pose_world = self.kin.compute_eef_jacobian_and_pose_world(qpos)
            pos_err, rot_err = compute_pose_error(target_eef_pose_world, eef_pose_world)
            mu = self.kin.manipulability_from_jacobian(jacobian)
            if not (
                np.all(np.isfinite(jacobian))
                and np.isfinite(pos_err)
                and np.isfinite(rot_err)
                and np.isfinite(mu)
            ):
                attempts.append(f"{seed_name}:invalid_kinematics({_solve_ms:.1f}ms)")
                return None, {
                    "method": "position_ik",
                    "failure_reason": "IK kinematic acceptance output is non-finite",
                    "failure_kind": IKFailureKind.INVALID_OUTPUT,
                    "attempts": attempts,
                }

            passed, tag = self._validate_ik_candidate(
                qpos,
                pos_err,
                rot_err,
                mu,
                previous_qpos_cmd,
                jump_limit,
                profile,
            )
            if not passed:
                if tag == "invalid_output":
                    return None, {
                        "method": "position_ik",
                        "failure_reason": "IK validation output is non-finite",
                        "failure_kind": IKFailureKind.INVALID_OUTPUT,
                        "attempts": attempts,
                    }
                attempts.append(f"{seed_name}:{tag}({_solve_ms:.1f}ms)")
                continue

            delta_current = self.ik_mgr.compute_qpos_delta(qpos, current_qpos)
            hw_dist_raw = float(np.max(np.abs(qpos - current_qpos)))
            hw_dist = float(np.max(np.abs(delta_current)))
            weighted_dist = self.ik_mgr.weighted_joint_distance(
                qpos, current_qpos, weights, delta=delta_current
            )
            if not (
                np.all(np.isfinite(delta_current))
                and np.isfinite(hw_dist_raw)
                and np.isfinite(hw_dist)
                and np.isfinite(weighted_dist)
            ):
                return None, {
                    "method": "position_ik",
                    "failure_reason": "IK candidate distance is non-finite",
                    "failure_kind": IKFailureKind.INVALID_OUTPUT,
                    "attempts": attempts,
                }

            _hw_band_mismatch = hw_dist_raw - hw_dist
            _hw_band_limit_rad = np.deg2rad(90.0)
            if _hw_band_mismatch > _hw_band_limit_rad:
                attempts.append(
                    f"{seed_name}:band_switch(raw={np.rad2deg(hw_dist_raw):.0f}deg, wrapped={np.rad2deg(hw_dist):.0f}deg)"
                )
                continue

            _hw_limit_rad = np.deg2rad(150.0)
            if hw_dist > _hw_limit_rad:
                attempts.append(
                    f"{seed_name}:hw_dist({_solve_ms:.1f}ms, {np.rad2deg(hw_dist):.0f}deg)"
                )
                continue

            attempts.append(f"{seed_name}:ok({_solve_ms:.1f}ms)")

            if (
                seed_name == "prev_cmd"
                and hw_dist <= fast_accept_rad
                # Fast-accept only when the previous-command solution tracks target.
                and pos_err <= 0.5 * profile.max_pose_error_pos_m
                and rot_err <= 0.5 * profile.max_pose_error_rot_rad
            ):
                return qpos, {
                    "method": "position_ik",
                    "seed": seed_name,
                    "attempts": attempts,
                }

            score = self._score_candidate(
                weighted_dist=weighted_dist,
                manipulability=mu,
                pos_err=pos_err,
                rot_err=rot_err,
                qpos=qpos,
                previous_qpos_cmd=previous_qpos_cmd,
                profile=profile,
                mu_current=mu_current,
            )
            if not np.isfinite(score):
                return None, {
                    "method": "position_ik",
                    "failure_reason": "IK candidate score is non-finite",
                    "failure_kind": IKFailureKind.INVALID_OUTPUT,
                    "attempts": attempts,
                }
            candidates.append((qpos.copy(), seed_name, score, mu))

        if candidates:
            candidates.sort(key=lambda c: c[2])  # lower score = better
            best_qpos, best_name, best_score, best_mu = candidates[0]
            return best_qpos, {
                "method": "position_ik",
                "seed": best_name,
                "num_candidates": len(candidates),
                "best_score": round(best_score, 4),
                "best_manipulability": round(best_mu, 4),
                "attempts": attempts,
            }

        return None, {
            "method": "position_ik",
            "failure_reason": f"all failed: {attempts}",
            "attempts": attempts,
        }

    def _validate_ik_candidate(
        self,
        qpos: np.ndarray,
        pos_err: float,
        rot_err: float,
        mu: float,
        previous_qpos_cmd: np.ndarray,
        jump_limit: np.ndarray,
        profile: OnlineIKConfig,
    ) -> tuple[bool, str]:
        """Validate IK pose, manipulability, branch continuity and endpoint self-collision.

        Returns (passed, tag) — tag is the first failing check name or "ok".
        """
        if pos_err > profile.max_pose_error_pos_m or rot_err > profile.max_pose_error_rot_rad:
            return False, "pose_err"

        if (
            profile.position_ik_min_manipulability > 0
            and mu < profile.position_ik_min_manipulability
        ):
            return False, "manipulability"

        delta_prev = self.ik_mgr.compute_qpos_delta(qpos, previous_qpos_cmd)
        if not np.all(np.isfinite(delta_prev)):
            return False, "invalid_output"
        if np.any(np.abs(delta_prev) > jump_limit):
            return False, "jump"

        if self._has_elbow_flip(qpos, previous_qpos_cmd):
            return False, "elbow_flip"

        # Catch multi-joint branch jumps that the J4-only elbow check misses.
        if float(np.linalg.norm(delta_prev)) > np.deg2rad(120):
            return False, "branch_jump_l2"

        if self.ik_mgr.has_self_collision(qpos):
            return False, "self_collision"

        return True, "ok"

    def _make_teleop_seeds(
        self,
        prev_cmd: np.ndarray,
        current_qpos: np.ndarray,
        profile: OnlineIKConfig,
    ) -> list[tuple[str, np.ndarray]]:
        """Try the previous command, measured pose, then perturbed seeds."""
        seeds: list[tuple[str, np.ndarray]] = [
            ("prev_cmd", prev_cmd.copy()),
            ("current_qpos", current_qpos.copy()),
        ]
        offsets_rad = np.deg2rad(profile.position_ik_seed_offset_deg)
        for i in range(profile.position_ik_num_random_seeds):
            seed = prev_cmd + self._rng.uniform(-offsets_rad, offsets_rad, self.kin.dof)
            seeds.append((f"random_{i}", seed))
        unique: list[tuple[str, np.ndarray]] = []
        for item in seeds:
            if not any(
                np.allclose(item[1], previous[1], atol=1e-8, rtol=0.0) for previous in unique
            ):
                unique.append(item)
        return unique

    def _score_candidate(
        self,
        weighted_dist: float,
        manipulability: float,
        pos_err: float,
        rot_err: float,
        qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: OnlineIKConfig,
        mu_current: float,
    ) -> float:
        """Score an IK candidate (lower is better).

        = weighted_joint_distance + velocity_weight*velocity_dist
          - manipulability_weight*normalized_mu + limit_penalty_weight*penalty
          + pose_accuracy_weight*pose_cost.
        """
        limit_penalty = self.ik_mgr.joint_limit_penalty(qpos, self.ik_mgr.joint_limits)

        vel_weights = (
            profile.velocity_joint_weights
            if profile.velocity_joint_weights is not None
            else profile.joint_weights
        )
        velocity_dist = self.ik_mgr.weighted_joint_distance(qpos, previous_qpos_cmd, vel_weights)

        # Normalize Yoshikawa manipulability to a unitless [0, 1] score.
        normalized_mu = min(manipulability / max(mu_current, 1e-9), 1.0)

        pose_cost = pos_err / max(
            profile.max_pose_error_pos_m, 1e-6
        ) + profile.position_ik_pose_rot_weight * (
            rot_err / max(profile.max_pose_error_rot_rad, 1e-6)
        )

        return (
            weighted_dist
            + profile.position_ik_velocity_weight * velocity_dist
            - profile.position_ik_manipulability_weight * normalized_mu
            + profile.position_ik_limit_penalty_weight * limit_penalty
            + profile.position_ik_pose_accuracy_weight * pose_cost
        )

    def _has_elbow_flip(self, candidate_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> bool:
        """Return True if candidate would cause an elbow branch flip vs previous command."""
        prev_j4 = float(previous_qpos_cmd[self._elbow_joint_index])
        cand_j4 = float(candidate_qpos[self._elbow_joint_index])
        delta_j4 = abs(cand_j4 - prev_j4)

        if prev_j4 < self._ELBOW_FLIP_NEG_THRESH_RAD and cand_j4 > self._ELBOW_FLIP_POS_THRESH_RAD:
            return bool(delta_j4 > self._ELBOW_FLIP_MIN_DELTA_RAD)
        if cand_j4 < self._ELBOW_FLIP_NEG_THRESH_RAD and prev_j4 > self._ELBOW_FLIP_POS_THRESH_RAD:
            return bool(delta_j4 > self._ELBOW_FLIP_MIN_DELTA_RAD)
        return False

    @staticmethod
    def _attempt_tag(attempt: str) -> str:
        """Extract the rejection tag from an attempt string ``seed:tag(...)``."""
        if ":" not in attempt:
            return attempt
        return attempt.split(":", 1)[1].split("(", 1)[0]

    @classmethod
    def _classify_attempts(cls, attempts: list[str]) -> str:
        """Classify a failed IK run from its per-attempt tags.

        ``unreachable`` is reserved for when *every* seed failed to converge.
        Otherwise report the candidate rejection criterion, including
        continuity, hardware distance, limits, pose error and self-collision.
        """
        tag_set = {cls._attempt_tag(a) for a in attempts}
        if not tag_set:
            return "unknown"
        if tag_set == {"mplib_failed"}:
            return "unreachable"
        if tag_set & {"jump", "elbow_flip", "branch_jump_l2"}:
            return "delta"
        if tag_set & {"hw_dist", "band_switch"}:
            return "hw_dist"
        if "limits" in tag_set:
            return "limits"
        if "pose_err" in tag_set:
            return "pose_error"
        if "manipulability" in tag_set:
            return "manipulability"
        if "self_collision" in tag_set:
            return "self_collision"
        return "all_filtered"

    @classmethod
    def _build_diagnostic(cls, report: dict[str, Any]) -> dict[str, Any]:
        """Classify failed IK attempts for local diagnostics."""
        attempts = report.get("attempts")
        classification = (
            "invalid_output"
            if report.get("failure_kind") == IKFailureKind.INVALID_OUTPUT
            else cls._classify_attempts(attempts or [])
        )
        failure_reason = str(report.get("failure_reason", ""))
        return {
            "classification": classification,
            "summary": f"Position IK [{classification}]: {failure_reason}",
        }

    def _failed_solution(
        self,
        qpos_cmd: np.ndarray,
        current_qpos: np.ndarray,
        reason: str,
        report: dict[str, Any],
        failure_kind: IKFailureKind = IKFailureKind.GEOMETRY_REJECTED,
        **extra: Any,
    ) -> IKResult:
        """Build a failed IK result."""
        try:
            qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
        except Exception:
            qpos_delta = None
        max_qpos_cmd_delta_deg: float | None = None
        if qpos_delta is not None and np.all(np.isfinite(qpos_delta)):
            max_qpos_cmd_delta_deg = float(np.rad2deg(np.max(np.abs(qpos_delta))))
        failure_report = {
            **report,
            **extra,
        }
        if max_qpos_cmd_delta_deg is not None:
            failure_report["max_qpos_cmd_delta_deg"] = max_qpos_cmd_delta_deg
        return IKResult(
            success=False,
            qpos=None,
            reason=reason,
            report=failure_report,
            failure_kind=failure_kind,
        )

    def _command_from_target_qpos(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        target_qpos: np.ndarray,
        profile: OnlineIKConfig,
        report: dict[str, Any],
    ) -> IKResult:
        """Refine, verify FK error, and assemble IKResult."""
        qpos_cmd = np.asarray(target_qpos, dtype=np.float64).copy()
        if not np.all(np.isfinite(qpos_cmd)):
            return self._failed_solution(
                qpos_cmd,
                current_qpos,
                "Final IK command is non-finite",
                report,
                failure_kind=IKFailureKind.INVALID_OUTPUT,
            )

        # Null-space repulsion preserves EEF pose only to first order; validate FK below.
        if profile.enable_nullspace_optimization:
            try:
                jacobian, _ = self.kin.compute_eef_jacobian_and_pose_world(qpos_cmd)
                qpos_cmd = apply_nullspace_optimization(
                    qpos_cmd,
                    jacobian,
                    self.ik_mgr.joint_limits,
                    step_size_rad=np.deg2rad(profile.nullspace_step_size_deg),
                    margin_deg=profile.nullspace_joint_limit_margin_deg,
                )
            except (ValueError, RuntimeError):
                _now = time.monotonic()
                if _now - self._nullspace_warn_last_s > 5.0:
                    logger.warning(
                        "Nullspace optimization failed — joint-limit repulsion degraded",
                        exc_info=True,
                    )
                    self._nullspace_warn_last_s = _now

        qpos_cmd = np.asarray(
            self.ik_mgr.canonicalize_qpos(qpos_cmd, current_qpos), dtype=np.float64
        )
        if not np.all(np.isfinite(qpos_cmd)):
            return self._failed_solution(
                qpos_cmd,
                current_qpos,
                "Final canonical IK command is non-finite",
                report,
                failure_kind=IKFailureKind.INVALID_OUTPUT,
            )
        outside, _ = self.ik_mgr.limit_violation(qpos_cmd, self.ik_mgr.joint_limits)
        if np.any(outside):
            return self._failed_solution(
                qpos_cmd,
                current_qpos,
                "Final IK command violates joint limits",
                report,
            )

        cmd_pos_error, cmd_rot_error = self.kin.compute_world_pose_error(
            target_eef_pose_world, qpos_cmd
        )
        if not (np.isfinite(cmd_pos_error) and np.isfinite(cmd_rot_error)):
            return self._failed_solution(
                qpos_cmd,
                current_qpos,
                "Final IK pose error is non-finite",
                report,
                failure_kind=IKFailureKind.INVALID_OUTPUT,
            )
        if (
            cmd_pos_error > profile.max_pose_error_pos_m
            or cmd_rot_error > profile.max_pose_error_rot_rad
        ):
            return self._failed_solution(
                qpos_cmd,
                current_qpos,
                "Final IK command exceeds pose-error limits",
                report,
                cmd_tracking_error_pos_m=cmd_pos_error,
                cmd_tracking_error_rot_rad=cmd_rot_error,
            )

        # Refinement can change a previously collision-free candidate.
        if self.ik_mgr.has_self_collision(qpos_cmd):
            return self._failed_solution(
                qpos_cmd,
                current_qpos,
                "self_collision",
                report,
            )

        qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
        if not np.all(np.isfinite(qpos_delta)):
            return self._failed_solution(
                qpos_cmd,
                current_qpos,
                "Final IK command delta is non-finite",
                report,
                failure_kind=IKFailureKind.INVALID_OUTPUT,
            )
        result_report = {
            **report,
            "cmd_tracking_error_pos_m": cmd_pos_error,
            "cmd_tracking_error_rot_rad": cmd_rot_error,
            "qpos_distance_to_current": float(np.linalg.norm(qpos_delta)),
            "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(qpos_delta)))),
        }
        return IKResult(success=True, qpos=qpos_cmd, report=result_report)


def nullspace_projector(J: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Compute null-space projector N = I - J⁺J via SVD.

    For xArm7 (6×7 Jacobian, rank 6): N is 7×7, symmetric, idempotent,
    with one eigenvalue ≈ 1 and six ≈ 0.
    """
    if J.ndim != 2 or not np.all(np.isfinite(J)):
        raise ValueError("Jacobian must be a finite 2-D array")
    return np.eye(J.shape[1]) - np.linalg.pinv(J, rcond=rcond) @ J


def joint_limit_gradient(
    qpos: np.ndarray,
    joint_limits: np.ndarray,
    margin_deg: float = 15.0,
) -> np.ndarray:
    """Quadratic repulsive gradient from joint limits (C¹ continuous).

    V(q) = ((margin-d)/margin)² for d < margin, else 0.
    NaN-safe: returns zeros on non-finite input.
    """
    if not np.all(np.isfinite(qpos)):
        return np.zeros_like(qpos)

    margin = np.deg2rad(margin_deg)
    low = joint_limits[:, 0]
    high = joint_limits[:, 1]
    grad = np.zeros(qpos.shape[0], dtype=np.float64)

    for i in range(qpos.shape[0]):
        d_low = qpos[i] - low[i]
        d_high = high[i] - qpos[i]

        if d_low < margin:
            grad[i] = 2.0 * (margin - d_low) / (margin * margin)
        elif d_high < margin:
            grad[i] = -2.0 * (margin - d_high) / (margin * margin)

    return grad


def apply_nullspace_optimization(
    qpos: np.ndarray,
    jacobian: np.ndarray,
    joint_limits: np.ndarray,
    step_size_rad: float = np.deg2rad(1.0),
    margin_deg: float = 15.0,
) -> np.ndarray:
    """Apply null-space joint-limit repulsion.

    Projects the limit gradient into the self-motion manifold using the
    null-space projector (J @ (qpos' - qpos) ≈ 0).  No posture objective is
    applied away from joint limits: a fixed-magnitude homeward step can cross
    the IK solution on successive frames and create a period-two command.
    """
    grad = joint_limit_gradient(qpos, joint_limits, margin_deg)

    if not np.any(grad):
        return qpos

    N = nullspace_projector(jacobian)
    dq = N @ grad
    dq_max = float(np.max(np.abs(dq)))

    if dq_max > step_size_rad and dq_max > 1e-12:
        dq *= step_size_rad / dq_max

    qpos_new = qpos + dq
    # Skip refinement if the projected qpos crosses a hard joint limit.
    if np.any(qpos_new < joint_limits[:, 0] - 1e-5) or np.any(qpos_new > joint_limits[:, 1] + 1e-5):
        return qpos
    return qpos_new
