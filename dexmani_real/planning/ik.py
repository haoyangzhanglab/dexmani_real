"""Teleoperation IK solver — position IK (MPlib) with deterministic seeding."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from .ik_candidates import IKCandidateManager
    from .kinematics import XArm7Kinematics

from .ik_candidates import is_mplib_success
from .pose_utils import compute_pose_error, ensure_qpos
from .types import IKResult, Pose, TeleopProfile

logger = get_logger(__name__)


class TeleopIKSolver:
    """MPlib position IK with prev_cmd seeding and fast-accept.

    Priority: prev_cmd seed (position IK) → multi-seed fallback → hold.
    Self-collision checks when TeleopProfile.check_self_collision=True.
    """

    # Elbow flip detection thresholds (ref: planner.py check_elbow_consistency).
    _ELBOW_FLIP_NEG_THRESH_RAD: float = np.deg2rad(-5.0)
    _ELBOW_FLIP_POS_THRESH_RAD: float = np.deg2rad(15.0)
    _ELBOW_FLIP_MIN_DELTA_RAD: float = np.deg2rad(40.0)

    def __init__(
        self,
        kin: XArm7Kinematics,
        ik_mgr: IKCandidateManager,
        teleop_profile: TeleopProfile,
        elbow_joint_index: int = 3,
        home_qpos: np.ndarray | None = None,
    ) -> None:
        self.kin = kin
        self.ik_mgr = ik_mgr
        self.profile = teleop_profile
        self._elbow_joint_index = elbow_joint_index
        self._home_qpos = home_qpos
        self._nullspace_warn_last_s: float = 0.0
        self._hold_start: float | None = None
        self._hold_warned: bool = False

    def solve(self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> IKResult:
        """Run teleop IK — prev_cmd seed, multi-seed fallback, hold on failure."""
        t_start = time.perf_counter()

        profile = self.profile
        current_qpos = ensure_qpos(current_qpos, self.kin.dof, "current_qpos")
        previous_qpos_cmd = ensure_qpos(previous_qpos_cmd, self.kin.dof, "previous_qpos_cmd")

        qpos, report = self._solve_position_ik(target_eef_pose_world, current_qpos, previous_qpos_cmd, profile)

        if qpos is not None:
            result = self._command_from_target_qpos(
                target_eef_pose_world=target_eef_pose_world,
                current_qpos=current_qpos,
                previous_qpos_cmd=previous_qpos_cmd,
                target_qpos=qpos,
                profile=profile,
                report=report,
            )
            # Re-canonicalize to current_qpos after nullspace optimization.
            # Nullspace perturbs by <1°, but defense-in-depth: ensure Mode 6
            # always receives a target on the shortest angular path from the
            # physical joint position.  (Root fix is in _solve_position_ik
            # where canonicalize_qpos now uses current_qpos as reference.)
            if result.success and result.qpos is not None:
                result.qpos = self.ik_mgr.canonicalize_qpos(result.qpos, current_qpos)
            # Propagate total solve time to success path (failure path already has it at line 103).
            result.report["ik_timing_ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)
        else:
            dt_total_ms = (time.perf_counter() - t_start) * 1000
            diagnostic = self._build_diagnostic(report)
            result = IKResult(
                success=False,
                qpos=previous_qpos_cmd.copy(),
                reason=diagnostic["summary"],
                held=True,
                report={
                    **report,
                    "held": True,
                    "diagnostic": diagnostic,
                    "ik_timing_ms": round(dt_total_ms, 1),
                },
            )

        if not result.success or result.held:
            if self._hold_start is None:
                self._hold_start = time.monotonic()
            elif time.monotonic() - self._hold_start > 2.0 and not self._hold_warned:
                logger.warning(
                    "IK holding for %.1fs — arm frozen (reason: %s)",
                    time.monotonic() - self._hold_start,
                    result.reason,
                )
                self._hold_warned = True
        else:
            self._hold_start = None
            self._hold_warned = False

        return result

    def _solve_position_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Position IK: prev_cmd seed with fast-accept, multi-seed fallback, scoring."""
        target_pose_base = self.kin.world_to_base_pose(target_eef_pose_world)
        jump_limit = np.deg2rad(self.ik_mgr.profile_array(profile.max_ik_jump_deg, "max_ik_jump_deg"))
        fast_accept_rad = profile.position_ik_fast_accept_rad
        weights = self.ik_mgr.profile_array(profile.joint_weights, "joint_weights")

        seeds = self._make_teleop_seeds(previous_qpos_cmd, current_qpos, profile)

        attempts: list[str] = []
        candidates: list[tuple[np.ndarray, str, float, float]] = []  # (qpos, seed_name, score, manipulability)
        best_fallback: tuple[np.ndarray, str, float] | None = None  # (qpos, seed_name, weighted_dist)

        for seed_name, seed, n_init in seeds:
            _tik0 = time.perf_counter()
            status, raw_qpos = self.ik_mgr.call_mplib_ik(
                target_pose_base,
                seed,
                n_init_qpos=n_init,
                return_closest=True,
            )
            _solve_ms = (time.perf_counter() - _tik0) * 1000.0
            if not is_mplib_success(status) or raw_qpos is None:
                attempts.append(f"{seed_name}:mplib_failed({_solve_ms:.1f}ms)")
                continue

            raw_qpos = np.asarray(raw_qpos, dtype=np.float64)
            # Canonicalize relative to physical encoder position (current_qpos)
            # to avoid Mode 6 taking the "long way" around 2π equivalent joints.
            # previous_qpos_cmd can drift into a different band over many frames.
            qpos = self.ik_mgr.canonicalize_qpos(raw_qpos, current_qpos)

            jacobian, eef_pose_world = self.kin.compute_eef_jacobian_and_pose_world(qpos)
            pos_err, rot_err = compute_pose_error(target_eef_pose_world, eef_pose_world)
            mu = self.kin.manipulability_from_jacobian(jacobian)

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
                attempts.append(f"{seed_name}:{tag}({_solve_ms:.1f}ms)")
                continue

            delta_current = self.ik_mgr.compute_qpos_delta(qpos, current_qpos)
            hw_dist_raw = float(np.max(np.abs(qpos - current_qpos)))
            hw_dist = float(np.max(np.abs(delta_current)))
            weighted_dist = self.ik_mgr.weighted_joint_distance(qpos, current_qpos, weights)

            # Band-mismatch gate: detect 2π band shift vs physical arm position.
            # J1/J3/J5/J7 wrap at 2π; canonicalize_qpos can be defeated at joint
            # limits.  raw - wrapped delta > 90° → reject (else arm rotates
            # ~328° physically while wrapped check sees only 32°).
            _hw_band_mismatch = hw_dist_raw - hw_dist
            _hw_band_limit_rad = np.deg2rad(90.0)
            if _hw_band_mismatch > _hw_band_limit_rad:
                attempts.append(
                    f"{seed_name}:band_switch(raw={np.rad2deg(hw_dist_raw):.0f}deg, wrapped={np.rad2deg(hw_dist):.0f}deg)"
                )
                continue

            # Hardware-distance gate: reject candidates too far from the
            # physical arm position (genuine tracking lag, not band mismatch).
            # Threshold 150° allows normal lag but blocks ~172° branch jumps.
            _hw_limit_rad = np.deg2rad(150.0)
            if hw_dist > _hw_limit_rad:
                attempts.append(f"{seed_name}:hw_dist({_solve_ms:.1f}ms, {np.rad2deg(hw_dist):.0f}deg)")
                continue

            if best_fallback is None or weighted_dist < best_fallback[2]:
                best_fallback = (qpos.copy(), seed_name, weighted_dist)
            attempts.append(f"{seed_name}:ok({_solve_ms:.1f}ms)")

            if seed_name == "prev_cmd" and hw_dist <= fast_accept_rad:
                return qpos, {"method": "position_ik", "seed": seed_name, "attempts": attempts}

            score = self._score_candidate(
                weighted_dist=weighted_dist,
                manipulability=mu,
                pos_err=pos_err,
                rot_err=rot_err,
                qpos=qpos,
                previous_qpos_cmd=previous_qpos_cmd,
                profile=profile,
            )
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

        if best_fallback is not None:
            qpos, seed_name, _ = best_fallback
            return qpos, {"method": "position_ik", "seed": seed_name, "fallback": True, "attempts": attempts}

        return None, {"method": "position_ik", "failure_reason": f"all failed: {attempts}"}

    def _validate_ik_candidate(
        self,
        qpos: np.ndarray,
        pos_err: float,
        rot_err: float,
        mu: float,
        previous_qpos_cmd: np.ndarray,
        jump_limit: np.ndarray,
        profile: TeleopProfile,
    ) -> tuple[bool, str]:
        """Validate IK result: pose error, manipulability, joint jump, elbow flip, branch jump L2.

        Returns (passed, tag) — tag is the first failing check name or "ok".
        """
        if pos_err > profile.max_pose_error_pos_m or rot_err > profile.max_pose_error_rot_rad:
            return False, "pose_err"

        if profile.position_ik_min_manipulability > 0 and mu < profile.position_ik_min_manipulability:
            return False, "manipulability"

        delta_prev = self.ik_mgr.compute_qpos_delta(qpos, previous_qpos_cmd)
        if np.any(np.abs(delta_prev) > jump_limit):
            return False, "jump"

        if self._has_elbow_flip(qpos, previous_qpos_cmd):
            return False, "elbow_flip"

        # L2 catch-all for multi-joint branch jumps the J4-only elbow check misses.
        # 120° L2 is 4-8× normal 16 Hz frame-to-frame motion (< 30°).
        if float(np.linalg.norm(delta_prev)) > np.deg2rad(120):
            return False, "branch_jump_l2"

        return True, "ok"

    def _make_teleop_seeds(
        self,
        prev_cmd: np.ndarray,
        current_qpos: np.ndarray,
        profile: TeleopProfile,
    ) -> list[tuple[str, np.ndarray, int]]:
        """Generate teleop IK seeds: prev_cmd (n_init_qpos=1) first, then current_qpos + random perturbations.

        n_init_qpos=1 avoids 3× MPlib perturbation overhead (~18 ms → ~6 ms).
        """
        seeds: list[tuple[str, np.ndarray, int]] = [
            ("prev_cmd", prev_cmd.copy(), 1),
            ("current_qpos", current_qpos.copy(), 1),
        ]
        seed = profile.teleop_ik_seed
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        offsets_rad = np.deg2rad(profile.position_ik_seed_offset_deg)
        for i in range(profile.position_ik_num_random_seeds):
            seed = prev_cmd + rng.uniform(-offsets_rad, offsets_rad, self.kin.dof)
            seeds.append((f"random_{i}", seed, 1))
        return seeds

    def _score_candidate(
        self,
        weighted_dist: float,
        manipulability: float,
        pos_err: float,
        rot_err: float,
        qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> float:
        """Score an IK candidate (lower is better).

        = weighted_joint_distance + velocity_weight*velocity_dist
          - manipulability_weight*mu + limit_penalty_weight*penalty + pose_accuracy_weight*pose_cost.
        """
        limit_penalty = self.ik_mgr.joint_limit_penalty(qpos, self.ik_mgr.joint_limits)

        vel_weights = (
            profile.velocity_joint_weights if profile.velocity_joint_weights is not None else profile.joint_weights
        )
        velocity_dist = self.ik_mgr.weighted_joint_distance(qpos, previous_qpos_cmd, vel_weights)

        pose_cost = pos_err / max(profile.max_pose_error_pos_m, 1e-6) + rot_err / max(
            profile.max_pose_error_rot_rad, 1e-6
        )

        return (
            weighted_dist
            + profile.position_ik_velocity_weight * velocity_dist
            - profile.position_ik_manipulability_weight * manipulability
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
    def _build_diagnostic(report: dict[str, Any]) -> dict[str, Any]:
        """Build structured IK failure diagnostic."""
        diagnostic: dict[str, Any] = {"classification": "unknown", "summary": ""}

        failure_reason = str(report.get("failure_reason", ""))
        if "all failed" in failure_reason:
            attempts_str = str(report.get("attempts", failure_reason))
            if "mplib_failed" in attempts_str and "ok" not in attempts_str:
                diagnostic["classification"] = "unreachable"
            elif "ok" in attempts_str:
                diagnostic["classification"] = "all_filtered"
            else:
                diagnostic["classification"] = "unreachable"
        elif "mplib_failed" in failure_reason:
            diagnostic["classification"] = "unreachable"
        elif "pose_err" in failure_reason:
            diagnostic["classification"] = "pose_error"
        elif "jump" in failure_reason:
            diagnostic["classification"] = "delta"
        else:
            diagnostic["classification"] = "other"

        diagnostic["summary"] = f"Position IK [{diagnostic['classification']}]: {failure_reason}"
        return diagnostic

    def _check_teleop_collision_gate(
        self,
        qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> tuple[str | None, dict[str, Any]]:
        """Self-collision gate. Returns (reason, extra_report) or (None, {})."""
        if not profile.check_self_collision:
            return None, {}

        if self.ik_mgr.has_self_collision(qpos_cmd):
            info = self.ik_mgr.check_self_collision(qpos_cmd)
            if info:
                return (
                    f"IK result in self-collision ({info.summary}), holding.",
                    {"collision_type": "self", "collision": info.to_dict()},
                )
        return None, {}

    def _make_collision_held(
        self,
        qpos_cmd: np.ndarray,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        reason: str,
        report: dict[str, Any],
        **extra: Any,
    ) -> IKResult:
        """Build a held IKResult for collision-gate rejection."""
        qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
        return IKResult(
            success=False,
            qpos=previous_qpos_cmd.copy(),
            reason=reason,
            report={
                **report,
                "held": True,
                "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(qpos_delta)))),
                **extra,
            },
            held=True,
        )

    def _command_from_target_qpos(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        target_qpos: np.ndarray,
        profile: TeleopProfile,
        report: dict[str, Any],
    ) -> IKResult:
        """Nullspace-optimize, collision-check, and assemble IKResult."""
        qpos_cmd = target_qpos.copy()

        # EEF pose computed once, reused for tracking error.
        # Nullspace preserves EEF pose (J · dq_null = 0), so it stays valid.
        eef_pose_world: Pose | None = None

        # Null-space joint-limit repulsion (zero EEF error by construction).
        if profile.enable_nullspace_optimization:
            try:
                jacobian, eef_pose_world = self.kin.compute_eef_jacobian_and_pose_world(qpos_cmd)

                qpos_cmd = apply_nullspace_optimization(
                    qpos_cmd,
                    jacobian,
                    self.ik_mgr.joint_limits,
                    step_size_rad=np.deg2rad(profile.nullspace_step_size_deg),
                    margin_deg=profile.nullspace_joint_limit_margin_deg,
                    home_qpos=self._home_qpos,
                )
            except (ValueError, RuntimeError):
                _now = time.monotonic()
                if _now - self._nullspace_warn_last_s > 5.0:
                    logger.warning(
                        "Nullspace optimization failed — joint-limit repulsion degraded",
                        exc_info=True,
                    )
                    self._nullspace_warn_last_s = _now

        # Collision safety gate.
        collision_reason, collision_extra = self._check_teleop_collision_gate(qpos_cmd, profile)
        if collision_reason is not None:
            return self._make_collision_held(
                qpos_cmd,
                current_qpos,
                previous_qpos_cmd,
                collision_reason,
                report,
                **collision_extra,
            )

        qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
        if eef_pose_world is not None:
            cmd_pos_error, cmd_rot_error = compute_pose_error(target_eef_pose_world, eef_pose_world)
        else:
            cmd_pos_error, cmd_rot_error = self.kin.compute_world_pose_error(target_eef_pose_world, qpos_cmd)
        result_report = {
            **report,
            "cmd_tracking_error_pos_m": cmd_pos_error,
            "cmd_tracking_error_rot_rad": cmd_rot_error,
            "qpos_distance_to_current": float(np.linalg.norm(qpos_delta)),
            "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(qpos_delta)))),
        }
        return IKResult(success=True, qpos=qpos_cmd, report=result_report)


# Null-space optimization helpers


def nullspace_projector(J: np.ndarray) -> np.ndarray:
    """Compute null-space projector N = I - J⁺J via SVD.

    For xArm7 (6×7 Jacobian, rank 6): N is 7×7, symmetric, idempotent,
    with one eigenvalue ≈ 1 and six ≈ 0.
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    V = Vt.T  # 7×6
    return np.eye(J.shape[1]) - V @ V.T


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
    home_qpos: np.ndarray | None = None,
) -> np.ndarray:
    """Apply null-space joint-limit repulsion + optional home bias.

    Projects the combined gradient (limit repulsion + homeward bias) into the
    self-motion manifold using the null-space projector (J @ (qpos' - qpos) ≈ 0).
    Skips SVD when no joint is near a limit and no home bias is configured.

    The home bias is normalised to a unit-length direction so it shares the
    *step_size_rad* budget equally with joint-limit repulsion.  When only one
    force is active, it gets the full budget.
    """
    grad = joint_limit_gradient(qpos, joint_limits, margin_deg)

    if home_qpos is not None:
        home_dir = home_qpos - qpos
        home_max = float(np.max(np.abs(home_dir)))
        if home_max > 1e-12:
            home_dir = home_dir / home_max  # unit-length direction toward home
            grad = grad + home_dir if np.any(grad) else home_dir

    if not np.any(grad):
        return qpos

    N = nullspace_projector(jacobian)
    dq = N @ grad
    dq_max = float(np.max(np.abs(dq)))

    if dq_max > step_size_rad and dq_max > 1e-12:
        dq *= step_size_rad / dq_max

    return qpos + dq
