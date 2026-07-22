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
    """Teleoperation IK — MPlib position IK with prev_cmd seeding.

    Priority:
      1. Position IK (MPlib) — seeded from previous_qpos_cmd,
         return_closest=True, n_init_qpos=3. Branch-stable via elbow flip
         check + fast-accept within 15°.
      2. Fail → hold previous command.

    Speed limiting is handled by ArmInnerLoop (Mode 6): per-step joint
    delta clamp + firmware online trajectory planning.
    Self-collision checks are done when TeleopProfile.check_self_collision=True.

    ref: LeFranX current_distance penalty, ssik seed_tolerance.
    """

    # xArm7 joint4 (elbow) — distinguishes elbow-up vs elbow-down IK branches.
    _ELBOW_JOINT_INDEX: int = 3

    # Elbow flip detection thresholds (ref: planner.py check_elbow_consistency).
    _ELBOW_FLIP_NEG_THRESH_RAD: float = np.deg2rad(-5.0)
    _ELBOW_FLIP_POS_THRESH_RAD: float = np.deg2rad(15.0)
    _ELBOW_FLIP_MIN_DELTA_RAD: float = np.deg2rad(40.0)

    def __init__(self, kin: XArm7Kinematics, ik_mgr: IKCandidateManager, teleop_profile: TeleopProfile) -> None:
        self.kin = kin
        self.ik_mgr = ik_mgr
        self.profile = teleop_profile
        self._nullspace_warn_throttle: int = 0
        self._hold_start: float | None = None
        self._hold_warned: bool = False

    # ── Public API ──

    def solve(
        self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray
    ) -> IKResult:
        """Teleop IK — MPlib position IK, hold on failure.

        Tries prev_cmd seed (n_init_qpos=3) → current_qpos seed (n_init_qpos=1).
        Both use return_closest=True for deterministic, branch-stable results.
        """
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
        else:
            # ── Hold ──
            dt_total_ms = (time.perf_counter() - t_start) * 1000
            diagnostic = self._build_diagnostic(report)
            result = IKResult(
                success=False,
                qpos=previous_qpos_cmd.copy(),
                reason=diagnostic["summary"],
                report={
                    **report,
                    "held": True,
                    "diagnostic": diagnostic,
                    "ik_timing_ms": round(dt_total_ms, 1),
                },
            )

        # ── Hold timeout tracking ──
        if not result.success or result.held:
            if self._hold_start is None:
                self._hold_start = time.time()
            elif time.time() - self._hold_start > 2.0 and not self._hold_warned:
                logger.warning(
                    "IK holding for %.1fs — arm frozen (reason: %s)",
                    time.time() - self._hold_start,
                    result.reason,
                )
                self._hold_warned = True
        else:
            self._hold_start = None
            self._hold_warned = False

        return result

    # ── Position IK ──

    def _solve_position_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Position IK with multi-candidate scoring for dexterous manipulation.

        Phase 1: prev_cmd seed, fast-accept within 15° (deterministic, branch-stable).
        Phase 2: multi-seed search (5 seeds) with manipulability-weighted scoring.
        """
        target_pose_base = self.kin.world_to_base_pose(target_eef_pose_world)
        jump_limit = np.deg2rad(self.ik_mgr.profile_array(profile.max_ik_jump_deg, "max_ik_jump_deg"))
        fast_accept_rad = profile.position_ik_fast_accept_rad
        weights = self.ik_mgr.profile_array(profile.joint_weights, "joint_weights")

        seeds = self._make_teleop_seeds(previous_qpos_cmd, current_qpos, profile)

        attempts: list[str] = []
        candidates: list[tuple[np.ndarray, str, float, float]] = []  # (qpos, seed_name, score, manipulability)
        best_fallback: tuple[np.ndarray, str, float] | None = None  # fallback: (qpos, seed_name, weighted_dist)

        for seed_name, seed, n_init in seeds:
            status, raw_qpos = self.ik_mgr.call_mplib_ik(
                target_pose_base, seed, n_init_qpos=n_init, return_closest=True,
            )

            if not is_mplib_success(status) or raw_qpos is None:
                attempts.append(f"{seed_name}:mplib_failed")
                continue

            raw_qpos = np.asarray(raw_qpos, dtype=np.float64)
            qpos = self.ik_mgr.canonicalize_qpos(raw_qpos, previous_qpos_cmd)
            pos_err, rot_err = self.kin.compute_world_pose_error(target_eef_pose_world, qpos)
            delta_prev = self.ik_mgr.compute_qpos_delta(qpos, previous_qpos_cmd)

            if pos_err > profile.max_pose_error_pos_m or rot_err > profile.max_pose_error_rot_rad:
                attempts.append(f"{seed_name}:pose_err")
                continue

            if np.any(np.abs(delta_prev) > jump_limit):
                attempts.append(f"{seed_name}:jump")
                continue

            if self._has_elbow_flip(qpos, previous_qpos_cmd):
                attempts.append(f"{seed_name}:elbow_flip")
                continue

            # L2 catch-all for unexpected multi-joint branch jumps (shoulder
            # reconfiguration, wrist singularity flips) that the J4-only elbow
            # check misses.  120° L2 is 4-8× normal frame-to-frame motion at
            # 16 Hz (< 30°), conservative against any single-frame discontinuity.
            if float(np.linalg.norm(delta_prev)) > np.deg2rad(120):
                attempts.append(f"{seed_name}:branch_jump_l2")
                continue

            hw_dist = float(np.max(np.abs(self.ik_mgr.compute_qpos_delta(qpos, current_qpos))))
            weighted_dist = self.ik_mgr.weighted_joint_distance(qpos, current_qpos, weights)

            # Track best fallback (closest to current by weighted distance).
            if best_fallback is None or weighted_dist < best_fallback[2]:
                best_fallback = (qpos.copy(), seed_name, weighted_dist)
            attempts.append(f"{seed_name}:ok")

            # Fast path: prev_cmd seed, close to hardware → accept immediately.
            if seed_name == "prev_cmd" and hw_dist <= fast_accept_rad:
                return qpos, {"method": "position_ik", "seed": seed_name, "attempts": attempts}

            # Phase 2: collect candidates for scoring.
            mu = self.kin.compute_manipulability(qpos)
            score = self._score_candidate(
                weighted_dist=weighted_dist,
                manipulability=mu,
                qpos=qpos,
                previous_qpos_cmd=previous_qpos_cmd,
                profile=profile,
            )
            candidates.append((qpos.copy(), seed_name, score, mu))

        # ── Return best candidate by score ──
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

        # ── No scored candidate → use best fallback (closest to current) ──
        if best_fallback is not None:
            qpos, seed_name, _ = best_fallback
            return qpos, {"method": "position_ik", "seed": seed_name, "fallback": True, "attempts": attempts}

        return None, {"method": "position_ik", "failure_reason": f"all failed: {attempts}"}

    def _make_teleop_seeds(
        self, prev_cmd: np.ndarray, current_qpos: np.ndarray, profile: TeleopProfile,
    ) -> list[tuple[str, np.ndarray, int]]:
        """Generate teleop IK seeds: prev_cmd, current_qpos, + random around prev_cmd."""
        seeds: list[tuple[str, np.ndarray, int]] = [
            ("prev_cmd", prev_cmd.copy(), 3),
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
        qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> float:
        """Score an IK candidate. Lower is better.

        score = weighted_joint_distance               (smoothness vs hardware)
              + velocity_weight * velocity_dist       (temporal coherence vs prev cmd)
              - manipulability_weight * mu            (dexterity)
              + limit_penalty_weight * penalty        (adjustment room)

        The velocity term uses dedicated velocity_joint_weights (tuned for
        inertia/responsiveness) when set, falling back to joint_weights.
        """
        limits = self.ik_mgr.joint_limits
        center = 0.5 * (limits[:, 0] + limits[:, 1])
        half_range = np.maximum(0.5 * (limits[:, 1] - limits[:, 0]), 1e-6)
        limit_penalty = float(np.sum(((qpos - center) / half_range) ** 2))

        vel_weights = (
            profile.velocity_joint_weights
            if profile.velocity_joint_weights is not None
            else profile.joint_weights
        )
        velocity_dist = self.ik_mgr.weighted_joint_distance(qpos, previous_qpos_cmd, vel_weights)

        return (
            weighted_dist
            + profile.position_ik_velocity_weight * velocity_dist
            - profile.position_ik_manipulability_weight * manipulability
            + profile.position_ik_limit_penalty_weight * limit_penalty
        )

    # ── Elbow branch flip detection ──

    def _has_elbow_flip(self, candidate_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> bool:
        """Return True if candidate would cause an elbow branch flip vs previous command."""
        prev_j4 = float(previous_qpos_cmd[self._ELBOW_JOINT_INDEX])
        cand_j4 = float(candidate_qpos[self._ELBOW_JOINT_INDEX])
        delta_j4 = abs(cand_j4 - prev_j4)

        if prev_j4 < self._ELBOW_FLIP_NEG_THRESH_RAD and cand_j4 > self._ELBOW_FLIP_POS_THRESH_RAD:
            return bool(delta_j4 > self._ELBOW_FLIP_MIN_DELTA_RAD)
        if cand_j4 < self._ELBOW_FLIP_NEG_THRESH_RAD and prev_j4 > self._ELBOW_FLIP_POS_THRESH_RAD:
            return bool(delta_j4 > self._ELBOW_FLIP_MIN_DELTA_RAD)
        return False

    # ── Diagnostics ──

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

    # ── Command assembly ──

    def _check_teleop_collision_gate(
        self, qpos_cmd: np.ndarray, profile: TeleopProfile,
    ) -> tuple[str | None, dict[str, Any]]:
        """Self + env collision gate. Returns (reason, extra_report) or (None, {})."""
        if not profile.check_self_collision and not profile.check_env_collision:
            return None, {}

        if profile.check_self_collision:
            if self.ik_mgr.has_self_collision(qpos_cmd):
                info = self.ik_mgr.check_self_collision(qpos_cmd)
                if info:
                    return (
                        f"IK result in self-collision ({info.summary}), holding.",
                        {"collision": info.to_dict()},
                    )

        if profile.check_env_collision and self.ik_mgr.has_env_collision(qpos_cmd):
            return (
                "IK result in environment collision (table/obstacle), holding.",
                {"env_collision": True},
            )
        return None, {}

    def _command_from_target_qpos(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        target_qpos: np.ndarray,
        profile: TeleopProfile,
        report: dict[str, Any],
    ) -> IKResult:
        """Nullspace-optimize, collision-check, and assemble IKResult.

        canonicalize_qpos already applied at ik.py:124 to each raw MPlib result;
        nullspace optimization perturbs by <1° so a second wrap is a no-op.
        """
        qpos_cmd = target_qpos.copy()

        # EEF pose at qpos_cmd — computed once, reused for tracking error below.
        # When nullspace is enabled, compute_eef_jacobian_and_pose_world returns
        # both Jacobian and world-frame pose in a single FK call.  Nullspace
        # preserves EEF pose by construction (J · dq_null = 0), so the original
        # pose_world is still valid after nullspace adjustment.
        eef_pose_world: Pose | None = None

        # Null-space joint-limit repulsion (zero EEF error by construction).
        if profile.enable_nullspace_optimization:
            try:
                jacobian, eef_pose_world = self.kin.compute_eef_jacobian_and_pose_world(qpos_cmd)
                from .nullspace import apply_nullspace_optimization

                qpos_cmd = apply_nullspace_optimization(
                    qpos_cmd,
                    jacobian,
                    self.ik_mgr.joint_limits,
                    step_size_rad=np.deg2rad(profile.nullspace_step_size_deg),
                    margin_deg=profile.nullspace_joint_limit_margin_deg,
                )
            except (ValueError, RuntimeError):
                if not self._nullspace_warn_throttle:
                    logger.warning(
                        "Nullspace optimization failed — joint-limit repulsion degraded",
                        exc_info=True,
                    )
                    self._nullspace_warn_throttle = 80  # ~5s at 16Hz
                if self._nullspace_warn_throttle > 0:
                    self._nullspace_warn_throttle -= 1

        # ── Collision safety gates ──
        collision_reason, collision_extra = self._check_teleop_collision_gate(qpos_cmd, profile)
        if collision_reason is not None:
            if "environment collision" in collision_reason:
                current_eef = self.kin.compute_eef_pose_world(current_qpos)
                if target_eef_pose_world.p[2] > current_eef.p[2] + 0.001:
                    logger.warning(
                        "Allowing upward recovery through env collision gate (target_z=%.3f, current_z=%.3f)",
                        target_eef_pose_world.p[2], current_eef.p[2],
                    )
                else:
                    qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
                    return IKResult(
                        success=False,
                        qpos=previous_qpos_cmd.copy(),
                        reason=collision_reason,
                        report={
                            **report,
                            "held": True,
                            "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(qpos_delta)))),
                            **collision_extra,
                        },
                        held=True,
                    )
            else:
                qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
                return IKResult(
                    success=False,
                    qpos=previous_qpos_cmd.copy(),
                    reason=collision_reason,
                    report={
                        **report,
                        "held": True,
                        "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(qpos_delta)))),
                        **collision_extra,
                    },
                    held=True,
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