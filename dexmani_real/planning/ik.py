"""Teleoperation IK solver — differential IK (primary) with position IK fallback."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.utils.ik_profiler import IKProfiler
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from .ik_candidates import IKCandidateManager
    from .kinematics import XArm7Kinematics

from .pose_utils import ensure_qpos, pose_error_vector
from .types import IKResult, Pose, TeleopProfile

logger = get_logger(__name__)

# Opt-in py-spy profiler for DLS timing anomaly diagnosis.
# Enable with DEXMANI_IK_PROFILE=1.  Triggers when DLS > 5 ms/iter
# (normal: ~16 μs; production baseline: 0.6–2.8 ms).
_ik_profiler = IKProfiler(dls_threshold_ms=5.0, cooldown_s=30.0)


class TeleopIKSolver:
    """Teleoperation IK — deterministic primary, stochastic fallback.

    ref: BunnyVisionPro DLS (damped least-squares as primary, deterministic).
         LeFranX current_distance penalty (hardware-closest candidate selection).

    Priority:
      1. Differential IK — deterministic, no branch switching. Same seed +
         same target = same result. Fast (< 1ms).
      2. Position IK (MPlib) — stochastic fallback for large tracking
         errors or near-singularity where diff IK can't converge.
         Picks the hardware-closest valid candidate to avoid branch switching.
      3. Both fail → hold previous command.

    Speed limiting is handled by ArmInnerLoop (50 Hz, Mode 6): per-step joint
    delta clamp + firmware online trajectory planning.
    Self-collision checks are done when TeleopProfile.check_self_collision=True.


    References:
      - BunnyVisionPro DLS + LeFranX scoring)
      - BunnyVisionPro DLS fallback)
      - LeFranX current_distance penalty)
      - ssik explain=True pattern)
      - ssik max_solutions=1
      - ssik seed_tolerance hard boundary)
    """

    # xArm7 joint4 (elbow) — distinguishes elbow-up vs elbow-down IK branches.
    # Range [-11°, 225°]: negative values ≈ elbow-up branch, large positive ≈ elbow-down.
    _ELBOW_JOINT_INDEX: int = 3

    # Elbow flip detection thresholds (ref: planner.py check_elbow_consistency).
    _ELBOW_FLIP_NEG_THRESH_RAD: float = np.deg2rad(-5.0)  # below this → elbow-up branch
    _ELBOW_FLIP_POS_THRESH_RAD: float = np.deg2rad(15.0)  # above this → elbow-down branch
    _ELBOW_FLIP_MIN_DELTA_RAD: float = np.deg2rad(40.0)  # minimum J4 change to count as flip

    # F3: Wall-clock budget for teleop IK solve.  At 16 Hz (62.5 ms/frame),
    # 10 ms leaves ample headroom for the rest of the pipeline (mapper, retargeting,
    # validate_action, send).  Exceeding this budget means the solver is stuck on a
    # near-singular or ill-conditioned pose — better to hold and try next frame.
    _IK_TIMEOUT_S: float = 0.020  # 10→20ms: 62.5ms frame leaves 42.5ms headroom; covers GC pauses + iter count

    def __init__(self, kin: XArm7Kinematics, ik_mgr: IKCandidateManager, teleop_profile: TeleopProfile) -> None:
        self.kin = kin
        self.ik_mgr = ik_mgr
        self.profile = teleop_profile
        self._ik_timing_log_throttle: int = 0  # countdown for ~5s throttled timing log
        # Pre-allocated buffer for DLS (6x6 J·J^T), reused every iteration.
        # Eliminates ~5 np.array allocations per DLS call (was ~50-70/IK solve).
        self._JJt_buf = np.empty((6, 6), dtype=np.float64)

    # Public API

    def solve(self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> IKResult:
        """Deterministic IK for real-time teleop (ref: BunnyVisionPro DLS + LeFranX scoring).

        Priority (deterministic first, stochastic fallback):
          1. Differential IK (damped least-squares, deterministic) — primary.
             Same seed + same target = same result, no branch switching.
          2. Position IK (MPlib, stochastic) — fallback for large tracking
             errors or near-singularity where diff IK fails.
          3. Both fail → hold previous command.
        """
        t_start = time.perf_counter()

        profile = self.profile
        current_qpos = ensure_qpos(current_qpos, self.kin.dof, "current_qpos")
        previous_qpos_cmd = ensure_qpos(previous_qpos_cmd, self.kin.dof, "previous_qpos_cmd")

        # ── Step 1: Differential IK (deterministic, primary) ──
        diff_ik_failed = False
        diff_ik_reason = ""
        dt_diff_s: float = 0.0
        if profile.use_differential_ik_fallback:
            diff_result = self.solve_differential_ik(
                target_eef_pose_world,
                current_qpos,
                previous_qpos_cmd,
                profile,
            )
            dt_diff_s = time.perf_counter() - t_start
            if diff_result.success:
                # Verify the diff IK result actually reaches the target.
                pos_err, rot_err = self.kin.compute_world_pose_error(
                    target_eef_pose_world,
                    diff_result.qpos,  # type: ignore[arg-type]  # IKResult.qpos is Optional, but success=True ⇒ non-None
                )
                if pos_err <= profile.max_pose_error_pos_m and rot_err <= profile.max_pose_error_rot_rad:
                    self._maybe_log_ik_timing(dt_diff_s, 0.0, diff_result.report.get("iterations", 0))
                    return diff_result
                # Diff IK converged but pose error too large → fall through to position IK.
                diff_ik_failed = True
                diff_ik_reason = f"pose_error({pos_err:.4f}m,{rot_err:.4f}rad)"
            else:
                diff_ik_failed = True
                diff_ik_reason = diff_result.reason

        # ── Step 2: Position IK (stochastic fallback) ──
        dt_pos_s: float = 0.0
        position_report: dict[str, Any] = {}
        if profile.use_position_ik:
            # F3: skip the expensive MPlib position-IK fallback if diff IK
            # already exhausted the time budget — better to hold one frame
            # than to blow the 62.5 ms pipeline window.
            if time.perf_counter() - t_start > self._IK_TIMEOUT_S:
                logger.warning(
                    "IK timeout: diff IK took %.1fms, skipping position IK fallback",
                    dt_diff_s * 1000,
                )
                position_report = {"teleop_ik_method": "position_ik", "failure_reason": "timeout_skipped"}
            else:
                qpos, position_report = self.solve_position_ik(
                    target_eef_pose_world,
                    current_qpos,
                    previous_qpos_cmd,
                    profile,
                )
                dt_pos_s = time.perf_counter() - t_start - dt_diff_s
                if qpos is not None:
                    self._maybe_log_ik_timing(dt_diff_s, dt_pos_s, 0)
                    return self.command_from_target_qpos(
                        target_eef_pose_world=target_eef_pose_world,
                        current_qpos=current_qpos,
                        previous_qpos_cmd=previous_qpos_cmd,
                        target_qpos=qpos,
                        profile=profile,
                        report=position_report,
                        method="position_ik",
                    )

        # ── All strategies failed → hold ──
        dt_total_s = time.perf_counter() - t_start
        self._maybe_log_ik_timing(dt_diff_s, dt_pos_s, 0)
        diagnostic = self._build_ik_diagnostic(
            diff_ik_failed=diff_ik_failed,
            diff_ik_reason=diff_ik_reason if diff_ik_failed else "",
            position_report=position_report,
        )
        return IKResult(
            success=False,
            qpos=previous_qpos_cmd.copy(),
            reason=diagnostic["summary"],
            report={"held": True, "diagnostic": diagnostic, "ik_timing_ms": round(dt_total_s * 1000, 1)},
        )

    # IK timing diagnostic (throttled ~5s or on threshold exceed)

    def _maybe_log_ik_timing(self, dt_diff_s: float, dt_pos_s: float, diff_iters: int) -> None:
        """Log IK timing diagnostic, throttled to ~once per 5s or on >15ms total.

        The warning threshold (15ms) sits below the diff-IK timeout budget
        (``_IK_TIMEOUT_S`` = 20ms) so we get a heads-up before the solver
        actually aborts.
        """
        dt_total_ms = (dt_diff_s + dt_pos_s) * 1000
        threshold_exceeded = dt_total_ms > 15.0

        if self._ik_timing_log_throttle > 0 and not threshold_exceeded:
            self._ik_timing_log_throttle -= 1
            return

        if threshold_exceeded:
            logger.warning(
                "IK timing: diff=%.1fms (iters=%d) + pos=%.1fms = total=%.1fms (>15ms, budget=20ms)",
                dt_diff_s * 1000,
                diff_iters,
                dt_pos_s * 1000,
                dt_total_ms,
            )
        else:
            logger.debug(
                "IK timing: diff=%.1fms (iters=%d) + pos=%.1fms = total=%.1fms",
                dt_diff_s * 1000,
                diff_iters,
                dt_pos_s * 1000,
                dt_total_ms,
            )
        self._ik_timing_log_throttle = 80  # ~5s at 16Hz

    # Position IK — simplified single-seed with fallback

    def solve_position_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Position IK fallback: try prev_cmd seed → current_qpos seed.

        Only called when diff IK fails (large tracking error, near-singularity).
        Picks the hardware-closest valid candidate to avoid branch switching.
        """
        target_pose_base = self.kin.world_to_base_pose(target_eef_pose_world)
        jump_limit = np.deg2rad(self.ik_mgr.profile_array(profile.max_ik_jump_deg, "max_ik_jump_deg"))
        fast_accept_rad = profile.position_ik_fast_accept_rad

        seeds: list[tuple[str, np.ndarray, int]] = [
            ("prev_cmd", previous_qpos_cmd.copy(), 3),
            ("current_qpos", current_qpos.copy(), 2),
        ]

        attempts: list[str] = []
        best: tuple[np.ndarray, str, float] | None = None  # (qpos, seed_name, weighted_dist)
        weights = self.ik_mgr.profile_array(profile.joint_weights, "joint_weights")

        for seed_name, seed, n_init in seeds:
            status, raw_qpos = self.ik_mgr.call_mplib_ik(
                target_pose_base,
                seed,
                n_init_qpos=n_init,
                return_closest=True,
            )

            if not status.lower().startswith("success") or raw_qpos is None:
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

            # Elbow branch consistency: reject candidates that would flip the
            # elbow (joint4) from one branch to the other relative to the
            # previous command.  MPlib's stochastic IK can produce valid-looking
            # solutions on either elbow branch; this guards against a visually
            # jarring arm "snap" when the position IK fallback picks the wrong branch.
            # (ref: planner.py check_elbow_consistency — same thresholds)
            if self._has_elbow_flip(qpos, previous_qpos_cmd):
                attempts.append(f"{seed_name}:elbow_flip")
                continue

            # Per-frame safety gate: max single-joint delta (L-infinity).
            # Guards against any individual joint jumping too far regardless of
            # how "cheap" the weighted metric considers that joint.
            hw_dist = float(np.max(np.abs(self.ik_mgr.compute_qpos_delta(qpos, current_qpos))))

            # Candidate ranking: weighted, range-normalised L2 distance.
            # Base joints (high weight) are penalised more than wrist joints,
            # so the solver prefers solutions that keep the base stable and
            # move the wrist for orientation tracking.
            # Ref: LeFranX weighted_ik.cpp:62-69 calculate_normalized_weighted_distance.
            weighted_dist = self.ik_mgr.weighted_joint_distance(qpos, current_qpos, weights)

            if best is None or weighted_dist < best[2]:
                best = (qpos.copy(), seed_name, weighted_dist)
                attempts.append(f"{seed_name}:ok")

            # Fast path: prev_cmd seed, close to hardware → accept immediately.
            # Uses L-infinity (not weighted distance) as a per-joint safety gate
            # — if ANY single joint has jumped more than fast_accept_rad, we
            # keep searching for a safer candidate.
            # (ref: ssik seed_tolerance hard boundary)
            if seed_name == "prev_cmd" and hw_dist <= fast_accept_rad:
                return qpos, {"teleop_ik_method": "position_ik", "seed": seed_name, "attempts": attempts}

            # Early exit: any seed with excellent quality → stop searching
            # (ref: ssik max_solutions=1 — first good solution wins)
            quality_pos = pos_err < profile.max_pose_error_pos_m * 0.3
            quality_rot = rot_err < profile.max_pose_error_rot_rad * 0.3
            quality_delta = hw_dist < fast_accept_rad * 0.5
            if quality_pos and quality_rot and quality_delta:
                return qpos, {
                    "teleop_ik_method": "position_ik",
                    "seed": seed_name,
                    "attempts": attempts,
                    "early_exit": "excellent_quality",
                }

        if best is not None:
            qpos, seed_name, _ = best
            return qpos, {"teleop_ik_method": "position_ik", "seed": seed_name, "fallback": True, "attempts": attempts}

        return None, {"teleop_ik_method": "position_ik", "failure_reason": f"all failed: {attempts}"}

    # ── Elbow branch flip detection ──

    def _has_elbow_flip(self, candidate_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> bool:
        """Return True if ``candidate_qpos`` would cause an elbow branch flip vs ``previous_qpos_cmd``.

        xArm7 joint4 (elbow, index 3) has range [-11°, 225°].  Negative values
        correspond to the elbow-up branch and large positive values to the
        elbow-down branch.  A branch flip is detected when the previous command
        and candidate are on opposite sides of the threshold band AND the
        absolute J4 change exceeds the minimum delta.

        Thresholds match the offline path planner's ``check_elbow_consistency``
        (planner.py:606-617).
        """
        prev_j4 = float(previous_qpos_cmd[self._ELBOW_JOINT_INDEX])
        cand_j4 = float(candidate_qpos[self._ELBOW_JOINT_INDEX])
        delta_j4 = abs(cand_j4 - prev_j4)

        # One side on the elbow-up branch (negative), the other on elbow-down (positive).
        if prev_j4 < self._ELBOW_FLIP_NEG_THRESH_RAD and cand_j4 > self._ELBOW_FLIP_POS_THRESH_RAD:
            return bool(delta_j4 > self._ELBOW_FLIP_MIN_DELTA_RAD)
        if cand_j4 < self._ELBOW_FLIP_NEG_THRESH_RAD and prev_j4 > self._ELBOW_FLIP_POS_THRESH_RAD:
            return bool(delta_j4 > self._ELBOW_FLIP_MIN_DELTA_RAD)
        return False

    # Structured IK diagnostics (ref: ssik explain=True pattern)

    @staticmethod
    def _build_ik_diagnostic(
        diff_ik_failed: bool,
        diff_ik_reason: str,
        position_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Build structured IK failure diagnostic categorising why each tier failed.

        Categories map to actionable root causes:
          - ``singular``: near-singularity, damping exhausted → adjust target or
            reduce pose step
          - ``pose_error``: converged but FK residual too large → target may be
            out of workspace
          - ``self_collision``: solution would cause arm self-collision → adjust
            target or add collision margin
          - ``unreachable``: MPlib IK returned no solution → target truly out of
            reachable workspace
          - ``filtered``: MPlib found solutions but all were filtered (limits /
            delta / pose_error / collision) → relax filter thresholds or adjust seed
        """
        diagnostic: dict[str, Any] = {
            "diff_ik": {},
            "position_ik": {},
            "classification": "unknown",
            "summary": "",
        }

        # ── Categorise diff IK failure ──
        if diff_ik_failed:
            reason_lower = diff_ik_reason.lower()
            if "singular" in reason_lower:
                category = "singular"
            elif "self-collision" in reason_lower:
                category = "self_collision"
            elif "pose_error" in reason_lower:
                category = "pose_error"
            else:
                category = "other"
            diagnostic["diff_ik"] = {
                "failed": True,
                "category": category,
                "detail": diff_ik_reason,
            }
        else:
            diagnostic["diff_ik"] = {"failed": False}

        # ── Categorise position IK failure ──
        if position_report:
            pos_diag: dict[str, Any] = {"failed": True}
            failure_reason = str(position_report.get("failure_reason", ""))
            pos_diag["detail"] = failure_reason

            # Classify based on the failure reason string
            if "all failed" in failure_reason:
                # Parse attempt strings to categorise individual failures
                attempts_str = str(position_report.get("attempts", failure_reason))
                if "mplib_failed" in attempts_str and "ok" not in attempts_str:
                    pos_diag["category"] = "unreachable"
                elif "ok" in attempts_str:
                    pos_diag["category"] = "filtered"
                else:
                    pos_diag["category"] = "unreachable"
            elif "mplib_failed" in failure_reason:
                pos_diag["category"] = "unreachable"
            elif "pose_err" in failure_reason:
                pos_diag["category"] = "pose_error"
            elif "jump" in failure_reason:
                pos_diag["category"] = "delta"
            else:
                pos_diag["category"] = "other"
            diagnostic["position_ik"] = pos_diag
        else:
            diagnostic["position_ik"] = {"failed": False, "disabled": True}

        # ── Overall classification ──
        diff_cat = diagnostic["diff_ik"].get("category", "")
        pos_cat = diagnostic["position_ik"].get("category", "")

        if diff_ik_failed and not diagnostic["position_ik"].get("disabled"):
            if pos_cat == "unreachable":
                diagnostic["classification"] = "unreachable"
            elif pos_cat == "filtered":
                diagnostic["classification"] = "all_filtered"
            else:
                diagnostic["classification"] = "all_methods_exhausted"
        elif diff_ik_failed and diagnostic["position_ik"].get("disabled"):
            diagnostic["classification"] = f"diff_ik_failed:{diff_cat}"
        else:
            diagnostic["classification"] = "held"

        # ── Human-readable summary ──
        parts: list[str] = []
        if diff_ik_failed:
            parts.append(f"Diff IK [{diagnostic['diff_ik'].get('category', '?')}]: {diff_ik_reason}")
        if position_report and not position_report.get("disabled"):
            parts.append(f"Position IK [{pos_cat}]: {position_report.get('failure_reason', '?')}")
        diagnostic["summary"] = " | ".join(parts) if parts else "All IK strategies failed"

        return diagnostic

    # Command assembly

    def _check_teleop_collision_gate(
        self,
        qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> tuple[str | None, dict[str, Any]]:
        """Self + env collision gate for teleop IK result validation.

        Returns ``(reason, extra_report)`` — ``(None, {})`` when all clear.
        Env collision uses the full two-tier check for accuracy (Tier1 Z-min
        pre-filter + Tier2 FCL mesh-mesh).
        """
        if not profile.check_self_collision and not profile.check_env_collision:
            return None, {}

        # Self: fast bool gate (stop_at_first=True), detail only on hit.
        if profile.check_self_collision:
            if self.ik_mgr.has_self_collision(qpos_cmd):
                info = self.ik_mgr.check_self_collision(qpos_cmd)
                if info:
                    return (
                        f"IK result in self-collision ({info.summary}), holding.",
                        {"collision": info.to_dict()},
                    )

        # Env: Tier2 FCL (Tier1 pre-filter → Tier2 mesh only when near table)
        if profile.check_env_collision and self.ik_mgr.has_env_collision(qpos_cmd):
            return (
                "IK result in environment collision (table/obstacle), holding.",
                {"env_collision": True},
            )
        return None, {}

    def command_from_target_qpos(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        target_qpos: np.ndarray,
        profile: TeleopProfile,
        report: dict[str, Any],
        method: str,
        jacobian: np.ndarray | None = None,
    ) -> IKResult:
        """Canonicalize IK result and compute tracking error.

        Speed limiting is NOT done here — it is handled by ArmInnerLoop
        (per-step joint delta clamp + Mode 6 firmware trajectory planning).

        Args:
            jacobian: Optional pre-computed 6×7 EEF Jacobian at ``target_qpos``.
                When provided (from diff IK's final iteration), avoids ~80 μs
                FK+Jacobian recomputation in the nullspace optimisation path (P5).
        """
        qpos_cmd = self.ik_mgr.canonicalize_qpos(target_qpos, previous_qpos_cmd)

        # Null-space joint-limit repulsion (post-IK, zero EEF error by construction).
        # Adjusts the redundant DOF to push joints away from limits without
        # altering the end-effector pose.  Runs before collision check so the
        # safety gate covers the adjusted result.
        if profile.enable_nullspace_optimization:
            try:
                if jacobian is None:
                    jacobian, _eef_world = self.kin.compute_eef_jacobian_and_pose_world(qpos_cmd)
                from .nullspace import apply_nullspace_optimization

                qpos_cmd = apply_nullspace_optimization(
                    qpos_cmd,
                    jacobian,
                    self.ik_mgr.joint_limits,
                    step_size_rad=np.deg2rad(profile.nullspace_step_size_deg),
                    margin_deg=profile.nullspace_joint_limit_margin_deg,
                )
            except Exception:
                # Null-space failure is non-critical — skip and use raw IK result.
                pass

        # ── Collision safety gates ──
        # Self: FCL (stop_at_first=True for fast gate, detail on hit).
        # Env: full two-tier (Tier1 Z-min + Tier2 FCL mesh-mesh).
        collision_reason, collision_extra = self._check_teleop_collision_gate(
            qpos_cmd,
            profile,
        )
        if collision_reason is not None:
            # Direction-aware recovery for env collision: when the target EEF
            # is moving upward (away from the table), allow the command through
            # to prevent permanent stuck state at low heights where all IK
            # solutions trigger the collision gate.
            if "environment collision" in collision_reason:
                current_eef = self.kin.compute_eef_pose_world(current_qpos)
                if target_eef_pose_world.p[2] > current_eef.p[2] + 0.001:
                    # Moving up — allow recovery, skip collision gate.
                    logger.warning(
                        "Allowing upward recovery through env collision gate " "(target_z=%.3f, current_z=%.3f)",
                        target_eef_pose_world.p[2],
                        current_eef.p[2],
                    )
                else:
                    qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
                    return IKResult(
                        success=False,
                        qpos=previous_qpos_cmd.copy(),
                        reason=collision_reason,
                        report={
                            **report,
                            "teleop_ik_method": method,
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
                        "teleop_ik_method": method,
                        "held": True,
                        "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(qpos_delta)))),
                        **collision_extra,
                    },
                    held=True,
                )

        qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
        cmd_pos_error, cmd_rot_error = self.kin.compute_world_pose_error(target_eef_pose_world, qpos_cmd)
        result_report = {
            **report,
            "teleop_ik_method": method,
            "cmd_tracking_error_pos_m": cmd_pos_error,
            "cmd_tracking_error_rot_rad": cmd_rot_error,
            "qpos_distance_to_current": float(np.linalg.norm(qpos_delta)),
            "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(qpos_delta)))),
        }
        return IKResult(success=True, qpos=qpos_cmd, report=result_report)

    # Differential IK helpers

    def _solve_damped_least_squares(
        self,
        jacobian: np.ndarray,
        error: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        """Damped least-squares: dq = J^T (J J^T + λ² I)^{-1} error.

        Uses pre-allocated ``self._JJt_buf`` (6×6) to eliminate the ~5 numpy
        array allocations per call — the dominant source of IK slowdown
        (was 0.8-8 ms/iter, now ~0.02 ms/iter).
        """
        np.matmul(jacobian, jacobian.T, out=self._JJt_buf)
        self._JJt_buf.flat[::7] += damping * damping  # in-place diag add λ²
        return jacobian.T @ np.linalg.solve(self._JJt_buf, error)

    # Differential IK fallback

    def solve_differential_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> IKResult:
        """Iterative damped least-squares differential IK.

        Ref: BunnyVisionPro xarm7_ability.py:136-159 compute_ik()
        - Seeded from previous_qpos_cmd: the previous command already reached the
          previous target, so the new target is only ~20ms of hand motion away —
          converges in a few iterations instead of iterating from the (lagging)
          hardware pose.  Also keeps the command stream branch-continuous.
        - Up to max_iterations DLS steps per solve
        - Fixed damping λ²=1e-5 (no adaptive)
        - Early-exit at 50% of the teleop acceptance thresholds
          (max_pose_error_pos_m / max_pose_error_rot_rad) — iterating further is
          wasted work the acceptance gate in solve() cannot distinguish
        - Step size 0.05 per iteration
        """
        qpos = previous_qpos_cmd.copy()
        damping = float(profile.differential_ik_damping)
        max_iter = profile.differential_ik_max_iterations
        step = profile.differential_ik_gain

        # Adaptive damping (optional, disabled by default — aligned with BVP).
        if profile.adaptive_damping:
            _jacobian, _ = self.kin.compute_eef_jacobian_and_pose_world(current_qpos)
            mu = self.kin.manipulability_from_jacobian(_jacobian)
            threshold = profile.manipulability_threshold
            if mu > threshold:
                damping = profile.differential_ik_min_damping
            elif mu < 1e-6:
                damping = profile.differential_ik_max_damping
            else:
                ratio = mu / threshold
                damping = profile.differential_ik_min_damping + (
                    profile.differential_ik_max_damping - profile.differential_ik_min_damping
                ) * (1.0 - ratio)

        iterations = 0
        converged = False
        last_jacobian: np.ndarray | None = None  # P5: reuse in nullspace path
        _tik_start = time.perf_counter()  # F3: wall-clock budget for this diff-IK call
        # Per-iteration phase timing accumulators (logged on >10ms total for root-cause)
        _t_fk_acc = 0.0
        _t_err_acc = 0.0
        _t_dls_acc = 0.0
        for iterations in range(max_iter):
            # F3: abort diff IK if we've blown the time budget.  Skip the check
            # on iteration 0 so at least one FK+Jacobian (needed for nullspace)
            # always completes — the first iteration is always cheap (~0.3 ms).
            if iterations > 0 and time.perf_counter() - _tik_start > self._IK_TIMEOUT_S:
                logger.warning(
                    "Diff IK timeout: %d iterations in %.1fms, aborting",
                    iterations,
                    (time.perf_counter() - _tik_start) * 1000,
                )
                break
            _t_fk = time.perf_counter()
            jacobian, current_pose = self.kin.compute_eef_jacobian_and_pose_world(qpos)
            _dt_fk = time.perf_counter() - _t_fk
            _t_fk_acc += _dt_fk
            if _dt_fk > 0.002:  # >2ms — abnormal (baseline ~0.3ms); signals GC/thermal/GIL
                logger.warning(
                    "Slow FK+J: %.1fms (iter %d/%d, total=%.1fms)",
                    _dt_fk * 1000,
                    iterations + 1,
                    max_iter,
                    (time.perf_counter() - _tik_start) * 1000,
                )
            last_jacobian = jacobian  # cache for nullspace FK reuse (P5)

            # 6D error in world frame (no step limit during internal iterations).
            _t_err = time.perf_counter()
            error_world = pose_error_vector(
                target=target_eef_pose_world,
                actual=current_pose,
                max_pos_step=float("inf"),
                max_rot_step=float("inf"),
            )

            # Convergence: within 50% of the teleop acceptance thresholds.
            # The 50% margin keeps the post-hoc pose check in solve() (which
            # re-evaluates after canonicalization + nullspace adjustment) from
            # flapping a boundary result into the expensive mplib fallback.
            if (
                np.linalg.norm(error_world[:3]) <= 0.5 * profile.max_pose_error_pos_m
                and np.linalg.norm(error_world[3:]) <= 0.5 * profile.max_pose_error_rot_rad
            ):
                converged = True
                _t_err_acc += time.perf_counter() - _t_err
                break
            _t_err_acc += time.perf_counter() - _t_err

            # DLS solve: dq = J^T (J·J^T + λ²I)^(-1) · error · step
            # J has been rotated to world frame by compute_eef_jacobian_and_pose_world,
            # so J and error are now in consistent (world) frames.
            # Ref: BunnyVisionPro xarm7_ability.py:136-159 — consistent frame J+error.
            _t_dls = time.perf_counter()
            try:
                dq = self._solve_damped_least_squares(jacobian, error_world, damping)
            except np.linalg.LinAlgError:
                _t_dls_acc += time.perf_counter() - _t_dls
                if iterations == 0:
                    return IKResult(
                        success=False,
                        qpos=None,
                        reason="Differential IK: singular linear system.",
                        report={"differential_ik_status": "linear_solve_failed"},
                    )
                break  # use last good qpos from previous iteration
            _t_dls_acc += time.perf_counter() - _t_dls

            qpos = qpos + dq * step

        # Per-joint canonicalization: wrap to [-π, π] and prefer the branch
        # closest to the previous command. Joint-level delta safety is enforced
        # by the inner loop (per-step delta clamp + Mode 6 firmware limits).
        _t_post = time.perf_counter()
        raw_target_qpos = self.ik_mgr.canonicalize_qpos(qpos, previous_qpos_cmd)

        diff_report = {
            "differential_ik_status": "success",
            "fallback_method": "differential_ik",
            "iterations": iterations + 1,
            "converged": converged,
        }
        result = self.command_from_target_qpos(
            target_eef_pose_world=target_eef_pose_world,
            current_qpos=current_qpos,
            previous_qpos_cmd=previous_qpos_cmd,
            target_qpos=raw_target_qpos,
            profile=profile,
            report=diff_report,
            method="differential_ik",
            jacobian=last_jacobian,
        )
        _dt_post = time.perf_counter() - _t_post

        # Per-iteration phase breakdown when diff IK is slow (>10ms).
        # Complements the top-level "IK timing" log in _maybe_log_ik_timing
        # by showing where time went inside the iteration loop.
        # Threshold is intentionally lower than _IK_TIMEOUT_S (20ms) to
        # capture the per-phase breakdown before the solver hits the budget.
        _t_diff_total = time.perf_counter() - _tik_start
        if _t_diff_total > 0.010:
            n = max(iterations + 1, 1)
            logger.warning(
                "IK detail: fk=%.1f err=%.1f dls=%.1f post=%.1fms "
                "| %d iters | total=%.1fms (fk/iter=%.2f dls/iter=%.2f)",
                _t_fk_acc * 1000,
                _t_err_acc * 1000,
                _t_dls_acc * 1000,
                _dt_post * 1000,
                n,
                _t_diff_total * 1000,
                _t_fk_acc / n * 1000,
                _t_dls_acc / n * 1000,
            )
            # Trigger background py-spy profiling when DLS/iter is abnormally
            # slow (opt-in: DEXMANI_IK_PROFILE=1, threshold=3ms/iter).
            _ik_profiler.maybe_profile(
                _t_dls_acc / n * 1000,
                _t_fk_acc / n * 1000,
            )
        return result
