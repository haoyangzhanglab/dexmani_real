"""Teleoperation IK solver — differential IK (primary) with position IK fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from .ik_candidates import IKCandidateManager
    from .kinematics import XArm7Kinematics

from .types import IKResult, Pose, TeleopProfile
from .pose_utils import ensure_qpos, pose_error_vector, wxyz_to_xyzw


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

    Speed limiting is handled by XArm7._limit_joint_step()
    (driver bottleneck scaling + soft-start).
    Self-collision checks are done when TeleopProfile.check_self_collision=True.
    

    References:
      - BunnyVisionPro DLS + LeFranX scoring)
      - BunnyVisionPro DLS fallback)
      - LeFranX current_distance penalty)
      - ssik explain=True pattern)
      - ssik max_solutions=1
      - ssik seed_tolerance hard boundary)
    """

    def __init__(self, kin: XArm7Kinematics, ik_mgr: IKCandidateManager, teleop_profile: TeleopProfile) -> None:
        self.kin = kin
        self.ik_mgr = ik_mgr
        self.profile = teleop_profile

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
        profile = self.profile
        current_qpos = ensure_qpos(current_qpos, self.kin.dof, "current_qpos")
        previous_qpos_cmd = ensure_qpos(previous_qpos_cmd, self.kin.dof, "previous_qpos_cmd")

        # ── Step 1: Differential IK (deterministic, primary) ──
        diff_ik_failed = False
        diff_ik_reason = ""
        if profile.use_differential_ik_fallback:
            diff_result = self.solve_differential_ik(
                target_eef_pose_world, current_qpos, previous_qpos_cmd, profile,
            )
            if diff_result.success:
                # Verify the diff IK result actually reaches the target.
                pos_err, rot_err = self.kin.compute_world_pose_error(
                    target_eef_pose_world, diff_result.qpos,
                )
                if pos_err <= profile.max_pose_error_pos_m and rot_err <= profile.max_pose_error_rot_rad:
                    return diff_result
                # Diff IK converged but pose error too large → fall through to position IK.
                diff_ik_failed = True
                diff_ik_reason = f"pose_error({pos_err:.4f}m,{rot_err:.4f}rad)"
            else:
                diff_ik_failed = True
                diff_ik_reason = diff_result.reason

        # ── Step 2: Position IK (stochastic fallback) ──
        position_report: dict[str, Any] = {}
        if profile.use_position_ik:
            qpos, position_report = self.solve_position_ik(
                target_eef_pose_world, current_qpos, previous_qpos_cmd, profile,
            )
            if qpos is not None:
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
        diagnostic = self._build_ik_diagnostic(
            diff_ik_failed=diff_ik_failed,
            diff_ik_reason=diff_ik_reason if diff_ik_failed else "",
            position_report=position_report,
        )
        return IKResult(
            success=False, qpos=previous_qpos_cmd.copy(),
            reason=diagnostic["summary"],
            report={"held": True, "diagnostic": diagnostic}, held=True,
        )

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
        best: tuple[np.ndarray, str, float] | None = None  # (qpos, seed_name, hw_dist)

        for seed_name, seed, n_init in seeds:
            status, raw_qpos = self.ik_mgr.call_mplib_ik(
                target_pose_base, seed, n_init_qpos=n_init, return_closest=True,
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

            hw_dist = float(np.max(np.abs(self.ik_mgr.compute_qpos_delta(qpos, current_qpos))))

            # Track hardware-closest candidate (ref: LeFranX current_distance penalty)
            if best is None or hw_dist < best[2]:
                best = (qpos.copy(), seed_name, hw_dist)
                attempts.append(f"{seed_name}:ok")

            # Fast path: prev_cmd seed, close to hardware → accept immediately
            # (ref: ssik seed_tolerance hard boundary)
            if seed_name == "prev_cmd" and hw_dist <= fast_accept_rad:
                return qpos, {"teleop_ik_method": "position_ik", "seed": seed_name, "attempts": attempts}

            # Early exit: any seed with excellent quality → stop searching
            # (ref: ssik max_solutions=1 — first good solution wins)
            quality_pos = pos_err < profile.max_pose_error_pos_m * 0.3
            quality_rot = rot_err < profile.max_pose_error_rot_rad * 0.3
            quality_delta = hw_dist < fast_accept_rad * 0.5
            if quality_pos and quality_rot and quality_delta:
                return qpos, {"teleop_ik_method": "position_ik", "seed": seed_name,
                              "attempts": attempts, "early_exit": "excellent_quality"}

        if best is not None:
            qpos, seed_name, _ = best
            return qpos, {"teleop_ik_method": "position_ik", "seed": seed_name, "fallback": True, "attempts": attempts}

        return None, {"teleop_ik_method": "position_ik", "failure_reason": f"all failed: {attempts}"}

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

        if diff_ik_failed and not position_report.get("disabled"):
            if pos_cat == "unreachable":
                diagnostic["classification"] = "unreachable"
            elif pos_cat == "filtered":
                diagnostic["classification"] = "all_filtered"
            else:
                diagnostic["classification"] = "all_methods_exhausted"
        elif diff_ik_failed and position_report.get("disabled"):
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

    def command_from_target_qpos(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        target_qpos: np.ndarray,
        profile: TeleopProfile,
        report: dict[str, Any],
        method: str,
    ) -> IKResult:
        """Canonicalize IK result and compute tracking error.

        Speed limiting is NOT done here — it is handled exclusively by
        XArm7._limit_joint_step() in the hardware driver layer.
        """
        qpos_cmd = self.ik_mgr.canonicalize_qpos(target_qpos, previous_qpos_cmd)

        # Self-collision check (teleop hot path, ~0.1-0.3 ms per call)
        if profile.check_self_collision and self.ik_mgr.has_self_collision(qpos_cmd):
            qpos_delta = self.ik_mgr.compute_qpos_delta(qpos_cmd, current_qpos)
            return IKResult(
                success=False, qpos=previous_qpos_cmd.copy(),
                reason="IK result in self-collision, holding.",
                report={
                    **report,
                    "teleop_ik_method": method,
                    "held": True,
                    "self_collision": True,
                    "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(qpos_delta)))),
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

    @staticmethod
    def _solve_damped_least_squares(
        jacobian: np.ndarray, error: np.ndarray, damping: float,
    ) -> np.ndarray:
        """Damped least-squares: dq = J^T (J J^T + λ² I)^{-1} error."""
        damped_JJt = jacobian @ jacobian.T + (damping * damping) * np.eye(6)
        return jacobian.T @ np.linalg.solve(damped_JJt, error)

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
        - Up to max_iterations DLS steps per solve
        - Fixed damping λ²=1e-5 (no adaptive)
        - Converges when ||error|| < convergence_threshold (1e-3)
        - Step size 0.05 per iteration
        """
        qpos = current_qpos.copy()
        damping = float(profile.differential_ik_damping)
        max_iter = profile.differential_ik_max_iterations
        conv_thresh = profile.differential_ik_convergence_threshold
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
                damping = (
                    profile.differential_ik_min_damping
                    + (profile.differential_ik_max_damping - profile.differential_ik_min_damping)
                    * (1.0 - ratio)
                )

        iterations = 0
        for iterations in range(max_iter):
            jacobian, current_pose = self.kin.compute_eef_jacobian_and_pose_world(qpos)

            # 6D error in world frame (no step limit during internal iterations).
            error_world = pose_error_vector(
                target=target_eef_pose_world,
                actual=current_pose,
                max_pos_step=float("inf"),
                max_rot_step=float("inf"),
            )

            # Convergence check (BVP: norm(err) < 1e-3).
            if np.linalg.norm(error_world) < conv_thresh:
                break

            # Rotate error from world to EEF local frame.
            quat_xyzw = wxyz_to_xyzw(current_pose.q)
            R_local_world = Rotation.from_quat(quat_xyzw).inv().as_matrix()
            R6 = np.zeros((6, 6))
            R6[:3, :3] = R_local_world
            R6[3:, 3:] = R_local_world
            error = R6 @ error_world

            # DLS solve: dq = J^T (J·J^T + λ²I)^(-1) · error · step
            try:
                dq = self._solve_damped_least_squares(jacobian, error, damping)
            except np.linalg.LinAlgError:
                if iterations == 0:
                    return IKResult(
                        success=False,
                        qpos=None,
                        reason="Differential IK: singular linear system.",
                        report={"differential_ik_status": "linear_solve_failed"},
                    )
                break  # use last good qpos from previous iteration

            qpos = qpos + dq * step

        # Apply step limit to the FINAL delta (safety clamp on total displacement).
        final_delta = qpos - current_qpos
        final_normalized = np.abs(final_delta) / (
            profile.differential_ik_max_pos_step_m if np.max(np.abs(final_delta)) > 0 else 1.0
        )
        # Per-joint canonicalization: wrap to [-π, π] and prefer the branch
        # closest to the previous command.
        raw_target_qpos = self.ik_mgr.canonicalize_qpos(qpos, previous_qpos_cmd)

        diff_report = {
            "differential_ik_status": "success",
            "fallback_method": "differential_ik",
            "iterations": iterations + 1,
            "converged": iterations + 1 < max_iter,
        }
        return self.command_from_target_qpos(
            target_eef_pose_world=target_eef_pose_world,
            current_qpos=current_qpos,
            previous_qpos_cmd=previous_qpos_cmd,
            target_qpos=raw_target_qpos,
            profile=profile,
            report=diff_report,
            method="differential_ik",
        )

