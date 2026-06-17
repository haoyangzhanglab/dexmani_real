from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from .ik_candidates import IKCandidateManager
    from .kinematics import XArm7Kinematics

from .types import IKResult, Pose, TeleopProfile
from .pose_utils import ensure_qpos, pose_error_vector, wxyz_to_xyzw

logger = logging.getLogger(__name__)


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
    Self-collision checks are only done during path planning, not teleop.
    """

    def __init__(self, kin: XArm7Kinematics, ik_mgr: IKCandidateManager, teleop_profile: TeleopProfile) -> None:
        self.kin = kin
        self.ik_mgr = ik_mgr
        self.profile = teleop_profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        diff_failed = False
        diff_reason = ""
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
                diff_failed = True
                diff_reason = f"pose_error({pos_err:.4f}m,{rot_err:.4f}rad)"
            else:
                diff_failed = True
                diff_reason = diff_result.reason

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
        parts = []
        if diff_failed:
            parts.append(f"Diff IK: {diff_reason}")
        if position_report:
            parts.append(str(position_report.get("failure_reason", "no position IK candidate")))
        reason = "\n".join(parts) if parts else "All IK strategies failed"

        return IKResult(
            success=False, qpos=previous_qpos_cmd.copy(), reason=reason,
            report={"held": True}, held=True,
        )

    # ------------------------------------------------------------------
    # Position IK — simplified single-seed with fallback
    # ------------------------------------------------------------------

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
        _HW_BRANCH_CHECK_RAD = np.deg2rad(15.0)

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
            if seed_name == "prev_cmd" and hw_dist <= _HW_BRANCH_CHECK_RAD:
                return qpos, {"teleop_ik_method": "position_ik", "seed": seed_name, "attempts": attempts}

        if best is not None:
            qpos, seed_name, _ = best
            return qpos, {"teleop_ik_method": "position_ik", "seed": seed_name, "fallback": True, "attempts": attempts}

        return None, {"teleop_ik_method": "position_ik", "failure_reason": f"all failed: {attempts}"}

    # ------------------------------------------------------------------
    # Command assembly
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Differential IK fallback
    # ------------------------------------------------------------------

    def solve_differential_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> IKResult:
        """Damped least-squares differential IK.

        Uses full tracking gain=1.0 — speed safety is handled downstream
        by XArm7._limit_joint_step() bottleneck scaling.
        """
        current_pose = self.kin.compute_eef_pose_world(current_qpos)
        error_world = pose_error_vector(
            target=target_eef_pose_world,
            actual=current_pose,
            max_pos_step=profile.differential_ik_max_pos_step_m,
            max_rot_step=profile.differential_ik_max_rot_step_rad,
        )
        error_world = profile.differential_ik_gain * error_world

        # Jacobian is in EEF local frame; rotate error from world to local frame.
        quat_xyzw = wxyz_to_xyzw(current_pose.q)
        R_local_world = Rotation.from_quat(quat_xyzw).inv().as_matrix()
        R6 = np.zeros((6, 6))
        R6[:3, :3] = R_local_world
        R6[3:, 3:] = R_local_world
        error = R6 @ error_world

        jacobian = self.kin.compute_eef_jacobian(current_qpos)
        damping = float(profile.differential_ik_damping)
        lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(6)
        try:
            dq = jacobian.T @ np.linalg.solve(lhs, error)
        except np.linalg.LinAlgError:
            return IKResult(
                success=False,
                qpos=None,
                reason="Differential IK fallback failed: singular linear system.",
                report={"differential_ik_status": "linear_solve_failed"},
            )

        raw_target_qpos = current_qpos + dq
        raw_target_qpos = self.ik_mgr.canonicalize_qpos(raw_target_qpos, previous_qpos_cmd)

        diff_report = {
            "differential_ik_status": "success",
            "fallback_method": "differential_ik",
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

