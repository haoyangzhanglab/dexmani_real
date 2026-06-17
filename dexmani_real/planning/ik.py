from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from .ik_candidates import IKCandidateManager
    from .kinematics import XArm7Kinematics

from .types import IKResult, Pose, TeleopProfile
from .pose_utils import compute_pose_error, ensure_qpos, pose_error_vector, wxyz_to_xyzw

logger = logging.getLogger(__name__)


class TeleopIKSolver:
    """Teleoperation IK as local tracking.

    Position IK is used when it returns a continuous branch. Differential IK is
    used as a local fallback when discrete IK fails or only returns far branches.
    """

    def __init__(self, kin: XArm7Kinematics, ik_mgr: IKCandidateManager, teleop_profile: TeleopProfile) -> None:
        self.kin = kin
        self.ik_mgr = ik_mgr
        self.profile = teleop_profile

    def solve(self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> IKResult:
        profile = self.profile
        current_qpos = ensure_qpos(current_qpos, self.kin.dof, "current_qpos")
        previous_qpos_cmd = ensure_qpos(previous_qpos_cmd, self.kin.dof, "previous_qpos_cmd")

        current_pose = self.kin.compute_eef_pose_world(current_qpos)
        pos_err, rot_err = compute_pose_error(target_eef_pose_world, current_pose)
        tracking_delta = pos_err + 0.1 * rot_err

        position_report: dict[str, Any] = {}
        if profile.use_position_ik:
            # ~10cm pos error → n_init=15 (max); 0.1 weight maps 1rad rot ≈ 0.1m pos
            n_init = max(2, min(15, int(tracking_delta * 150)))
            qpos, position_report = self.solve_position_ik(
                target_eef_pose_world,
                current_qpos,
                previous_qpos_cmd,
                profile,
                n_init,
                tracking_delta,
            )
            if qpos is not None:
                result = self.command_from_target_qpos(
                    target_eef_pose_world=target_eef_pose_world,
                    current_qpos=current_qpos,
                    previous_qpos_cmd=previous_qpos_cmd,
                    target_qpos=qpos,
                    profile=profile,
                    report=position_report,
                    method="position_ik",
                )
                if profile.check_self_collision and self.ik_mgr.has_self_collision(result.qpos):
                    return IKResult(
                        success=False, qpos=previous_qpos_cmd.copy(),
                        reason="Self-collision detected in teleop IK result",
                        report={**result.report, "held": True, "self_collision": True},
                        held=True,
                    )
                return result
        else:
            position_report = {
                "teleop_ik_method": "position_ik_disabled",
                "failure_reason": "Position IK disabled by TeleopProfile.",
            }

        reason = str(position_report.get("failure_reason", "Position IK did not produce a continuous candidate."))
        report: dict[str, Any] = position_report

        if profile.use_differential_ik_fallback:
            diff_result = self.solve_differential_ik(
                target_eef_pose_world, current_qpos, previous_qpos_cmd, profile, current_pose=current_pose
            )
            if diff_result.success:
                diff_result.report["position_ik_report"] = {
                    k: position_report[k]
                    for k in (
                        "failure_reason",
                        "teleop_ik_success_count",
                        "teleop_ik_rejected_success_count",
                        "teleop_ik_reject_counts",
                        "max_raw_delta_deg",
                    )
                    if k in position_report
                }
                if profile.check_self_collision and self.ik_mgr.has_self_collision(diff_result.qpos):
                    return IKResult(
                        success=False, qpos=previous_qpos_cmd.copy(),
                        reason="Self-collision detected in teleop IK result (diff IK)",
                        report={**diff_result.report, "held": True, "self_collision": True},
                        held=True,
                    )
                return diff_result
            reason = reason + "\nDifferential IK fallback failed."
            report = {**position_report, "differential_ik_report": diff_result.report}

        if profile.hold_on_failure:
            return IKResult(
                success=False, qpos=previous_qpos_cmd.copy(), reason=reason, report={**report, "held": True}, held=True
            )
        return IKResult(success=False, qpos=None, reason=reason, report=report)

    def solve_position_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
        n_init: int,
        tracking_delta: float,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        target_pose_base = self.kin.world_to_base_pose(target_eef_pose_world)
        jump_limit = np.deg2rad(self.ik_mgr.profile_array(profile.max_ik_jump_deg, "max_ik_jump_deg"))

        # Greedy: return first valid candidate (prefers prev_cmd branch for teleop stability).
        # Offline collect_ik_candidates sorts all candidates by score instead.
        seed_specs: list[tuple[str, np.ndarray, int]] = [
            ("prev_cmd", previous_qpos_cmd.copy(), n_init),
            ("current_qpos", current_qpos.copy(), max(2, n_init // 2)),
        ]
        if tracking_delta > 0.10:
            seed_specs.append(("prev_cmd_explore", previous_qpos_cmd.copy(), min(n_init * 2, 12)))

        attempts: list[dict[str, Any]] = []
        reject_counts: dict[str, int] = {}

        for seed_index, (seed_name, seed, n_init_seed) in enumerate(seed_specs):
            status, raw_qpos = self.ik_mgr.call_mplib_ik(
                target_pose_base, seed, n_init_qpos=n_init_seed, return_closest=True
            )
            attempt = {
                "seed_index": seed_index,
                "seed_name": seed_name,
                "n_init_qpos": n_init_seed,
                "status": status,
            }
            attempts.append(attempt)

            if not status.lower().startswith("success") or raw_qpos is None:
                reject_counts["mplib_ik_failed"] = reject_counts.get("mplib_ik_failed", 0) + 1
                continue

            raw_qpos = np.asarray(raw_qpos, dtype=np.float64)
            qpos = self.ik_mgr.canonicalize_qpos(raw_qpos, previous_qpos_cmd)
            pos_err, rot_err = self.kin.compute_world_pose_error(target_eef_pose_world, qpos)
            delta = self.ik_mgr.compute_qpos_delta(qpos, previous_qpos_cmd)

            if pos_err > profile.max_pose_error_pos_m or rot_err > profile.max_pose_error_rot_rad:
                reject_counts["pose_error"] = reject_counts.get("pose_error", 0) + 1
                continue

            if np.any(np.abs(delta) > jump_limit):
                reject_counts["extreme_branch_jump"] = reject_counts.get("extreme_branch_jump", 0) + 1
                continue

            best_report = {
                "teleop_ik_seed_index": seed_index,
                "teleop_ik_seed_name": seed_name,
                "teleop_ik_n_init_qpos": n_init_seed,
                "max_raw_delta_deg": float(np.rad2deg(np.max(np.abs(delta)))),
                "raw_pose_error_pos_m": pos_err,
                "raw_pose_error_rot_rad": rot_err,
                "qpos_delta_norm": float(np.linalg.norm(delta)),
            }
            report = self.build_position_report(
                attempts=attempts,
                accepted_count=1,
                rejected_count=seed_index,
                reject_counts=reject_counts,
                best_report=best_report,
                profile=profile,
            )
            return qpos, report

        report = self.build_position_report(
            attempts=attempts,
            accepted_count=0,
            rejected_count=len(seed_specs),
            reject_counts=reject_counts,
            best_report=None,
            profile=profile,
        )
        report["failure_reason"] = f"No continuous position IK candidate. Reject counts: {reject_counts}"
        return None, report

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
        XArm7._limit_joint_step() in the hardware driver layer, following
        the BunnyVisionPro architecture (single speed limit point).
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

    def solve_differential_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
        current_pose: Pose | None = None,
    ) -> IKResult:
        if current_pose is None:
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

    @staticmethod
    def build_position_report(
        attempts: list[dict[str, Any]],
        accepted_count: int,
        rejected_count: int,
        reject_counts: dict[str, int],
        best_report: dict[str, Any] | None,
        profile: TeleopProfile,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "teleop_ik_method": "position_ik",
            "teleop_ik_success_count": accepted_count,
            "teleop_ik_rejected_success_count": rejected_count,
            "teleop_ik_reject_counts": reject_counts,
        }
        if best_report is not None:
            report.update(best_report)
        key = "teleop_ik_attempts" if profile.debug else "teleop_ik_attempt_count"
        report[key] = attempts if profile.debug else len(attempts)
        return report
