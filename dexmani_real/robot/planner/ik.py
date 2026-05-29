from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .planner_types import IKResult, Pose, TeleopProfile
    from .pose_utils import compute_pose_error, ensure_qpos, pose_error_vector
except ImportError:
    from planner_types import IKResult, Pose, TeleopProfile
    from pose_utils import compute_pose_error, ensure_qpos, pose_error_vector


class TeleopIKSolver:
    """Teleoperation IK as local tracking.

    Position IK is used when it returns a continuous branch. Differential IK is
    used as a local fallback when discrete IK fails or only returns far branches.
    """

    def __init__(self, planner: Any) -> None:
        self.planner = planner

    def solve(self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> IKResult:
        profile = self.planner.teleop_profile
        current_qpos = ensure_qpos(current_qpos, self.planner.dof, "current_qpos")
        previous_qpos_cmd = ensure_qpos(previous_qpos_cmd, self.planner.dof, "previous_qpos_cmd")

        if profile.use_position_ik:
            qpos, position_report = self.solve_position_ik(target_eef_pose_world, current_qpos, previous_qpos_cmd, profile)
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
        else:
            position_report = {
                "teleop_ik_method": "position_ik_disabled",
                "failure_reason": "Position IK disabled by TeleopProfile.",
            }

        if profile.use_differential_ik_fallback:
            differential_result = self.solve_differential_ik(target_eef_pose_world, current_qpos, previous_qpos_cmd, profile)
            if differential_result.success:
                differential_result.report["position_ik_report"] = self.compact_position_report(position_report)
                return differential_result

            report = {**position_report, "differential_ik_report": differential_result.report}
            reason = (
                str(position_report.get("failure_reason", "Position IK did not produce a continuous candidate."))
                + "\nDifferential IK fallback failed."
            )
            return self.hold_or_fail(previous_qpos_cmd, reason, report, profile)

        reason = str(position_report.get("failure_reason", "Position IK did not produce a continuous candidate."))
        return self.hold_or_fail(previous_qpos_cmd, reason, position_report, profile)

    def solve_position_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        target_pose_base = self.planner.world_to_base_pose(target_eef_pose_world)
        seeds = self.generate_seeds(current_qpos, previous_qpos_cmd)
        jump_limit = np.deg2rad(self.planner.profile_array(profile.max_ik_jump_deg, "max_ik_jump_deg"))

        accepted: list[tuple[np.ndarray, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        reject_counts: dict[str, int] = {}

        for seed_index, (seed_name, seed, n_init_qpos) in enumerate(seeds):
            status, raw_qpos = self.planner.call_mplib_ik(target_pose_base, seed, n_init_qpos=n_init_qpos, return_closest=True)
            attempt = {"seed_index": seed_index, "seed_name": seed_name, "n_init_qpos": n_init_qpos, "status": status}
            attempts.append(attempt)

            if status != "Success" or raw_qpos is None:
                reject_counts["mplib_ik_failed"] = reject_counts.get("mplib_ik_failed", 0) + 1
                continue

            raw_qpos = np.asarray(raw_qpos, dtype=np.float64)
            qpos = self.planner.nearest_equivalent_qpos(raw_qpos, previous_qpos_cmd)
            qpos = self.planner.wrap_qpos_to_limits(qpos, previous_qpos_cmd, self.planner.joint_limits)
            raw_pose_error_pos, raw_pose_error_rot = self.planner.compute_world_pose_error(target_eef_pose_world, qpos)
            delta = self.planner.compute_qpos_delta(qpos, previous_qpos_cmd)
            max_raw_delta = float(np.max(np.abs(delta)))

            candidate_report = {
                "teleop_ik_seed_index": seed_index,
                "teleop_ik_seed_name": seed_name,
                "teleop_ik_n_init_qpos": n_init_qpos,
                "raw_pose_error_pos_m": raw_pose_error_pos,
                "raw_pose_error_rot_rad": raw_pose_error_rot,
                "max_raw_delta_deg": float(np.rad2deg(max_raw_delta)),
                "qpos_delta_norm": float(np.linalg.norm(delta)),
                "teleop_ik_fallback_used": seed_index > 0,
            }

            if raw_pose_error_pos > profile.max_pose_error_pos_m or raw_pose_error_rot > profile.max_pose_error_rot_rad:
                candidate_report["reject_reason"] = "pose_error"
                rejected.append((candidate_report | {"qpos": qpos}))
                reject_counts["pose_error"] = reject_counts.get("pose_error", 0) + 1
                continue

            if np.any(np.abs(delta) > jump_limit):
                candidate_report["reject_reason"] = "extreme_branch_jump"
                rejected.append(candidate_report | {"qpos": qpos})
                reject_counts["extreme_branch_jump"] = reject_counts.get("extreme_branch_jump", 0) + 1
                continue

            score = self.score_candidate(delta, raw_pose_error_pos, raw_pose_error_rot, seed_index)
            candidate_report["teleop_ik_score"] = score
            accepted.append((qpos, candidate_report))

        if accepted:
            accepted.sort(key=lambda item: item[1]["teleop_ik_score"])
            qpos, best_report = accepted[0]
            report = self.build_position_report(
                attempts=attempts,
                accepted_count=len(accepted),
                rejected_count=len(rejected),
                reject_counts=reject_counts,
                best_report=best_report,
                profile=profile,
            )
            return qpos, report

        best_rejected = self.select_best_rejected(rejected)
        reason = "MPlib position IK failed for all local seeds."
        if best_rejected is not None and best_rejected.get("reject_reason") == "extreme_branch_jump":
            reason = "No continuous position IK candidate; best solution is a far branch."
        elif best_rejected is not None and best_rejected.get("reject_reason") == "pose_error":
            reason = "Position IK converged, but pose error exceeded threshold."

        report = self.build_position_report(
            attempts=attempts,
            accepted_count=0,
            rejected_count=len(rejected),
            reject_counts=reject_counts,
            best_report=best_rejected,
            profile=profile,
        )
        report["failure_reason"] = reason
        return None, report

    def generate_seeds(self, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
        seeds: list[tuple[str, np.ndarray, int]] = [
            ("previous_qpos_cmd", previous_qpos_cmd.copy(), 1),
            ("current_qpos", current_qpos.copy(), 1),
        ]
        for radius_deg in (3.0, 8.0):
            radius = np.deg2rad(radius_deg)
            for joint_index in range(self.planner.dof):
                for sign, sign_name in ((1.0, "+"), (-1.0, "-")):
                    seed = previous_qpos_cmd.copy()
                    seed[joint_index] += sign * radius
                    seeds.append((f"previous{sign_name}{radius_deg:g}deg_j{joint_index + 1}", seed, 1))
        seeds.append(("previous_qpos_cmd_restart", previous_qpos_cmd.copy(), 4))
        seeds.append(("current_qpos_restart", current_qpos.copy(), 4))
        return seeds

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
        max_step = np.deg2rad(self.planner.profile_array(profile.max_qpos_cmd_speed_deg, "max_qpos_cmd_speed_deg")) * profile.teleop_dt
        raw_delta = self.planner.compute_qpos_delta(target_qpos, previous_qpos_cmd)
        clipped_delta = np.clip(raw_delta, -max_step, max_step)
        qpos_cmd = previous_qpos_cmd + clipped_delta
        qpos_cmd = self.planner.nearest_equivalent_qpos(qpos_cmd, previous_qpos_cmd)
        clipped = bool(np.any(np.abs(clipped_delta - raw_delta) > 1e-12))

        cmd_pos_error, cmd_rot_error = self.planner.compute_world_pose_error(target_eef_pose_world, qpos_cmd)
        result_report = {
            **report,
            "teleop_ik_method": method,
            "clipped": clipped,
            "cmd_tracking_error_pos_m": cmd_pos_error,
            "cmd_tracking_error_rot_rad": cmd_rot_error,
            "qpos_distance_to_current": float(np.linalg.norm(qpos_cmd - current_qpos)),
            "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(clipped_delta)))),
        }
        return IKResult(success=True, qpos=qpos_cmd, report=result_report)

    def solve_differential_ik(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        previous_qpos_cmd: np.ndarray,
        profile: TeleopProfile,
    ) -> IKResult:
        current_pose = self.planner.compute_eef_pose_world(current_qpos)
        error = pose_error_vector(
            target=target_eef_pose_world,
            actual=current_pose,
            max_pos_step=profile.differential_ik_max_pos_step_m,
            max_rot_step=profile.differential_ik_max_rot_step_rad,
        )
        error = profile.differential_ik_gain * error

        jacobian = self.planner.compute_eef_jacobian(current_qpos)
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

        max_step = np.deg2rad(self.planner.profile_array(profile.max_qpos_cmd_speed_deg, "max_qpos_cmd_speed_deg")) * profile.teleop_dt
        clipped_dq = np.clip(dq, -max_step, max_step)
        qpos_cmd = previous_qpos_cmd + clipped_dq
        qpos_cmd = self.planner.nearest_equivalent_qpos(qpos_cmd, previous_qpos_cmd)

        cmd_pos_error, cmd_rot_error = self.planner.compute_world_pose_error(target_eef_pose_world, qpos_cmd)
        report = {
            "teleop_ik_method": "differential_ik",
            "fallback_method": "differential_ik",
            "differential_ik_status": "success",
            "cmd_tracking_error_pos_m": cmd_pos_error,
            "cmd_tracking_error_rot_rad": cmd_rot_error,
            "max_qpos_cmd_delta_deg": float(np.rad2deg(np.max(np.abs(clipped_dq)))),
            "clipped": bool(np.any(np.abs(clipped_dq - dq) > 1e-12)),
        }
        return IKResult(success=True, qpos=qpos_cmd, report=report)

    def hold_or_fail(self, previous_qpos_cmd: np.ndarray, reason: str, report: dict[str, Any], profile: TeleopProfile) -> IKResult:
        if profile.hold_on_failure:
            return IKResult(success=False, qpos=previous_qpos_cmd.copy(), reason=reason, report={**report, "held": True})
        return IKResult(success=False, qpos=None, reason=reason, report=report)

    def score_candidate(self, delta: np.ndarray, pos_error: float, rot_error: float, seed_index: int) -> float:
        return float(np.linalg.norm(delta) + np.max(np.abs(delta)) + 0.2 * pos_error + 0.05 * rot_error + 0.01 * seed_index)

    def select_best_rejected(self, rejected: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rejected:
            return None
        return min(rejected, key=lambda item: float(item.get("qpos_delta_norm", float("inf"))))

    def build_position_report(
        self,
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
            for key, value in best_report.items():
                if key != "qpos":
                    report[key] = value
        if profile.debug:
            report["teleop_ik_attempts"] = attempts
        else:
            report["teleop_ik_attempt_count"] = len(attempts)
        return report

    def compact_position_report(self, report: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "failure_reason",
            "teleop_ik_success_count",
            "teleop_ik_rejected_success_count",
            "teleop_ik_reject_counts",
            "max_raw_delta_deg",
        )
        return {key: report[key] for key in keys if key in report}
