from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np

from .ik import TeleopIKSolver
from .ik_candidates import IKCandidateManager
from .kinematics import XArm7Kinematics
from .planner_types import IKResult, PathResult, PlanningProfile, Pose, TeleopProfile, XArm7PlannerConfig
from .pose_utils import compute_pose_error, ensure_qpos


class XArm7MotionPlanner:
    """Arm-only xArm7 planner.

    The backend is MPlib. This wrapper keeps three task-specific policies explicit:
    local IK branch selection, full-turn joint canonicalization, and path quality
    validation for data collection.

    Internally delegates to XArm7Kinematics (FK/Jacobian/pose transforms) and
    IKCandidateManager (IK candidate generation/filtering/scoring/canonicalization).
    """

    def __init__(
        self,
        config: XArm7PlannerConfig,
        planning_profile: PlanningProfile | None = None,
        teleop_profile: TeleopProfile | None = None,
    ) -> None:
        import os

        import mplib as mp

        self.mp = mp
        self.config = config
        self.planning_profile = planning_profile or PlanningProfile()
        self.teleop_profile = teleop_profile or TeleopProfile()

        joint_vel_limits = np.deg2rad(np.asarray(config.joint_vel_limits_deg, dtype=np.float64))
        joint_acc_limits = joint_vel_limits * float(config.joint_acc_scale)
        if not os.path.exists(config.srdf_path):
            mp.urdf_utils.generate_srdf(config.urdf_path, config.srdf_path)

        self.mp_planner = self.mp.Planner(
            urdf=str(config.urdf_path),
            srdf=str(config.srdf_path),
            move_group=config.eef_link_name,
            use_convex=config.use_convex,
            joint_vel_limits=joint_vel_limits.tolist(),
            joint_acc_limits=joint_acc_limits.tolist(),
        )
        self.pinocchio_model = self.mp_planner.pinocchio_model

        link_names = list(self.pinocchio_model.get_link_names())
        if config.eef_link_name not in link_names:
            raise ValueError(f"Link {config.eef_link_name!r} not found. Available links: {link_names}")
        eef_link_id = int(link_names.index(config.eef_link_name))

        self._elbow_joint_index = list(self.pinocchio_model.get_joint_names()).index("joint4")
        joint_limits = np.asarray(self.mp_planner.joint_limits, dtype=np.float64)
        dof = int(joint_limits.shape[0])
        equivalent_joint_mask = (joint_limits[:, 1] - joint_limits[:, 0]) >= np.pi

        base_pose_world = config.base_pose_world.copy()

        self.kin = XArm7Kinematics(
            mp_planner=self.mp_planner,
            pinocchio_model=self.pinocchio_model,
            eef_link_id=eef_link_id,
            dof=dof,
            joint_limits=joint_limits,
            equivalent_joint_mask=equivalent_joint_mask,
            base_pose_world=base_pose_world,
            config=config,
            mp=self.mp,
        )
        self.ik_mgr = IKCandidateManager(self.kin)
        self.mp_planner.set_base_pose(self.kin.to_mplib_pose(base_pose_world))

        self.teleop_solver = TeleopIKSolver(self)

        # Convenience aliases (used by teleop_solver and external code)
        self.dof = dof
        self.joint_limits = joint_limits
        self.equivalent_joint_mask = equivalent_joint_mask

    # --- Public API ---

    def set_base_pose(self, base_pose_world: Pose) -> None:
        self.kin.set_base_pose(base_pose_world)

    @contextmanager
    def with_planning_profile(self, profile: PlanningProfile):
        saved = self.planning_profile
        self.planning_profile = profile
        try:
            yield
        finally:
            self.planning_profile = saved

    def solve_ik(self, target_eef_pose_world: Pose, current_qpos: np.ndarray) -> IKResult:
        current_qpos = ensure_qpos(current_qpos, self.dof, "current_qpos")
        candidates, ik_report = self.collect_ik_candidates(target_eef_pose_world, current_qpos, self.planning_profile)
        if not candidates:
            return IKResult(success=False, qpos=None, reason="No valid IK candidate.", report=ik_report)
        qpos, report = candidates[0]
        return IKResult(success=True, qpos=qpos, report={**report, "candidate_summary": ik_report})

    def solve_teleop_ik(
        self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray
    ) -> IKResult:
        return self.teleop_solver.solve(target_eef_pose_world, current_qpos, previous_qpos_cmd)

    def plan_path(self, target_eef_pose_world: Pose, current_qpos: np.ndarray) -> PathResult:
        profile = self.planning_profile
        current_qpos = ensure_qpos(current_qpos, self.dof, "current_qpos")
        current_qpos = self.canonicalize_qpos(
            current_qpos, current_qpos, self.resolve_planning_limits(profile, current_qpos)
        )

        current_pose = self.compute_eef_pose_world(current_qpos)
        pos_error, rot_error = compute_pose_error(target_eef_pose_world, current_pose)
        if pos_error <= profile.max_pose_error_pos_m and rot_error <= profile.max_pose_error_rot_rad:
            return PathResult(
                success=True, qpos_path=current_qpos.reshape(1, -1), source="hold", report={"num_waypoints": 1}
            )

        screw_result = self.try_screw_plan(target_eef_pose_world, current_qpos, profile)
        if screw_result.success:
            screw_result.report["num_planning_attempts"] = 1
            screw_result.report["num_valid_plans"] = 1
            return screw_result

        candidates, ik_report = self.collect_ik_candidates(target_eef_pose_world, current_qpos, profile)
        results: list[PathResult] = [screw_result]
        if candidates:
            results.extend(self.try_multi_rrt_plan(target_eef_pose_world, current_qpos, candidates, profile))
        else:
            results.append(
                PathResult(
                    success=False,
                    qpos_path=None,
                    source="",
                    reason=f"IK failed: 0/{ik_report.get('num_seeds', '?')} seeds produced valid candidates. "
                    f"Reject counts: {ik_report.get('reject_counts', {})}",
                    report={"ik": ik_report},
                )
            )

        valid_results = [result for result in results if result.success and result.qpos_path is not None]
        if not valid_results:
            reason_counts: dict[str, int] = {}
            for result in results:
                r = result.reason or "unknown"
                reason_counts[r] = reason_counts.get(r, 0) + 1
            reason = (
                "; ".join(f"{text} x{count}" for text, count in reason_counts.items())
                or "All planning strategies failed."
            )
            return PathResult(
                success=False,
                qpos_path=None,
                source="",
                reason=reason,
                report={
                    "ik": ik_report,
                    "num_planning_attempts": len(results),
                    "planning_reject_counts": reason_counts,
                },
            )

        valid_results.sort(key=lambda result: result.report.get("path_score", float("inf")))
        best = valid_results[0]
        best.report.setdefault("ik", ik_report)
        best.report["num_planning_attempts"] = len(results)
        best.report["num_valid_plans"] = len(valid_results)
        return best

    # ── Public API: Kinematics delegation ──

    def world_to_base_pose(self, pose_world: Pose) -> Pose:
        return self.kin.world_to_base_pose(pose_world)

    def compute_eef_pose_world(self, qpos: np.ndarray) -> Pose:
        return self.kin.compute_eef_pose_world(qpos)

    def compute_eef_jacobian(self, qpos: np.ndarray) -> np.ndarray:
        return self.kin.compute_eef_jacobian(qpos)

    def compute_manipulability(self, qpos: np.ndarray) -> float:
        return self.kin.compute_manipulability(qpos)

    def compute_world_pose_error(self, target_eef_pose_world: Pose, qpos: np.ndarray) -> tuple[float, float]:
        return self.kin.compute_world_pose_error(target_eef_pose_world, qpos)

    def to_mplib_pose(self, pose: Pose) -> Any:
        return self.kin.to_mplib_pose(pose)

    # ── Public API: IK Candidates delegation ──

    def call_mplib_ik(
        self, target_pose_base: Pose, seed_qpos: np.ndarray, n_init_qpos: int, return_closest: bool
    ) -> tuple[str, Any]:
        return self.ik_mgr.call_mplib_ik(target_pose_base, seed_qpos, n_init_qpos, return_closest)

    def collect_ik_candidates(
        self, target_eef_pose_world: Pose, current_qpos: np.ndarray, profile: PlanningProfile
    ) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], dict[str, Any]]:
        return self.ik_mgr.collect_ik_candidates(target_eef_pose_world, current_qpos, profile)

    def filter_ik_candidate(
        self, qpos: np.ndarray, raw_qpos: np.ndarray, target_eef_pose_world: Pose,
        current_qpos: np.ndarray, profile: PlanningProfile, limits: np.ndarray,
    ) -> tuple[bool, dict[str, Any]]:
        return self.ik_mgr.filter_ik_candidate(qpos, raw_qpos, target_eef_pose_world, current_qpos, profile, limits)

    def resolve_planning_limits(self, profile: PlanningProfile, reference_qpos: np.ndarray | None = None) -> np.ndarray:
        return self.ik_mgr.resolve_planning_limits(profile, reference_qpos)

    def canonicalize_qpos(
        self, qpos: np.ndarray, reference_qpos: np.ndarray, limits: np.ndarray | None = None, limit_tol: float = 1e-5
    ) -> np.ndarray:
        return self.ik_mgr.canonicalize_qpos(qpos, reference_qpos, limits, limit_tol)

    def canonicalize_path_to_planning_limits(
        self, path: np.ndarray, current_qpos: np.ndarray, profile: PlanningProfile
    ) -> np.ndarray:
        return self.ik_mgr.canonicalize_path_to_planning_limits(path, current_qpos, profile)

    def snap_path_to_nearest_equivalent(self, path: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        return self.ik_mgr.snap_path_to_nearest_equivalent(path, reference_qpos)

    def compute_qpos_delta(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        return self.ik_mgr.compute_qpos_delta(qpos, reference_qpos)

    def limit_violation(
        self, qpos: np.ndarray, limits: np.ndarray, limit_tol: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.ik_mgr.limit_violation(qpos, limits, limit_tol)

    def path_limit_violation(
        self, path: np.ndarray, limits: np.ndarray, limit_tol: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.ik_mgr.path_limit_violation(path, limits, limit_tol)

    def has_self_collision(self, qpos: np.ndarray) -> bool:
        return self.ik_mgr.has_self_collision(qpos)

    def check_path_collisions(self, path: np.ndarray) -> dict[str, Any]:
        return self.ik_mgr.check_path_collisions(path)

    def normalized_joint_distance(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> float:
        return self.ik_mgr.normalized_joint_distance(qpos, reference_qpos)

    def joint_limit_penalty(self, qpos: np.ndarray, limits: np.ndarray) -> float:
        return self.ik_mgr.joint_limit_penalty(qpos, limits)

    def profile_array(self, values: tuple[float, ...], name: str) -> np.ndarray:
        return self.ik_mgr.profile_array(values, name)

    # ── Planning strategies (internal) ──

    def try_screw_plan(
        self, target_eef_pose_world: Pose, current_qpos: np.ndarray, profile: PlanningProfile
    ) -> PathResult:
        result = self.mp_planner.plan_screw(
            goal_pose=self.to_mplib_pose(target_eef_pose_world),
            current_qpos=current_qpos,
            time_step=profile.path_dt,
            qpos_step=profile.screw_qpos_step,
            wrt_world=True,
        )
        return self.result_from_mplib(result, target_eef_pose_world, current_qpos, source="screw", profile=profile)

    def try_multi_rrt_plan(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        candidates: list[tuple[np.ndarray, dict[str, Any]]],
        profile: PlanningProfile,
    ) -> list[PathResult]:
        goal_qposes = [qpos for qpos, report in candidates]
        results: list[PathResult] = []
        for rrt_range in profile.rrt_range_options:
            for attempt_index in range(profile.num_rrt_attempts):
                result = self.mp_planner.plan_qpos(
                    goal_qposes=goal_qposes,
                    current_qpos=current_qpos,
                    time_step=profile.path_dt,
                    rrt_range=rrt_range,
                    planning_time=profile.rrt_time_limit,
                    simplify=profile.simplify_path,
                )
                path_result = self.result_from_mplib(
                    result, target_eef_pose_world, current_qpos, source="rrt", profile=profile
                )
                path_result.report["rrt_range"] = rrt_range
                path_result.report["rrt_attempt_index"] = attempt_index
                results.append(path_result)
        return results

    def result_from_mplib(
        self,
        result: dict[str, Any],
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        source: str,
        profile: PlanningProfile,
    ) -> PathResult:
        if not isinstance(result, dict):
            return PathResult(
                success=False,
                qpos_path=None,
                source=source,
                reason=f"MPlib {source} failed: invalid result type",
                report={"mplib_status": "Invalid"},
            )
        status = str(result.get("status", ""))
        if not status.lower().startswith("success"):
            return PathResult(
                success=False,
                qpos_path=None,
                source=source,
                reason=f"MPlib {source} failed: {status}",
                report={"mplib_status": status},
            )

        path = np.asarray(result.get("position", []), dtype=np.float64)
        if len(path) == 0:
            import warnings

            warnings.warn(f"MPlib {source} returned success but empty position; falling back to current_qpos.")
            path = current_qpos.reshape(1, -1)
        path_result = self.validate_path(path, target_eef_pose_world, current_qpos, source=source, profile=profile)
        path_result.report.update(mplib_status=status)
        return path_result

    # ── Path validation (internal) ──

    def shortcut_smooth_path(
        self, path: np.ndarray, current_qpos: np.ndarray, profile: PlanningProfile
    ) -> np.ndarray:
        limits = self.resolve_planning_limits(profile, current_qpos)
        path = np.asarray(path, dtype=np.float64).copy()
        if len(path) <= 2:
            return path

        for _ in range(3):
            changed = False
            idx = 1
            while idx < len(path) - 1:
                prev = path[idx - 1]
                nxt = path[idx + 1]
                mid = 0.5 * (prev + nxt)
                if self._is_shortcut_valid(mid, limits, profile):
                    path = np.delete(path, idx, axis=0)
                    changed = True
                else:
                    idx += 1
            if not changed:
                break
        return path

    def _is_shortcut_valid(self, qpos: np.ndarray, limits: np.ndarray, profile: PlanningProfile) -> bool:
        outside, _ = self.limit_violation(qpos, limits)
        if np.any(outside):
            return False
        if profile.check_self_collision and self.has_self_collision(qpos):
            return False
        return True

    def validate_path(
        self,
        path: np.ndarray,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        source: str,
        profile: PlanningProfile,
    ) -> PathResult:
        try:
            path = self.snap_path_to_nearest_equivalent(path, current_qpos)
            path = self.canonicalize_path_to_planning_limits(path, current_qpos, profile)
            path = self.shortcut_smooth_path(path, current_qpos, profile)
        except ValueError as error:
            return PathResult(success=False, qpos_path=None, source=source, reason=str(error))

        report = self.compute_path_metrics(path, target_eef_pose_world, current_qpos, profile)
        if report["limit_violation"]:
            return PathResult(
                success=False, qpos_path=None, source=source, reason="Path violates planning limits.", report=report
            )
        has_flip, flip_info = self.check_elbow_consistency(path)
        if has_flip:
            report.update(flip_info)
            return PathResult(
                success=False, qpos_path=None, source=source, reason="Elbow branch flip detected.", report=report
            )
        if profile.check_self_collision:
            collision_report = self.check_path_collisions(path)
            report.update(collision_report)
            if collision_report.get("path_self_collision"):
                return PathResult(
                    success=False, qpos_path=None, source=source, reason="Path contains self-collision.", report=report
                )
        if report["start_qpos_error_rad"] > np.deg2rad(profile.max_waypoint_delta_deg) + 1e-12:
            return PathResult(
                success=False,
                qpos_path=None,
                source=source,
                reason="Path start is too far from current_qpos.",
                report=report,
            )
        if report["max_waypoint_delta_rad"] > np.deg2rad(profile.max_waypoint_delta_deg) + 1e-12:
            return PathResult(
                success=False, qpos_path=None, source=source, reason="Path waypoint delta too large.", report=report
            )
        if (
            report["terminal_pos_error_m"] > profile.max_pose_error_pos_m
            or report["terminal_rot_error_rad"] > profile.max_pose_error_rot_rad
        ):
            return PathResult(
                success=False, qpos_path=None, source=source, reason="Terminal pose error too large.", report=report
            )
        report["path_score"] = float(
            report.get("joint_path_length", 0.0)
            + 2.0 * report.get("max_waypoint_delta_rad", 0.0)
            + 3.0 * (1.0 - report.get("eef_efficiency", 1.0))
        )
        return PathResult(success=True, qpos_path=path, source=source, report=report)

    def compute_path_metrics(
        self, path: np.ndarray, target_eef_pose_world: Pose, current_qpos: np.ndarray, profile: PlanningProfile
    ) -> dict[str, Any]:
        diff = np.diff(path, axis=0) if len(path) > 1 else np.zeros((0, self.dof), dtype=np.float64)
        max_step = float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0
        path_length = float(np.sum(np.linalg.norm(diff, axis=1))) if len(diff) > 0 else 0.0
        terminal_pos_error, terminal_rot_error = self.compute_world_pose_error(target_eef_pose_world, path[-1])

        eef_efficiency = 1.0
        if len(path) >= 3:
            eef_positions = np.array(
                [self.compute_eef_pose_world(q).p for q in path], dtype=np.float64
            )
            eef_deltas = np.diff(eef_positions, axis=0)
            eef_path_len = float(np.sum(np.linalg.norm(eef_deltas, axis=1)))
            eef_straight = float(np.linalg.norm(eef_positions[-1] - eef_positions[0]))
            if eef_path_len > 1e-8:
                eef_efficiency = eef_straight / eef_path_len
        limits = self.resolve_planning_limits(profile, current_qpos)
        outside, violation = self.path_limit_violation(path, limits)
        report = {
            "num_waypoints": int(len(path)),
            "joint_path_length": path_length,
            "max_waypoint_delta_rad": max_step,
            "max_waypoint_delta_deg": float(np.rad2deg(max_step)),
            "start_qpos_error_rad": float(np.max(np.abs(self.compute_qpos_delta(path[0], current_qpos)))),
            "terminal_pos_error_m": terminal_pos_error,
            "terminal_rot_error_rad": terminal_rot_error,
            "limit_violation": bool(np.any(outside)),
            "eef_efficiency": eef_efficiency,
        }
        if np.any(outside):
            waypoint_indices, joint_indices = np.where(outside)
            report["limit_violation_waypoint_index"] = int(waypoint_indices[0])
            report["limit_violation_joint_index_1based"] = int(joint_indices[0] + 1)
            report["max_limit_violation_deg"] = float(np.rad2deg(np.max(violation[outside])))
            report["limit_violation_qpos_deg"] = np.rad2deg(path[waypoint_indices[0]].copy())
        return report

    def check_elbow_consistency(self, path: np.ndarray) -> tuple[bool, dict[str, Any]]:
        values = path[:, self._elbow_joint_index]
        v_min, v_max = float(np.min(values)), float(np.max(values))
        span = v_max - v_min
        if v_min < np.deg2rad(-5) and v_max > np.deg2rad(15) and span > np.deg2rad(45):
            return True, {
                "elbow_branch_flip": True,
                "elbow_min_deg": float(np.rad2deg(v_min)),
                "elbow_max_deg": float(np.rad2deg(v_max)),
                "elbow_span_deg": float(np.rad2deg(span)),
            }
        return False, {}

    # ── Elbow consistency check (internal) ──
