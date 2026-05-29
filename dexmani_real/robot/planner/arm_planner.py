from __future__ import annotations

from typing import Any
import importlib

import numpy as np

try:
    from .ik import TeleopIKSolver
    from .planner_types import IKResult, PathResult, PlanningProfile, Pose, TeleopProfile, XArm7PlannerConfig
    from .pose_utils import compose_pose, compute_pose_error, ensure_qpos, interpolate_qpos_path, invert_pose
except ImportError:
    from ik import TeleopIKSolver
    from planner_types import IKResult, PathResult, PlanningProfile, Pose, TeleopProfile, XArm7PlannerConfig
    from pose_utils import compose_pose, compute_pose_error, ensure_qpos, interpolate_qpos_path, invert_pose


class XArm7MotionPlanner:
    """Arm-only xArm7 planner.

    The backend is MPlib. This wrapper keeps three task-specific policies explicit:
    local IK branch selection, full-turn joint canonicalization, and path quality
    validation for data collection.
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
        self.eef_link_id = self.find_link_id(config.eef_link_name)
        self.joint_limits = np.asarray(self.mp_planner.joint_limits, dtype=np.float64)
        self.dof = int(self.joint_limits.shape[0])
        self.equivalent_joint_mask = self.compute_equivalent_joint_mask()

        self.base_pose_world = Pose.identity()
        self.base_pose_inverse = Pose.identity()
        self.point_cloud_names: set[str] = set()
        self.object_names: set[str] = set()
        self.teleop_solver = TeleopIKSolver(self)

        self.set_base_pose(config.base_pose_world)

    # Public API

    def set_base_pose(self, base_pose_world: Pose) -> None:
        self.base_pose_world = base_pose_world.copy()
        self.base_pose_inverse = invert_pose(self.base_pose_world)
        self.mp_planner.set_base_pose(self.to_mplib_pose(self.base_pose_world))

    def get_eef_pose(self, qpos: np.ndarray) -> Pose:
        return self.compute_eef_pose_world(qpos)

    def solve_ik(self, target_eef_pose_world: Pose, current_qpos: np.ndarray) -> IKResult:
        current_qpos = ensure_qpos(current_qpos, self.dof, "current_qpos")
        candidates, ik_report = self.collect_ik_candidates(target_eef_pose_world, current_qpos, self.planning_profile)
        if not candidates:
            return IKResult(success=False, qpos=None, reason="No valid IK candidate.", report=ik_report)
        qpos, report = candidates[0]
        return IKResult(success=True, qpos=qpos, report={**report, "candidate_summary": ik_report})

    def solve_teleop_ik(self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray) -> IKResult:
        return self.teleop_solver.solve(target_eef_pose_world, current_qpos, previous_qpos_cmd)

    def plan_path(self, target_eef_pose_world: Pose, current_qpos: np.ndarray) -> PathResult:
        profile = self.planning_profile
        current_qpos = ensure_qpos(current_qpos, self.dof, "current_qpos")
        current_qpos = self.wrap_qpos_to_limits(current_qpos, current_qpos, self.resolve_planning_limits(profile, current_qpos))

        current_pose = self.compute_eef_pose_world(current_qpos)
        pos_error, rot_error = compute_pose_error(target_eef_pose_world, current_pose)
        if pos_error <= profile.max_pose_error_pos_m and rot_error <= profile.max_pose_error_rot_rad:
            return PathResult(success=True, qpos_path=current_qpos.reshape(1, -1), source="hold", report={"num_waypoints": 1})

        candidates, ik_report = self.collect_ik_candidates(target_eef_pose_world, current_qpos, profile)
        if not candidates:
            return PathResult(success=False, qpos_path=None, source="failed", reason="No valid IK candidate.", report={"ik": ik_report})

        results: list[PathResult] = []
        if profile.use_joint_interpolation:
            results.extend(self.try_joint_interpolation(target_eef_pose_world, current_qpos, candidates, profile))
        if profile.use_screw:
            results.append(self.try_screw_plan(target_eef_pose_world, current_qpos, profile))
        if profile.use_rrt:
            results.extend(self.try_multi_rrt_plan(target_eef_pose_world, current_qpos, candidates, profile))

        valid_results = [result for result in results if result.success and result.qpos_path is not None]
        if not valid_results:
            reason_counts = self.count_result_reasons(results)
            reason = "; ".join(f"{text} x{count}" for text, count in reason_counts.items()) or "All planning strategies failed."
            return PathResult(
                success=False,
                qpos_path=None,
                source="failed",
                reason=reason,
                report={"ik": ik_report, "num_planning_attempts": len(results), "planning_reject_counts": reason_counts},
            )

        valid_results.sort(key=lambda result: result.report.get("path_score", float("inf")))
        best = valid_results[0]
        best.report.setdefault("ik", ik_report)
        best.report["num_planning_attempts"] = len(results)
        best.report["num_valid_plans"] = len(valid_results)
        return best

    # Scene API

    def update_point_cloud(self, points_world: np.ndarray, resolution: float = 0.02, name: str = "scene_pcd") -> None:
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_world must have shape (N, 3), got {points.shape}.")
        self.mp_planner.update_point_cloud(points=points, resolution=resolution, name=name)
        self.point_cloud_names.add(name)

    def remove_point_cloud(self, name: str = "scene_pcd") -> bool:
        removed = bool(self.mp_planner.remove_point_cloud(name=name))
        self.point_cloud_names.discard(name)
        return removed

    def add_box_world(self, name: str, size: tuple[float, float, float], pose_world: Pose, replace: bool = True) -> None:
        if replace:
            self.remove_object(name)
        fcl = self.load_fcl()
        box = fcl.Box(np.asarray(size, dtype=np.float64))
        collision_object = self.make_collision_object(fcl, box, pose_world)
        self.add_collision_object_to_world(name, collision_object)
        self.object_names.add(name)

    def remove_object(self, name: str) -> bool:
        world = self.mp_planner.planning_world
        removed = False
        if hasattr(self.mp_planner, "remove_object"):
            try:
                removed = bool(self.mp_planner.remove_object(name))
            except Exception:
                removed = False
        for method_name in ("remove_object", "remove_normal_object"):
            if removed or not hasattr(world, method_name):
                continue
            try:
                removed = bool(getattr(world, method_name)(name))
            except Exception:
                removed = False
        self.object_names.discard(name)
        return removed

    def clear_scene(self) -> None:
        for name in list(self.point_cloud_names):
            self.remove_point_cloud(name)
        for name in list(self.object_names):
            self.remove_object(name)

    # FK and frame helpers

    def world_to_base_pose(self, pose_world: Pose) -> Pose:
        return compose_pose(self.base_pose_inverse, pose_world)

    def base_to_world_pose(self, pose_base: Pose) -> Pose:
        return compose_pose(self.base_pose_world, pose_base)

    def compute_eef_pose_base(self, qpos: np.ndarray) -> Pose:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        full_qpos = self.pad_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        return self.from_mplib_pose(self.pinocchio_model.get_link_pose(self.eef_link_id))

    def compute_eef_pose_world(self, qpos: np.ndarray) -> Pose:
        return self.base_to_world_pose(self.compute_eef_pose_base(qpos))

    def compute_eef_jacobian(self, qpos: np.ndarray) -> np.ndarray:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        full_qpos = self.pad_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        jacobian = np.asarray(self.pinocchio_model.compute_single_link_jacobian(full_qpos, self.eef_link_id, False), dtype=np.float64)
        if jacobian.shape[1] < self.dof:
            raise RuntimeError(f"Jacobian has {jacobian.shape[1]} columns but planner dof is {self.dof}.")
        return jacobian[:, : self.dof]

    # Offline IK helpers

    def call_mplib_ik(self, target_pose_base: Pose, seed_qpos: np.ndarray, n_init_qpos: int, return_closest: bool) -> tuple[str, Any]:
        return self.mp_planner.IK(
            goal_pose=self.to_mplib_pose(target_pose_base),
            start_qpos=seed_qpos,
            mask=None,
            n_init_qpos=n_init_qpos,
            threshold=1e-3,
            return_closest=return_closest,
        )

    def generate_ik_seeds(self, current_qpos: np.ndarray, profile: PlanningProfile) -> list[np.ndarray]:
        limits = self.resolve_planning_limits(profile, current_qpos)
        low, high = limits[:, 0], limits[:, 1]
        current = self.wrap_qpos_to_limits(current_qpos, current_qpos, limits)
        seeds: list[np.ndarray] = [current.copy()]

        for radius_deg in profile.ik_seed_offsets_deg:
            radius = float(np.deg2rad(radius_deg))
            for joint_index in range(self.dof):
                for sign in (-1.0, 1.0):
                    seed = current.copy()
                    seed[joint_index] += sign * radius
                    seed = self.wrap_qpos_to_limits(seed, current, limits)
                    seeds.append(np.clip(seed, low, high))

        if profile.num_random_ik_seeds > 0:
            rng = np.random.default_rng(profile.random_seed)
            radius = float(np.deg2rad(max(profile.ik_seed_offsets_deg or (5.0,))))
            for random_index in range(profile.num_random_ik_seeds):
                seed = current + rng.uniform(-radius, radius, size=self.dof)
                seed = self.wrap_qpos_to_limits(seed, current, limits)
                seeds.append(np.clip(seed, low, high))
        return self.unique_qpos_list(seeds)

    def collect_ik_candidates(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        profile: PlanningProfile,
    ) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], dict[str, Any]]:
        target_pose_base = self.world_to_base_pose(target_eef_pose_world)
        seeds = self.generate_ik_seeds(current_qpos, profile)
        candidates: list[tuple[np.ndarray, dict[str, Any]]] = []
        reject_counts: dict[str, int] = {}
        reject_examples: list[dict[str, Any]] = []
        raw_success_count = 0

        for seed_index, seed in enumerate(seeds):
            status, raw_qpos = self.call_mplib_ik(target_pose_base, seed, n_init_qpos=1, return_closest=True)
            if status != "Success" or raw_qpos is None:
                reason = "mplib_ik_failed"
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue

            raw_success_count += 1
            raw_qpos = np.asarray(raw_qpos, dtype=np.float64)
            qpos = self.nearest_equivalent_qpos(raw_qpos, current_qpos)
            valid, report = self.filter_ik_candidate(qpos, raw_qpos, target_eef_pose_world, current_qpos, profile)
            if valid:
                report["ik_score"] = self.score_ik_candidate(qpos, current_qpos, report)
                report["seed_index"] = seed_index
                candidates.append((qpos.copy(), report))
                continue

            reason = str(report.get("reason", "rejected"))
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            if len(reject_examples) < 5:
                reject_examples.append(self.compact_reject_report(seed_index, report))

        candidates.sort(key=lambda item: item[1]["ik_score"])
        summary: dict[str, Any] = {
            "num_seeds": len(seeds),
            "raw_ik_success_count": raw_success_count,
            "valid_candidate_count": len(candidates),
            "returned_candidate_count": min(len(candidates), profile.num_ik_candidates),
            "reject_counts": reject_counts,
            "random_seed": profile.random_seed,
        }
        if reject_examples:
            summary["reject_examples"] = reject_examples
        return candidates[: profile.num_ik_candidates], summary

    def filter_ik_candidate(
        self,
        qpos: np.ndarray,
        raw_qpos: np.ndarray,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        profile: PlanningProfile,
    ) -> tuple[bool, dict[str, Any]]:
        limits = self.resolve_planning_limits(profile, current_qpos)
        low, high = limits[:, 0], limits[:, 1]
        qpos[:] = self.wrap_qpos_to_limits(qpos, current_qpos, limits)
        report: dict[str, Any] = {"raw_qpos": raw_qpos.copy()}

        outside, violation = self.limit_violation(qpos, limits)
        if np.any(outside):
            indices = np.where(outside)[0]
            report.update(
                reason="IK candidate outside planning limits.",
                outside_joint_indices_1based=(indices + 1).tolist(),
                max_limit_violation_deg=float(np.rad2deg(np.max(violation[indices]))),
                qpos_deg=np.rad2deg(qpos.copy()),
                low_deg=np.rad2deg(low.copy()),
                high_deg=np.rad2deg(high.copy()),
            )
            return False, report

        np.clip(qpos, low, high, out=qpos)
        delta = self.compute_qpos_delta(qpos, current_qpos)
        max_delta = np.deg2rad(self.profile_array(profile.max_ik_delta_deg, "max_ik_delta_deg"))
        over_delta = np.abs(delta) > max_delta
        if np.any(over_delta):
            indices = np.where(over_delta)[0]
            violation_delta = np.maximum(np.abs(delta) - max_delta, 0.0)
            report.update(
                reason="IK candidate exceeds max_ik_delta_deg.",
                max_delta_joint_indices_1based=(indices + 1).tolist(),
                max_delta_violation_deg=float(np.rad2deg(np.max(violation_delta[indices]))),
            )
            return False, report

        pose_error_pos, pose_error_rot = self.compute_world_pose_error(target_eef_pose_world, qpos)
        report.update(
            pose_error_pos_m=pose_error_pos,
            pose_error_rot_rad=pose_error_rot,
            qpos_distance=float(np.linalg.norm(delta)),
            max_qpos_delta=float(np.max(np.abs(delta))),
            max_qpos_delta_deg=float(np.rad2deg(np.max(np.abs(delta)))),
            joint_limit_penalty=self.joint_limit_penalty(qpos, limits),
        )

        if pose_error_pos > profile.max_pose_error_pos_m or pose_error_rot > profile.max_pose_error_rot_rad:
            report["reason"] = "IK candidate pose error exceeds threshold."
            return False, report
        if profile.enable_self_collision_check and self.has_self_collision(qpos):
            report["reason"] = "IK candidate in self-collision."
            return False, report
        if profile.enable_env_collision_check and self.has_env_collision(qpos):
            report["reason"] = "IK candidate in environment collision."
            return False, report
        return True, report

    def score_ik_candidate(self, qpos: np.ndarray, current_qpos: np.ndarray, report: dict[str, Any]) -> float:
        delta = self.compute_qpos_delta(qpos, current_qpos)
        return float(
            np.linalg.norm(delta)
            + 0.5 * np.max(np.abs(delta))
            + 0.2 * report.get("joint_limit_penalty", 0.0)
            + 0.2 * (report.get("pose_error_pos_m", 0.0) + report.get("pose_error_rot_rad", 0.0))
        )

    # Planning strategies

    def try_joint_interpolation(
        self,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        candidates: list[tuple[np.ndarray, dict[str, Any]]],
        profile: PlanningProfile,
    ) -> list[PathResult]:
        results: list[PathResult] = []
        max_step = np.deg2rad(profile.max_waypoint_delta_deg)
        for goal_qpos, ik_report in candidates:
            path = interpolate_qpos_path(current_qpos, goal_qpos, max_step)
            result = self.validate_path(path, target_eef_pose_world, current_qpos, source="interpolation", profile=profile)
            if result.success:
                result.report["ik"] = ik_report
            results.append(result)
        return results

    def try_screw_plan(self, target_eef_pose_world: Pose, current_qpos: np.ndarray, profile: PlanningProfile) -> PathResult:
        try:
            result = self.mp_planner.plan_screw(
                goal_pose=self.to_mplib_pose(target_eef_pose_world),
                current_qpos=current_qpos,
                time_step=profile.path_dt,
                qpos_step=profile.screw_qpos_step,
                wrt_world=True,
            )
        except TypeError:
            result = self.mp_planner.plan_screw(
                goal_pose=self.to_mplib_pose(target_eef_pose_world),
                current_qpos=current_qpos,
                time_step=profile.path_dt,
                qpos_step=profile.screw_qpos_step,
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
                try:
                    result = self.mp_planner.plan_qpos(
                        goal_qposes=goal_qposes,
                        current_qpos=current_qpos,
                        time_step=profile.path_dt,
                        rrt_range=rrt_range,
                        planning_time=profile.rrt_time_limit,
                        simplify=profile.simplify_path,
                    )
                except TypeError:
                    result = self.mp_planner.plan_qpos(goal_qposes, current_qpos, time_step=profile.path_dt)
                path_result = self.result_from_mplib(result, target_eef_pose_world, current_qpos, source="rrt", profile=profile)
                path_result.report["rrt_range"] = rrt_range
                path_result.report["rrt_attempt_index"] = attempt_index
                results.append(path_result)
        return results

    def result_from_mplib(self, result: dict[str, Any], target_eef_pose_world: Pose, current_qpos: np.ndarray, source: str, profile: PlanningProfile) -> PathResult:
        status = result.get("status", "Unknown") if isinstance(result, dict) else "Invalid"
        if not isinstance(result, dict) or status != "Success":
            return PathResult(success=False, qpos_path=None, source=source, reason=f"MPlib {source} failed: {status}", report={"mplib_status": status})

        path = np.asarray(result.get("position", []), dtype=np.float64)
        if len(path) == 0:
            path = current_qpos.reshape(1, -1)
        path_result = self.validate_path(path, target_eef_pose_world, current_qpos, source=source, profile=profile)
        path_result.report.update(mplib_status=status)
        path_result.qvel_path = result.get("velocity")
        path_result.qacc_path = result.get("acceleration")
        path_result.time_path = result.get("time")
        path_result.duration = result.get("duration")
        return path_result

    # Path validation

    def validate_path(
        self,
        path: np.ndarray,
        target_eef_pose_world: Pose,
        current_qpos: np.ndarray,
        source: str,
        profile: PlanningProfile,
    ) -> PathResult:
        try:
            path = self.unwrap_qpos_path(path, current_qpos)
            path = self.wrap_path_to_planning_limits(path, current_qpos, profile)
        except ValueError as error:
            return PathResult(success=False, qpos_path=None, source=source, reason=str(error))

        report = self.build_path_report(path, target_eef_pose_world, current_qpos, profile)
        if report["limit_violation"]:
            return PathResult(success=False, qpos_path=None, source=source, reason="Path violates planning limits.", report=report)
        if report["start_qpos_error_rad"] > np.deg2rad(profile.max_waypoint_delta_deg) + 1e-12:
            return PathResult(success=False, qpos_path=None, source=source, reason="Path start is too far from current_qpos.", report=report)
        if report["max_waypoint_delta_rad"] > np.deg2rad(profile.max_waypoint_delta_deg) + 1e-12:
            return PathResult(success=False, qpos_path=None, source=source, reason="Path waypoint delta too large.", report=report)
        if report["terminal_pos_error_m"] > profile.max_pose_error_pos_m or report["terminal_rot_error_rad"] > profile.max_pose_error_rot_rad:
            return PathResult(success=False, qpos_path=None, source=source, reason="Terminal pose error too large.", report=report)
        if profile.enable_self_collision_check or profile.enable_env_collision_check:
            collision_report = self.path_collision_report(path, profile)
            report.update(collision_report)
            if collision_report["collision"]:
                return PathResult(success=False, qpos_path=None, source=source, reason="Path in collision.", report=report)
        report["path_score"] = self.score_path(report)
        return PathResult(success=True, qpos_path=path, source=source, report=report)

    def build_path_report(self, path: np.ndarray, target_eef_pose_world: Pose, current_qpos: np.ndarray, profile: PlanningProfile) -> dict[str, Any]:
        diff = np.diff(path, axis=0) if len(path) > 1 else np.zeros((0, self.dof), dtype=np.float64)
        max_step = float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0
        path_length = float(np.sum(np.linalg.norm(diff, axis=1))) if len(diff) > 0 else 0.0
        terminal_pos_error, terminal_rot_error = self.compute_world_pose_error(target_eef_pose_world, path[-1])
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
        }
        if np.any(outside):
            waypoint_indices, joint_indices = np.where(outside)
            report["limit_violation_waypoint_index"] = int(waypoint_indices[0])
            report["limit_violation_joint_index_1based"] = int(joint_indices[0] + 1)
            report["max_limit_violation_deg"] = float(np.rad2deg(np.max(violation[outside])))
            report["limit_violation_qpos_deg"] = np.rad2deg(path[waypoint_indices[0]].copy())
        return report

    def path_collision_report(self, path: np.ndarray, profile: PlanningProfile) -> dict[str, Any]:
        for index, qpos in enumerate(path):
            if profile.enable_self_collision_check and self.has_self_collision(qpos):
                return {"collision": True, "collision_type": "self", "collision_index": int(index)}
            if profile.enable_env_collision_check and self.has_env_collision(qpos):
                return {"collision": True, "collision_type": "env", "collision_index": int(index)}
        return {"collision": False}

    def score_path(self, report: dict[str, Any]) -> float:
        return float(report.get("joint_path_length", 0.0) + 2.0 * report.get("max_waypoint_delta_rad", 0.0))

    # Joint canonicalization

    def resolve_planning_limits(self, profile: PlanningProfile, reference_qpos: np.ndarray | None = None) -> np.ndarray:
        if profile.planning_limits_deg is not None:
            limits = np.deg2rad(np.asarray(profile.planning_limits_deg, dtype=np.float64))
            if limits.shape != self.joint_limits.shape:
                raise ValueError(f"planning_limits_deg must have shape {self.joint_limits.shape}, got {limits.shape}.")
            return limits

        limits = self.joint_limits.copy()
        if reference_qpos is None:
            reference_qpos = np.zeros(self.dof, dtype=np.float64)
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")

        for joint_index in range(self.dof):
            if not self.equivalent_joint_mask[joint_index]:
                continue
            hardware_low = self.joint_limits[joint_index, 0]
            hardware_high = self.joint_limits[joint_index, 1]
            local_low = reference_qpos[joint_index] - np.pi
            local_high = reference_qpos[joint_index] + np.pi
            limits[joint_index, 0] = max(hardware_low, local_low)
            limits[joint_index, 1] = min(hardware_high, local_high)
        return limits

    def nearest_equivalent_qpos(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")
        result = qpos.copy()
        period = 2.0 * np.pi
        low, high = self.joint_limits[:, 0], self.joint_limits[:, 1]
        for joint_index in range(self.dof):
            if not self.equivalent_joint_mask[joint_index]:
                continue
            k = np.round((reference_qpos[joint_index] - result[joint_index]) / period)
            k_min = np.ceil((low[joint_index] - result[joint_index]) / period)
            k_max = np.floor((high[joint_index] - result[joint_index]) / period)
            result[joint_index] += np.clip(k, k_min, k_max) * period
        return result

    def wrap_qpos_to_limits(self, qpos: np.ndarray, reference_qpos: np.ndarray, limits: np.ndarray, limit_tol: float = 1e-5) -> np.ndarray:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")
        result = qpos.copy()
        period = 2.0 * np.pi
        for joint_index in range(self.dof):
            low = limits[joint_index, 0]
            high = limits[joint_index, 1]
            width = high - low
            can_wrap = self.equivalent_joint_mask[joint_index] or width >= period - 1e-5
            if not can_wrap:
                continue
            k_min = int(np.ceil((low - result[joint_index] - limit_tol) / period))
            k_max = int(np.floor((high - result[joint_index] + limit_tol) / period))
            if k_min > k_max:
                continue
            values = np.array([result[joint_index] + k * period for k in range(k_min, k_max + 1)], dtype=np.float64)
            best_index = int(np.argmin(np.abs(values - reference_qpos[joint_index])))
            result[joint_index] = values[best_index]
        np.clip(result, limits[:, 0], limits[:, 1], out=result)
        return result

    def wrap_path_to_planning_limits(self, path: np.ndarray, current_qpos: np.ndarray, profile: PlanningProfile) -> np.ndarray:
        path = np.asarray(path, dtype=np.float64).copy()
        if path.ndim != 2 or path.shape[1] != self.dof:
            raise ValueError(f"path must have shape (N, {self.dof}), got {path.shape}.")
        limits = self.resolve_planning_limits(profile, current_qpos)
        path[0] = self.wrap_qpos_to_limits(path[0], current_qpos, limits)
        for index in range(1, len(path)):
            path[index] = self.wrap_qpos_to_limits(path[index], path[index - 1], limits)
        return path

    def unwrap_qpos_path(self, path: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        path = np.asarray(path, dtype=np.float64).copy()
        if path.ndim != 2 or path.shape[1] != self.dof:
            raise ValueError(f"path must have shape (N, {self.dof}), got {path.shape}.")
        if len(path) == 0:
            return path
        path[0] = self.nearest_equivalent_qpos(path[0], reference_qpos)
        for index in range(1, len(path)):
            path[index] = self.nearest_equivalent_qpos(path[index], path[index - 1])
        return path

    def compute_qpos_delta(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        reference_qpos = ensure_qpos(reference_qpos, self.dof, "reference_qpos")
        delta = qpos - reference_qpos
        period = 2.0 * np.pi
        for joint_index in range(self.dof):
            if self.equivalent_joint_mask[joint_index]:
                delta[joint_index] = (delta[joint_index] + np.pi) % period - np.pi
        return delta

    def limit_violation(self, qpos: np.ndarray, limits: np.ndarray, limit_tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        below = qpos < limits[:, 0] - limit_tol
        above = qpos > limits[:, 1] + limit_tol
        outside = below | above
        lower = np.maximum(limits[:, 0] - qpos, 0.0)
        upper = np.maximum(qpos - limits[:, 1], 0.0)
        return outside, np.maximum(lower, upper)

    def path_limit_violation(self, path: np.ndarray, limits: np.ndarray, limit_tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        below = path < limits[None, :, 0] - limit_tol
        above = path > limits[None, :, 1] + limit_tol
        outside = below | above
        lower = np.maximum(limits[None, :, 0] - path, 0.0)
        upper = np.maximum(path - limits[None, :, 1], 0.0)
        return outside, np.maximum(lower, upper)

    # Small utilities

    def compute_world_pose_error(self, target_eef_pose_world: Pose, qpos: np.ndarray) -> tuple[float, float]:
        return compute_pose_error(target_eef_pose_world, self.compute_eef_pose_world(qpos))

    def has_self_collision(self, qpos: np.ndarray) -> bool:
        try:
            return bool(self.mp_planner.check_for_self_collision(qpos))
        except Exception:
            return bool(self.mp_planner.check_for_self_collision(self.pad_qpos(qpos)))

    def has_env_collision(self, qpos: np.ndarray) -> bool:
        try:
            return bool(self.mp_planner.check_for_env_collision(qpos))
        except Exception:
            return bool(self.mp_planner.check_for_env_collision(self.pad_qpos(qpos)))

    def joint_limit_penalty(self, qpos: np.ndarray, limits: np.ndarray) -> float:
        center = 0.5 * (limits[:, 0] + limits[:, 1])
        half_range = np.maximum(0.5 * (limits[:, 1] - limits[:, 0]), 1e-6)
        return float(np.sum(((qpos - center) / half_range) ** 2))

    def compute_equivalent_joint_mask(self) -> np.ndarray:
        widths = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        return widths > (2.0 * np.pi - 1e-6)

    def profile_array(self, values: tuple[float, ...], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape == (1,):
            return np.repeat(array, self.dof)
        if array.shape != (self.dof,):
            raise ValueError(f"{name} must have length 1 or {self.dof}, got {array.shape[0]}.")
        return array

    def unique_qpos_list(self, qpos_list: list[np.ndarray], atol: float = 1e-8) -> list[np.ndarray]:
        unique: list[np.ndarray] = []
        for qpos in qpos_list:
            if not any(np.allclose(qpos, item, atol=atol, rtol=0.0) for item in unique):
                unique.append(qpos)
        return unique

    def compact_reject_report(self, seed_index: int, report: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {"seed_index": seed_index, "reason": report.get("reason")}
        keys = (
            "outside_joint_indices_1based",
            "max_limit_violation_deg",
            "max_delta_joint_indices_1based",
            "max_delta_violation_deg",
            "pose_error_pos_m",
            "pose_error_rot_rad",
            "max_qpos_delta_deg",
        )
        for key in keys:
            if key in report:
                compact[key] = report[key]
        return compact

    def count_result_reasons(self, results: list[PathResult]) -> dict[str, int]:
        reason_counts: dict[str, int] = {}
        for result in results:
            reason = result.reason or "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return reason_counts

    def pad_qpos(self, qpos: np.ndarray) -> np.ndarray:
        if hasattr(self.mp_planner, "pad_move_group_qpos"):
            return self.mp_planner.pad_move_group_qpos(qpos)
        return qpos

    def find_link_id(self, link_name: str) -> int:
        if hasattr(self.pinocchio_model, "get_link_names"):
            names = list(self.pinocchio_model.get_link_names())
        elif hasattr(self.pinocchio_model, "link_names"):
            names = list(self.pinocchio_model.link_names)
        else:
            raise RuntimeError("Cannot get link names from MPlib Pinocchio model.")
        if link_name not in names:
            raise ValueError(f"Link {link_name!r} not found. Available links: {names}")
        return int(names.index(link_name))

    def to_mplib_pose(self, pose: Pose) -> Any:
        return self.mp.Pose(p=pose.p, q=pose.q)

    def from_mplib_pose(self, pose: Any) -> Pose:
        return Pose(p=np.asarray(pose.p, dtype=np.float64), q=np.asarray(pose.q, dtype=np.float64))

    def load_fcl(self) -> Any:
        for module_name in ("mplib.collision_detection.fcl", "mplib.pymp.collision_detection.fcl"):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                continue
        raise ImportError("Cannot import MPlib FCL module. Use update_point_cloud(...) if this MPlib build has no FCL binding.")

    def make_collision_object(self, fcl: Any, geometry: Any, pose_world: Pose) -> Any:
        pose = self.to_mplib_pose(pose_world)
        try:
            return fcl.CollisionObject(geometry, pose)
        except TypeError:
            return fcl.CollisionObject(geometry, pose.p, pose.q)

    def add_collision_object_to_world(self, name: str, collision_object: Any) -> None:
        world = self.mp_planner.planning_world
        if hasattr(world, "add_object"):
            try:
                world.add_object(name, collision_object)
            except TypeError:
                world.add_object(collision_object, name)
            return
        for method_name in ("add_normal_object", "set_normal_object"):
            if not hasattr(world, method_name):
                continue
            method = getattr(world, method_name)
            try:
                method(collision_object, name)
            except TypeError:
                method(name, collision_object)
            return
        raise RuntimeError("PlanningWorld cannot add collision objects in this MPlib version.")
