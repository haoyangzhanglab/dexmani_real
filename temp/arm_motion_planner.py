from __future__ import annotations

from pathlib import Path
from typing import Any

import mplib as mp
import numpy as np
from mplib.collision_detection import fcl

from planner_utils import (
    evaluate_path,
    pose_error,
    relative_pose,
    transform_pose,
    unwrap_path_to_reference,
    wrap_to_reference,
)


IK_WEIGHTS = np.array((3.0, 2.0, 2.5, 1.5, 1.5, 1.0, 1.0), dtype=float)


class ArmMotionPlanner:
    def __init__(
        self,
        urdf_path: str | Path,
        srdf_path: str | Path,
        move_group: str,
        joint_vel_limits: np.ndarray | None = None,
        joint_acc_limits: np.ndarray | None = None,
        base_pose: mp.Pose | None = None,
        use_convex: bool = False,
        equivalent_joint_indices: list[int] | None = None,
    ):
        urdf_path = Path(urdf_path)
        srdf_path = Path(srdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(urdf_path)
        if not srdf_path.is_file():
            raise FileNotFoundError(srdf_path)

        self.planner = mp.Planner(
            urdf=str(urdf_path),
            srdf=str(srdf_path),
            move_group=move_group,
            use_convex=use_convex,
            joint_vel_limits=None if joint_vel_limits is None else np.asarray(joint_vel_limits, dtype=float),
            joint_acc_limits=None if joint_acc_limits is None else np.asarray(joint_acc_limits, dtype=float),
        )

        self.joint_limits = np.asarray(self.planner.joint_limits, dtype=float).copy()
        if self.joint_limits.shape[0] != 7:
            raise ValueError("ArmMotionPlanner is tuned for xArm7 and expects exactly 7 planning joints.")

        if equivalent_joint_indices is None:
            joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
            equivalent_joint_indices = np.where(joint_range > 2.0 * np.pi + 1e-6)[0].tolist()
        self.equivalent_joint_indices = equivalent_joint_indices
        self.last_goal_qpos: np.ndarray | None = None

        if base_pose is None:
            base_pose = mp.Pose(p=np.zeros(3, dtype=float), q=np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
        self.set_base_pose(base_pose)

    def set_base_pose(self, base_pose: mp.Pose):
        self.base_pose = mp.Pose(
            p=np.asarray(base_pose.p, dtype=float),
            q=np.asarray(base_pose.q, dtype=float),
        )
        self.planner.set_base_pose(self.base_pose)

    def canonicalize_qpos(self, qpos: np.ndarray, strict: bool = True) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float).copy()
        if qpos.shape != (len(self.joint_limits),):
            raise ValueError(f"qpos must have shape ({len(self.joint_limits)},).")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos must contain only finite values.")

        wrapped = bool(self.planner.wrap_joint_limit(qpos))
        if strict and not wrapped:
            raise ValueError("qpos cannot be wrapped into the configured joint limits.")
        return qpos

    def validate_qpos_limits(self, qpos: np.ndarray, tolerance: float = 1e-6) -> bool:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (len(self.joint_limits),):
            return False
        if not np.all(np.isfinite(qpos)):
            return False
        return bool(
            np.all(qpos >= self.joint_limits[:, 0] - tolerance)
            and np.all(qpos <= self.joint_limits[:, 1] + tolerance)
        )

    def validate_final_pose(
        self,
        final_qpos: np.ndarray,
        goal_pose_world: mp.Pose,
        pos_threshold: float,
        rot_threshold: float,
    ) -> tuple[bool, float, float]:
        if pos_threshold <= 0.0 or rot_threshold <= 0.0:
            raise ValueError("final pose validation thresholds must be positive.")

        pos_err, rot_err = pose_error(self.fk_world(final_qpos), goal_pose_world)
        return pos_err <= pos_threshold and rot_err <= rot_threshold, pos_err, rot_err

    def check_current_state(self, current_qpos: np.ndarray) -> dict[str, Any] | None:
        if not self.validate_qpos_limits(current_qpos):
            return {"status": "Planning Failed", "mode_used": "precheck", "debug": {"reason": "current_joint_limit"}}
        if len(self.planner.check_for_self_collision(current_qpos)) > 0:
            return {"status": "Planning Failed", "mode_used": "precheck", "debug": {"reason": "current_self_collision"}}
        if len(self.planner.check_for_env_collision(current_qpos)) > 0:
            return {"status": "Planning Failed", "mode_used": "precheck", "debug": {"reason": "current_env_collision"}}
        return None

    def build_timed_path(
        self,
        plan: dict[str, Any],
        current_qpos: np.ndarray,
        time_step: float,
        verbose: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_path = np.asarray(plan["position"], dtype=float)
        path = unwrap_path_to_reference(
            raw_path,
            current_qpos,
            self.joint_limits,
            self.equivalent_joint_indices,
        )

        offset = path - raw_path
        max_offset = float(np.max(np.abs(offset))) if len(offset) > 0 else 0.0
        max_offset_step = float(np.max(np.abs(np.diff(offset, axis=0)))) if len(offset) > 1 else 0.0

        debug = {
            "unwrap_changed": max_offset > 1e-9,
            "unwrap_offset_is_constant": max_offset_step <= 1e-9,
            "max_unwrap_delta_deg": round(float(np.rad2deg(max_offset)), 6),
            "max_unwrap_offset_step_deg": round(float(np.rad2deg(max_offset_step)), 6),
        }

        if max_offset_step <= 1e-9:
            debug["timing_source"] = "mplib_original"
            return {
                "qpos": path,
                "qvel": np.asarray(plan["velocity"], dtype=float),
                "qacc": np.asarray(plan["acceleration"], dtype=float),
                "time": np.asarray(plan["time"], dtype=float),
                "duration": float(plan["duration"]),
            }, debug

        time, qpos, qvel, qacc, duration = self.planner.TOPP(path, step=time_step, verbose=verbose)
        debug["timing_source"] = "topp_after_unwrap"
        return {
            "qpos": np.asarray(qpos, dtype=float),
            "qvel": np.asarray(qvel, dtype=float),
            "qacc": np.asarray(qacc, dtype=float),
            "time": np.asarray(time, dtype=float),
            "duration": float(duration),
        }, debug

    def fk_world(self, qpos: np.ndarray) -> mp.Pose:
        qpos = self.canonicalize_qpos(qpos)
        model = self.planner.pinocchio_model
        model.compute_forward_kinematics(qpos)
        return transform_pose(self.base_pose, model.get_link_pose(self.planner.move_group_link_id))

    def ik_world(
        self,
        goal_pose_world: mp.Pose,
        current_qpos: np.ndarray,
        threshold: float = 1e-3,
        pos_threshold: float | None = None,
        rot_threshold: float = float(np.deg2rad(0.5)),
        random_seed_count: int = 16,
        max_candidates: int = 8,
        score_mode: str = "nearest",
        verbose: bool = False,
    ) -> list[np.ndarray]:
        current_qpos = self.canonicalize_qpos(current_qpos)
        goal_pose_base = relative_pose(self.base_pose, goal_pose_world)
        if pos_threshold is None:
            pos_threshold = threshold
        if pos_threshold <= 0.0 or rot_threshold <= 0.0:
            raise ValueError("IK pose validation thresholds must be positive.")

        seeds = [current_qpos.copy()]
        if self.last_goal_qpos is not None:
            seeds.append(self.last_goal_qpos.copy())

        center = 0.5 * (self.joint_limits[:, 0] + self.joint_limits[:, 1])
        seeds.append(center)

        for base_seed in list(seeds):
            for joint_id in self.equivalent_joint_indices:
                for direction in (-1.0, 1.0):
                    seed = base_seed.copy()
                    seed[joint_id] += direction * 2.0 * np.pi
                    if self.joint_limits[joint_id, 0] <= seed[joint_id] <= self.joint_limits[joint_id, 1]:
                        seeds.append(seed)

        if random_seed_count > 0:
            local_count = random_seed_count // 2
            global_count = random_seed_count - local_count
            if local_count > 0:
                local = current_qpos + np.random.normal(scale=0.25, size=(local_count, 7))
                local = np.clip(local, self.joint_limits[:, 0], self.joint_limits[:, 1])
                seeds.extend(local)
            if global_count > 0:
                seeds.extend(np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1], size=(global_count, 7)))

        scored_goals: list[tuple[tuple[float, ...], np.ndarray]] = []
        for seed in seeds:
            status, ik_result = self.planner.IK(
                goal_pose=goal_pose_base,
                start_qpos=np.asarray(seed, dtype=float),
                n_init_qpos=2,
                threshold=threshold,
                return_closest=False,
                verbose=verbose,
            )
            if status != "Success" or ik_result is None:
                continue

            for goal_qpos in [ik_result] if isinstance(ik_result, np.ndarray) else ik_result:
                goal_qpos = self.canonicalize_qpos(goal_qpos)
                goal_qpos = wrap_to_reference(goal_qpos, current_qpos, self.joint_limits, self.equivalent_joint_indices)
                if len(self.planner.check_for_self_collision(goal_qpos)) > 0:
                    continue
                if len(self.planner.check_for_env_collision(goal_qpos)) > 0:
                    continue

                pos_err, rot_err = pose_error(self.fk_world(goal_qpos), goal_pose_world)
                if pos_err > pos_threshold or rot_err > rot_threshold:
                    continue

                distance = float(np.sum(IK_WEIGHTS * (goal_qpos - current_qpos) ** 2))
                margin = float(np.min(np.minimum(goal_qpos - self.joint_limits[:, 0], self.joint_limits[:, 1] - goal_qpos)))
                branch = 0.0
                if self.last_goal_qpos is not None:
                    branch = 0.02 * float(np.sum(np.abs(goal_qpos - self.last_goal_qpos)))

                if score_mode == "continuous":
                    score = (branch, distance, -margin)
                elif score_mode == "margin":
                    score = (-margin, distance, branch)
                else:
                    score = (distance, branch, -margin)
                scored_goals.append((score, goal_qpos))

        goals: list[np.ndarray] = []
        for _, goal_qpos in sorted(scored_goals, key=lambda item: item[0]):
            if not any(np.allclose(goal_qpos, q, atol=1e-5) for q in goals):
                goals.append(goal_qpos)
            if len(goals) >= max_candidates:
                break
        return goals

    def ik_best_world(
        self,
        goal_pose_world: mp.Pose,
        current_qpos: np.ndarray,
        threshold: float = 1e-3,
        pos_threshold: float | None = None,
        rot_threshold: float = float(np.deg2rad(0.5)),
        random_seed_count: int = 64,
        score_mode: str = "nearest",
        verbose: bool = False,
    ) -> np.ndarray | None:
        candidates = self.ik_world(
            goal_pose_world,
            current_qpos,
            threshold=threshold,
            pos_threshold=pos_threshold,
            rot_threshold=rot_threshold,
            random_seed_count=random_seed_count,
            max_candidates=1,
            score_mode=score_mode,
            verbose=verbose,
        )
        return None if len(candidates) == 0 else candidates[0]

    def update_point_cloud(
        self,
        points: np.ndarray,
        resolution: float = 1e-3,
        name: str = "scene_pcd",
    ):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point cloud must have shape (N, 3) in world frame.")
        if resolution <= 0.0:
            raise ValueError("point cloud resolution must be positive.")
        self.planner.update_point_cloud(points, resolution=resolution, name=name)

    def remove_point_cloud(self, name: str = "scene_pcd") -> bool:
        return bool(self.planner.remove_point_cloud(name=name))

    def add_obstacle_box(
        self,
        size: np.ndarray,
        pose: mp.Pose,
        name: str = "obstacle_box",
        replace: bool = True,
    ):
        size = np.asarray(size, dtype=float)
        if size.shape != (3,):
            raise ValueError("obstacle box size must have shape (3,).")
        if np.any(size <= 0.0):
            raise ValueError("obstacle box size must be positive.")

        if replace:
            self.planner.remove_object(name)

        box = fcl.Box(size)
        obj = fcl.CollisionObject(box, pose)
        self.planner.planning_world.add_object(name, obj)

    def remove_obstacle_box(self, name: str = "obstacle_box") -> bool:
        return bool(self.planner.remove_object(name))

    def plan_near_pose_world(
        self,
        goal_pose_world: mp.Pose,
        current_qpos: np.ndarray,
        pos_err: float,
        rot_err: float,
        constraint: dict[str, Any] | None = None,
        time_step: float = 0.05,
        screw_qpos_step: float = 0.1,
        final_pos_threshold: float = 1e-3,
        final_rot_threshold: float = float(np.deg2rad(0.5)),
        verbose: bool = False,
    ) -> dict[str, Any] | None:
        if final_pos_threshold <= 0.0 or final_rot_threshold <= 0.0:
            raise ValueError("final pose validation thresholds must be positive.")

        if pos_err <= 1e-4 and rot_err <= np.deg2rad(0.5):
            self.last_goal_qpos = current_qpos.copy()
            return {
                "status": "Success",
                "mode_used": "shortcut",
                "qpos": current_qpos.reshape(1, -1),
                "qvel": np.zeros((1, len(current_qpos)), dtype=float),
                "qacc": np.zeros((1, len(current_qpos)), dtype=float),
                "time": np.array([0.0], dtype=float),
                "duration": 0.0,
                "goal_qpos": current_qpos.copy(),
                "ik_score": None,
                "path_score": None,
                "debug": {"reason": "pose_shortcut"},
            }

        if constraint is not None:
            return None
        if pos_err > 0.05 or rot_err > np.deg2rad(10.0):
            return None

        plan = self.planner.plan_screw(
            goal_pose=goal_pose_world,
            current_qpos=current_qpos,
            qpos_step=screw_qpos_step,
            time_step=time_step,
            wrt_world=True,
            verbose=verbose,
        )
        if plan.get("status", "") != "Success":
            return None

        timed, timing_debug = self.build_timed_path(plan, current_qpos, time_step, verbose=verbose)
        path = timed["qpos"]
        final_qpos = path[-1].copy()
        valid, path_score, debug = evaluate_path(path, current_qpos, final_qpos, self.last_goal_qpos)
        debug.update(timing_debug)
        final_pose_valid, final_pos_err, final_rot_err = self.validate_final_pose(
            final_qpos,
            goal_pose_world,
            final_pos_threshold,
            final_rot_threshold,
        )
        debug.update({
            "final_pos_err": round(final_pos_err, 8),
            "final_rot_err_deg": round(float(np.rad2deg(final_rot_err)), 6),
        })
        if not valid or not final_pose_valid:
            return None

        self.last_goal_qpos = final_qpos.copy()
        return {
            "status": "Success",
            "mode_used": "screw",
            "qpos": timed["qpos"],
            "qvel": timed["qvel"],
            "qacc": timed["qacc"],
            "time": timed["time"],
            "duration": timed["duration"],
            "goal_qpos": final_qpos,
            "ik_score": None,
            "path_score": path_score,
            "debug": debug,
        }

    def plan_qpos_candidates_world(
        self,
        goal_pose_world: mp.Pose,
        current_qpos: np.ndarray,
        constraint: dict[str, Any] | None = None,
        time_step: float = 0.05,
        planning_time: float = 1.0,
        rrt_range: float = 0.1,
        simplify: bool = True,
        ik_threshold: float = 1e-3,
        ik_pos_threshold: float | None = None,
        ik_rot_threshold: float = float(np.deg2rad(0.5)),
        ik_random_seed_count: int = 16,
        max_goal_candidates: int = 8,
        verbose: bool = False,
    ) -> dict[str, Any]:
        final_pos_threshold = ik_threshold if ik_pos_threshold is None else ik_pos_threshold
        if final_pos_threshold <= 0.0 or ik_rot_threshold <= 0.0:
            raise ValueError("final pose validation thresholds must be positive.")

        goal_qpos_candidates = self.ik_world(
            goal_pose_world,
            current_qpos,
            threshold=ik_threshold,
            pos_threshold=ik_pos_threshold,
            rot_threshold=ik_rot_threshold,
            random_seed_count=ik_random_seed_count,
            max_candidates=max_goal_candidates,
            score_mode="continuous",
            verbose=verbose,
        )
        if len(goal_qpos_candidates) == 0:
            return {"status": "IK Failed", "mode_used": "ik", "debug": {"ik_candidates": 0}}

        constraint_kwargs = {}
        simplify_for_plan = simplify
        if constraint is not None:
            constraint_function = constraint.get("function")
            constraint_jacobian = constraint.get("jacobian")
            if constraint_function is None or constraint_jacobian is None:
                raise ValueError("constraint must provide both 'function' and 'jacobian'.")

            constraint_tolerance = float(constraint.get("tolerance", 1e-3))
            if constraint_tolerance <= 0.0:
                raise ValueError("constraint tolerance must be positive.")

            constraint_kwargs = {
                "constraint_function": constraint_function,
                "constraint_jacobian": constraint_jacobian,
                "constraint_tolerance": constraint_tolerance,
            }
            # MPlib/OMPL constrained planning does not support path simplification.
            simplify_for_plan = False

        best_result: dict[str, Any] | None = None
        plan_attempts = 0
        plan_successes = 0
        path_rejects = 0
        last_reject_reason = None

        for goal_qpos in goal_qpos_candidates:
            distance = float(np.sum(IK_WEIGHTS * (goal_qpos - current_qpos) ** 2))
            margin = float(np.min(np.minimum(goal_qpos - self.joint_limits[:, 0], self.joint_limits[:, 1] - goal_qpos)))
            branch_score = 0.0
            if self.last_goal_qpos is not None:
                branch_score = 0.02 * float(np.sum(np.abs(goal_qpos - self.last_goal_qpos)))
            ik_score = (branch_score, distance, -margin)

            if not self.validate_qpos_limits(goal_qpos):
                path_rejects += 1
                last_reject_reason = "goal_joint_limit"
                continue

            plan_attempts += 1
            plan = self.planner.plan_qpos(
                goal_qposes=[goal_qpos.copy()],
                current_qpos=current_qpos,
                time_step=time_step,
                rrt_range=rrt_range,
                planning_time=planning_time,
                simplify=simplify_for_plan,
                verbose=verbose,
                **constraint_kwargs,
            )
            if plan.get("status", "") != "Success":
                last_reject_reason = plan.get("status", "plan_qpos_failed")
                continue

            plan_successes += 1
            timed, timing_debug = self.build_timed_path(plan, current_qpos, time_step, verbose=verbose)
            path = timed["qpos"]
            final_qpos = path[-1].copy()
            valid, path_score, debug = evaluate_path(path, current_qpos, final_qpos, self.last_goal_qpos)
            debug.update(timing_debug)
            final_pose_valid, final_pos_err, final_rot_err = self.validate_final_pose(
                final_qpos,
                goal_pose_world,
                final_pos_threshold,
                ik_rot_threshold,
            )
            debug.update({
                "final_pos_err": round(final_pos_err, 8),
                "final_rot_err_deg": round(float(np.rad2deg(final_rot_err)), 6),
            })
            if not valid:
                path_rejects += 1
                last_reject_reason = debug.get("reason")
                continue
            if not final_pose_valid:
                path_rejects += 1
                last_reject_reason = "final_pose_error"
                continue

            result = {
                "status": "Success",
                "mode_used": "qpos",
                "qpos": timed["qpos"],
                "qvel": timed["qvel"],
                "qacc": timed["qacc"],
                "time": timed["time"],
                "duration": timed["duration"],
                "goal_qpos": final_qpos,
                "ik_score": ik_score,
                "path_score": path_score,
                "debug": debug,
            }
            if best_result is None or (result["path_score"], result["ik_score"]) < (best_result["path_score"], best_result["ik_score"]):
                best_result = result

        plan_debug = {
            "ik_candidates": len(goal_qpos_candidates),
            "qpos_plan_attempts": plan_attempts,
            "qpos_plan_successes": plan_successes,
            "path_rejects": path_rejects,
            "last_reject_reason": last_reject_reason,
            "simplify_used": simplify_for_plan,
            "constraint_used": constraint is not None,
        }
        if best_result is None:
            return {"status": "Planning Failed", "mode_used": "qpos", "debug": plan_debug}

        best_result["debug"].update(plan_debug)
        self.last_goal_qpos = best_result["goal_qpos"].copy()
        return best_result

    def plan_pose_world(
        self,
        goal_pose_world: mp.Pose,
        current_qpos: np.ndarray,
        constraint: dict[str, Any] | None = None,
        time_step: float = 0.05,
        planning_time: float = 1.0,
        rrt_range: float = 0.1,
        simplify: bool = True,
        ik_threshold: float = 1e-3,
        ik_pos_threshold: float | None = None,
        ik_rot_threshold: float = float(np.deg2rad(0.5)),
        ik_random_seed_count: int = 16,
        max_goal_candidates: int = 8,
        screw_qpos_step: float = 0.1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        current_qpos = self.canonicalize_qpos(current_qpos)
        current_state_error = self.check_current_state(current_qpos)
        if current_state_error is not None:
            return current_state_error

        final_pos_threshold = ik_threshold if ik_pos_threshold is None else ik_pos_threshold
        if final_pos_threshold <= 0.0 or ik_rot_threshold <= 0.0:
            raise ValueError("final pose validation thresholds must be positive.")

        pos_err, rot_err = pose_error(self.fk_world(current_qpos), goal_pose_world)

        near_result = self.plan_near_pose_world(
            goal_pose_world=goal_pose_world,
            current_qpos=current_qpos,
            pos_err=pos_err,
            rot_err=rot_err,
            constraint=constraint,
            time_step=time_step,
            screw_qpos_step=screw_qpos_step,
            final_pos_threshold=final_pos_threshold,
            final_rot_threshold=ik_rot_threshold,
            verbose=verbose,
        )
        if near_result is not None:
            return near_result

        return self.plan_qpos_candidates_world(
            goal_pose_world=goal_pose_world,
            current_qpos=current_qpos,
            constraint=constraint,
            time_step=time_step,
            planning_time=planning_time,
            rrt_range=rrt_range,
            simplify=simplify,
            ik_threshold=ik_threshold,
            ik_pos_threshold=ik_pos_threshold,
            ik_rot_threshold=ik_rot_threshold,
            ik_random_seed_count=ik_random_seed_count,
            max_goal_candidates=max_goal_candidates,
            verbose=verbose,
        )
