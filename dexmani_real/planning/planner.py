"""xArm7 motion planner with MPlib backend — IK, path planning, collision checks."""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

from .collision_model import CollisionModel
from .desk_safety import FingertipDeskSafety
from .ik import TeleopIKSolver
from .ik_candidates import IKCandidateManager
from .kinematics import XArm7Kinematics
from .pose_utils import compute_pose_error, ensure_qpos
from .types import IKResult, PathResult, PlanningProfile, Pose, TeleopProfile, XArm7PlannerConfig
from .workspace_safety import WorkspaceSafety

__all__ = [
    "XArm7MotionPlanner",
]

# Path scoring weights (Phase 4.2)
_PATH_SCORE_JOINT_LENGTH_WEIGHT = 1.0
_PATH_SCORE_WAYPOINT_DELTA_WEIGHT = 2.0
_PATH_SCORE_EEF_EFFICIENCY_WEIGHT = 3.0


class XArm7MotionPlanner:
    """Arm-only xArm7 motion planner with MPlib backend.

    Three internal subsystems:
      - ``kin`` (:class:`XArm7Kinematics`): FK / Jacobian / pose transforms.
      - ``ik_mgr`` (:class:`IKCandidateManager`): IK candidate generation,
        filtering, scoring, canonicalization.
      - ``mp_planner`` (:class:`mplib.Planner`): raw MPlib plan_screw /
        plan_qpos calls.

    Public API (prefer these over direct subsystem access):
      - ``solve_ik`` / ``solve_teleop_ik`` — single-shot IK suitable for teleop.
      - ``plan_path`` — multi-strategy path planning (screw → RRT).
      - ``compute_eef_pose_world`` / ``compute_eef_jacobian`` — FK queries.
      - ``has_self_collision`` / ``has_env_collision`` — collision queries.
    """

    def __init__(
        self,
        config: XArm7PlannerConfig,
        planning_profile: PlanningProfile | None = None,
        teleop_profile: TeleopProfile | None = None,
        hand_dof: bool = True,
    ) -> None:
        import mplib

        self.mplib = mplib
        self.config = config
        self.planning_profile = planning_profile or PlanningProfile()
        self.teleop_profile = teleop_profile or TeleopProfile()

        joint_vel_limits = np.deg2rad(np.asarray(config.joint_vel_limits_deg, dtype=np.float64))
        joint_acc_limits = joint_vel_limits * float(config.joint_acc_scale)
        if not os.path.exists(config.srdf_path):
            mplib.urdf_utils.generate_srdf(config.urdf_path, config.srdf_path)

        self.mplib_planner = self.mplib.Planner(
            urdf=str(config.urdf_path),
            srdf=str(config.srdf_path),
            move_group=config.eef_link_name,
            use_convex=config.use_convex,
            joint_vel_limits=joint_vel_limits.tolist(),
            joint_acc_limits=joint_acc_limits.tolist(),
        )
        self.pinocchio_model = self.mplib_planner.pinocchio_model

        link_names = list(self.pinocchio_model.get_link_names())
        if config.eef_link_name not in link_names:
            raise ValueError(f"Link {config.eef_link_name!r} not found. Available links: {link_names}")
        eef_link_id = int(link_names.index(config.eef_link_name))

        self._elbow_joint_index = list(self.pinocchio_model.get_joint_names()).index("joint4")
        joint_limits = np.asarray(self.mplib_planner.joint_limits, dtype=np.float64)
        dof = int(joint_limits.shape[0])
        equivalent_joint_mask = (joint_limits[:, 1] - joint_limits[:, 0]) >= np.pi

        base_pose_world = config.base_pose_world.copy()
        self.workspace_safety: WorkspaceSafety | None = (
            WorkspaceSafety(config.workspace_bounds) if config.workspace_bounds is not None else None
        )

        self.kin = XArm7Kinematics(
            mp_planner=self.mplib_planner,
            pinocchio_model=self.pinocchio_model,
            eef_link_id=eef_link_id,
            dof=dof,
            joint_limits=joint_limits,
            equivalent_joint_mask=equivalent_joint_mask,
            base_pose_world=base_pose_world,
            mplib=self.mplib,
        )
        self.collision_model = CollisionModel(hand_dof=hand_dof, collision_config=config.collision)
        self.ik_mgr = IKCandidateManager(self.kin, collision_model=self.collision_model)
        self.mplib_planner.set_base_pose(self.kin.to_mplib_pose(base_pose_world))

        self.teleop_solver = TeleopIKSolver(self.kin, self.ik_mgr, self.teleop_profile)

        # Convenience aliases (used by teleop_solver and external code)
        self.dof = dof
        self.joint_limits = joint_limits
        self.equivalent_joint_mask = equivalent_joint_mask

        # Geometric FK desk safety — fingertip Z vs desk
        self.desk_safety: FingertipDeskSafety | None = None
        if config.collision is not None:
            try:
                self.desk_safety = FingertipDeskSafety(
                    collision_model=self.collision_model,
                    collision_config=config.collision,
                )
            except (ValueError, RuntimeError, IndexError):
                logger.warning("FingertipDeskSafety init failed — desk FK checks disabled.", exc_info=True)
                # desk_safety remains None — desk FK checks skipped

    def __getattr__(self, name: str):
        """Proxy passthrough methods to self.kin, self.ik_mgr, or self.mplib_planner.

        Eliminates 24 pure-delegation methods (ref: code-simplification-review).
        Callers use ``planner.compute_eef_pose_world(q)`` as before — the proxy
        routes to ``self.kin.compute_eef_pose_world(q)`` transparently.

        Only fires when normal attribute lookup fails (i.e. the method is not
        defined on XArm7MotionPlanner directly).  ``self.kin``, ``self.ik_mgr``,
        and ``self.mplib_planner`` are regular attributes set in ``__init__`` and
        are never proxied.
        """
        for delegate in (self.kin, self.ik_mgr, self.mplib_planner):
            if hasattr(delegate, name):
                return getattr(delegate, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        """Set current hand joint configuration for 19-DOF collision detection.

        In ``hand_dof`` mode, the CollisionModel auto-expands 7-DOF arm qpos to
        19-DOF by concatenating with this buffer.  Call each teleop frame before
        arm IK to keep the hand geometry up-to-date for collision checks.

        No-op in 7-DOF mode.
        """
        self.collision_model.set_hand_qpos(hand_qpos)

    # --- Public API ---

    def set_base_pose(self, base_pose_world: Pose) -> None:
        self.kin.set_base_pose(base_pose_world)

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

        # Diagnostic: when hand_dof=True, CollisionModel auto-expands arm qpos
        # with _hand_qpos.  Warn if the buffer was never initialized — all-zero
        # is ambiguous (could be a valid home pose), so use the dedicated flag.
        if self.collision_model.hand_dof and not self.collision_model._hand_qpos_initialized:
            logger.warning(
                "plan_path: _hand_qpos was never set — "
                "CollisionModel env/self checks use zero (open-hand) pose.  "
                "Call set_hand_qpos() before plan_path() to sync."
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
                reason = result.reason or "unknown"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
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

    # ── Pass-through delegates ──
    # All kinematics, IK candidate, and collision-check methods are proxied via
    # __getattr__ to self.kin / self.ik_mgr / self.mplib_planner.
    # See __getattr__ docstring for rationale.
    #
    # Explicit delegates retained only for methods that add semantic value:
    #   - solve_ik / solve_teleop_ik / plan_path  (planning orchestration)
    #   - set_base_pose  (coordinates kin + mp_planner)
    #
    # Direct passthrough callers use the same syntax as before (e.g.
    # planner.compute_eef_pose_world(q)), now routed via __getattr__.

    def try_screw_plan(
        self, target_eef_pose_world: Pose, current_qpos: np.ndarray, profile: PlanningProfile
    ) -> PathResult:
        result = self.mplib_planner.plan_screw(
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
                result = self.mplib_planner.plan_qpos(
                    goal_qposes=goal_qposes,
                    current_qpos=current_qpos,
                    time_step=profile.path_dt,
                    rrt_range=rrt_range,
                    planning_time=profile.rrt_time_limit,
                    simplify=profile.simplify_path,
                    verbose=False,  # MPlib verbose=False suppresses debug output (P3.2)
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
            warnings.warn(f"MPlib {source} returned success but empty position; falling back to current_qpos.")
            path = current_qpos.reshape(1, -1)
        path_result = self.validate_path(path, target_eef_pose_world, current_qpos, source=source, profile=profile)
        path_result.report.update(mplib_status=status)
        return path_result

    # ── Path validation (internal) ──

    def shortcut_smooth_path(self, path: np.ndarray, current_qpos: np.ndarray, profile: PlanningProfile) -> np.ndarray:
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
                if self._is_shortcut_valid(prev, nxt, limits, profile):
                    path = np.delete(path, idx, axis=0)
                    changed = True
                else:
                    idx += 1
            if not changed:
                break
        return path

    def _is_shortcut_valid(
        self,
        prev: np.ndarray,
        nxt: np.ndarray,
        limits: np.ndarray,
        profile: PlanningProfile,
    ) -> bool:
        """Check if the direct prev→nxt shortcut segment is collision-free.

        Checks three points along the segment (¼, ½, ¾) for self-collision
        and environment collision.  Joint limits are only checked at the
        midpoint (limit bounds are convex, so midpoint-outside implies the
        segment is problematic).
        """
        # Joint limits check at midpoint (convex → midpoint suffices)
        mid = 0.5 * (prev + nxt)
        outside, _ = self.limit_violation(mid, limits)
        if np.any(outside):
            return False

        # Sample three points (α = ¼, ½, ¾) along prev→nxt
        diff = nxt - prev
        q_quarter = prev + 0.25 * diff
        q_three_quarter = prev + 0.75 * diff
        samples = [q_quarter, mid, q_three_quarter]

        if profile.check_self_collision:
            for q in samples:
                if self.ik_mgr.has_self_collision(q):
                    return False
        if profile.check_env_collision:
            for q in samples:
                if self.ik_mgr.has_env_collision(q):
                    return False
        # Geometric FK desk safety — check the segment endpoints
        if self.desk_safety is not None and profile.check_env_collision:
            seg = np.array([prev, nxt], dtype=np.float64)
            desk_safe, _min_z, _viol_idx = self.desk_safety.check_path_desk_safety(seg)
            if not desk_safe:
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
        """Validate a planned path through a chain of independent checks.

        Each check returns None on pass or a failure PathResult.  Checks are
        ordered by cost (cheapest first) to fail fast.

        If the shortcut-smoothed path fails validation, the original unsmoothed
        path is retried as a fallback.  Shortcut smoothing can create waypoint
        gaps larger than max_waypoint_delta_deg when the arm is far from the
        target (e.g. return-to-home from a stretched pose).
        """
        # ── Preprocessing ──
        try:
            path = self.snap_path_to_nearest_equivalent(path, current_qpos)
            path = self.canonicalize_path_to_planning_limits(path, current_qpos, profile)
            path_before_smooth = path.copy()
            path = self.shortcut_smooth_path(path, current_qpos, profile)
        except ValueError as error:
            return PathResult(success=False, qpos_path=None, source=source, reason=str(error))

        # ── Try smoothed path first; fall back to unsmoothed on failure ──
        smoothed_failure_reason: str | None = None
        for attempt_label, candidate in (
            ("smoothed", path),
            ("unsmoothed", path_before_smooth),
        ):
            if attempt_label == "unsmoothed" and len(candidate) == len(path):
                # Shortcut smoothing didn't change the path — no point retrying
                break

            report = self.compute_path_metrics(candidate, target_eef_pose_world, current_qpos, profile)
            failure = None
            for check in (
                self._check_limit_violation,
                self._check_elbow_consistency,
                self._check_start_distance,
                self._check_waypoint_delta,
                self._check_terminal_pose,
                self._check_self_collision,
                self._check_env_collision,
                self._check_workspace_bounds,
                self._check_desk_safety,
            ):
                failure = check(candidate, report, source, profile)
                if failure is not None:
                    break

            if failure is None:
                # ── All checks passed ──
                report["path_score"] = float(
                    _PATH_SCORE_JOINT_LENGTH_WEIGHT * report.get("joint_path_length", 0.0)
                    + _PATH_SCORE_WAYPOINT_DELTA_WEIGHT * report.get("max_waypoint_delta_rad", 0.0)
                    + _PATH_SCORE_EEF_EFFICIENCY_WEIGHT * (1.0 - report.get("eef_efficiency", 1.0))
                )
                if attempt_label == "unsmoothed":
                    logger.debug(
                        "validate_path: smoothed path failed (%s), unsmoothed fallback passed "
                        "(%d waypoints, max_delta=%.1f°)",
                        smoothed_failure_reason or "?",
                        len(candidate),
                        report.get("max_waypoint_delta_deg", 0),
                    )
                return PathResult(success=True, qpos_path=candidate, source=source, report=report)

            if attempt_label == "smoothed":
                smoothed_failure_reason = failure.reason

        return failure  # type: ignore[return-value]  # both attempts failed

    # ── Path validators (each returns None on pass, PathResult on failure) ──

    @staticmethod
    def _make_failure(reason: str, source: str, report: dict) -> PathResult:
        return PathResult(success=False, qpos_path=None, source=source, reason=reason, report=report)

    def _check_limit_violation(self, _path, report, source, _profile):
        if report.get("limit_violation"):
            return self._make_failure("Path violates planning limits.", source, report)
        return None

    def _check_elbow_consistency(self, path, report, source, _profile):
        has_flip, flip_info = self.check_elbow_consistency(path)
        if has_flip:
            report.update(flip_info)
            return self._make_failure("Elbow branch flip detected.", source, report)
        return None

    def _check_start_distance(self, _path, report, source, profile):
        if report["start_qpos_error_rad"] > np.deg2rad(profile.max_waypoint_delta_deg) + 1e-12:
            return self._make_failure("Path start is too far from current_qpos.", source, report)
        return None

    def _check_waypoint_delta(self, _path, report, source, profile):
        if report["max_waypoint_delta_rad"] > np.deg2rad(profile.max_waypoint_delta_deg) + 1e-12:
            return self._make_failure("Path waypoint delta too large.", source, report)
        return None

    def _check_terminal_pose(self, _path, report, source, profile):
        if (
            report["terminal_pos_error_m"] > profile.max_pose_error_pos_m
            or report["terminal_rot_error_rad"] > profile.max_pose_error_rot_rad
        ):
            return self._make_failure("Terminal pose error too large.", source, report)
        return None

    def _check_self_collision(self, path, report, source, profile):
        if not profile.check_self_collision:
            return None
        collision_report = self.check_path_collisions(path)
        report.update(collision_report)
        if collision_report.get("path_self_collision"):
            collision_dict = collision_report.get("collision")
            reason = "Path contains self-collision."
            if isinstance(collision_dict, dict) and collision_dict.get("in_collision"):
                num = collision_dict.get("num_contacts", 0)
                pairs = collision_dict.get("collision_pairs", [])
                link_names = ", ".join(f"{p['link1']}↔{p['link2']}" for p in pairs[:5])
                if len(pairs) > 5:
                    link_names += f" ... +{len(pairs) - 5} more"
                reason = f"Path contains self-collision ({num} contact(s): {link_names})."
            return self._make_failure(reason, source, report)
        return None

    def _check_env_collision(self, path, report, source, profile):
        if not profile.check_env_collision:
            return None
        env_collision_report = self.check_path_env_collisions(path)
        report.update(env_collision_report)
        if env_collision_report.get("path_env_collision"):
            return self._make_failure("Path contains environment collision.", source, report)
        return None

    def _check_workspace_bounds(self, path, report, source, _profile):
        if self.workspace_safety is None:
            return None
        eef_positions = np.array([self.compute_eef_pose_world(q).p for q in path], dtype=np.float64)
        # Skip waypoint 0 (start position) — the arm may already be outside
        # workspace; the path is valid as long as subsequent waypoints are safe.
        for i, eef_p in enumerate(eef_positions):
            if i == 0:
                continue
            if not self.workspace_safety.check(eef_p):
                violations = []
                for ax in range(3):
                    if eef_p[ax] < self.workspace_safety.bounds[ax, 0]:
                        violations.append(
                            f"axis={'XYZ'[ax]} val={eef_p[ax]:.3f} < {self.workspace_safety.bounds[ax, 0]:.3f}"
                        )
                    elif eef_p[ax] > self.workspace_safety.bounds[ax, 1]:
                        violations.append(
                            f"axis={'XYZ'[ax]} val={eef_p[ax]:.3f} > {self.workspace_safety.bounds[ax, 1]:.3f}"
                        )
                report["workspace_violation_index"] = i
                report["workspace_violation_summary"] = "; ".join(violations)
                return self._make_failure(
                    f"Path contains workspace violations: waypoint[{i}] ({'; '.join(violations)})",
                    source,
                    report,
                )
        return None

    def _check_desk_safety(self, path, report, source, profile):
        if self.desk_safety is None or not profile.check_env_collision:
            return None
        desk_safe, min_z, viol_idx = self.desk_safety.check_path_desk_safety(path)
        report["desk_safety_min_fingertip_z"] = float(min_z)
        report["desk_safety_violation_index"] = int(viol_idx)
        if not desk_safe:
            return self._make_failure(
                f"Path contains desk collision (fingertip z_min={min_z:.3f}m < "
                f"safe={self.desk_safety.config.fingertip_threshold:.3f}m, segment {viol_idx})",
                source,
                report,
            )
        return None

    def compute_path_metrics(
        self, path: np.ndarray, target_eef_pose_world: Pose, current_qpos: np.ndarray, profile: PlanningProfile
    ) -> dict[str, Any]:
        diff = np.diff(path, axis=0) if len(path) > 1 else np.zeros((0, self.dof), dtype=np.float64)
        max_step = float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0
        path_length = float(np.sum(np.linalg.norm(diff, axis=1))) if len(diff) > 0 else 0.0
        terminal_pos_error, terminal_rot_error = self.compute_world_pose_error(target_eef_pose_world, path[-1])

        eef_efficiency = 1.0
        if len(path) >= 3:
            eef_positions = np.array([self.compute_eef_pose_world(q).p for q in path], dtype=np.float64)
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
