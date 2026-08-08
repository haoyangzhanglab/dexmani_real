"""xArm7 motion planner with MPlib backend — IK, path planning, collision checks."""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from dexmani_real.utils.log import ThrottledWarner, get_logger

logger = get_logger(__name__)

_warn_hand_qpos_unset_teleop = ThrottledWarner(interval_s=30.0)

from .collision_model import CollisionModel
from .ik import TeleopIKSolver
from .ik_candidates import IKCandidateManager, is_mplib_success
from .kinematics import XArm7Kinematics
from .path_utils import interpolate_waypoints
from .pose_utils import compute_pose_error, ensure_qpos
from .types import IKResult, PathResult, PlanningProfile, Pose, TeleopProfile, XArm7PlannerConfig

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
      - ``solve_teleop_ik`` — single-shot IK for teleop.
      - ``plan_path`` — multi-strategy path planning (screw → RRT).
      - ``compute_eef_pose_world`` / ``compute_eef_jacobian`` — FK queries.
      - ``has_self_collision`` — collision queries.
    """

    def __init__(
        self,
        config: XArm7PlannerConfig,
        planning_profile: PlanningProfile | None = None,
        teleop_profile: TeleopProfile | None = None,
        hand_dof: bool = True,
        home_qpos: np.ndarray | None = None,
    ) -> None:
        import mplib

        self.mplib = mplib
        self.config = config
        self.planning_profile = planning_profile or PlanningProfile()
        self.teleop_profile = teleop_profile or TeleopProfile()
        self.workspace_bounds = None
        if config.workspace_bounds is not None:
            bounds = np.asarray(config.workspace_bounds, dtype=np.float64)
            if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("workspace_bounds must be finite shape (3, 2) with lower <= upper")
            self.workspace_bounds = bounds.copy()

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

        # Cross-check URDF (mplib) limits against defaults.arm (Python config).
        # They should be identical; divergence indicates a stale hardcoded copy.
        from dexmani_real.config.defaults import arm as _arm_cfg

        _cfg_lower = np.asarray(_arm_cfg.joint_limit_lower, dtype=np.float64)
        _cfg_upper = np.asarray(_arm_cfg.joint_limit_upper, dtype=np.float64)
        if not np.allclose(joint_limits[:, 0], _cfg_lower, atol=1e-3) or not np.allclose(
            joint_limits[:, 1], _cfg_upper, atol=1e-3
        ):
            logger.warning(
                "URDF joint limits differ from defaults.arm — using URDF values.\n"
                "  URDF lower:  %s\n  defaults:    %s\n"
                "  URDF upper:  %s\n  defaults:    %s",
                joint_limits[:, 0],
                _cfg_lower,
                joint_limits[:, 1],
                _cfg_upper,
            )
        else:
            logger.debug("URDF joint limits match defaults.arm (ok)")

        dof = int(joint_limits.shape[0])
        equivalent_joint_mask = (joint_limits[:, 1] - joint_limits[:, 0]) > 2 * np.pi

        base_pose_world = config.base_pose_world.copy()

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
        # Dual-model design: this Pinocchio CollisionModel validates paths
        # post-hoc with dense interpolation (0.02 rad step), complementing
        # MPlib's internal FCL collision checking during RRT tree expansion.
        # Both use the same SRDF (xarm7_xhand.srdf) so collision pair rules
        # are consistent.  See CLAUDE.md §Collision Detection Layers for the
        # full architecture.
        self.collision_model = CollisionModel(hand_dof=hand_dof)
        self.ik_mgr = IKCandidateManager(self.kin, collision_model=self.collision_model)
        self.mplib_planner.set_base_pose(self.kin.to_mplib_pose(base_pose_world))

        self.teleop_solver = TeleopIKSolver(
            self.kin,
            self.ik_mgr,
            self.teleop_profile,
            elbow_joint_index=self._elbow_joint_index,
            home_qpos=home_qpos,
        )

        # Convenience aliases (used by teleop_solver and external code)
        self.dof = dof
        self.joint_limits = joint_limits
        self.equivalent_joint_mask = equivalent_joint_mask

    @classmethod
    def create_default(
        cls,
        planning_profile: PlanningProfile | None = None,
        teleop_profile: TeleopProfile | None = None,
        home_qpos: np.ndarray | None = None,
    ) -> "XArm7MotionPlanner":
        """Factory with canonical URDF/SRDF and identity base_pose_world.

        Centralises the invariant planner setup shared by keyboard_teleop,
        calibrate_camera, and replay_traj.  Callers pass their own
        *planning_profile* / *teleop_profile* to match their use case (teleop
        tolerances are intentionally looser than the dataclass defaults).
        """
        from pathlib import Path

        _ASSET_DIR = Path(__file__).parent.parent.parent / "assets"
        urdf_path = str(_ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
        srdf_path = str(_ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf")
        cfg = XArm7PlannerConfig(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            base_pose_world=Pose(
                p=np.array([0.0, 0.0, 0.0]),
                q=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        )
        return cls(
            cfg,
            planning_profile=planning_profile or PlanningProfile(),
            teleop_profile=teleop_profile or TeleopProfile(),
            home_qpos=home_qpos,
        )

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

    def set_base_pose(self, base_pose_world: Pose) -> None:
        self.kin.set_base_pose(base_pose_world)

    def solve_teleop_ik(
        self, target_eef_pose_world: Pose, current_qpos: np.ndarray, previous_qpos_cmd: np.ndarray
    ) -> IKResult:
        # Mirror plan_path's hand_qpos guard (line ~227): warn when
        # collision checks will use the fallback open-hand proxy pose.
        if self.collision_model.hand_dof and self.collision_model._hand_qpos is None:
            _warn_hand_qpos_unset_teleop(
                "solve_teleop_ik: hand_qpos not set — "
                "collision checks use home (open-hand) pose. "
                "Call set_hand_qpos() before solve_teleop_ik()."
            )
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
        if self.collision_model.hand_dof and self.collision_model._hand_qpos is None:
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

    # Kinematics, IK candidate, and collision-check methods are proxied via
    # __getattr__ to self.kin / self.ik_mgr / self.mplib_planner.

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
        if not is_mplib_success(status):
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

    # Path validation — each returns None on pass, PathResult on failure.

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
        # Preprocessing
        try:
            path = self.snap_path_to_nearest_equivalent(path, current_qpos)
            path = self.canonicalize_path_to_planning_limits(path, current_qpos, profile)
            path_before_smooth = path.copy()
            path = self.shortcut_smooth_path(path, current_qpos, profile)
        except ValueError as error:
            logger.warning("validate_path preprocessing failed: %s", error, exc_info=True)
            return PathResult(success=False, qpos_path=None, source=source, reason=str(error))

        # Try smoothed path first; fall back to unsmoothed on failure.
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
                self._check_workspace,
                self._check_self_collision,
            ):
                failure = check(candidate, report, source, profile)
                if failure is not None:
                    break

            if failure is None:
                # All checks passed.
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

    # Path validators — each returns None on pass, PathResult on failure.

    @staticmethod
    def _make_failure(reason: str, source: str, report: dict) -> PathResult:
        return PathResult(success=False, qpos_path=None, source=source, reason=reason, report=report)

    def _check_limit_violation(self, _path, report, source, _profile):
        """Fail if the planner report flags a joint limit violation."""
        if report.get("limit_violation"):
            return self._make_failure("Path violates planning limits.", source, report)
        return None

    def _check_elbow_consistency(self, path, report, source, _profile):
        """Fail if the path contains an elbow branch flip (J4 crossing ±360° bands)."""
        has_flip, flip_info = self.check_elbow_consistency(path)
        if has_flip:
            report.update(flip_info)
            return self._make_failure("Elbow branch flip detected.", source, report)
        return None

    def _check_start_distance(self, _path, report, source, profile):
        """Fail if the path start is too far from current joint positions."""
        if report["start_qpos_error_rad"] > np.deg2rad(profile.max_waypoint_delta_deg) + 1e-12:
            return self._make_failure("Path start is too far from current_qpos.", source, report)
        return None

    def _check_waypoint_delta(self, _path, report, source, profile):
        """Fail if any consecutive waypoint step exceeds the delta limit."""
        if report["max_waypoint_delta_rad"] > np.deg2rad(profile.max_waypoint_delta_deg) + 1e-12:
            return self._make_failure("Path waypoint delta too large.", source, report)
        return None

    def _check_terminal_pose(self, _path, report, source, profile):
        """Fail if the final waypoint EEF pose error exceeds thresholds."""
        if (
            report["terminal_pos_error_m"] > profile.max_pose_error_pos_m
            or report["terminal_rot_error_rad"] > profile.max_pose_error_rot_rad
        ):
            return self._make_failure("Terminal pose error too large.", source, report)
        return None

    def _check_workspace(self, path, report, source, _profile):
        """Fail if any returned waypoint leaves configured world-frame bounds."""
        if self.workspace_bounds is None:
            return None
        dense_path = interpolate_waypoints(path, max_step=0.02)
        for index, qpos in enumerate(dense_path):
            try:
                position = self.compute_eef_pose_world(qpos).p
            except (ValueError, RuntimeError):
                position = np.full(3, np.nan)
            if (
                not np.all(np.isfinite(position))
                or np.any(position < self.workspace_bounds[:, 0])
                or np.any(position > self.workspace_bounds[:, 1])
            ):
                report["workspace_violation_index"] = index
                report["workspace_violation_position_m"] = position.copy()
                return self._make_failure("Path leaves configured workspace.", source, report)
        return None

    def is_workspace_segment_safe(self, start_qpos: np.ndarray, end_qpos: np.ndarray) -> bool:
        """Check a short commanded segment against configured EEF bounds."""
        if self.workspace_bounds is None:
            return True
        path = interpolate_waypoints(np.stack([start_qpos, end_qpos]), max_step=0.02)
        for qpos in path:
            try:
                position = self.compute_eef_pose_world(qpos).p
            except (ValueError, RuntimeError):
                return False
            if not np.all(np.isfinite(position)):
                return False
            if np.any(position < self.workspace_bounds[:, 0]) or np.any(position > self.workspace_bounds[:, 1]):
                return False
        return True

    def _check_self_collision(self, path, report, source, profile):
        """Fail if any waypoint is in self-collision (per SRDF collision pairs)."""
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

    # Elbow branch-flip detection thresholds (path-wide min/max check).
    # The -5°/15° band boundaries are shared with TeleopIKSolver; the span
    # threshold (45°) is deliberately larger than the single-step check (40°)
    # — a path can have a legitimate elbow crossing without any single frame
    # being a "flip."
    _ELBOW_NEG_BAND_RAD: float = np.deg2rad(-5.0)
    _ELBOW_POS_BAND_RAD: float = np.deg2rad(15.0)
    _ELBOW_MIN_SPAN_RAD: float = np.deg2rad(45.0)

    def check_elbow_consistency(self, path: np.ndarray) -> tuple[bool, dict[str, Any]]:
        values = path[:, self._elbow_joint_index]
        v_min, v_max = float(np.min(values)), float(np.max(values))
        span = v_max - v_min
        if v_min < self._ELBOW_NEG_BAND_RAD and v_max > self._ELBOW_POS_BAND_RAD and span > self._ELBOW_MIN_SPAN_RAD:
            return True, {
                "elbow_branch_flip": True,
                "elbow_min_deg": float(np.rad2deg(v_min)),
                "elbow_max_deg": float(np.rad2deg(v_max)),
                "elbow_span_deg": float(np.rad2deg(span)),
            }
        return False, {}
