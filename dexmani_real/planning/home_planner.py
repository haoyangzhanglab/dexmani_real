"""Bounded HOME planning; final path admission remains fail-closed."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np

from dexmani_real.config.environment import StaticCollisionBox, TableCollisionConfig
from .collision import CollisionModel
from .kinematics.arm_fk import make_arm_kinematics
from .kinematics.pose import compute_pose_error
from .paths import WORKSPACE_BOUNDS_TOLERANCE_M, interpolate_waypoints

logger = logging.getLogger(__name__)

RRT_TIME_S = 0.5
RRT_RANGE = 0.12
PATH_DT = 1 / 15
MAX_WAYPOINT_DELTA_RAD = np.deg2rad(15.0)
MAX_POSE_ERROR_POS_M = 0.005
MAX_POSE_ERROR_ROT_RAD = 0.05


class HomePlanner:
    """Nominal-URDF arm kinematics plus the 19-DOF arm/hand collision model."""

    def __init__(
        self,
        *,
        workspace_bounds: np.ndarray | None = None,
        static_boxes: Iterable[StaticCollisionBox] = (),
        table: TableCollisionConfig | None = None,
    ) -> None:
        self.workspace_bounds = None
        if workspace_bounds is not None:
            bounds = np.asarray(workspace_bounds, dtype=np.float64)
            if (
                bounds.shape != (3, 2)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])
            ):
                raise ValueError("workspace_bounds must be finite shape (3, 2) with lower <= upper")
            self.workspace_bounds = bounds.copy()
        self.kin = make_arm_kinematics()
        self.elbow_joint_index = list(self.kin.pinocchio_model.get_joint_names()).index("joint4")
        self.collision_model = CollisionModel(static_boxes=static_boxes, table=table)

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.collision_model.set_hand_qpos(hand_qpos)

    def plan_joint_qpos_path(self, target_qpos: np.ndarray, current_qpos: np.ndarray) -> np.ndarray:
        """Bounded HOME RRT fallback; reject malformed or unsafe output with ValueError."""
        current_qpos = np.asarray(current_qpos, dtype=np.float64)
        target_qpos = np.asarray(target_qpos, dtype=np.float64)
        if any(q.shape != (7,) or not np.isfinite(q).all() for q in (current_qpos, target_qpos)):
            raise ValueError("HOME planning requires finite (7,) joint vectors")
        target_qpos = self.kin.canonicalize_qpos(target_qpos, current_qpos)
        target_pose = self.kin.compute_eef_pose_base(target_qpos)
        result = self.kin.mp_planner.plan_qpos(
            goal_qposes=[target_qpos],
            current_qpos=current_qpos,
            time_step=PATH_DT,
            rrt_range=RRT_RANGE,
            planning_time=RRT_TIME_S,
            simplify=True,
            verbose=False,
        )
        if not isinstance(result, dict) or result.get("status") != "Success":
            raise ValueError(f"MPlib HOME planning failed: {result}")
        return self.validate_rrt_path(result.get("position", []), target_pose, current_qpos)

    @staticmethod
    def outside_limits(qpos, limits):
        return bool(np.any(qpos < limits[:, 0] - 1e-5) or np.any(qpos > limits[:, 1] + 1e-5))

    def shortcut_smooth_path(self, path, limits):
        path = path.copy()
        for _ in range(3):
            changed = False
            index = 1
            while index < len(path) - 1:
                prev, nxt = path[index - 1], path[index + 1]
                if self.is_shortcut_valid(prev, nxt, limits):
                    path = np.delete(path, index, axis=0)
                    changed = True
                else:
                    index += 1
            if not changed:
                break
        return path

    def is_shortcut_valid(self, prev, nxt, limits):
        """Probe shortcuts at ¼, ½ and ¾; final admission densely checks the whole path."""
        mid = 0.5 * (prev + nxt)
        if self.outside_limits(mid, limits):
            return False
        delta = nxt - prev
        try:
            return not any(
                self.collision_model.check_collision(q)
                for q in (
                    prev + 0.25 * delta,
                    mid,
                    prev + 0.75 * delta,
                )
            )
        except Exception:
            logger.warning("shortcut collision check failed closed", exc_info=True)
            return False

    def validate_rrt_path(self, path, target_pose, current_qpos):
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 7 or len(path) == 0 or not np.isfinite(path).all():
            raise ValueError("MPlib HOME path must be nonempty finite shape (N, 7)")
        path = self.kin.canonicalize_path(path, current_qpos)
        limits = self.kin.planning_limits(current_qpos)
        smoothed = self.shortcut_smooth_path(path, limits)
        candidates = (smoothed, path) if len(smoothed) != len(path) else (smoothed,)
        for candidate in candidates:
            try:
                self.check_rrt_candidate(candidate, target_pose, current_qpos, limits)
            except ValueError as exc:
                reason = str(exc)
            else:
                return candidate
        raise ValueError(reason)

    def check_rrt_candidate(self, path, target_pose, current_qpos, limits):
        if self.outside_limits(path, limits):
            raise ValueError("Path violates planning limits.")
        elbow = path[:, self.elbow_joint_index]
        if (
            np.min(elbow) < np.deg2rad(-5.0)
            and np.max(elbow) > np.deg2rad(15.0)
            and np.ptp(elbow) > np.deg2rad(45.0)
        ):
            raise ValueError("Elbow branch flip detected.")
        if (
            np.max(np.abs(self.kin.compute_qpos_delta(path[0], current_qpos)))
            > MAX_WAYPOINT_DELTA_RAD + 1e-12
        ):
            raise ValueError("Path start is too far from current_qpos.")
        if (
            len(path) > 1
            and np.max(np.abs(np.diff(path, axis=0))) > MAX_WAYPOINT_DELTA_RAD + 1e-12
        ):
            raise ValueError("Path waypoint delta too large.")
        pos_error, rot_error = compute_pose_error(
            target_pose, self.kin.compute_eef_pose_base(path[-1])
        )
        if (
            not np.isfinite((pos_error, rot_error)).all()
            or pos_error > MAX_POSE_ERROR_POS_M
            or rot_error > MAX_POSE_ERROR_ROT_RAD
        ):
            raise ValueError("Terminal pose error too large.")
        self.check_workspace_path(path)
        try:
            collision = self.collision_model.check_path(path)
        except Exception as exc:
            raise ValueError(f"Path collision check failed closed: {exc}") from exc
        if collision is not None:
            index, info = collision
            raise ValueError(f"Path collision at segment {index}: {info}")

    def check_workspace_path(self, path: np.ndarray) -> None:
        """Check EEF workspace at joint intervals of at most 0.02 rad; reject failed FK."""
        if self.workspace_bounds is None:
            return
        for index, qpos in enumerate(interpolate_waypoints(path, max_step=0.02)):
            try:
                position = self.kin.compute_eef_pose_base(qpos).p
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f"workspace FK failed at sample {index}: {exc}") from exc
            if (
                not np.isfinite(position).all()
                or np.any(position < self.workspace_bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M)
                or np.any(position > self.workspace_bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M)
            ):
                raise ValueError(f"Path leaves configured workspace at sample {index}: {position}")
