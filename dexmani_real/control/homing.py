"""HOME 完整候选路径按行程与停稳开销排序，共用几何校验。"""

from __future__ import annotations

from functools import cache
from itertools import pairwise

import numpy as np

from dexmani_real.config.control import EnvironmentConfig
from dexmani_real.config.xarm7 import HOME_DWELL_S, HOME_TARGET_TIMEOUT_S, ArmConfig
from dexmani_real.control.collision import CollisionModel
from dexmani_real.control.constraints import wrap_nearest_equivalent
from dexmani_real.robot.model import ARM_JOINT_SHAPE

PROXIMAL_MASK = np.array([True, True, True, True, False, False, False], dtype=bool)


def home_segment_timeout_s(distance: float, arm: ArmConfig) -> float:
    """静止到静止的运动估计，加既有运动余量、停稳及查询预算。"""
    speed = arm.homing.max_speed_rad_per_s
    acceleration = arm.max_joint_acceleration_rad_per_s2
    nominal = (
        2 * np.sqrt(distance / acceleration)
        if distance <= speed * speed / acceleration
        else distance / speed + speed / acceleration
    )
    return (
        2 * nominal + HOME_TARGET_TIMEOUT_S + HOME_DWELL_S
        + 2 * arm.feedback_max_age_s + 1 / arm.feedback_hz
    )


class HomePathRejected(ValueError):
    """几何拒绝允许尝试其他路径；技术异常直接传播。"""


class HomePlanner:
    def __init__(
        self,
        environment: EnvironmentConfig,
        arm: ArmConfig,
        *,
        table_plane_abcd: tuple[float, float, float, float] | None,
    ) -> None:
        self.arm = arm
        self.environment = environment
        self.limits = np.column_stack((arm.joint_limit_lower, arm.joint_limit_upper))
        margin = environment.eef_workspace_margin_m
        self.workspace_bounds = environment.workspace.as_array() + (-margin, margin)
        self.collision_model = CollisionModel(
            static_boxes=environment.static_boxes,
            table=environment.table,
            table_plane_abcd=table_plane_abcd,
        )
        if self.collision_model.table is None:
            if not np.isfinite(environment.table_z_surface_m):
                raise ValueError("table_z_surface_m must be finite")
            if not np.isfinite(environment.hand_safety_margin_m) or environment.hand_safety_margin_m < 0:
                raise ValueError("hand_safety_margin_m must be finite and non-negative")

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.collision_model.set_hand_qpos(hand_qpos)

    def plan(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        current, target = (np.asarray(q, dtype=np.float64) for q in (current, target))
        if any(q.shape != ARM_JOINT_SHAPE or not np.isfinite(q).all() for q in (current, target)):
            raise ValueError("HOME states must be finite shape (7,)")
        if np.any(np.stack((current, target)) < self.limits[:, 0]) or np.any(
            np.stack((current, target)) > self.limits[:, 1]
        ):
            raise HomePathRejected("home path violates operational joint limits")

        # 缓存只属于本次规划；手部姿态变化后不能复用。路径上的退出软间隙规则仍逐点检查。
        check_state = cache(self._check_state)
        check_state(tuple(current))
        self._check_path(target[None], check_state, allow_table_soft_escape=False)
        if np.max(np.abs(target - current)) <= self.arm.homing.convergence_rad:
            self._check_path(np.stack((current, target)), check_state, allow_table_soft_escape=False)
            return np.empty((0, 7), dtype=np.float64)

        nearest = wrap_nearest_equivalent(target, current, self.limits[:, 0], self.limits[:, 1])
        paths: dict[bytes, np.ndarray] = {}
        for endpoint in (target, nearest):
            for mask in (np.ones(7, dtype=bool), PROXIMAL_MASK, ~PROXIMAL_MASK):
                mid = np.where(mask, endpoint, current)
                path = unique_milestones(current, mid, endpoint, target)
                paths.setdefault(path.tobytes(), path)
        # 行程/速度是时间下界；每段还需要一次停稳确认，不把 SDK 的实际耗时当作已知。
        speed = self.arm.homing.max_speed_rad_per_s
        candidates = sorted(paths.values(), key=lambda p: (
            np.max(np.abs(np.diff(p, axis=0)), axis=1).sum() / speed
            + (len(p) - 1) * HOME_DWELL_S
        ))
        rejections = []
        for path in candidates:
            try:
                self._check_path(path, check_state, allow_table_soft_escape=True)
                return path
            except HomePathRejected as exc:
                rejections.append(str(exc))
        raise HomePathRejected("no safe home path; " + "; ".join(rejections))

    def _check_state(self, qpos):
        reason = self.collision_model.rejection(np.asarray(qpos))
        if reason is not None:
            raise HomePathRejected(f"collision: {reason}")
        position = self.collision_model.eef_position()
        if np.any(position < self.workspace_bounds[:, 0]) or np.any(
            position > self.workspace_bounds[:, 1]
        ):
            raise HomePathRejected(f"path leaves workspace: {position}")
        clearance, measurement, source = self._table_clearance()
        if not np.isfinite(clearance):
            raise RuntimeError("table_check_nonfinite")
        return clearance, measurement, source

    def _table_clearance(self) -> tuple[float, float, str]:
        """返回净间隙、原始测量和测量来源。"""
        collision_model = self.collision_model
        if collision_model.table is not None:
            distance_m = float(collision_model.minimum_table_distance())
            clearance_m = distance_m - float(collision_model.table.soft_clearance_m)
            return clearance_m, distance_m, "calibrated_mesh_distance"
        hand_min_z_m = float(collision_model.minimum_hand_frame_z())
        environment = self.environment
        clearance_m = (
            hand_min_z_m - environment.hand_safety_margin_m - environment.table_z_surface_m
        )
        return clearance_m, hand_min_z_m, "hand_frame_proxy"

    def _check_path(
        self,
        path: np.ndarray,
        check_state,
        *,
        allow_table_soft_escape: bool,
    ) -> None:
        # 候选节点由限位内的端点组合而成，分段线性插值不会越过关节限位。
        path = interpolate_waypoints(path, np.deg2rad(1.0))
        # 先查已有采样的中点，尽早拒绝穿过障碍的长路径；随后仍完整、按序检查。
        check_state(tuple(path[len(path) // 2]))
        best_clearance = -np.inf
        for index, waypoint in enumerate(path):
            clearance, measurement, source = check_state(tuple(waypoint))
            # 退出软间隙前的回退容差相对历史最佳间隙，退出后不能重新进入。
            if clearance < 0.0 and (
                not allow_table_soft_escape
                or best_clearance >= 0.0
                or clearance < best_clearance - 5e-4
                or index == len(path) - 1
            ):
                raise HomePathRejected(
                    f"table_clearance at waypoint {index}: clearance={clearance:g} m, "
                    f"{source}={measurement:g} m"
                )
            best_clearance = max(best_clearance, clearance)


def interpolate_waypoints(path: np.ndarray, max_step: float) -> np.ndarray:
    """插值后每个关节的步长不超过 max_step，单位 rad。"""
    if len(path) <= 1:
        return path.astype(np.float64)
    dense = [path[:1]]
    for a, b in pairwise(path):
        count = int(np.ceil(np.max(np.abs(b - a)) / max_step))
        if count:
            fractions = np.arange(1, count + 1)[:, None] / count
            dense.append(a + fractions * (b - a))
            dense[-1][-1] = b  # 精确保留路点，也让相邻候选复用同一个姿态检查。
    return np.concatenate(dense).astype(np.float64, copy=False)


def unique_milestones(*points: np.ndarray) -> np.ndarray:
    unique = [np.asarray(points[0], dtype=np.float64)]
    for point in points[1:]:
        point = np.asarray(point, dtype=np.float64)
        if float(np.max(np.abs(point - unique[-1]))) > 1e-9:
            unique.append(point)
    return np.asarray(unique, dtype=np.float64)
