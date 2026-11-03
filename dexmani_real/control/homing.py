"""检查当前姿态到配置 HOME 的直接路径。"""

from __future__ import annotations

import numpy as np

from dexmani_real.config.control import EnvironmentConfig, HomeConfigurationError
from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.control.collision import CollisionModel
from dexmani_real.robot.model import ARM_JOINT_SHAPE


class HomePathRejected(ValueError):
    """直接路径几何拒绝；需要操作者调整姿态。"""


class HomePlanner:
    """使用已由 load_table_plane 准入的桌面平面规划 HOME 路径。"""

    def __init__(
        self,
        environment: EnvironmentConfig,
        arm: ArmConfig,
        *,
        table_plane_abcd: tuple[float, float, float, float],
    ) -> None:
        try:
            environment.table.validate()
        except (TypeError, ValueError) as exc:
            raise HomeConfigurationError(f"HOME table configuration: {exc}") from exc
        self.arm = arm
        self.table_plane_abcd = table_plane_abcd
        self.limits = np.column_stack((arm.joint_limit_lower, arm.joint_limit_upper))
        margin = environment.eef_workspace_margin_m
        self.workspace_bounds = environment.workspace.as_array() + (-margin, margin)
        self.collision_model = CollisionModel(
            static_boxes=environment.static_boxes,
            table=environment.table,
            table_plane_abcd=table_plane_abcd,
        )

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.collision_model.set_hand_qpos(hand_qpos)

    def plan(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        current, target = (np.asarray(q, dtype=np.float64) for q in (current, target))
        if any(q.shape != ARM_JOINT_SHAPE or not np.isfinite(q).all() for q in (current, target)):
            raise ValueError("HOME states must be finite shape (7,)")
        path = np.stack((current, target))
        if np.any(path < self.limits[:, 0]) or np.any(
            path > self.limits[:, 1]
        ):
            raise HomePathRejected("home path violates operational joint limits")

        steps = int(np.ceil(np.max(np.abs(target - current)) / np.deg2rad(1.0)))
        for index, waypoint in enumerate(np.linspace(current, target, steps + 1)):
            clearance, distance = self._check_state(waypoint)
            if clearance < 0:
                raise HomePathRejected(
                    f"table_clearance at waypoint {index}: clearance={clearance:g} m, "
                    f"distance={distance:g} m"
                )
        if np.max(np.abs(target - current)) <= self.arm.homing.convergence_rad:
            return np.empty((0, 7), dtype=np.float64)
        return path

    def _check_state(self, qpos):
        reason = self.collision_model.rejection(np.asarray(qpos))
        if reason is not None:
            raise HomePathRejected(f"collision: {reason}")
        position = self.collision_model.eef_position()
        if np.any(position < self.workspace_bounds[:, 0]) or np.any(
            position > self.workspace_bounds[:, 1]
        ):
            raise HomePathRejected(f"path leaves workspace: {position}")
        distance = float(self.collision_model.minimum_table_distance())
        clearance = distance - float(self.collision_model.table.soft_clearance_m)
        return clearance, distance
