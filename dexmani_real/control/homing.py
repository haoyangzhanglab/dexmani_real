"""HOME 候选路径、等价分支对齐与桌面间隙。"""

from __future__ import annotations

import numpy as np

from dexmani_real.control.collision import CollisionModel
from dexmani_real.control.config import EnvironmentConfig
from dexmani_real.control.constraints import (
    equivalent_joint_delta,
    wrap_nearest_equivalent,
)
from dexmani_real.robot.model import ARM_JOINT_SHAPE, XARM7_EQUIVALENT_JOINT_MASK
from dexmani_real.robot.xarm7.config import ArmConfig

PROXIMAL_MASK = np.array([True, True, True, True, False, False, False], dtype=bool)

# 工作空间边缘允许 FK 非线性插值产生 1 mm 外扩。
WORKSPACE_BOUNDS_TOLERANCE_M = 1e-3


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
        self.workspace_bounds = environment.workspace.as_array()
        self.collision_model = CollisionModel(
            static_boxes=environment.static_boxes,
            table=environment.table,
            table_plane_abcd=table_plane_abcd,
        )

    def canonicalize_qpos(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        return wrap_nearest_equivalent(qpos, reference_qpos, self.limits[:, 0], self.limits[:, 1])

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.collision_model.set_hand_qpos(hand_qpos)

    def plan(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        current, target = (np.asarray(q, dtype=np.float64) for q in (current, target))
        if any(q.shape != ARM_JOINT_SHAPE or not np.isfinite(q).all() for q in (current, target)):
            raise ValueError("HOME states must be finite shape (7,)")
        try:
            return self._candidate_path(current, target, use_canonical_target=True)
        except HomePathRejected:
            # 只有等价表示不同才尝试另一组路径。
            if np.array_equal(self.canonicalize_qpos(target, current), target):
                raise
        wrapped = self._candidate_path(current, target)
        start = wrapped[-1] if len(wrapped) else self.canonicalize_qpos(target, current)
        alignment = self._align_branch(start, target)
        if len(alignment):
            tail = alignment[1:] if len(wrapped) else alignment
            return np.concatenate((wrapped, tail))
        return wrapped

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
        *,
        allow_table_soft_escape: bool,
    ) -> None:
        # 分段线性插值在端点限位内时，整段都在限位内。
        if np.any(path < self.limits[:, 0]) or np.any(path > self.limits[:, 1]):
            raise HomePathRejected("home path violates operational joint limits")
        path = interpolate_waypoints(path, np.deg2rad(1.0))
        table_samples = []
        for index, waypoint in enumerate(path):
            reason = self.collision_model.rejection(waypoint)
            if reason is not None:
                raise HomePathRejected(f"collision at sample {index}: {reason}")
            position = self.collision_model.eef_position()
            if np.any(
                position < self.workspace_bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M
            ) or np.any(position > self.workspace_bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M):
                raise HomePathRejected(f"path leaves workspace at sample {index}: {position}")
            clearance, measurement, source = self._table_clearance()
            if not np.isfinite(clearance):
                raise RuntimeError(f"table_check_nonfinite at waypoint {index}")
            table_samples.append((index, clearance, measurement, source))

        negative = [sample for sample in table_samples if sample[1] < 0.0]
        if not negative:
            return
        best_clearance = table_samples[0][1]
        monotonic_escape = True
        for sample in table_samples[1:]:
            if best_clearance >= 0:
                break
            if sample[1] < best_clearance - 5e-4:
                monotonic_escape = False
                break
            best_clearance = max(best_clearance, sample[1])
        if (
            allow_table_soft_escape
            and table_samples[0][1] < 0.0
            and all(sample[0] == index for index, sample in enumerate(negative))
            and monotonic_escape
            and table_samples[-1][1] >= 0.0
        ):
            return
        index, clearance, measurement, source = negative[0]
        raise HomePathRejected(
            f"table_clearance at waypoint {index}: clearance={clearance:g} m, "
            f"{source}={measurement:g} m"
        )

    def _candidate_path(
        self,
        qpos: np.ndarray,
        home_qpos: np.ndarray,
        *,
        use_canonical_target: bool = False,
    ) -> np.ndarray:
        if use_canonical_target:
            # 绝对角差保留跨 2π 分支的回零需求。
            target = home_qpos
            delta = float(np.max(np.abs(target - qpos)))
        else:
            target = self.canonicalize_qpos(home_qpos, qpos)
            delta = float(np.max(np.abs(equivalent_joint_delta(target, qpos))))
        if delta <= self.arm.homing.convergence_rad:
            self._check_path(np.stack([qpos, target]), allow_table_soft_escape=False)
            return np.empty((0, 7), dtype=np.float64)

        rejections = []
        for name, mask in (
            ("direct", None),
            ("proximal_first", PROXIMAL_MASK),
            ("distal_first", ~PROXIMAL_MASK),
        ):
            mid = qpos.copy()
            if mask is not None:
                mid[mask] = target[mask]
            milestones = (
                np.stack((qpos, target)) if mask is None else unique_milestones(qpos, mid, target)
            )
            try:
                self._check_path(milestones, allow_table_soft_escape=True)
            except HomePathRejected as exc:
                rejections.append(f"{name}: {exc}")
                continue
            return milestones
        raise HomePathRejected("no safe home path; " + "; ".join(rejections))

    def _align_branch(
        self,
        wrapped_home: np.ndarray,
        canonical_home: np.ndarray,
    ) -> np.ndarray:
        delta = np.rad2deg(np.abs(wrapped_home - canonical_home))
        if not np.any(delta[np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)] > 1.0):
            return np.empty((0, 7), dtype=np.float64)
        milestones = np.stack([wrapped_home, canonical_home])
        try:
            self._check_path(
                milestones,
                allow_table_soft_escape=False,
            )
        except HomePathRejected as exc:
            raise HomePathRejected(f"band_alignment: {exc}") from exc
        return milestones


def interpolate_waypoints(path: np.ndarray, max_step: float) -> np.ndarray:
    """插值后每个关节的步长不超过 max_step，单位 rad。"""
    if len(path) <= 1:
        return path.astype(np.float64)
    dense = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        count = int(np.ceil(np.max(np.abs(b - a)) / max_step))
        if count:
            fractions = np.arange(1, count + 1)[:, None] / count
            dense.append(a + fractions * (b - a))
    return np.concatenate(dense).astype(np.float64, copy=False)


def unique_milestones(*points: np.ndarray) -> np.ndarray:
    unique = [np.asarray(points[0], dtype=np.float64)]
    for point in points[1:]:
        point = np.asarray(point, dtype=np.float64)
        if float(np.max(np.abs(point - unique[-1]))) > 1e-9:
            unique.append(point)
    return np.asarray(unique, dtype=np.float64)
