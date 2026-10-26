"""HOME 候选路径、等价分支对齐与桌面间隙。"""

from __future__ import annotations

import numpy as np

from dexmani_real.control.collision import CollisionModel
from dexmani_real.control.config import EnvironmentConfig
from dexmani_real.control.constraints import (
    equivalent_joint_delta,
    wrap_nearest_equivalent,
)
from dexmani_real.geometry.kinematics import ArmFK
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
        self.limits = np.column_stack((arm.joint_limit_lower, arm.joint_limit_upper))
        self.workspace_bounds = environment.workspace.as_array()
        self.arm_fk = ArmFK()
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
            return compute_joint_home_path(current, target, self, use_canonical_target=True)
        except HomePathRejected:
            # 只有等价表示不同才尝试另一组路径。
            if np.array_equal(self.canonicalize_qpos(target, current), target):
                raise
        wrapped = compute_joint_home_path(current, target, self)
        start = wrapped[-1] if len(wrapped) else self.canonicalize_qpos(target, current)
        alignment = compute_band_alignment_path(start, target, self)
        if len(alignment):
            tail = alignment[1:] if len(wrapped) else alignment
            return np.concatenate((wrapped, tail))
        return wrapped


def table_clearance_m(
    planner: HomePlanner,
    arm_qpos: np.ndarray,
) -> tuple[float, float, str]:
    """返回净间隙、原始测量和测量来源。"""
    collision_model = planner.collision_model
    if collision_model.table is not None:
        distance_m = float(collision_model.minimum_table_distance(arm_qpos))
        clearance_m = distance_m - float(collision_model.table.soft_clearance_m)
        return clearance_m, distance_m, "calibrated_mesh_distance"
    hand_min_z_m = float(collision_model.minimum_hand_frame_z(arm_qpos))
    clearance_m = hand_min_z_m - planner.arm.hand_safety_margin_m - planner.arm.table_z_surface_m
    return clearance_m, hand_min_z_m, "hand_frame_proxy"


def interpolate_waypoints(path: np.ndarray, max_step: float) -> np.ndarray:
    """插值后每个关节的步长不超过 max_step，单位 rad。"""
    if len(path) <= 1:
        return path.astype(np.float64)
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)


def check_home_path(
    path: np.ndarray,
    planner: HomePlanner,
    *,
    allow_table_soft_escape: bool,
) -> None:
    # 分段线性插值在端点限位内时，整段都在限位内。
    if np.any(path < planner.limits[:, 0]) or np.any(path > planner.limits[:, 1]):
        raise HomePathRejected("home path violates operational joint limits")
    path = interpolate_waypoints(path, np.deg2rad(1.0))
    table_samples = []
    for index, waypoint in enumerate(path):
        reason = planner.collision_model.rejection(waypoint)
        if reason is not None:
            raise HomePathRejected(f"collision at sample {index}: {reason}")
        position, _ = planner.arm_fk.forward(waypoint)
        if np.any(
            position < planner.workspace_bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M
        ) or np.any(position > planner.workspace_bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M):
            raise HomePathRejected(f"path leaves workspace at sample {index}: {position}")
        clearance, measurement, source = table_clearance_m(planner, waypoint)
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


def unique_milestones(*points: np.ndarray) -> np.ndarray:
    unique = [np.asarray(points[0], dtype=np.float64)]
    for point in points[1:]:
        point = np.asarray(point, dtype=np.float64)
        if float(np.max(np.abs(point - unique[-1]))) > 1e-9:
            unique.append(point)
    return np.asarray(unique, dtype=np.float64)


def compute_joint_home_path(
    qpos: np.ndarray,
    home_qpos: np.ndarray,
    planner: HomePlanner,
    *,
    use_canonical_target: bool = False,
) -> np.ndarray:
    if use_canonical_target:
        # 绝对角差保留跨 2π 分支的回零需求。
        target = home_qpos
        delta = float(np.max(np.abs(target - qpos)))
    else:
        target = planner.canonicalize_qpos(home_qpos, qpos)
        delta = float(np.max(np.abs(equivalent_joint_delta(target, qpos))))
    if delta <= planner.arm.homing.convergence_rad:
        check_home_path(np.stack([qpos, target]), planner, allow_table_soft_escape=False)
        return np.empty((0, 7), dtype=np.float64)

    rejections = []

    def admitted(milestones, name):
        try:
            check_home_path(
                milestones,
                planner,
                allow_table_soft_escape=True,
            )
        except HomePathRejected as exc:
            rejections.append(f"{name}: {exc}")
            return False
        return True

    direct = np.stack([qpos, target])
    if admitted(direct, "direct"):
        return direct
    mid = qpos.copy()
    mid[PROXIMAL_MASK] = target[PROXIMAL_MASK]
    proximal = unique_milestones(qpos, mid, target)
    if admitted(proximal, "proximal_first"):
        return proximal
    mid = qpos.copy()
    mid[~PROXIMAL_MASK] = target[~PROXIMAL_MASK]
    distal = unique_milestones(qpos, mid, target)
    if admitted(distal, "distal_first"):
        return distal

    raise HomePathRejected("no safe home path; " + "; ".join(rejections))


def compute_band_alignment_path(
    wrapped_home: np.ndarray,
    canonical_home: np.ndarray,
    planner: HomePlanner,
) -> np.ndarray:
    delta = np.rad2deg(np.abs(wrapped_home - canonical_home))
    if not np.any(delta[np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)] > 1.0):
        return np.empty((0, 7), dtype=np.float64)
    milestones = np.stack([wrapped_home, canonical_home])
    try:
        check_home_path(
            milestones,
            planner,
            allow_table_soft_escape=False,
        )
    except HomePathRejected as exc:
        raise HomePathRejected(f"band_alignment: {exc}") from exc
    return milestones
