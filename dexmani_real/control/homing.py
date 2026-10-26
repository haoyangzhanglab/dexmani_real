"""HOME 几何、候选路径、等价分支对齐与密集采样准入。"""

from __future__ import annotations

import numpy as np

from dexmani_real.control.collision import CollisionModel
from dexmani_real.control.config import EnvironmentConfig
from dexmani_real.control.joint_angles import (
    equivalent_joint_delta,
    wrap_nearest_equivalent,
)
from dexmani_real.geometry.kinematics.arm_fk import ArmFK
from dexmani_real.robot.model import ARM_JOINT_SHAPE, XARM7_EQUIVALENT_JOINT_MASK

# The proximal-first HOME candidate moves J1–J4 before J5–J7.
PROXIMAL_MASK = np.array([True, True, True, True, False, False, False], dtype=bool)

# Allow a small outward tolerance for nonlinear FK interpolation at workspace edges.
WORKSPACE_BOUNDS_TOLERANCE_M = 1e-3


class HomePathRejected(ValueError):
    """几何约束拒绝了 HOME 路径；技术故障不属于可尝试其他候选的拒绝。"""


class HomePlanner:
    """Nominal-URDF arm kinematics plus the 19-DOF arm/hand collision model."""

    def __init__(
        self,
        environment: EnvironmentConfig,
        *,
        table_plane_abcd: tuple[float, float, float, float] | None,
    ) -> None:
        self.workspace_bounds = environment.workspace.as_array()
        self.arm_fk = ArmFK()
        self.collision_model = CollisionModel(
            static_boxes=environment.static_boxes,
            table=environment.table,
            table_plane_abcd=table_plane_abcd,
        )

    def canonicalize_qpos(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        model = self.arm_fk.model
        return wrap_nearest_equivalent(
            qpos, reference_qpos, model.lowerPositionLimit, model.upperPositionLimit
        )

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.collision_model.set_hand_qpos(hand_qpos)

    def check_workspace_path(self, path: np.ndarray) -> None:
        """Check the owner's dense HOME samples against the EEF workspace."""
        for index, qpos in enumerate(path):
            position, _ = self.arm_fk.compute(qpos)
            if (
                np.any(position < self.workspace_bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M)
                or np.any(position > self.workspace_bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M)
            ):
                raise HomePathRejected(
                    f"Path leaves configured workspace at sample {index}: {position}"
                )


def table_clearance_m(
    planner: HomePlanner,
    arm_qpos: np.ndarray,
    *,
    table_z_surface_m: float,
    hand_safety_margin_m: float,
) -> tuple[float, float, str]:
    """Return ``(clearance, raw_measurement, source)`` for one arm pose."""
    collision_model = planner.collision_model
    if collision_model.has_table:
        distance_m = float(collision_model.minimum_table_distance(arm_qpos))
        clearance_m = distance_m - float(collision_model.table_soft_clearance_m)
        return clearance_m, distance_m, "calibrated_mesh_distance"
    hand_min_z_m = float(collision_model.minimum_hand_frame_z(arm_qpos))
    clearance_m = hand_min_z_m - hand_safety_margin_m - table_z_surface_m
    return clearance_m, hand_min_z_m, "hand_frame_proxy"


def interpolate_waypoints(path: np.ndarray, max_step: float) -> np.ndarray:
    """Densify (W,J) waypoints into float64 (D,J) with per-joint steps ≤ max_step radians."""
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
    table_z_surface_m: float,
    hand_safety_margin_m: float,
    allow_table_soft_escape: bool,
) -> None:
    """Check one shared 1° sampling; table soft escape never permits mesh collision."""
    path = interpolate_waypoints(path, np.deg2rad(1.0))
    collision = planner.collision_model.check_path(path)
    if collision is not None:
        index, info = collision
        raise HomePathRejected(f"collision at sample {index}: {info}")

    planner.check_workspace_path(path)

    table_samples = []
    for index, waypoint in enumerate(path):
        clearance, measurement, source = table_clearance_m(
            planner,
            waypoint,
            table_z_surface_m=table_z_surface_m,
            hand_safety_margin_m=hand_safety_margin_m,
        )
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


def home_joint_vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != ARM_JOINT_SHAPE or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite shape (7,)")
    return value


def compute_joint_home_path(
    qpos: np.ndarray,
    home_qpos: np.ndarray,
    planner: HomePlanner,
    *,
    table_z_surface_m: float,
    hand_safety_margin_m: float,
    use_canonical_target: bool = False,
) -> np.ndarray:
    """Return admitted milestones, an empty (0,7) path if home, or raise HomePathRejected."""
    qpos = home_joint_vector(qpos, "current qpos")
    home_qpos = home_joint_vector(home_qpos, "home qpos")
    if use_canonical_target:
        # Absolute HOME deltas preserve equivalent-joint band mismatches.
        target = home_qpos
        delta = float(np.max(np.abs(target - qpos)))
    else:
        target = planner.canonicalize_qpos(home_qpos, qpos)
        delta = float(np.max(np.abs(equivalent_joint_delta(target, qpos))))
    if delta < np.deg2rad(0.15):
        return np.empty((0, 7), dtype=np.float64)

    rejections = []

    def admitted(milestones, name):
        try:
            check_home_path(
                milestones,
                planner,
                table_z_surface_m=table_z_surface_m,
                hand_safety_margin_m=hand_safety_margin_m,
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
    *,
    table_z_surface_m: float,
    hand_safety_margin_m: float,
) -> np.ndarray:
    """Admit absolute equivalent-angle alignment without table soft escape."""
    wrapped_home = home_joint_vector(wrapped_home, "wrapped home")
    canonical_home = home_joint_vector(canonical_home, "canonical home")
    delta = np.rad2deg(np.abs(wrapped_home - canonical_home))
    if not np.any(delta[np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)] > 1.0):
        return np.empty((0, 7), dtype=np.float64)
    milestones = np.stack([wrapped_home, canonical_home])
    try:
        check_home_path(
            milestones,
            planner,
            table_z_surface_m=table_z_surface_m,
            hand_safety_margin_m=hand_safety_margin_m,
            allow_table_soft_escape=False,
        )
    except HomePathRejected as exc:
        raise HomePathRejected(f"band_alignment: {exc}") from exc
    return milestones


def home_path(current, target, planner, *, table_z_surface_m, hand_safety_margin_m):
    options = dict(
        table_z_surface_m=table_z_surface_m,
        hand_safety_margin_m=hand_safety_margin_m,
    )
    try:
        return compute_joint_home_path(
            current, target, planner, use_canonical_target=True, **options
        )
    except HomePathRejected:
        # Retry only when the equivalent representation gives different candidates.
        if np.array_equal(planner.canonicalize_qpos(target, current), target):
            raise
    wrapped = compute_joint_home_path(current, target, planner, **options)
    start = wrapped[-1] if len(wrapped) else planner.canonicalize_qpos(target, current)
    alignment = compute_band_alignment_path(start, target, planner, **options)
    if len(alignment):
        tail = alignment[1:] if len(wrapped) else alignment
        return np.concatenate((wrapped, tail))
    return wrapped
