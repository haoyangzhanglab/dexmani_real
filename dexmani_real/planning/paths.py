"""Joint-space HOME candidates, dense safety checks and equivalent-angle handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, XARM7_EQUIVALENT_JOINT_MASK

if TYPE_CHECKING:
    from dexmani_real.planning.home_planner import HomePlanner


# The proximal-first HOME candidate moves J1–J4 before J5–J7.
PROXIMAL_MASK = np.array([True, True, True, True, False, False, False], dtype=bool)

# Allow a small outward tolerance for nonlinear FK interpolation at workspace edges.
WORKSPACE_BOUNDS_TOLERANCE_M = 1e-3


def table_clearance_m(
    planner: "HomePlanner",
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


def wrap_nearest_equivalent(
    qpos: np.ndarray,
    reference: np.ndarray,
    joint_limit_lower: tuple[float, ...] | np.ndarray,
    joint_limit_upper: tuple[float, ...] | np.ndarray,
    *,
    equivalent_mask=XARM7_EQUIVALENT_JOINT_MASK,
) -> np.ndarray:
    """Return a float64 copy with equivalent axes shifted by 2π to the nearest limit-valid radian values."""
    result = np.asarray(qpos, dtype=np.float64).copy()
    lo = np.asarray(joint_limit_lower, dtype=np.float64)
    hi = np.asarray(joint_limit_upper, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if (
        result.ndim != 1
        or ref.shape != result.shape
        or lo.shape != result.shape
        or hi.shape != result.shape
    ):
        raise ValueError("qpos, reference, and joint limits must be matching 1-D arrays")
    if not all(np.all(np.isfinite(values)) for values in (result, ref, lo, hi)):
        raise ValueError("qpos, reference, and joint limits must be finite")
    if np.any(lo > hi):
        raise ValueError("joint lower limits must not exceed upper limits")
    is_equiv = np.asarray(equivalent_mask, dtype=bool)
    if is_equiv.shape != result.shape:
        raise ValueError("equivalent mask must match joint shape")

    if not np.any(is_equiv):
        return result

    period = 2.0 * np.pi
    lo_equiv = lo[is_equiv]
    hi_equiv = hi[is_equiv]
    res_equiv = result[is_equiv]
    ref_equiv = ref[is_equiv]
    k_min = np.ceil((lo_equiv - res_equiv) / period)
    k_max = np.floor((hi_equiv - res_equiv) / period)
    valid = k_min <= k_max
    nearest_k = np.rint((ref_equiv - res_equiv) / period)
    nearest_k = np.minimum(np.maximum(nearest_k, k_min), k_max)
    # Preserve invalid raw values so downstream validation can reject them.
    result[is_equiv] = np.where(valid, res_equiv + nearest_k * period, res_equiv)

    return result


def check_home_path(
    path: np.ndarray,
    planner: "HomePlanner",
    *,
    table_z_surface_m: float,
    hand_safety_margin_m: float,
    allow_table_soft_escape: bool,
) -> None:
    """Reject the first unsafe sample; table soft escape never permits mesh collision."""
    try:
        collision = planner.collision_model.check_path(path)
    except Exception as exc:
        raise ValueError(f"collision_check_error: {exc}") from exc
    if collision is not None:
        index, info = collision
        raise ValueError(f"collision at segment {index}: {info}")

    try:
        planner.check_workspace_path(path)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"workspace: {exc}") from exc

    table_samples = []
    for index, waypoint in enumerate(path):
        try:
            clearance, measurement, source = table_clearance_m(
                planner,
                waypoint,
                table_z_surface_m=table_z_surface_m,
                hand_safety_margin_m=hand_safety_margin_m,
            )
        except Exception as exc:
            raise ValueError(f"table_check_error at waypoint {index}: {exc}") from exc
        if not np.isfinite(clearance):
            raise ValueError(f"table_check_nonfinite at waypoint {index}")
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
    raise ValueError(
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
    planner: "HomePlanner",
    *,
    table_z_surface_m: float,
    hand_safety_margin_m: float,
    use_canonical_target: bool = False,
) -> np.ndarray:
    """Return admitted milestones, an empty (0,7) path if home, or raise ValueError."""
    qpos = home_joint_vector(qpos, "current qpos")
    home_qpos = home_joint_vector(home_qpos, "home qpos")
    if use_canonical_target:
        # Absolute HOME deltas preserve equivalent-joint band mismatches.
        target = home_qpos
        delta = float(np.max(np.abs(target - qpos)))
    else:
        target = planner.kin.canonicalize_qpos(home_qpos, qpos)
        delta = float(np.max(np.abs(planner.kin.compute_qpos_delta(target, qpos))))
    if delta < np.deg2rad(0.15):
        return np.empty((0, 7), dtype=np.float64)

    rejections = []

    def admitted(milestones, name):
        try:
            check_home_path(
                interpolate_waypoints(milestones, np.deg2rad(1.0)),
                planner,
                table_z_surface_m=table_z_surface_m,
                hand_safety_margin_m=hand_safety_margin_m,
                allow_table_soft_escape=True,
            )
        except ValueError as exc:
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

    raise ValueError("no safe home path; " + "; ".join(rejections))


def compute_band_alignment_path(
    wrapped_home: np.ndarray,
    canonical_home: np.ndarray,
    planner: "HomePlanner",
    *,
    table_z_surface_m: float,
    hand_safety_margin_m: float,
) -> np.ndarray:
    """Admit absolute equivalent-angle alignment without table soft escape."""
    wrapped_home = home_joint_vector(wrapped_home, "wrapped home")
    canonical_home = home_joint_vector(canonical_home, "canonical home")
    delta = np.rad2deg(np.abs(wrapped_home - canonical_home))
    if not np.any(delta[planner.kin.equivalent_joint_mask] > 1.0):
        return np.empty((0, 7), dtype=np.float64)
    milestones = np.stack([wrapped_home, canonical_home])
    try:
        check_home_path(
            interpolate_waypoints(milestones, np.deg2rad(1.0)),
            planner,
            table_z_surface_m=table_z_surface_m,
            hand_safety_margin_m=hand_safety_margin_m,
            allow_table_soft_escape=False,
        )
    except ValueError as exc:
        raise ValueError(f"band_alignment: {exc}") from exc
    return milestones
