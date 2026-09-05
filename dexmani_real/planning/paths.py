"""Pure joint paths and typed collision-checked home-path results.

``HomePathResult`` makes already-home, safe-to-command, and unsafe outcomes
explicit so callers cannot confuse an empty path with a rejected motion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.config.defaults import arm as _arm_cfg
from dexmani_real.robot.model import ARM_JOINT_SHAPE

if TYPE_CHECKING:
    from dexmani_real.planning.planner import XArm7MotionPlanner


# Proximal (shoulder/elbow) vs distal (wrist) joint mask for two-stage homing.
# J1-J4 move first (arm repositioning), then J5-J7 (wrist orientation).
_PROXIMAL_MASK = np.array([True, True, True, True, False, False, False], dtype=bool)

# Allow a small outward tolerance for nonlinear FK interpolation at workspace edges.
WORKSPACE_BOUNDS_TOLERANCE_M = 1e-3


@dataclass(kw_only=True)
class PathResult:
    success: bool
    qpos_path: np.ndarray | None
    reason: str = ""
    source: str = ""
    report: dict[str, Any] = field(default_factory=dict)


class HomePathStatus(str, Enum):
    """Explicit outcome of collision-checked home-path planning."""

    ALREADY_HOME = "already_home"
    SAFE = "safe"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class HomeCollisionPair:
    """Links involved in one rejected home-path collision sample."""

    link1: str
    link2: str
    kind: str = ""


@dataclass(frozen=True)
class HomePathCandidate:
    """Bounded diagnostic for one densely validated path candidate."""

    name: str
    safe: bool
    reason: str = ""
    sample_count: int = 0
    detail: str = ""
    collision_waypoint_index: int | None = None
    collision_pairs: tuple[HomeCollisionPair, ...] = ()
    workspace_segment_index: int | None = None
    table_waypoint_index: int | None = None
    table_check_source: str = ""
    table_raw_measurement_m: float | None = None
    clearance_m: float | None = None
    minimum_table_clearance_m: float | None = None
    table_soft_escape: bool = False


@dataclass(frozen=True)
class HomePathResult:
    """Typed home-path result; status, path, and diagnostics travel together."""

    status: HomePathStatus
    waypoints: np.ndarray
    selected_candidate: str | None = None
    candidates: tuple[HomePathCandidate, ...] = ()

    def __post_init__(self) -> None:
        waypoints = np.asarray(self.waypoints, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[1:] != ARM_JOINT_SHAPE:
            raise ValueError("home-path waypoints must have shape (N, 7)")
        if not np.all(np.isfinite(waypoints)):
            raise ValueError("home-path waypoints must be finite")
        if self.status is HomePathStatus.SAFE and len(waypoints) < 2:
            raise ValueError("a safe home path must contain at least two milestones")
        if self.status is not HomePathStatus.SAFE and len(waypoints) != 0:
            raise ValueError("non-safe home-path results must not contain motion")
        object.__setattr__(self, "waypoints", waypoints.copy())


def _table_clearance_m(
    planner: "XArm7MotionPlanner",
    arm_qpos: np.ndarray,
    *,
    table_z_surface_m: float | None,
    hand_safety_margin_m: float,
) -> tuple[float, float, str]:
    """Return ``(clearance, raw_measurement, source)`` for one arm pose."""
    collision_model = planner.collision_model
    if bool(getattr(collision_model, "has_table", False)):
        distance_m = float(collision_model.minimum_table_distance(arm_qpos))
        clearance_m = distance_m - float(collision_model.table_soft_clearance_m)
        return clearance_m, distance_m, "calibrated_mesh_distance"
    if table_z_surface_m is None:
        return float("inf"), float("inf"), "disabled"
    hand_min_z_m = float(collision_model.minimum_hand_frame_z(arm_qpos))
    clearance_m = hand_min_z_m - hand_safety_margin_m - table_z_surface_m
    return clearance_m, hand_min_z_m, "hand_frame_proxy"


def interpolate_waypoints(path: np.ndarray, max_step: float) -> np.ndarray:
    """Linearly densify a sparse joint path so each step ≤ max_step rad.

    Args:
        path:     (W, J) array of joint-space waypoints.
        max_step: maximum joint change per step (radians).

    Returns:
        (D, J) dense path (D ≥ W).
    """
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
    joint_limit_lower: tuple[float, ...],
    joint_limit_upper: tuple[float, ...],
) -> np.ndarray:
    """Select the nearest limit-valid equivalent of each joint.

    Equivalent joints are those where the joint limit span exceeds 2*pi
    (J1, J3, J5, J7 on xArm7).  For each such joint, the value is shifted
    by an integer multiple of 2*pi so it is as close as possible to the
    corresponding reference value while remaining inside hardware limits.

    Non-equivalent joints are returned unchanged.

    Args:
        qpos:               Joint positions to wrap (shape broadcastable to 1D).
        reference:          Reference positions (same shape as *qpos*).
        joint_limit_lower:  Hardware lower limits (rad).
        joint_limit_upper:  Hardware upper limits (rad).

    Returns:
        Wrapped qpos (float64, same shape as input).
    """
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
        raise ValueError(
            "qpos, reference, and joint limits must be matching 1-D arrays"
        )
    if not all(np.all(np.isfinite(values)) for values in (result, ref, lo, hi)):
        raise ValueError("qpos, reference, and joint limits must be finite")
    if np.any(lo > hi):
        raise ValueError("joint lower limits must not exceed upper limits")
    joint_range = hi - lo
    is_equiv = joint_range > 2.0 * np.pi

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


def _empty_home_path() -> np.ndarray:
    return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)


def _collision_pairs(collision: object) -> tuple[HomeCollisionPair, ...]:
    """Normalize planner collision payloads at their untyped boundary."""
    if not isinstance(collision, Mapping):
        return ()
    raw_pairs = collision.get("collision_pairs", ())
    if not isinstance(raw_pairs, (list, tuple)):
        return ()
    pairs: list[HomeCollisionPair] = []
    for raw_pair in raw_pairs:
        if isinstance(raw_pair, Mapping):
            pairs.append(
                HomeCollisionPair(
                    link1=str(raw_pair.get("link1", "?")),
                    link2=str(raw_pair.get("link2", "?")),
                    kind=str(raw_pair.get("type", "")),
                )
            )
    return tuple(pairs)


def _check_home_path_candidate(
    path: np.ndarray,
    candidate_name: str,
    planner: "XArm7MotionPlanner | None",
    *,
    table_z_surface_m: float | None,
    hand_safety_margin_m: float,
    allow_table_soft_escape: bool,
) -> HomePathCandidate:
    """Validate one dense path and return its first safety rejection."""
    sample_count = len(path)
    have_collision = (
        planner is not None and planner.planning_profile.check_self_collision
    )
    if have_collision:
        assert planner is not None
        path_check = getattr(
            planner.ik_mgr,
            "check_path_combined_collisions",
            planner.ik_mgr.check_path_collisions,
        )
        try:
            collision_result = path_check(path)
        except Exception as exc:
            return HomePathCandidate(
                candidate_name,
                False,
                reason="collision_check_error",
                sample_count=sample_count,
                detail=str(exc),
            )
        if collision_result.get(
            "path_collision", collision_result.get("path_self_collision", False)
        ):
            pairs = _collision_pairs(collision_result.get("collision"))
            reason = (
                "environment_collision"
                if any(pair.kind == "environment" for pair in pairs)
                else "self_collision"
            )
            return HomePathCandidate(
                candidate_name,
                False,
                reason=reason,
                sample_count=sample_count,
                collision_waypoint_index=collision_result.get(
                    "collision_waypoint_index"
                ),
                collision_pairs=pairs,
            )

    if planner is not None:
        try:
            for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
                if not planner.is_workspace_segment_safe(start, end):
                    return HomePathCandidate(
                        candidate_name,
                        False,
                        reason="workspace",
                        sample_count=sample_count,
                        workspace_segment_index=segment_index,
                    )
        except (ValueError, RuntimeError) as exc:
            return HomePathCandidate(
                candidate_name,
                False,
                reason="workspace_check_error",
                sample_count=sample_count,
                detail=str(exc),
            )

    check_table = planner is not None and (
        bool(getattr(planner.collision_model, "has_table", False))
        or table_z_surface_m is not None
    )
    if not check_table:
        return HomePathCandidate(candidate_name, True, sample_count=sample_count)

    assert planner is not None
    table_samples: list[tuple[int, float, float, str]] = []
    for waypoint_index, waypoint in enumerate(path):
        try:
            clearance_m, raw_measurement_m, source = _table_clearance_m(
                planner,
                waypoint,
                table_z_surface_m=table_z_surface_m,
                hand_safety_margin_m=hand_safety_margin_m,
            )
        except Exception as exc:
            return HomePathCandidate(
                candidate_name,
                False,
                reason="table_check_error",
                sample_count=sample_count,
                detail=str(exc),
                table_waypoint_index=waypoint_index,
            )
        if not np.isfinite(clearance_m):
            return HomePathCandidate(
                candidate_name,
                False,
                reason="table_check_nonfinite",
                sample_count=sample_count,
                table_waypoint_index=waypoint_index,
            )
        table_samples.append((waypoint_index, clearance_m, raw_measurement_m, source))

    minimum_clearance_m = min(sample[1] for sample in table_samples)
    negative_samples = [sample for sample in table_samples if sample[1] < 0.0]
    if not negative_samples:
        return HomePathCandidate(
            candidate_name,
            True,
            sample_count=sample_count,
            minimum_table_clearance_m=minimum_clearance_m,
        )

    starts_inside_soft_zone = table_samples[0][1] < 0.0
    negative_is_initial_prefix = all(
        sample[0] == index for index, sample in enumerate(negative_samples)
    )
    monotonic_escape = all(
        current[1] >= previous[1] - 5e-4
        for previous, current in zip(table_samples[:-1], table_samples[1:])
        if previous[1] < 0.0
    )
    reaches_clearance = table_samples[-1][1] >= 0.0
    is_soft_escape = (
        allow_table_soft_escape
        and starts_inside_soft_zone
        and negative_is_initial_prefix
        and monotonic_escape
        and reaches_clearance
    )
    if is_soft_escape:
        return HomePathCandidate(
            candidate_name,
            True,
            sample_count=sample_count,
            minimum_table_clearance_m=minimum_clearance_m,
            table_soft_escape=True,
        )

    bad_index, bad_clearance_m, bad_measurement_m, bad_source = negative_samples[0]
    return HomePathCandidate(
        candidate_name,
        False,
        reason="table_clearance",
        sample_count=sample_count,
        table_waypoint_index=bad_index,
        table_check_source=bad_source,
        table_raw_measurement_m=float(bad_measurement_m),
        clearance_m=float(bad_clearance_m),
        minimum_table_clearance_m=minimum_clearance_m,
    )


def _unique_milestones(*points: np.ndarray) -> np.ndarray:
    unique = [np.asarray(points[0], dtype=np.float64)]
    for point in points[1:]:
        point = np.asarray(point, dtype=np.float64)
        if float(np.max(np.abs(point - unique[-1]))) > 1e-9:
            unique.append(point)
    return np.asarray(unique, dtype=np.float64)


def compute_joint_home_path(
    qpos: np.ndarray,
    home_qpos: np.ndarray,
    planner: "XArm7MotionPlanner | None" = None,
    *,
    table_z_surface_m: float | None = None,
    hand_safety_margin_m: float | None = None,
    use_canonical_target: bool = False,
) -> HomePathResult:
    """Return a typed, collision-checked path from current to home joints."""
    if hand_safety_margin_m is None:
        hand_safety_margin_m = _arm_cfg.hand_safety_margin_m

    if use_canonical_target:
        # Canonical home uses raw absolute deltas; equivalent joints may rotate a full band.
        target = np.asarray(home_qpos, dtype=np.float64)
        # Use raw delta for the already-home check; wrapped deltas hide band mismatches.
        delta = float(np.max(np.abs(target - qpos)))
    else:
        if planner is not None:
            target = planner.ik_mgr.nearest_equivalent_qpos(home_qpos, qpos)
            delta = float(
                np.max(np.abs(planner.ik_mgr.compute_qpos_delta(target, qpos)))
            )
        else:
            target = wrap_nearest_equivalent(
                home_qpos,
                qpos,
                _arm_cfg.joint_limit_lower,
                _arm_cfg.joint_limit_upper,
            )
            delta = float(np.max(np.abs(qpos - target)))
    if delta < np.deg2rad(0.15):
        return HomePathResult(HomePathStatus.ALREADY_HOME, _empty_home_path())

    candidates: list[HomePathCandidate] = []

    def try_candidate(milestones: np.ndarray, name: str) -> HomePathResult | None:
        samples = interpolate_waypoints(milestones, np.deg2rad(1.0))
        candidate = _check_home_path_candidate(
            samples,
            name,
            planner,
            table_z_surface_m=table_z_surface_m,
            hand_safety_margin_m=float(hand_safety_margin_m),
            allow_table_soft_escape=True,
        )
        candidates.append(candidate)
        if not candidate.safe:
            return None
        return HomePathResult(
            HomePathStatus.SAFE,
            milestones,
            selected_candidate=name,
            candidates=tuple(candidates),
        )

    direct_milestones = np.stack([qpos, target])
    result = try_candidate(direct_milestones, "direct")
    if result is not None:
        return result

    mid = qpos.copy()
    mid[_PROXIMAL_MASK] = target[_PROXIMAL_MASK]
    proximal_first = _unique_milestones(qpos, mid, target)
    result = try_candidate(proximal_first, "proximal_first")
    if result is not None:
        return result

    distal_mid = qpos.copy()
    distal_mid[~_PROXIMAL_MASK] = target[~_PROXIMAL_MASK]
    distal_first = _unique_milestones(qpos, distal_mid, target)
    result = try_candidate(distal_first, "distal_first")
    if result is not None:
        return result

    if planner is not None and hasattr(planner, "plan_joint_qpos_path"):
        try:
            planned = planner.plan_joint_qpos_path(target, qpos, planning_time_s=0.5)
        except (ValueError, RuntimeError) as exc:
            candidates.append(
                HomePathCandidate(
                    "joint_qpos_rrt",
                    False,
                    reason="planner_error",
                    detail=str(exc),
                )
            )
        else:
            if (
                planned.success
                and planned.qpos_path is not None
                and len(planned.qpos_path) >= 2
            ):
                rrt_milestones = np.asarray(planned.qpos_path, dtype=np.float64)
                if float(np.max(np.abs(rrt_milestones[-1] - target))) > 1e-6:
                    rrt_milestones = np.concatenate(
                        [rrt_milestones, target.reshape((1, *ARM_JOINT_SHAPE))], axis=0
                    )
                result = try_candidate(rrt_milestones, "joint_qpos_rrt")
                if result is not None:
                    return result
            else:
                candidates.append(
                    HomePathCandidate(
                        "joint_qpos_rrt",
                        False,
                        reason="planner_failed",
                        detail=planned.reason,
                    )
                )

    return HomePathResult(
        HomePathStatus.UNSAFE,
        _empty_home_path(),
        candidates=tuple(candidates),
    )


def compute_band_alignment_path(
    wrapped_home: np.ndarray,
    canonical_home: np.ndarray,
    planner: "XArm7MotionPlanner | None" = None,
    *,
    table_z_surface_m: float | None = None,
    hand_safety_margin_m: float | None = None,
) -> HomePathResult:
    """Return a typed safety result for equivalent-joint band alignment."""
    if hand_safety_margin_m is None:
        hand_safety_margin_m = _arm_cfg.hand_safety_margin_m

    equivalent_mask = (
        np.array(_arm_cfg.joint_limit_upper) - np.array(_arm_cfg.joint_limit_lower)
    ) > 2.0 * np.pi
    raw_delta_deg = np.rad2deg(np.abs(wrapped_home - canonical_home))
    if not np.any(raw_delta_deg[equivalent_mask] > 1.0):
        return HomePathResult(HomePathStatus.ALREADY_HOME, _empty_home_path())

    milestones = np.stack([wrapped_home, canonical_home])
    dense_path = interpolate_waypoints(milestones, np.deg2rad(1.0))
    candidate = _check_home_path_candidate(
        dense_path,
        "band_alignment",
        planner,
        table_z_surface_m=table_z_surface_m,
        hand_safety_margin_m=float(hand_safety_margin_m),
        allow_table_soft_escape=False,
    )
    if not candidate.safe:
        return HomePathResult(
            HomePathStatus.UNSAFE,
            _empty_home_path(),
            candidates=(candidate,),
        )
    return HomePathResult(
        HomePathStatus.SAFE,
        milestones,
        selected_candidate=candidate.name,
        candidates=(candidate,),
    )
