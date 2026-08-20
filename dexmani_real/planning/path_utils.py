"""Path processing utilities — densification, interpolation, smoothing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from dexmani_real.config.defaults import arm as _arm_cfg
from dexmani_real.utils.schema import ARM_JOINT_SHAPE

if TYPE_CHECKING:
    from dexmani_real.planning.planner import XArm7MotionPlanner


# Proximal (shoulder/elbow) vs distal (wrist) joint mask for two-stage homing.
# J1-J4 move first (arm repositioning), then J5-J7 (wrist orientation).
_PROXIMAL_MASK = np.array([True, True, True, True, False, False, False], dtype=bool)


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
    return clearance_m, hand_min_z_m, "legacy_hand_frame_proxy"


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
    if result.ndim != 1 or ref.shape != result.shape or lo.shape != result.shape or hi.shape != result.shape:
        raise ValueError("qpos, reference, and joint limits must be matching 1-D arrays")
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


def plan_joint_home_path(
    qpos: np.ndarray,
    home_qpos: np.ndarray,
    planner: "XArm7MotionPlanner | None" = None,
    *,
    table_z_surface_m: float | None = None,
    hand_safety_margin_m: float | None = None,
    use_canonical_target: bool = False,
    report: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Plan a collision-safe joint-space path from *qpos* to *home_qpos*.

    Returns:
        * ``None`` — arm is already at home (delta < 0.15°); caller should
          send ``home_qpos`` directly (no-op).
        * ``(M, 7)`` execution milestones (normally 2–3) — every segment has
          been densely sampled and collision-checked at ≤1° joint increments.
        * ``(0, 7)`` empty array — no candidate passed all configured safety
          checks; caller MUST NOT fall back to unchecked interpolation.  The
          arm should hold position and the operator should be alerted.

    This function only reads the collision model — it does NOT need a live
    XArm7 connection.

    Args:
        qpos:              Current arm joint positions (7,).
        home_qpos:          Target home joint positions (7,).
        planner:            Planner with collision model.  When None, only
                            joint-limit-aware wrapping is performed.
        table_z_surface_m:  Fixed-Z fallback used only when the planner
                            has no calibrated table geometry.  With
                            ``environment.table`` enabled, every waypoint uses
                            exact robot-mesh distance to the tilted table box.
        hand_safety_margin_m:
                            Conservative padding from link-frame origins to the
                            lowest hand collision surface.
                            Defaults to ``arm.hand_safety_margin_m`` (0.05 m)
                            when ``None``. Used only by the fixed-Z fallback.
        use_canonical_target:
                            When True, plan directly to the canonical
                            ``home_qpos`` (no nearest-equivalent wrapping).
                            Equivalent joints whose encoder sits in a different
                            2π band rotate through the full raw difference,
                            concurrent with the rest of the arm, in a single
                            motion.  When False (default), wrap ``home_qpos``
                            to the nearest equivalent band first.
        report:             Optional mutable diagnostics dictionary. It is
                            cleared and populated with every attempted path and
                            its first rejection reason.
    """
    if report is None:
        report = {}
    else:
        report.clear()
    report["candidates"] = []

    if hand_safety_margin_m is None:
        hand_safety_margin_m = _arm_cfg.hand_safety_margin_m

    if use_canonical_target:
        # Canonical home uses raw absolute deltas; equivalent joints may rotate a full band.
        target = np.asarray(home_qpos, dtype=np.float64)
        # Use raw delta for the already-home check; wrapped deltas hide band mismatches.
        delta = float(np.max(np.abs(target - qpos)))
    else:
        # Wrap to the nearest equivalent to avoid unnecessary full-band motion.
        if planner is not None:
            target = planner.ik_mgr.nearest_equivalent_qpos(home_qpos, qpos)
            delta = float(np.max(np.abs(planner.ik_mgr.compute_qpos_delta(target, qpos))))
        else:
            # Without a planner, wrap using the URDF-mirrored arm limits.
            target = wrap_nearest_equivalent(
                home_qpos,
                qpos,
                _arm_cfg.joint_limit_lower,
                _arm_cfg.joint_limit_upper,
            )
            delta = float(np.max(np.abs(qpos - target)))
    if delta < np.deg2rad(0.15):
        report["status"] = "already_home"
        return None  # caller can send home_qpos directly

    have_collision = planner is not None and planner.planning_profile.check_self_collision
    _check_table = planner is not None and (
        bool(getattr(planner.collision_model, "has_table", False)) or table_z_surface_m is not None
    )

    def _check_safe(path: np.ndarray, candidate_name: str) -> bool:
        candidate: dict[str, Any] = {
            "name": candidate_name,
            "sample_count": len(path),
        }
        report["candidates"].append(candidate)

        if have_collision:
            assert planner is not None
            _path_check = getattr(
                planner.ik_mgr,
                "check_path_combined_collisions",
                planner.ik_mgr.check_path_collisions,
            )
            try:
                result = _path_check(path)
            except Exception as exc:
                candidate.update(safe=False, reason="collision_check_error", detail=str(exc))
                return False
            if result.get("path_collision", result.get("path_self_collision", False)):
                _collision = result.get("collision")
                _pairs = _collision.get("collision_pairs", []) if isinstance(_collision, dict) else []
                _reason = (
                    "environment_collision"
                    if any(pair.get("type") == "environment" for pair in _pairs)
                    else "self_collision"
                )
                candidate.update(
                    safe=False,
                    reason=_reason,
                    collision_waypoint_index=result.get("collision_waypoint_index"),
                    collision=result.get("collision"),
                )
                return False

        # Homing applies the configured world-frame EEF boundary explicitly.
        if planner is not None:
            try:
                for _segment_index, (_start, _end) in enumerate(zip(path[:-1], path[1:])):
                    if not planner.is_workspace_segment_safe(_start, _end):
                        candidate.update(
                            safe=False,
                            reason="workspace",
                            workspace_segment_index=_segment_index,
                        )
                        return False
            except (ValueError, RuntimeError) as exc:
                candidate.update(safe=False, reason="workspace_check_error", detail=str(exc))
                return False

        # Prefer calibrated table geometry for clearance checks.
        if _check_table:
            _minimum_clearance_m = float("inf")
            _table_samples: list[tuple[int, float, float, str]] = []
            for _waypoint_index, _wp in enumerate(path):
                try:
                    _clearance_m, _raw_table_measurement_m, _table_source = _table_clearance_m(
                        planner,  # type: ignore[arg-type]
                        _wp,
                        table_z_surface_m=table_z_surface_m,
                        hand_safety_margin_m=float(hand_safety_margin_m),
                    )
                except Exception as exc:
                    candidate.update(
                        safe=False,
                        reason="table_check_error",
                        table_waypoint_index=_waypoint_index,
                        detail=str(exc),
                    )
                    return False  # FK/frame failure → treat as unsafe
                if not np.isfinite(_clearance_m):
                    candidate.update(
                        safe=False,
                        reason="table_check_nonfinite",
                        table_waypoint_index=_waypoint_index,
                    )
                    return False
                _minimum_clearance_m = min(_minimum_clearance_m, _clearance_m)
                _table_samples.append(
                    (_waypoint_index, _clearance_m, _raw_table_measurement_m, _table_source)
                )
            _negative = [sample for sample in _table_samples if sample[1] < 0.0]
            if _negative:
                _starts_inside_soft_zone = _table_samples[0][1] < 0.0
                _negative_is_initial_prefix = all(sample[0] == index for index, sample in enumerate(_negative))
                _monotonic_escape = all(
                    current[1] >= previous[1] - 5e-4
                    for previous, current in zip(_table_samples[:-1], _table_samples[1:])
                    if previous[1] < 0.0
                )
                _reaches_clearance = _table_samples[-1][1] >= 0.0
                if not (
                    _starts_inside_soft_zone
                    and _negative_is_initial_prefix
                    and _monotonic_escape
                    and _reaches_clearance
                ):
                    _bad = _negative[0]
                    candidate.update(
                        safe=False,
                        reason="table_clearance",
                        table_waypoint_index=_bad[0],
                        table_check_source=_bad[3],
                        table_raw_measurement_m=float(_bad[2]),
                        clearance_m=float(_bad[1]),
                    )
                    return False
                candidate["table_soft_escape"] = True
            candidate["minimum_table_clearance_m"] = _minimum_clearance_m
        candidate["safe"] = True
        return True

    def _validate_milestones(milestones: np.ndarray, candidate_name: str) -> bool:
        samples = interpolate_waypoints(milestones, np.deg2rad(1.0))
        return _check_safe(samples, candidate_name)

    def _unique_milestones(*points: np.ndarray) -> np.ndarray:
        unique = [np.asarray(points[0], dtype=np.float64)]
        for point in points[1:]:
            point = np.asarray(point, dtype=np.float64)
            if float(np.max(np.abs(point - unique[-1]))) > 1e-9:
                unique.append(point)
        return np.asarray(unique, dtype=np.float64)

    # Dense samples validate the path; firmware receives only its endpoints.

    # Attempt 1: direct joint-space segment.
    direct_milestones = np.stack([qpos, target])
    if _validate_milestones(direct_milestones, "direct"):
        report.update(status="safe", selected_candidate="direct")
        return direct_milestones

    # Attempt 2: move shoulder/elbow first while keeping the wrist fixed.
    mid = qpos.copy()
    mid[_PROXIMAL_MASK] = target[_PROXIMAL_MASK]

    proximal_first = _unique_milestones(qpos, mid, target)
    if _validate_milestones(proximal_first, "proximal_first"):
        report.update(status="safe", selected_candidate="proximal_first")
        return proximal_first

    # Also try the wrist-first ordering for extended poses.
    distal_mid = qpos.copy()
    distal_mid[~_PROXIMAL_MASK] = target[~_PROXIMAL_MASK]
    distal_first = _unique_milestones(qpos, distal_mid, target)
    if _validate_milestones(distal_first, "distal_first"):
        report.update(status="safe", selected_candidate="distal_first")
        return distal_first

    # Use bounded joint-space RRT as a final candidate after heuristic paths fail.
    if planner is not None and hasattr(planner, "plan_joint_qpos_path"):
        try:
            planned = planner.plan_joint_qpos_path(target, qpos, planning_time_s=0.5)
        except (ValueError, RuntimeError) as exc:
            report["candidates"].append(
                {"name": "joint_qpos_rrt", "safe": False, "reason": "planner_error", "detail": str(exc)}
            )
        else:
            if planned.success and planned.qpos_path is not None and len(planned.qpos_path) >= 2:
                rrt_milestones = np.asarray(planned.qpos_path, dtype=np.float64)
                if float(np.max(np.abs(rrt_milestones[-1] - target))) > 1e-6:
                    rrt_milestones = np.concatenate([rrt_milestones, target.reshape((1, *ARM_JOINT_SHAPE))], axis=0)
                if _validate_milestones(rrt_milestones, "joint_qpos_rrt"):
                    report.update(status="safe", selected_candidate="joint_qpos_rrt")
                    return rrt_milestones
            else:
                report["candidates"].append(
                    {
                        "name": "joint_qpos_rrt",
                        "safe": False,
                        "reason": "planner_failed",
                        "detail": planned.reason,
                    }
                )

    report["status"] = "unsafe"
    return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)  # caller must hold position


def plan_band_alignment_path(
    wrapped_home: np.ndarray,
    canonical_home: np.ndarray,
    planner: "XArm7MotionPlanner | None" = None,
    *,
    table_z_surface_m: float | None = None,
    hand_safety_margin_m: float | None = None,
) -> np.ndarray | None:
    """Plan a collision-safe band-alignment path from *wrapped_home* to *canonical_home*.

    *wrapped_home* is ``home_qpos`` shifted by integer multiples of 2π on
    equivalent joints (J1/J3/J5/J7) to match the current encoder band — the
    arm is physically at the home pose but the encoder values differ.

    This function densely validates the 2π joint-space segment at 1°/step,
    then returns only its endpoints for Mode 0 MoveJoint execution.  Only the
    band-mismatched equivalent joints move; self-collision, workspace bounds,
    and orientation-aware hand/table clearance are checked on every sample.

    Returns:
        * ``None`` — no alignment needed (*wrapped_home* ≈ *canonical_home*).
        * ``(2, 7)`` execution milestones — densely validated safe alignment.
        * ``(0, 7)`` empty array — alignment needed but **no safe path**;
          the caller SHOULD keep the arm at *wrapped_home*.
    """
    if hand_safety_margin_m is None:
        hand_safety_margin_m = _arm_cfg.hand_safety_margin_m

    # Plan only when raw deltas show an equivalent-joint band mismatch.
    _eq_mask = (np.array(_arm_cfg.joint_limit_upper) - np.array(_arm_cfg.joint_limit_lower)) > 2.0 * np.pi
    _raw_delta_deg = np.rad2deg(np.abs(wrapped_home - canonical_home))
    if not np.any(_raw_delta_deg[_eq_mask] > 1.0):
        return None  # same band — no alignment needed

    # Interpolate the band-alignment rotation at 1° per step.
    path = interpolate_waypoints(np.stack([wrapped_home, canonical_home]), np.deg2rad(1.0))

    have_collision = planner is not None and planner.planning_profile.check_self_collision
    _check_table = planner is not None and (
        bool(getattr(planner.collision_model, "has_table", False)) or table_z_surface_m is not None
    )

    if have_collision:
        assert planner is not None
        _path_check = getattr(
            planner.ik_mgr,
            "check_path_combined_collisions",
            planner.ik_mgr.check_path_collisions,
        )
        try:
            result = _path_check(path)
        except Exception:
            return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)
        if result.get("path_collision", result.get("path_self_collision", False)):
            return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)

    if planner is not None:
        try:
            for _start, _end in zip(path[:-1], path[1:]):
                if not planner.is_workspace_segment_safe(_start, _end):
                    return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)
        except (ValueError, RuntimeError):
            return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)

    if _check_table:
        for _wp in path:
            try:
                _clearance_m, _, _ = _table_clearance_m(
                    planner,  # type: ignore[arg-type]
                    _wp,
                    table_z_surface_m=table_z_surface_m,
                    hand_safety_margin_m=float(hand_safety_margin_m),
                )
            except Exception:
                return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)
            if not np.isfinite(_clearance_m):
                return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)
            if _clearance_m < 0.0:
                return np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)

    return np.stack([wrapped_home, canonical_home])
