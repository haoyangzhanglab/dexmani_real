"""Path processing utilities — densification, interpolation, smoothing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.defaults import arm as _arm_cfg

if TYPE_CHECKING:
    from dexmani_real.planning.planner import XArm7MotionPlanner


# Proximal (shoulder/elbow) vs distal (wrist) joint mask for two-stage homing.
# J1-J4 move first (arm repositioning), then J5-J7 (wrist orientation).
_PROXIMAL_MASK = np.array([True, True, True, True, False, False, False], dtype=bool)


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
    """Wrap equivalent joints to the nearest +/-2*pi band of *reference*.

    Equivalent joints are those where the joint limit span exceeds 2*pi
    (J1, J3, J5, J7 on xArm7).  For each such joint, the value is shifted
    by an integer multiple of 2*pi so it is as close as possible to the
    corresponding reference value, then clipped to hardware limits.

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
    joint_range = hi - lo
    is_equiv = joint_range > 2.0 * np.pi

    if not np.any(is_equiv):
        return result

    ref = np.asarray(reference, dtype=np.float64)
    period = 2.0 * np.pi

    # ── k-bounds expansion (backported from ik_candidates.canonicalize_qpos) ──
    # When the physical arm is near a joint-limit boundary (e.g. +2π) and the
    # raw value is near the opposite limit (e.g. -2π), the ideal wrapping factor
    # k may push the result slightly past the limit; the final np.clip snaps it
    # back.  Without the ±1 expansion, k_max clips the wrapping factor and the
    # wrapped result ends up ~2π away from reference → "joint turning a full
    # circle" during teleop.
    lo_equiv = lo[is_equiv]
    hi_equiv = hi[is_equiv]
    res_equiv = result[is_equiv]
    ref_equiv = ref[is_equiv]
    k_min = np.ceil((lo_equiv - res_equiv) / period)
    k_max = np.floor((hi_equiv - res_equiv) / period)
    k = np.round((ref_equiv - res_equiv) / period)
    valid = k_min <= k_max
    k = np.where(valid, np.clip(k, k_min - 1, k_max + 1), 0.0)

    result[is_equiv] += k * period
    np.clip(result, lo, hi, out=result)
    return result


def plan_joint_home_path(
    qpos: np.ndarray,
    home_qpos: np.ndarray,
    planner: "XArm7MotionPlanner | None" = None,
    *,
    table_z_surface_m: float | None = None,
    hand_safety_margin_m: float | None = None,
) -> np.ndarray | None:
    """Plan a collision-safe joint-space path from *qpos* to *home_qpos*.

    Returns:
        * ``None`` — arm is already at home (delta < 0.5°); caller should
          send ``home_qpos`` directly (no-op).
        * ``(D, 7)`` dense waypoint array (D ≥ 2) — collision-safe path found.
        * ``(0, 7)`` empty array — **no safe path exists**; caller MUST NOT
          fall back to linear interpolation.  The arm should hold position
          and the operator should be alerted.

    Raises no exceptions.

    This function only reads the collision model — it does NOT need a live
    XArm7 connection.

    Args:
        qpos:              Current arm joint positions (7,).
        home_qpos:          Target home joint positions (7,).
        planner:            Planner with collision model.  When None, only
                            joint-limit-aware wrapping is performed.
        table_z_surface_m:  Table top Z in arm-base frame (metres).  When
                            provided, an additional EEF-z check ensures the
                            fingertips cannot reach below the table surface
                            along the planned path.  This complements the
                            Pinocchio self-collision check — mesh-based collision
                            detection can produce false positives for hand links
                            (coarse convex hulls), while the EEF-z check is a
                            simple, reliable geometric bound.
        hand_safety_margin_m:
                            Conservative estimate of the maximum vertical
                            distance from EEF to the lowest fingertip.
                            Defaults to ``arm.hand_safety_margin_m`` (0.05 m)
                            when ``None``.  Used with *table_z_surface_m*.
    """
    if hand_safety_margin_m is None:
        hand_safety_margin_m = _arm_cfg.hand_safety_margin_m

    # ── Wrap home_qpos to nearest equivalent of current qpos ──
    # Prevents interpolate_waypoints from generating up to 360° of unnecessary
    # rotation for equivalent joints (J1/J3/J5/J7 on xArm7, 720° range).
    # Wrapping home→qpos (not qpos→home) keeps all waypoints in the arm's
    # current encoder band — critical because Mode 6 firmware plans from the
    # physical encoder position to each waypoint target.
    if planner is not None:
        _home = planner.ik_mgr.nearest_equivalent_qpos(home_qpos, qpos)
        delta = float(np.max(np.abs(planner.ik_mgr.compute_qpos_delta(_home, qpos))))
    else:
        # Fallback when planner is unavailable: wrap to nearest equivalent using
        # the arm config limits (which mirror the URDF).
        _home = wrap_nearest_equivalent(
            home_qpos,
            qpos,
            _arm_cfg.joint_limit_lower,
            _arm_cfg.joint_limit_upper,
        )
        delta = float(np.max(np.abs(qpos - _home)))
    if delta < np.deg2rad(0.5):
        return None  # caller can send home_qpos directly

    have_collision = planner is not None and planner.planning_profile.check_self_collision
    _check_table = table_z_surface_m is not None and planner is not None

    def _check_safe(path: np.ndarray) -> bool:
        # ── Self-collision check (Pinocchio mesh-based) ──
        if have_collision:
            result = planner.ik_mgr.check_path_collisions(path)  # type: ignore[union-attr]
            if result.get("path_self_collision", False):
                return False

        # ── Table clearance check (EEF z-based, complements mesh check) ──
        # Pinocchio convex-hull collision meshes for hand links can be
        # conservatively large, producing false positives against the table
        # box.  The EEF-z check provides a simple, reliable bound: the
        # lowest fingertip is at most *hand_safety_margin_m* below the EEF.
        if _check_table:
            for _wp in path:
                try:
                    _eef_p = planner.kin.compute_eef_pose_world(_wp).p  # type: ignore[union-attr]
                except Exception:
                    return False  # FK failure → treat as unsafe
                if not np.all(np.isfinite(_eef_p)):
                    return False
                if float(_eef_p[2]) - hand_safety_margin_m < table_z_surface_m:  # type: ignore[operator]
                    return False  # hand may collide with table
        return True

    # ── Attempt 1: direct linear joint-space interpolation ──
    path = interpolate_waypoints(np.stack([qpos, _home]), np.deg2rad(1.0))
    if _check_safe(path):
        return path

    if not have_collision and not _check_table:
        return None

    # ── Attempt 2: two-stage detour (proximal → wrist) ──
    # Move shoulder/elbow joints to home first while keeping wrist fixed,
    # then move wrist joints.  This avoids the hand swinging through the body.
    mid = qpos.copy()
    mid[_PROXIMAL_MASK] = _home[_PROXIMAL_MASK]

    path1 = interpolate_waypoints(np.stack([qpos, mid]), np.deg2rad(1.0))
    path2_full = interpolate_waypoints(np.stack([mid, _home]), np.deg2rad(1.0))
    path2 = path2_full[1:]  # skip mid (already at end of path1)

    staged = np.concatenate([path1, path2], axis=0) if len(path2) > 0 else path1
    if _check_safe(staged):
        return staged

    return np.empty((0, 7), dtype=np.float64)  # sentinel: no safe path — caller MUST hold position


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

    This function generates a dense 1°/step joint-space path that rotates
    only the band-mismatched joints through exactly one 2π wrap, then checks
    self-collision and table clearance.  On a pure wrist-roll alignment
    (J7 only), the EEF position is invariant so the table check is a no-op;
    Pinocchio self-collision is the primary guard.

    Returns:
        * ``None`` — no alignment needed (*wrapped_home* ≈ *canonical_home*).
        * ``(D, 7)`` dense waypoint array (D ≥ 2) — safe alignment path.
        * ``(0, 7)`` empty array — alignment needed but **no safe path**;
          the caller SHOULD keep the arm at *wrapped_home*.
    """
    if hand_safety_margin_m is None:
        hand_safety_margin_m = _arm_cfg.hand_safety_margin_m

    # Only plan if the two home positions differ on equivalent joints.
    # Use RAW (unwrapped) delta — compute_qpos_delta wraps 2π apart to 0,
    # defeating the purpose of detecting band mismatches.
    # Equivalent joints: J1(idx=0), J3(2), J5(4), J7(6) — 720° range.
    _EQ_MASK = np.array([True, False, True, False, True, False, True])
    _raw_delta_deg = np.rad2deg(np.abs(wrapped_home - canonical_home))
    if not np.any(_raw_delta_deg[_EQ_MASK] > 1.0):
        return None  # same band — no alignment needed

    # Linear joint-space interpolation at 1°/step — this produces a pure
    # equivalent-joint rotation (e.g. J7: -360° → 0° at 1°/step = 360 steps).
    # The path is fully wrapping-aware because interpolate_waypoints computes
    # raw (unwrapped) deltas between waypoints.
    path = interpolate_waypoints(np.stack([wrapped_home, canonical_home]), np.deg2rad(1.0))

    have_collision = planner is not None and planner.planning_profile.check_self_collision
    _check_table = table_z_surface_m is not None and planner is not None

    # ── Safety checks (same as plan_joint_home_path._check_safe) ──
    if have_collision:
        result = planner.ik_mgr.check_path_collisions(path)  # type: ignore[union-attr]
        if result.get("path_self_collision", False):
            return np.empty((0, 7), dtype=np.float64)

    if _check_table:
        for _wp in path:
            try:
                _eef_p = planner.kin.compute_eef_pose_world(_wp).p  # type: ignore[union-attr]
            except Exception:
                return np.empty((0, 7), dtype=np.float64)
            if not np.all(np.isfinite(_eef_p)):
                return np.empty((0, 7), dtype=np.float64)
            if float(_eef_p[2]) - hand_safety_margin_m < table_z_surface_m:  # type: ignore[operator]
                return np.empty((0, 7), dtype=np.float64)

    return path
