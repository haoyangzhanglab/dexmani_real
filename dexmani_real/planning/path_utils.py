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
) -> np.ndarray | None:
    """Plan a collision-safe joint-space path from *qpos* to *home_qpos*.

    Returns a dense (D×7) waypoint array, or ``None`` when the arm is already
    close enough that no waypoints are needed.  Raises no exceptions — on any
    failure to plan, returns ``None`` (caller falls back to linear interpolation).

    This function only reads the collision model — it does NOT need a live
    XArm7 connection.
    """
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
            home_qpos, qpos,
            _arm_cfg.joint_limit_lower, _arm_cfg.joint_limit_upper,
        )
        delta = float(np.max(np.abs(qpos - _home)))
    if delta < np.deg2rad(0.5):
        return None  # caller can send home_qpos directly

    have_collision = (
        planner is not None
        and planner.planning_profile.check_self_collision
    )

    def _check_safe(path: np.ndarray) -> bool:
        if not have_collision:
            return True
        result = planner.ik_mgr.check_path_collisions(path)  # type: ignore[union-attr]
        return not result.get("path_self_collision", False)

    # ── Attempt 1: direct linear joint-space interpolation ──
    path = interpolate_waypoints(np.stack([qpos, _home]), np.deg2rad(1.0))
    if _check_safe(path):
        return path

    if not have_collision:
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

    return None  # no safe path — caller falls back to direct linear interpolation
