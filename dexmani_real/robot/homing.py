"""Collision-checked arm homing shared by experiment entry points.

The planner densely validates a joint-space path (self/table/environment
collision, joint limits) and returns sparse milestones.  The requester queues
``(waypoints, final_qpos)`` to the arm worker, which drives them as a
blocking ``XArm7.home()`` in Mode 0; completion is observed from the arm
state ring (a fresh Mode-6 frame at the canonical home), not from an RPC
acknowledgement.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from queue import Full
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm
from dexmani_real.utils.schema import ARM_JOINT_SHAPE
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_HOME_RESULT_POLL_S = 0.1
_ARM_HEARTBEAT_MAX_AGE_S = 1.0
_MIN_HOME_TIMEOUT_S = 10.0
_HOME_TIMEOUT_PADDING_S = 5.0
_RESULT_TIMEOUT_PADDING_S = 2.0


def _describe_band_diff(wrapped: np.ndarray, canonical: np.ndarray) -> str:
    """Describe which equivalent joints differ between wrapped and canonical home.

    Returns a short string like ``"J7:-360→0°"`` or ``"same band"``.
    """
    delta_deg = np.rad2deg(np.abs(wrapped - canonical))
    joint_names = {0: "J1", 2: "J3", 4: "J5", 6: "J7"}
    parts: list[str] = []
    for joint_index, joint_name in joint_names.items():
        if delta_deg[joint_index] > 1.0:
            parts.append(
                f"{joint_name}:{np.rad2deg(wrapped[joint_index]):.0f}→" f"{np.rad2deg(canonical[joint_index]):.0f}°"
            )
    return ", ".join(parts) if parts else "same band"


def _latch_operator_estop(shared: SharedStorage, callback: Callable[[], bool] | None) -> bool:
    if callback is None:
        return False
    try:
        requested = bool(callback())
    except Exception:
        logger.error("arm home e-stop callback failed", exc_info=True)
        requested = True
    if requested:
        shared.estop_request.value = True
    return requested


def _arm_heartbeat_issue(heartbeat_s: float, checked_s: float, max_age_s: float) -> str | None:
    """Return why an arm heartbeat is unusable, or ``None`` when fresh."""
    if not np.isfinite(heartbeat_s) or heartbeat_s <= 0.0:
        return "arm heartbeat is missing or invalid"
    if not np.isfinite(checked_s):
        return "arm heartbeat check time is invalid"
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        return "arm heartbeat maximum age is invalid"
    age_s = checked_s - heartbeat_s
    if age_s < 0.0:
        return f"arm heartbeat is {-age_s:.3f}s in the future"
    if age_s > max_age_s:
        return f"arm heartbeat stale ({age_s:.1f}s)"
    return None


def _estimate_home_timeout_s(
    waypoints: np.ndarray,
    *,
    max_speed_rad_s: float = np.deg2rad(arm.homing.max_speed_deg_s),
    target_timeout_s: float = arm.homing.target_timeout_s,
) -> float:
    """Deadline derived from milestone path length and feedback settle overhead."""
    if not np.isfinite(max_speed_rad_s) or max_speed_rad_s <= 0.0:
        raise ValueError("homing max speed must be finite and positive")
    if not np.isfinite(target_timeout_s) or target_timeout_s <= 0.0:
        raise ValueError("homing target timeout must be finite and positive")
    if len(waypoints) < 2:
        return _MIN_HOME_TIMEOUT_S
    segment_motion = np.max(np.abs(np.diff(waypoints, axis=0)), axis=1)
    nominal_s = float(np.sum(segment_motion)) / max_speed_rad_s
    moving_segments = int(np.count_nonzero(segment_motion > 1e-9))
    settle_s = moving_segments * target_timeout_s
    return max(_MIN_HOME_TIMEOUT_S, 2.0 * nominal_s + settle_s + _HOME_TIMEOUT_PADDING_S)


def _wait_for_prehome_state(
    shared: SharedStorage,
    *,
    newer_than_ns: int,
    timeout_s: float,
    max_velocity_rad_s: float,
    heartbeat: bool,
    arm_heartbeat_max_age_s: float,
    estop_requested: Callable[[], bool] | None,
) -> np.ndarray | None:
    """Wait for fresh stationary feedback after invalidating queued actions."""
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("pre-home settle timeout must be finite and positive")
    if not np.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0.0:
        raise ValueError("pre-home velocity threshold must be finite and positive")
    deadline_s = time.monotonic() + timeout_s
    consecutive_stationary = 0
    last_stationary_sequence = 0
    while time.monotonic() < deadline_s:
        loop_now_s = time.monotonic()
        if heartbeat:
            shared.set_heartbeat("policy", loop_now_s)
        if _latch_operator_estop(shared, estop_requested):
            return None
        if (
            not shared.is_running.value
            or shared.error_state.value
            or int(shared.safety_state.value) == int(SafetyState.FAULT)
        ):
            return None
        arm_heartbeat_s = shared.get_heartbeat("arm")
        heartbeat_checked_s = time.monotonic()
        if _arm_heartbeat_issue(arm_heartbeat_s, heartbeat_checked_s, arm_heartbeat_max_age_s) is not None:
            return None

        state_result = shared.arm_state_ring.read_latest()
        if state_result is not None:
            state, _publish_ns, sequence = state_result
            state_checked_ns = time.monotonic_ns()
            source_ns = int(state["source_monotonic_ns"][0])
            qpos = np.asarray(state["qpos"][0], dtype=np.float64)
            qvel = np.asarray(state["qvel"][0], dtype=np.float64)
            healthy = (
                newer_than_ns <= source_ns <= state_checked_ns
                and bool(state["state_valid"][0])
                and bool(state["connected"][0])
                and int(state["error_code"][0]) == 0
                and qpos.shape == ARM_JOINT_SHAPE
                and qvel.shape == ARM_JOINT_SHAPE
                and np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
            )
            if healthy and float(np.max(np.abs(qvel))) <= max_velocity_rad_s:
                if int(sequence) > last_stationary_sequence:
                    last_stationary_sequence = int(sequence)
                    consecutive_stationary += 1
                    if consecutive_stationary >= 2:
                        return qpos.copy()
            else:
                consecutive_stationary = 0
                last_stationary_sequence = 0
        time.sleep(_HOME_RESULT_POLL_S)
    return None


def _format_home_candidate_rejection(candidate: dict[str, Any]) -> str:
    """Format one path-candidate diagnostic without dumping large arrays."""
    name = str(candidate.get("name", "unknown"))
    reason = str(candidate.get("reason", "unknown"))
    if reason in ("self_collision", "environment_collision", "collision"):
        collision = candidate.get("collision") or {}
        pairs = collision.get("collision_pairs", []) if isinstance(collision, dict) else []
        pair_names = [f"{pair.get('link1', '?')}<->{pair.get('link2', '?')}" for pair in pairs[:2]]
        pair_text = ",".join(pair_names) if pair_names else "pair unavailable"
        return f"{name}: collision sample={candidate.get('collision_waypoint_index', '?')} ({pair_text})"
    if reason == "table_clearance":
        clearance_mm = 1000.0 * float(candidate.get("clearance_m", float("nan")))
        return (
            f"{name}: table_clearance sample={candidate.get('table_waypoint_index', '?')} "
            f"margin={clearance_mm:+.1f}mm"
        )
    if reason == "workspace":
        return f"{name}: workspace segment={candidate.get('workspace_segment_index', '?')}"
    detail = str(candidate.get("detail", "")).strip()
    return f"{name}: {reason}" + (f" ({detail})" if detail else "")


def _wait_for_home_completion(
    shared: SharedStorage,
    home_qpos: np.ndarray,
    *,
    newer_than_ns: int,
    timeout_s: float,
    tol_rad: float,
    settled_velocity_rad_s: float,
    heartbeat: bool,
    estop_requested: Callable[[], bool] | None,
    arm_heartbeat_max_age_s: float,
    verbose: bool,
) -> bool:
    """Block until the arm worker publishes a fresh stationary frame at ``home_qpos``.

    HOME is a blocking worker operation.  Completion is observed from the arm
    state ring: a frame published after the request whose ``qpos`` converged to
    ``home_qpos`` and whose ``qvel`` has settled.  Milestone frames published
    while the arm is still moving fail the velocity check, so only the settled
    end state (or an already-at-home arm) satisfies it.
    """
    if not np.isfinite(tol_rad) or tol_rad <= 0.0:
        raise ValueError("arm home result tolerance must be finite and positive")
    deadline = time.monotonic() + timeout_s
    abort_reason: str | None = None
    while time.monotonic() < deadline:
        now_s = time.monotonic()
        if heartbeat:
            shared.set_heartbeat("policy", now_s)
        if _latch_operator_estop(shared, estop_requested):
            abort_reason = "e-stop requested by operator"
            break
        if not shared.is_running.value:
            abort_reason = "shutdown requested"
            break
        if shared.error_state.value:
            abort_reason = "sticky error_state set"
            break
        if int(shared.safety_state.value) == int(SafetyState.FAULT):
            abort_reason = "safety state is FAULT"
            break
        if _arm_heartbeat_issue(
            shared.get_heartbeat("arm"), now_s, arm_heartbeat_max_age_s
        ) is not None:
            abort_reason = "arm heartbeat became stale during homing"
            break
        state_result = shared.arm_state_ring.read_latest()
        if state_result is not None:
            state, _publish_ns, _sequence = state_result
            if int(state["source_monotonic_ns"][0]) > newer_than_ns and bool(
                state["state_valid"][0]
            ):
                qpos = np.asarray(state["qpos"][0], dtype=np.float64)
                qvel = np.asarray(state["qvel"][0], dtype=np.float64)
                if (
                    qpos.shape == ARM_JOINT_SHAPE
                    and qvel.shape == ARM_JOINT_SHAPE
                    and np.all(np.isfinite(qpos))
                    and np.all(np.isfinite(qvel))
                    and float(np.max(np.abs(qpos - home_qpos))) < tol_rad
                    and float(np.max(np.abs(qvel))) <= settled_velocity_rad_s
                ):
                    if verbose:
                        print("  arm: home reached", flush=True)
                    return True
        time.sleep(_HOME_RESULT_POLL_S)
    if verbose:
        if abort_reason is not None:
            print(f"  arm: home wait aborted — {abort_reason}", flush=True)
        else:
            print(f"  arm: home acknowledgement timed out after {timeout_s:.1f}s", flush=True)
    return False


def send_arm_home(
    shared: SharedStorage,
    home_qpos: np.ndarray,
    *,
    planner: Any | None = None,
    table_z_surface_m: float = 0.0,
    current_qpos: np.ndarray | None = None,
    queue_timeout: float = arm.homing.request_queue_timeout_s,
    converge_timeout_s: float = arm.homing.convergence_timeout_s,
    state_max_age_s: float = arm.homing.state_max_age_s,
    heartbeat: bool = True,
    estop_requested: Callable[[], bool] | None = None,
    homing_max_speed_rad_s: float = np.deg2rad(arm.homing.max_speed_deg_s),
    homing_target_timeout_s: float = arm.homing.target_timeout_s,
    arm_heartbeat_max_age_s: float = _ARM_HEARTBEAT_MAX_AGE_S,
    preplan_velocity_rad_s: float = arm.homing.velocity_convergence_rad_s,
    result_tolerance_rad: float = arm.homing.convergence_rad,
    verbose: bool = True,
) -> bool:
    """Send the arm to home via a collision-safe path and wait for convergence.

    Encapsulates the full home sequence used by all entry points:
    1. Invalidate pending policy commands and wait for fresh stationary feedback.
    2. Densely validate collision-safe segments and retain sparse milestones.
    3. Queue ``(waypoints, final_qpos, generation)`` to ``arm_home_q``.
    4. Wait for a fresh Mode-6 frame at home via ``_wait_for_home_completion``.

    ``current_qpos`` is only an optional boundary-validation hint; planning
    always starts from post-invalidation feedback.  A planner is required.
    Missing state, planning errors, and unsafe paths fail closed.

    Returns True if home reached, False on timeout or error.
    """
    from dexmani_real.planning.path_utils import plan_band_alignment_path, plan_joint_home_path

    if _latch_operator_estop(shared, estop_requested):
        if verbose:
            print("  arm: homing cancelled — e-stop requested", flush=True)
        return False
    if not np.isfinite(result_tolerance_rad) or result_tolerance_rad <= 0.0:
        raise ValueError("arm home result tolerance must be finite and positive")

    # Reject HOME after a sticky fault; the arm loop no longer consumes actions.
    if not shared.is_running.value:
        if verbose:
            print("  arm: homing cancelled — shutdown in progress", flush=True)
        return False
    if shared.error_state.value or shared.safety_state.value == int(SafetyState.FAULT):
        if verbose:
            print("  arm: homing cancelled — system is in FAULT; restart after inspection", flush=True)
        return False
    if int(shared.safety_state.value) != int(SafetyState.ARMED):
        if verbose:
            print("  arm: homing cancelled — safety state is not ARMED", flush=True)
        return False

    if current_qpos is not None:
        current_qpos = np.asarray(current_qpos, dtype=np.float64)
        if current_qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(current_qpos)):
            if verbose:
                print("  arm: invalid current qpos hint — homing cancelled", flush=True)
            return False

    from dexmani_real.policy.safety import advance_run_generation

    home_generation = advance_run_generation(shared)
    _run_advanced_ns = time.monotonic_ns()
    _fresh_qpos = _wait_for_prehome_state(
        shared,
        newer_than_ns=_run_advanced_ns,
        timeout_s=max(float(converge_timeout_s), float(state_max_age_s)),
        max_velocity_rad_s=preplan_velocity_rad_s,
        heartbeat=heartbeat,
        arm_heartbeat_max_age_s=arm_heartbeat_max_age_s,
        estop_requested=estop_requested,
    )
    if _fresh_qpos is None:
        if verbose:
            print("  arm: no fresh stationary state after cancelling pending actions — homing cancelled", flush=True)
        return False
    current_qpos = _fresh_qpos

    if planner is None:
        if verbose:
            print("  arm: no collision planner — homing cancelled", flush=True)
        return False

    # Prefer a direct collision-safe path to canonical home_qpos.
    _home_path_report: dict[str, Any] = {}
    try:
        _waypoints = plan_joint_home_path(
            current_qpos,
            home_qpos,
            planner,
            table_z_surface_m=table_z_surface_m,
            use_canonical_target=True,
            report=_home_path_report,
        )
    except Exception as exc:
        logger.warning("send_arm_home: planning failed", exc_info=True)
        if verbose:
            print(f"  arm: home path planning failed — holding ({exc})", flush=True)
        return False

    if _latch_operator_estop(shared, estop_requested):
        if verbose:
            print("  arm: homing cancelled during path planning — e-stop requested", flush=True)
        return False

    if _waypoints is not None and len(_waypoints) == 0:
        # Fall back to the wrapped two-phase path when direct homing is unsafe.
        if verbose:
            _canonical_rejections = "; ".join(
                _format_home_candidate_rejection(candidate)
                for candidate in _home_path_report.get("candidates", [])
                if not candidate.get("safe", False)
            )
            print(
                "  arm: canonical home path rejected — falling back to wrapped+alignment"
                f" ({_canonical_rejections or 'no candidate diagnostics'})",
                flush=True,
            )
        try:
            _waypoints = plan_joint_home_path(
                current_qpos,
                home_qpos,
                planner,
                table_z_surface_m=table_z_surface_m,
                use_canonical_target=False,
                report=_home_path_report,
            )
        except Exception as exc:
            logger.warning("send_arm_home: fallback planning failed", exc_info=True)
            if verbose:
                print(f"  arm: fallback home path planning failed — holding ({exc})", flush=True)
            return False

        if _latch_operator_estop(shared, estop_requested):
            if verbose:
                print("  arm: homing cancelled during fallback planning — e-stop requested", flush=True)
            return False

        if _waypoints is not None and len(_waypoints) == 0:
            if verbose:
                _candidate_text = "; ".join(
                    _format_home_candidate_rejection(candidate)
                    for candidate in _home_path_report.get("candidates", [])
                    if not candidate.get("safe", False)
                )
                _qpos_text = np.array2string(np.rad2deg(current_qpos), precision=1, separator=",")
                print(
                    "  arm: no validated home-path candidate — holding\n"
                    f"       current_qpos_deg={_qpos_text}\n"
                    f"       rejected={_candidate_text or 'no candidate diagnostics'}",
                    flush=True,
                )
            return False

        _wrapped_home = (
            _waypoints[-1].copy()
            if _waypoints is not None and len(_waypoints) > 0
            else planner.ik_mgr.nearest_equivalent_qpos(home_qpos, current_qpos)
        )
        if _waypoints is None:
            _waypoints = np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)

        # Align the wrapped home path to the canonical joint band.
        try:
            _align_path = plan_band_alignment_path(_wrapped_home, home_qpos, planner, table_z_surface_m=table_z_surface_m)
        except Exception as exc:
            logger.warning("send_arm_home: band-alignment planning failed", exc_info=True)
            if verbose:
                print(f"  arm: band-alignment planning failed — holding ({exc})", flush=True)
            return False

        if _latch_operator_estop(shared, estop_requested):
            if verbose:
                print("  arm: homing cancelled during band alignment — e-stop requested", flush=True)
            return False

        if _align_path is not None:
            if len(_align_path) == 0:
                if verbose:
                    _desc = _describe_band_diff(_wrapped_home, home_qpos)
                    print(f"  arm: band-alignment UNSAFE ({_desc}) — holding", flush=True)
                return False
            # Keep the first alignment waypoint only when there is no main path.
            _tail = _align_path[1:] if len(_waypoints) > 0 else _align_path
            _waypoints = np.concatenate([_waypoints, _tail], axis=0)
            if verbose:
                _desc = _describe_band_diff(_wrapped_home, home_qpos)
                print(f"  arm: band-alignment appended ({len(_tail)} milestones, {_desc})", flush=True)

    if verbose and _waypoints is not None:
        _selected = _home_path_report.get("selected_candidate", "unknown")
        print(f"  arm: home path selected={_selected} milestones={len(_waypoints)}", flush=True)

    if _waypoints is None:
        _waypoints = np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)

    if _latch_operator_estop(shared, estop_requested):
        if verbose:
            print("  arm: homing cancelled before queue publication — e-stop requested", flush=True)
        return False
    # A generation change during planning/settle invalidates the stale path;
    # cancel without enqueuing so the arm never executes a superseded plan.
    if int(shared.run_generation.value) != home_generation:
        if verbose:
            print("  arm: homing cancelled — run generation changed during planning", flush=True)
        return False

    _queued_ns = time.monotonic_ns()
    try:
        shared.arm_home_q.put(
            (
                np.asarray(_waypoints, dtype=np.float64),
                np.asarray(home_qpos, dtype=np.float64).copy(),
                home_generation,
            ),
            timeout=queue_timeout,
        )
    except Full:
        if verbose:
            print("  arm: home queue is full — homing request was not queued", flush=True)
        return False
    except Exception:
        logger.warning("send_arm_home: failed to queue HOME request", exc_info=True)
        if verbose:
            print("  arm: failed to queue homing request", flush=True)
        return False

    return _wait_for_home_completion(
        shared,
        home_qpos,
        newer_than_ns=_queued_ns,
        timeout_s=_estimate_home_timeout_s(
            _waypoints,
            max_speed_rad_s=homing_max_speed_rad_s,
            target_timeout_s=homing_target_timeout_s,
        )
        + _RESULT_TIMEOUT_PADDING_S,
        tol_rad=result_tolerance_rad,
        settled_velocity_rad_s=preplan_velocity_rad_s,
        heartbeat=heartbeat,
        estop_requested=estop_requested,
        arm_heartbeat_max_age_s=arm_heartbeat_max_age_s,
        verbose=verbose,
    )
