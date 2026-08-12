"""Collision-checked arm homing commands shared by experiment entry points."""

from __future__ import annotations

import time
from collections.abc import Callable
from queue import Empty, Full
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm
from dexmani_real.utils.schema import ARM_JOINT_SHAPE
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import HOME_SENTINEL, HomeRequest, HomeResult, SharedStorage
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_FAULT_RESULT_GRACE_S = 0.25
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


def wait_for_arm_home(
    shared: SharedStorage,
    home_qpos: np.ndarray,
    *,
    request_id: int,
    timeout_s: float = 20.0,
    tol_rad: float = arm.homing.convergence_rad,
    heartbeat: bool = False,
    estop_requested: Callable[[], bool] | None = None,
    arm_heartbeat_max_age_s: float = _ARM_HEARTBEAT_MAX_AGE_S,
    verbose: bool = True,
) -> bool:
    """Wait for the arm worker's acknowledgement for *request_id*."""
    if not np.isfinite(tol_rad) or tol_rad <= 0.0:
        raise ValueError("arm home result tolerance must be finite and positive")
    _deadline = time.monotonic() + timeout_s
    _abort_reason: str | None = None
    _fault_ack_deadline: float | None = None
    while time.monotonic() < _deadline:
        _loop_now_s = time.monotonic()
        if heartbeat:
            shared.policy_heartbeat_s.value = _loop_now_s

        if _latch_operator_estop(shared, estop_requested):
            _abort_reason = "e-stop requested by operator"
            break

        _arm_heartbeat_s = float(shared.arm_heartbeat_s.value)
        _heartbeat_checked_s = time.monotonic()
        _heartbeat_issue = _arm_heartbeat_issue(
            _arm_heartbeat_s,
            _heartbeat_checked_s,
            arm_heartbeat_max_age_s,
        )
        if _heartbeat_issue is not None:
            _abort_reason = _heartbeat_issue
            break

        _result = None
        try:
            _result = shared.arm_home_result_q.get(
                timeout=min(_HOME_RESULT_POLL_S, max(0.0, _deadline - _heartbeat_checked_s))
            )
        except Empty:
            pass
        if _result is not None and (not isinstance(_result, HomeResult) or _result.request_id != request_id):
            logger.warning("wait_for_arm_home: discarded stale/malformed result %r", _result)
            continue
        if isinstance(_result, HomeResult):
            _q = np.asarray(_result.final_qpos, dtype=np.float64)
            _converged = (
                _q.shape == home_qpos.shape
                and np.all(np.isfinite(_q))
                and float(np.max(np.abs(_q - home_qpos))) < tol_rad
            )
            if _result.success and _converged:
                if verbose:
                    print("  arm: home reached", flush=True)
                return True
            if verbose:
                print(f"  arm: home failed — {_result.reason}", flush=True)
            return False

        if not shared.is_running.value:
            _abort_reason = "shutdown requested"
            break
        _fault_reason: str | None = None
        if shared.error_state.value:
            _fault_reason = "sticky error_state set"
        elif shared.safety_state.value == int(SafetyState.FAULT):
            _fault_reason = "safety state is FAULT"
        if _fault_reason is not None:
            # Queue feeder delivery can lag the worker's error latch briefly.
            if _fault_ack_deadline is None:
                _fault_ack_deadline = min(_deadline, _heartbeat_checked_s + _FAULT_RESULT_GRACE_S)
            if _heartbeat_checked_s < _fault_ack_deadline:
                continue
            _abort_reason = _fault_reason
            break
    if verbose:
        if _abort_reason is not None:
            print(f"  arm: home wait aborted — {_abort_reason}", flush=True)
        else:
            print(f"  arm: home acknowledgement timed out after {timeout_s:.1f}s", flush=True)
    return False


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
            shared.policy_heartbeat_s.value = loop_now_s
        if _latch_operator_estop(shared, estop_requested):
            return None
        if (
            not shared.is_running.value
            or shared.error_state.value
            or int(shared.safety_state.value) == int(SafetyState.FAULT)
        ):
            return None
        arm_heartbeat_s = float(shared.arm_heartbeat_s.value)
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
    """Send arm to home via collision-safe path and wait for convergence.

    Encapsulates the full home sequence used by all entry points:
    1. Invalidate pending policy commands and wait for fresh stationary feedback.
    2. Densely validate collision-safe segments and retain sparse milestones.
    3. Queue a correlated ``HomeRequest`` to ``arm_action_q``.
    4. Wait for convergence via ``wait_for_arm_home``.

    ``current_qpos`` is only an optional boundary-validation hint; planning
    always starts from post-invalidation feedback. A planner is required.
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

    # HOME is a motion command, not a fault-recovery command.  arm_loop stops
    # consuming actions after a sticky fault, so rejecting here prevents
    # impossible requests from filling the bounded action queue.
    if not shared.is_running.value:
        if verbose:
            print("  arm: homing cancelled — shutdown in progress", flush=True)
        return False
    if shared.error_state.value or shared.safety_state.value == int(SafetyState.FAULT):
        if verbose:
            print("  arm: homing cancelled — system is in FAULT; restart after inspection", flush=True)
        return False
    if shared.safety_state.value not in (int(SafetyState.ARMED), int(SafetyState.RUNNING)):
        if verbose:
            print("  arm: homing cancelled — safety state is not ARMED/RUNNING", flush=True)
        return False

    if current_qpos is not None:
        current_qpos = np.asarray(current_qpos, dtype=np.float64)
        if current_qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(current_qpos)):
            if verbose:
                print("  arm: invalid current qpos hint — homing cancelled", flush=True)
            return False

    from dexmani_real.policy.safety import advance_policy_epoch

    advance_policy_epoch(shared)
    _epoch_advanced_ns = time.monotonic_ns()
    _fresh_qpos = _wait_for_prehome_state(
        shared,
        newer_than_ns=_epoch_advanced_ns,
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

    # Plan a collision-safe path to the nearest equivalent home.
    _home_path_report: dict[str, Any] = {}
    try:
        _waypoints = plan_joint_home_path(
            current_qpos,
            home_qpos,
            planner,
            table_z_surface_m=table_z_surface_m,
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

    if verbose and _waypoints is not None:
        _selected = _home_path_report.get("selected_candidate", "unknown")
        print(f"  arm: home path selected={_selected} milestones={len(_waypoints)}", flush=True)

    _wrapped_home = (
        _waypoints[-1].copy()
        if _waypoints is not None and len(_waypoints) > 0
        else planner.ik_mgr.nearest_equivalent_qpos(home_qpos, current_qpos)
    )
    if _waypoints is None:
        _waypoints = np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)

    # Align the nearest equivalent home to the canonical joint band.
    # plan_joint_home_path wraps home_qpos to the nearest 2π band of the
    # arm's current encoder position.  The returned waypoints end at this
    # wrapped position (_home).  For strategy learning we need the arm at
    # the canonical home_qpos, so we append a collision-checked alignment
    # segment that rotates only the band-mismatched equivalent joints.
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

    # Queue the validated path.
    _request_id = time.monotonic_ns()
    _execution_timeout_s = max(
        float(converge_timeout_s),
        _estimate_home_timeout_s(
            _waypoints,
            max_speed_rad_s=homing_max_speed_rad_s,
            target_timeout_s=homing_target_timeout_s,
        ),
    )
    # A prior caller may have abandoned a result.  Homing is serialized, so it
    # is safe to drain stale acknowledgements before publishing the new ID.
    while True:
        try:
            shared.arm_home_result_q.get_nowait()
        except Empty:
            break
    if _latch_operator_estop(shared, estop_requested):
        if verbose:
            print("  arm: homing cancelled before queue publication — e-stop requested", flush=True)
        return False
    try:
        _request = HomeRequest(
            request_id=_request_id,
            waypoints=np.asarray(_waypoints, dtype=np.float64),
            final_qpos=np.asarray(home_qpos, dtype=np.float64).copy(),
            execution_timeout_s=_execution_timeout_s,
        )
        shared.arm_action_q.put((HOME_SENTINEL, _request), timeout=queue_timeout)
    except Full:
        if verbose:
            print("  arm: action queue is full — homing request was not queued", flush=True)
        return False
    except Exception:
        logger.warning("send_arm_home: failed to queue HOME request", exc_info=True)
        if verbose:
            print("  arm: failed to queue homing request", flush=True)
        return False

    return wait_for_arm_home(
        shared,
        home_qpos,
        request_id=_request_id,
        timeout_s=_execution_timeout_s + _RESULT_TIMEOUT_PADDING_S,
        tol_rad=result_tolerance_rad,
        heartbeat=heartbeat,
        estop_requested=estop_requested,
        arm_heartbeat_max_age_s=arm_heartbeat_max_age_s,
        verbose=verbose,
    )
