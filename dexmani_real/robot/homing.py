"""Collision-checked arm homing commands shared by experiment entry points."""

from __future__ import annotations

import time
from collections.abc import Callable
from queue import Empty, Full
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm
from dexmani_real.utils.schema import ARM_JOINT_SHAPE
from dexmani_real.robot.arm_sdk import (
    ArmLoopConfig,
    _read_live_error_code,
    _require_sdk_ok,
)
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
            shared.set_heartbeat("policy", _loop_now_s)

        if _latch_operator_estop(shared, estop_requested):
            _abort_reason = "e-stop requested by operator"
            break

        _arm_heartbeat_s = shared.get_heartbeat("arm")
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

    from dexmani_real.policy.safety import advance_run_generation

    advance_run_generation(shared)
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

    # Plan a collision-safe path.  Prefer a single-phase path directly to the
    # canonical home_qpos (equivalent joints rotate concurrently with the rest
    # of the arm).  If no canonical candidate is safe, fall back to the proven
    # two-phase path: nearest-equivalent home, then a band-alignment rotation.
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
        # No safe single-phase path to canonical home.  Fall back to the
        # two-phase wrapped path + band alignment.
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

        # Align the nearest equivalent home to the canonical joint band.
        # plan_joint_home_path wraps home_qpos to the nearest 2π band of the
        # arm's current encoder position.  The returned waypoints end at this
        # wrapped position.  For strategy learning we need the arm at the
        # canonical home_qpos, so we append a collision-checked alignment
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

    if verbose and _waypoints is not None:
        _selected = _home_path_report.get("selected_candidate", "unknown")
        print(f"  arm: home path selected={_selected} milestones={len(_waypoints)}", flush=True)

    if _waypoints is None:
        _waypoints = np.empty((0, *ARM_JOINT_SHAPE), dtype=np.float64)

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


def _result_impl(
    request: HomeRequest,
    success: bool,
    reason: str,
    qpos: np.ndarray,
) -> HomeResult:
    return HomeResult(
        request_id=request.request_id,
        success=success,
        reason=reason,
        final_qpos=np.asarray(qpos, dtype=np.float64).copy(),
        completed_at_s=time.monotonic(),
    )


def _shared_abort_reason_impl(shared: Any) -> str | None:
    if shared is None:
        return None
    if not shared.is_running.value:
        return "shutdown requested"
    if shared.estop_request.value:
        return "e-stop requested"
    if shared.error_state.value:
        return "sticky error_state set during homing"
    if shared.safety_state.value == SafetyState.FAULT:
        return "FAULT during homing"
    return None


def _confirm_home_dwell_impl(
    arm: Any,
    request: HomeRequest,
    cfg: ArmLoopConfig,
    home_qpos: np.ndarray,
    shared: Any,
    current: np.ndarray,
    current_qvel: np.ndarray,
    failure_reason: str,
) -> HomeResult:
    if (
        float(np.max(np.abs(current - home_qpos))) > cfg.homing_convergence_rad
        or float(np.max(np.abs(current_qvel)))
        > cfg.homing_velocity_convergence_rad_s
    ):
        return _result_impl(request, False, failure_reason, current)
    stable_since = time.monotonic()
    while time.monotonic() - stable_since < cfg.homing_dwell_s:
        abort_reason = _shared_abort_reason_impl(shared)
        if abort_reason is not None:
            return _result_impl(request, False, abort_reason, current)
        time.sleep(min(cfg.homing_step_interval_s, cfg.homing_dwell_s))
        code, states = arm.get_joint_states(is_radian=True, num=3)
        if code != 0 or len(states) <= 1:
            return _result_impl(request, 
                False, "state/qvel unavailable during home dwell", current
            )
        current = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
        current_qvel = np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
        if (
            current.shape != ARM_JOINT_SHAPE
            or current_qvel.shape != ARM_JOINT_SHAPE
            or not np.all(np.isfinite(current))
            or not np.all(np.isfinite(current_qvel))
            or float(np.max(np.abs(current - home_qpos)))
            > cfg.homing_convergence_rad
            or float(np.max(np.abs(current_qvel)))
            > cfg.homing_velocity_convergence_rad_s
        ):
            return _result_impl(request, 
                False, "home dwell interrupted by position/velocity", current
            )
    return _result_impl(request, True, "already at canonical home and settled", current)


def _execute_mode0_milestones_impl(
    arm: Any,
    request: HomeRequest,
    cfg: ArmLoopConfig,
    home_qpos: np.ndarray,
    shared: Any,
    feedback_callback: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None,
    execution_targets: np.ndarray,
    current: np.ndarray,
) -> tuple[HomeResult, np.ndarray]:
    _overall_deadline = time.monotonic() + request.execution_timeout_s
    _milestone_tol = min(cfg.homing_convergence_rad, np.deg2rad(0.5))

    for _target_index, _target in enumerate(execution_targets, start=1):
        if shared is not None:
            _abort_reason = _shared_abort_reason_impl(shared)
            if _abort_reason is not None:
                return _result_impl(request, False, _abort_reason, current), current
            shared.set_heartbeat("arm", time.monotonic())
        if time.monotonic() >= _overall_deadline:
            return _result_impl(request, 
                False,
                f"overall timeout before milestone {_target_index}/{len(execution_targets)}",
                current,
            ), current

        _segment_start = current.copy()
        _segment_started_s = time.monotonic()
        try:
            _code = arm.set_servo_angle(
                angle=_target,
                is_radian=True,
                speed=cfg.homing_max_speed_rad_per_s,
                mvacc=cfg.joint_max_acc_rad_per_s2,
                wait=False,
                radius=None,
            )
        except Exception:
            logger.warning("run_planned_homing: milestone send failed", exc_info=True)
            return _result_impl(request, False, f"milestone {_target_index} send raised", current), current
        if _code != 0:
            return _result_impl(request, 
                False,
                f"milestone {_target_index} rejected (SDK code={_code})",
                current,
            ), current

        _segment_timeout_s = _estimate_homing_segment_timeout_s(
            _segment_start, _target, cfg
        )
        _segment_deadline = min(
            _overall_deadline, _segment_started_s + _segment_timeout_s
        )
        _stable_since_s: float | None = None
        while time.monotonic() < _segment_deadline:
            if shared is not None:
                _abort_reason = _shared_abort_reason_impl(shared)
                if _abort_reason is not None:
                    return _result_impl(request, False, _abort_reason, current), current
                shared.set_heartbeat("arm", time.monotonic())
            try:
                _state_code, _states = arm.get_joint_states(is_radian=True, num=3)
            except Exception:
                logger.warning(
                    "run_planned_homing: milestone state read raised", exc_info=True
                )
                return _result_impl(request, 
                    False,
                    f"state read raised at milestone {_target_index}",
                    current,
                ), current
            if _state_code != 0 or len(_states) == 0:
                return _result_impl(request, 
                    False,
                    f"state read failed at milestone {_target_index} (code={_state_code})",
                    current,
                ), current
            current = np.asarray(_states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
            if current.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(current)):
                return _result_impl(request, 
                    False, f"invalid state at milestone {_target_index}", current
                ), current
            if len(_states) <= 1:
                return _result_impl(request, 
                    False, f"qvel unavailable at milestone {_target_index}", current
                ), current
            qvel = np.asarray(_states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
            tau = (
                np.asarray(_states[2], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
                if len(_states) > 2
                else np.zeros(ARM_JOINT_SHAPE)
            )
            try:
                _controller_error = _read_live_error_code(arm)
            except Exception:
                logger.warning(
                    "run_planned_homing: live controller error read failed at milestone %d",
                    _target_index,
                    exc_info=True,
                )
                return _result_impl(
                    request,
                    False,
                    f"live error read failed at milestone {_target_index}",
                    current,
                ), current
            if _controller_error != 0:
                return _result_impl(request, 
                    False,
                    f"controller error C{_controller_error} at milestone {_target_index}",
                    current,
                ), current
            if feedback_callback is not None:
                try:
                    feedback_callback(
                        current.copy(), qvel.copy(), tau.copy(), _target.copy()
                    )
                except Exception:
                    logger.warning(
                        "run_planned_homing: feedback publication failed",
                        exc_info=True,
                    )
            if (
                float(np.max(np.abs(current - _target))) <= _milestone_tol
                and float(np.max(np.abs(qvel)))
                <= cfg.homing_velocity_convergence_rad_s
            ):
                if _stable_since_s is None:
                    _stable_since_s = time.monotonic()
                if time.monotonic() - _stable_since_s >= cfg.homing_dwell_s:
                    break
            else:
                _stable_since_s = None
            time.sleep(cfg.homing_step_interval_s)
        else:
            _error = np.abs(current - _target)
            _joint = int(np.argmax(_error))
            _elapsed_s = time.monotonic() - _segment_started_s
            if time.monotonic() >= _overall_deadline:
                _timeout_kind = "overall timeout"
            else:
                _timeout_kind = "convergence timeout"
            return _result_impl(request, 
                False,
                f"{_timeout_kind} at milestone {_target_index}/{len(execution_targets)} "
                f"after {_elapsed_s:.2f}s (J{_joint + 1} error={np.rad2deg(_error[_joint]):.2f}deg)",
                current,
            ), current

    _final_error = float(np.max(np.abs(current - home_qpos)))
    if _final_error > cfg.homing_convergence_rad:
        return _result_impl(request, 
            False, f"final error {np.rad2deg(_final_error):.2f}deg", current
        ), current
    return _result_impl(request, True, "canonical home reached", current), current


def run_planned_homing(
    arm: Any,
    request: HomeRequest,
    cfg: ArmLoopConfig | None = None,
    *,
    shared: Any = None,
    feedback_callback: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None
    ) = None,
) -> HomeResult:
    """Execute collision-validated milestones with the firmware joint planner.

    The caller densely checks every joint-space segment for collision, but only
    the sparse segment endpoints cross the process boundary.  Homing temporarily
    enters Mode 0 and uses unblended ``MoveJoint`` commands so the controller,
    rather than this process, owns the point-to-point trajectory.  Normal Mode 6
    teleoperation is restored before returning from healthy paths; E-stop,
    shutdown, and controller-fault paths stop instead.  Completion is based
    only on fresh encoder feedback; no state is fabricated on SDK read failure.
    """
    _cfg = cfg or ArmLoopConfig()

    def _result(success: bool, reason: str, qpos: np.ndarray) -> HomeResult:
        return _result_impl(request, success, reason, qpos)

    def _shared_abort_reason() -> str | None:
        return _shared_abort_reason_impl(shared)

    waypoints = np.asarray(request.waypoints, dtype=np.float64)
    home_qpos = np.asarray(request.final_qpos, dtype=np.float64)
    if (
        not isinstance(request.request_id, (int, np.integer))
        or int(request.request_id) <= 0
    ):
        return _result(False, "invalid request_id", np.full(ARM_JOINT_SHAPE, np.nan))
    if (
        waypoints.ndim != 2
        or waypoints.shape[1:] != ARM_JOINT_SHAPE
        or not np.all(np.isfinite(waypoints))
    ):
        return _result(
            False, "invalid waypoint array", np.full(ARM_JOINT_SHAPE, np.nan)
        )
    if home_qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(home_qpos)):
        return _result(False, "invalid final_qpos", np.full(ARM_JOINT_SHAPE, np.nan))
    if (
        not np.isfinite(request.execution_timeout_s)
        or request.execution_timeout_s <= 0.0
    ):
        return _result(
            False, "invalid execution timeout", np.full(ARM_JOINT_SHAPE, np.nan)
        )
    _lower = np.asarray(_cfg.joint_limit_lower, dtype=np.float64)
    _upper = np.asarray(_cfg.joint_limit_upper, dtype=np.float64)
    if len(waypoints) > 0 and not np.all((waypoints >= _lower) & (waypoints <= _upper)):
        return _result(
            False, "waypoint violates joint limits", np.full(ARM_JOINT_SHAPE, np.nan)
        )
    if len(waypoints) > 0 and float(np.max(np.abs(waypoints[-1] - home_qpos))) > 1e-6:
        return _result(
            False,
            "final milestone does not match canonical home",
            np.full(ARM_JOINT_SHAPE, np.nan),
        )

    try:
        code, states = arm.get_joint_states(is_radian=True, num=3)
        if code == 0 and len(states) > 0:
            current = np.asarray(states[0], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
            current_qvel = (
                np.asarray(states[1], dtype=np.float64)[: ARM_JOINT_SHAPE[0]]
                if len(states) > 1
                else np.full(ARM_JOINT_SHAPE, np.inf)
            )
        else:
            return _result(
                False,
                f"initial state read failed (code={code})",
                np.full(ARM_JOINT_SHAPE, np.nan),
            )
    except Exception:
        logger.warning("run_planned_homing: initial state read raised", exc_info=True)
        return _result(
            False, "initial state read raised", np.full(ARM_JOINT_SHAPE, np.nan)
        )
    if current.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(current)):
        return _result(
            False, "initial state is invalid", np.full(ARM_JOINT_SHAPE, np.nan)
        )

    def _confirm_home_dwell(failure_reason: str) -> HomeResult:
        return _confirm_home_dwell_impl(
            arm, request, _cfg, home_qpos, shared, current, current_qvel, failure_reason
        )

    if len(waypoints) == 0:
        return _confirm_home_dwell(
            "empty path while away from stationary canonical home"
        )
    if float(np.max(np.abs(current - waypoints[0]))) > _cfg.homing_convergence_rad:
        return _result(
            False, "current state moved too far from planned path start", current
        )

    _execution_targets = waypoints[1:]
    if len(_execution_targets) == 0:
        return _confirm_home_dwell(
            "single-point path is not at stationary canonical home"
        )
    _preflight_abort = _shared_abort_reason()
    if _preflight_abort is not None:
        return _result(False, _preflight_abort, current)

    def _execute_mode0_milestones() -> HomeResult:
        nonlocal current
        _home_result, current = _execute_mode0_milestones_impl(
            arm, request, _cfg, home_qpos, shared, feedback_callback,
            _execution_targets, current,
        )
        return _home_result

    # Mode 6 is designed for continuously changing online targets and its
    # per-joint velocity profiles need not be synchronous.  A planned homing
    # path instead uses Mode 0 MoveJoint.  Explicitly restore Mode 6 after
    # healthy entry/execution failures so the worker never silently changes
    # semantics; global-stop and controller-fault paths remain stopped.
    _mode_switch_attempted = False
    try:
        logger.info(
            "homing: entering Mode 0 MoveJoint (%d motion milestones, speed=%.1fdeg/s)",
            len(_execution_targets),
            np.rad2deg(_cfg.homing_max_speed_rad_per_s),
        )
        _mode_switch_attempted = True
        _require_sdk_ok("set_mode(0)", arm.set_mode(0))
        _require_sdk_ok("set_state(0) after Mode 0", arm.set_state(0))
    except Exception as exc:
        logger.error("run_planned_homing: failed to enter Mode 0", exc_info=True)
        _home_result = _result(False, f"Mode 0 entry failed: {exc}", current)
    else:
        _home_result = _execute_mode0_milestones()

    _restore_error: Exception | None = None
    _post_homing_abort = _shared_abort_reason()
    try:
        _controller_error_after_home = _read_live_error_code(arm)
    except Exception:
        # Fail-closed: without a live error read we do not restore Mode 6.
        _controller_error_after_home = -1
    _restore_mode6 = _post_homing_abort is None and _controller_error_after_home == 0
    if _mode_switch_attempted and _restore_mode6:
        try:
            _require_sdk_ok("restore set_mode(6)", arm.set_mode(6))
            _require_sdk_ok("restore set_state(0)", arm.set_state(0))
        except Exception as exc:
            _restore_error = exc
            logger.error("run_planned_homing: failed to restore Mode 6", exc_info=True)
    elif _mode_switch_attempted:
        _stop_reason = (
            _post_homing_abort or f"controller error C{_controller_error_after_home}"
        )
        try:
            _require_sdk_ok("stop after interrupted homing", arm.set_state(4))
        except Exception as exc:
            _restore_error = exc
            logger.error(
                "run_planned_homing: failed to stop after interrupted homing",
                exc_info=True,
            )
        if _home_result.success:
            _home_result = _result(
                False, f"homing interrupted after convergence: {_stop_reason}", current
            )
    if _restore_error is not None:
        _operation = "Mode 6 restore" if _restore_mode6 else "safe stop"
        return _result(
            False,
            f"{_home_result.reason}; {_operation} failed: {_restore_error}",
            current,
        )
    if _restore_mode6:
        logger.info("homing: restored Mode 6")
    return _home_result


def _estimate_homing_segment_timeout_s(
    start: np.ndarray, target: np.ndarray, cfg: ArmLoopConfig
) -> float:
    """Deadline for one firmware-planned milestone, including settle time."""
    delta_rad = float(np.max(np.abs(np.asarray(target) - np.asarray(start))))
    nominal_s = delta_rad / max(cfg.homing_max_speed_rad_per_s, 1e-6)
    return max(
        cfg.homing_target_timeout_s, 2.0 * nominal_s + cfg.homing_target_timeout_s
    )
