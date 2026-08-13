#!/usr/bin/env python3
"""Keyboard teleoperation entry point with measured XHand feedback by default.

Uses production workers (arm_loop, hand_loop), the action safety gate, and
the SharedStorage IPC plane.  The full experiment logic lives here rather
than in the ``dexmani_real`` package — that keeps the package focused on
reusable library code and avoids accumulating entry-point logic.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.planning import Pose, TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.policy.safety import (
    ActionSafetyGateConfig,
    planner_action_safety_gate,
    publish_hand_home_and_wait_applied,
    publish_joint_targets,
    request_arm_decelerated_stop,
)
from dexmani_real.robot.arm_loop import ArmLoopConfig, arm_loop
from dexmani_real.robot.hand_process import HandProcessConfig, hand_loop
from dexmani_real.robot.homing import send_arm_home
from dexmani_real.robot.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.shm.shared_storage import (
    SharedStorage,
    SharedStorageConfig,
    read_arm_state_dict,
    read_hand_state_dict,
)
from dexmani_real.teleop.keyboard import (
    GlobalKeyState,
    eef_delta_from_keys,
    validate_arm_feedback,
    validate_hand_feedback,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

_INITIAL_STATE_POLL_S = 0.05
_IK_WARNING_INTERVAL_S = 1.0
_BOUNDARY_WARN_INTERVAL_S = 2.0


def _fmt_vec3(v: np.ndarray, decimals: int = 3) -> str:
    """Format a 3D vector as fixed-width comma-separated triple."""
    w = decimals + 3  # sign + digit + dot + decimals
    return f"({v[0]:{w}.{decimals}f}, {v[1]:{w}.{decimals}f}, {v[2]:{w}.{decimals}f})"


def _fmt_keys(keys: tuple[str, ...]) -> str:
    """Format held-key tuple for compact display."""
    if not keys:
        return "-"
    short = {"up": "UP", "down": "DN", "left": "LT", "right": "RT"}
    return "".join(short.get(k, k.upper()) for k in keys)


def _workspace(runtime: ResolvedRuntimeConfig) -> np.ndarray:
    bounds = runtime.policy.workspace
    return np.array(
        [
            [bounds.x_min, bounds.x_max],
            [bounds.y_min, bounds.y_max],
            [bounds.z_min, bounds.z_max],
        ],
        dtype=np.float64,
    )


def _build_planner_and_gate(
    runtime: ResolvedRuntimeConfig,
) -> tuple[XArm7MotionPlanner, Any]:
    planner = XArm7MotionPlanner.create_default(
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=float(runtime.keyboard_teleop.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(
                runtime.keyboard_teleop.ik_max_pose_error_rot_rad
            ),
        ),
        static_boxes=tuple(runtime.environment.static_boxes),
        table=runtime.environment.table,
    )
    planner.workspace_bounds = _workspace(runtime)
    planner.set_hand_qpos(
        np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
    )
    gate = planner_action_safety_gate(
        ActionSafetyGateConfig(
            arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
        ),
        planner=planner,
    )
    return planner, gate


def _runtime_issue(
    shared: SharedStorage,
    arm_process: Any,
    hand_process: Any | None,
    *,
    hand_enabled: bool,
    heartbeat_timeouts_s: dict[str, float],
) -> str | None:
    if shared.estop_request.value:
        return "e-stop is requested"
    if shared.error_state.value:
        return "a worker set the sticky error latch"
    if int(shared.safety_state.value) == int(SafetyState.FAULT):
        return "safety state is FAULT"
    if not arm_process.is_alive():
        return "arm worker exited"
    if hand_enabled and (hand_process is None or not hand_process.is_alive()):
        return "hand worker exited"

    heartbeats = [("arm", shared.get_heartbeat("arm"))]
    if hand_enabled:
        heartbeats.append(("hand", shared.get_heartbeat("hand")))
    now_s = time.monotonic()
    for name, heartbeat_s in heartbeats:
        timeout_s = float(heartbeat_timeouts_s[name])
        age_s = now_s - heartbeat_s
        if (
            not np.isfinite(heartbeat_s)
            or heartbeat_s <= 0.0
            or heartbeat_s > now_s
            or not np.isfinite(timeout_s)
            or timeout_s <= 0.0
            or age_s > timeout_s
        ):
            return f"{name} heartbeat stale ({age_s:.2f}s)"
    return None


def _hand_feedback_issue(
    state: dict[str, Any] | None, *, now_ns: int, max_age_s: float
) -> str | None:
    if state is None:
        return "hand state ring is empty"
    return validate_hand_feedback(
        connected=state["connected"],
        error_state=state["error_state"],
        state_valid=state["state_valid"],
        send_healthy=state["send_healthy"],
        read_healthy=state["read_healthy"],
        source_monotonic_ns=state["source_monotonic_ns"],
        now_monotonic_ns=now_ns,
        max_age_s=max_age_s,
        qpos=state["qpos"],
    )


def _read_initial_arm(
    shared: SharedStorage, runtime: ResolvedRuntimeConfig
) -> dict[str, Any] | None:
    deadline_s = time.monotonic() + float(runtime.safety.readiness_timeouts_s["arm"])
    while time.monotonic() < deadline_s:
        state = read_arm_state_dict(shared)
        if state is not None:
            issue = validate_arm_feedback(
                connected=state["connected"],
                state_valid=state["state_valid"],
                source_monotonic_ns=state["source_monotonic_ns"],
                now_monotonic_ns=time.monotonic_ns(),
                max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
                qpos=state["qpos"],
                qvel=state["qvel"],
                eef_pos=state["eef_pos"],
                eef_rot6d=state["eef_rot6d"],
            )
            if issue is None and state["error_code"] == 0:
                return state
        time.sleep(_INITIAL_STATE_POLL_S)
    return None


def _start_workers(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    context: Any,
    processes: list[Any],
    *,
    hand_requested: bool,
) -> tuple[Any, Any | None, bool]:
    arm_process = context.Process(
        target=arm_loop,
        args=(shared, ArmLoopConfig.from_runtime(runtime)),
        name="arm",
        daemon=False,
    )
    processes.append(arm_process)
    arm_process.start()
    arm_timeout_s = float(runtime.safety.readiness_timeouts_s["arm"])
    if not wait_subsystem_ready(
        shared, [("arm", shared.arm_ready, arm_timeout_s)], processes
    ):
        return arm_process, None, False

    if not hand_requested:
        return arm_process, None, False

    hand_process = context.Process(
        target=hand_loop,
        args=(
            shared,
            HandProcessConfig.from_runtime(runtime, startup_failure_is_fatal=True),
        ),
        name="hand",
        daemon=False,
    )
    processes.append(hand_process)
    hand_process.start()
    deadline_s = time.monotonic() + float(runtime.safety.readiness_timeouts_s["hand"])
    while (
        hand_process.is_alive()
        and not shared.hand_ready.is_set()
        and time.monotonic() < deadline_s
    ):
        time.sleep(_INITIAL_STATE_POLL_S)

    if not shared.hand_ready.is_set():
        return arm_process, hand_process, False

    hand_state = read_hand_state_dict(shared)
    if (
        _hand_feedback_issue(
            hand_state,
            now_ns=time.monotonic_ns(),
            max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
        )
        is not None
    ):
        return arm_process, hand_process, False
    return arm_process, hand_process, True


def _set_fault(shared: SharedStorage, reason: str, *, estop: bool = False) -> None:
    logger.error("Keyboard teleop fault: %s", reason)
    if estop:
        shared.estop_request.value = True
    shared.error_state.value = True
    transition(shared, SafetyState.FAULT)


def _stop_arm_motion(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    *,
    reason: str,
) -> np.ndarray | None:
    """Brake in controller State 6 and return its settled measured position."""
    return request_arm_decelerated_stop(
        shared,
        prepare_timeout_s=float(runtime.policy.action_prepare_timeout_s),
        apply_timeout_s=float(runtime.policy.action_apply_timeout_s),
        settle_velocity_rad_s=float(runtime.arm.homing.velocity_convergence_rad_s),
        reason=f"keyboard-{reason}",
        heartbeat=False,
    )


def _run_control_loop(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    safety_gate: Any,
    keys: GlobalKeyState,
    arm_process: Any,
    hand_process: Any | None,
    *,
    hand_enabled: bool,
) -> bool:
    """Run until Q (True) or a fault/e-stop (False)."""
    state = _read_initial_arm(shared, runtime)
    if state is None:
        _set_fault(shared, "initial arm feedback is unavailable or unhealthy")
        return False

    cfg = runtime.keyboard_teleop
    policy = runtime.policy
    workspace = _workspace(runtime)
    dt_s = 1.0 / float(cfg.control_hz)
    current_qpos = np.asarray(state["qpos"], dtype=np.float64)
    previous_command = current_qpos.copy()
    pose = planner.kin.compute_eef_pose_world(current_qpos)
    target_pos = pose.p.copy()
    target_quat = pose.q.copy()

    recoverable_errors = frozenset(int(code) for code in runtime.arm.recoverable_errors)
    collision_errors = frozenset(
        int(code) for code in runtime.arm.collision_fault_errors
    )
    heartbeat_timeouts = dict(runtime.safety.heartbeat_timeouts)
    rate = RateManager(float(cfg.control_hz))
    state_failures = 0
    home_key_down = False
    motion_active = False
    tracking_fault_frames = 0
    previous_active_keys: tuple[str, ...] | None = None
    blocked_keys: tuple[str, ...] | None = None
    frame = 0
    last_ik_warning_s = 0.0
    last_boundary_warn_s = 0.0
    started_s = time.monotonic()

    print(
        "Keyboard active (terminal input captured through shutdown): "
        "WASD/arrows/IJKL move, R home, Q quit, ESC e-stop"
    )
    while shared.is_running.value:
        rate.wait()
        frame += 1

        if keys.is_pressed("esc"):
            _set_fault(shared, "operator e-stop", estop=True)
            return False
        if not keys.healthy:
            _set_fault(shared, "keyboard listener exited", estop=True)
            return False

        issue = _runtime_issue(
            shared,
            arm_process,
            hand_process,
            hand_enabled=hand_enabled,
            heartbeat_timeouts_s=heartbeat_timeouts,
        )
        if issue is not None:
            _set_fault(shared, issue)
            return False
        quit_requested = keys.is_pressed("q")
        state = read_arm_state_dict(shared)
        feedback_issue: str | None
        if state is None:
            feedback_issue = "arm state ring is empty"
        else:
            current_qpos = np.asarray(state["qpos"], dtype=np.float64)
            feedback_issue = validate_arm_feedback(
                connected=state["connected"],
                state_valid=state["state_valid"],
                source_monotonic_ns=state["source_monotonic_ns"],
                now_monotonic_ns=time.monotonic_ns(),
                max_age_s=float(policy.arm_state_stale_threshold_s),
                qpos=current_qpos,
                qvel=state["qvel"],
                eef_pos=state["eef_pos"],
                eef_rot6d=state["eef_rot6d"],
            )
        if feedback_issue is not None:
            if quit_requested:
                _set_fault(
                    shared, f"cannot confirm a decelerated quit stop: {feedback_issue}"
                )
                return False
            state_failures += 1
            if state_failures >= int(policy.max_consecutive_errors):
                _set_fault(shared, feedback_issue)
                return False
            continue
        assert state is not None

        state_failures = 0
        error_code = int(state["error_code"])
        if error_code in recoverable_errors:
            if quit_requested:
                _set_fault(
                    shared,
                    f"cannot request a decelerated quit stop during arm error C{error_code}",
                )
                return False
            tracking_fault_frames = 0
            continue
        if error_code != 0:
            category = "collision" if error_code in collision_errors else "controller"
            _set_fault(shared, f"arm {category} error C{error_code}")
            return False
        if hand_enabled:
            hand_state = read_hand_state_dict(shared)
            hand_issue = _hand_feedback_issue(
                hand_state,
                now_ns=time.monotonic_ns(),
                max_age_s=float(heartbeat_timeouts["hand"]),
            )
            if hand_issue is not None:
                _set_fault(shared, hand_issue)
                return False
            assert hand_state is not None
            planner.set_hand_qpos(np.asarray(hand_state["qpos"], dtype=np.float64))

        if quit_requested:
            if _stop_arm_motion(shared, runtime, reason="quit") is None:
                _set_fault(shared, "decelerated quit stop did not settle")
                return False
            if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                require_transition(shared, SafetyState.ARMED)
            return True

        home_pressed = keys.is_pressed("r")
        if home_pressed and not home_key_down:
            if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                require_transition(shared, SafetyState.ARMED)
            hand_home_accepted = True
            if hand_enabled:
                hand_home = np.deg2rad(
                    np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)
                )
                hand_home_accepted = publish_hand_home_and_wait_applied(
                    shared,
                    hand_home,
                    command_lower_rad=np.asarray(
                        runtime.hand.qpos_min_rad, dtype=np.float64
                    ),
                    command_upper_rad=np.asarray(
                        runtime.hand.qpos_max_rad, dtype=np.float64
                    ),
                    mechanical_lower_rad=np.asarray(
                        runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
                    ),
                    mechanical_upper_rad=np.asarray(
                        runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
                    ),
                    max_command_delta_rad=runtime.hand.max_delta_rad,
                    timeout_s=float(runtime.hand.home_command_ack_timeout_s),
                    heartbeat=False,
                    abort_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
                )
                if hand_home_accepted:
                    planner.set_hand_qpos(hand_home)

            home_ok = False
            if hand_home_accepted:
                home_ok = send_arm_home(
                    shared,
                    np.asarray(runtime.arm.home_qpos, dtype=np.float64),
                    planner=planner,
                    table_z_surface_m=float(runtime.arm.table_z_surface_m),
                    current_qpos=current_qpos,
                    queue_timeout=float(runtime.arm.homing.request_queue_timeout_s),
                    converge_timeout_s=float(runtime.arm.homing.convergence_timeout_s),
                    state_max_age_s=float(runtime.arm.homing.state_max_age_s),
                    heartbeat=False,
                    estop_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
                    homing_max_speed_rad_s=float(
                        np.deg2rad(runtime.arm.homing.max_speed_deg_s)
                    ),
                    homing_target_timeout_s=float(runtime.arm.homing.target_timeout_s),
                    arm_heartbeat_max_age_s=float(
                        runtime.safety.heartbeat_timeouts["arm"]
                    ),
                    preplan_velocity_rad_s=float(
                        runtime.arm.homing.velocity_convergence_rad_s
                    ),
                    result_tolerance_rad=float(runtime.arm.homing.convergence_rad),
                    verbose=True,
                )
            else:
                logger.warning(
                    "Return-home cancelled: hand-home command was not accepted"
                )
            if shared.estop_request.value:
                _set_fault(shared, "operator e-stop during homing")
                return False
            refreshed = _read_initial_arm(shared, runtime)
            if refreshed is None:
                _set_fault(shared, "fresh arm feedback unavailable after homing")
                return False
            current_qpos = np.asarray(refreshed["qpos"], dtype=np.float64)
            previous_command = current_qpos.copy()
            pose = planner.kin.compute_eef_pose_world(current_qpos)
            target_pos, target_quat = pose.p.copy(), pose.q.copy()
            if not home_ok:
                logger.warning("Return-home request was not executed")
            motion_active = False
            tracking_fault_frames = 0
            rate.reset()
            home_key_down = home_pressed
            continue
        home_key_down = home_pressed

        active_keys = keys.pressed_keys()
        dx, drpy = eef_delta_from_keys(
            keys, float(cfg.delta_pos_m), float(cfg.delta_rpy_rad)
        )
        if active_keys != previous_active_keys:
            label = _fmt_keys(active_keys)
            if active_keys:
                print(
                    f"  [{label:8s}] δpos={_fmt_vec3(dx)}m  δrot={_fmt_vec3(drpy)}rad",
                    flush=True,
                )
            else:
                print("  [--------] released", flush=True)
            previous_active_keys = active_keys
        moving = bool(np.any(dx != 0.0) or np.any(drpy != 0.0))
        if blocked_keys is not None:
            if active_keys == blocked_keys:
                if frame % int(cfg.idle_interval_frames) == 0:
                    elapsed = time.monotonic() - started_s
                    pose = planner.kin.compute_eef_pose_world(current_qpos)
                    print(
                        f"{_fmt_keys(active_keys):8s} !  [{elapsed:4.0f}s f={frame:5d}] "
                        f"eef={_fmt_vec3(pose.p)}",
                        flush=True,
                    )
                continue
            blocked_keys = None
        # ── Idle: no keys pressed ──────────────────────────────────
        # Mode 6 is a position servo — the arm naturally converges to the
        # last commanded endpoint and holds it with full active stiffness.
        # No explicit stop or resume is needed; just stop publishing.
        if not moving:
            if motion_active:
                require_transition(shared, SafetyState.ARMED)
                motion_active = False
                rate.reset()
            tracking_fault_frames = 0
            # Track measured position during idle so the velocity check on
            # the next motion tick uses a fresh reference.
            previous_command = current_qpos.copy()
            if frame % int(cfg.idle_interval_frames) == 0:
                elapsed = time.monotonic() - started_s
                pose = planner.kin.compute_eef_pose_world(current_qpos)
                print(
                    f"idle    [{elapsed:4.0f}s f={frame:5d}] eef={_fmt_vec3(pose.p)}",
                    flush=True,
                )
            continue

        # ── Moving: keys are held ──────────────────────────────────
        if not motion_active:
            require_transition(shared, SafetyState.RUNNING)
            motion_active = True
            rate.reset()

        # Advance a virtual Cartesian command at delta/dt, but cap its lead
        # over fresh feedback.  This gives Mode 6 enough following error to
        # reach the requested speed without allowing an unbounded target.
        measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
        workspace_margin_m = float(cfg.workspace_command_margin_m)
        command_low = workspace[:, 0] + workspace_margin_m
        command_high = workspace[:, 1] - workspace_margin_m
        _desired_pos = target_pos + dx
        target_pos = np.clip(_desired_pos, command_low, command_high)
        boundary_indicator = ""
        _clipped = np.abs(_desired_pos - target_pos) > 1e-9
        if np.any(_clipped):
            _axis_names = ("x", "y", "z")
            _parts: list[str] = []
            for _i in range(3):
                if _clipped[_i]:
                    _side = "⁺" if _desired_pos[_i] > target_pos[_i] else "⁻"
                    _parts.append(f"{_axis_names[_i]}{_side}{target_pos[_i]:.3f}")
            boundary_indicator = "  " + " ".join(_parts)
            _now_s = time.monotonic()
            if _now_s - last_boundary_warn_s >= _BOUNDARY_WARN_INTERVAL_S:
                logger.warning("Workspace boundary: %s", " ".join(_parts))
                last_boundary_warn_s = _now_s
        position_lead = target_pos - measured_pose.p
        max_position_lead_m = float(cfg.command_lookahead_frames) * float(
            cfg.delta_pos_m
        )
        position_lead_norm = float(np.linalg.norm(position_lead))
        if position_lead_norm > max_position_lead_m:
            target_pos = measured_pose.p + position_lead * (
                max_position_lead_m / position_lead_norm
            )
        if np.any(drpy != 0.0):
            delta_quat = Rotation.from_euler("xyz", drpy).as_quat(scalar_first=True)
            target_quat = quat_multiply(delta_quat, target_quat)
        measured_rotation = Rotation.from_quat(measured_pose.q, scalar_first=True)
        target_rotation = Rotation.from_quat(target_quat, scalar_first=True)
        rotation_lead = (target_rotation * measured_rotation.inv()).as_rotvec()
        rotation_lead_norm = float(np.linalg.norm(rotation_lead))
        max_rotation_lead_rad = float(cfg.command_lookahead_frames) * float(
            cfg.delta_rpy_rad
        )
        if rotation_lead_norm > max_rotation_lead_rad:
            target_rotation = (
                Rotation.from_rotvec(
                    rotation_lead * (max_rotation_lead_rad / rotation_lead_norm)
                )
                * measured_rotation
            )
            target_quat = target_rotation.as_quat(scalar_first=True)

        result = planner.solve_teleop_ik(
            Pose(p=target_pos, q=target_quat), current_qpos, previous_command
        )
        if not result.success or result.qpos is None:
            tracking_fault_frames = 0
            now_s = time.monotonic()
            if now_s - last_ik_warning_s >= _IK_WARNING_INTERVAL_S:
                logger.warning("IK rejected target: %s", result.reason or "unknown")
                last_ik_warning_s = now_s
            # Don't stop the arm — the last valid Mode 6 endpoint is still
            # active and the arm is converging to a safe position.  Just
            # block this key combination until the user changes keys.
            blocked_keys = active_keys
            continue

        published = publish_joint_targets(
            shared,
            result.qpos,
            prepare_timeout_s=float(policy.action_prepare_timeout_s),
            dt_s=dt_s,
            safety_gate=safety_gate,
        )
        if published is None or published.arm_qpos is None:
            logger.warning(
                "Keyboard motion command rejected — blocked until keys change"
            )
            blocked_keys = active_keys
            continue
        previous_command = np.asarray(published.arm_qpos, dtype=np.float64).copy()

        tracking_error = float(
            np.max(
                np.abs(
                    planner.ik_mgr.compute_qpos_delta(previous_command, current_qpos)
                )
            )
        )
        if tracking_error > float(cfg.tracking_fault_rad):
            tracking_fault_frames += 1
            if tracking_fault_frames >= int(cfg.tracking_fault_frames):
                _set_fault(
                    shared, f"persistent tracking divergence ({tracking_error:.2f}rad)"
                )
                return False
        else:
            tracking_fault_frames = 0

        if frame % int(cfg.status_interval_frames) == 0:
            elapsed = time.monotonic() - started_s
            measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
            _err_m = float(np.linalg.norm(target_pos - measured_pose.p))
            print(
                f"{_fmt_keys(active_keys):8s} [{elapsed:4.0f}s f={frame:5d}] "
                f"eef={_fmt_vec3(measured_pose.p)} → {_fmt_vec3(target_pos)}  "
                f"Δ={_err_m:.3f}m{boundary_indicator}",
                flush=True,
            )
    return False


def _run_keyboard_session(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    context: Any,
    processes: list[Any],
    keys: GlobalKeyState,
    planner: XArm7MotionPlanner,
    safety_gate: Any,
    *,
    no_hand: bool,
) -> bool:
    """Start validated workers and run the control loop until a terminal request."""
    arm_process, hand_process, hand_enabled = _start_workers(
        shared,
        runtime,
        context,
        processes,
        hand_requested=bool(runtime.policy.hand_enabled) and not no_hand,
    )
    if not shared.arm_ready.is_set():
        logger.error("Arm worker did not become ready")
        return False
    if bool(runtime.policy.hand_enabled) and not no_hand and not hand_enabled:
        logger.error("XHand is required but did not become ready")
        return False
    if bool(runtime.policy.hand_enabled) and not no_hand and hand_process is not None:
        if hand_process.is_alive() and not shared.hand_ready.is_set():
            logger.error("XHand startup timed out in an indeterminate state")
            return False
    if shared.error_state.value:
        logger.error("A worker failed during startup")
        return False

    initial_state = _read_initial_arm(shared, runtime)
    if initial_state is None:
        logger.error("Initial arm feedback is unavailable or unhealthy")
        return False
    if hand_enabled:
        hand_state = read_hand_state_dict(shared)
        assert hand_state is not None
        planner.set_hand_qpos(hand_state["qpos"])

    keys.start()
    require_transition(shared, SafetyState.ARMED)
    # Actively servo the hand at its home position.  Without this the hand
    # stays at its power-up position and joints near the mechanical limit
    # (particularly J5) may drift or repeatedly trigger feedback-bound
    # tolerance warnings.
    if hand_enabled:
        hand_home = np.deg2rad(
            np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)
        )
        if publish_hand_home_and_wait_applied(
            shared,
            hand_home,
            command_lower_rad=np.asarray(
                runtime.hand.qpos_min_rad, dtype=np.float64
            ),
            command_upper_rad=np.asarray(
                runtime.hand.qpos_max_rad, dtype=np.float64
            ),
            mechanical_lower_rad=np.asarray(
                runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ),
            mechanical_upper_rad=np.asarray(
                runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ),
            max_command_delta_rad=runtime.hand.max_delta_rad,
            timeout_s=float(runtime.hand.home_command_ack_timeout_s),
            heartbeat=False,
            abort_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
        ):
            planner.set_hand_qpos(hand_home)
        else:
            logger.warning(
                "Startup hand-home command was not accepted; "
                "hand may be outside command limits"
            )
    print(
        f"Keyboard teleop: {runtime.keyboard_teleop.control_hz:g}Hz, "
        f"hand={'measured' if hand_enabled else 'assumed-home'}, config={runtime.sha256[:12]}"
    )
    return _run_control_loop(
        shared,
        runtime,
        planner,
        safety_gate,
        keys,
        arm_process,
        hand_process,
        hand_enabled=hand_enabled,
    )


def run_keyboard_experiment(
    runtime: ResolvedRuntimeConfig,
    *,
    no_hand: bool,
) -> int:
    if not bool(runtime.policy.hand_enabled) and not no_hand:
        logger.error("Hand-disabled operation must be acknowledged with --no-hand")
        return 2
    planner, safety_gate = _build_planner_and_gate(runtime)
    context = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix=f"dexmani_keyboard_{os.getpid()}",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=context,
    )
    processes: list[Any] = []
    keys = GlobalKeyState(
        suppress_echo=True,
        estop_callback=lambda: _set_fault(
            shared, "operator e-stop callback", estop=True
        ),
    )
    clean_exit = False
    exit_code = 1
    try:
        clean_exit = _run_keyboard_session(
            shared,
            runtime,
            context,
            processes,
            keys,
            planner,
            safety_gate,
            no_hand=no_hand,
        )
        exit_code = 0 if clean_exit else 1
    except KeyboardInterrupt:
        if int(shared.safety_state.value) != int(SafetyState.DISARMED):
            _set_fault(shared, "KeyboardInterrupt")
        exit_code = 130
    except Exception:
        _set_fault(shared, "unexpected main-process exception")
        logger.error("Keyboard teleop failed", exc_info=True)
        exit_code = 1
    finally:
        # Keep the listener and terminal suppression active until all device
        # processes and shared resources are closed. Restoring it here used to
        # expose the remaining 1–2 s shutdown window to the shell.
        keys.quiesce()
        started = [process for process in processes if process.pid is not None]
        if started:
            try:
                shutdown_report = shutdown_processes(
                    shared,
                    started,
                    graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                    disarm_if_clean=clean_exit and exit_code == 0,
                )
            except RuntimeError:
                logger.critical(
                    "child process remains alive; leaving SharedStorage linked",
                    exc_info=True,
                )
                exit_code = 1
            else:
                if exit_code == 0 and not shutdown_report.clean:
                    logger.error(
                        "verified shutdown invalidated the clean control exit: %s",
                        shutdown_report,
                    )
                    exit_code = 1
        else:
            try:
                if not shared.close():
                    _set_fault(shared, "SharedStorage cleanup was incomplete")
                    exit_code = 1
            except Exception:
                _set_fault(shared, "SharedStorage cleanup failed")
                logger.error("SharedStorage cleanup failed", exc_info=True)
                exit_code = 1
        try:
            if not keys.wait_for_release(timeout_s=2.0):
                logger.warning(
                    "Keyboard keys remained held while restoring terminal input"
                )
            keys.stop()
        except Exception:
            logger.error("Keyboard listener cleanup failed", exc_info=True)
            exit_code = 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Keyboard teleoperation for xArm7")
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not connect XHand; it must be absent or secured at configured home",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Validated experiment YAML"
    )
    args = parser.parse_args(argv)
    try:
        runtime = resolve_runtime_config(
            yaml_path=args.config,
            cli_overrides={
                "policy.hand_enabled": False if args.no_hand else None,
            },
        )
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        parser.error(f"invalid experiment config: {exc}")
    return run_keyboard_experiment(runtime, no_hand=args.no_hand)


if __name__ == "__main__":
    raise SystemExit(main())
