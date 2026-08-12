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
    advance_run_generation,
    planner_action_safety_gate,
    publish_joint_targets,
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
from dexmani_real.utils.signal_utils import ema_smooth_pose

logger = get_logger(__name__)

_INITIAL_STATE_POLL_S = 0.05
_IK_WARNING_INTERVAL_S = 1.0


def _workspace(runtime: ResolvedRuntimeConfig) -> np.ndarray:
    bounds = runtime.policy.workspace
    return np.array(
        [[bounds.x_min, bounds.x_max], [bounds.y_min, bounds.y_max], [bounds.z_min, bounds.z_max]],
        dtype=np.float64,
    )


def _build_planner_and_gate(runtime: ResolvedRuntimeConfig) -> tuple[XArm7MotionPlanner, Any]:
    planner = XArm7MotionPlanner.create_default(
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=float(runtime.keyboard_teleop.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(runtime.keyboard_teleop.ik_max_pose_error_rot_rad),
        ),
        static_boxes=tuple(runtime.environment.static_boxes),
    )
    planner.workspace_bounds = _workspace(runtime)
    planner.set_hand_qpos(np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)))
    gate = planner_action_safety_gate(
        ActionSafetyGateConfig(
            arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
            arm_max_velocity_rad_s=float(np.deg2rad(runtime.arm.max_joint_velocity_deg_per_s)),
            hand_max_velocity_rad_s=(
                float(runtime.hand.max_delta_rad) * float(runtime.keyboard_teleop.control_hz)
                if runtime.hand.max_delta_rad is not None
                else float(np.deg2rad(runtime.hand.safety_gate_max_velocity_deg_per_s))
            ),
            require_geometry_checks=True,
        ),
        planner=planner,
        table_z_surface_m=float(runtime.arm.table_z_surface_m),
        hand_safety_margin_m=float(runtime.arm.hand_safety_margin_m),
        enable_table_check=False,
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

    heartbeats = [("arm", float(shared.arm_heartbeat_s.value))]
    if hand_enabled:
        heartbeats.append(("hand", float(shared.hand_heartbeat_s.value)))
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


def _hand_feedback_issue(state: dict[str, Any] | None, *, now_ns: int, max_age_s: float) -> str | None:
    if state is None:
        return "hand state ring is empty"
    return validate_hand_feedback(
        connected=state["connected"],
        error_state=state["error_state"],
        qpos_stale=state["qpos_stale"],
        state_valid=state["state_valid"],
        send_healthy=state["send_healthy"],
        read_healthy=state["read_healthy"],
        source_monotonic_ns=state["source_monotonic_ns"],
        now_monotonic_ns=now_ns,
        max_age_s=max_age_s,
        qpos=state["qpos"],
    )


def _read_initial_arm(shared: SharedStorage, runtime: ResolvedRuntimeConfig) -> dict[str, Any] | None:
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
    if not wait_subsystem_ready(shared, [("arm", shared.arm_ready, arm_timeout_s)], processes):
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
    while hand_process.is_alive() and not shared.hand_ready.is_set() and time.monotonic() < deadline_s:
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


def _publish_measured_quit_hold(
    shared: SharedStorage,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None,
    *,
    dt_s: float,
    prepare_timeout_s: float,
    apply_timeout_s: float,
    safety_gate: Any,
) -> bool:
    """Invalidate pending motion and wait for one measured hold to reach the SDKs."""
    advance_run_generation(shared)
    return (
        publish_joint_targets(
            shared,
            arm_qpos,
            hand_qpos,
            is_hold=True,
            prepare_timeout_s=prepare_timeout_s,
            dt_s=dt_s,
            safety_gate=safety_gate,
            wait_applied=True,
            apply_timeout_s=apply_timeout_s,
        )
        is not None
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
    ema_pos = target_pos.copy()
    ema_quat = target_quat.copy()

    recoverable_errors = frozenset(int(code) for code in runtime.arm.recoverable_errors)
    collision_errors = frozenset(int(code) for code in runtime.arm.collision_fault_errors)
    heartbeat_timeouts = dict(runtime.safety.heartbeat_timeouts)
    rate = RateManager(float(cfg.control_hz))
    state_failures = 0
    home_key_down = False
    motion_active = False
    tracking_fault_frames = 0
    frame = 0
    last_ik_warning_s = 0.0
    started_s = time.monotonic()

    print("Keyboard active: WASD/arrows/IJKL move, R home, Q quit, ESC e-stop")
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
                _set_fault(shared, f"cannot publish a measured quit hold: {feedback_issue}")
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
                _set_fault(shared, f"cannot publish a measured quit hold during arm error C{error_code}")
                return False
            tracking_fault_frames = 0
            continue
        if error_code != 0:
            category = "collision" if error_code in collision_errors else "controller"
            _set_fault(shared, f"arm {category} error C{error_code}")
            return False
        measured_hand_qpos: np.ndarray | None = None
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
            measured_hand_qpos = np.asarray(hand_state["qpos"], dtype=np.float64)
            planner.set_hand_qpos(measured_hand_qpos)

        if quit_requested:
            if not _publish_measured_quit_hold(
                shared,
                current_qpos,
                measured_hand_qpos,
                dt_s=dt_s,
                prepare_timeout_s=float(policy.action_prepare_timeout_s),
                apply_timeout_s=float(policy.action_apply_timeout_s),
                safety_gate=safety_gate,
            ):
                _set_fault(shared, "measured quit hold was not applied")
                return False
            if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                require_transition(shared, SafetyState.ARMED)
            return True

        home_pressed = keys.is_pressed("r")
        if home_pressed and not home_key_down:
            if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                require_transition(shared, SafetyState.ARMED)
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
                homing_max_speed_rad_s=float(np.deg2rad(runtime.arm.homing.max_speed_deg_s)),
                homing_target_timeout_s=float(runtime.arm.homing.target_timeout_s),
                arm_heartbeat_max_age_s=float(runtime.safety.heartbeat_timeouts["arm"]),
                preplan_velocity_rad_s=float(runtime.arm.homing.velocity_convergence_rad_s),
                result_tolerance_rad=float(runtime.arm.homing.convergence_rad),
                verbose=True,
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
            ema_pos, ema_quat = target_pos.copy(), target_quat.copy()
            if not home_ok:
                logger.warning("Return-home request was not executed")
            motion_active = False
            tracking_fault_frames = 0
            rate.reset()
            home_key_down = home_pressed
            continue
        home_key_down = home_pressed

        dx, drpy = eef_delta_from_keys(keys, float(cfg.delta_pos_m), float(cfg.delta_rpy_rad))
        moving = bool(np.any(dx != 0.0) or np.any(drpy != 0.0))
        if moving and not motion_active:
            require_transition(shared, SafetyState.RUNNING)
        elif not moving and motion_active:
            require_transition(shared, SafetyState.ARMED)
            held_pose = planner.kin.compute_eef_pose_world(previous_command)
            target_pos, target_quat = held_pose.p.copy(), held_pose.q.copy()
            ema_pos, ema_quat = target_pos.copy(), target_quat.copy()
        motion_active = moving
        if not moving:
            tracking_fault_frames = 0
            if frame % int(cfg.idle_interval_frames) == 0:
                pose = planner.kin.compute_eef_pose_world(current_qpos)
                print(f"[idle {time.monotonic() - started_s:.0f}s] eef={np.round(pose.p, 3)}m", flush=True)
            continue

        target_pos = np.clip(target_pos + dx, workspace[:, 0], workspace[:, 1])
        if np.any(drpy != 0.0):
            delta_quat = Rotation.from_euler("xyz", drpy).as_quat(scalar_first=True)
            target_quat = quat_multiply(delta_quat, target_quat)
        ik_pos, ik_quat = ema_smooth_pose(
            target_pos,
            target_quat,
            ema_pos,
            ema_quat,
            float(policy.ema.alpha_pos),
            float(policy.ema.alpha_rot),
        )
        ema_pos, ema_quat = ik_pos.copy(), ik_quat.copy()
        if float(cfg.cartesian_kp) > 0.0:
            measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
            position_error = target_pos - measured_pose.p
            if np.linalg.norm(position_error) > float(cfg.cartesian_deadband_m):
                ik_pos = np.clip(
                    ik_pos + float(cfg.cartesian_kp) * position_error,
                    workspace[:, 0],
                    workspace[:, 1],
                )

        result = planner.solve_teleop_ik(Pose(p=ik_pos, q=ik_quat), current_qpos, previous_command)
        if not result.success or result.qpos is None:
            tracking_fault_frames = 0
            now_s = time.monotonic()
            if now_s - last_ik_warning_s >= _IK_WARNING_INTERVAL_S:
                logger.warning("IK rejected target: %s", result.reason or "unknown")
                last_ik_warning_s = now_s
            measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
            target_pos, target_quat = measured_pose.p.copy(), measured_pose.q.copy()
            ema_pos, ema_quat = target_pos.copy(), target_quat.copy()
            continue

        published = publish_joint_targets(
            shared,
            result.qpos,
            prepare_timeout_s=float(policy.action_prepare_timeout_s),
            dt_s=dt_s,
            safety_gate=safety_gate,
            wait_applied=True,
            apply_timeout_s=float(policy.action_apply_timeout_s),
        )
        if published is None or published.arm_qpos is None:
            _set_fault(shared, "arm publish failed")
            return False
        previous_command = np.asarray(published.arm_qpos, dtype=np.float64).copy()

        tracking_error = float(np.max(np.abs(planner.ik_mgr.compute_qpos_delta(previous_command, current_qpos))))
        if tracking_error > float(cfg.tracking_fault_rad):
            tracking_fault_frames += 1
            if tracking_fault_frames >= int(cfg.tracking_fault_frames):
                _set_fault(shared, f"persistent tracking divergence ({tracking_error:.2f}rad)")
                return False
        else:
            tracking_fault_frames = 0

        if frame % int(cfg.status_interval_frames) == 0:
            measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
            print(
                f"[{time.monotonic() - started_s:.0f}s f={frame}] "
                f"eef={np.round(measured_pose.p, 3)}m target={np.round(target_pos, 3)}m",
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
        estop_callback=lambda: _set_fault(shared, "operator e-stop callback", estop=True),
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
        try:
            keys.stop()
        except Exception:
            _set_fault(shared, "keyboard listener cleanup failed")
            logger.error("Keyboard listener cleanup failed", exc_info=True)
            exit_code = 1
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
                logger.critical("child process remains alive; leaving SharedStorage linked", exc_info=True)
                exit_code = 1
            else:
                if exit_code == 0 and not shutdown_report.clean:
                    logger.error("verified shutdown invalidated the clean control exit: %s", shutdown_report)
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
    return exit_code


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Keyboard teleoperation for xArm7")
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not connect XHand; it must be absent or secured at configured home",
    )
    parser.add_argument("--config", type=Path, default=None, help="Validated experiment YAML")
    args = parser.parse_args(argv)
    try:
        runtime = resolve_runtime_config(
            yaml_path=args.config,
            cli_overrides={
                "policy.hand_enabled": False if args.no_hand else None,
            },
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"invalid experiment config: {exc}")
    return run_keyboard_experiment(runtime, no_hand=args.no_hand)


if __name__ == "__main__":
    raise SystemExit(main())
