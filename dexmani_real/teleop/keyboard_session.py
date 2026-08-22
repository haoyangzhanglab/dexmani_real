"""Keyboard teleoperation session lifecycle and real-time control flow.

The CLI remains in ``examples/keyboard_teleop.py``. This module owns the
concrete keyboard session, including its workers, safety state, and control
loop without adding another abstraction layer.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.config.runtime import ArmLoopConfig, ResolvedRuntimeConfig
from dexmani_real.planning import Pose, TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.policy.safety import (
    planner_action_safety_gate,
    publish_hand_home_and_wait_applied,
    publish_joint_targets,
)
from dexmani_real.robot.arm_loop import arm_loop
from dexmani_real.robot.hand_process import hand_loop
from dexmani_real.robot.homing import ArmHomeConfig, execute_arm_home
from dexmani_real.robot.safety import (
    SafetyState,
    advance_run_generation,
    require_transition,
    transition,
)
from dexmani_real.runtime.processes import WorkerSpec, build_processes, start_processes
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.shm.shared_storage import (
    SharedStorage,
    SharedStorageConfig,
    read_arm_state_dict,
    read_hand_state_dict,
)
from dexmani_real.teleop.keyboard import GlobalKeyState, eef_delta_from_keys
from dexmani_real.utils.hand_health import validate_arm_feedback, validate_hand_feedback
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
        # Match VR teleoperation: retain self/static-box checks but do not
        # reject intentional tabletop contact.
        table=None,
    )
    planner.workspace_bounds = _workspace(runtime)
    planner.set_hand_qpos(
        np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
    )
    gate = planner_action_safety_gate(
        planner=planner,
        arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
        arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
        hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
        hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
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
        state_valid=state["state_valid"],
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
    arm_spec = WorkerSpec(
        "arm", arm_loop, (shared, ArmLoopConfig.from_runtime(runtime)), ready_name="arm"
    )
    arm_process = build_processes(context, [arm_spec])[0]
    processes.append(arm_process)
    start_processes([arm_process])
    arm_timeout_s = float(runtime.safety.readiness_timeouts_s["arm"])
    if not wait_subsystem_ready(shared, [("arm", arm_timeout_s)], processes):
        return arm_process, None, False

    if not hand_requested:
        return arm_process, None, False

    hand_spec = WorkerSpec(
        "hand",
        hand_loop,
        (shared, runtime.hand),
        ready_name="hand",
    )
    hand_process = build_processes(context, [hand_spec])[0]
    processes.append(hand_process)
    start_processes([hand_process])
    deadline_s = time.monotonic() + float(runtime.safety.readiness_timeouts_s["hand"])
    while (
        hand_process.is_alive()
        and not shared.is_ready("hand")
        and time.monotonic() < deadline_s
    ):
        time.sleep(_INITIAL_STATE_POLL_S)

    if not shared.is_ready("hand"):
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


def _keyboard_command_anchor(
    planner: XArm7MotionPlanner,
    measured_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild joint and Cartesian command baselines from measured feedback."""
    qpos = np.asarray(measured_qpos, dtype=np.float64).copy()
    pose = planner.kin.compute_eef_pose_world(qpos)
    return qpos, pose.p.copy(), pose.q.copy()


@dataclass(frozen=True)
class _KeyboardFeedback:
    arm_state: dict[str, Any] | None
    arm_qpos_rad: np.ndarray | None
    hand_qpos_rad: np.ndarray | None
    issue: str | None
    retryable: bool = False


@dataclass(frozen=True)
class _KeyboardTargetUpdate:
    target_pos_world_m: np.ndarray
    target_quat_wxyz: np.ndarray
    boundary_text: str


class _KeyboardPublishStatus(str, Enum):
    PUBLISHED = "published"
    IK_REJECTED = "ik_rejected"
    SAFETY_REJECTED = "safety_rejected"


@dataclass(frozen=True)
class _KeyboardPublishResult:
    status: _KeyboardPublishStatus
    arm_qpos_rad: np.ndarray | None = None
    detail: str = ""


def _read_keyboard_feedback(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    *,
    hand_enabled: bool,
) -> _KeyboardFeedback:
    """Read and validate one arm/hand feedback snapshot without policy changes."""
    arm_state = read_arm_state_dict(shared)
    if arm_state is None:
        return _KeyboardFeedback(
            None,
            None,
            None,
            "arm state ring is empty",
            retryable=True,
        )

    arm_qpos_rad = np.asarray(arm_state["qpos"], dtype=np.float64)
    arm_issue = validate_arm_feedback(
        connected=arm_state["connected"],
        state_valid=arm_state["state_valid"],
        source_monotonic_ns=arm_state["source_monotonic_ns"],
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
        qpos=arm_qpos_rad,
        qvel=arm_state["qvel"],
    )
    if arm_issue is not None:
        return _KeyboardFeedback(
            arm_state,
            arm_qpos_rad,
            None,
            arm_issue,
            retryable=True,
        )
    error_code = int(arm_state["error_code"])
    if error_code != 0:
        return _KeyboardFeedback(
            arm_state,
            arm_qpos_rad,
            None,
            f"arm controller error C{error_code}",
        )
    if not hand_enabled:
        return _KeyboardFeedback(arm_state, arm_qpos_rad, None, None)

    hand_state = read_hand_state_dict(shared)
    hand_issue = _hand_feedback_issue(
        hand_state,
        now_ns=time.monotonic_ns(),
        max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
    )
    if hand_issue is not None:
        return _KeyboardFeedback(arm_state, arm_qpos_rad, None, hand_issue)
    assert hand_state is not None
    return _KeyboardFeedback(
        arm_state,
        arm_qpos_rad,
        np.asarray(hand_state["qpos"], dtype=np.float64),
        None,
    )


def _compute_keyboard_target_update(
    measured_pose_world: Pose,
    target_pos_world_m: np.ndarray,
    target_quat_wxyz: np.ndarray,
    delta_pos_world_m: np.ndarray,
    delta_rpy_rad: np.ndarray,
    workspace_world_m: np.ndarray,
    *,
    workspace_margin_m: float,
    command_lookahead_frames: float,
    position_step_m: float,
    rotation_step_rad: float,
) -> _KeyboardTargetUpdate:
    """Advance and bound the virtual keyboard target without side effects."""
    command_low_world_m = workspace_world_m[:, 0] + workspace_margin_m
    command_high_world_m = workspace_world_m[:, 1] - workspace_margin_m
    desired_pos_world_m = target_pos_world_m + delta_pos_world_m
    bounded_pos_world_m = np.clip(
        desired_pos_world_m,
        command_low_world_m,
        command_high_world_m,
    )

    clipped = np.abs(desired_pos_world_m - bounded_pos_world_m) > 1e-9
    boundary_parts: list[str] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        if clipped[axis_index]:
            side = (
                "⁺"
                if desired_pos_world_m[axis_index] > bounded_pos_world_m[axis_index]
                else "⁻"
            )
            boundary_parts.append(
                f"{axis_name}{side}{bounded_pos_world_m[axis_index]:.3f}"
            )

    position_lead_world_m = bounded_pos_world_m - measured_pose_world.p
    max_position_lead_m = command_lookahead_frames * position_step_m
    position_lead_norm_m = float(np.linalg.norm(position_lead_world_m))
    if position_lead_norm_m > max_position_lead_m > 0.0:
        bounded_pos_world_m = measured_pose_world.p + position_lead_world_m * (
            max_position_lead_m / position_lead_norm_m
        )

    bounded_quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float64).copy()
    if np.any(delta_rpy_rad != 0.0):
        delta_quat_wxyz = Rotation.from_euler("xyz", delta_rpy_rad).as_quat(
            scalar_first=True
        )
        bounded_quat_wxyz = quat_multiply(delta_quat_wxyz, bounded_quat_wxyz)

    measured_rotation = Rotation.from_quat(
        measured_pose_world.q,
        scalar_first=True,
    )
    target_rotation = Rotation.from_quat(bounded_quat_wxyz, scalar_first=True)
    rotation_lead_rad = (target_rotation * measured_rotation.inv()).as_rotvec()
    rotation_lead_norm_rad = float(np.linalg.norm(rotation_lead_rad))
    max_rotation_lead_rad = command_lookahead_frames * rotation_step_rad
    if rotation_lead_norm_rad > max_rotation_lead_rad > 0.0:
        target_rotation = (
            Rotation.from_rotvec(
                rotation_lead_rad * (max_rotation_lead_rad / rotation_lead_norm_rad)
            )
            * measured_rotation
        )
        bounded_quat_wxyz = target_rotation.as_quat(scalar_first=True)

    return _KeyboardTargetUpdate(
        target_pos_world_m=np.asarray(bounded_pos_world_m, dtype=np.float64),
        target_quat_wxyz=np.asarray(bounded_quat_wxyz, dtype=np.float64),
        boundary_text=" ".join(boundary_parts),
    )


def _run_keyboard_home(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    keys: GlobalKeyState,
    current_qpos_rad: np.ndarray,
    *,
    hand_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Home enabled actuators and rebuild command anchors from fresh feedback."""
    if int(shared.safety_state.value) == int(SafetyState.RUNNING):
        require_transition(shared, SafetyState.ARMED)

    hand_home_accepted = True
    if hand_enabled:
        hand_home_qpos_rad = np.deg2rad(
            np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)
        )
        hand_home_accepted = publish_hand_home_and_wait_applied(
            shared,
            hand_home_qpos_rad,
            command_lower_rad=np.asarray(
                runtime.hand.qpos_min_rad,
                dtype=np.float64,
            ),
            command_upper_rad=np.asarray(
                runtime.hand.qpos_max_rad,
                dtype=np.float64,
            ),
            mechanical_lower_rad=np.asarray(
                runtime.hand.mechanical_qpos_min_rad,
                dtype=np.float64,
            ),
            mechanical_upper_rad=np.asarray(
                runtime.hand.mechanical_qpos_max_rad,
                dtype=np.float64,
            ),
            hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
            timeout_s=float(runtime.hand.home_command_ack_timeout_s),
            heartbeat=False,
            abort_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
        )
        if hand_home_accepted:
            planner.set_hand_qpos(hand_home_qpos_rad)

    if hand_home_accepted:
        home_result = execute_arm_home(
            shared,
            np.asarray(runtime.arm.home_qpos, dtype=np.float64),
            planner=planner,
            config=ArmHomeConfig.from_runtime(
                runtime,
                publish_policy_heartbeat=False,
            ),
            table_z_surface_m=float(runtime.arm.table_z_surface_m),
            current_qpos=current_qpos_rad,
            estop_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
            progress=lambda message: print(f"  {message}", flush=True),
        )
    else:
        logger.warning("Return-home cancelled: hand-home command was not accepted")
        home_result = None

    if shared.estop_request.value:
        _set_fault(shared, "operator e-stop during homing")
        return None
    refreshed = _read_initial_arm(shared, runtime)
    if refreshed is None:
        _set_fault(shared, "fresh arm feedback unavailable after homing")
        return None
    if home_result is not None and not home_result.succeeded:
        logger.warning("Return-home request was not executed: %s", home_result.detail)
    return _keyboard_command_anchor(
        planner,
        np.asarray(refreshed["qpos"], dtype=np.float64),
    )


def _publish_keyboard_target(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    safety_gate: Any,
    target_pos_world_m: np.ndarray,
    target_quat_wxyz: np.ndarray,
    current_qpos_rad: np.ndarray,
    previous_command_qpos_rad: np.ndarray,
) -> _KeyboardPublishResult:
    """Solve and publish one keyboard target through the shared safety boundary."""
    ik_result = planner.solve_teleop_ik(
        Pose(p=target_pos_world_m, q=target_quat_wxyz),
        current_qpos_rad,
        previous_command_qpos_rad,
    )
    if not ik_result.success or ik_result.qpos is None:
        return _KeyboardPublishResult(
            _KeyboardPublishStatus.IK_REJECTED,
            detail=ik_result.reason or "unknown",
        )

    publish_result = publish_joint_targets(
        shared,
        ik_result.qpos,
        prepare_timeout_s=float(runtime.policy.action_prepare_timeout_s),
        safety_gate=safety_gate,
        hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
    )
    candidate = publish_result.candidate
    if not publish_result.succeeded or candidate is None or candidate.arm_qpos is None:
        return _KeyboardPublishResult(
            _KeyboardPublishStatus.SAFETY_REJECTED,
            detail=publish_result.reason,
        )
    return _KeyboardPublishResult(
        _KeyboardPublishStatus.PUBLISHED,
        arm_qpos_rad=np.asarray(candidate.arm_qpos, dtype=np.float64).copy(),
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
    current_qpos = np.asarray(state["qpos"], dtype=np.float64)
    previous_command, target_pos, target_quat = _keyboard_command_anchor(
        planner,
        current_qpos,
    )

    heartbeat_timeouts = dict(runtime.safety.heartbeat_timeouts)
    rate = RateManager(float(cfg.control_hz), label="keyboard_teleop")
    state_failures = 0
    home_key_down = False
    motion_active = False
    quit_quiesced = False
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
        if quit_requested and not quit_quiesced:
            # Establish the terminal command-silence boundary before any
            # remaining feedback/fault classification work in this iteration.
            advance_run_generation(shared)
            quit_quiesced = True
        feedback = _read_keyboard_feedback(
            shared,
            runtime,
            hand_enabled=hand_enabled,
        )
        if feedback.issue is not None:
            if not feedback.retryable:
                _set_fault(shared, feedback.issue)
                return False
            state_failures += 1
            if state_failures >= int(policy.max_consecutive_errors):
                _set_fault(shared, feedback.issue)
                return False
            continue
        state_failures = 0
        assert feedback.arm_state is not None
        assert feedback.arm_qpos_rad is not None
        current_qpos = feedback.arm_qpos_rad
        if feedback.hand_qpos_rad is not None:
            planner.set_hand_qpos(feedback.hand_qpos_rad)

        if quit_requested:
            if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                require_transition(shared, SafetyState.ARMED)
            return True

        home_pressed = keys.is_pressed("r")
        if home_pressed and not home_key_down:
            home_anchor = _run_keyboard_home(
                shared,
                runtime,
                planner,
                keys,
                current_qpos,
                hand_enabled=hand_enabled,
            )
            if home_anchor is None:
                return False
            current_qpos, target_pos, target_quat = home_anchor
            previous_command = current_qpos.copy()
            motion_active = False
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
        # Idle: stop publishing commands while Mode 6 settles its last endpoint.
        if not moving:
            if motion_active:
                advance_run_generation(shared)
                require_transition(shared, SafetyState.ARMED)
                motion_active = False
                rate.reset()
            # Rebuild baselines so new key presses start from current feedback.
            previous_command, target_pos, target_quat = _keyboard_command_anchor(
                planner,
                current_qpos,
            )
            if frame % int(cfg.idle_interval_frames) == 0:
                elapsed = time.monotonic() - started_s
                pose = planner.kin.compute_eef_pose_world(current_qpos)
                print(
                    f"idle    [{elapsed:4.0f}s f={frame:5d}] eef={_fmt_vec3(pose.p)}",
                    flush=True,
                )
            continue

        # Moving: keys are held.
        if not motion_active:
            require_transition(shared, SafetyState.RUNNING)
            motion_active = True
            rate.reset()

        measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
        target_update = _compute_keyboard_target_update(
            measured_pose,
            target_pos,
            target_quat,
            dx,
            drpy,
            workspace,
            workspace_margin_m=float(cfg.workspace_command_margin_m),
            command_lookahead_frames=float(cfg.command_lookahead_frames),
            position_step_m=float(cfg.delta_pos_m),
            rotation_step_rad=float(cfg.delta_rpy_rad),
        )
        target_pos = target_update.target_pos_world_m
        target_quat = target_update.target_quat_wxyz
        boundary_indicator = (
            f"  {target_update.boundary_text}" if target_update.boundary_text else ""
        )
        if target_update.boundary_text:
            now_s = time.monotonic()
            if now_s - last_boundary_warn_s >= _BOUNDARY_WARN_INTERVAL_S:
                logger.warning("Workspace boundary: %s", target_update.boundary_text)
                last_boundary_warn_s = now_s

        publish_result = _publish_keyboard_target(
            shared,
            runtime,
            planner,
            safety_gate,
            target_pos,
            target_quat,
            current_qpos,
            previous_command,
        )
        if publish_result.status is _KeyboardPublishStatus.IK_REJECTED:
            now_s = time.monotonic()
            if now_s - last_ik_warning_s >= _IK_WARNING_INTERVAL_S:
                logger.warning("IK rejected target: %s", publish_result.detail)
                last_ik_warning_s = now_s
            blocked_keys = active_keys
            continue
        if publish_result.status is _KeyboardPublishStatus.SAFETY_REJECTED:
            logger.warning(
                "Keyboard motion command rejected (%s) — blocked until keys change",
                publish_result.detail,
            )
            blocked_keys = active_keys
            continue
        assert publish_result.arm_qpos_rad is not None
        previous_command = publish_result.arm_qpos_rad

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
    if not shared.is_ready("arm"):
        logger.error("Arm worker did not become ready")
        return False
    if bool(runtime.policy.hand_enabled) and not no_hand and not hand_enabled:
        logger.error("XHand is required but did not become ready")
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
    # Servo the hand home position to prevent drift near mechanical limits.
    if hand_enabled:
        hand_home = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
        if publish_hand_home_and_wait_applied(
            shared,
            hand_home,
            command_lower_rad=np.asarray(runtime.hand.qpos_min_rad, dtype=np.float64),
            command_upper_rad=np.asarray(runtime.hand.qpos_max_rad, dtype=np.float64),
            mechanical_lower_rad=np.asarray(
                runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ),
            mechanical_upper_rad=np.asarray(
                runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ),
            hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
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
        # processes and shared resources are closed.
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
