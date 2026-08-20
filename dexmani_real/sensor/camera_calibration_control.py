"""Arm-motion state and safety lifecycle for interactive camera calibration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.planning import Pose, XArm7MotionPlanner
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.policy.safety import SafetyGate, publish_joint_targets
from dexmani_real.robot.homing import ArmHomeConfig, execute_arm_home
from dexmani_real.robot.safety import (
    SafetyState,
    advance_run_generation,
    require_transition,
    transition,
)
from dexmani_real.sensor.camera_calibration import CalibrationConfig, CalibrationSamples
from dexmani_real.shm.shared_storage import SharedStorage, read_arm_state_dict
from dexmani_real.teleop.keyboard import GlobalKeyState, eef_delta_from_keys
from dexmani_real.utils.hand_health import validate_arm_feedback
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

_INITIAL_STATE_POLL_S = 0.05
_IK_WARNING_INTERVAL_S = 1.0
_BOUNDARY_WARN_INTERVAL_S = 2.0


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


def _set_fault(shared: SharedStorage, reason: str, *, estop: bool = False) -> None:
    logger.error("Calibration fault: %s", reason)
    if estop:
        shared.estop_request.value = True
    shared.error_state.value = True
    transition(shared, SafetyState.FAULT)


@dataclass
class _CalibrationLoopState:
    """Mutable operator and motion state for one calibration control loop."""

    samples: CalibrationSamples
    current_qpos: np.ndarray
    previous_command: np.ndarray
    target_pos: np.ndarray
    target_quat: np.ndarray
    calibration_saved: bool = False
    home_key_down: bool = False
    motion_active: bool = False
    frame: int = 0
    last_ik_warning_s: float = 0.0
    blocked_keys: tuple[str, ...] | None = None
    last_boundary_warning_s: float = 0.0

    @classmethod
    def from_arm_state(
        cls, planner: XArm7MotionPlanner, arm_state: dict[str, Any]
    ) -> "_CalibrationLoopState":
        current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
        pose = planner.kin.compute_eef_pose_world(current_qpos)
        return cls(
            samples=CalibrationSamples(),
            current_qpos=current_qpos,
            previous_command=current_qpos.copy(),
            target_pos=pose.p.copy(),
            target_quat=pose.q.copy(),
        )


@dataclass(frozen=True)
class _CalibrationArmFeedback:
    qpos: np.ndarray | None
    issue: str = ""
    error_code: int = 0


class _HomeKeyOutcome(str, Enum):
    IDLE = "idle"
    COMPLETED = "completed"
    FAULT = "fault"


def _read_calibration_arm_feedback(
    shared: SharedStorage, runtime: ResolvedRuntimeConfig
) -> _CalibrationArmFeedback:
    arm_state = read_arm_state_dict(shared)
    if arm_state is None:
        return _CalibrationArmFeedback(None, "arm state is unavailable")
    qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
    issue = validate_arm_feedback(
        connected=arm_state["connected"],
        state_valid=arm_state["state_valid"],
        source_monotonic_ns=arm_state["source_monotonic_ns"],
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
        qpos=qpos,
        qvel=arm_state["qvel"],
    )
    return _CalibrationArmFeedback(
        None if issue is not None else qpos,
        issue or "",
        int(arm_state["error_code"]),
    )


def _publish_calibration_quit_hold(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    safety_gate: SafetyGate,
    current_qpos: np.ndarray,
    *,
    calibration_saved: bool,
) -> int:
    """Invalidate queued motion, publish measured hold, and return exit status."""
    advance_run_generation(shared)
    published = publish_joint_targets(
        shared,
        current_qpos,
        is_hold=True,
        prepare_timeout_s=float(runtime.policy.action_prepare_timeout_s),
        safety_gate=safety_gate,
        wait_applied=True,
        apply_timeout_s=float(runtime.policy.action_apply_timeout_s),
        hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
    )
    if not published.succeeded:
        _set_fault(
            shared,
            f"measured quit hold was not applied: {published.reason}",
        )
        return 1
    if int(shared.safety_state.value) == int(SafetyState.RUNNING):
        require_transition(shared, SafetyState.ARMED)
    return 0 if calibration_saved else 2


def _handle_calibration_home_key(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    keys: GlobalKeyState,
    rate: RateManager,
    state: _CalibrationLoopState,
) -> _HomeKeyOutcome:
    """Handle one return-home key edge and re-anchor the motion state."""
    home_pressed = keys.is_pressed("r")
    if not home_pressed:
        state.home_key_down = False
        return _HomeKeyOutcome.IDLE
    if state.home_key_down:
        return _HomeKeyOutcome.IDLE
    state.home_key_down = True

    if int(shared.safety_state.value) == int(SafetyState.RUNNING):
        require_transition(shared, SafetyState.ARMED)
    home_result = execute_arm_home(
        shared,
        np.asarray(runtime.arm.home_qpos, dtype=np.float64),
        planner=planner,
        config=ArmHomeConfig.from_runtime(
            runtime,
            publish_policy_heartbeat=False,
        ),
        table_z_surface_m=float(runtime.arm.table_z_surface_m),
        current_qpos=state.current_qpos,
        estop_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    if shared.estop_request.value:
        _set_fault(shared, "operator e-stop during homing")
        return _HomeKeyOutcome.FAULT
    refreshed = _read_initial_arm(shared, runtime)
    if refreshed is None:
        _set_fault(shared, "fresh arm feedback unavailable after homing")
        return _HomeKeyOutcome.FAULT

    state.current_qpos = np.asarray(refreshed["qpos"], dtype=np.float64)
    state.previous_command = state.current_qpos.copy()
    fresh_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
    state.target_pos = fresh_pose.p.copy()
    state.target_quat = fresh_pose.q.copy()
    if not home_result.succeeded:
        print("  WARNING: return-home request was not executed")
    state.motion_active = False
    rate.reset()
    return _HomeKeyOutcome.COMPLETED


def _log_workspace_clipping(
    desired_pos: np.ndarray,
    clipped_pos: np.ndarray,
    last_warning_s: float,
) -> float:
    clipped = np.abs(desired_pos - clipped_pos) > 1e-9
    if not np.any(clipped):
        return last_warning_s
    parts: list[str] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        if clipped[axis_index]:
            side = "⁺" if desired_pos[axis_index] > clipped_pos[axis_index] else "⁻"
            parts.append(f"{axis_name}{side}{clipped_pos[axis_index]:.3f}")
    now_s = time.monotonic()
    if now_s - last_warning_s >= _BOUNDARY_WARN_INTERVAL_S:
        logger.warning("Workspace boundary: %s", " ".join(parts))
        return now_s
    return last_warning_s


def _run_calibration_motion_tick(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    safety_gate: SafetyGate,
    workspace: np.ndarray,
    keys: GlobalKeyState,
    state: _CalibrationLoopState,
    calib_cfg: CalibrationConfig,
) -> None:
    """Translate held motion keys into one gated arm command or an idle hold."""
    dx, drpy = eef_delta_from_keys(keys, calib_cfg.delta_pos_m, calib_cfg.delta_rpy_rad)
    moving = bool(np.any(dx != 0.0) or np.any(drpy != 0.0))
    active_keys = keys.pressed_keys()
    if state.blocked_keys is not None and active_keys == state.blocked_keys:
        return
    state.blocked_keys = None

    if moving and not state.motion_active:
        require_transition(shared, SafetyState.RUNNING)
    elif not moving and state.motion_active:
        require_transition(shared, SafetyState.ARMED)
        held_pose = planner.kin.compute_eef_pose_world(state.previous_command)
        state.target_pos = held_pose.p.copy()
        state.target_quat = held_pose.q.copy()
    state.motion_active = moving

    if not moving:
        state.previous_command = state.current_qpos.copy()
        idle_interval = int(runtime.keyboard_teleop.idle_interval_frames)
        if state.frame % idle_interval == 0:
            measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
            print(
                f"[f={state.frame}] samples={len(state.samples)} "
                f"eef={np.round(measured_pose.p, 3)}m",
                flush=True,
            )
        return

    workspace_margin_m = float(runtime.keyboard_teleop.workspace_command_margin_m)
    command_low = workspace[:, 0] + workspace_margin_m
    command_high = workspace[:, 1] - workspace_margin_m
    desired_pos = state.target_pos + dx
    state.target_pos = np.clip(desired_pos, command_low, command_high)
    state.last_boundary_warning_s = _log_workspace_clipping(
        desired_pos,
        state.target_pos,
        state.last_boundary_warning_s,
    )
    if np.any(drpy != 0.0):
        delta_quat = Rotation.from_euler("xyz", drpy).as_quat(scalar_first=True)
        state.target_quat = quat_multiply(delta_quat, state.target_quat)

    ik_result = planner.solve_teleop_ik(
        Pose(p=state.target_pos, q=state.target_quat),
        state.current_qpos,
        state.previous_command,
    )
    if not ik_result.success or ik_result.qpos is None:
        now_s = time.monotonic()
        if now_s - state.last_ik_warning_s >= _IK_WARNING_INTERVAL_S:
            logger.warning("IK rejected target: %s", ik_result.reason or "unknown")
            state.last_ik_warning_s = now_s
        measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
        state.target_pos = measured_pose.p.copy()
        state.target_quat = measured_pose.q.copy()
        return

    published = publish_joint_targets(
        shared,
        ik_result.qpos,
        prepare_timeout_s=float(runtime.policy.action_prepare_timeout_s),
        safety_gate=safety_gate,
        wait_applied=True,
        apply_timeout_s=float(runtime.policy.action_apply_timeout_s),
        hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
    )
    candidate = published.candidate
    if not published.succeeded or candidate is None or candidate.arm_qpos is None:
        logger.warning(
            "arm motion command rejected (%s) — blocked until keys change",
            published.reason,
        )
        state.blocked_keys = active_keys
        return
    state.previous_command = np.asarray(candidate.arm_qpos, dtype=np.float64).copy()

    if state.frame % calib_cfg.status_interval_frames == 0:
        measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
        print(
            f"[f={state.frame}] samples={len(state.samples)} "
            f"eef={np.round(measured_pose.p, 3)}m "
            f"target={np.round(state.target_pos, 3)}m",
            flush=True,
        )
