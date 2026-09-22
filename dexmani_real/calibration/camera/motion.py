"""Arm-motion state and safety lifecycle for interactive camera calibration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.calibration.camera.solver import CalibrationConfig, CalibrationSamples
from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.ipc.channels import RuntimeChannels, read_arm_state_dict
from dexmani_real.planning import Pose, XArm7MotionPlanner
from dexmani_real.planning.kinematics.pose import quat_multiply
from dexmani_real.robot.arm_homing import ArmHomeConfig, execute_arm_home
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.robot.projection import project_arm_command
from dexmani_real.runtime.observation import sample_is_fresh
from dexmani_real.runtime.operator_input import KeyboardInput
from dexmani_real.runtime.safety import SafetyState, begin_motion, revoke_motion
from dexmani_real.teleop.jog import (
    any_jog_key_held,
    compute_cartesian_jog_delta,
    limit_cartesian_pose_lead,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)

_INITIAL_STATE_POLL_S = 0.05
_IK_WARNING_INTERVAL_S = 1.0
_BOUNDARY_WARN_INTERVAL_S = 2.0


def read_initial_arm(shared: RuntimeChannels, runtime: ExperimentConfig) -> dict[str, Any] | None:
    deadline_s = time.monotonic() + float(runtime.safety.readiness_timeouts_s["arm"])
    while time.monotonic() < deadline_s:
        state = read_arm_state_dict(shared)
        if state is not None and sample_is_fresh(
            state["timestamp_ns"], runtime.arm.feedback_max_age_s
        ):
            return state
        time.sleep(_INITIAL_STATE_POLL_S)
    return None


def set_calibration_fault(shared: RuntimeChannels, reason: str, *, estop: bool = False) -> None:
    logger.error("Calibration fault: %s", reason)
    if estop:
        shared.estop_request.value = True
    shared.error_state.value = True
    revoke_motion(shared, SafetyState.FAULT)


@dataclass
class CalibrationLoopState:
    """Mutable operator and motion state for one calibration control loop."""

    samples: CalibrationSamples
    current_qpos: np.ndarray
    previous_command: np.ndarray
    calibration_saved: bool = False
    home_key_down: bool = False
    frame: int = 0
    last_ik_warning_s: float = 0.0
    blocked_until_release: bool = False
    last_boundary_warning_s: float = 0.0

    @classmethod
    def from_arm_state(cls, arm_state: dict[str, Any]) -> "CalibrationLoopState":
        current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
        return cls(
            samples=CalibrationSamples(),
            current_qpos=current_qpos,
            previous_command=current_qpos.copy(),
        )


class HomeKeyOutcome(str, Enum):
    IDLE = "idle"
    COMPLETED = "completed"
    FAULT = "fault"


def finish_calibration_motion(shared, *, calibration_saved):
    revoke_motion(shared)
    return 0 if calibration_saved else 2


def handle_calibration_home_key(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    planner: XArm7MotionPlanner,
    keys: KeyboardInput,
    rate: LoopRate,
    state: CalibrationLoopState,
) -> HomeKeyOutcome:
    """Handle one return-home key edge and re-anchor the motion state."""
    home_pressed = keys.is_pressed("r")
    if not home_pressed:
        state.home_key_down = False
        return HomeKeyOutcome.IDLE
    if state.home_key_down:
        return HomeKeyOutcome.IDLE
    state.home_key_down = True

    if int(shared.safety_state.value) == int(SafetyState.RUNNING):
        if not revoke_motion(shared, SafetyState.ARMED):
            set_calibration_fault(shared, "failed to stop calibration motion before home")
            return HomeKeyOutcome.FAULT
    home_result = execute_arm_home(
        shared,
        np.asarray(runtime.arm.home_qpos, dtype=np.float64),
        planner=planner,
        config=ArmHomeConfig.from_runtime(runtime),
        estop_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    if shared.estop_request.value:
        set_calibration_fault(shared, "operator e-stop during homing")
        return HomeKeyOutcome.FAULT
    refreshed = read_initial_arm(shared, runtime)
    if refreshed is None:
        set_calibration_fault(shared, "fresh arm feedback unavailable after homing")
        return HomeKeyOutcome.FAULT

    state.current_qpos = np.asarray(refreshed["qpos"], dtype=np.float64)
    state.previous_command = state.current_qpos.copy()
    if not home_result.ok:
        print("  WARNING: return-home request was not executed")
    state.blocked_until_release = False
    rate.reset()
    return HomeKeyOutcome.COMPLETED


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


def _reject_calibration_motion(
    shared: RuntimeChannels, state: CalibrationLoopState, reason: str
) -> None:
    """Close the rejected jog epoch and require a fresh physical key press."""
    if (
        shared.error_state.value
        or shared.estop_request.value
        or not shared.is_running.value
        or int(shared.safety_state.value) not in (int(SafetyState.ARMED), int(SafetyState.RUNNING))
    ):
        set_calibration_fault(shared, reason)
        return
    if int(shared.safety_state.value) == int(SafetyState.RUNNING):
        if not revoke_motion(shared, SafetyState.ARMED):
            set_calibration_fault(shared, "failed to stop rejected calibration motion")
            return
    state.previous_command = state.current_qpos.copy()
    state.blocked_until_release = True


def run_calibration_motion_tick(
    shared: RuntimeChannels,
    runtime: ExperimentConfig,
    planner: XArm7MotionPlanner,
    workspace: np.ndarray,
    keys: KeyboardInput,
    state: CalibrationLoopState,
    calib_cfg: CalibrationConfig,
) -> None:
    """Advance from the previous high-level target with measured-pose bounded lookahead."""
    if shared.stop_request.value:
        _reject_calibration_motion(shared, state, "command admission revoked")
        with shared.motion_lock:
            shared.stop_request.value = 0
    active_keys = keys.pressed_keys()
    if state.blocked_until_release:
        if not any_jog_key_held(active_keys):
            state.blocked_until_release = False
            state.previous_command = state.current_qpos.copy()
        return
    safety_state = int(shared.safety_state.value)
    if safety_state not in (int(SafetyState.ARMED), int(SafetyState.RUNNING)):
        set_calibration_fault(shared, "unexpected calibration motion state")
        return
    dx, drpy = compute_cartesian_jog_delta(keys, calib_cfg.delta_pos_m, calib_cfg.delta_rpy_rad)
    moving = bool(np.any(dx != 0.0) or np.any(drpy != 0.0))
    if not moving:
        if safety_state == int(SafetyState.RUNNING):
            if not revoke_motion(shared, SafetyState.ARMED):
                set_calibration_fault(shared, "failed to stop calibration motion")
                return
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

    if safety_state == int(SafetyState.ARMED):
        state.previous_command = state.current_qpos.copy()
        if not begin_motion(shared):
            set_calibration_fault(shared, "failed to enter calibration motion")
            return

    epoch = int(shared.run_id.value)
    measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
    anchor_pose = planner.kin.compute_eef_pose_world(state.previous_command)
    workspace_margin_m = float(runtime.keyboard_teleop.workspace_command_margin_m)
    command_low = workspace[:, 0] + workspace_margin_m
    command_high = workspace[:, 1] - workspace_margin_m
    desired_pos = anchor_pose.p + dx
    proposed_pos = np.clip(desired_pos, command_low, command_high)
    state.last_boundary_warning_s = _log_workspace_clipping(
        desired_pos,
        proposed_pos,
        state.last_boundary_warning_s,
    )
    proposed_quat = anchor_pose.q.copy()
    if np.any(drpy != 0.0):
        delta_quat = Rotation.from_euler("xyz", drpy).as_quat(scalar_first=True)
        proposed_quat = quat_multiply(delta_quat, proposed_quat)

    proposed_pos, proposed_quat = limit_cartesian_pose_lead(
        measured_pose.p,
        measured_pose.q,
        proposed_pos,
        proposed_quat,
        max_position_lead_m=calib_cfg.command_lookahead_frames * calib_cfg.delta_pos_m,
        max_rotation_lead_rad=calib_cfg.command_lookahead_frames * calib_cfg.delta_rpy_rad,
    )

    ik_result = planner.solve_teleop_ik(
        Pose(p=proposed_pos, q=proposed_quat),
        state.current_qpos,
        state.previous_command,
    )
    if not ik_result.success or ik_result.qpos is None:
        now_s = time.monotonic()
        if now_s - state.last_ik_warning_s >= _IK_WARNING_INTERVAL_S:
            logger.warning("IK rejected target: %s", ik_result.reason or "unknown")
            state.last_ik_warning_s = now_s
        _reject_calibration_motion(shared, state, ik_result.reason or "IK rejected")
        return

    q_cmd = project_arm_command(
        ik_result.qpos,
        state.current_qpos,
        joint_lower_rad=runtime.arm.joint_limit_lower,
        joint_upper_rad=runtime.arm.joint_limit_upper,
    )
    if publish_command(shared, RobotCommand(epoch, q_cmd)):
        state.previous_command = q_cmd

    if state.frame % calib_cfg.status_interval_frames == 0:
        measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
        print(
            f"[f={state.frame}] samples={len(state.samples)} "
            f"eef={np.round(measured_pose.p, 3)}m "
            f"target={np.round(proposed_pos, 3)}m",
            flush=True,
        )
