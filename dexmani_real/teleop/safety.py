"""Safety holds, re-anchoring, contact guards, and homing for teleoperation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, hand, safety
from dexmani_real.utils.schema import ARM_JOINT_SHAPE
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.planning.pose_utils import rot6d_to_quat_wxyz
from dexmani_real.policy.action_protocol import AckStatus, ActionSafetyGate, SafeCommandPublisher, hand_home_converge
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.robot.homing import send_arm_home
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.hand_control import _reset_hand_retargeter
from dexmani_real.teleop.snapshot import _read_arm_state
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def _candidate_application_state(shared: Any, candidate: ActionCandidate) -> str:
    """Return pending/applied/failed from exact identity-correlated worker ACKs."""
    arm_status = (
        AckStatus.APPLIED
        if candidate.arm_qpos is None
        else SafeCommandPublisher._ack_status(shared.arm_ack_ring, candidate)
    )
    hand_status = (
        AckStatus.APPLIED
        if candidate.hand_qpos is None
        else SafeCommandPublisher._ack_status(shared.hand_ack_ring, candidate)
    )
    failed = {AckStatus.REJECTED, AckStatus.SDK_FAILED}
    if arm_status in failed or hand_status in failed:
        return "failed"
    if arm_status is AckStatus.APPLIED and hand_status is AckStatus.APPLIED:
        return "applied" if _candidate_applied_monotonic_ns(shared, candidate) is not None else "pending"
    return "pending"


def _candidate_applied_monotonic_ns(shared: Any, candidate: ActionCandidate) -> int | None:
    """Return the latest enabled-worker SDK apply time for one exact candidate."""
    applied_times: list[int] = []
    for qpos, ack_ring in (
        (candidate.arm_qpos, shared.arm_ack_ring),
        (candidate.hand_qpos, shared.hand_ack_ring),
    ):
        if qpos is None:
            continue
        result = ack_ring.read_latest()
        if result is None or not SafeCommandPublisher._ack_matches(result[0], candidate):
            return None
        ack = result[0]
        if int(ack["status"][0]) != int(AckStatus.APPLIED):
            return None
        applied_ns = int(ack["applied_monotonic_ns"][0])
        if applied_ns <= 0:
            return None
        applied_times.append(applied_ns)
    return max(applied_times) if applied_times else None


def _feedback_after_hold(
    arm_state: np.ndarray,
    hand_state: np.ndarray | None,
    candidate: ActionCandidate,
    applied_monotonic_ns: int,
) -> bool:
    """Require measured feedback produced after every coordinated hold apply."""
    arm_names = arm_state.dtype.names or ()
    if "source_monotonic_ns" not in arm_names:
        return False
    if int(arm_state["source_monotonic_ns"][0]) < applied_monotonic_ns:
        return False
    if candidate.hand_qpos is None:
        return True
    if hand_state is None:
        return False
    hand_names = hand_state.dtype.names or ()
    return "source_monotonic_ns" in hand_names and int(hand_state["source_monotonic_ns"][0]) >= applied_monotonic_ns


def _vr_after_hold(vr_frame: dict[str, Any] | None, applied_monotonic_ns: int) -> bool:
    """Return whether the VR sample was received after coordinated hold apply."""
    if vr_frame is None:
        return False
    try:
        return int(vr_frame.get("local_recv_ns", 0)) >= applied_monotonic_ns
    except (TypeError, ValueError):
        return False


def _reset_mapper_from_frames(
    mapper: ArmWristMapper,
    arm_state: np.ndarray | None,
    vr_frame: dict[str, Any] | None,
) -> bool:
    """Atomically re-anchor wrist mapping from fresh measured arm/VR poses."""
    if arm_state is None or vr_frame is None:
        mapper.clear()
        return False
    names = arm_state.dtype.names or ()
    if "state_valid" in names and not bool(arm_state["state_valid"][0]):
        mapper.clear()
        return False
    if "connected" in names and not bool(arm_state["connected"][0]):
        mapper.clear()
        return False
    eef_pos = np.asarray(arm_state["eef_pos"][0], dtype=np.float64)
    eef_rot6d = np.asarray(arm_state["eef_rot6d"][0], dtype=np.float64)
    if eef_pos.shape != (3,) or eef_rot6d.shape != (6,):
        mapper.clear()
        return False
    if not np.all(np.isfinite(eef_pos)) or not np.all(np.isfinite(eef_rot6d)):
        mapper.clear()
        return False
    c1 = eef_rot6d[:3]
    c2 = eef_rot6d[3:]
    if np.linalg.norm(c1) < 1e-12 or np.linalg.norm(np.cross(c1, c2)) < 1e-12:
        mapper.clear()
        return False

    try:
        head_quat = vr_frame.get("head_quat_wxyz")
        if head_quat is not None:
            mapper.set_heading(np.asarray(head_quat, dtype=np.float64))
        wrist_pos = vr_frame.get("wrist_pos")
        wrist_quat_wxyz = vr_frame.get("wrist_quat_wxyz")
        if wrist_pos is None or wrist_quat_wxyz is None:
            mapper.clear()
            return False
        wrist_pos_array = np.asarray(wrist_pos, dtype=np.float64)
        wrist_quat_array = np.asarray(wrist_quat_wxyz, dtype=np.float64)
    except (TypeError, ValueError):
        mapper.clear()
        return False
    mapper.reset(
        wrist_pos=wrist_pos_array,
        wrist_quat_wxyz=wrist_quat_array,
        eef_pos=eef_pos,
        eef_quat_wxyz=rot6d_to_quat_wxyz(eef_rot6d),
    )
    return mapper.is_ready()


def _transition_collision_free(
    planner: XArm7MotionPlanner,
    arm_start: np.ndarray,
    arm_end: np.ndarray,
    hand_start: np.ndarray,
    hand_end: np.ndarray,
) -> bool:
    """Fail closed when the conservative arm-hand transition check cannot complete."""
    try:
        return planner.collision_model.check_transition_collision_free(arm_start, arm_end, hand_start, hand_end)
    except (ValueError, RuntimeError):
        logger.warning("teleop: arm-hand collision check failed", exc_info=True)
        return False


def _contact_stall_detected(
    arm_qpos: np.ndarray,
    arm_qvel: np.ndarray,
    previous_arm_cmd: np.ndarray,
    eef_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    table_z_surface_m: float,
    table_context_height_m: float,
    min_downward_target_m: float,
    tracking_error_rad: float,
    max_closing_speed_rad_s: float,
) -> bool:
    """Detect a blocked downward command without treating the table as forbidden."""
    qpos = np.asarray(arm_qpos, dtype=np.float64)
    qvel = np.asarray(arm_qvel, dtype=np.float64)
    command = np.asarray(previous_arm_cmd, dtype=np.float64)
    eef = np.asarray(eef_pos, dtype=np.float64)
    target = np.asarray(target_pos, dtype=np.float64)
    if qpos.shape != ARM_JOINT_SHAPE or qvel.shape != ARM_JOINT_SHAPE or command.shape != ARM_JOINT_SHAPE:
        return False
    if eef.shape != (3,) or target.shape != (3,):
        return False
    if not all(np.all(np.isfinite(values)) for values in (qpos, qvel, command, eef, target)):
        return False

    near_table = eef[2] <= table_z_surface_m + table_context_height_m
    downward_intent = target[2] <= eef[2] - min_downward_target_m
    command_error = command - qpos
    if not near_table or not downward_intent or np.max(np.abs(command_error)) < tracking_error_rad:
        return False

    error_norm = float(np.linalg.norm(command_error))
    if error_norm <= 1e-12:
        return False
    closing_speed = float(np.dot(qvel, command_error) / error_norm)
    return closing_speed <= max_closing_speed_rad_s


def _do_teleop_home(
    shared: SharedStorage,
    *,
    hand_available: bool,
    fixed_hand_home_acknowledged: bool = False,
    prev_hand_qpos: np.ndarray,
    planner,
    audio,
    hand_home_qpos: np.ndarray,
    table_z_surface_m: float,
    hand_home_timeout_s: float = hand.home_settle_timeout_s,
    hand_home_tolerance_rad: float = hand.home_settle_tol_rad,
    arm_home_convergence_timeout_s: float = arm.homing.convergence_timeout_s,
    arm_home_request_queue_timeout_s: float = arm.homing.request_queue_timeout_s,
    arm_home_state_max_age_s: float = arm.homing.state_max_age_s,
    arm_home_max_speed_rad_s: float = np.deg2rad(arm.homing.max_speed_deg_s),
    arm_home_target_timeout_s: float = arm.homing.target_timeout_s,
    arm_home_velocity_convergence_rad_s: float = arm.homing.velocity_convergence_rad_s,
    arm_home_result_tolerance_rad: float = arm.homing.convergence_rad,
    arm_heartbeat_timeout_s: float = safety.heartbeat_timeouts["arm"],
    estop_requested: Callable[[], bool] | None = None,
    arm_mapper=None,
    hand_retargeter=None,
    heartbeat: bool = True,
    action_safety_gate: ActionSafetyGate | None = None,
    arm_home_qpos: np.ndarray | None = None,
) -> np.ndarray:
    """Home hand first, then arm. Returns updated *prev_hand_qpos*.

    If *arm_mapper* and *hand_retargeter* are both provided, clears EMA
    state and re-seeds retargeter before homing (active-teleop H path).
    Post-teleop callers pass ``None`` for both — the state is already cleared.
    """
    if arm_mapper is not None:
        arm_mapper.clear()
    if hand_retargeter is not None:
        _reset_hand_retargeter(hand_retargeter)

    # Step 1: hand home first (prevents arm sweeping while hand is in grasp).
    if hand_available and not shared.error_state.value:
        hand_reached, final_qpos = hand_home_converge(
            shared,
            hand_home_qpos,
            timeout_s=hand_home_timeout_s,
            tol_rad=hand_home_tolerance_rad,
            heartbeat=heartbeat,
            check_is_running=True,
            verbose=True,
            safety_gate=action_safety_gate,
            abort_requested=estop_requested,
        )
        if final_qpos is not None:
            prev_hand_qpos = final_qpos
            planner.set_hand_qpos(prev_hand_qpos)
        if not hand_reached:
            logger.warning("arm home cancelled: hand did not reach a fresh, healthy home state")
            return prev_hand_qpos
    elif fixed_hand_home_acknowledged:
        prev_hand_qpos = np.asarray(hand_home_qpos, dtype=np.float64).copy()
        planner.set_hand_qpos(prev_hand_qpos)
        print("  hand: using explicitly acknowledged fixed-home geometry", flush=True)
    else:
        print("  hand: not connected — arm home cancelled (hand pose unknown)", flush=True)
        return prev_hand_qpos

    # Step 2: arm home (collision-checked path via HOME_SENTINEL).
    # Re-read after hand homing: the old loop-local qpos may be several seconds
    # stale, and collision/path execution must start from current encoder state.
    _arm_state = _read_arm_state(shared)
    if _arm_state is None:
        logger.warning("arm home cancelled: no current arm state")
        return prev_hand_qpos
    _state_age_s = time.monotonic() - float(_arm_state["timestamp"][0])
    if (
        _state_age_s > arm_home_state_max_age_s
        or not bool(_arm_state["connected"][0])
        or int(_arm_state["error_code"][0]) != 0
        or not np.all(np.isfinite(_arm_state["qpos"][0]))
    ):
        logger.warning("arm home cancelled: arm state is stale or unhealthy (age=%.3fs)", _state_age_s)
        return prev_hand_qpos
    arm_qpos = np.asarray(_arm_state["qpos"][0], dtype=np.float64).copy()
    _home_qpos = np.array(arm.home_qpos if arm_home_qpos is None else arm_home_qpos, dtype=np.float64)
    _ok = send_arm_home(
        shared,
        _home_qpos,
        planner=planner,
        table_z_surface_m=table_z_surface_m,
        current_qpos=arm_qpos,
        heartbeat=heartbeat,
        converge_timeout_s=arm_home_convergence_timeout_s,
        queue_timeout=arm_home_request_queue_timeout_s,
        state_max_age_s=arm_home_state_max_age_s,
        homing_max_speed_rad_s=arm_home_max_speed_rad_s,
        homing_target_timeout_s=arm_home_target_timeout_s,
        preplan_velocity_rad_s=arm_home_velocity_convergence_rad_s,
        result_tolerance_rad=arm_home_result_tolerance_rad,
        arm_heartbeat_max_age_s=arm_heartbeat_timeout_s,
        estop_requested=estop_requested,
        verbose=True,
    )
    if _ok:
        audio.play("home_done")
        print("  arm: home reached", flush=True)
    else:
        logger.warning("arm home failed or was cancelled; see correlated HOME result above")

    return prev_hand_qpos


def _do_configured_teleop_home(
    shared: SharedStorage,
    config: TeleopConfig,
    *,
    hand_available: bool,
    prev_hand_qpos: np.ndarray,
    planner: XArm7MotionPlanner,
    audio: Any,
    estop_requested: Callable[[], bool],
    action_safety_gate: ActionSafetyGate,
    arm_mapper: ArmWristMapper | None = None,
    hand_retargeter: Any = None,
) -> np.ndarray:
    """Apply the validated experiment config to the hand-first homing protocol."""
    return _do_teleop_home(
        shared,
        hand_available=hand_available,
        fixed_hand_home_acknowledged=not config.hand_enabled,
        prev_hand_qpos=prev_hand_qpos,
        planner=planner,
        audio=audio,
        hand_home_qpos=np.deg2rad(np.asarray(config.hand_home_qpos_deg, dtype=np.float64)),
        hand_home_timeout_s=config.hand_home_settle_timeout_s,
        hand_home_tolerance_rad=config.hand_home_settle_tolerance_rad,
        arm_home_convergence_timeout_s=config.arm_home_convergence_timeout_s,
        arm_home_request_queue_timeout_s=config.arm_home_request_queue_timeout_s,
        arm_home_state_max_age_s=config.arm_home_state_max_age_s,
        arm_home_max_speed_rad_s=config.arm_home_max_speed_rad_s,
        arm_home_target_timeout_s=config.arm_home_target_timeout_s,
        arm_home_velocity_convergence_rad_s=config.arm_home_velocity_convergence_rad_s,
        arm_home_result_tolerance_rad=config.arm_home_result_tolerance_rad,
        arm_heartbeat_timeout_s=config.arm_heartbeat_timeout_s,
        estop_requested=estop_requested,
        table_z_surface_m=config.contact_stall_table_z_surface_m,
        arm_mapper=arm_mapper,
        hand_retargeter=hand_retargeter,
        action_safety_gate=action_safety_gate,
        arm_home_qpos=np.asarray(config.arm_home_qpos, dtype=np.float64),
    )
