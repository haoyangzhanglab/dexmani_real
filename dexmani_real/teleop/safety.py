"""Safety holds, re-anchoring, contact guards, and homing for teleoperation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, hand, safety
from dexmani_real.control.arm_home import send_arm_home
from dexmani_real.control.hand_home import publish_hand_home_and_wait_applied
from dexmani_real.ipc.causal import read_arm_state_causal
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning import XArm7MotionPlanner
from dexmani_real.planning.arm_fk import make_arm_fk
from dexmani_real.planning.poses import rot6d_to_quat_wxyz
from dexmani_real.runtime.safety import invalidate_coupled_commands
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_state import CommandQuiescence, TeleopLoopState
from dexmani_real.teleop.hand_control import reset_hand_retargeter
from dexmani_real.utils.feedback import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import ThrottledWarner, get_logger

logger = get_logger(__name__)


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
    try:
        eef_pos, eef_rot6d = make_arm_fk().compute(
            np.asarray(arm_state["qpos"][0], dtype=np.float64)
        )
    except Exception:
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


def _do_teleop_home(
    shared: RuntimeChannels,
    *,
    hand_available: bool,
    fixed_hand_home_acknowledged: bool = False,
    prev_hand_qpos: np.ndarray,
    planner,
    audio,
    hand_home_qpos: np.ndarray,
    table_z_surface_m: float,
    hand_command_lower_rad: tuple[float, ...] | np.ndarray = hand.qpos_min_rad,
    hand_command_upper_rad: tuple[float, ...] | np.ndarray = hand.qpos_max_rad,
    hand_mechanical_lower_rad: (
        tuple[float, ...] | np.ndarray
    ) = hand.mechanical_qpos_min_rad,
    hand_mechanical_upper_rad: (
        tuple[float, ...] | np.ndarray
    ) = hand.mechanical_qpos_max_rad,
    hand_home_ack_timeout_s: float = hand.home_command_ack_timeout_s,
    arm_home_convergence_timeout_s: float = arm.homing.convergence_timeout_s,
    arm_home_request_queue_timeout_s: float = arm.homing.request_queue_timeout_s,
    arm_home_state_max_age_s: float = arm.homing.state_max_age_s,
    arm_home_max_speed_rad_s: float = np.deg2rad(arm.homing.max_speed_deg_s),
    arm_home_target_timeout_s: float = arm.homing.target_timeout_s,
    arm_home_velocity_convergence_rad_s: float = arm.homing.velocity_convergence_rad_s,
    arm_home_result_tolerance_rad: float = arm.homing.convergence_rad,
    arm_heartbeat_timeout_s: float = safety.heartbeat_timeouts["arm"],
    hand_feedback_max_age_s: float = safety.heartbeat_timeouts["hand"],
    estop_requested: Callable[[], bool] | None = None,
    arm_mapper=None,
    hand_retargeter=None,
    heartbeat: bool = True,
    arm_home_qpos: np.ndarray | None = None,
) -> np.ndarray:
    """Apply hand-home, acknowledge its SDK send, then home the arm.

    If *arm_mapper* and *hand_retargeter* are both provided, clears EMA
    state and re-seeds retargeter before homing (active-teleop H path).
    Post-teleop callers pass ``None`` for both — the state is already cleared.
    Hand execution convergence is intentionally not inspected.
    """
    if arm_mapper is not None:
        arm_mapper.clear()
    if hand_retargeter is not None:
        reset_hand_retargeter(hand_retargeter)

    if hand_available and not shared.error_state.value:
        hand_accepted = publish_hand_home_and_wait_applied(
            shared,
            np.asarray(hand_home_qpos, dtype=np.float64),
            command_lower_rad=np.asarray(hand_command_lower_rad, dtype=np.float64),
            command_upper_rad=np.asarray(hand_command_upper_rad, dtype=np.float64),
            mechanical_lower_rad=np.asarray(
                hand_mechanical_lower_rad, dtype=np.float64
            ),
            mechanical_upper_rad=np.asarray(
                hand_mechanical_upper_rad, dtype=np.float64
            ),
            hand_feedback_max_age_s=hand_feedback_max_age_s,
            timeout_s=hand_home_ack_timeout_s,
            heartbeat=heartbeat,
            abort_requested=estop_requested,
        )
        if not hand_accepted:
            logger.warning(
                "arm home cancelled: hand-home command was not accepted by the worker/SDK"
            )
            return prev_hand_qpos
        prev_hand_qpos = np.asarray(hand_home_qpos, dtype=np.float64).copy()
        planner.set_hand_qpos(prev_hand_qpos)
    elif fixed_hand_home_acknowledged:
        prev_hand_qpos = np.asarray(hand_home_qpos, dtype=np.float64).copy()
        planner.set_hand_qpos(prev_hand_qpos)
        print("  hand: using explicitly acknowledged fixed-home geometry", flush=True)
    else:
        print(
            "  hand: not connected — arm home cancelled (hand pose unknown)", flush=True
        )
        return prev_hand_qpos

    _arm_state = read_arm_state_causal(shared)
    if _arm_state is None:
        logger.warning("arm home cancelled: no current arm state")
        return prev_hand_qpos
    _state_age_s = (
        time.monotonic_ns() - int(_arm_state["source_monotonic_ns"][0])
    ) * 1e-9
    if (
        _state_age_s > arm_home_state_max_age_s
        or not bool(_arm_state["connected"][0])
        or int(_arm_state["error_code"][0]) != 0
        or not np.all(np.isfinite(_arm_state["qpos"][0]))
    ):
        logger.warning(
            "arm home cancelled: arm state is stale or unhealthy (age=%.3fs)",
            _state_age_s,
        )
        return prev_hand_qpos
    arm_qpos = np.asarray(_arm_state["qpos"][0], dtype=np.float64).copy()
    _home_qpos = np.array(
        arm.home_qpos if arm_home_qpos is None else arm_home_qpos, dtype=np.float64
    )
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
        # Keep the departure cue intact; AudioFeedback.queue() serializes this
        # completion cue after it instead of cancelling it mid-sentence.
        audio.queue("home_done")
        print("  arm: home reached", flush=True)
    else:
        logger.warning("arm home failed or was cancelled")

    return prev_hand_qpos


def do_configured_teleop_home(
    shared: RuntimeChannels,
    config: TeleopConfig,
    *,
    hand_available: bool,
    prev_hand_qpos: np.ndarray,
    planner: XArm7MotionPlanner,
    audio: Any,
    estop_requested: Callable[[], bool],
    arm_mapper: ArmWristMapper | None = None,
    hand_retargeter: Any = None,
) -> np.ndarray:
    """Apply the validated experiment config to the hand-first homing protocol."""
    return _do_teleop_home(
        shared,
        hand_available=hand_available,
        fixed_hand_home_acknowledged=not config.runtime.policy.hand_enabled,
        prev_hand_qpos=prev_hand_qpos,
        planner=planner,
        audio=audio,
        hand_home_qpos=np.deg2rad(
            np.asarray(config.runtime.hand.home_qpos_deg, dtype=np.float64)
        ),
        hand_command_lower_rad=np.asarray(
            config.runtime.hand.qpos_min_rad, dtype=np.float64
        ),
        hand_command_upper_rad=np.asarray(
            config.runtime.hand.qpos_max_rad, dtype=np.float64
        ),
        hand_mechanical_lower_rad=np.asarray(
            config.runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
        ),
        hand_mechanical_upper_rad=np.asarray(
            config.runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
        ),
        hand_home_ack_timeout_s=config.runtime.hand.home_command_ack_timeout_s,
        arm_home_convergence_timeout_s=config.runtime.arm.homing.convergence_timeout_s,
        arm_home_request_queue_timeout_s=config.runtime.arm.homing.request_queue_timeout_s,
        arm_home_state_max_age_s=config.runtime.arm.homing.state_max_age_s,
        arm_home_max_speed_rad_s=float(
            np.deg2rad(config.runtime.arm.homing.max_speed_deg_s)
        ),
        arm_home_target_timeout_s=config.runtime.arm.homing.target_timeout_s,
        arm_home_velocity_convergence_rad_s=config.runtime.arm.homing.velocity_convergence_rad_s,
        arm_home_result_tolerance_rad=config.runtime.arm.homing.convergence_rad,
        arm_heartbeat_timeout_s=float(config.runtime.safety.heartbeat_timeouts["arm"]),
        hand_feedback_max_age_s=float(config.runtime.safety.heartbeat_timeouts["hand"]),
        estop_requested=estop_requested,
        table_z_surface_m=config.runtime.arm.table_z_surface_m,
        arm_mapper=arm_mapper,
        hand_retargeter=hand_retargeter,
        arm_home_qpos=np.asarray(config.runtime.arm.home_qpos, dtype=np.float64),
    )


def arm_feedback_issue(
    state: np.ndarray | None,
    *,
    now_monotonic_ns: int,
    max_age_s: float,
) -> str | None:
    """Return why an arm state is unsafe to consume, including controller faults."""
    if state is None:
        return "arm feedback unavailable"
    issue = validate_arm_feedback(
        connected=bool(state["connected"][0]),
        error_code=int(state["error_code"][0]),
        state_valid=bool(state["state_valid"][0]),
        source_monotonic_ns=int(state["source_monotonic_ns"][0]),
        now_monotonic_ns=int(now_monotonic_ns),
        max_age_s=max_age_s,
        qpos=np.asarray(state["qpos"][0]),
        qvel=np.asarray(state["qvel"][0]),
    )
    if issue is not None:
        return issue
    return None


def advance_arm_feedback_error_count(
    current_count: int,
    issue: str | None,
    *,
    max_consecutive_errors: int,
) -> tuple[int, bool]:
    """Reset on valid feedback; fault exactly at the configured invalid-frame limit."""
    if issue is None:
        return 0, False
    next_count = current_count + 1
    return next_count, next_count >= max_consecutive_errors


def hand_feedback_issue(
    cfg: TeleopConfig,
    state: np.ndarray | None,
) -> str | None:
    if not cfg.runtime.policy.hand_enabled:
        return None
    if state is None:
        return "hand feedback unavailable"
    return validate_hand_feedback(
        connected=bool(state["connected"][0]),
        state_valid=bool(state["state_valid"][0]),
        source_monotonic_ns=int(state["source_monotonic_ns"][0]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
        qpos=np.asarray(state["qpos"][0]),
    )


def enter_command_quiescence(
    ctx: TeleopLoopState,
    shared: RuntimeChannels,
    quiescence: CommandQuiescence,
    arm_mapper: ArmWristMapper,
    reason: str,
    *,
    start_new_run: bool = False,
    replace_existing_reason: bool = False,
) -> None:
    """Invalidate pending commands, then remain silent until re-anchored.

    Repeated observations preserve the existing boundary and do not advance
    the generation again. Explicit operator transitions may replace the reason
    while retaining that boundary: C marks a resumable pause, whereas a
    session-ending signal cancels that eligibility. A distinct BEGIN supersedes
    the prior pause boundary and always creates a new run generation.
    """
    if start_new_run and replace_existing_reason:
        raise ValueError("BEGIN cannot reuse an existing quiescence boundary")
    if start_new_run:
        previous_reason, _entered_ns = quiescence.clear()
        if previous_reason is not None:
            logger.info(
                "teleop_loop: new run supersedes %s command quiescence",
                previous_reason,
            )
    first_entry = quiescence.enter(
        reason,
        entered_monotonic_ns=time.monotonic_ns(),
    )
    if first_entry:
        # ``begin_motion`` already created the generation for a new run. All
        # other quiescence paths remain RUNNING but must still invalidate the
        # active coupled-command ticket atomically.
        run_generation = (
            int(shared.run_generation.value)
            if start_new_run
            else invalidate_coupled_commands(shared)
        )
        logger.info(
            "teleop_loop: entered %s command quiescence (run=%d)",
            reason,
            run_generation,
        )
    else:
        previous_reason = quiescence.reason
        if replace_existing_reason:
            quiescence.relabel(reason)
        logger.debug(
            "teleop_loop: remaining in %s command quiescence "
            "(observed %s; prior reason=%s)",
            quiescence.reason,
            reason,
            previous_reason,
        )
    arm_mapper.clear()
    ctx.ema_prev_pos = ctx.ema_prev_quat = None
    ctx.hand_ramp_start = None
    ctx.hand_ramp_step = 0
    # No cached observation may cross a generation/quiescence boundary.
    ctx.hand_retarget_cache.reset()


def complete_reanchor(
    ctx: TeleopLoopState,
    arm_mapper: ArmWristMapper,
    validate_warn: ThrottledWarner,
    hand_available: bool,
    current_arm_state: np.ndarray,
    current_vr_frame: dict[str, Any],
    current_hand_state: np.ndarray | None,
) -> bool:
    """Reset temporal state; the caller suppresses this grid's publication."""
    if not _reset_mapper_from_frames(arm_mapper, current_arm_state, current_vr_frame):
        validate_warn(
            "teleop_loop: re-anchor inputs invalid — remaining command-silent"
        )
        return False
    ctx.ema_prev_pos = ctx.ema_prev_quat = None
    hand_anchor: np.ndarray | None = None
    if hand_available:
        if current_hand_state is not None and np.all(
            np.isfinite(current_hand_state["qpos"][0])
        ):
            ctx.prev_hand_qpos = np.asarray(
                current_hand_state["qpos"][0], dtype=np.float64
            ).copy()
        if ctx.prev_hand_qpos is None:
            validate_warn(
                "teleop_loop: hand re-anchor unavailable — remaining command-silent"
            )
            return False
        hand_anchor = ctx.prev_hand_qpos.copy()
    ctx.hand_ramp_start = hand_anchor
    ctx.hand_ramp_step = 0
    ctx.hand_retarget_cache.reset()
    reset_hand_retargeter(ctx.hand_retargeter, hand_anchor)
    return True
