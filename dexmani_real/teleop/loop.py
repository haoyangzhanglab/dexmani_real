"""VR teleoperation coordinator over causal shared-memory snapshots.

This module owns policy-process state transitions, command quiescence, action
proposal orchestration, safety-gated publication, and recording decisions. It
does not own a hardware SDK; arm, hand, VR, camera, and RecorderIO live in
their respective workers.
"""

from __future__ import annotations

import gc
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.constants import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.planning.hand_kinematics import HandKinematics
from dexmani_real.planning.kinematics import make_arm_fk
from dexmani_real.planning.pose_utils import normalize_quat_wxyz, quat_wxyz_to_rot6d
from dexmani_real.policy.loop_timing import StageTimer
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.policy.safety import (
    CommandPublishResult,
    CommandPublishStatus,
    GateRejectCode,
    SafetyGate,
    advance_run_generation,
    build_action_candidate,
    planner_action_safety_gate,
    validate_and_send_candidate,
)
from dexmani_real.recording.recorder_client import RecorderClient, RecorderPhase
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.causal_reader import (
    read_arm_state_causal,
    read_camera_frame_causal,
    read_hand_state_causal,
    read_hand_tactile_causal,
    read_vr_frame_causal,
)
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.teleop.action_proposal import (
    compute_arm_joint_proposal,
    compute_hand_joint_proposal,
    compute_target_eef_pose,
)
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.audio_feedback import AudioFeedback, update_motion_gate
from dexmani_real.teleop.camera_freshness import CameraFreshnessTracker
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_state import CommandQuiescence
from dexmani_real.teleop.episode_samples import (
    _FRAME_IK_FAIL,
    _FRAME_OK,
    _FRAME_RETARGET_FAIL,
    _FRAME_SAFETY_REJECT,
    _record_frame,
    _record_held,
    _stop_recording,
)
from dexmani_real.teleop.hand_control import (
    HandRetargetObservationCache,
    _hand_ramp_frame_count,
    _reset_hand_retargeter,
    _seed_hand_retargeter,
)
from dexmani_real.teleop.hand_retarget import (
    TAGHandRetargeter,
    XHandRetargeter,
    _tag_config_with_urdf,
)
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.recording_session import (
    QuitRecordingDecision,
    await_quit_recording_decision,
)
from dexmani_real.teleop.safety import (
    _do_configured_teleop_home,
    _reset_mapper_from_frames,
)
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.hand_health import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

_END_AUDIO_GRACE_S = 2.0
_NS_PER_SECOND = 1_000_000_000
_VALIDATION_WARN_INTERVAL_S = 2.0
_ARM_FEEDBACK_WARN_INTERVAL_S = 3.0


def _build_safety_gate(config: TeleopConfig, planner: XArm7MotionPlanner) -> SafetyGate:
    """Build the teleoperation safety gate from control-domain limits."""
    return planner_action_safety_gate(
        planner=planner,
        arm_joint_lower_rad=tuple(config.runtime.arm.joint_limit_lower),
        arm_joint_upper_rad=tuple(config.runtime.arm.joint_limit_upper),
        hand_joint_lower_rad=tuple(config.runtime.hand.qpos_min_rad),
        hand_joint_upper_rad=tuple(config.runtime.hand.qpos_max_rad),
    )


def _arm_feedback_issue(
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
        state_valid=bool(state["state_valid"][0]),
        source_monotonic_ns=int(state["source_monotonic_ns"][0]),
        now_monotonic_ns=int(now_monotonic_ns),
        max_age_s=max_age_s,
        qpos=np.asarray(state["qpos"][0]),
        qvel=np.asarray(state["qvel"][0]),
    )
    if issue is not None:
        return issue
    error_code = int(state["error_code"][0])
    return None if error_code == 0 else f"arm controller error C{error_code}"


def _advance_arm_feedback_error_count(
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


def _policy_exit_fault(
    *,
    error_state: bool,
    estop_request: bool,
    safety_fault: bool,
) -> str | None:
    """Classify terminal policy state without losing an e-stop or sticky fault."""
    if estop_request:
        return "policy exited after e-stop request"
    if error_state or safety_fault:
        return "policy exited with sticky fault"
    return None


def _start_keyboard(shared: SharedStorage) -> KeyboardHandler | None:
    """Start the required operator input boundary, failing closed on startup errors."""
    keyboard = KeyboardHandler(
        estop_callback=lambda: setattr(shared.estop_request, "value", True)
    )
    try:
        keyboard.start()
    except Exception:
        logger.error("teleop_loop: keyboard startup failed", exc_info=True)
        shared.error_state.value = True
        return None
    return keyboard


def _hand_feedback_issue_impl(
    cfg: TeleopConfig,
    state: np.ndarray | None,
) -> str | None:
    if not cfg.runtime.policy.hand_enabled:
        return None
    if state is None:
        return "hand feedback unavailable"
    return validate_hand_feedback(
        connected=bool(state["connected"][0]),
        error_state=bool(state["error_state"][0]),
        state_valid=bool(state["state_valid"][0]),
        send_healthy=bool(state["send_healthy"][0]),
        read_healthy=bool(state["read_healthy"][0]),
        source_monotonic_ns=int(state["source_monotonic_ns"][0]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
        qpos=np.asarray(state["qpos"][0]),
    )


@dataclass
class TeleopLoopState:
    """Mutable per-session state shared between the teleop loop body and its
    extracted module-level helper functions.

    The helpers mutate these fields in place (no ``nonlocal``), so the loop
    body and helpers see the same object.  Read-only dependencies (shared,
    config, gate, planner, mapper, quiescence, …) are passed as explicit
    parameters instead of being bundled here.
    """

    hand_retargeter: Any = None
    prev_qpos_cmd: np.ndarray | None = None
    prev_hand_qpos: np.ndarray | None = None
    ema_prev_pos: np.ndarray | None = None
    ema_prev_quat: np.ndarray | None = None
    hand_ramp_start: np.ndarray | None = None
    hand_ramp_step: int = 0
    # Solver results are keyed by verified VR sequence, not every ramp tick.
    hand_retarget_cache: HandRetargetObservationCache = field(
        default_factory=HandRetargetObservationCache
    )
    hand_available: bool = False
    hand_disconnected_at_s: float | None = None
    teleop_active: bool = False
    recording_active: bool = False
    begin_audio_gate_deadline_s: float | None = None
    ignore_begin_audio_until_silent: bool = False
    quit_pending: bool = False
    quit_after_recording: bool = False
    quit_recording_deadline_s: float = 0.0
    post_teleop_deadline_s: float = 0.0
    arm_feedback_error_count: int = 0
    consecutive_ik_hold_frames: int = 0
    ik_hold_started_s: float = 0.0
    last_target_eef_pos: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    last_target_eef_rot6d: np.ndarray = field(
        default_factory=lambda: np.full(6, np.nan)
    )
    sigterm_requested: bool = False


class CoordinatorDirective(str, Enum):
    """Control-flow decision returned by one coordinator state handler."""

    NORMAL = "normal"
    CONTINUE = "continue"
    BREAK = "break"


@dataclass(frozen=True)
class TeleopCommandLimits:
    """Resolved per-process command bounds with units made explicit."""

    arm_joint_lower_rad: np.ndarray
    arm_joint_upper_rad: np.ndarray
    arm_max_delta_rad_per_tick: np.ndarray | None
    hand_home_qpos_rad: np.ndarray
    hand_command_lower_rad: np.ndarray
    hand_command_upper_rad: np.ndarray
    hand_mechanical_lower_rad: np.ndarray
    hand_mechanical_upper_rad: np.ndarray
    workspace_bounds_world_m: np.ndarray

    @classmethod
    def from_config(cls, config: TeleopConfig) -> TeleopCommandLimits:
        arm_joint_lower_rad = np.asarray(
            config.runtime.arm.joint_limit_lower, dtype=np.float64
        ).copy()
        arm_joint_upper_rad = np.asarray(
            config.runtime.arm.joint_limit_upper, dtype=np.float64
        ).copy()
        configured_max_delta = config.runtime.policy.arm_max_delta_rad_per_tick
        arm_max_delta_rad_per_tick = (
            None
            if configured_max_delta is None
            else np.broadcast_to(
                np.asarray(configured_max_delta, dtype=np.float64),
                arm_joint_lower_rad.shape,
            ).copy()
        )
        return cls(
            arm_joint_lower_rad=arm_joint_lower_rad,
            arm_joint_upper_rad=arm_joint_upper_rad,
            arm_max_delta_rad_per_tick=arm_max_delta_rad_per_tick,
            hand_home_qpos_rad=np.deg2rad(
                np.asarray(config.runtime.hand.home_qpos_deg, dtype=np.float64)
            ),
            hand_command_lower_rad=np.asarray(
                config.runtime.hand.qpos_min_rad, dtype=np.float64
            ).copy(),
            hand_command_upper_rad=np.asarray(
                config.runtime.hand.qpos_max_rad, dtype=np.float64
            ).copy(),
            hand_mechanical_lower_rad=np.asarray(
                config.runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ).copy(),
            hand_mechanical_upper_rad=np.asarray(
                config.runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ).copy(),
            workspace_bounds_world_m=np.asarray(
                config.runtime.policy.workspace.as_tuple(), dtype=np.float64
            ).copy(),
        )


@dataclass(frozen=True)
class TeleopControlResources:
    """Planning, mapping, safety, and recording resources for one policy process."""

    planner: XArm7MotionPlanner
    arm_mapper: ArmWristMapper
    safety_gate: SafetyGate
    recorder: RecorderClient | None


@dataclass(frozen=True)
class TeleopOperatorResources:
    """Resources used only while applying operator control signals."""

    control: TeleopControlResources
    keyboard: KeyboardHandler
    audio: AudioFeedback
    limiter: RateManager
    quiescence: CommandQuiescence
    camera_freshness: CameraFreshnessTracker


@dataclass(frozen=True)
class TeleopGridResources:
    """Read-only dependencies used to execute one control-grid observation."""

    control: TeleopControlResources
    command_limits: TeleopCommandLimits
    quiescence: CommandQuiescence
    camera_freshness: CameraFreshnessTracker
    stage_timer: StageTimer
    validation_warn: ThrottledWarner
    arm_feedback_warn: ThrottledWarner
    hand_fk: HandKinematics | None
    handbase_position_eef_m: np.ndarray
    handbase_quat_eef_wxyz: np.ndarray
    hand_ramp_total_frames: int
    audio: AudioFeedback


@dataclass(frozen=True)
class TeleopGridObservation:
    """One validated causal observation ready for command computation."""

    arm_state: np.ndarray
    arm_qpos_rad: np.ndarray
    vr_frame: dict[str, Any]
    camera_frame: dict[str, Any] | None
    hand_state: np.ndarray | None
    hand_tactile: np.ndarray | None
    anchor_monotonic_ns: int


@dataclass(frozen=True)
class TeleopActionComputation:
    """Mapped targets, solver result, and diagnostics for one grid tick."""

    target_position_world_m: np.ndarray
    target_quat_world_wxyz: np.ndarray
    raw_target_position_world_m: np.ndarray
    raw_target_quat_world_wxyz: np.ndarray
    position_before_workspace_clamp_world_m: np.ndarray
    hand_qpos_rad: np.ndarray
    raw_hand_qpos_rad: np.ndarray
    hand_retarget_succeeded: bool
    hand_validation_issue: str | None
    hand_retarget_time_ms: float
    ik_qpos_rad: np.ndarray | None
    ik_failure_reason: str
    ik_solve_time_ms: float
    policy_map_time_ms: float
    policy_compute_started_s: float


def _load_control_resources(
    shared: SharedStorage,
    config: TeleopConfig,
    *,
    recording_enabled: bool,
) -> TeleopControlResources:
    """Load the non-hardware resources owned by one teleoperation policy process."""
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(
                p=np.array([0.0, 0.0, 0.0]),
                q=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            workspace_bounds=np.asarray(
                config.runtime.policy.workspace.as_tuple(), dtype=np.float64
            ),
        ),
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=config.runtime.policy.ik_max_pose_error_pos_m,
            max_pose_error_rot_rad=config.runtime.policy.ik_max_pose_error_rot_rad,
            nullspace_step_size_deg=(
                config.runtime.policy.ik_nullspace_step_rate_deg_s
                / config.runtime.policy.control_hz
            ),
        ),
        hand_dof=True,
        static_boxes=config.runtime.environment.static_boxes,
        table=config.runtime.environment.table,
    )

    vr_config_path = Path(__file__).resolve().parents[2] / config.vr_transform_path
    vr_calibration = load_vr_transform(vr_config_path)
    rotation_robot_vr = vr_calibration.transform
    logger.info("VR transform loaded: theta=%.6g°", vr_calibration.theta_deg)
    arm_mapper = ArmWristMapper(
        pos_scale=config.runtime.policy.vr_mapping.pos_scale,
        rot_scale=config.runtime.policy.vr_mapping.rot_scale,
        vr_to_base_rot=rotation_robot_vr,
        T_vr_to_robot=rotation_robot_vr,
        max_delta_rot_rad=config.runtime.policy.vr_mapping.max_delta_rot_rad,
        base_to_world_rot=np.eye(3, dtype=np.float64),
    )
    return TeleopControlResources(
        planner=planner,
        arm_mapper=arm_mapper,
        safety_gate=_build_safety_gate(config, planner),
        recorder=RecorderClient(shared) if recording_enabled else None,
    )


def _try_load_hand_kinematics(
    config: TeleopConfig,
    *,
    recording_enabled: bool,
) -> HandKinematics | None:
    """Load optional recording-only hand FK, falling back to NaN fingertips."""
    if not recording_enabled or not config.hand_urdf_path:
        return None
    try:
        hand_fk = HandKinematics(
            config.hand_urdf_path,
            list(config.runtime.hand.fingertip_link_names),
        )
    except Exception:
        logger.warning("Hand FK initialization failed", exc_info=True)
        return None
    if hand_fk.is_ready():
        logger.info("Hand FK ready")
    else:
        logger.warning("Hand FK not ready — fingertips will be NaN")
    return hand_fk


def _wait_for_enabled_capabilities(
    shared: SharedStorage,
    config: TeleopConfig,
    *,
    recording_enabled: bool,
) -> tuple[str, float] | None:
    """Wait for required process readiness and return the first timeout."""
    capability_names = ["arm", "vr"]
    if recording_enabled:
        capability_names += ["camera", "recorder"]
    if config.runtime.policy.hand_enabled:
        capability_names.insert(1, "hand")
    for capability_name in capability_names:
        timeout_s = float(
            dict(config.runtime.safety.readiness_timeouts_s)[capability_name]
        )
        if not shared.wait_ready(capability_name, timeout_s):
            return capability_name, timeout_s
    return None


def _try_init_hand_retargeter_impl(ctx: TeleopLoopState, cfg: TeleopConfig) -> bool:
    """Lazily initialize ctx.hand_retargeter if not already created."""
    if ctx.hand_retargeter is not None:
        return True
    try:
        if cfg.runtime.policy.hand_retargeting_type == "tag":
            ctx.hand_retargeter = TAGHandRetargeter(
                hand_type="right",
                fingertip_link_names=cfg.runtime.hand.fingertip_link_names,
                tag_config=_tag_config_with_urdf(
                    cfg.runtime.tag_retargeting, cfg.hand_urdf_path
                ),
            )
        else:
            ctx.hand_retargeter = XHandRetargeter(
                hand_type="right",
                retargeting_type=cfg.runtime.policy.hand_retargeting_type,
                dexpilot_config=cfg.runtime.dexpilot_retargeting,
            )
        logger.info(
            "Hand retargeter ready (type=%s)", cfg.runtime.policy.hand_retargeting_type
        )
        return True
    except Exception:
        logger.error("Hand retargeter initialization failed", exc_info=True)
        ctx.hand_retargeter = None
        return False


def _init_and_seed_hand_retargeter_impl(
    ctx: TeleopLoopState, cfg: TeleopConfig, shared: SharedStorage
) -> np.ndarray | None:
    """Lazy-init retargeter and seed NLP warm-start from hardware qpos.

    Returns the seeded qpos (for updating ``ctx.prev_hand_qpos``) or None.
    """
    if not cfg.runtime.policy.hand_enabled:
        return None
    if not _try_init_hand_retargeter_impl(ctx, cfg):
        return None
    hs = read_hand_state_causal(shared)
    qpos = (
        hs["qpos"][0]
        if _hand_feedback_issue_impl(cfg, hs) is None and hs is not None
        else None
    )
    return _seed_hand_retargeter(ctx.hand_retargeter, qpos)


def _enter_command_quiescence_impl(
    ctx: TeleopLoopState,
    shared: SharedStorage,
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
        run_generation = advance_run_generation(shared)
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


def _complete_reanchor_impl(
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
    _reset_hand_retargeter(ctx.hand_retargeter, hand_anchor)
    return True


def _transition_or_fault(
    shared: SharedStorage,
    new_state: SafetyState,
    reason: str,
) -> bool:
    """Apply one safety transition and make any rejection a sticky fault."""
    if transition(shared, new_state):
        return True
    logger.error(
        "teleop_loop: safety transition to %s failed during %s",
        new_state.name,
        reason,
    )
    shared.error_state.value = True
    return False


def _enter_operator_quiescence(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    resources: TeleopOperatorResources,
    reason: str,
    *,
    start_new_run: bool = False,
    replace_existing_reason: bool = False,
) -> None:
    _enter_command_quiescence_impl(
        ctx,
        shared,
        resources.quiescence,
        resources.control.arm_mapper,
        reason,
        start_new_run=start_new_run,
        replace_existing_reason=replace_existing_reason,
    )


def _handoff_quiescence_to_home(resources: TeleopOperatorResources) -> None:
    reason, _entered_ns = resources.quiescence.clear()
    if reason is not None:
        logger.info(
            "teleop_loop: homing supersedes %s command quiescence",
            reason,
        )


def _keyboard_estop_requested(keyboard: KeyboardHandler) -> bool:
    return keyboard.estop_latched or not keyboard.healthy


def _apply_quit_signal(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopOperatorResources,
) -> CoordinatorDirective:
    """Enter the bounded post-teleop state after an operator quit request."""
    recorder = resources.control.recorder
    print("\nQ: 退出")
    resources.audio.play("quit")
    _enter_operator_quiescence(
        ctx,
        shared,
        resources,
        "quit",
        replace_existing_reason=True,
    )
    ctx.teleop_active = False
    if not _transition_or_fault(shared, SafetyState.ARMED, "quit"):
        return CoordinatorDirective.BREAK

    if ctx.recording_active:
        resources.audio.queue("quit_save_prompt")
        print(
            "  [S] 保存并退出  [D] 丢弃并退出  [H] 保存并归位 "
            f"({cfg.runtime.policy.quit_save_timeout_s:.0f}s 超时默认丢弃)"
        )
        decision = await_quit_recording_decision(
            shared,
            resources.keyboard,
            timeout_s=cfg.runtime.policy.quit_save_timeout_s,
        )
        save = decision in (
            QuitRecordingDecision.SAVE,
            QuitRecordingDecision.SAVE_AND_HOME,
        )
        if decision is QuitRecordingDecision.ESTOP:
            resources.audio.play("emergency")
        else:
            resources.audio.play("save" if save else "discard")
        _stop_recording(
            recorder,
            ctx.recording_active,
            save=save,
            shared=shared,
        )
        ctx.recording_active = False
        if decision is QuitRecordingDecision.TIMEOUT:
            print("  超时，默认丢弃请求已提交")
        elif decision is QuitRecordingDecision.DISCARD:
            print("  丢弃请求已提交")
        elif save:
            print("  保存请求已提交")

        if decision is QuitRecordingDecision.SAVE_AND_HOME and shared.is_running.value:
            resources.audio.play("home")
            ctx.ema_prev_pos = ctx.ema_prev_quat = None
            _handoff_quiescence_to_home(resources)
            ctx.prev_hand_qpos = _do_configured_teleop_home(
                shared,
                cfg,
                hand_available=ctx.hand_available,
                prev_hand_qpos=ctx.prev_hand_qpos,
                planner=resources.control.planner,
                audio=resources.audio,
                estop_requested=lambda: _keyboard_estop_requested(resources.keyboard),
                arm_mapper=resources.control.arm_mapper,
                hand_retargeter=ctx.hand_retargeter,
            )

    ctx.quit_pending = True
    ctx.post_teleop_deadline_s = (
        time.perf_counter() + cfg.runtime.policy.post_teleop_timeout_s
    )
    print(
        f"\n[H] return_home  [Q] quit  ({cfg.runtime.policy.post_teleop_timeout_s:.0f}s timeout)",
        flush=True,
    )
    return CoordinatorDirective.CONTINUE


def _apply_home_signal(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopOperatorResources,
) -> CoordinatorDirective:
    """Stop the current session and execute the configured home operation."""
    print("\nH: return_home")
    resources.audio.play("home")
    _stop_recording(
        resources.control.recorder,
        ctx.recording_active,
        save=True,
        shared=shared,
    )
    ctx.recording_active = False
    ctx.teleop_active = False
    if not _transition_or_fault(shared, SafetyState.ARMED, "home"):
        return CoordinatorDirective.BREAK
    ctx.ema_prev_pos = ctx.ema_prev_quat = None
    _handoff_quiescence_to_home(resources)
    ctx.prev_hand_qpos = _do_configured_teleop_home(
        shared,
        cfg,
        hand_available=ctx.hand_available,
        prev_hand_qpos=ctx.prev_hand_qpos,
        planner=resources.control.planner,
        audio=resources.audio,
        estop_requested=lambda: _keyboard_estop_requested(resources.keyboard),
        arm_mapper=resources.control.arm_mapper,
        hand_retargeter=ctx.hand_retargeter,
    )
    resources.keyboard.drain_signal(ControlSignal.HOME)
    resources.limiter.reset()
    return CoordinatorDirective.CONTINUE


def _apply_pause_signal(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    resources: TeleopOperatorResources,
) -> bool:
    """Pause or resume one existing session; return false on a safety fault."""
    if ctx.teleop_active:
        _enter_operator_quiescence(
            ctx,
            shared,
            resources,
            "pause",
            replace_existing_reason=True,
        )
        ctx.teleop_active = False
        if not _transition_or_fault(shared, SafetyState.ARMED, "pause"):
            return False
    else:
        if resources.quiescence.reason != "pause":
            print("\nC: 没有可恢复的暂停 session — 请按 B 开始新的遥操作 session")
            return True
        if shared.safety_state.value != SafetyState.ARMED:
            print(
                f"\nC: safety_state={shared.safety_state.value} — must be ARMED to resume"
            )
            return True
        if not _transition_or_fault(shared, SafetyState.RUNNING, "resume"):
            return False
        ctx.teleop_active = True
    state_str = "恢复" if ctx.teleop_active else "暂停"
    print(f"\nC: {state_str}遥操作")
    resources.audio.play("resume" if ctx.teleop_active else "pause")
    return True


def _apply_begin_signal(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopOperatorResources,
) -> bool:
    """Start a new run and optional recording transaction."""
    if ctx.teleop_active or ctx.recording_active:
        print(
            "\nB: session already active — use C to pause/resume, S to save, or D to discard"
        )
        return True
    if shared.safety_state.value != SafetyState.ARMED:
        print(
            f"\nB: safety_state={shared.safety_state.value} — must be ARMED({SafetyState.ARMED})"
        )
        return True
    if read_vr_frame_causal(shared) is None:
        print("\nB: 无 VR 帧，无法开始遥操作")
        return True
    begin_hand_state = (
        read_hand_state_causal(shared) if cfg.runtime.policy.hand_enabled else None
    )
    begin_hand_issue = _hand_feedback_issue_impl(cfg, begin_hand_state)
    if begin_hand_issue is not None:
        print(f"\nB: hand feedback unhealthy ({begin_hand_issue}) — cannot begin")
        return True

    recorder = resources.control.recorder
    _stop_recording(
        recorder,
        ctx.recording_active,
        save=ctx.recording_active,
        shared=shared,
    )
    gc.collect()
    if recorder is None:
        ctx.recording_active = False
        shared.is_recording.value = False
        begin_reason = "begin"
        begin_message = "\nB: 遥操作开始（未启用录制 capability）"
    else:
        if not recorder.start_episode(
            task_label=cfg.task_label,
            operator=cfg.operator,
        ):
            print("  ⚠ 无法开始录制")
            return True
        ctx.recording_active = True
        resources.camera_freshness.reset(time.monotonic())
        shared.is_recording.value = True
        begin_reason = "begin recording"
        begin_message = f"\nB: 遥操作+录制开始  episode={recorder.frame_count}"

    resources.keyboard.drain_signal(ControlSignal.BEGIN)
    if not _transition_or_fault(shared, SafetyState.RUNNING, begin_reason):
        _stop_recording(
            recorder,
            ctx.recording_active,
            save=False,
            shared=shared,
            reason="safety_transition_failed",
        )
        ctx.recording_active = False
        return False
    _enter_operator_quiescence(ctx, shared, resources, "begin", start_new_run=True)
    ctx.teleop_active = True
    logger.debug("teleop_loop: RUNNING")
    seeded_qpos = _init_and_seed_hand_retargeter_impl(ctx, cfg, shared)
    if seeded_qpos is not None:
        ctx.prev_hand_qpos = seeded_qpos
    resources.audio.play("begin")
    control_period_s = 1.0 / cfg.runtime.policy.control_hz
    ctx.begin_audio_gate_deadline_s = time.monotonic() + max(
        0.0,
        cfg.runtime.policy.begin_motion_gate_timeout_s - control_period_s,
    )
    ctx.ignore_begin_audio_until_silent = False
    print(begin_message)
    resources.limiter.reset()
    return True


def _apply_operator_controls(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopOperatorResources,
    controls: tuple[ControlSignal, ...],
) -> CoordinatorDirective:
    """Apply queued controls while preserving their original ordering semantics."""
    skip_control_tick = False
    for control in controls:
        if control is ControlSignal.EMERGENCY_STOP:
            print("\nESC: emergency_stop")
            resources.audio.play("emergency")
            shared.estop_request.value = True
            _stop_recording(
                resources.control.recorder,
                ctx.recording_active,
                save=False,
                shared=shared,
            )
            ctx.recording_active = False
            return CoordinatorDirective.BREAK
        if control is ControlSignal.QUIT:
            return _apply_quit_signal(ctx, shared, cfg, resources)
        if control is ControlSignal.HOME:
            return _apply_home_signal(ctx, shared, cfg, resources)
        if control in (ControlSignal.STOP, ControlSignal.DISCARD):
            save_episode = control is ControlSignal.STOP
            stop_reason = "stop" if save_episode else "discard"
            print("\nS: 停止录制" if save_episode else "\nD: 丢弃录制")
            resources.audio.play("save" if save_episode else "discard")
            _enter_operator_quiescence(
                ctx,
                shared,
                resources,
                stop_reason,
                replace_existing_reason=True,
            )
            _stop_recording(
                resources.control.recorder,
                ctx.recording_active,
                save=save_episode,
                shared=shared,
            )
            ctx.recording_active = False
            ctx.teleop_active = False
            if not _transition_or_fault(shared, SafetyState.ARMED, stop_reason):
                return CoordinatorDirective.BREAK
            skip_control_tick = True
        elif control is ControlSignal.PAUSE:
            if not _apply_pause_signal(ctx, shared, resources):
                return CoordinatorDirective.BREAK
            skip_control_tick = True
        elif control is ControlSignal.BEGIN:
            if not _apply_begin_signal(ctx, shared, cfg, resources):
                return CoordinatorDirective.BREAK
            skip_control_tick = True

    if (
        shared.estop_request.value
        or shared.quit_requested.value
        or not shared.is_running.value
        or shared.error_state.value
    ):
        return CoordinatorDirective.BREAK
    if skip_control_tick:
        return CoordinatorDirective.CONTINUE
    return CoordinatorDirective.NORMAL


def _poll_recording_lifecycle(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    recorder: RecorderClient | None,
    audio: AudioFeedback,
    *,
    enter_quiescence: Callable[..., None],
    transition_or_fault: Callable[[SafetyState, str], bool],
) -> bool:
    """Poll asynchronous recorder state and handle writer failure fail-closed."""
    if recorder is not None:
        stop_result = recorder.poll_stop()
        reached_limit = (
            stop_result.phase
            in (
                RecorderPhase.FINALIZING,
                RecorderPhase.COMPLETED,
                RecorderPhase.ERROR,
            )
            and stop_result.reason == "max_frames"
            and (ctx.teleop_active or ctx.recording_active)
        )
        if reached_limit:
            enter_quiescence("max_frames", replace_existing_reason=True)
            ctx.teleop_active = False
            ctx.recording_active = False
            shared.is_recording.value = False
            if not transition_or_fault(SafetyState.ARMED, "maximum recording duration"):
                return False
            print("  已达到最大录制时长：正在自动保存，遥操作进入静默暂停")
            audio.play("pause")
        if stop_result.done:
            ctx.recording_active = False
            shared.is_recording.value = False
            if stop_result.error:
                path_label = f": {stop_result.path}" if stop_result.path else ""
                print(f"  ⚠ 录制终结失败 ({stop_result.error}){path_label}")
            elif stop_result.saved:
                print(
                    f"  录制已保存: {stop_result.path}  ({stop_result.frame_count} 帧)"
                )
                if not stop_result.min_frames_met:
                    print("  ⚠ 已保存，但未达到配置的最短质量时长")
            else:
                print(f"  录制已丢弃 ({stop_result.frame_count} 帧)")
            gc.collect()
            if ctx.quit_after_recording:
                shared.quit_requested.value = True
        elif stop_result.phase is RecorderPhase.FINALIZING and stop_result.error:
            print("  ⚠ 录制终结超过时限；仍在安全回收，本会话将标记为失败")

    writer_error = (
        recorder.camera_writer_error
        if recorder is not None and ctx.recording_active
        else None
    )
    if writer_error is None:
        return True
    logger.error(
        "Camera writer failed — discarding current episode: %s",
        writer_error,
    )
    print(f"  ⚠ 相机写盘失败，当前 episode 已废弃: {writer_error}")
    _stop_recording(
        recorder,
        ctx.recording_active,
        save=False,
        shared=shared,
        reason="camera_writer_error",
    )
    ctx.recording_active = False
    return True


def _advance_post_teleop_state(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    kb: KeyboardHandler,
    audio: AudioFeedback,
    planner: XArm7MotionPlanner,
    limiter: RateManager,
    recorder: RecorderClient | None,
    *,
    handoff_quiescence_to_home: Callable[[], None],
    keyboard_estop_requested: Callable[[], bool],
) -> CoordinatorDirective:
    """Keep workers alive after Q for optional home and bounded recorder exit."""
    if not ctx.quit_pending:
        return CoordinatorDirective.NORMAL

    home_handled = False
    for control in kb.poll(timeout=0.1):
        if control is ControlSignal.HOME:
            if home_handled:
                continue
            home_handled = True
            print("  H: return_home")
            audio.play("home")
            handoff_quiescence_to_home()
            ctx.prev_hand_qpos = _do_configured_teleop_home(
                shared,
                cfg,
                hand_available=ctx.hand_available,
                prev_hand_qpos=ctx.prev_hand_qpos,
                planner=planner,
                audio=audio,
                estop_requested=keyboard_estop_requested,
            )
            kb.drain_signal(ControlSignal.HOME)
            limiter.reset()
            print("  [Q] quit", flush=True)
        elif control in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
            if control is ControlSignal.EMERGENCY_STOP:
                shared.estop_request.value = True
            elif recorder is not None and recorder.stop_pending:
                if not ctx.quit_after_recording:
                    ctx.quit_after_recording = True
                    ctx.quit_recording_deadline_s = (
                        time.monotonic() + cfg.runtime.policy.quit_save_timeout_s
                    )
                print("  录制仍在终结；完成后自动退出", flush=True)
            else:
                shared.quit_requested.value = True
                break

    if (
        shared.estop_request.value
        or shared.quit_requested.value
        or not shared.is_running.value
    ):
        return CoordinatorDirective.BREAK

    recording_stop_pending = recorder is not None and recorder.stop_pending
    if (
        ctx.quit_after_recording
        and recording_stop_pending
        and time.monotonic() >= ctx.quit_recording_deadline_s
    ):
        print("  录制终结超时 — 退出并将本会话标记为失败")
        shared.quit_requested.value = True
        return CoordinatorDirective.BREAK

    if time.perf_counter() <= ctx.post_teleop_deadline_s:
        return CoordinatorDirective.CONTINUE
    if recording_stop_pending:
        if not ctx.quit_after_recording:
            ctx.quit_after_recording = True
            ctx.quit_recording_deadline_s = (
                time.monotonic() + cfg.runtime.policy.quit_save_timeout_s
            )
            print("  timeout — 等待录制终结后自动退出", flush=True)
        elif time.monotonic() >= ctx.quit_recording_deadline_s:
            print("  录制终结超时 — 退出并将本会话标记为失败")
            shared.quit_requested.value = True
            return CoordinatorDirective.BREAK
    else:
        print("  timeout — auto exit")
        shared.quit_requested.value = True
        return CoordinatorDirective.BREAK
    return CoordinatorDirective.CONTINUE


def _record_grid_hold(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
    *,
    action_candidate: ActionCandidate | None = None,
    frame_status: int | None = None,
    retarget_ok: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    """Record one fallback command with the common causal grid provenance."""
    if not ctx.recording_active:
        return
    assert ctx.prev_qpos_cmd is not None
    assert ctx.prev_hand_qpos is not None
    kwargs: dict[str, Any] = {}
    if frame_status is not None:
        kwargs["frame_status"] = frame_status
    _record_held(
        resources.control.recorder,
        observation.arm_state,
        ctx.prev_qpos_cmd,
        ctx.prev_hand_qpos,
        observation.vr_frame,
        observation.camera_frame,
        hand_state=observation.hand_state,
        hand_tactile=observation.hand_tactile,
        retarget_ok=retarget_ok,
        arm_qpos_sent=ctx.prev_qpos_cmd.copy(),
        diagnostics=diagnostics,
        target_eef_pos=ctx.last_target_eef_pos,
        target_eef_rot6d=ctx.last_target_eef_rot6d,
        hand_fk=resources.hand_fk,
        T_eef_handbase_pos=resources.handbase_position_eef_m,
        T_eef_handbase_quat_wxyz=resources.handbase_quat_eef_wxyz,
        observation_anchor_monotonic_ns=observation.anchor_monotonic_ns,
        shared=shared,
        action_candidate=action_candidate,
        **kwargs,
    )


def _read_control_grid_observation(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    *,
    loop_count: int,
    observation_anchor_monotonic_ns: int,
) -> tuple[CoordinatorDirective, TeleopGridObservation | None]:
    """Read and validate one causal sensor cut, remaining silent when unsafe."""
    assert ctx.prev_qpos_cmd is not None
    assert ctx.prev_hand_qpos is not None
    arm_mapper = resources.control.arm_mapper
    recorder = resources.control.recorder
    _quiescence = resources.quiescence
    _camera_freshness = resources.camera_freshness
    stage_timer = resources.stage_timer
    _validate_warn = resources.validation_warn
    _arm_feedback_warn = resources.arm_feedback_warn
    _hand_fk = resources.hand_fk
    _T_eef_handbase_pos = resources.handbase_position_eef_m
    _T_eef_handbase_quat_wxyz = resources.handbase_quat_eef_wxyz
    _current_grid_anchor_ns = observation_anchor_monotonic_ns
    audio = resources.audio

    def _enter_command_quiescence(reason: str) -> None:
        _enter_command_quiescence_impl(
            ctx,
            shared,
            _quiescence,
            arm_mapper,
            reason,
        )

    def _complete_reanchor(
        current_arm_state: np.ndarray,
        current_vr_frame: dict[str, Any],
        current_hand_state: np.ndarray | None,
    ) -> bool:
        return _complete_reanchor_impl(
            ctx,
            arm_mapper,
            _validate_warn,
            ctx.hand_available,
            current_arm_state,
            current_vr_frame,
            current_hand_state,
        )

    arm_state = read_arm_state_causal(
        shared, anchor_monotonic_ns=_current_grid_anchor_ns
    )
    arm_issue = _arm_feedback_issue(
        arm_state,
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=cfg.runtime.policy.arm_state_stale_threshold_s,
    )
    ctx.arm_feedback_error_count, arm_feedback_fault = (
        _advance_arm_feedback_error_count(
            ctx.arm_feedback_error_count,
            arm_issue,
            max_consecutive_errors=cfg.runtime.policy.max_consecutive_errors,
        )
    )
    if arm_issue is not None:
        _arm_feedback_warn(
            "teleop_loop: invalid arm feedback (%d/%d): %s",
            ctx.arm_feedback_error_count,
            cfg.runtime.policy.max_consecutive_errors,
            arm_issue,
        )
        if ctx.teleop_active and not _quiescence.active:
            _enter_command_quiescence("arm_feedback")
        if arm_feedback_fault:
            logger.error("teleop_loop: arm feedback fault: %s", arm_issue)
            shared.error_state.value = True
            return CoordinatorDirective.BREAK, None
        return CoordinatorDirective.CONTINUE, None
    assert arm_state is not None  # validation above proved availability
    arm_qpos = arm_state["qpos"][0].copy()

    vr_frame = read_vr_frame_causal(shared, anchor_monotonic_ns=_current_grid_anchor_ns)
    vr_stale = vr_frame is None or (
        (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0))
        > cfg.runtime.policy.vr_mapping.stale_threshold_s * 1e9
    )
    stage_timer.mark("vr")

    # VR control does not consume camera pixels.  Scan/copy the large
    # payload only while the policy-owned recorder requests it.
    cam = (
        read_camera_frame_causal(shared, anchor_monotonic_ns=_current_grid_anchor_ns)
        if ctx.recording_active
        else None
    )
    if ctx.recording_active:
        cam, _camera_stalled = _camera_freshness.observe(cam)
        if _camera_stalled:
            logger.error(
                "Camera source stale for %.1fs — discarding episode; teleoperation remains RUNNING",
                cfg.runtime.camera.recording_stall_abort_s,
            )
            print("  ⚠ 相机连续失帧超过阈值，当前 episode 已废弃；遥操作继续")
            _stop_recording(
                recorder,
                ctx.recording_active,
                save=False,
                shared=shared,
                reason="camera_stall",
            )
            ctx.recording_active = False
    stage_timer.mark("cam")

    hand_state = read_hand_state_causal(
        shared, anchor_monotonic_ns=_current_grid_anchor_ns
    )
    hand_tactile = read_hand_tactile_causal(
        shared, anchor_monotonic_ns=_current_grid_anchor_ns
    )

    hand_issue = _hand_feedback_issue_impl(cfg, hand_state)
    if cfg.runtime.policy.hand_enabled and hand_issue is not None:
        now_s = time.monotonic()
        if ctx.hand_disconnected_at_s is None:
            ctx.hand_disconnected_at_s = now_s
            logger.warning("Hand feedback unhealthy — pausing motion: %s", hand_issue)
        if ctx.teleop_active and not _quiescence.active:
            _enter_command_quiescence("hand_feedback")
        unhealthy_duration_s = now_s - ctx.hand_disconnected_at_s
        if unhealthy_duration_s >= cfg.runtime.policy.hand_disconnect_timeout_s:
            logger.error(
                "Hand feedback remained unhealthy for %.1fs: %s",
                unhealthy_duration_s,
                hand_issue,
            )
            shared.error_state.value = True
            return CoordinatorDirective.BREAK, None
    elif cfg.runtime.policy.hand_enabled and ctx.hand_disconnected_at_s is not None:
        unhealthy_duration_s = time.monotonic() - ctx.hand_disconnected_at_s
        ctx.hand_disconnected_at_s = None
        logger.info(
            "Hand feedback recovered after %.1fs — waiting for fresh re-anchor",
            unhealthy_duration_s,
        )

    if loop_count % cfg.runtime.policy.status_print_interval == 0:
        _arm_age = (
            (time.monotonic_ns() - int(arm_state["source_monotonic_ns"][0])) * 1e-9
            if arm_state is not None
            else -1.0
        )
        _qdepth = -1  # latest-wins arm ring has no queue depth
        _print_status(
            loop_count,
            arm_state,
            vr_frame,
            ctx.teleop_active,
            ctx.recording_active,
            ctx.arm_feedback_error_count,
            arm_q_depth=_qdepth,
            arm_state_age_s=_arm_age,
        )

    if ctx.teleop_active and vr_stale and not _quiescence.active:
        _enter_command_quiescence("vr_stale")

    # BEGIN remains command-silent until its audio cue releases or times out.
    _audio_playing = audio.is_playing
    (
        _hold_for_audio,
        ctx.begin_audio_gate_deadline_s,
        ctx.ignore_begin_audio_until_silent,
    ) = update_motion_gate(
        audio_playing=_audio_playing,
        begin_deadline_s=ctx.begin_audio_gate_deadline_s,
        ignore_begin_until_silent=ctx.ignore_begin_audio_until_silent,
        now_s=time.monotonic(),
    )
    if ctx.teleop_active and _hold_for_audio and not _quiescence.active:
        _enter_command_quiescence("audio_gate")

    if not ctx.teleop_active or vr_stale or _quiescence.active:
        # Resume only with feedback newer than the quiescence boundary.
        if (
            ctx.teleop_active
            and not vr_stale
            and not _hold_for_audio
            and _quiescence.active
            and vr_frame is not None
            and (not ctx.hand_available or hand_state is not None)
            and (not ctx.hand_available or hand_issue is None)
            and _quiescence.feedback_is_newer(
                arm_source_monotonic_ns=int(arm_state["source_monotonic_ns"][0]),
                vr_receive_monotonic_ns=int(vr_frame["local_recv_ns"]),
                hand_source_monotonic_ns=(
                    int(hand_state["source_monotonic_ns"][0])
                    if ctx.hand_available and hand_state is not None
                    else None
                ),
            )
        ):
            if _complete_reanchor(arm_state, vr_frame, hand_state):
                quiescence_reason = _quiescence.reason
                _quiescence.clear()
                logger.info(
                    "teleop_loop: released %s command quiescence after fresh re-anchor",
                    quiescence_reason,
                )
        # Track measured position while silent without publishing a hold target.
        ctx.prev_qpos_cmd = arm_qpos.copy()
        ctx.ema_prev_pos = ctx.ema_prev_quat = None
        return CoordinatorDirective.CONTINUE, None

    if vr_frame is None:
        logger.warning(
            "teleop_loop: vr_frame is None after vr_stale check — suppressing publication"
        )
        if ctx.recording_active:
            _record_held(
                recorder,
                arm_state,
                ctx.prev_qpos_cmd,
                ctx.prev_hand_qpos,
                None,  # vr_frame
                cam,
                hand_state=hand_state,
                hand_tactile=hand_tactile,
                arm_qpos_sent=ctx.prev_qpos_cmd.copy(),
                target_eef_pos=ctx.last_target_eef_pos,
                target_eef_rot6d=ctx.last_target_eef_rot6d,
                hand_fk=_hand_fk,
                T_eef_handbase_pos=_T_eef_handbase_pos,
                T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                shared=shared,
            )
        return CoordinatorDirective.CONTINUE, None

    return (
        CoordinatorDirective.NORMAL,
        TeleopGridObservation(
            arm_state=arm_state,
            arm_qpos_rad=arm_qpos,
            vr_frame=vr_frame,
            camera_frame=cam,
            hand_state=hand_state,
            hand_tactile=hand_tactile,
            anchor_monotonic_ns=observation_anchor_monotonic_ns,
        ),
    )


def _compute_action_computation(
    ctx: TeleopLoopState,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
) -> TeleopActionComputation | None:
    """Map one validated observation and solve its arm/hand proposal."""
    assert ctx.prev_qpos_cmd is not None
    assert ctx.prev_hand_qpos is not None
    compute_started_s = time.perf_counter()
    map_started_s = time.perf_counter()
    mapped = resources.control.arm_mapper.map(
        observation.vr_frame["wrist_pos"],
        observation.vr_frame["wrist_quat_wxyz"],
    )
    if mapped is None:
        return None

    command_limits = resources.command_limits
    target = compute_target_eef_pose(
        mapped["pos"],
        mapped["quat_wxyz"],
        previous_position_world_m=ctx.ema_prev_pos,
        previous_quat_world_wxyz=ctx.ema_prev_quat,
        workspace_bounds_world_m=command_limits.workspace_bounds_world_m,
        ema_alpha_position=cfg.runtime.policy.ema.alpha_pos,
        ema_alpha_rotation=cfg.runtime.policy.ema.alpha_rot,
    )
    if target.smoothing_state_incomplete:
        logger.warning("teleop_loop: previous EEF quaternion is missing — skipping EMA")
    policy_map_time_ms = (time.perf_counter() - map_started_s) * 1000.0
    resources.stage_timer.mark("map")

    hand = compute_hand_joint_proposal(
        ctx.hand_retargeter,
        observation.vr_frame,
        ctx.prev_hand_qpos,
        hand_available=ctx.hand_available,
        retarget_cache=ctx.hand_retarget_cache,
        ramp_start_qpos_rad=ctx.hand_ramp_start,
        ramp_step=ctx.hand_ramp_step,
        ramp_total_frames=resources.hand_ramp_total_frames,
        command_lower_rad=command_limits.hand_command_lower_rad,
        command_upper_rad=command_limits.hand_command_upper_rad,
        mechanical_lower_rad=command_limits.hand_mechanical_lower_rad,
        mechanical_upper_rad=command_limits.hand_mechanical_upper_rad,
    )
    ctx.hand_ramp_start = hand.next_ramp_start_qpos_rad
    ctx.hand_ramp_step = hand.next_ramp_step
    if hand.validation_issue is not None:
        resources.validation_warn(
            "teleop_loop: invalid hand command — holding: %s",
            hand.validation_issue,
        )

    planner = resources.control.planner
    # The arm collision model must see the hand pose from this same observation.
    planner.set_hand_qpos(hand.qpos_rad)
    ik_started_s = time.perf_counter()
    ik_result = planner.solve_teleop_ik(
        Pose(p=target.position_world_m, q=target.quat_world_wxyz),
        observation.arm_qpos_rad,
        ctx.prev_qpos_cmd,
    )
    ik_solve_time_ms = (time.perf_counter() - ik_started_s) * 1000.0
    resources.stage_timer.mark("ik")
    return TeleopActionComputation(
        target_position_world_m=target.position_world_m,
        target_quat_world_wxyz=target.quat_world_wxyz,
        raw_target_position_world_m=target.raw_position_world_m,
        raw_target_quat_world_wxyz=target.raw_quat_world_wxyz,
        position_before_workspace_clamp_world_m=(
            target.position_before_workspace_clamp_world_m
        ),
        hand_qpos_rad=hand.qpos_rad,
        raw_hand_qpos_rad=hand.raw_qpos_rad,
        hand_retarget_succeeded=hand.retarget_succeeded,
        hand_validation_issue=hand.validation_issue,
        hand_retarget_time_ms=hand.compute_time_ms,
        ik_qpos_rad=ik_result.qpos if ik_result.success else None,
        ik_failure_reason=ik_result.reason,
        ik_solve_time_ms=ik_solve_time_ms,
        policy_map_time_ms=policy_map_time_ms,
        policy_compute_started_s=compute_started_s,
    )


def _publish_arm_safety_hold(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
    *,
    failure_context: str,
    frame_status: int,
    retarget_ok: bool,
) -> CoordinatorDirective:
    """Publish and record an arm-only hold after a rejected proposal."""
    assert ctx.prev_qpos_cmd is not None
    hold_result = _safe_joint_publish(
        shared,
        ctx.prev_qpos_cmd.copy(),
        None,
        is_hold=True,
        timeout=cfg.runtime.policy.action_prepare_timeout_s,
        observation_id=int(observation.vr_frame["ring_sequence"]),
        observation_anchor_monotonic_ns=int(observation.vr_frame["local_recv_ns"]),
        safety_gate=resources.control.safety_gate,
        hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
    )
    published_hold = hold_result.candidate
    if not hold_result.succeeded or published_hold is None:
        logger.error(
            "teleop_loop: %s hold publish failed: %s",
            failure_context,
            hold_result.reason,
        )
        shared.error_state.value = True
        return CoordinatorDirective.BREAK
    _record_grid_hold(
        ctx,
        shared,
        resources,
        observation,
        action_candidate=published_hold,
        frame_status=frame_status,
        retarget_ok=retarget_ok,
    )
    return CoordinatorDirective.CONTINUE


def _publish_ik_failure_hold(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
    computation: TeleopActionComputation,
) -> CoordinatorDirective:
    """Publish a bounded hold while preserving independent safe hand motion."""
    assert ctx.prev_qpos_cmd is not None
    assert ctx.prev_hand_qpos is not None
    if ctx.consecutive_ik_hold_frames == 0:
        ctx.ik_hold_started_s = time.monotonic()
        logger.warning(
            "teleop_loop: IK hold started: %s",
            computation.ik_failure_reason or "no feasible solution",
        )
    ctx.consecutive_ik_hold_frames += 1

    if ctx.hand_available:
        safe_hand_qpos = (
            computation.hand_qpos_rad
            if computation.hand_validation_issue is None
            else None
        )
        publish_result = _safe_joint_publish(
            shared,
            ctx.prev_qpos_cmd.copy(),
            safe_hand_qpos,
            is_hold=True,
            timeout=cfg.runtime.policy.action_prepare_timeout_s,
            observation_id=int(observation.vr_frame["ring_sequence"]),
            observation_anchor_monotonic_ns=int(observation.vr_frame["local_recv_ns"]),
            safety_gate=resources.control.safety_gate,
            hand_mechanical_lower_rad=(
                resources.command_limits.hand_mechanical_lower_rad
            ),
            hand_mechanical_upper_rad=(
                resources.command_limits.hand_mechanical_upper_rad
            ),
            hand_feedback_max_age_s=float(
                cfg.runtime.safety.heartbeat_timeouts["hand"]
            ),
        )
    else:
        publish_result = _safe_arm_queue_put(
            shared,
            {"qpos": ctx.prev_qpos_cmd.copy(), "is_hold": True},
            timeout=cfg.runtime.policy.action_prepare_timeout_s,
            observation_id=int(observation.vr_frame["ring_sequence"]),
            observation_anchor_monotonic_ns=int(observation.vr_frame["local_recv_ns"]),
            safety_gate=resources.control.safety_gate,
            hand_feedback_max_age_s=float(
                cfg.runtime.safety.heartbeat_timeouts["hand"]
            ),
        )

    published_candidate = publish_result.candidate
    if not publish_result.succeeded or published_candidate is None:
        logger.error(
            "teleop_loop: IK-failure hold publish failed: %s",
            publish_result.reason,
        )
        shared.error_state.value = True
        return CoordinatorDirective.BREAK
    if ctx.hand_available:
        if published_candidate.arm_qpos is not None:
            ctx.prev_qpos_cmd = np.asarray(
                published_candidate.arm_qpos, dtype=np.float64
            )
        if published_candidate.hand_qpos is not None:
            ctx.prev_hand_qpos = np.asarray(
                published_candidate.hand_qpos, dtype=np.float64
            ).copy()

    arm_names = observation.arm_state.dtype.names or ()
    diagnostics = {
        "tracking_error": (
            float(observation.arm_state["tracking_err"][0])
            if "tracking_err" in arm_names
            else 0.0
        ),
        "ik_solve_time_ms": computation.ik_solve_time_ms,
        "target_pos_before_clamp": (
            computation.position_before_workspace_clamp_world_m.copy()
        ),
        "head_quat_wxyz": np.asarray(
            observation.vr_frame.get("head_quat_wxyz", np.full(4, np.nan)),
            dtype=np.float64,
        ),
        "target_eef_pos_raw": computation.raw_target_position_world_m.copy(),
        "target_eef_rot6d_raw": quat_wxyz_to_rot6d(
            normalize_quat_wxyz(computation.raw_target_quat_world_wxyz)
        ),
        "action_hand_joint_raw": computation.raw_hand_qpos_rad.copy(),
        "policy_map_time_ms": computation.policy_map_time_ms,
        "hand_retarget_time_ms": computation.hand_retarget_time_ms,
        "transition_check_time_ms": 0.0,
        "policy_compute_time_ms": (
            time.perf_counter() - computation.policy_compute_started_s
        )
        * 1000.0,
    }
    _record_grid_hold(
        ctx,
        shared,
        resources,
        observation,
        action_candidate=published_candidate,
        frame_status=_FRAME_IK_FAIL,
        retarget_ok=computation.hand_retarget_succeeded,
        diagnostics=diagnostics,
    )
    return CoordinatorDirective.CONTINUE


def _publish_solved_action(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    observation: TeleopGridObservation,
    computation: TeleopActionComputation,
) -> CoordinatorDirective:
    """Validate, publish, and record one successful IK solution."""
    assert ctx.prev_qpos_cmd is not None
    assert ctx.prev_hand_qpos is not None
    assert computation.ik_qpos_rad is not None
    planner = resources.control.planner
    gate = resources.control.safety_gate
    recorder = resources.control.recorder
    command_limits = resources.command_limits
    stage_timer = resources.stage_timer
    _hand_fk = resources.hand_fk
    _T_eef_handbase_pos = resources.handbase_position_eef_m
    _T_eef_handbase_quat_wxyz = resources.handbase_quat_eef_wxyz
    _current_grid_anchor_ns = observation.anchor_monotonic_ns
    arm_state = observation.arm_state
    vr_frame = observation.vr_frame
    cam = observation.camera_frame
    hand_state = observation.hand_state
    hand_tactile = observation.hand_tactile
    target_pos = computation.target_position_world_m
    target_quat = computation.target_quat_world_wxyz
    target_pos_raw = computation.raw_target_position_world_m
    target_quat_raw = computation.raw_target_quat_world_wxyz
    target_pos_before_clamp = computation.position_before_workspace_clamp_world_m
    hand_cmd = computation.hand_qpos_rad
    hand_cmd_raw = computation.raw_hand_qpos_rad
    retarget_ok = computation.hand_retarget_succeeded
    hand_cmd_valid = computation.hand_validation_issue is None
    hand_retarget_time_ms = computation.hand_retarget_time_ms
    ik_solve_time_ms = computation.ik_solve_time_ms
    policy_map_time_ms = computation.policy_map_time_ms
    _policy_compute_t0 = computation.policy_compute_started_s

    if ctx.consecutive_ik_hold_frames:
        logger.info(
            "teleop_loop: IK recovered after %d frames (%.3fs)",
            ctx.consecutive_ik_hold_frames,
            time.monotonic() - ctx.ik_hold_started_s,
        )
        ctx.consecutive_ik_hold_frames = 0
        ctx.ik_hold_started_s = 0.0

    arm_proposal = compute_arm_joint_proposal(
        computation.ik_qpos_rad,
        ctx.prev_qpos_cmd,
        joint_lower_rad=command_limits.arm_joint_lower_rad,
        joint_upper_rad=command_limits.arm_joint_upper_rad,
        max_delta_rad_per_tick=(command_limits.arm_max_delta_rad_per_tick),
        compute_qpos_delta=planner.compute_qpos_delta,
    )
    arm_cmd = arm_proposal.qpos_rad
    arm_cmd_raw = arm_proposal.raw_qpos_rad

    reject_reason = arm_proposal.validation_issue
    if reject_reason is None and not hand_cmd_valid:
        reject_reason = "hand command validation failed"
    if reject_reason is not None:
        resources.validation_warn(
            "teleop_loop: action rejected — %s",
            reject_reason,
        )
        return _publish_arm_safety_hold(
            ctx,
            shared,
            cfg,
            resources,
            observation,
            failure_context="rejected-action",
            frame_status=_FRAME_SAFETY_REJECT,
            retarget_ok=computation.hand_retarget_succeeded,
        )

    publish_result = _safe_joint_publish(
        shared,
        arm_cmd.copy(),
        hand_cmd.copy() if ctx.hand_available else None,
        timeout=cfg.runtime.policy.action_prepare_timeout_s,
        observation_id=int(vr_frame["ring_sequence"]),
        observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
        safety_gate=gate,
        hand_mechanical_lower_rad=command_limits.hand_mechanical_lower_rad,
        hand_mechanical_upper_rad=command_limits.hand_mechanical_upper_rad,
        hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
    )
    published_candidate = publish_result.candidate
    workspace_rejected = (
        publish_result.status == CommandPublishStatus.GATE_REJECTED
        and publish_result.gate_code
        in (GateRejectCode.WORKSPACE, GateRejectCode.WORKSPACE_CHECK_FAILED)
    )
    if workspace_rejected:
        resources.validation_warn(
            "teleop_loop: action rejected — %s; publishing hold",
            publish_result.reason,
        )
        return _publish_arm_safety_hold(
            ctx,
            shared,
            cfg,
            resources,
            observation,
            failure_context="workspace-rejection",
            frame_status=_FRAME_SAFETY_REJECT,
            retarget_ok=computation.hand_retarget_succeeded,
        )
    if not publish_result.succeeded or published_candidate is None:
        # Recoverable holds keep arm and hand in place without latching a fault.
        hold_status = publish_result.status in (
            CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY,
            CommandPublishStatus.HAND_FEEDBACK_UNAVAILABLE,
        )
        if publish_result.runtime_gated:
            logger.info(
                "teleop_loop: joint publication stopped by runtime gate: %s",
                publish_result.reason,
            )
            if publish_result.status != CommandPublishStatus.SAFETY_STATE_GATED:
                return CoordinatorDirective.BREAK
            hold_status = True
        if not hold_status:
            logger.error("teleop_loop: joint publish failed: %s", publish_result.reason)
            shared.error_state.value = True
            return CoordinatorDirective.BREAK
        _record_grid_hold(ctx, shared, resources, observation)
        return CoordinatorDirective.CONTINUE
    stage_timer.mark("send")

    if published_candidate.arm_qpos is not None:
        arm_cmd = np.asarray(published_candidate.arm_qpos, dtype=np.float64)
    if published_candidate.hand_qpos is not None:
        hand_cmd = np.asarray(published_candidate.hand_qpos, dtype=np.float64)
    ctx.prev_qpos_cmd = arm_cmd.copy()
    ctx.prev_hand_qpos = hand_cmd.copy()
    ctx.ema_prev_pos = target_pos.copy()
    ctx.ema_prev_quat = target_quat.copy()

    if ctx.recording_active:
        policy_compute_time_ms = (time.perf_counter() - _policy_compute_t0) * 1000.0
        ctx.last_target_eef_pos = target_pos.copy()
        ctx.last_target_eef_rot6d = quat_wxyz_to_rot6d(normalize_quat_wxyz(target_quat))
        if not retarget_ok and ctx.hand_available:
            _f_status = _FRAME_RETARGET_FAIL
        else:
            _f_status = _FRAME_OK
        _record_frame(
            recorder,
            arm_state,
            hand_state,
            arm_cmd,
            hand_cmd,
            target_pos,
            target_quat,
            vr_frame,
            cam,
            ik_solve_time_ms,
            target_pos_before_clamp,
            hand_tactile,
            retarget_ok=retarget_ok,
            frame_status=_f_status,
            target_eef_pos_raw=target_pos_raw,
            target_eef_rot6d_raw=quat_wxyz_to_rot6d(
                normalize_quat_wxyz(target_quat_raw)
            ),
            action_hand_joint_raw=hand_cmd_raw,
            action_arm_joint_raw=arm_cmd_raw,
            policy_map_time_ms=policy_map_time_ms,
            hand_retarget_time_ms=hand_retarget_time_ms,
            policy_compute_time_ms=policy_compute_time_ms,
            hand_fk=_hand_fk,
            T_eef_handbase_pos=_T_eef_handbase_pos,
            T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
            observation_anchor_monotonic_ns=_current_grid_anchor_ns,
            shared=shared,
            action_candidate=published_candidate,
        )
    stage_timer.mark("rec")

    return CoordinatorDirective.NORMAL


def _run_control_grid_tick(
    ctx: TeleopLoopState,
    shared: SharedStorage,
    cfg: TeleopConfig,
    resources: TeleopGridResources,
    *,
    loop_count: int,
    observation_anchor_monotonic_ns: int,
) -> CoordinatorDirective:
    """Consume one causal observation and publish at most one action."""
    assert ctx.prev_qpos_cmd is not None
    gate = resources.control.safety_gate
    observation_directive, observation = _read_control_grid_observation(
        ctx,
        shared,
        cfg,
        resources,
        loop_count=loop_count,
        observation_anchor_monotonic_ns=observation_anchor_monotonic_ns,
    )
    if observation_directive is not CoordinatorDirective.NORMAL:
        return observation_directive
    assert observation is not None
    vr_frame = observation.vr_frame
    computation = _compute_action_computation(ctx, cfg, resources, observation)
    if computation is None:
        hold_result = _safe_arm_queue_put(
            shared,
            {"qpos": ctx.prev_qpos_cmd.copy(), "is_hold": True},
            timeout=cfg.runtime.policy.action_prepare_timeout_s,
            observation_id=int(vr_frame["ring_sequence"]),
            observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
            safety_gate=gate,
            hand_feedback_max_age_s=float(
                cfg.runtime.safety.heartbeat_timeouts["hand"]
            ),
        )
        published_hold = hold_result.candidate
        if not hold_result.succeeded or published_hold is None:
            logger.error(
                "teleop_loop: mapper hold publish failed: %s",
                hold_result.reason,
            )
            shared.error_state.value = True
            return CoordinatorDirective.BREAK
        _record_grid_hold(
            ctx,
            shared,
            resources,
            observation,
            action_candidate=published_hold,
        )
        return CoordinatorDirective.CONTINUE

    if computation.ik_qpos_rad is None:
        return _publish_ik_failure_hold(
            ctx,
            shared,
            cfg,
            resources,
            observation,
            computation,
        )

    return _publish_solved_action(
        ctx,
        shared,
        cfg,
        resources,
        observation,
        computation,
    )


def teleop_loop(shared: SharedStorage, config: TeleopConfig | None = None) -> None:
    """Teleoperation process entry point used by ``collect_teleop.py``.

    Reads from rings (vr, arm_state, hand_state, camera), writes actions
    to the actuator rings (arm_cmd_ring, hand_cmd_ring), owns recording.
    """
    cfg = config or TeleopConfig()
    ctx = TeleopLoopState()

    logger.debug("teleop_loop: LOADING")
    command_limits = TeleopCommandLimits.from_config(cfg)
    recording_enabled = bool(cfg.runtime.policy.recording_enabled)

    try:
        resources = _load_control_resources(
            shared,
            cfg,
            recording_enabled=recording_enabled,
        )
    except Exception:
        logger.error("teleop_loop: init failed", exc_info=True)
        shared.error_state.value = True
        return
    planner = resources.planner
    arm_mapper = resources.arm_mapper
    recorder = resources.recorder

    kb = _start_keyboard(shared)
    if kb is None:
        return

    def _keyboard_estop_requested() -> bool:
        return kb.estop_latched or not kb.healthy

    audio = AudioFeedback()

    ctx.hand_available = False
    ctx.hand_disconnected_at_s = None  # monotonic timestamp of first bad frame
    _hand_ramp_total_frames = _hand_ramp_frame_count(
        cfg.runtime.policy.hand_ramp_duration_s, cfg.runtime.policy.control_hz
    )

    def _try_init_hand_retargeter() -> bool:
        return _try_init_hand_retargeter_impl(ctx, cfg)

    def _hand_feedback_issue(state: np.ndarray | None) -> str | None:
        return _hand_feedback_issue_impl(cfg, state)

    _hand_fk = _try_load_hand_kinematics(
        cfg,
        recording_enabled=recording_enabled,
    )
    _T_eef_handbase_pos = np.array(
        cfg.runtime.hand.T_eef_handbase_pos_xyz, dtype=np.float64
    )
    _T_eef_handbase_quat_wxyz = np.array(
        cfg.runtime.hand.T_eef_handbase_quat_wxyz, dtype=np.float64
    )
    logger.info("Teleop: waiting for enabled capabilities...")
    readiness_timeout = _wait_for_enabled_capabilities(
        shared,
        cfg,
        recording_enabled=recording_enabled,
    )
    if readiness_timeout is not None:
        capability_name, timeout_s = readiness_timeout
        logger.error("Teleop: %s startup timeout (%.1fs)", capability_name, timeout_s)
        shared.error_state.value = True
        kb.stop()
        return
    logger.info("Teleop: all subsystems ready")

    # hand_loop publishes its initial state before setting hand_ready.
    if not cfg.runtime.policy.hand_enabled:
        ctx.hand_available = False
        logger.info(
            "Hand explicitly disabled — using the configured fixed-home collision assumption"
        )
    else:
        _init_hand_state = read_hand_state_causal(shared)
        initial_hand_issue = _hand_feedback_issue(_init_hand_state)
        if initial_hand_issue is not None:
            logger.error(
                "Teleop: initial hand feedback rejected: %s", initial_hand_issue
            )
            shared.error_state.value = True
            kb.stop()
            return
        ctx.hand_available = True
        if not _try_init_hand_retargeter():
            logger.error("teleop_loop: hand retargeter initialization failed")
            shared.error_state.value = True
            kb.stop()
            return

    shared.set_heartbeat("policy", time.monotonic())
    shared.set_ready("policy")
    logger.debug("teleop_loop: READY")

    home_qpos = np.array(cfg.runtime.arm.home_qpos, dtype=np.float64)
    arm_state = read_arm_state_causal(shared)
    hand_state = read_hand_state_causal(shared)
    if arm_state is None:
        arm_qpos = home_qpos.copy()
    else:
        arm_qpos = arm_state["qpos"][0].copy()
    ctx.prev_qpos_cmd = arm_qpos.copy()
    ctx.prev_hand_qpos = (
        hand_state["qpos"][0].copy()
        if hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0]))
        else command_limits.hand_home_qpos_rad.copy()
    )
    planner.set_hand_qpos(ctx.prev_hand_qpos)  # sync hand pose for collision checks

    ctx.teleop_active = False
    ctx.recording_active = False
    ctx.begin_audio_gate_deadline_s = None
    ctx.ignore_begin_audio_until_silent = False
    _quiescence = CommandQuiescence()

    ctx.quit_pending = False
    ctx.quit_after_recording = False
    ctx.quit_recording_deadline_s = 0.0
    ctx.post_teleop_deadline_s = 0.0

    # Coordination runs at coordinator_hz; control and recording use control_hz.
    limiter = RateManager(cfg.runtime.policy.coordinator_hz, label="teleop")
    _grid_period_ns = int(round(_NS_PER_SECOND / cfg.runtime.policy.control_hz))
    _next_grid_ns = time.monotonic_ns() + _grid_period_ns
    _current_grid_anchor_ns = _next_grid_ns
    _pending_controls: list[ControlSignal] = []
    stage_timer = StageTimer(window=cfg.runtime.policy.status_print_interval)
    _validate_warn = ThrottledWarner(interval_s=_VALIDATION_WARN_INTERVAL_S)
    _arm_feedback_warn = ThrottledWarner(interval_s=_ARM_FEEDBACK_WARN_INTERVAL_S)
    loop_count = 0
    ctx.arm_feedback_error_count = 0
    ctx.consecutive_ik_hold_frames = 0
    ctx.ik_hold_started_s = 0.0
    ctx.last_target_eef_pos = np.full(
        3, np.nan
    )  # last valid IK target (held frame continuity)
    ctx.last_target_eef_rot6d = np.full(6, np.nan)
    _camera_freshness = CameraFreshnessTracker(
        max_age_s=cfg.runtime.camera.max_frame_age_s,
        abort_after_s=cfg.runtime.camera.recording_stall_abort_s,
    )
    operator_resources = TeleopOperatorResources(
        control=resources,
        keyboard=kb,
        audio=audio,
        limiter=limiter,
        quiescence=_quiescence,
        camera_freshness=_camera_freshness,
    )
    grid_resources = TeleopGridResources(
        control=resources,
        command_limits=command_limits,
        quiescence=_quiescence,
        camera_freshness=_camera_freshness,
        stage_timer=stage_timer,
        validation_warn=_validate_warn,
        arm_feedback_warn=_arm_feedback_warn,
        hand_fk=_hand_fk,
        handbase_position_eef_m=_T_eef_handbase_pos,
        handbase_quat_eef_wxyz=_T_eef_handbase_quat_wxyz,
        hand_ramp_total_frames=_hand_ramp_total_frames,
        audio=audio,
    )

    def _transition_shared_or_fault(
        new_state: SafetyState,
        reason: str,
    ) -> bool:
        return _transition_or_fault(shared, new_state, reason)

    def _handoff_operator_quiescence_to_home() -> None:
        _handoff_quiescence_to_home(operator_resources)

    def _enter_command_quiescence(
        reason: str,
        *,
        start_new_run: bool = False,
        replace_existing_reason: bool = False,
    ) -> None:
        _enter_command_quiescence_impl(
            ctx,
            shared,
            _quiescence,
            arm_mapper,
            reason,
            start_new_run=start_new_run,
            replace_existing_reason=replace_existing_reason,
        )

    def _on_sigterm(signum: int, frame: object) -> None:
        ctx.sigterm_requested = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    logger.info(
        "Teleop: entering coordinator loop @ %.0f Hz (observation/action grid %.0f Hz)",
        cfg.runtime.policy.coordinator_hz,
        cfg.runtime.policy.control_hz,
    )

    try:
        while shared.is_running.value and not ctx.sigterm_requested:
            shared.set_heartbeat("policy", time.monotonic())
            limiter.wait()

            if not _poll_recording_lifecycle(
                ctx,
                shared,
                recorder,
                audio,
                enter_quiescence=_enter_command_quiescence,
                transition_or_fault=_transition_shared_or_fault,
            ):
                break

            post_teleop_directive = _advance_post_teleop_state(
                ctx,
                shared,
                cfg,
                kb,
                audio,
                planner,
                limiter,
                recorder,
                handoff_quiescence_to_home=_handoff_operator_quiescence_to_home,
                keyboard_estop_requested=_keyboard_estop_requested,
            )
            if post_teleop_directive is CoordinatorDirective.BREAK:
                break
            if post_teleop_directive is CoordinatorDirective.CONTINUE:
                continue
            _pending_controls.extend(kb.poll(timeout=0.0))
            _coordinator_now_ns = time.monotonic_ns()
            _grid_due = _coordinator_now_ns >= _next_grid_ns
            if _grid_due:
                # Skip missed deadlines rather than executing catch-up bursts;
                # every produced sample still has exactly one causal grid slot.
                _missed_periods = max(
                    1, (_coordinator_now_ns - _next_grid_ns) // _grid_period_ns + 1
                )
                _current_grid_anchor_ns = (
                    _next_grid_ns + int(_missed_periods - 1) * _grid_period_ns
                )
                _next_grid_ns += int(_missed_periods) * _grid_period_ns
                loop_count += 1
                stage_timer.tick()
                stage_timer.mark("coordinator")
            if not _grid_due and not _pending_controls:
                continue

            _controls = tuple(_pending_controls)
            _pending_controls.clear()
            operator_directive = _apply_operator_controls(
                ctx,
                shared,
                cfg,
                operator_resources,
                _controls,
            )
            if operator_directive is CoordinatorDirective.BREAK:
                break
            if operator_directive is CoordinatorDirective.CONTINUE:
                continue
            if not _grid_due:
                continue

            grid_directive = _run_control_grid_tick(
                ctx,
                shared,
                cfg,
                grid_resources,
                loop_count=loop_count,
                observation_anchor_monotonic_ns=_current_grid_anchor_ns,
            )
            if grid_directive is CoordinatorDirective.BREAK:
                break
            if grid_directive is CoordinatorDirective.CONTINUE:
                continue

    finally:
        if ctx.recording_active:
            _stop_recording(
                recorder, True, save=False, shared=shared, reason="policy_shutdown"
            )
        kb.stop()
        audio.play("end")
        if not audio.wait_until_idle(timeout_s=_END_AUDIO_GRACE_S):
            logger.warning("End audio did not finish within %.1fs", _END_AUDIO_GRACE_S)
        audio.close()
        exit_fault = _policy_exit_fault(
            error_state=bool(shared.error_state.value),
            estop_request=bool(shared.estop_request.value),
            safety_fault=shared.safety_state.value == SafetyState.FAULT,
        )
        if exit_fault is not None:
            logger.error("teleop_loop: %s", exit_fault)
        else:
            logger.debug("teleop_loop: STOPPED")
        logger.info("Teleop: loop exited")


def _safe_arm_queue_put(
    shared: SharedStorage,
    action,
    *,
    timeout: float,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    safety_gate: SafetyGate | None = None,
    hand_feedback_max_age_s: float,
) -> CommandPublishResult:
    """Publish a single arm command through the safety gate (fire-and-forget)."""
    try:
        return _safe_joint_publish(
            shared,
            np.asarray(action["qpos"], dtype=np.float64),
            None,
            is_hold=bool(action.get("is_hold", False)),
            timeout=timeout,
            observation_id=observation_id,
            observation_anchor_monotonic_ns=observation_anchor_monotonic_ns,
            safety_gate=safety_gate,
            hand_feedback_max_age_s=hand_feedback_max_age_s,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("teleop_loop: rejected invalid arm action: %s", exc)
        return CommandPublishResult(
            CommandPublishStatus.INVALID_CANDIDATE, detail=str(exc)
        )


def _safe_joint_publish(
    shared: SharedStorage,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None,
    *,
    is_hold: bool = False,
    timeout: float,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    safety_gate: SafetyGate | None = None,
    hand_mechanical_lower_rad: np.ndarray | None = None,
    hand_mechanical_upper_rad: np.ndarray | None = None,
    hand_feedback_max_age_s: float,
) -> CommandPublishResult:
    """Validate through SafetyGate and publish via fire-and-forget send_command."""
    if safety_gate is None:
        logger.error("joint target rejected: SafetyGate is required")
        return CommandPublishResult(CommandPublishStatus.NO_SAFETY_GATE)
    gate = safety_gate

    try:
        candidate = build_action_candidate(
            shared,
            arm_qpos,
            hand_qpos,
            is_hold=is_hold,
            observation_id=observation_id,
            observation_anchor_monotonic_ns=observation_anchor_monotonic_ns,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("joint target rejected: invalid candidate: %s", exc)
        return CommandPublishResult(
            CommandPublishStatus.INVALID_CANDIDATE, detail=str(exc)
        )
    if candidate is None:
        return CommandPublishResult(CommandPublishStatus.INVALID_OBSERVATION_ANCHOR)

    return validate_and_send_candidate(
        shared,
        candidate,
        gate=gate,
        hand_feedback_max_age_s=hand_feedback_max_age_s,
        prepare_timeout_s=timeout,
        hand_mechanical_lower_rad=hand_mechanical_lower_rad,
        hand_mechanical_upper_rad=hand_mechanical_upper_rad,
    )


def _print_status(
    loop_count: int,
    arm_state: np.ndarray | None,
    vr_frame: dict | None,
    teleop_active: bool,
    recording_active: bool,
    error_count: int,
    arm_q_depth: int = -1,
    arm_state_age_s: float = -1.0,
) -> None:
    """Periodic status print."""
    if arm_state is not None:
        try:
            _e, _ = make_arm_fk().compute(
                np.asarray(arm_state["qpos"][0], dtype=np.float64)
            )
            eef_str = f"eef={_e[0]:.3f},{_e[1]:.3f},{_e[2]:.3f}"
        except Exception:
            eef_str = "eef=?,?,?"
    else:
        eef_str = "eef=?,?,?"
    if vr_frame is not None:
        vr_age_ms = (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) / 1e6
        vr_str = f"vr={vr_age_ms:.0f}ms"
    else:
        vr_str = "vr=?ms"
    parts = [
        f"f={loop_count:>5d}",
        eef_str,
        f"T={'1' if teleop_active else '0'}",
        f"R={'1' if recording_active else '0'}",
        vr_str,
        f"err={error_count}",
    ]
    if arm_q_depth >= 0:
        parts.append(f"q={arm_q_depth}")
    if arm_state_age_s >= 0:
        parts.append(f"arm_age={arm_state_age_s:.2f}s")
    print("  ".join(parts), flush=True)
