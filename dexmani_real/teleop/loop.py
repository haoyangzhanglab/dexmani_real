"""Readable VR teleoperation experiment loop over shared-memory snapshots."""

from __future__ import annotations

import gc
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.planning import (PlanningProfile, Pose, TeleopProfile,
                                   XArm7MotionPlanner, XArm7PlannerConfig)
from dexmani_real.planning.hand_kinematics import HandKinematics
from dexmani_real.planning.pose_utils import (normalize_quat_wxyz,
                                              quat_wxyz_to_rot6d)
from dexmani_real.policy.loop_timing import StageTimer
from dexmani_real.policy.safety import (CommandPublishResult,
                                        CommandPublishStatus, GateRejectCode,
                                        SafetyGate, advance_run_generation,
                                        build_action_candidate,
                                        planner_action_safety_gate,
                                        validate_and_send_candidate)
from dexmani_real.recording.recorder_client import DirectRecorderClient, RecorderClient, RecorderPhase
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.audio_feedback import (AudioFeedback,
                                                update_motion_gate)
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_state import CommandQuiescence
from dexmani_real.teleop.episode_samples import (_FRAME_IK_FAIL, _FRAME_OK,
                                                 _FRAME_RETARGET_FAIL,
                                                 _FRAME_SAFETY_REJECT,
                                                 _record_frame, _record_held,
                                                 _stop_recording)
from dexmani_real.teleop.hand_control import (HandRetargetObservationCache,
                                              _compute_hand_command,
                                              _get_raw_hand_command,
                                              _hand_ramp_frame_count,
                                              _reset_hand_retargeter,
                                              _sanitize_hand_command,
                                              _seed_hand_retargeter,
                                              _smoothstep_hand_ramp)
from dexmani_real.teleop.hand_retarget import (TAGHandRetargeter,
                                               XHandRetargeter,
                                               _tag_config_with_urdf)
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.hand_health import (validate_arm_feedback,
                                            validate_hand_feedback)
from dexmani_real.teleop.recording_session import (
    QuitRecordingDecision, await_quit_recording_decision)
from dexmani_real.teleop.safety import (_do_configured_teleop_home,
                                        _reset_mapper_from_frames)
from dexmani_real.teleop.snapshot import (CameraFreshnessTracker,
                                          _read_arm_state, _read_camera_frame,
                                          _read_hand_state, _read_hand_tactile,
                                          _read_vr_frame)
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import ema_smooth_pose

logger = get_logger(__name__)

_END_AUDIO_GRACE_S = 2.0
_NS_PER_SECOND = 1_000_000_000
_VALIDATION_WARN_INTERVAL_S = 2.0
_ARM_FEEDBACK_WARN_INTERVAL_S = 3.0


def _load_vr_transform(path: Path) -> tuple[np.ndarray, str]:
    """Compatibility wrapper around the shared schema-v1 calibration loader."""
    calibration = load_vr_transform(path)
    return calibration.transform, f"{calibration.theta_deg:.6g}"


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
        eef_pos=np.asarray(state["eef_pos"][0]),
        eef_rot6d=np.asarray(state["eef_rot6d"][0]),
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
    keyboard = KeyboardHandler(estop_callback=lambda: setattr(shared.estop_request, "value", True))
    try:
        keyboard.start()
    except Exception:
        logger.error("teleop_loop: keyboard startup failed", exc_info=True)
        shared.error_state.value = True
        return None
    return keyboard


def _transition_or_fault_impl(
    shared: SharedStorage,
    transition: Any,
    new_state: Any,
    reason: str,
) -> bool:
    if transition(shared, new_state):
        return True
    logger.error("teleop_loop: safety transition to %s failed during %s", new_state.name, reason)
    shared.error_state.value = True
    return False


def _keyboard_estop_requested_impl(kb: KeyboardHandler) -> bool:
    return kb.estop_latched or not kb.healthy


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
    # Solver result/failure keyed by verified VR ring sequence. Unlike the
    # ramp, this state advances only on a new observation.
    hand_retarget_cache: HandRetargetObservationCache = field(
        default_factory=HandRetargetObservationCache
    )
    sigterm_requested: bool = False


def _try_init_hand_retargeter_impl(ctx: TeleopLoopState, cfg: TeleopConfig) -> bool:
    """Lazily initialize ctx.hand_retargeter if not already created."""
    if ctx.hand_retargeter is not None:
        return True
    try:
        if cfg.runtime.policy.hand_retargeting_type == "tag":
            ctx.hand_retargeter = TAGHandRetargeter(
                hand_type="right",
                fingertip_link_names=cfg.runtime.hand.fingertip_link_names,
                tag_config=_tag_config_with_urdf(cfg.runtime.tag_retargeting, cfg.hand_urdf_path),
            )
        else:
            ctx.hand_retargeter = XHandRetargeter(
                hand_type="right",
                retargeting_type=cfg.runtime.policy.hand_retargeting_type,
                dexpilot_config=cfg.runtime.dexpilot_retargeting,
            )
        logger.info("Hand retargeter ready (type=%s)", cfg.runtime.policy.hand_retargeting_type)
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
    hs = _read_hand_state(shared)
    qpos = hs["qpos"][0] if _hand_feedback_issue_impl(cfg, hs) is None and hs is not None else None
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


def _handoff_command_quiescence_to_home_impl(quiescence: CommandQuiescence) -> None:
    """Let the homing protocol supersede command quiescence."""
    reason, _entered_ns = quiescence.clear()
    if reason is not None:
        logger.info(
            "teleop_loop: homing supersedes %s command quiescence",
            reason,
        )


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
        validate_warn("teleop_loop: re-anchor inputs invalid — remaining command-silent")
        return False
    ctx.ema_prev_pos = ctx.ema_prev_quat = None
    hand_anchor: np.ndarray | None = None
    if hand_available:
        if current_hand_state is not None and np.all(np.isfinite(current_hand_state["qpos"][0])):
            ctx.prev_hand_qpos = np.asarray(current_hand_state["qpos"][0], dtype=np.float64).copy()
        if ctx.prev_hand_qpos is None:
            validate_warn("teleop_loop: hand re-anchor unavailable — remaining command-silent")
            return False
        hand_anchor = ctx.prev_hand_qpos.copy()
    ctx.hand_ramp_start = hand_anchor
    ctx.hand_ramp_step = 0
    # Keep the observation cursor and stateful backend on the same reset epoch.
    ctx.hand_retarget_cache.reset()
    _reset_hand_retargeter(ctx.hand_retargeter, hand_anchor)
    return True


def teleop_loop(shared: SharedStorage, config: TeleopConfig | None = None) -> None:
    """Teleoperation process entry point used by ``collect_teleop.py``.

    Reads from rings (vr, arm_state, hand_state, camera), writes actions
    to queues/rings (arm_action_q, hand_cmd_ring), owns recording.
    """
    from dexmani_real.robot.safety import SafetyState, transition

    cfg = config or TeleopConfig()
    ctx = TeleopLoopState()

    def _transition_or_fault(new_state: SafetyState, reason: str) -> bool:
        return _transition_or_fault_impl(shared, transition, new_state, reason)

    logger.debug("teleop_loop: LOADING")
    ctrl_dt = 1.0 / cfg.runtime.policy.control_hz
    joint_lower_rad = np.asarray(cfg.runtime.arm.joint_limit_lower, dtype=np.float64)
    joint_upper_rad = np.asarray(cfg.runtime.arm.joint_limit_upper, dtype=np.float64)
    arm_max_delta_rad_per_tick: np.ndarray | float | None = cfg.runtime.policy.arm_max_delta_rad_per_tick
    recording_enabled = bool(cfg.runtime.policy.recording_enabled)
    recording_mode = cfg.runtime.policy.recording_mode
    # The simplified direct backend keeps the complete v17 data contract.  A
    # recording session therefore always starts the RGB-D path; mechanism
    # simplification must not silently remove modalities or metadata.
    camera_recording_enabled = recording_enabled
    if arm_max_delta_rad_per_tick is not None:
        arm_max_delta_rad_per_tick = np.broadcast_to(
            np.asarray(arm_max_delta_rad_per_tick, dtype=np.float64), joint_lower_rad.shape
        )
    hand_home_qpos_rad = np.deg2rad(np.asarray(cfg.runtime.hand.home_qpos_deg, dtype=np.float64))
    hand_qpos_lower_rad = np.asarray(cfg.runtime.hand.qpos_min_rad, dtype=np.float64)
    hand_qpos_upper_rad = np.asarray(cfg.runtime.hand.qpos_max_rad, dtype=np.float64)
    hand_mechanical_lower_rad = np.asarray(cfg.runtime.hand.mechanical_qpos_min_rad, dtype=np.float64)
    hand_mechanical_upper_rad = np.asarray(cfg.runtime.hand.mechanical_qpos_max_rad, dtype=np.float64)

    try:
        urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
        srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf")
        planner = XArm7MotionPlanner(
            XArm7PlannerConfig(
                urdf_path=urdf_path,
                srdf_path=srdf_path,
                base_pose_world=Pose(
                    p=np.array([0.0, 0.0, 0.0]),
                    q=np.array([1.0, 0.0, 0.0, 0.0]),
                ),
                workspace_bounds=np.asarray(cfg.runtime.policy.workspace.as_tuple(), dtype=np.float64),
            ),
            planning_profile=PlanningProfile(),
            teleop_profile=TeleopProfile(
                max_pose_error_pos_m=cfg.runtime.policy.ik_max_pose_error_pos_m,
                max_pose_error_rot_rad=cfg.runtime.policy.ik_max_pose_error_rot_rad,
                nullspace_step_size_deg=cfg.runtime.policy.ik_nullspace_step_rate_deg_s / cfg.runtime.policy.control_hz,
            ),
            hand_dof=True,  # 19-DOF — hand geometry follows set_hand_qpos()
            static_boxes=cfg.runtime.environment.static_boxes,
            table=cfg.runtime.environment.table,
        )

        vr_config_path = Path(__file__).resolve().parents[2] / cfg.vr_transform_path
        vr_to_robot, vr_heading_deg = _load_vr_transform(vr_config_path)
        logger.info("VR transform loaded: theta=%s°", vr_heading_deg)

        arm_mapper = ArmWristMapper(
            pos_scale=cfg.runtime.policy.vr_mapping.pos_scale,
            rot_scale=cfg.runtime.policy.vr_mapping.rot_scale,
            vr_to_base_rot=vr_to_robot,
            T_vr_to_robot=vr_to_robot,
            max_delta_rot_rad=cfg.runtime.policy.vr_mapping.max_delta_rot_rad,
            base_to_world_rot=np.eye(3, dtype=np.float64),
        )

        gate = _build_safety_gate(cfg, planner)
        recorder: Any
        if not recording_enabled:
            recorder = None
        elif recording_mode == "v17":
            recorder = RecorderClient(shared)
        else:
            recorder = DirectRecorderClient(
                shared,
                data_dir=str(Path(__file__).resolve().parents[2] / cfg.runtime.policy.episodes_dir),
                max_frames=int(round(cfg.runtime.policy.max_record_duration_s * cfg.runtime.policy.control_hz)),
                control_hz=cfg.runtime.policy.control_hz,
                min_frames=int(round(cfg.runtime.policy.min_record_duration_s * cfg.runtime.policy.control_hz)),
                resolved_config_sha256=cfg.runtime.sha256,
                align_mode=cfg.runtime.camera.align_mode,
                provenance=dict(cfg.recording_provenance),
                rgb_shape=tuple(cfg.runtime.camera.rgb_shape),
                depth_shape=tuple(cfg.runtime.camera.depth_shape),
                writer_queue_size=int(cfg.runtime.camera.writer_queue_size),
            )
    except Exception:
        logger.error("teleop_loop: init failed", exc_info=True)
        shared.error_state.value = True
        return

    kb = _start_keyboard(shared)
    if kb is None:
        return

    def _keyboard_estop_requested() -> bool:
        return _keyboard_estop_requested_impl(kb)

    audio = AudioFeedback()

    hand_available = False
    _hand_disconnected_at: float | None = None  # monotonic timestamp of first bad frame
    _hand_ramp_total_frames = _hand_ramp_frame_count(cfg.runtime.policy.hand_ramp_duration_s, cfg.runtime.policy.control_hz)

    def _try_init_hand_retargeter() -> bool:
        return _try_init_hand_retargeter_impl(ctx, cfg)

    def _init_and_seed_hand_retargeter() -> np.ndarray | None:
        return _init_and_seed_hand_retargeter_impl(ctx, cfg, shared)

    def _hand_feedback_issue(state: np.ndarray | None) -> str | None:
        return _hand_feedback_issue_impl(cfg, state)

    _hand_fk: HandKinematics | None = None
    _T_eef_handbase_pos = np.array(cfg.runtime.hand.T_eef_handbase_pos_xyz, dtype=np.float64)
    _T_eef_handbase_quat_wxyz = np.array(cfg.runtime.hand.T_eef_handbase_quat_wxyz, dtype=np.float64)
    if recording_enabled and cfg.hand_urdf_path:
        try:
            _hand_fk = HandKinematics(cfg.hand_urdf_path, list(cfg.runtime.hand.fingertip_link_names))
            if _hand_fk.is_ready():
                logger.info("Hand FK ready")
            else:
                logger.warning("Hand FK not ready — fingertips will be NaN")
        except Exception:
            logger.warning("Hand FK initialization failed", exc_info=True)

    logger.info("Teleop: waiting for enabled capabilities...")
    _ready_names = ["arm", "vr"]
    if camera_recording_enabled:
        _ready_names += ["camera"]
    if recording_mode == "v17" and recording_enabled:
        _ready_names += ["recorder"]
    if cfg.runtime.policy.hand_enabled:
        _ready_names.insert(1, "hand")
    for name in _ready_names:
        timeout_s = float(dict(cfg.runtime.safety.readiness_timeouts_s)[name])
        if not shared.wait_ready(name, timeout_s):
            logger.error("Teleop: %s startup timeout (%.1fs)", name, timeout_s)
            shared.error_state.value = True
            kb.stop()
            return
    logger.info("Teleop: all subsystems ready")

    # hand_loop publishes its initial state before setting hand_ready.
    if not cfg.runtime.policy.hand_enabled:
        hand_available = False
        logger.info("Hand explicitly disabled — using the configured fixed-home collision assumption")
    else:
        _init_hand_state = _read_hand_state(shared)
        initial_hand_issue = _hand_feedback_issue(_init_hand_state)
        if initial_hand_issue is not None:
            logger.error("Teleop: initial hand feedback rejected: %s", initial_hand_issue)
            shared.error_state.value = True
            kb.stop()
            return
        hand_available = True
        if not _try_init_hand_retargeter():
            logger.error("teleop_loop: hand retargeter initialization failed")
            shared.error_state.value = True
            kb.stop()
            return

    # Publish before policy_ready so Main never observes a ready worker with a zero heartbeat.
    shared.set_heartbeat("policy", time.monotonic())
    shared.set_ready("policy")
    logger.debug("teleop_loop: READY")

    home_qpos = np.array(cfg.runtime.arm.home_qpos, dtype=np.float64)
    arm_state = _read_arm_state(shared)
    hand_state = _read_hand_state(shared)
    if arm_state is None:
        arm_qpos = home_qpos.copy()
    else:
        arm_qpos = arm_state["qpos"][0].copy()
    ctx.prev_qpos_cmd = arm_qpos.copy()
    ctx.prev_hand_qpos = (
        hand_state["qpos"][0].copy()
        if hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0]))
        else hand_home_qpos_rad.copy()
    )
    planner.set_hand_qpos(ctx.prev_hand_qpos)  # sync hand pose for collision checks

    teleop_active = False
    recording_active = False
    _begin_audio_gate_deadline_s: float | None = None
    _ignore_begin_audio_until_silent = False
    _quiescence = CommandQuiescence()

    # First Q leaves teleop active for an optional home; the next Q exits.
    quit_pending = False
    quit_after_recording = False
    quit_recording_deadline_s = 0.0
    post_teleop_deadline = 0.0


    # Coordinator duties run at coordinator_hz; observations, actions, and
    # recording remain on the configured control_hz grid.
    limiter = RateManager(cfg.runtime.policy.coordinator_hz)
    _grid_period_ns = int(round(_NS_PER_SECOND / cfg.runtime.policy.control_hz))
    _next_grid_ns = time.monotonic_ns() + _grid_period_ns
    _current_grid_anchor_ns = _next_grid_ns
    _pending_controls: list[ControlSignal] = []
    stage_timer = StageTimer(window=cfg.runtime.policy.status_print_interval)
    _validate_warn = ThrottledWarner(interval_s=_VALIDATION_WARN_INTERVAL_S)
    _arm_feedback_warn = ThrottledWarner(interval_s=_ARM_FEEDBACK_WARN_INTERVAL_S)
    loop_count = 0
    error_count = 0
    _last_target_eef_pos = np.full(3, np.nan)  # last valid IK target (held frame continuity)
    _last_target_eef_rot6d = np.full(6, np.nan)
    _camera_freshness = CameraFreshnessTracker(
        max_age_s=cfg.runtime.camera.max_frame_age_s,
        abort_after_s=cfg.runtime.camera.recording_stall_abort_s,
    )

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

    def _handoff_command_quiescence_to_home() -> None:
        _handoff_command_quiescence_to_home_impl(_quiescence)

    def _complete_reanchor(
        current_arm_state: np.ndarray,
        current_vr_frame: dict[str, Any],
        current_hand_state: np.ndarray | None,
    ) -> bool:
        return _complete_reanchor_impl(ctx, arm_mapper, _validate_warn, hand_available, current_arm_state, current_vr_frame, current_hand_state)

    # SIGTERM is intercepted so the active direct/v17 recorder can finish its
    # episode transaction before the policy process exits.
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

            if recorder is not None:
                _stop_result = recorder.poll_stop()
                if (
                    _stop_result.phase in (RecorderPhase.FINALIZING, RecorderPhase.COMPLETED, RecorderPhase.ERROR)
                    and _stop_result.reason == "max_frames"
                    and (teleop_active or recording_active)
                ):
                    _enter_command_quiescence(
                        "max_frames",
                        replace_existing_reason=True,
                    )
                    teleop_active = False
                    recording_active = False
                    shared.is_recording.value = False
                    if not _transition_or_fault(SafetyState.ARMED, "maximum recording duration"):
                        break
                    print("  已达到最大录制时长：正在自动保存，遥操作进入静默暂停")
                    audio.play("pause")
                if _stop_result.done:
                    recording_active = False
                    shared.is_recording.value = False
                    if _stop_result.error:
                        path_label = f": {_stop_result.path}" if _stop_result.path else ""
                        print(f"  ⚠ 录制终结失败 ({_stop_result.error}){path_label}")
                    elif _stop_result.saved:
                        print(f"  录制已保存: {_stop_result.path}  ({_stop_result.frame_count} 帧)")
                        if not _stop_result.min_frames_met:
                            print("  ⚠ 已保存，但未达到配置的最短质量时长")
                    else:
                        print(f"  录制已丢弃 ({_stop_result.frame_count} 帧)")
                    gc.collect()
                    if quit_after_recording:
                        shared.quit_requested.value = True
                elif (
                    _stop_result.phase is RecorderPhase.FINALIZING
                    and _stop_result.error
                ):
                    print(
                        "  ⚠ 录制终结超过时限；仍在安全回收，"
                        "本会话将标记为失败"
                    )

            if recorder is not None and recording_active and recorder.camera_writer_error is not None:
                _writer_error = recorder.camera_writer_error
                logger.error("Camera writer failed — discarding current episode: %s", _writer_error)
                print(f"  ⚠ 相机写盘失败，当前 episode 已废弃: {_writer_error}")
                _stop_recording(
                    recorder,
                    recording_active,
                    save=False,
                    shared=shared,
                    reason="camera_writer_error",
                )
                recording_active = False

            # Entered after Q key stops teleop. The teleop loop stays alive with
            # heartbeats ticking so arm/hand/vr continue running — H (return_home)
            # can still queue HOME_SENTINEL and wait for convergence.
            if quit_pending:
                # Show prompt once per entry; re-shown after H completes.
                for _sig in kb.poll(timeout=0.1):
                    if _sig == ControlSignal.HOME:
                        print("  H: return_home")
                        _handoff_command_quiescence_to_home()
                        ctx.prev_hand_qpos = _do_configured_teleop_home(
                            shared,
                            cfg,
                            hand_available=hand_available,
                            prev_hand_qpos=ctx.prev_hand_qpos,
                            planner=planner,
                            audio=audio,
                            estop_requested=_keyboard_estop_requested,
                        )
                        limiter.reset()
                        print("  [Q] quit", flush=True)

                    elif _sig in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
                        if _sig == ControlSignal.EMERGENCY_STOP:
                            shared.estop_request.value = True
                        else:
                            if recorder is not None and recorder.stop_pending:
                                if not quit_after_recording:
                                    quit_after_recording = True
                                    quit_recording_deadline_s = (
                                        time.monotonic() + cfg.runtime.policy.quit_save_timeout_s
                                    )
                                print("  录制仍在终结；完成后自动退出", flush=True)
                            else:
                                shared.quit_requested.value = True
                                break

                if shared.estop_request.value or shared.quit_requested.value or not shared.is_running.value:
                    break

                if (
                    quit_after_recording
                    and recorder is not None
                    and recorder.stop_pending
                    and time.monotonic() >= quit_recording_deadline_s
                ):
                    print("  录制终结超时 — 退出并将本会话标记为失败")
                    shared.quit_requested.value = True
                    break

                if time.perf_counter() > post_teleop_deadline:
                    if recorder is not None and recorder.stop_pending:
                        if not quit_after_recording:
                            quit_after_recording = True
                            quit_recording_deadline_s = (
                                time.monotonic() + cfg.runtime.policy.quit_save_timeout_s
                            )
                            print("  timeout — 等待录制终结后自动退出", flush=True)
                        elif time.monotonic() >= quit_recording_deadline_s:
                            print("  录制终结超时 — 退出并将本会话标记为失败")
                            shared.quit_requested.value = True
                            break
                    else:
                        print("  timeout — auto exit")
                        shared.quit_requested.value = True
                        break

                continue  # stay in quit_pending, don't process normal teleop

            _pending_controls.extend(kb.poll(timeout=0.0))
            _coordinator_now_ns = time.monotonic_ns()
            _grid_due = _coordinator_now_ns >= _next_grid_ns
            if _grid_due:
                # Skip missed deadlines rather than executing catch-up bursts;
                # every produced sample still has exactly one causal grid slot.
                _missed_periods = max(1, (_coordinator_now_ns - _next_grid_ns) // _grid_period_ns + 1)
                _current_grid_anchor_ns = _next_grid_ns + int(_missed_periods - 1) * _grid_period_ns
                _next_grid_ns += int(_missed_periods) * _grid_period_ns
                loop_count += 1
                stage_timer.tick()
                stage_timer.mark("coordinator")
            if not _grid_due and not _pending_controls:
                continue

            _controls = tuple(_pending_controls)
            _pending_controls.clear()
            skip_rest = False
            for sig in _controls:
                if sig == ControlSignal.EMERGENCY_STOP:
                    print("\nESC: emergency_stop")
                    audio.play("emergency")
                    shared.estop_request.value = True
                    _stop_recording(recorder, recording_active, save=False, shared=shared)
                    recording_active = False
                    break

                elif sig == ControlSignal.QUIT:
                    print("\nQ: 退出")
                    audio.play("quit")
                    _enter_command_quiescence(
                        "quit",
                        replace_existing_reason=True,
                    )
                    teleop_active = False
                    if not _transition_or_fault(SafetyState.ARMED, "quit"):
                        break

                    if recording_active:
                        audio.queue("quit_save_prompt")  # queue: plays after "quit" finishes
                        print(
                            "  [S] 保存并退出  [D] 丢弃并退出  [H] 保存并归位 "
                            f"({cfg.runtime.policy.quit_save_timeout_s:.0f}s 超时默认丢弃)"
                        )
                        decision = await_quit_recording_decision(shared, kb, timeout_s=cfg.runtime.policy.quit_save_timeout_s)
                        if decision in (QuitRecordingDecision.SAVE, QuitRecordingDecision.SAVE_AND_HOME):
                            audio.play("save")
                            _stop_recording(recorder, recording_active, save=True, shared=shared)
                            recording_active = False
                            print("  保存请求已提交")
                        elif decision is QuitRecordingDecision.DISCARD:
                            audio.play("discard")
                            _stop_recording(recorder, recording_active, save=False, shared=shared)
                            recording_active = False
                            print("  丢弃请求已提交")
                        elif decision is QuitRecordingDecision.ESTOP:
                            audio.play("emergency")
                            _stop_recording(recorder, recording_active, save=False, shared=shared)
                            recording_active = False
                        elif decision is QuitRecordingDecision.TIMEOUT:
                            audio.play("discard")
                            _stop_recording(recorder, recording_active, save=False, shared=shared)
                            recording_active = False
                            print("  超时，默认丢弃请求已提交")

                        if decision is QuitRecordingDecision.SAVE_AND_HOME and shared.is_running.value:
                            audio.play("home")
                            ctx.ema_prev_pos = ctx.ema_prev_quat = None
                            _handoff_command_quiescence_to_home()
                            ctx.prev_hand_qpos = _do_configured_teleop_home(
                                shared,
                                cfg,
                                hand_available=hand_available,
                                prev_hand_qpos=ctx.prev_hand_qpos,
                                planner=planner,
                                audio=audio,
                                estop_requested=_keyboard_estop_requested,
                                arm_mapper=arm_mapper,
                                hand_retargeter=ctx.hand_retargeter,
                            )

                    # Enter post-teleop state (two-stage Q) instead of immediate exit.
                    # Teleop stays alive with heartbeats ticking; arm/hand/vr
                    # continue running so H (return_home) still works.
                    quit_pending = True
                    post_teleop_deadline = time.perf_counter() + cfg.runtime.policy.post_teleop_timeout_s
                    print(
                        f"\n[H] return_home  [Q] quit  ({cfg.runtime.policy.post_teleop_timeout_s:.0f}s timeout)",
                        flush=True,
                    )
                    skip_rest = True
                    break  # break from for-sig loop, re-enter main loop in quit_pending state

                elif sig == ControlSignal.HOME:
                    print("\nH: return_home")
                    audio.play("home")
                    _stop_recording(recorder, recording_active, save=True, shared=shared)
                    recording_active = False
                    teleop_active = False
                    if not _transition_or_fault(SafetyState.ARMED, "home"):
                        break
                    ctx.ema_prev_pos = ctx.ema_prev_quat = None
                    _handoff_command_quiescence_to_home()
                    ctx.prev_hand_qpos = _do_configured_teleop_home(
                        shared,
                        cfg,
                        hand_available=hand_available,
                        prev_hand_qpos=ctx.prev_hand_qpos,
                        planner=planner,
                        audio=audio,
                        estop_requested=_keyboard_estop_requested,
                        arm_mapper=arm_mapper,
                        hand_retargeter=ctx.hand_retargeter,
                    )
                    limiter.reset()
                    skip_rest = True

                elif sig in (ControlSignal.STOP, ControlSignal.DISCARD):
                    save_episode = sig is ControlSignal.STOP
                    stop_reason = "stop" if save_episode else "discard"
                    print("\nS: 停止录制" if save_episode else "\nD: 丢弃录制")
                    audio.play("save" if save_episode else "discard")
                    _enter_command_quiescence(
                        stop_reason,
                        replace_existing_reason=True,
                    )
                    _stop_recording(recorder, recording_active, save=save_episode, shared=shared)
                    recording_active = False
                    teleop_active = False
                    if not _transition_or_fault(SafetyState.ARMED, stop_reason):
                        break
                    skip_rest = True

                elif sig == ControlSignal.PAUSE:
                    if teleop_active:
                        # C is an explicit operator pause. If an automatic gate
                        # is already silent, retain its freshness boundary but
                        # make this state resumable by the next C.
                        _enter_command_quiescence(
                            "pause",
                            replace_existing_reason=True,
                        )
                        teleop_active = False
                        if not _transition_or_fault(SafetyState.ARMED, "pause"):
                            break
                    else:
                        if not _quiescence.active or _quiescence.reason != "pause":
                            print("\nC: 没有可恢复的暂停 session — 请按 B 开始新的遥操作 session")
                            skip_rest = True
                            continue
                        if shared.safety_state.value != SafetyState.ARMED:
                            print(f"\nC: safety_state={shared.safety_state.value} — must be ARMED to resume")
                            skip_rest = True
                            continue
                        if not _transition_or_fault(SafetyState.RUNNING, "resume"):
                            break
                        teleop_active = True
                    state_str = "恢复" if teleop_active else "暂停"
                    print(f"\nC: {state_str}遥操作")
                    if teleop_active:
                        audio.play("resume")
                    else:
                        audio.play("pause")
                    skip_rest = True

                elif sig == ControlSignal.BEGIN:
                    if teleop_active or recording_active:
                        print("\nB: session already active — use C to pause/resume, S to save, or D to discard")
                        skip_rest = True
                        continue
                    # Only Main can arm the system; B only performs ARMED -> RUNNING.
                    if shared.safety_state.value != SafetyState.ARMED:
                        print(f"\nB: safety_state={shared.safety_state.value} — must be ARMED({SafetyState.ARMED})")
                        skip_rest = True
                        continue
                    vr_frame = _read_vr_frame(shared)
                    if vr_frame is None:
                        print("\nB: 无 VR 帧，无法开始遥操作")
                        skip_rest = True
                        continue
                    begin_hand_state = _read_hand_state(shared) if cfg.runtime.policy.hand_enabled else None
                    begin_hand_issue = _hand_feedback_issue(begin_hand_state)
                    if begin_hand_issue is not None:
                        print(f"\nB: hand feedback unhealthy ({begin_hand_issue}) — cannot begin")
                        skip_rest = True
                        continue
                    _stop_recording(recorder, recording_active, save=recording_active, shared=shared)
                    gc.collect()

                    if recorder is None:
                        recording_active = False
                        shared.is_recording.value = False
                        begin_reason = "begin"
                        begin_message = "\nB: 遥操作开始（未启用录制 capability）"
                    else:
                        if not recorder.start_episode(task_label=cfg.task_label, operator=cfg.operator):
                            print("  ⚠ 无法开始录制")
                            skip_rest = True
                            continue
                        recording_active = True
                        _camera_freshness.reset(time.monotonic())
                        shared.is_recording.value = True
                        begin_reason = "begin recording"
                        begin_message = f"\nB: 遥操作+录制开始  episode={recorder.frame_count}"

                    kb.drain_signal(ControlSignal.BEGIN)
                    if not _transition_or_fault(SafetyState.RUNNING, begin_reason):
                        _stop_recording(
                            recorder,
                            recording_active,
                            save=False,
                            shared=shared,
                            reason="safety_transition_failed",
                        )
                        recording_active = False
                        break
                    # BEGIN is a distinct run boundary even when a prior STOP,
                    # DISCARD, or max-duration stop already invalidated its own
                    # generation. Require feedback newer than this BEGIN before
                    # the one-grid re-anchor.
                    _enter_command_quiescence("begin", start_new_run=True)
                    teleop_active = True
                    logger.debug("teleop_loop: RUNNING")
                    _seeded = _init_and_seed_hand_retargeter()
                    if _seeded is not None:
                        ctx.prev_hand_qpos = _seeded
                    audio.play("begin")
                    _begin_audio_gate_deadline_s = time.monotonic() + max(
                        0.0, cfg.runtime.policy.begin_motion_gate_timeout_s - ctrl_dt
                    )
                    _ignore_begin_audio_until_silent = False
                    print(begin_message)
                    limiter.reset()
                    skip_rest = True

            if shared.estop_request.value or shared.quit_requested.value or not shared.is_running.value:
                break
            if shared.error_state.value:
                break
            if skip_rest:
                continue
            if not _grid_due:
                continue

            arm_state = _read_arm_state(shared, anchor_monotonic_ns=_current_grid_anchor_ns)
            arm_issue = _arm_feedback_issue(
                arm_state,
                now_monotonic_ns=time.monotonic_ns(),
                max_age_s=cfg.runtime.policy.arm_state_stale_threshold_s,
            )
            error_count, arm_feedback_fault = _advance_arm_feedback_error_count(
                error_count,
                arm_issue,
                max_consecutive_errors=cfg.runtime.policy.max_consecutive_errors,
            )
            if arm_issue is not None:
                _arm_feedback_warn(
                    "teleop_loop: invalid arm feedback (%d/%d): %s",
                    error_count,
                    cfg.runtime.policy.max_consecutive_errors,
                    arm_issue,
                )
                if teleop_active and not _quiescence.active:
                    _enter_command_quiescence("arm_feedback")
                if arm_feedback_fault:
                    logger.error("teleop_loop: arm feedback fault: %s", arm_issue)
                    shared.error_state.value = True
                    break
                continue
            assert arm_state is not None  # validation above proved availability
            arm_qpos = arm_state["qpos"][0].copy()

            vr_frame = _read_vr_frame(shared, anchor_monotonic_ns=_current_grid_anchor_ns)
            vr_stale = vr_frame is None or (
                (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) > cfg.runtime.policy.vr_mapping.stale_threshold_s * 1e9
            )
            stage_timer.mark("vr")

            # VR control does not consume camera pixels.  Scan/copy the large
            # payload only while the policy-owned recorder requests it.
            cam = _read_camera_frame(shared, anchor_monotonic_ns=_current_grid_anchor_ns) if recording_active else None
            if recording_active:
                cam, _camera_stalled = _camera_freshness.observe(cam)
                if _camera_stalled:
                    logger.error(
                        "Camera source stale for %.1fs — discarding episode; teleoperation remains RUNNING",
                        cfg.runtime.camera.recording_stall_abort_s,
                    )
                    print("  ⚠ 相机连续失帧超过阈值，当前 episode 已废弃；遥操作继续")
                    _stop_recording(
                        recorder,
                        recording_active,
                        save=False,
                        shared=shared,
                        reason="camera_stall",
                    )
                    recording_active = False
            stage_timer.mark("cam")

            hand_state = _read_hand_state(shared, anchor_monotonic_ns=_current_grid_anchor_ns)
            hand_tactile = _read_hand_tactile(shared, anchor_monotonic_ns=_current_grid_anchor_ns)

            hand_issue = _hand_feedback_issue(hand_state)
            if cfg.runtime.policy.hand_enabled and hand_issue is not None:
                now_s = time.monotonic()
                if _hand_disconnected_at is None:
                    _hand_disconnected_at = now_s
                    logger.warning("Hand feedback unhealthy — pausing motion: %s", hand_issue)
                if teleop_active and not _quiescence.active:
                    _enter_command_quiescence("hand_feedback")
                unhealthy_duration_s = now_s - _hand_disconnected_at
                if unhealthy_duration_s >= cfg.runtime.policy.hand_disconnect_timeout_s:
                    logger.error(
                        "Hand feedback remained unhealthy for %.1fs: %s",
                        unhealthy_duration_s,
                        hand_issue,
                    )
                    shared.error_state.value = True
                    break
            elif cfg.runtime.policy.hand_enabled and _hand_disconnected_at is not None:
                unhealthy_duration_s = time.monotonic() - _hand_disconnected_at
                _hand_disconnected_at = None
                logger.info(
                    "Hand feedback recovered after %.1fs — waiting for fresh re-anchor",
                    unhealthy_duration_s,
                )

            if loop_count % cfg.runtime.policy.status_print_interval == 0:
                _arm_age = time.monotonic() - float(arm_state["timestamp"][0]) if arm_state is not None else -1.0
                _qdepth = shared.arm_action_q.qsize()
                _print_status(
                    loop_count,
                    arm_state,
                    vr_frame,
                    teleop_active,
                    recording_active,
                    error_count,
                    arm_q_depth=_qdepth,
                    arm_state_age_s=_arm_age,
                )

            if teleop_active and vr_stale and not _quiescence.active:
                _enter_command_quiescence("vr_stale")

            # State-transition audio gates motion by invalidating old commands
            # once and publishing nothing until the cue has released.
            _audio_playing = audio.is_playing
            _hold_for_audio, _begin_audio_gate_deadline_s, _ignore_begin_audio_until_silent = update_motion_gate(
                audio_playing=_audio_playing,
                begin_deadline_s=_begin_audio_gate_deadline_s,
                ignore_begin_until_silent=_ignore_begin_audio_until_silent,
                now_s=time.monotonic(),
            )
            if teleop_active and _hold_for_audio and not _quiescence.active:
                _enter_command_quiescence("audio_gate")

            if not teleop_active or vr_stale or _quiescence.active:
                # Resumption requires source feedback newer than the original
                # quiescence boundary.  This grid only re-anchors; command
                # publication starts on the following grid.
                if (
                    teleop_active
                    and not vr_stale
                    and not _hold_for_audio
                    and _quiescence.active
                    and vr_frame is not None
                    and (not hand_available or hand_state is not None)
                    and (not hand_available or hand_issue is None)
                    and _quiescence.feedback_is_newer(
                        arm_source_monotonic_ns=int(arm_state["source_monotonic_ns"][0]),
                        vr_receive_monotonic_ns=int(vr_frame["local_recv_ns"]),
                        hand_source_monotonic_ns=(
                            int(hand_state["source_monotonic_ns"][0])
                            if hand_available and hand_state is not None
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
                # Track measured arm position while silent.  This updates only
                # the local command baseline; it does not publish a hold target.
                ctx.prev_qpos_cmd = arm_qpos.copy()
                ctx.ema_prev_pos = ctx.ema_prev_quat = None
                continue

            if vr_frame is None:
                logger.warning("teleop_loop: vr_frame is None after vr_stale check — suppressing publication")
                if recording_active:
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
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                        hand_fk=_hand_fk,
                        T_eef_handbase_pos=_T_eef_handbase_pos,
                        T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                        observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                        shared=shared,
                    )
                continue
            _policy_compute_t0 = time.perf_counter()
            _map_t0 = time.perf_counter()
            mapped = arm_mapper.map(vr_frame["wrist_pos"], vr_frame["wrist_quat_wxyz"])
            if mapped is None:
                hold_result = _safe_arm_queue_put(
                    shared,
                    {"qpos": ctx.prev_qpos_cmd.copy(), "is_hold": True},
                    timeout=cfg.runtime.policy.action_prepare_timeout_s,
                    observation_id=int(vr_frame["ring_sequence"]),
                    observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                    safety_gate=gate,
                    hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
                )
                published_hold = hold_result.candidate
                if not hold_result.succeeded or published_hold is None:
                    logger.error("teleop_loop: mapper hold publish failed: %s", hold_result.reason)
                    shared.error_state.value = True
                    break
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        ctx.prev_qpos_cmd,
                        ctx.prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        arm_qpos_sent=ctx.prev_qpos_cmd.copy(),
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                        hand_fk=_hand_fk,
                        T_eef_handbase_pos=_T_eef_handbase_pos,
                        T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                        observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                        shared=shared,
                        action_candidate=published_hold,
                    )
                continue

            target_pos_raw = np.asarray(mapped["pos"], dtype=np.float64).copy()
            target_quat_raw = np.asarray(mapped["quat_wxyz"], dtype=np.float64).copy()
            target_pos = target_pos_raw.copy()
            target_quat = target_quat_raw.copy()

            if ctx.ema_prev_pos is not None:
                if ctx.ema_prev_quat is None:
                    logger.warning("teleop_loop: ctx.ema_prev_quat is None but ctx.ema_prev_pos is set — skipping EMA")
                else:
                    target_pos, target_quat = ema_smooth_pose(
                        target_pos,
                        target_quat,
                        ctx.ema_prev_pos,
                        ctx.ema_prev_quat,
                        cfg.runtime.policy.ema.alpha_pos,
                        cfg.runtime.policy.ema.alpha_rot,
                    )

            target_pos_before_clamp = target_pos.copy()
            for axis in range(3):
                lo, hi = cfg.runtime.policy.workspace.as_tuple()[axis]
                target_pos[axis] = np.clip(target_pos[axis], lo, hi)
            policy_map_time_ms = (time.perf_counter() - _map_t0) * 1000.0
            stage_timer.mark("map")

            # Compute the hand command first so the arm IK collision model sees
            # the current-frame (post-shaping) hand pose, not the previous
            # frame's applied command.  Hand-only motion stays independent of
            # the IK outcome; the hand-only hold below still publishes on IK
            # failure.
            _hand_retarget_t0 = time.perf_counter()
            hand_cmd, retarget_ok = _compute_hand_command(
                ctx.hand_retargeter,
                vr_frame,
                ctx.prev_hand_qpos,
                hand_available,
                ctx.hand_retarget_cache,
            )
            # Cache hits intentionally measure lookup time, not a synthetic
            # repeated solver duration; the recording data dictionary documents
            # this distinction.
            hand_retarget_time_ms = (time.perf_counter() - _hand_retarget_t0) * 1000.0
            hand_cmd_raw = _get_raw_hand_command(ctx.hand_retargeter, hand_cmd, retarget_ok)
            # Rate-independent smoothstep ramp from the measured pose captured
            # when command quiescence ends. The last configured frame reaches
            # the live target exactly, avoiding an extra one-frame tail.
            if ctx.hand_ramp_start is not None and ctx.hand_ramp_step < _hand_ramp_total_frames:
                hand_cmd = _smoothstep_hand_ramp(
                    ctx.hand_ramp_start,
                    hand_cmd,
                    ctx.hand_ramp_step,
                    _hand_ramp_total_frames,
                )
                ctx.hand_ramp_step += 1
                if ctx.hand_ramp_step >= _hand_ramp_total_frames:
                    ctx.hand_ramp_start = None
            elif ctx.hand_ramp_start is not None:
                ctx.hand_ramp_start = None
                ctx.hand_ramp_step = 0

            # Hand command-floor clip: project the command into the
            # operator-set anti-clogging command box instead of rejecting the
            # whole action; measured feedback is never clipped here.
            hand_cmd = np.clip(hand_cmd, hand_qpos_lower_rad, hand_qpos_upper_rad)
            hand_cmd_valid = True
            try:
                hand_cmd = _sanitize_hand_command(
                    hand_cmd,
                    hand_qpos_lower_rad,
                    hand_qpos_upper_rad,
                    hand_mechanical_lower_rad,
                    hand_mechanical_upper_rad,
                )
            except ValueError as exc:
                _validate_warn("teleop_loop: invalid hand command — holding: %s", exc)
                hand_cmd = ctx.prev_hand_qpos.copy()
                hand_cmd_valid = False
                retarget_ok = False

            planner.set_hand_qpos(hand_cmd)  # current-frame hand pose for collision checks
            target_pose = Pose(p=target_pos, q=target_quat)
            _ik_t0 = time.perf_counter()
            ik_result = planner.solve_teleop_ik(target_pose, arm_qpos, ctx.prev_qpos_cmd)
            ik_solve_time_ms = (time.perf_counter() - _ik_t0) * 1000.0
            stage_timer.mark("ik")

            if not ik_result.success or ik_result.qpos is None:
                # Arm is held; hand motion proceeds independently.
                # SafetyGate validates well-formedness + joint limits only for hand-only commands.
                if hand_available:
                    safe_hand_cmd = hand_cmd if hand_cmd_valid else None
                    publish_result = _safe_joint_publish(
                        shared,
                        ctx.prev_qpos_cmd.copy(),
                        safe_hand_cmd,
                        is_hold=True,
                        timeout=cfg.runtime.policy.action_prepare_timeout_s,
                        observation_id=int(vr_frame["ring_sequence"]),
                        observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                        safety_gate=gate,
                        hand_mechanical_lower_rad=hand_mechanical_lower_rad,
                        hand_mechanical_upper_rad=hand_mechanical_upper_rad,
                        hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
                    )
                    published_candidate = publish_result.candidate
                    if not publish_result.succeeded or published_candidate is None:
                        logger.error("teleop_loop: IK-failure hold publish failed: %s", publish_result.reason)
                        shared.error_state.value = True
                        break
                    if published_candidate.arm_qpos is not None:
                        ctx.prev_qpos_cmd = np.asarray(published_candidate.arm_qpos, dtype=np.float64)
                    if published_candidate.hand_qpos is not None:
                        ctx.prev_hand_qpos = np.asarray(published_candidate.hand_qpos, dtype=np.float64).copy()
                else:
                    publish_result = _safe_arm_queue_put(
                        shared,
                        {"qpos": ctx.prev_qpos_cmd.copy(), "is_hold": True},
                        timeout=cfg.runtime.policy.action_prepare_timeout_s,
                        observation_id=int(vr_frame["ring_sequence"]),
                        observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                        safety_gate=gate,
                        hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
                    )
                    published_candidate = publish_result.candidate
                    if not publish_result.succeeded or published_candidate is None:
                        logger.error("teleop_loop: IK-failure arm hold publish failed: %s", publish_result.reason)
                        shared.error_state.value = True
                        break
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        ctx.prev_qpos_cmd,
                        ctx.prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        frame_status=_FRAME_IK_FAIL,
                        retarget_ok=retarget_ok,
                        arm_qpos_sent=ctx.prev_qpos_cmd.copy(),
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                        hand_fk=_hand_fk,
                        T_eef_handbase_pos=_T_eef_handbase_pos,
                        T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                        observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                        shared=shared,
                        action_candidate=published_candidate,
                    )
                continue

            arm_cmd_raw = np.asarray(ik_result.qpos, dtype=np.float64).copy()
            arm_cmd = arm_cmd_raw.copy()

            # Keep IK output inside the firmware-accepted joint limits;
            # velocity/acceleration smoothing is Mode 6 firmware's job.
            arm_cmd = np.clip(arm_cmd, joint_lower_rad, joint_upper_rad)

            # Arm delta-clip fallback: cap the command-to-command joint step so
            # a distant IK solution (multi-seed fallback / nullspace jump)
            # degrades to a bounded ramp instead of an incoherent jerk. This is
            # NOT application-side interpolation — it edits the single Mode-6
            # endpoint per grid tick and Mode 6 still owns trajectory
            # smoothing. compute_qpos_delta wraps J1/J3/J5/J7 into (-pi, pi],
            # so crossing the +-pi seam is never mistaken for a ~2pi jump, and
            # re-adding keeps the command in prev_qpos_cmd's unwrapped frame.
            if (
                arm_max_delta_rad_per_tick is not None
                and ctx.prev_qpos_cmd is not None
                and np.all(np.isfinite(ctx.prev_qpos_cmd))
            ):
                _arm_delta = planner.compute_qpos_delta(arm_cmd, ctx.prev_qpos_cmd)
                _arm_delta = np.clip(_arm_delta, -arm_max_delta_rad_per_tick, arm_max_delta_rad_per_tick)
                arm_cmd = ctx.prev_qpos_cmd + _arm_delta
                arm_cmd = np.clip(arm_cmd, joint_lower_rad, joint_upper_rad)

            # Pre-flight checks specific to teleop command assembly. Joint
            # limits and workspace are enforced once by SafetyGate.
            _reject = False
            _reject_reason = ""
            if not np.all(np.isfinite(arm_cmd)):
                _reject = True
                _reject_reason = "arm_cmd NaN/Inf"
            elif not hand_cmd_valid:
                _reject = True
                _reject_reason = "hand command validation failed"
            if _reject:
                _validate_warn("teleop_loop: action rejected — %s", _reject_reason)
                hold_result = _safe_joint_publish(
                    shared,
                    ctx.prev_qpos_cmd.copy(),
                    None,
                    is_hold=True,
                    timeout=cfg.runtime.policy.action_prepare_timeout_s,
                    observation_id=int(vr_frame["ring_sequence"]),
                    observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                    safety_gate=gate,
                    hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
                )
                published_hold = hold_result.candidate
                if not hold_result.succeeded or published_hold is None:
                    logger.error("teleop_loop: rejected-action hold publish failed: %s", hold_result.reason)
                    shared.error_state.value = True
                    break
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        ctx.prev_qpos_cmd,
                        ctx.prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        frame_status=_FRAME_SAFETY_REJECT,
                        retarget_ok=retarget_ok,
                        arm_qpos_sent=ctx.prev_qpos_cmd.copy(),
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                        hand_fk=_hand_fk,
                        T_eef_handbase_pos=_T_eef_handbase_pos,
                        T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                        observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                        shared=shared,
                        action_candidate=published_hold,
                    )
                continue

            publish_result = _safe_joint_publish(
                shared,
                arm_cmd.copy(),
                hand_cmd.copy() if hand_available else None,
                timeout=cfg.runtime.policy.action_prepare_timeout_s,
                observation_id=int(vr_frame["ring_sequence"]),
                observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                safety_gate=gate,
                hand_mechanical_lower_rad=hand_mechanical_lower_rad,
                hand_mechanical_upper_rad=hand_mechanical_upper_rad,
                hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
            )
            published_candidate = publish_result.candidate
            workspace_rejected = (
                publish_result.status == CommandPublishStatus.GATE_REJECTED
                and publish_result.gate_code in (GateRejectCode.WORKSPACE, GateRejectCode.WORKSPACE_CHECK_FAILED)
            )
            if workspace_rejected:
                _validate_warn("teleop_loop: action rejected — %s; publishing hold", publish_result.reason)
                hold_result = _safe_joint_publish(
                    shared,
                    ctx.prev_qpos_cmd.copy(),
                    None,
                    is_hold=True,
                    timeout=cfg.runtime.policy.action_prepare_timeout_s,
                    observation_id=int(vr_frame["ring_sequence"]),
                    observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                    safety_gate=gate,
                    hand_feedback_max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
                )
                published_hold = hold_result.candidate
                if not hold_result.succeeded or published_hold is None:
                    logger.error("teleop_loop: workspace-rejection hold publish failed: %s", hold_result.reason)
                    shared.error_state.value = True
                    break
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        ctx.prev_qpos_cmd,
                        ctx.prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        frame_status=_FRAME_SAFETY_REJECT,
                        retarget_ok=retarget_ok,
                        arm_qpos_sent=ctx.prev_qpos_cmd.copy(),
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                        hand_fk=_hand_fk,
                        T_eef_handbase_pos=_T_eef_handbase_pos,
                        T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                        observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                        shared=shared,
                        action_candidate=published_hold,
                    )
                continue
            if not publish_result.succeeded or published_candidate is None:
                if publish_result.runtime_gated:
                    logger.info("teleop_loop: joint publication stopped by runtime gate: %s", publish_result.reason)
                    if publish_result.status == CommandPublishStatus.SAFETY_STATE_GATED:
                        if recording_active:
                            _record_held(
                                recorder,
                                arm_state,
                                ctx.prev_qpos_cmd,
                                ctx.prev_hand_qpos,
                                vr_frame,
                                cam,
                                hand_state=hand_state,
                                hand_tactile=hand_tactile,
                                arm_qpos_sent=ctx.prev_qpos_cmd.copy(),
                                target_eef_pos=_last_target_eef_pos,
                                target_eef_rot6d=_last_target_eef_rot6d,
                                hand_fk=_hand_fk,
                                T_eef_handbase_pos=_T_eef_handbase_pos,
                                T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                                observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                                shared=shared,
                            )
                        continue
                    break
                logger.error("teleop_loop: joint publish failed: %s", publish_result.reason)
                shared.error_state.value = True
                break
            stage_timer.mark("send")

            if published_candidate.arm_qpos is not None:
                arm_cmd = np.asarray(published_candidate.arm_qpos, dtype=np.float64)
            if published_candidate.hand_qpos is not None:
                hand_cmd = np.asarray(published_candidate.hand_qpos, dtype=np.float64)
            ctx.prev_qpos_cmd = arm_cmd.copy()
            ctx.prev_hand_qpos = hand_cmd.copy()
            ctx.ema_prev_pos = target_pos.copy()
            ctx.ema_prev_quat = target_quat.copy()

            if recording_active:
                policy_compute_time_ms = (time.perf_counter() - _policy_compute_t0) * 1000.0
                # Track last valid IK target for held-frame continuity.
                _last_target_eef_pos = target_pos.copy()
                _last_target_eef_rot6d = quat_wxyz_to_rot6d(normalize_quat_wxyz(target_quat))
                # Hand retargeting fail but IK success → mark separately
                if not retarget_ok and hand_available:
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
                    target_eef_rot6d_raw=quat_wxyz_to_rot6d(normalize_quat_wxyz(target_quat_raw)),
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

    finally:
        if recording_active:
            _stop_recording(recorder, True, save=False, shared=shared, reason="policy_shutdown")
        # Direct finalization lives in this process; join its bounded writer
        # before policy exits so a published episode is never truncated.
        if isinstance(recorder, DirectRecorderClient) and recorder.stop_pending:
            result = recorder.join_stop(timeout=60.0)
            if result.error:
                logger.error("direct recorder finalization failed: %s", result.error)
        kb.stop()
        audio.play("end")
        time.sleep(_END_AUDIO_GRACE_S)
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
        return CommandPublishResult(CommandPublishStatus.INVALID_CANDIDATE, detail=str(exc))


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
        return CommandPublishResult(CommandPublishStatus.INVALID_CANDIDATE, detail=str(exc))
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
    """Periodic status print (~1 Hz)."""
    if arm_state is not None:
        _e = arm_state["eef_pos"][0]
        eef_str = f"eef={_e[0]:.3f},{_e[1]:.3f},{_e[2]:.3f}"
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
