"""VR teleoperation coordinator over causal shared-memory snapshots.

This module builds policy-process resources, waits for readiness, schedules
operator control and causal-grid work, and performs bounded cleanup.
``operator_controls`` owns operator transitions and recording decisions;
``control_grid`` owns one observation-to-publication tick.  Hardware SDKs stay
inside the arm, hand, VR, camera, and RecorderIO workers.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path

import numpy as np

from dexmani_real.control.safety_gate import SafetyGate, planner_action_safety_gate
from dexmani_real.ipc.causal import read_arm_state_causal, read_hand_state_causal
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.hand_fk import HandKinematics
from dexmani_real.recording.client import RecorderClient
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import SafetyState
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.camera_freshness import CameraFreshnessTracker
from dexmani_real.teleop.config import TeleopCommandLimits, TeleopConfig
from dexmani_real.teleop.control_grid import (
    TeleopControlResources,
    TeleopGridResources,
    run_control_grid_tick,
)
from dexmani_real.teleop.control_state import (
    CommandQuiescence,
    CoordinatorDirective,
    TeleopLoopState,
)
from dexmani_real.teleop.episode_samples import stop_recording
from dexmani_real.teleop.hand_control import hand_ramp_frame_count
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.operator_controls import (
    TeleopOperatorResources,
    advance_post_teleop_state,
    apply_operator_controls,
    handoff_quiescence_to_home,
    poll_recording_lifecycle,
    transition_or_fault,
    try_init_hand_retargeter,
)
from dexmani_real.teleop.safety import enter_command_quiescence, hand_feedback_issue
from dexmani_real.teleop.timing import StageTimer
from dexmani_real.teleop.vr_transform import load_vr_transform
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.rate import LoopRate

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


def _start_keyboard(shared: RuntimeChannels) -> KeyboardHandler | None:
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


def _load_control_resources(
    shared: RuntimeChannels,
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
        # Table contact is intentional during fine teleoperation. Homing still
        # applies its independent table-clearance validation.
        table=None,
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
    shared: RuntimeChannels,
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


def teleop_loop(shared: RuntimeChannels, config: TeleopConfig | None = None) -> None:
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
    _hand_ramp_total_frames = hand_ramp_frame_count(
        cfg.runtime.policy.hand_ramp_duration_s, cfg.runtime.policy.control_hz
    )

    def _try_init_hand_retargeter() -> bool:
        return try_init_hand_retargeter(ctx, cfg)

    def _hand_feedback_issue(state: np.ndarray | None) -> str | None:
        return hand_feedback_issue(cfg, state)

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

    # The coordinator polls controls; only the control grid publishes motion.
    # Avoid spending a CPU core's precision spin window or warning on harmless
    # sub-grid scheduler jitter. Actual skipped control grids are reported below.
    limiter = LoopRate(
        cfg.runtime.policy.coordinator_hz,
        label="teleop",
        busy_wait=False,
        warn_on_overrun=False,
    )
    _grid_period_ns = int(round(_NS_PER_SECOND / cfg.runtime.policy.control_hz))
    _next_grid_ns = time.monotonic_ns() + _grid_period_ns
    _current_grid_anchor_ns = _next_grid_ns
    _pending_controls: list[ControlSignal] = []
    stage_timer = StageTimer(window=cfg.runtime.policy.status_print_interval)
    _validate_warn = ThrottledWarner(interval_s=_VALIDATION_WARN_INTERVAL_S)
    _arm_feedback_warn = ThrottledWarner(interval_s=_ARM_FEEDBACK_WARN_INTERVAL_S)
    _grid_overrun_warn = ThrottledWarner(interval_s=_VALIDATION_WARN_INTERVAL_S)
    missed_control_grid_total = 0
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
        return transition_or_fault(shared, new_state, reason)

    def _handoff_operator_quiescence_to_home() -> None:
        handoff_quiescence_to_home(operator_resources)

    def _enter_command_quiescence(
        reason: str,
        *,
        start_new_run: bool = False,
        replace_existing_reason: bool = False,
    ) -> None:
        enter_command_quiescence(
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

            if not poll_recording_lifecycle(
                ctx,
                shared,
                recorder,
                audio,
                enter_quiescence=_enter_command_quiescence,
                transition_or_fault=_transition_shared_or_fault,
            ):
                break

            post_teleop_directive = advance_post_teleop_state(
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
                if _missed_periods > 1:
                    missed_control_grid_total += int(_missed_periods - 1)
                    _grid_overrun_warn(
                        "teleop_loop: skipped %d control-grid slots (total=%d, lateness=%.1fms)",
                        _missed_periods - 1,
                        missed_control_grid_total,
                        (_coordinator_now_ns - _next_grid_ns) / 1e6,
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
            operator_directive = apply_operator_controls(
                ctx,
                shared,
                cfg,
                operator_resources,
                _controls,
            )
            if operator_directive is CoordinatorDirective.BREAK:
                break
            if operator_directive is CoordinatorDirective.REANCHOR_GRID:
                # Synchronous home is an intentional command-silent interval,
                # not missed teleop work. Re-anchor both clocks here, where the
                # control-grid deadline is owned, and wait for a new causal slot.
                limiter.reset()
                _next_grid_ns = time.monotonic_ns() + _grid_period_ns
                _current_grid_anchor_ns = _next_grid_ns
                continue
            if operator_directive is CoordinatorDirective.CONTINUE:
                continue
            if not _grid_due:
                continue

            grid_directive = run_control_grid_tick(
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
            stop_recording(
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
