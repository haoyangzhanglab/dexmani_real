"""VR teleoperation control loop over causal shared-memory snapshots.

This module builds policy-process resources, waits for readiness, schedules
operator control and causal-grid work, and performs bounded cleanup.
The loop directly owns operator transitions, pause boundaries, recording, and
grid cadence; ``control_grid`` owns algorithm state and one causal tick. Hardware SDKs stay
inside the arm, hand, VR, camera, and RecorderIO workers.
"""

from __future__ import annotations

import gc
import signal
import time
from pathlib import Path

import numpy as np

from dexmani_real.control.safety_gate import SafetyGate, planner_action_safety_gate
from dexmani_real.ipc.causal import (
    read_arm_state_causal,
    read_hand_state_causal,
    read_hand_tactile_causal,
    read_vr_frame_causal,
    vr_frame_is_fresh,
)
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.hand_fk import HandKinematics
from dexmani_real.recording.client import RecorderClient, RecorderPhase
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    invalidate_coupled_commands,
    transition,
)
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.camera_freshness import CameraFreshnessTracker
from dexmani_real.teleop.config import TeleopCommandLimits, TeleopConfig
from dexmani_real.teleop.control_grid import (
    TeleopController,
    TeleopGridResources,
    run_control_grid_tick,
)
from dexmani_real.teleop.episode_samples import (
    RECORDING_TACTILE_MAX_AGE_NS,
    stop_recording,
)
from dexmani_real.teleop.hand_control import hand_ramp_frame_count, seed_hand_retargeter
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.recording_session import (
    QuitRecordingDecision,
    await_quit_recording_decision,
)
from dexmani_real.teleop.retarget.facade import (
    TAGHandRetargeter,
    XHandRetargeter,
)
from dexmani_real.teleop.safety import do_configured_teleop_home, hand_feedback_issue
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
) -> tuple[
    XArm7MotionPlanner,
    ArmWristMapper,
    SafetyGate,
    RecorderClient | None,
]:
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
    vr_to_robot_rot = vr_calibration.transform
    logger.info("VR transform loaded: theta=%.6g°", vr_calibration.theta_deg)
    arm_mapper = ArmWristMapper(
        pos_scale=config.runtime.policy.vr_mapping.pos_scale,
        rot_scale=config.runtime.policy.vr_mapping.rot_scale,
        vr_to_robot_rot=vr_to_robot_rot,
        max_delta_rot_rad=config.runtime.policy.vr_mapping.max_delta_rot_rad,
        base_to_world_rot=np.eye(3, dtype=np.float64),
    )
    return (
        planner,
        arm_mapper,
        _build_safety_gate(config, planner),
        RecorderClient(shared) if recording_enabled else None,
    )


def _build_hand_retargeter(config: TeleopConfig):
    """Build the configured hand retargeter without owning runtime state."""
    if not config.runtime.policy.hand_enabled:
        return None
    if config.runtime.policy.hand_retargeting_type == "tag":
        return TAGHandRetargeter(
            fingertip_link_names=config.runtime.hand.fingertip_link_names,
            tag_config=config.runtime.tag_retargeting,
            urdf_path=config.hand_urdf_path,
        )
    return XHandRetargeter(
        hand_type="right",
        retargeting_type=config.runtime.policy.hand_retargeting_type,
        dexpilot_config=config.runtime.dexpilot_retargeting,
    )


def _transition_or_fault(
    shared: RuntimeChannels,
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


def _load_hand_kinematics(
    config: TeleopConfig,
    *,
    recording_enabled: bool,
) -> HandKinematics | None:
    """Load the hand FK required to produce valid hand recording samples."""
    if not recording_enabled or not config.runtime.policy.hand_enabled:
        return None
    hand_fk = HandKinematics(
        config.hand_urdf_path,
        list(config.runtime.hand.fingertip_link_names),
    )
    if not hand_fk.is_ready():
        raise RuntimeError("Hand FK is not ready")
    logger.info("Hand FK ready")
    return hand_fk


def _begin_feedback_issue(
    cfg: TeleopConfig,
    vr_frame: dict | None,
    hand_state: np.ndarray | None,
    hand_tactile: np.ndarray | None,
    *,
    recording_enabled: bool,
    now_monotonic_ns: int,
) -> str | None:
    """Return the first data-admission issue before beginning a session."""
    if not vr_frame_is_fresh(
        vr_frame,
        now_monotonic_ns=now_monotonic_ns,
        max_age_s=cfg.runtime.policy.vr_mapping.stale_threshold_s,
    ):
        return "VR hand feedback is unavailable or stale"
    hand_issue = hand_feedback_issue(cfg, hand_state)
    if hand_issue is not None:
        return hand_issue
    if not recording_enabled or not cfg.runtime.policy.hand_enabled:
        return None
    if hand_tactile is None:
        return "tactile feedback is unavailable"
    tactile = hand_tactile[0]
    tactile_source_ns = int(tactile["source_monotonic_ns"])
    if not (
        bool(tactile["fresh"])
        and 0 < tactile_source_ns <= now_monotonic_ns
        and now_monotonic_ns - tactile_source_ns <= RECORDING_TACTILE_MAX_AGE_NS
    ):
        return "tactile feedback is stale"
    if not bool(tactile["calibrated"]):
        return "tactile feedback is not calibrated"
    return None


def teleop_loop(shared: RuntimeChannels, config: TeleopConfig) -> None:
    """Teleoperation process entry point used by ``collect_teleop.py``.

    Reads from rings (vr, arm_state, hand_state, camera), writes actions
    to the coherent actuator command ring, owns recording.
    """
    cfg = config
    logger.debug("teleop_loop: LOADING")
    command_limits = TeleopCommandLimits.from_config(cfg)
    recording_enabled = bool(cfg.runtime.policy.recording_enabled)
    try:
        planner, arm_mapper, safety_gate, recorder = _load_control_resources(
            shared, cfg, recording_enabled=recording_enabled
        )
        hand_fk = _load_hand_kinematics(cfg, recording_enabled=recording_enabled)
    except Exception:
        logger.error("teleop_loop: init failed", exc_info=True)
        shared.error_state.value = True
        return
    kb = _start_keyboard(shared)
    if kb is None:
        return
    audio = AudioFeedback()

    handbase_position_eef_m = np.asarray(
        cfg.runtime.hand.T_eef_handbase_pos_xyz, dtype=np.float64
    )
    handbase_quat_eef_wxyz = np.asarray(
        cfg.runtime.hand.T_eef_handbase_quat_wxyz, dtype=np.float64
    )
    arm_state = read_arm_state_causal(shared)
    hand_state = read_hand_state_causal(shared)
    if cfg.runtime.policy.hand_enabled:
        initial_hand_issue = hand_feedback_issue(cfg, hand_state)
        if initial_hand_issue is not None:
            logger.error(
                "Teleop: initial hand feedback rejected: %s", initial_hand_issue
            )
            shared.error_state.value = True
            kb.stop()
            return
        try:
            hand_retargeter = _build_hand_retargeter(cfg)
        except Exception:
            logger.error("Hand retargeter initialization failed", exc_info=True)
            shared.error_state.value = True
            kb.stop()
            return
        logger.info(
            "Hand retargeter ready (type=%s)", cfg.runtime.policy.hand_retargeting_type
        )
    else:
        hand_retargeter = None
        logger.info(
            "Hand explicitly disabled — using the configured fixed-home collision assumption"
        )
    arm_qpos = (
        np.asarray(arm_state["qpos"][0], dtype=np.float64).copy()
        if arm_state is not None
        else np.asarray(cfg.runtime.arm.home_qpos, dtype=np.float64).copy()
    )
    hand_qpos = (
        np.asarray(hand_state["qpos"][0], dtype=np.float64).copy()
        if hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0]))
        else command_limits.hand_home_qpos_rad.copy()
    )
    controller = TeleopController(
        planner=planner,
        arm_mapper=arm_mapper,
        config=cfg,
        command_limits=command_limits,
        initial_arm_qpos_rad=arm_qpos,
        initial_hand_qpos_rad=hand_qpos,
        hand_retargeter=hand_retargeter,
    )

    shared.set_heartbeat("policy", time.monotonic())
    shared.set_ready("policy")
    logger.debug("teleop_loop: READY")

    teleop_active = False
    recording_active = False
    pause_since_ns = 0
    pause_reason: str | None = None
    quit_pending = False
    quit_after_recording = False
    quit_recording_deadline_s = 0.0
    post_teleop_deadline_s = 0.0
    arm_feedback_error_count = 0
    hand_disconnected_at_s: float | None = None
    sigterm_requested = False

    limiter = LoopRate(
        cfg.runtime.policy.executor_poll_hz,
        label="teleop",
        busy_wait=False,
        warn_on_overrun=False,
    )
    grid_period_ns = int(round(_NS_PER_SECOND / cfg.runtime.policy.control_hz))
    next_grid_ns = time.monotonic_ns() + grid_period_ns
    current_grid_anchor_ns = next_grid_ns
    pending_controls: list[ControlSignal] = []
    stage_timer = StageTimer(window=cfg.runtime.policy.status_print_interval)
    validate_warn = ThrottledWarner(interval_s=_VALIDATION_WARN_INTERVAL_S)
    arm_feedback_warn = ThrottledWarner(interval_s=_ARM_FEEDBACK_WARN_INTERVAL_S)
    grid_overrun_warn = ThrottledWarner(interval_s=_VALIDATION_WARN_INTERVAL_S)
    missed_control_grid_total = 0
    loop_count = 0
    camera_freshness = CameraFreshnessTracker(
        max_age_s=cfg.runtime.camera.max_frame_age_s,
        abort_after_s=cfg.runtime.camera.recording_stall_abort_s,
    )
    grid_resources = TeleopGridResources(
        planner=planner,
        safety_gate=safety_gate,
        recorder=recorder,
        command_limits=command_limits,
        camera_freshness=camera_freshness,
        stage_timer=stage_timer,
        validation_warn=validate_warn,
        arm_feedback_warn=arm_feedback_warn,
        hand_fk=hand_fk,
        handbase_position_eef_m=handbase_position_eef_m,
        handbase_quat_eef_wxyz=handbase_quat_eef_wxyz,
        hand_ramp_total_frames=hand_ramp_frame_count(
            cfg.runtime.policy.hand_ramp_duration_s, cfg.runtime.policy.control_hz
        ),
        max_observation_skew_s=cfg.runtime.policy.max_observation_skew_s,
    )

    def enter_pause(
        reason: str,
        *,
        start_new_run: bool = False,
        relabel: bool = False,
    ) -> None:
        nonlocal pause_since_ns, pause_reason
        now_ns = time.monotonic_ns()
        if start_new_run:
            if pause_reason is not None:
                logger.info(
                    "teleop_loop: new run supersedes %s pause boundary",
                    pause_reason,
                )
            pause_since_ns = now_ns
            pause_reason = reason
            run_generation = int(shared.run_generation.value)
        elif pause_reason is None:
            pause_since_ns = now_ns
            pause_reason = reason
            run_generation = invalidate_coupled_commands(shared)
        else:
            if relabel:
                pause_reason = reason
            controller.clear_reference()
            logger.debug(
                "teleop_loop: remaining in %s pause boundary (observed %s)",
                pause_reason,
                reason,
            )
            return
        controller.clear_reference()
        logger.info(
            "teleop_loop: entered %s pause boundary (run=%d)",
            reason,
            run_generation,
        )

    def clear_pause_for_home() -> None:
        nonlocal pause_since_ns, pause_reason
        if pause_reason is not None:
            logger.info(
                "teleop_loop: homing supersedes %s pause boundary", pause_reason
            )
        pause_since_ns = 0
        pause_reason = None
        controller.clear_reference()

    def run_home() -> None:
        clear_pause_for_home()
        controller.prev_hand_qpos = do_configured_teleop_home(
            shared,
            cfg,
            hand_available=controller.hand_enabled,
            prev_hand_qpos=controller.prev_hand_qpos,
            planner=planner,
            audio=audio,
            estop_requested=lambda: kb.estop_latched or not kb.healthy,
            arm_mapper=controller.arm_mapper,
            hand_retargeter=controller.hand_retargeter,
        )

    def on_sigterm(signum: int, frame: object) -> None:
        nonlocal sigterm_requested
        sigterm_requested = True

    signal.signal(signal.SIGTERM, on_sigterm)
    logger.info(
        "Teleop: entering control loop @ %.0f Hz (observation/action grid %.0f Hz)",
        cfg.runtime.policy.executor_poll_hz,
        cfg.runtime.policy.control_hz,
    )

    try:
        while shared.is_running.value and not sigterm_requested:
            shared.set_heartbeat("policy", time.monotonic())
            limiter.wait()

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
                    and (teleop_active or recording_active)
                )
                if reached_limit:
                    enter_pause("max_frames", relabel=True)
                    teleop_active = False
                    recording_active = False
                    shared.is_recording.value = False
                    if not _transition_or_fault(
                        shared, SafetyState.ARMED, "maximum recording duration"
                    ):
                        break
                    print("  已达到最大录制时长：正在自动保存，遥操作进入静默暂停")
                    audio.play("pause")
                if stop_result.done:
                    recording_active = False
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
                    if quit_after_recording:
                        shared.quit_requested.value = True
                elif (
                    stop_result.phase is RecorderPhase.FINALIZING and stop_result.error
                ):
                    print("  ⚠ 录制终结超过时限；仍在安全回收，本会话将标记为失败")
                if recording_active and recorder.camera_writer_error is not None:
                    logger.error(
                        "Camera writer failed — discarding current episode: %s",
                        recorder.camera_writer_error,
                    )
                    stop_recording(
                        recorder,
                        True,
                        save=False,
                        shared=shared,
                        reason="camera_writer_error",
                    )
                    recording_active = False

            if (
                shared.estop_request.value
                or shared.quit_requested.value
                or shared.error_state.value
                or not shared.is_running.value
            ):
                break

            if quit_pending:
                home_handled = False
                for control in kb.poll(timeout=0.1):
                    if control is ControlSignal.HOME and not home_handled:
                        home_handled = True
                        print("  H: return_home")
                        audio.play("home")
                        run_home()
                        kb.drain_signal(ControlSignal.HOME)
                        limiter.reset()
                        print("  [Q] quit", flush=True)
                    elif control in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
                        if control is ControlSignal.EMERGENCY_STOP:
                            shared.estop_request.value = True
                        elif recorder is not None and recorder.stop_pending:
                            if not quit_after_recording:
                                quit_after_recording = True
                                quit_recording_deadline_s = (
                                    time.monotonic()
                                    + cfg.runtime.policy.quit_save_timeout_s
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
                    break
                recording_stop_pending = recorder is not None and recorder.stop_pending
                if (
                    quit_after_recording
                    and recording_stop_pending
                    and time.monotonic() >= quit_recording_deadline_s
                ):
                    print("  录制终结超时 — 退出并将本会话标记为失败")
                    shared.quit_requested.value = True
                    break
                if time.perf_counter() <= post_teleop_deadline_s:
                    continue
                if recording_stop_pending:
                    if not quit_after_recording:
                        quit_after_recording = True
                        quit_recording_deadline_s = (
                            time.monotonic() + cfg.runtime.policy.quit_save_timeout_s
                        )
                        print("  timeout — 等待录制终结后自动退出", flush=True)
                    continue
                print("  timeout — auto exit")
                shared.quit_requested.value = True
                break

            pending_controls.extend(kb.poll(timeout=0.0))
            loop_now_ns = time.monotonic_ns()
            grid_due = loop_now_ns >= next_grid_ns
            if grid_due:
                missed_periods = max(
                    1, (loop_now_ns - next_grid_ns) // grid_period_ns + 1
                )
                if missed_periods > 1:
                    missed_control_grid_total += int(missed_periods - 1)
                    grid_overrun_warn(
                        "teleop_loop: skipped %d control-grid slots (total=%d, lateness=%.1fms)",
                        missed_periods - 1,
                        missed_control_grid_total,
                        (loop_now_ns - next_grid_ns) / 1e6,
                    )
                current_grid_anchor_ns = (
                    next_grid_ns + int(missed_periods - 1) * grid_period_ns
                )
                next_grid_ns += int(missed_periods) * grid_period_ns
                loop_count += 1
                stage_timer.tick()
                stage_timer.mark("control_loop")
            if not grid_due and not pending_controls:
                continue

            controls = tuple(pending_controls)
            pending_controls.clear()
            skip_control_tick = False
            reanchor_grid = False
            break_loop = False
            for control in controls:
                if control is ControlSignal.EMERGENCY_STOP:
                    print("\nESC: emergency_stop")
                    audio.play("emergency")
                    shared.estop_request.value = True
                    stop_recording(
                        recorder, recording_active, save=False, shared=shared
                    )
                    recording_active = False
                    break_loop = True
                    break
                if control is ControlSignal.QUIT:
                    print("\nQ: 退出")
                    audio.play("quit")
                    enter_pause("quit", relabel=True)
                    teleop_active = False
                    if not _transition_or_fault(shared, SafetyState.ARMED, "quit"):
                        break_loop = True
                        break
                    if recording_active:
                        audio.queue("quit_save_prompt")
                        print(
                            "  [S] 保存并退出  [D] 丢弃并退出  [H] 保存并归位 "
                            f"({cfg.runtime.policy.quit_save_timeout_s:.0f}s 超时默认丢弃)"
                        )
                        decision = await_quit_recording_decision(
                            shared, kb, timeout_s=cfg.runtime.policy.quit_save_timeout_s
                        )
                        save = decision in (
                            QuitRecordingDecision.SAVE,
                            QuitRecordingDecision.SAVE_AND_HOME,
                        )
                        audio.play(
                            "emergency"
                            if decision is QuitRecordingDecision.ESTOP
                            else ("save" if save else "discard")
                        )
                        stop_recording(recorder, True, save=save, shared=shared)
                        recording_active = False
                        if decision is QuitRecordingDecision.TIMEOUT:
                            print("  超时，默认丢弃请求已提交")
                        elif decision is QuitRecordingDecision.DISCARD:
                            print("  丢弃请求已提交")
                        elif save:
                            print("  保存请求已提交")
                        if (
                            decision is QuitRecordingDecision.SAVE_AND_HOME
                            and shared.is_running.value
                        ):
                            audio.play("home")
                            run_home()
                    quit_pending = True
                    post_teleop_deadline_s = (
                        time.perf_counter() + cfg.runtime.policy.post_teleop_timeout_s
                    )
                    print(
                        f"\n[H] return_home  [Q] quit  ({cfg.runtime.policy.post_teleop_timeout_s:.0f}s timeout)",
                        flush=True,
                    )
                    skip_control_tick = True
                    break
                if control is ControlSignal.HOME:
                    print("\nH: return_home")
                    audio.play("home")
                    stop_recording(recorder, recording_active, save=True, shared=shared)
                    recording_active = False
                    teleop_active = False
                    if not _transition_or_fault(shared, SafetyState.ARMED, "home"):
                        break_loop = True
                        break
                    run_home()
                    kb.drain_signal(ControlSignal.HOME)
                    reanchor_grid = True
                    skip_control_tick = True
                    break
                if control in (ControlSignal.STOP, ControlSignal.DISCARD):
                    save_episode = control is ControlSignal.STOP
                    reason = "stop" if save_episode else "discard"
                    print("\nS: 停止录制" if save_episode else "\nD: 丢弃录制")
                    audio.play("save" if save_episode else "discard")
                    enter_pause(reason, relabel=True)
                    stop_recording(
                        recorder, recording_active, save=save_episode, shared=shared
                    )
                    recording_active = False
                    teleop_active = False
                    if not _transition_or_fault(shared, SafetyState.ARMED, reason):
                        break_loop = True
                        break
                    skip_control_tick = True
                elif control is ControlSignal.PAUSE:
                    pause_signal_applied = False
                    if teleop_active:
                        enter_pause("pause", relabel=True)
                        teleop_active = False
                        if not _transition_or_fault(shared, SafetyState.ARMED, "pause"):
                            break_loop = True
                            break
                        pause_signal_applied = True
                    elif pause_reason == "pause":
                        if shared.safety_state.value == SafetyState.ARMED:
                            if not _transition_or_fault(
                                shared, SafetyState.RUNNING, "resume"
                            ):
                                break_loop = True
                                break
                            teleop_active = True
                            pause_signal_applied = True
                        else:
                            print(
                                f"\nC: safety_state={shared.safety_state.value} — must be ARMED to resume"
                            )
                    else:
                        print(
                            "\nC: 没有可恢复的暂停 session — 请按 B 开始新的遥操作 session"
                        )
                    if pause_signal_applied:
                        print(f"\nC: {'恢复' if teleop_active else '暂停'}遥操作")
                        audio.play("resume" if teleop_active else "pause")
                    skip_control_tick = True
                elif control is ControlSignal.BEGIN:
                    if teleop_active or recording_active:
                        print(
                            "\nB: session already active — use C to pause/resume, S to save, or D to discard"
                        )
                        skip_control_tick = True
                        continue
                    if shared.safety_state.value != SafetyState.ARMED:
                        print(
                            f"\nB: safety_state={shared.safety_state.value} — must be ARMED"
                        )
                        skip_control_tick = True
                        continue
                    begin_now_ns = time.monotonic_ns()
                    vr_frame = read_vr_frame_causal(shared)
                    begin_hand_state = (
                        read_hand_state_causal(shared)
                        if cfg.runtime.policy.hand_enabled
                        else None
                    )
                    begin_hand_tactile = (
                        read_hand_tactile_causal(shared)
                        if recorder is not None and cfg.runtime.policy.hand_enabled
                        else None
                    )
                    begin_issue = _begin_feedback_issue(
                        cfg,
                        vr_frame,
                        begin_hand_state,
                        begin_hand_tactile,
                        recording_enabled=recorder is not None,
                        now_monotonic_ns=begin_now_ns,
                    )
                    if begin_issue is not None:
                        print(f"\nB: {begin_issue} — cannot begin")
                        skip_control_tick = True
                        continue
                    wrist_pos = vr_frame["wrist_pos"]
                    wrist_quat_wxyz = vr_frame["wrist_quat_wxyz"]
                    print(
                        "\nB: wrist_pose "
                        f"pos=[{' '.join(f'{value:.6f}' for value in wrist_pos)}] + "
                        f"wxyz=[{' '.join(f'{value:.6f}' for value in wrist_quat_wxyz)}]",
                        flush=True,
                    )
                    gc.collect()
                    if recorder is not None:
                        if not recorder.start_episode(
                            task_label=cfg.task_label, operator=cfg.operator
                        ):
                            print("  ⚠ 无法开始录制")
                            skip_control_tick = True
                            continue
                        recording_active = True
                        camera_freshness.reset(time.monotonic())
                        shared.is_recording.value = True
                        begin_message = (
                            f"\nB: 遥操作+录制开始  episode={recorder.frame_count}"
                        )
                    else:
                        shared.is_recording.value = False
                        begin_message = "\nB: 遥操作开始（未启用录制 capability）"
                    kb.drain_signal(ControlSignal.BEGIN)
                    if not _transition_or_fault(shared, SafetyState.RUNNING, "begin"):
                        stop_recording(
                            recorder,
                            recording_active,
                            save=False,
                            shared=shared,
                            reason="safety_transition_failed",
                        )
                        recording_active = False
                        break_loop = True
                        break
                    enter_pause("begin", start_new_run=True)
                    teleop_active = True
                    if controller.hand_enabled:
                        assert begin_hand_state is not None
                        seeded_qpos = seed_hand_retargeter(
                            controller.hand_retargeter,
                            np.asarray(begin_hand_state["qpos"][0], dtype=np.float64),
                        )
                        if seeded_qpos is not None:
                            controller.prev_hand_qpos = seeded_qpos
                    audio.play("begin")
                    print(begin_message)
                    limiter.reset()
                    skip_control_tick = True

            if break_loop or shared.error_state.value or shared.estop_request.value:
                break
            if reanchor_grid:
                limiter.reset()
                next_grid_ns = time.monotonic_ns() + grid_period_ns
                current_grid_anchor_ns = next_grid_ns
                continue
            if skip_control_tick or not grid_due:
                continue

            tick_result = run_control_grid_tick(
                controller,
                shared,
                cfg,
                grid_resources,
                teleop_active=teleop_active,
                recording_active=recording_active,
                pause_since_ns=pause_since_ns,
                pause_reason=pause_reason,
                arm_feedback_error_count=arm_feedback_error_count,
                hand_disconnected_at_s=hand_disconnected_at_s,
                loop_count=loop_count,
                observation_anchor_monotonic_ns=current_grid_anchor_ns,
            )
            recording_active = tick_result.recording_active
            arm_feedback_error_count = tick_result.arm_feedback_error_count
            hand_disconnected_at_s = tick_result.hand_disconnected_at_s
            if tick_result.pause_reason is not None:
                enter_pause(tick_result.pause_reason)
            if tick_result.pause_released:
                pause_since_ns = 0
                pause_reason = None
            if not tick_result.keep_running:
                break
    finally:
        if recording_active:
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
