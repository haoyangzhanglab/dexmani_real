"""Readable VR teleoperation experiment loop over shared-memory snapshots."""

from __future__ import annotations

import gc
import json
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.planning import (PlanningProfile, Pose, TeleopProfile,
                                   XArm7MotionPlanner, XArm7PlannerConfig)
from dexmani_real.planning.hand_kinematics import HandKinematics
from dexmani_real.planning.pose_utils import (normalize_quat_wxyz,
                                              quat_wxyz_to_rot6d,
                                              rot6d_to_quat_wxyz)
from dexmani_real.policy.loop_timing import StageTimer
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.policy.safety import (SafetyGate, advance_run_generation,
                                        send_command)
from dexmani_real.recording.io_process import RecorderClient
from dexmani_real.runtime.status import ComponentPhase, FaultCode
from dexmani_real.shm.shared_storage import (SharedStorage,
                                             publish_component_status)
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.audio_feedback import (AudioFeedback,
                                                update_motion_gate)
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.control_state import ControlHold
from dexmani_real.teleop.episode_samples import (_FRAME_IK_FAIL, _FRAME_OK,
                                                 _FRAME_RETARGET_FAIL,
                                                 _FRAME_SAFETY_REJECT,
                                                 _record_frame, _record_held,
                                                 _stop_recording)
from dexmani_real.teleop.hand_control import (_compute_hand_command,
                                              _get_raw_hand_command,
                                              _hand_ramp_frame_count,
                                              _reset_hand_retargeter,
                                              _sanitize_hand_command,
                                              _seed_hand_retargeter,
                                              _smoothstep_hand_ramp)
from dexmani_real.teleop.hand_retarget import (TAGHandRetargeter,
                                               XHandRetargeter,
                                               _tag_config_with_urdf)
from dexmani_real.teleop.keyboard import (ControlSignal, KeyboardHandler,
                                          validate_arm_feedback,
                                          validate_hand_feedback)
from dexmani_real.teleop.recording_session import (
    QuitRecordingDecision, await_quit_recording_decision)
from dexmani_real.teleop.safety import (_contact_stall_detected,
                                        _do_configured_teleop_home,
                                        _feedback_after_send,
                                        _reset_mapper_from_frames,
                                        _vr_after_send)
from dexmani_real.teleop.snapshot import (CameraFreshnessTracker,
                                          _read_arm_state, _read_camera_frame,
                                          _read_hand_state, _read_hand_tactile,
                                          _read_vr_frame)
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import ema_smooth_pose

logger = get_logger(__name__)

_END_AUDIO_GRACE_S = 2.0
_NS_PER_SECOND = 1_000_000_000
_VALIDATION_WARN_INTERVAL_S = 2.0
_ARM_FEEDBACK_WARN_INTERVAL_S = 3.0


def _load_vr_transform(path: Path) -> tuple[np.ndarray, str]:
    """Load the required calibrated VR-to-robot rotation."""
    if not path.is_file():
        raise FileNotFoundError(f"VR transform config not found: {path}")
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    transform = np.asarray(config["T_vr_to_robot"], dtype=np.float64)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("T_vr_to_robot must be a finite 3x3 matrix")
    return transform, str(config.get("theta_deg", "?"))


def _build_safety_gate(config: TeleopConfig) -> SafetyGate:
    """Build the teleoperation safety gate from control-domain limits."""
    return SafetyGate(
        arm_joint_lower_rad=tuple(config.joint_limit_lower),
        arm_joint_upper_rad=tuple(config.joint_limit_upper),
        hand_joint_lower_rad=tuple(config.hand_qpos_lower_rad),
        hand_joint_upper_rad=tuple(config.hand_qpos_upper_rad),
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
) -> tuple[FaultCode, str] | None:
    """Classify terminal policy state without losing an e-stop or sticky fault."""
    if estop_request:
        return FaultCode.ESTOP, "policy exited after e-stop request"
    if error_state or safety_fault:
        return FaultCode.COMMAND_INVALID, "policy exited with sticky fault"
    return None


def _start_keyboard(shared: SharedStorage) -> KeyboardHandler | None:
    """Start the required operator input boundary, failing closed on startup errors."""
    keyboard = KeyboardHandler(estop_callback=lambda: setattr(shared.estop_request, "value", True))
    try:
        keyboard.start()
    except Exception:
        logger.error("teleop_loop: keyboard startup failed", exc_info=True)
        shared.error_state.value = True
        publish_component_status(
            shared,
            "policy",
            ComponentPhase.FAULT,
            fault_code=FaultCode.STARTUP_FAILED,
            detail="keyboard startup failed",
        )
        return None
    return keyboard


def teleop_loop(shared: SharedStorage, config: TeleopConfig | None = None) -> None:
    """Teleoperation process entry point used by ``collect_teleop.py``.

    Reads from rings (vr, arm_state, hand_state, camera), writes actions
    to queues/rings (arm_action_q, hand_cmd_ring), owns recording.
    """
    from dexmani_real.robot.safety import SafetyState, transition

    cfg = config or TeleopConfig()

    def _transition_or_fault(new_state: SafetyState, reason: str) -> bool:
        if transition(shared, new_state):
            return True
        logger.error("teleop_loop: safety transition to %s failed during %s", new_state.name, reason)
        shared.error_state.value = True
        return False

    publish_component_status(shared, "policy", ComponentPhase.LOADING)
    ctrl_dt = 1.0 / cfg.runtime.policy.control_hz
    arm_cmd_max_step_rad = float(np.deg2rad(cfg.joint_max_speed_deg_s)) * ctrl_dt
    joint_lower_rad = np.asarray(cfg.runtime.arm.joint_limit_lower, dtype=np.float64)
    joint_upper_rad = np.asarray(cfg.runtime.arm.joint_limit_upper, dtype=np.float64)
    hand_home_qpos_rad = np.deg2rad(np.asarray(cfg.hand_home_qpos_deg, dtype=np.float64))
    hand_qpos_lower_rad = np.asarray(cfg.hand_qpos_lower_rad, dtype=np.float64)
    hand_qpos_upper_rad = np.asarray(cfg.hand_qpos_upper_rad, dtype=np.float64)

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
                workspace_bounds=np.asarray(cfg.workspace_bounds, dtype=np.float64),
            ),
            planning_profile=PlanningProfile(),
            teleop_profile=TeleopProfile(
                max_pose_error_pos_m=cfg.runtime.policy.ik_max_pose_error_pos_m,
                max_pose_error_rot_rad=cfg.runtime.policy.ik_max_pose_error_rot_rad,
                nullspace_step_size_deg=cfg.runtime.policy.ik_nullspace_step_rate_deg_s / cfg.runtime.policy.control_hz,
            ),
            hand_dof=True,  # 19-DOF — hand geometry follows set_hand_qpos()
            static_boxes=cfg.static_collision_boxes,
            table=cfg.table_collision,
        )

        vr_config_path = Path(__file__).resolve().parents[2] / cfg.vr_transform_path
        vr_to_robot, vr_heading_deg = _load_vr_transform(vr_config_path)
        logger.info("VR transform loaded: theta=%s°", vr_heading_deg)

        arm_mapper = ArmWristMapper(
            pos_scale=cfg.vr_pos_scale,
            rot_scale=cfg.vr_rot_scale,
            vr_to_base_rot=vr_to_robot,
            T_vr_to_robot=vr_to_robot,
            max_delta_rot_rad=cfg.vr_max_delta_rot_rad,
            base_to_world_rot=np.eye(3, dtype=np.float64),
        )

        gate = _build_safety_gate(cfg)
        # SafetyGate validates well-formedness, joint limits, and workspace
        # only.  Collision and transition checks were removed (2026-08-12);
        # xArm Mode 6 firmware provides the hardware backstop (C22/C31/C24).
        # Collision-free homing paths are planned independently through
        # plan_joint_home_path / plan_band_alignment_path.
        gate.workspace_check = planner.is_workspace_segment_safe

        recorder = RecorderClient(shared) if cfg.runtime.policy.recording_enabled else None
    except Exception:
        logger.error("teleop_loop: init failed", exc_info=True)
        shared.error_state.value = True
        publish_component_status(
            shared,
            "policy",
            ComponentPhase.FAULT,
            fault_code=FaultCode.STARTUP_FAILED,
            detail="VR policy initialization failed",
        )
        return

    kb = _start_keyboard(shared)
    if kb is None:
        return

    def _keyboard_estop_requested() -> bool:
        return kb.estop_latched or not kb.healthy

    audio = AudioFeedback()

    hand_retargeter: TAGHandRetargeter | XHandRetargeter | None = None
    hand_available = False
    _hand_disconnected_at: float | None = None  # monotonic timestamp of first bad frame
    _hand_ramp_start: np.ndarray | None = None
    _hand_ramp_step = 0
    _hand_ramp_total_frames = _hand_ramp_frame_count(cfg.runtime.policy.hand_ramp_duration_s, cfg.runtime.policy.control_hz)

    def _try_init_hand_retargeter() -> bool:
        """Lazily initialize hand_retargeter if not already created."""
        nonlocal hand_retargeter
        if hand_retargeter is not None:
            return True
        try:
            if cfg.runtime.policy.hand_retargeting_type == "tag":
                hand_retargeter = TAGHandRetargeter(
                    hand_type="right",
                    smoothing_alpha=cfg.runtime.policy.hand_output_smoothing_alpha,
                    qpos_lower_rad=cfg.hand_qpos_lower_rad,
                    qpos_upper_rad=cfg.hand_qpos_upper_rad,
                    fingertip_link_names=cfg.runtime.hand.fingertip_link_names,
                    tag_config=_tag_config_with_urdf(cfg.tag_retargeting_config, cfg.hand_urdf_path),
                )
            else:
                hand_retargeter = XHandRetargeter(
                    hand_type="right",
                    retargeting_type=cfg.runtime.policy.hand_retargeting_type,
                    smoothing_alpha=cfg.runtime.policy.hand_output_smoothing_alpha,
                )
            logger.info("Hand retargeter ready (type=%s)", cfg.runtime.policy.hand_retargeting_type)
            return True
        except Exception:
            logger.error("Hand retargeter initialization failed", exc_info=True)
            hand_retargeter = None
            return False

    def _init_and_seed_hand_retargeter() -> np.ndarray | None:
        """Lazy-init retargeter and seed NLP warm-start from hardware qpos.

        Returns the seeded qpos (for updating ``prev_hand_qpos``) or None.
        """
        if not cfg.runtime.policy.hand_enabled:
            return None
        if not _try_init_hand_retargeter():
            return None
        hs = _read_hand_state(shared)
        qpos = hs["qpos"][0] if _hand_feedback_issue(hs) is None and hs is not None else None
        return _seed_hand_retargeter(hand_retargeter, qpos)

    def _hand_feedback_issue(state: np.ndarray | None) -> str | None:
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
            max_age_s=cfg.hand_heartbeat_timeout_s,
            qpos=np.asarray(state["qpos"][0]),
        )

    _hand_fk: HandKinematics | None = None
    _T_eef_handbase_pos = np.array(cfg.runtime.hand.T_eef_handbase_pos_xyz, dtype=np.float64)
    _T_eef_handbase_quat_wxyz = np.array(cfg.runtime.hand.T_eef_handbase_quat_wxyz, dtype=np.float64)
    if cfg.runtime.policy.recording_enabled and cfg.hand_urdf_path:
        try:
            _hand_fk = HandKinematics(cfg.hand_urdf_path, list(cfg.runtime.hand.fingertip_link_names))
            if _hand_fk.is_ready():
                logger.info("Hand FK ready")
            else:
                logger.warning("Hand FK not ready — fingertips will be NaN")
        except Exception:
            logger.warning("Hand FK initialization failed", exc_info=True)

    logger.info("Teleop: waiting for enabled capabilities...")
    _ready_events = [("arm", shared.arm_ready), ("vr", shared.vr_ready)]
    if cfg.runtime.policy.recording_enabled:
        _ready_events.append(("camera", shared.camera_ready))
        _ready_events.append(("recorder", shared.recorder_ready))
    if cfg.runtime.policy.hand_enabled:
        _ready_events.insert(1, ("hand", shared.hand_ready))
    for name, ev in _ready_events:
        timeout_s = float(cfg.readiness_timeouts_s[name])
        if not ev.wait(timeout=timeout_s):
            logger.error("Teleop: %s startup timeout (%.1fs)", name, timeout_s)
            shared.error_state.value = True
            publish_component_status(
                shared,
                "policy",
                ComponentPhase.FAULT,
                fault_code=FaultCode.STARTUP_FAILED,
                detail=f"startup timeout waiting for {name}",
            )
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
            publish_component_status(
                shared,
                "policy",
                ComponentPhase.FAULT,
                fault_code=FaultCode.STARTUP_FAILED,
                detail=initial_hand_issue,
            )
            kb.stop()
            return
        hand_available = True
        if not _try_init_hand_retargeter():
            publish_component_status(
                shared,
                "policy",
                ComponentPhase.FAULT,
                fault_code=FaultCode.STARTUP_FAILED,
                detail="hand retargeter initialization failed",
            )
            shared.error_state.value = True
            kb.stop()
            return

    # Publish before policy_ready so Main never observes a ready worker with a zero heartbeat.
    shared.policy_heartbeat_s.value = time.monotonic()
    shared.policy_ready.set()
    publish_component_status(shared, "policy", ComponentPhase.READY)

    home_qpos = np.array(cfg.arm_home_qpos, dtype=np.float64)
    arm_state = _read_arm_state(shared)
    hand_state = _read_hand_state(shared)
    if arm_state is None:
        arm_qpos = home_qpos.copy()
    else:
        arm_qpos = arm_state["qpos"][0].copy()
    prev_qpos_cmd = arm_qpos.copy()
    prev_hand_qpos = (
        hand_state["qpos"][0].copy()
        if hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0]))
        else hand_home_qpos_rad.copy()
    )
    planner.set_hand_qpos(prev_hand_qpos)  # sync hand pose for collision checks

    teleop_active = False
    recording_active = False
    recording_paused = False
    _begin_audio_gate_deadline_s: float | None = None
    _ignore_begin_audio_until_silent = False
    _control_hold = ControlHold()
    _hold_sent_at_s: float | None = None
    _reanchor_pending_reason: str | None = None

    # First Q leaves teleop active for an optional home; the next Q exits.
    quit_pending = False
    post_teleop_deadline = 0.0

    ema_prev_pos: np.ndarray | None = None
    ema_prev_quat: np.ndarray | None = None

    # Coordinator duties run at coordinator_hz; observations, actions, and
    # recording remain on the configured control_hz grid.
    limiter = RateManager(cfg.runtime.policy.coordinator_hz)
    _grid_period_ns = int(round(_NS_PER_SECOND / cfg.runtime.policy.control_hz))
    _next_grid_ns = time.monotonic_ns() + _grid_period_ns
    _current_grid_anchor_ns = _next_grid_ns
    _pending_controls: list[ControlSignal] = []
    stage_timer = StageTimer(window=cfg.status_every)
    _validate_warn = ThrottledWarner(interval_s=_VALIDATION_WARN_INTERVAL_S)
    _arm_feedback_warn = ThrottledWarner(interval_s=_ARM_FEEDBACK_WARN_INTERVAL_S)
    loop_count = 0
    error_count = 0
    _last_target_eef_pos = np.full(3, np.nan)  # last valid IK target (held frame continuity)
    _last_target_eef_rot6d = np.full(6, np.nan)
    _camera_freshness = CameraFreshnessTracker(
        max_age_s=cfg.camera_max_frame_age_s,
        abort_after_s=cfg.camera_recording_stall_abort_s,
    )

    def _enter_hand_feedback_pause(issue: str) -> None:
        """Invalidate pending motion without publishing from unhealthy hand feedback."""
        nonlocal _reanchor_pending_reason, recording_paused
        nonlocal ema_prev_pos, ema_prev_quat, _hand_ramp_start, _hand_ramp_step

        run_generation = advance_run_generation(shared)
        _control_hold.pause_without_candidate("hand_feedback")
        _reanchor_pending_reason = "hand_recovered"
        recording_paused = True
        arm_mapper.clear()
        ema_prev_pos = ema_prev_quat = None
        _hand_ramp_start = None
        _hand_ramp_step = 0
        logger.warning(
            "teleop_loop: hand feedback pause invalidated run=%d without publishing: %s",
            run_generation,
            issue,
        )

    def _enter_measured_hold(reason: str) -> bool:
        """Invalidate old endpoints and publish one measured arm-only hold."""
        nonlocal _reanchor_pending_reason
        nonlocal prev_qpos_cmd, ema_prev_pos, ema_prev_quat
        nonlocal _hand_ramp_start, _hand_ramp_step

        if _control_hold.active:
            _control_hold.relabel(reason)
            _reanchor_pending_reason = reason
            if _control_hold.candidate is None:
                logger.error("teleop_loop: cannot enter %s without a measured arm hold candidate", reason)
                return False
            return True

        latest_arm = _read_arm_state(shared)
        latest_arm_issue = _arm_feedback_issue(
            latest_arm,
            now_monotonic_ns=time.monotonic_ns(),
            max_age_s=cfg.runtime.policy.arm_state_stale_threshold_s,
        )
        if latest_arm_issue is not None:
            logger.error("teleop_loop: cannot enter %s hold: %s", reason, latest_arm_issue)
            return False
        assert latest_arm is not None
        measured_arm = np.asarray(latest_arm["qpos"][0], dtype=np.float64)

        if hand_available:
            latest_hand = _read_hand_state(shared)
            latest_hand_issue = _hand_feedback_issue(latest_hand)
            if latest_hand_issue is not None:
                logger.error(
                    "teleop_loop: cannot enter %s hold with unhealthy hand feedback: %s",
                    reason,
                    latest_hand_issue,
                )
                return False
            assert latest_hand is not None

        run_generation = advance_run_generation(shared)
        candidate = _safe_joint_publish(
            shared,
            measured_arm,
            None,
            is_hold=True,
            timeout=cfg.runtime.policy.action_prepare_timeout_s,
            safety_gate=gate,
        )
        if candidate is None:
            logger.error("teleop_loop: failed to publish %s hold after advancing to run=%d", reason, run_generation)
            return False

        _control_hold.begin(reason, candidate, deadline_s=time.monotonic() + cfg.runtime.policy.action_apply_timeout_s)
        _hold_sent_at_s = time.monotonic()
        _reanchor_pending_reason = reason
        prev_qpos_cmd = np.asarray(candidate.arm_qpos, dtype=np.float64).copy()
        arm_mapper.clear()
        ema_prev_pos = ema_prev_quat = None
        _hand_ramp_start = None
        _hand_ramp_step = 0
        logger.info(
            "teleop_loop: %s hold published (run=%d action_id=%d)",
            reason,
            run_generation,
            candidate.action_id,
        )
        return True

    def _handoff_control_hold_to_home() -> None:
        """Stop reviewing a hold that the homing protocol will invalidate."""
        nonlocal _reanchor_pending_reason, recording_paused

        handoff_state, hold_reason = _control_hold.clear()
        if handoff_state != "idle":
            logger.info(
                "teleop_loop: homing supersedes %s %s hold candidate",
                handoff_state,
                hold_reason,
            )
        _reanchor_pending_reason = None
        recording_paused = False

    def _complete_reanchor(
        current_arm_state: np.ndarray,
        current_vr_frame: dict[str, Any],
        current_hand_state: np.ndarray | None,
    ) -> bool:
        """Reset arm/hand temporal state; caller holds the re-anchor grid frame."""
        nonlocal ema_prev_pos, ema_prev_quat, prev_hand_qpos
        nonlocal _hand_ramp_start, _hand_ramp_step
        if not _reset_mapper_from_frames(arm_mapper, current_arm_state, current_vr_frame):
            _validate_warn("teleop_loop: re-anchor inputs invalid — remaining held")
            return False
        ema_prev_pos = ema_prev_quat = None
        if hand_available and current_hand_state is not None and np.all(np.isfinite(current_hand_state["qpos"][0])):
            prev_hand_qpos = np.asarray(current_hand_state["qpos"][0], dtype=np.float64).copy()
        _hand_ramp_start = prev_hand_qpos.copy() if hand_available else None
        _hand_ramp_step = 0
        _reset_hand_retargeter(hand_retargeter, prev_hand_qpos.copy() if hand_available else None)
        return True

    # SIGTERM is intercepted so RecorderIO can finish its episode transaction.
    _sigterm_requested = False

    def _on_sigterm(signum: int, frame: object) -> None:
        nonlocal _sigterm_requested
        _sigterm_requested = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    logger.info(
        "Teleop: entering coordinator loop @ %.0f Hz (observation/action grid %.0f Hz)",
        cfg.runtime.policy.coordinator_hz,
        cfg.runtime.policy.control_hz,
    )

    try:
        while shared.is_running.value and not _sigterm_requested:
            shared.policy_heartbeat_s.value = time.monotonic()
            limiter.wait()
            if _control_hold.application_pending:
                _hold_result = _control_hold.observe_delivery(
                    _hold_sent_at_s,
                    now_s=time.monotonic(),
                )
                if _hold_result == "applied":
                    logger.info(
                        "teleop_loop: %s hold applied (action_id=%d)",
                        _control_hold.reason,
                        _control_hold.candidate.action_id if _control_hold.candidate else 0,
                    )
                elif _hold_result == "failed":
                    logger.error(
                        "teleop_loop: %s hold delivery timed out",
                        _control_hold.reason,
                    )
                    shared.error_state.value = True
                    break

            if recorder is not None:
                _stop_result = recorder.poll_stop()
                if _stop_result.done and _stop_result.path is not None:
                    if _stop_result.error:
                        print(f"  ⚠ 保存失败 ({_stop_result.error}): {_stop_result.path}")
                    elif _stop_result.success:
                        print(f"  录制已保存: {_stop_result.path}  ({_stop_result.frame_count} 帧)")
                    gc.collect()

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
                        _handoff_control_hold_to_home()
                        prev_hand_qpos = _do_configured_teleop_home(
                            shared,
                            cfg,
                            hand_available=hand_available,
                            prev_hand_qpos=prev_hand_qpos,
                            planner=planner,
                            audio=audio,
                            estop_requested=_keyboard_estop_requested,
                        )
                        limiter.reset()
                        print("  [Q] quit", flush=True)

                    elif _sig in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
                        if _sig == ControlSignal.EMERGENCY_STOP:
                            shared.estop_request.value = True
                        elif _control_hold.application_pending:
                            print("  measured hold is still applying; Q will be accepted after its ACK", flush=True)
                            continue
                        else:
                            shared.quit_requested.value = True
                        break

                if shared.estop_request.value or shared.quit_requested.value or not shared.is_running.value:
                    break

                if time.perf_counter() > post_teleop_deadline:
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
                    if teleop_active and not _enter_measured_hold("quit"):
                        shared.error_state.value = True
                        break
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
                            print("  已保存")
                        elif decision is QuitRecordingDecision.DISCARD:
                            audio.play("discard")
                            _stop_recording(recorder, recording_active, save=False, shared=shared)
                            recording_active = False
                            print("  已丢弃")
                        elif decision is QuitRecordingDecision.ESTOP:
                            audio.play("emergency")
                            _stop_recording(recorder, recording_active, save=False, shared=shared)
                            recording_active = False
                        elif decision is QuitRecordingDecision.TIMEOUT:
                            audio.play("discard")
                            _stop_recording(recorder, recording_active, save=False, shared=shared)
                            recording_active = False
                            print("  超时，默认丢弃")

                        if decision is QuitRecordingDecision.SAVE_AND_HOME and shared.is_running.value:
                            audio.play("home")
                            ema_prev_pos = ema_prev_quat = None
                            _handoff_control_hold_to_home()
                            prev_hand_qpos = _do_configured_teleop_home(
                                shared,
                                cfg,
                                hand_available=hand_available,
                                prev_hand_qpos=prev_hand_qpos,
                                planner=planner,
                                audio=audio,
                                estop_requested=_keyboard_estop_requested,
                                arm_mapper=arm_mapper,
                                hand_retargeter=hand_retargeter,
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
                    ema_prev_pos = ema_prev_quat = None
                    _handoff_control_hold_to_home()
                    prev_hand_qpos = _do_configured_teleop_home(
                        shared,
                        cfg,
                        hand_available=hand_available,
                        prev_hand_qpos=prev_hand_qpos,
                        planner=planner,
                        audio=audio,
                        estop_requested=_keyboard_estop_requested,
                        arm_mapper=arm_mapper,
                        hand_retargeter=hand_retargeter,
                    )
                    limiter.reset()
                    skip_rest = True

                elif sig in (ControlSignal.STOP, ControlSignal.DISCARD):
                    save_episode = sig is ControlSignal.STOP
                    stop_reason = "stop" if save_episode else "discard"
                    print("\nS: 停止录制" if save_episode else "\nD: 丢弃录制")
                    audio.play("save" if save_episode else "discard")
                    if teleop_active and not _enter_measured_hold(stop_reason):
                        shared.error_state.value = True
                        break
                    _stop_recording(recorder, recording_active, save=save_episode, shared=shared)
                    recording_active = False
                    teleop_active = False
                    if not _transition_or_fault(SafetyState.ARMED, stop_reason):
                        break
                    skip_rest = True

                elif sig == ControlSignal.PAUSE:
                    if teleop_active:
                        if not _enter_measured_hold("pause"):
                            shared.error_state.value = True
                            break
                        teleop_active = False
                        recording_paused = True
                        if not _transition_or_fault(SafetyState.ARMED, "pause"):
                            break
                    else:
                        if shared.safety_state.value != SafetyState.ARMED:
                            print(f"\nC: safety_state={shared.safety_state.value} — must be ARMED to resume")
                            skip_rest = True
                            continue
                        if not _transition_or_fault(SafetyState.RUNNING, "resume"):
                            break
                        teleop_active = True
                        _reanchor_pending_reason = "resume"
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
                    teleop_active = True
                    publish_component_status(shared, "policy", ComponentPhase.RUNNING)
                    recording_paused = False
                    _seeded = _init_and_seed_hand_retargeter()
                    if _seeded is not None:
                        prev_hand_qpos = _seeded
                    audio.play("begin")
                    _begin_audio_gate_deadline_s = time.monotonic() + max(
                        0.0, cfg.runtime.policy.begin_motion_gate_timeout_s - ctrl_dt
                    )
                    _ignore_begin_audio_until_silent = False
                    _reanchor_pending_reason = "begin"
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
                if arm_feedback_fault:
                    logger.error("teleop_loop: arm feedback fault: %s", arm_issue)
                    shared.error_state.value = True
                    break
                continue
            assert arm_state is not None  # validation above proved availability
            arm_qpos = arm_state["qpos"][0].copy()

            vr_frame = _read_vr_frame(shared, anchor_monotonic_ns=_current_grid_anchor_ns)
            vr_stale = vr_frame is None or (
                (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) > cfg.vr_stale_threshold_s * 1e9
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
                        cfg.camera_recording_stall_abort_s,
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
                if teleop_active and _control_hold.reason != "hand_feedback":
                    _enter_hand_feedback_pause(hand_issue)
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
                    "Hand feedback recovered after %.1fs — requiring measured hold and re-anchor", unhealthy_duration_s
                )
                if teleop_active:
                    _control_hold.clear()
                    if not _enter_measured_hold("hand_recovered"):
                        shared.error_state.value = True
                        break
                    recording_paused = True

            if loop_count % cfg.status_every == 0:
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

            if teleop_active and vr_stale and not _control_hold.active:
                if not _enter_measured_hold("vr_stale"):
                    shared.error_state.value = True
                    break

            if not teleop_active or vr_stale or _control_hold.active:
                # A stale-input/resume release is legal only after the exact
                # coordinated measured hold reached both enabled SDKs and a
                # fresh VR frame can be anchored to current measured FK.
                if (
                    teleop_active
                    and not vr_stale
                    and _control_hold.active
                    and _control_hold.applied
                    and _control_hold.candidate is not None
                    and _hold_sent_at_s is not None
                    and _vr_after_send(vr_frame, _hold_sent_at_s)
                    and _feedback_after_send(
                        arm_state,
                        hand_state,
                        _control_hold.candidate,
                        _hold_sent_at_s,
                    )
                ):
                    assert vr_frame is not None  # _vr_after_send() proved the frame exists
                    if _complete_reanchor(arm_state, vr_frame, hand_state):
                        logger.info("teleop_loop: released %s hold after fresh re-anchor", _control_hold.reason)
                        _control_hold.clear()
                        _reanchor_pending_reason = None
                        recording_paused = False
                if recording_active and not recording_paused:
                    _record_held(
                        recorder,
                        arm_state,
                        prev_qpos_cmd,
                        prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        arm_qpos_sent=prev_qpos_cmd.copy(),
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                        hand_fk=_hand_fk,
                        T_eef_handbase_pos=_T_eef_handbase_pos,
                        T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                        observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                        shared=shared,
                        action_candidate=_control_hold.take_record_candidate(),
                    )
                prev_qpos_cmd = arm_qpos.copy()
                ema_prev_pos = ema_prev_quat = None
                continue

            # Hold during state-transition voice prompts.  The begin cue is
            # special: it may block motion only for a bounded interval, then it
            # continues playing in the background.  Other safety/state cues keep
            # their existing full-duration gate.
            _audio_playing = audio.is_playing
            _hold_for_audio, _begin_audio_gate_deadline_s, _ignore_begin_audio_until_silent = update_motion_gate(
                audio_playing=_audio_playing,
                begin_deadline_s=_begin_audio_gate_deadline_s,
                ignore_begin_until_silent=_ignore_begin_audio_until_silent,
                now_s=time.monotonic(),
            )
            if _hold_for_audio:
                prev_qpos_cmd = arm_qpos.copy()
                # Refresh prev_hand_qpos from current hardware state so
                # the hand ramp starts from the actual joint position.
                if hand_available and hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0])):
                    prev_hand_qpos = hand_state["qpos"][0].copy()
                if _reanchor_pending_reason is None:
                    _reanchor_pending_reason = "audio_gate"
                continue

            if _reanchor_pending_reason is not None:
                # This flag is set directly by begin/resume/stale state changes;
                # audio is only an optional UX gate and cannot suppress reset.
                if vr_frame is None or not _complete_reanchor(arm_state, vr_frame, hand_state):
                    prev_qpos_cmd = arm_qpos.copy()
                    ema_prev_pos = ema_prev_quat = None
                    continue
                logger.info("teleop_loop: completed %s re-anchor", _reanchor_pending_reason)
                _reanchor_pending_reason = None
                recording_paused = False
                # Keep the re-anchor grid frame held.  Active mapping resumes
                # on the next configured grid slot from a zero reset-relative delta.
                prev_qpos_cmd = arm_qpos.copy()
                continue

            if vr_frame is None:
                logger.warning("teleop_loop: vr_frame is None after vr_stale check — holding")
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        prev_qpos_cmd,
                        prev_hand_qpos,
                        None,  # vr_frame
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        arm_qpos_sent=prev_qpos_cmd.copy(),
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
                published_hold = _safe_arm_queue_put(
                    shared,
                    {"qpos": prev_qpos_cmd.copy(), "is_hold": True},
                    timeout=cfg.runtime.policy.action_prepare_timeout_s,
                    observation_id=int(vr_frame["ring_sequence"]),
                    observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                    safety_gate=gate,
                )
                if published_hold is None:
                    shared.error_state.value = True
                    break
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        prev_qpos_cmd,
                        prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        arm_qpos_sent=prev_qpos_cmd.copy(),
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

            if ema_prev_pos is not None:
                if ema_prev_quat is None:
                    logger.warning("teleop_loop: ema_prev_quat is None but ema_prev_pos is set — skipping EMA")
                else:
                    target_pos, target_quat = ema_smooth_pose(
                        target_pos,
                        target_quat,
                        ema_prev_pos,
                        ema_prev_quat,
                        cfg.ema_alpha_pos,
                        cfg.ema_alpha_rot,
                    )

            target_pos_before_clamp = target_pos.copy()
            for axis in range(3):
                lo, hi = cfg.workspace_bounds[axis]
                target_pos[axis] = np.clip(target_pos[axis], lo, hi)
            policy_map_time_ms = (time.perf_counter() - _map_t0) * 1000.0
            stage_timer.mark("map")

            # Intentional tabletop contact is allowed. If a downward target has
            # accumulated substantial joint error while the arm is no longer
            # closing that error, discard the buried target and re-anchor VR at
            # the measured pose. This stops continued pushing without a table
            # exclusion zone or any speed/acceleration change.
            if (
                cfg.runtime.policy.contact_stall_enabled
                and arm_state is not None
                and _contact_stall_detected(
                    arm_qpos,
                    arm_state["qvel"][0],
                    prev_qpos_cmd,
                    arm_state["eef_pos"][0],
                    target_pos,
                    table_z_surface_m=cfg.contact_stall_table_z_surface_m,
                    table_context_height_m=cfg.runtime.policy.contact_stall_table_context_height_m,
                    min_downward_target_m=cfg.runtime.policy.contact_stall_min_downward_target_m,
                    tracking_error_rad=cfg.runtime.policy.contact_stall_tracking_error_rad,
                    max_closing_speed_rad_s=cfg.runtime.policy.contact_stall_max_closing_speed_rad_s,
                )
            ):
                command_error = prev_qpos_cmd - arm_qpos
                closing_speed = float(
                    np.dot(arm_state["qvel"][0], command_error) / max(np.linalg.norm(command_error), 1e-12)
                )
                logger.warning(
                    "teleop_loop: downward contact stall — resync measured pose "
                    "(tracking_err=%.3frad closing_speed=%.3frad/s eef_z=%.3fm)",
                    float(np.max(np.abs(command_error))),
                    closing_speed,
                    float(arm_state["eef_pos"][0][2]),
                )
                hold_qpos = arm_qpos.copy()
                arm_mapper.reset(
                    wrist_pos=vr_frame["wrist_pos"],
                    wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
                    eef_pos=arm_state["eef_pos"][0],
                    eef_quat_wxyz=rot6d_to_quat_wxyz(arm_state["eef_rot6d"][0]),
                )
                ema_prev_pos = ema_prev_quat = None
                prev_qpos_cmd = hold_qpos
                published_hold = _safe_arm_queue_put(
                    shared,
                    {"qpos": hold_qpos, "is_hold": True},
                    timeout=cfg.runtime.policy.action_prepare_timeout_s,
                    observation_id=int(vr_frame["ring_sequence"]),
                    observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                    safety_gate=gate,
                )
                if published_hold is None:
                    shared.error_state.value = True
                    break
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        hold_qpos,
                        prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        frame_status=_FRAME_SAFETY_REJECT,
                        arm_qpos_sent=hold_qpos.copy(),
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

            planner.set_hand_qpos(prev_hand_qpos)  # sync hand pose for collision checks
            target_pose = Pose(p=target_pos, q=target_quat)
            _ik_t0 = time.perf_counter()
            ik_result = planner.solve_teleop_ik(target_pose, arm_qpos, prev_qpos_cmd)
            ik_solve_time_ms = (time.perf_counter() - _ik_t0) * 1000.0
            stage_timer.mark("ik")

            # Compute hand retargeting before IK outcome handling. Hand-only motion
            # remains allowed on IK failure; the hold arm publishes with well-formedness + joint-limit validation through SafetyGate.
            _hand_retarget_t0 = time.perf_counter()
            hand_cmd, retarget_ok = _compute_hand_command(
                hand_retargeter,
                vr_frame,
                prev_hand_qpos,
                hand_available,
            )
            hand_retarget_time_ms = (time.perf_counter() - _hand_retarget_t0) * 1000.0
            hand_cmd_raw = _get_raw_hand_command(hand_retargeter, hand_cmd, retarget_ok)
            # Rate-independent smoothstep ramp from the measured pose captured
            # at hold exit.  The last configured frame reaches the live target
            # exactly, avoiding an extra one-frame tail.
            if _hand_ramp_start is not None and _hand_ramp_step < _hand_ramp_total_frames:
                hand_cmd = _smoothstep_hand_ramp(
                    _hand_ramp_start,
                    hand_cmd,
                    _hand_ramp_step,
                    _hand_ramp_total_frames,
                )
                _hand_ramp_step += 1
                if _hand_ramp_step >= _hand_ramp_total_frames:
                    _hand_ramp_start = None
            elif _hand_ramp_start is not None:
                _hand_ramp_start = None
                _hand_ramp_step = 0

            hand_start_qpos = prev_hand_qpos.copy()
            if hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0])):
                hand_start_qpos = np.asarray(hand_state["qpos"][0], dtype=np.float64).copy()
            hand_cmd_valid = True
            try:
                hand_cmd = _sanitize_hand_command(hand_cmd)
            except ValueError as exc:
                _validate_warn("teleop_loop: invalid hand command — holding: %s", exc)
                hand_cmd = prev_hand_qpos.copy()
                hand_cmd_valid = False
                retarget_ok = False

            if not ik_result.success or ik_result.qpos is None:
                # Arm is held; hand motion proceeds independently.
                # SafetyGate validates well-formedness + joint limits only for hand-only commands.
                if hand_available:
                    safe_hand_cmd = hand_cmd if hand_cmd_valid else None
                    published_candidate = _safe_joint_publish(
                        shared,
                        prev_qpos_cmd.copy(),
                        safe_hand_cmd,
                        is_hold=True,
                        timeout=cfg.runtime.policy.action_prepare_timeout_s,
                        observation_id=int(vr_frame["ring_sequence"]),
                        observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                        safety_gate=gate,
                    )
                    if published_candidate is None:
                        shared.error_state.value = True
                        break
                    if published_candidate.arm_qpos is not None:
                        prev_qpos_cmd = np.asarray(published_candidate.arm_qpos, dtype=np.float64)
                    if published_candidate.hand_qpos is not None:
                        prev_hand_qpos = np.asarray(published_candidate.hand_qpos, dtype=np.float64).copy()
                else:
                    published_candidate = _safe_arm_queue_put(
                        shared,
                        {"qpos": prev_qpos_cmd.copy(), "is_hold": True},
                        timeout=cfg.runtime.policy.action_prepare_timeout_s,
                        observation_id=int(vr_frame["ring_sequence"]),
                        observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                        safety_gate=gate,
                    )
                    if published_candidate is None:
                        shared.error_state.value = True
                        break
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        prev_qpos_cmd,
                        prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        frame_status=_FRAME_IK_FAIL,
                        retarget_ok=retarget_ok,
                        arm_qpos_sent=prev_qpos_cmd.copy(),
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

            # IK delta clamp
            arm_cmd_raw = np.asarray(ik_result.qpos, dtype=np.float64).copy()
            arm_cmd = arm_cmd_raw.copy()
            arm_cmd = prev_qpos_cmd + np.clip(arm_cmd - prev_qpos_cmd, -arm_cmd_max_step_rad, arm_cmd_max_step_rad)

            # Keep IK output inside the firmware-accepted joint limits.
            arm_cmd = np.clip(arm_cmd, joint_lower_rad, joint_upper_rad)

            # Pre-flight validation (NaN, health, workspace).
            _arm_ok = arm_state is not None and bool(arm_state["connected"][0])
            _reject = False
            _reject_reason = ""
            if not np.all(np.isfinite(arm_cmd)):
                _reject = True
                _reject_reason = "arm_cmd NaN/Inf"
            elif not hand_cmd_valid:
                _reject = True
                _reject_reason = "hand command validation failed"
            elif not _arm_ok:
                _reject = True
                _reject_reason = "arm disconnected"
            elif not planner.is_workspace_segment_safe(arm_qpos, arm_cmd):
                _reject = True
                _reject_reason = "final arm transition leaves workspace"
            if _reject:
                _validate_warn("teleop_loop: action rejected — %s", _reject_reason)
                published_hold = _safe_joint_publish(
                    shared,
                    prev_qpos_cmd.copy(),
                    None,
                    is_hold=True,
                    timeout=cfg.runtime.policy.action_prepare_timeout_s,
                    observation_id=int(vr_frame["ring_sequence"]),
                    observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                    safety_gate=gate,
                )
                if published_hold is None:
                    shared.error_state.value = True
                    break
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        prev_qpos_cmd,
                        prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        frame_status=_FRAME_SAFETY_REJECT,
                        retarget_ok=retarget_ok,
                        arm_qpos_sent=prev_qpos_cmd.copy(),
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

            # FAULT gate: do not send actions when system is in fault state.
            if shared.safety_state.value == SafetyState.FAULT:
                if recording_active:
                    _record_held(
                        recorder,
                        arm_state,
                        prev_qpos_cmd,
                        prev_hand_qpos,
                        vr_frame,
                        cam,
                        hand_state=hand_state,
                        hand_tactile=hand_tactile,
                        arm_qpos_sent=prev_qpos_cmd.copy(),
                        target_eef_pos=_last_target_eef_pos,
                        target_eef_rot6d=_last_target_eef_rot6d,
                        hand_fk=_hand_fk,
                        T_eef_handbase_pos=_T_eef_handbase_pos,
                        T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                        observation_anchor_monotonic_ns=_current_grid_anchor_ns,
                        shared=shared,
                    )
                continue
            published_candidate = _safe_joint_publish(
                shared,
                arm_cmd.copy(),
                hand_cmd.copy() if hand_available else None,
                timeout=cfg.runtime.policy.action_prepare_timeout_s,
                observation_id=int(vr_frame["ring_sequence"]),
                observation_anchor_monotonic_ns=int(vr_frame["local_recv_ns"]),
                safety_gate=gate,
            )
            if published_candidate is None:
                logger.error("teleop_loop: joint publish failed — actuator unresponsive")
                shared.error_state.value = True
                break
            stage_timer.mark("send")

            if published_candidate.arm_qpos is not None:
                arm_cmd = np.asarray(published_candidate.arm_qpos, dtype=np.float64)
            if published_candidate.hand_qpos is not None:
                hand_cmd = np.asarray(published_candidate.hand_qpos, dtype=np.float64)
            prev_qpos_cmd = arm_cmd.copy()
            prev_hand_qpos = hand_cmd.copy()
            ema_prev_pos = target_pos.copy()
            ema_prev_quat = target_quat.copy()

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
        if recorder is not None:
            if not recorder.join_stop(timeout=cfg.runtime.policy.quit_save_timeout_s):
                logger.error("teleop recorder did not finish before policy shutdown")
        kb.stop()
        audio.play("end")
        time.sleep(_END_AUDIO_GRACE_S)
        exit_fault = _policy_exit_fault(
            error_state=bool(shared.error_state.value),
            estop_request=bool(shared.estop_request.value),
            safety_fault=shared.safety_state.value == SafetyState.FAULT,
        )
        if exit_fault is not None:
            fault_code, detail = exit_fault
            publish_component_status(
                shared,
                "policy",
                ComponentPhase.FAULT,
                fault_code=fault_code,
                detail=detail,
            )
        else:
            publish_component_status(shared, "policy", ComponentPhase.STOPPED)
        logger.info("Teleop: loop exited")


def _safe_arm_queue_put(
    shared: SharedStorage,
    action,
    *,
    timeout: float,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    safety_gate: SafetyGate | None = None,
) -> ActionCandidate | None:
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
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("teleop_loop: rejected invalid arm action: %s", exc)
        return None


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
) -> ActionCandidate | None:
    """Validate through SafetyGate and publish via fire-and-forget send_command."""
    if safety_gate is None:
        logger.error("joint target rejected: SafetyGate is required")
        return None
    gate = safety_gate

    with shared.arm_command_seq.get_lock():
        action_id = int(shared.arm_command_seq.value) + 1
        shared.arm_command_seq.value = action_id
    now_ns = time.monotonic_ns()
    if observation_anchor_monotonic_ns is not None:
        anchor_ns = int(observation_anchor_monotonic_ns)
        if anchor_ns <= 0 or anchor_ns > now_ns:
            logger.warning("joint target rejected: invalid observation anchor")
            return None

    candidate = ActionCandidate(
        observation_id=action_id if observation_id is None else int(observation_id),
        run_generation=int(shared.run_generation.value),
        action_id=action_id,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=now_ns + int(float(shared.action_lead_time_s) * 1e9),
        valid_until_monotonic_ns=now_ns + int(0.5 * 1e9),
        arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
        hand_qpos=None if hand_qpos is None else np.asarray(hand_qpos, dtype=np.float64),
        is_hold=is_hold,
    )

    # Read current arm/hand feedback for the gate
    arm_result = shared.arm_state_ring.read_latest()
    if arm_result is None:
        logger.warning("joint target rejected: arm feedback unavailable")
        return None
    arm_record = arm_result[0][0]
    if not bool(arm_record["connected"]) or not bool(arm_record["state_valid"]):
        logger.warning("joint target rejected: arm feedback unhealthy")
        return None
    # ``arm_record`` is a scalar structured record, so qpos is already (7,).
    current_arm = np.asarray(arm_record["qpos"], dtype=np.float64)
    if current_arm.shape != (7,) or not np.all(np.isfinite(current_arm)):
        return None

    current_hand = np.zeros(12, dtype=np.float64)
    hand_result = shared.hand_state_ring.read_latest()
    if hand_result is not None:
        hand_record = hand_result[0][0]
        if bool(hand_record["connected"]) and bool(hand_record["state_valid"]):
            current_hand = np.asarray(hand_record["qpos"], dtype=np.float64)
    elif hand_qpos is not None:
        logger.warning("joint target rejected: hand feedback unavailable")
        return None

    ctrl_dt = 1.0 / float(shared.action_control_hz)
    gate_result = gate.validate(
        candidate,
        current_arm_qpos=current_arm,
        current_hand_qpos=current_hand,
        dt_s=ctrl_dt,
        run_generation=int(shared.run_generation.value),
    )
    if not gate_result.accepted or gate_result.candidate is None:
        logger.warning("joint target rejected by SafetyGate: %s", gate_result.reason)
        return None

    if not send_command(shared, gate_result.candidate, prepare_timeout_s=timeout):
        return None
    return gate_result.candidate


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
