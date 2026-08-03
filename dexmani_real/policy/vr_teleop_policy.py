"""Policy process — VR→IK pipeline, state machine, recording.

Runs as an independent mp.Process, exchanging data exclusively through SharedStorage.

Ref: ManiUniCon Policy process pattern.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real import ASSET_DIR
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.path_utils import plan_joint_home_path
from dexmani_real.planning.pose_utils import normalize_quat_wxyz, quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.config.defaults import arm, hand, policy
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.shm.shared_storage import (
    HAND_CMD_DTYPE, HOME_SENTINEL, SharedStorage,
    read_arm_state as _read_arm_state,
    read_hand_state as _read_hand_state,
    write_hand_cmd as _write_hand_cmd,
)
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.hand_retarget import XHandRetargeter
from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.policy.loop_timing import StageTimer
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import ema_smooth_pose

logger = get_logger(__name__)

# ── Frame quality codes (schema v11) ──
_FRAME_OK = 0
_FRAME_HELD = 1
_FRAME_IK_FAIL = 2
_FRAME_SAFETY_REJECT = 3
_FRAME_RETARGET_FAIL = 4


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PolicyConfig:
    """Policy process configuration — defaults from arm/hand/policy singletons."""

    # Control rate
    control_hz: float = field(default_factory=lambda: policy.control_hz)

    # Mode 6 firmware parameters (deg — matches CLI convention)
    joint_max_speed_deg_s: float = field(default_factory=lambda: arm.max_joint_velocity_deg_per_s)
    joint_max_acc_deg_s2: float = field(default_factory=lambda: arm.max_joint_acceleration_deg_per_s2)
    inner_loop_hz: float = field(default_factory=lambda: arm.loop_hz)

    # VR mapping
    vr_pos_scale: float = field(default_factory=lambda: policy.vr_mapping.pos_scale)
    vr_rot_scale: float = field(default_factory=lambda: policy.vr_mapping.rot_scale)
    vr_max_delta_rot_rad: float = field(default_factory=lambda: policy.vr_mapping.max_delta_rot_rad)
    vr_stale_threshold_s: float = field(default_factory=lambda: policy.vr_mapping.stale_threshold_s)
    # Workspace bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] (m)
    workspace_bounds: tuple = field(default_factory=lambda: policy.workspace.as_tuple())

    # Cartesian EMA smoothing (tuned at 16Hz)
    ema_alpha_pos: float = field(default_factory=lambda: policy.ema.alpha_pos)
    ema_alpha_rot: float = field(default_factory=lambda: policy.ema.alpha_rot)

    # Recording
    max_record_seconds: float = field(default_factory=lambda: policy.max_record_duration_s)
    min_record_seconds: float = field(default_factory=lambda: policy.min_record_duration_s)
    episodes_dir: str = field(default_factory=lambda: policy.episodes_dir)
    task_label: str = ""
    operator: str = ""

    # Status print interval (in control ticks)
    status_every: int = field(default_factory=lambda: policy.status_print_interval)

    # Error tolerance
    max_consecutive_errors: int = field(default_factory=lambda: policy.max_consecutive_errors)

    # Hand retargeting
    hand_enabled: bool = field(default_factory=lambda: policy.hand_enabled)
    hand_retargeting_type: str = field(default_factory=lambda: policy.hand_retargeting_type)
    hand_ramp_frames: int = field(default_factory=lambda: policy.hand_ramp_frame_count)
    hand_disconnect_timeout_s: float = field(default_factory=lambda: policy.hand_disconnect_timeout_s)


    # Joint-space hard limits — sourced from arm singleton via shared_storage.
    joint_limit_lower: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_lower)
    joint_limit_upper: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_upper)

    # Hand home position — sourced from hand singleton.
    hand_home_qpos_deg: tuple[float, ...] = field(default_factory=lambda: hand.home_qpos_deg)

    # VR transform config path (relative to repo root)
    vr_transform_path: str = "dexmani_real/config/vr_transform.json"


# ═══════════════════════════════════════════════════════════════════
# Policy loop
# ═══════════════════════════════════════════════════════════════════


def policy_loop(shared: SharedStorage, config: PolicyConfig | None = None) -> None:
    """Policy process entry point — called via mp.Process(target=policy_loop, ...).

    Reads from rings (vr, arm_state, hand_state, camera), writes actions
    to queues/rings (arm_action_q, hand_cmd_ring), owns recording.
    """
    from dexmani_real.robot.safety import SafetyState, transition

    cfg = config or PolicyConfig()
    ctrl_dt = 1.0 / cfg.control_hz
    arm_cmd_max_step_rad = float(np.deg2rad(cfg.joint_max_speed_deg_s)) * ctrl_dt
    JOINT_LO = np.array(cfg.joint_limit_lower, dtype=np.float64)
    JOINT_HI = np.array(cfg.joint_limit_upper, dtype=np.float64)
    HAND_HOME_QPOS = np.deg2rad(np.array(cfg.hand_home_qpos_deg, dtype=np.float64))

    # ── 1. Planner (MPlib + collision checking) ──
    try:
        urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
        srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf")
        planner = XArm7MotionPlanner(
            XArm7PlannerConfig(
                urdf_path=urdf_path,
                srdf_path=srdf_path,
                base_pose_world=Pose(
                    p=np.array([0.0, 0.0, 0.0]),
                    q=np.array([np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12)]),
                ),
            ),
            planning_profile=PlanningProfile(),
            teleop_profile=TeleopProfile(
                max_pose_error_pos_m=0.02,
                max_pose_error_rot_rad=np.deg2rad(5.0),
                nullspace_step_size_deg=1.0 * (50.0 / cfg.control_hz),
            ),
        )

        # ── 2. VR Arm Mapper ──
        _vr_cfg_path = Path(__file__).resolve().parents[2] / cfg.vr_transform_path
        if _vr_cfg_path.exists():
            with open(_vr_cfg_path) as _f:
                _vr_cfg = json.load(_f)
            _T_vr_fixed = np.array(_vr_cfg["T_vr_to_robot"], dtype=np.float64)
            logger.info("VR transform loaded: theta=%s°", _vr_cfg.get("theta_deg", "?"))
        else:
            _T_vr_fixed = np.eye(3, dtype=np.float64)
            logger.warning("VR transform config not found, using identity")

        arm_mapper = ArmWristMapper(
            pos_scale=cfg.vr_pos_scale,
            rot_scale=cfg.vr_rot_scale,
            vr_to_base_rot=_T_vr_fixed,
            T_vr_to_robot=_T_vr_fixed,
            max_delta_rot_rad=cfg.vr_max_delta_rot_rad,
            base_to_world_rot=Rotation.from_quat(
                [0.0, 0.0, np.sin(np.pi / 12), np.cos(np.pi / 12)]
            ).as_matrix(),
        )

        # ── 3. Recorder ──
        _repo_root = Path(__file__).resolve().parents[2]
        recorder = EpisodeRecorder(
            data_dir=str(_repo_root / cfg.episodes_dir),
            max_frames=int(round(cfg.max_record_seconds * cfg.control_hz)),
            control_hz=cfg.control_hz,
            min_frames=int(round(cfg.min_record_seconds * cfg.control_hz)),
            arm_sent_stream=True,
        )
    except Exception as e:
        logger.warning("policy_loop: init failed", exc_info=True)
        shared.is_running.value = False
        return

    # ── 4. Keyboard + Audio ──
    kb = KeyboardHandler()
    kb.start()
    audio = AudioFeedback()

    # ── 4b. Hand Retargeter (VR landmarks → XHand 12-DOF joints) ──
    hand_retargeter: XHandRetargeter | None = None
    hand_available = False
    _hand_disconnected_at: float | None = None  # monotonic timestamp of first bad frame
    _hand_ramp_start: np.ndarray | None = None
    _hand_ramp_frames = 0

    # Check if hand is connected by reading state ring (non-blocking peek).
    # Gate on config: --no-hand sets hand_enabled=False → skip entirely.
    if not cfg.hand_enabled:
        hand_available = False
        logger.info("Hand disabled (hand_enabled=False) — hold-position only")
    else:
        _init_hand_state = _read_hand_state(shared)
        if _init_hand_state is not None:
            hand_available = bool(_init_hand_state["connected"][0])
        else:
            hand_available = False

    if hand_available:
        try:
            hand_retargeter = XHandRetargeter(
                hand_type="right",
                retargeting_type=cfg.hand_retargeting_type,
            )
            logger.info("Hand retargeter ready (type=%s)", cfg.hand_retargeting_type)
        except Exception as e:
            logger.warning("Hand retargeter init failed: %s — degraded to hold-position", e)
            hand_available = False
            hand_retargeter = None
    else:
        logger.info("Hand not connected — hold-position only")

    # ── 5. Wait for subsystems ──
    logger.info("Policy: waiting for arm/hand/camera/vr ready...")
    for name, ev in [
        ("arm", shared.arm_ready),
        ("hand", shared.hand_ready),
        ("camera", shared.camera_ready),
        ("vr", shared.vr_ready),
    ]:
        if not ev.wait(timeout=120):
            logger.error("Policy: %s startup timeout (120s)", name)
            shared.is_running.value = False
            kb.stop()
            return
    logger.info("Policy: all subsystems ready")

    # Write heartbeat NOW — Main unblocks from vr_ready at the same moment and
    # enters the supervisor, which checks policy_heartbeat_s immediately.
    # Without this, the ~40 lines of init below race the first supervisor tick.
    shared.policy_heartbeat_s.value = time.monotonic()

    # ── 6. Initial state ──
    home_qpos = np.array(arm.home_qpos, dtype=np.float64)
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
        else np.zeros(12, dtype=np.float64)
    )

    # ── 7. State machine ──
    teleop_active = False
    recording_active = False
    recording_paused = False
    teleop_hold_for_audio = False

    # ── 8. EMA state ──
    ema_prev_pos: np.ndarray | None = None
    ema_prev_quat: np.ndarray | None = None

    # ── 9. Diagnostics ──
    limiter = RateManager(cfg.control_hz)
    stage_timer = StageTimer(window=cfg.status_every)
    _validate_warn = ThrottledWarner(interval_s=2.0)
    _arm_stale_warn = ThrottledWarner(interval_s=3.0)
    _hand_stale_warn = ThrottledWarner(interval_s=5.0)
    loop_count = 0
    error_count = 0
    prev_eef_pos: np.ndarray | None = None
    _last_target_eef_pos = np.full(3, np.nan)  # last valid IK target (held frame continuity)
    _last_target_eef_rot6d = np.full(6, np.nan)

    logger.info("Policy: entering main loop @ %.0f Hz", cfg.control_hz)
    print("\n控制: B=开始遥操作+录制 C=暂停 S=保存 D=丢弃 H=归位 Q=退出 ESC=急停")

    gc.disable()
    try:
        while shared.is_running.value:
            shared.policy_heartbeat_s.value = time.monotonic()
            stage_timer.tick()
            limiter.wait()
            stage_timer.mark("wait")

            # ── Non-blocking save completion ──
            _stop_result = recorder.poll_stop()
            if _stop_result.done and _stop_result.path is not None:
                if _stop_result.error:
                    print(f"  ⚠ 保存失败 ({_stop_result.error}): {_stop_result.path}")
                elif _stop_result.success:
                    print(f"  录制已保存: {_stop_result.path}  ({_stop_result.frame_count} 帧)")
                gc.collect()

            loop_count += 1

            # ── Keyboard ──
            skip_rest = False
            for sig in kb.poll(timeout=0.0):
                if sig == ControlSignal.EMERGENCY_STOP:
                    print("\nESC: emergency_stop")
                    audio.play("emergency")
                    shared.estop_request.value = True
                    shared.is_running.value = False
                    _stop_recording(recorder, recording_active, save=False, shared=shared)
                    recording_active = False
                    break

                elif sig == ControlSignal.QUIT:
                    print("\nQ: 退出")
                    audio.play("quit")

                    if recording_active:
                        audio.play("quit_save_prompt")
                        print("  [S] 保存并退出  [D] 丢弃并退出  [H] 保存并归位 (30s 超时默认丢弃)")

                        decision: bool | None = None
                        do_home: bool = False
                        deadline = time.perf_counter() + 30.0
                        while time.perf_counter() < deadline:
                            for post_sig in kb.poll(timeout=0.1):
                                if post_sig == ControlSignal.STOP:
                                    decision = True
                                    break
                                if post_sig == ControlSignal.DISCARD:
                                    decision = False
                                    break
                                if post_sig == ControlSignal.HOME:
                                    decision = True
                                    do_home = True
                                    break
                                if post_sig == ControlSignal.EMERGENCY_STOP:
                                    audio.play("emergency")
                                    shared.estop_request.value = True
                                    _stop_recording(recorder, recording_active, save=False, shared=shared)
                                    recording_active = False
                                    decision = None
                                    break
                            if decision is not None:
                                break
                            if not shared.is_running.value:
                                break

                        if decision is True:
                            audio.play("save")
                            _stop_recording(recorder, recording_active, save=True, shared=shared)
                            recording_active = False
                            print("  已保存")
                        elif decision is False:
                            audio.play("discard")
                            _stop_recording(recorder, recording_active, save=False, shared=shared)
                            recording_active = False
                            print("  已丢弃")
                        elif shared.is_running.value:
                            audio.play("discard")
                            _stop_recording(recorder, recording_active, save=False, shared=shared)
                            recording_active = False
                            print("  超时，默认丢弃")

                        if do_home and shared.is_running.value:
                            audio.play("home")
                            _home_qpos = np.array(arm.home_qpos, dtype=np.float64)
                            _home_wp = plan_joint_home_path(arm_qpos, _home_qpos, planner)
                            _safe_arm_queue_put(shared, (HOME_SENTINEL, _home_wp))
                            teleop_active = False
                            transition(shared, SafetyState.ARMED)
                            arm_mapper.clear()
                            ema_prev_pos = ema_prev_quat = None
                            _reset_hand_retargeter(hand_retargeter)
                            # Give arm process time to pick up HOME_SENTINEL and home (~8s)
                            _home_deadline = time.monotonic() + 15.0
                            while time.monotonic() < _home_deadline:
                                _hb = shared.arm_heartbeat_s.value
                                if _hb > 0 and time.monotonic() - _hb < 3.0:
                                    time.sleep(0.5)
                                else:
                                    break
                            audio.play("home_done")
                    else:
                        _stop_recording(recorder, recording_active, save=False, shared=shared)
                        recording_active = False

                    shared.is_running.value = False
                    break

                elif sig == ControlSignal.HOME:
                    print("\nH: return_home")
                    audio.play("home")
                    _stop_recording(recorder, recording_active, save=True, shared=shared)
                    recording_active = False
                    # Hand home (send before arm HOME_SENTINEL — hand is independent)
                    _write_hand_cmd(shared, HAND_HOME_QPOS)
                    prev_hand_qpos = HAND_HOME_QPOS.copy()

                    # Plan collision-safe path (uses policy's CollisionModel).
                    _home_qpos = np.array(arm.home_qpos, dtype=np.float64)
                    _waypoints = plan_joint_home_path(arm_qpos, _home_qpos, planner)
                    _safe_arm_queue_put(shared, (HOME_SENTINEL, _waypoints))

                    teleop_active = False
                    transition(shared, SafetyState.ARMED)
                    arm_mapper.clear()
                    ema_prev_pos = ema_prev_quat = None
                    _reset_hand_retargeter(hand_retargeter)
                    audio.play("home_done")
                    skip_rest = True

                elif sig == ControlSignal.STOP:
                    print("\nS: 停止录制")
                    audio.play("save")
                    _stop_recording(recorder, recording_active, save=True, shared=shared)
                    recording_active = False
                    teleop_active = False
                    transition(shared, SafetyState.ARMED)
                    skip_rest = True

                elif sig == ControlSignal.DISCARD:
                    print("\nD: 丢弃录制")
                    audio.play("discard")
                    _stop_recording(recorder, recording_active, save=False, shared=shared)
                    recording_active = False
                    teleop_active = False
                    transition(shared, SafetyState.ARMED)
                    skip_rest = True

                elif sig == ControlSignal.PAUSE:
                    # On resume: must be in RUNNING state
                    if not teleop_active and shared.safety_state.value != SafetyState.RUNNING:
                        print(f"\nC: safety_state={shared.safety_state.value} — cannot resume")
                        skip_rest = True
                        continue
                    teleop_active = not teleop_active
                    recording_paused = not teleop_active
                    state_str = "暂停" if not teleop_active else "恢复"
                    print(f"\nC: {state_str}遥操作")
                    if teleop_active:
                        audio.play("resume")
                        arm_mapper_reset = _try_reset_mapper(shared, arm_mapper)
                        if arm_mapper_reset:
                            # Warm-start hand retargeter NLP from current hardware
                            # pose so the first retarget() converges from near-optimum
                            # (matching B-press pattern at lines 527-531).
                            _hs = _read_hand_state(shared)
                            _reset_hand_retargeter(
                                hand_retargeter,
                                _hs["qpos"][0].copy()
                                if _hs is not None and np.all(np.isfinite(_hs["qpos"][0]))
                                else None,
                            )
                            if _hs is not None and np.all(np.isfinite(_hs["qpos"][0])):
                                prev_hand_qpos = _hs["qpos"][0].copy()
                            audio.play("calibrated")
                            # Trigger audio-hold gate — when audio finishes, the
                            # existing hold-exit path (lines 641-642) seeds the
                            # smoothstep ramp from hardware pose and clears EMA state.
                            teleop_hold_for_audio = True
                    else:
                        audio.play("pause")
                    skip_rest = True

                elif sig == ControlSignal.BEGIN:
                    # Require ARMED state before starting teleop (ManiUniCon P0)
                    if shared.safety_state.value != SafetyState.ARMED:
                        print(f"\nB: safety_state={shared.safety_state.value} — must be ARMED({SafetyState.ARMED})")
                        skip_rest = True
                        continue
                    vr_frame = _read_vr_frame(shared)
                    if vr_frame is None:
                        print("\nB: 无 VR 帧，无法开始遥操作")
                        skip_rest = True
                        continue
                    _stop_recording(recorder, recording_active, save=recording_active, shared=shared)
                    gc.collect()

                    # Read camera metadata from SharedStorage (set by camera_loop).
                    _cam_K = None
                    _cam_K_flat = list(shared.camera_K)
                    if any(v != 0.0 for v in _cam_K_flat):
                        _cam_K = np.array(_cam_K_flat, dtype=np.float64).reshape(3, 3)
                    _depth_scale = float(shared.camera_depth_scale.value) if shared.camera_depth_scale.value != 0.0 else None
                    _cam_serial_bytes = shared.camera_serial.value.rstrip(b"\x00")
                    _cam_serial = _cam_serial_bytes.decode() if _cam_serial_bytes else None
                    _cam_name = _cam_serial

                    if not recorder.start_episode(
                        task_label=cfg.task_label,
                        operator=cfg.operator,
                        camera_K=_cam_K,
                        camera_name=_cam_name,
                        camera_serial=_cam_serial,
                        depth_scale=_depth_scale,
                        record_config={
                            "ema_alpha_pos": cfg.ema_alpha_pos,
                            "ema_alpha_rot": cfg.ema_alpha_rot,
                            "joint_max_acc": cfg.joint_max_acc_deg_s2,
                            "joint_max_speed": cfg.joint_max_speed_deg_s,
                            "inner_loop_hz": cfg.inner_loop_hz,
                            "hand_available": hand_available,
                            "hand_retargeting_type": cfg.hand_retargeting_type,
                        },
                    ):
                        print("  ⚠ 无法开始录制")
                        skip_rest = True
                        continue
                    recording_active = True
                    shared.is_recording.value = True
                    kb.drain_signal(ControlSignal.BEGIN)
                    arm_state = _read_arm_state(shared)
                    if arm_state is not None and np.all(np.isfinite(arm_state["qpos"][0])):
                        arm_qpos = arm_state["qpos"][0].copy()
                        eef_pos = arm_state["eef_pos"][0].copy()
                        eef_rot6d = arm_state["eef_rot6d"][0].copy()
                        eef_quat_wxyz = rot6d_to_quat_wxyz(eef_rot6d)
                    else:
                        eef_pos = np.zeros(3)
                        eef_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
                    arm_mapper.reset(
                        wrist_pos=vr_frame["wrist_pos"],
                        wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
                        eef_pos=eef_pos,
                        eef_quat_wxyz=eef_quat_wxyz,
                    )
                    audio.play("calibrated")
                    teleop_active = True
                    transition(shared, SafetyState.RUNNING)
                    recording_paused = False
                    error_count = 0
                    _hand_disconnected_at = None
                    # Reset hand retargeter and seed with current hardware pose
                    # so the first retarget() frame converges from near-optimum.
                    _hand_state_for_reset = _read_hand_state(shared)
                    _reset_hand_retargeter(
                        hand_retargeter,
                        _hand_state_for_reset["qpos"][0].copy()
                        if _hand_state_for_reset is not None and np.all(np.isfinite(_hand_state_for_reset["qpos"][0]))
                        else None,
                    )
                    # Seed prev_hand_qpos from current state so first frame delta is zero
                    if _hand_state_for_reset is not None and np.all(np.isfinite(_hand_state_for_reset["qpos"][0])):
                        prev_hand_qpos = _hand_state_for_reset["qpos"][0].copy()
                    audio.play("begin")
                    teleop_hold_for_audio = True
                    print(f"\nB: 遥操作+录制开始  episode={recorder.frame_count}")
                    skip_rest = True

            if not shared.is_running.value:
                break
            if skip_rest:
                continue

            # ── Read arm state ──
            arm_state = _read_arm_state(shared)
            if arm_state is None:
                error_count += 1
                if error_count > cfg.max_consecutive_errors:
                    logger.error("连续 arm state 丢失，退出")
                    shared.is_running.value = False
                    break
                continue
            # S06: detect stale arm_state_ring (arm_loop hung but not crashed)
            _arm_state_age_s = time.monotonic() - float(arm_state["timestamp"][0])
            if _arm_state_age_s > 0.5:
                _arm_stale_warn("policy_loop: arm_state stale %.2fs", _arm_state_age_s)
                error_count += 1
                continue
            arm_qpos = arm_state["qpos"][0].copy()
            if not np.all(np.isfinite(arm_qpos)):
                error_count += 1
                continue
            error_count = 0

            # ── Read VR frame ──
            vr_frame = _read_vr_frame(shared)
            vr_stale = vr_frame is None or (
                (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) > cfg.vr_stale_threshold_s * 1e9
            )
            stage_timer.mark("vr")

            # ── Read camera frame ──
            cam = _read_camera_frame(shared)
            stage_timer.mark("cam")

            # ── Read hand state (before held-recording so held frames capture hand data) ──
            hand_state = _read_hand_state(shared)
            hand_tactile = _read_hand_tactile(shared)

            # ── Hand liveness: transient-glitch hold, persistent-disconnect detection ──
            # Single-frame RS485/EtherCAT glitches → retargeter keeps running (commands
            # land on a silent bus — harmless).  Only degrade to hold-position mode after
            # a persistent disconnect exceeding hand_disconnect_timeout_s.
            _hand_connected = bool(hand_state["connected"][0]) if hand_state is not None else False
            if _hand_connected:
                if not hand_available and _hand_disconnected_at is not None:
                    _dt = time.monotonic() - _hand_disconnected_at
                    logger.info("Hand recovered after %.1fs — re-enabling hand control", _dt)
                    hand_available = True
                    _reset_hand_retargeter(hand_retargeter, (
                        hand_state["qpos"][0].copy()
                        if hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0]))
                        else None
                    ))
                    if hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0])):
                        prev_hand_qpos = hand_state["qpos"][0].copy()
                _hand_disconnected_at = None
            elif hand_available:
                if _hand_disconnected_at is None:
                    _hand_disconnected_at = time.monotonic()
                elif time.monotonic() - _hand_disconnected_at > cfg.hand_disconnect_timeout_s:
                    logger.warning(
                        "Hand disconnected for %.1fs — disabling hand control",
                        time.monotonic() - _hand_disconnected_at,
                    )
                    hand_available = False

            # ── Periodic status ──
            if loop_count % cfg.status_every == 0:
                _arm_age = time.monotonic() - float(arm_state["timestamp"][0]) if arm_state is not None else -1.0
                _qdepth = shared.arm_action_q.qsize()
                _print_status(loop_count, arm_state, vr_frame, teleop_active, recording_active,
                             error_count, arm_q_depth=_qdepth, arm_state_age_s=_arm_age)

            # ── Held / inactive → record held frame → continue ──
            if not teleop_active or vr_stale:
                if recording_active and not recording_paused:
                    _record_held(recorder, arm_state, prev_qpos_cmd, prev_hand_qpos, vr_frame, cam,
                                 hand_state=hand_state, hand_tactile=hand_tactile,
                                 arm_qpos_sent=prev_qpos_cmd.copy(),
                                 target_eef_pos=_last_target_eef_pos,
                                 target_eef_rot6d=_last_target_eef_rot6d)
                prev_qpos_cmd = arm_qpos.copy()
                ema_prev_pos = ema_prev_quat = None
                continue

            # ── Audio hold ──
            if teleop_hold_for_audio:
                if audio.is_playing:
                    prev_qpos_cmd = arm_qpos.copy()
                    # Refresh prev_hand_qpos from current hardware state so
                    # the hand ramp starts from the actual joint position.
                    if hand_available and hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0])):
                        prev_hand_qpos = hand_state["qpos"][0].copy()
                    continue
                # Audio just ended — seed the hand ramp origin from current
                # hardware pose so retargeting transitions smoothly.
                ema_prev_pos = ema_prev_quat = None
                _hand_ramp_start = prev_hand_qpos.copy() if hand_available else None
                _hand_ramp_frames = cfg.hand_ramp_frames
                teleop_hold_for_audio = False

            # ── VR→EEF mapping ──
            assert vr_frame is not None  # guaranteed by vr_stale check above
            mapped = arm_mapper.map(vr_frame["wrist_pos"], vr_frame["wrist_quat_wxyz"])
            if mapped is None:
                if recording_active:
                    _record_held(recorder, arm_state, prev_qpos_cmd, prev_hand_qpos, vr_frame, cam,
                                 hand_state=hand_state, hand_tactile=hand_tactile,
                                 arm_qpos_sent=prev_qpos_cmd.copy(),
                                 target_eef_pos=_last_target_eef_pos,
                                 target_eef_rot6d=_last_target_eef_rot6d)
                if not _safe_arm_queue_put(shared, {"qpos": prev_qpos_cmd.copy()}):
                    shared.is_running.value = False
                    break
                continue

            target_pos = mapped["pos"]
            target_quat = mapped["quat_wxyz"]

            # ── Cartesian EMA ──
            if ema_prev_pos is not None:
                assert ema_prev_quat is not None  # set together with ema_prev_pos
                target_pos, target_quat = ema_smooth_pose(
                    target_pos, target_quat, ema_prev_pos, ema_prev_quat,
                    cfg.ema_alpha_pos, cfg.ema_alpha_rot,
                )

            # ── Workspace clamp ──
            target_pos_before_clamp = target_pos.copy()
            for axis in range(3):
                lo, hi = cfg.workspace_bounds[axis]
                target_pos[axis] = np.clip(target_pos[axis], lo, hi)
            stage_timer.mark("map")

            # ── IK solve ──
            target_pose = Pose(p=target_pos, q=target_quat)
            _ik_t0 = time.perf_counter()
            ik_result = planner.solve_teleop_ik(target_pose, arm_qpos, prev_qpos_cmd)
            ik_solve_time_ms = (time.perf_counter() - _ik_t0) * 1000.0
            stage_timer.mark("ik")

            # ── Hand: compute via retargeting from VR landmarks ──
            # (before IK fail/gate checks — hand retargeting is independent of arm IK,
            # so hand commands are sent even when IK fails)
            hand_cmd, retarget_ok = _compute_hand_command(
                hand_retargeter,
                vr_frame,
                prev_hand_qpos,
                hand_available,
            )
            # Smoothstep ramp: ease from home_qpos (captured at audio end)
            # to VR retargeting target over ~1s.  Smoothstep has zero
            # derivative at t=0 — the first few frames barely move,
            # giving the NLP warm-start time to converge.
            if _hand_ramp_frames > 0 and _hand_ramp_start is not None:
                t = 1.0 - (_hand_ramp_frames / cfg.hand_ramp_frames)
                t_smooth = t * t * (3.0 - 2.0 * t)
                hand_cmd = _hand_ramp_start + t_smooth * (hand_cmd - _hand_ramp_start)
                _hand_ramp_frames -= 1
            elif _hand_ramp_frames == 0:
                _hand_ramp_start = None  # ramp complete, release reference

            # G1: detect hand driver board lockout (hand_loop → qpos_stale in state ring).
            # When the hand stops executing commands despite being connected, hold the
            # last known-good command to prevent a gap jump on recovery.
            if hand_state is not None and bool(hand_state["qpos_stale"][0]) and hand_available:
                _hand_stale_warn("policy_loop: hand qpos stale — holding position")
                hand_cmd = prev_hand_qpos.copy()
                retarget_ok = False

            if not ik_result.success or ik_result.qpos is None:
                if recording_active:
                    _record_held(recorder, arm_state, prev_qpos_cmd, prev_hand_qpos, vr_frame, cam,
                                 hand_state=hand_state, hand_tactile=hand_tactile,
                                 frame_status=_FRAME_IK_FAIL, retarget_ok=retarget_ok,
                                 arm_qpos_sent=prev_qpos_cmd.copy(),
                                 target_eef_pos=_last_target_eef_pos,
                                 target_eef_rot6d=_last_target_eef_rot6d)
                if not _safe_arm_queue_put(shared, {"qpos": prev_qpos_cmd.copy()}):
                    shared.is_running.value = False
                    break
                # Hand retargeting is independent of arm IK — send hand
                # even when IK fails so the hand cmd ring doesn't go stale.
                if hand_available:
                    _write_hand_cmd(shared, hand_cmd)
                    prev_hand_qpos = hand_cmd.copy()
                continue

            # IK delta clamp
            arm_cmd = np.asarray(ik_result.qpos, dtype=np.float64)
            arm_cmd = prev_qpos_cmd + np.clip(arm_cmd - prev_qpos_cmd, -arm_cmd_max_step_rad, arm_cmd_max_step_rad)

            # Joint-space hard limit clip (P4 — prevents C9/C31 cascade when
            # IK drifts beyond firmware-accepted bounds).
            arm_cmd = np.clip(arm_cmd, JOINT_LO, JOINT_HI)

            # ── Validate (arm connection + NaN guard) ──
            # Error codes (C22/C24/C31) are handled by arm_loop independently
            # (auto-recover) or trigger FAULT (non-recoverable).  Policy only
            # gates on arm connectivity and command sanity.
            _arm_ok = arm_state is not None and bool(arm_state["connected"][0])
            _reject = False
            _reject_reason = ""
            if not np.all(np.isfinite(arm_cmd)):
                _reject = True
                _reject_reason = "arm_cmd NaN"
            elif not _arm_ok:
                _reject = True
                _reject_reason = "arm disconnected"
            elif not np.all(np.isfinite(hand_cmd)):
                _reject = True
                _reject_reason = "hand_cmd NaN"
            if _reject:
                _validate_warn("policy_loop: action rejected — %s", _reject_reason)
                if recording_active:
                    _record_held(recorder, arm_state, prev_qpos_cmd, prev_hand_qpos, vr_frame, cam,
                                 hand_state=hand_state, hand_tactile=hand_tactile,
                                 frame_status=_FRAME_SAFETY_REJECT, retarget_ok=retarget_ok,
                                 arm_qpos_sent=prev_qpos_cmd.copy(),
                                 target_eef_pos=_last_target_eef_pos,
                                 target_eef_rot6d=_last_target_eef_rot6d)
                if not _safe_arm_queue_put(shared, {"qpos": prev_qpos_cmd.copy()}):
                    shared.is_running.value = False
                    break
                # Hand retargeting is independent of arm — send hand even on arm reject,
                # but do NOT send when the hand command itself was rejected (NaN).
                if hand_available and _reject_reason != "hand_cmd NaN":
                    _write_hand_cmd(shared, hand_cmd)
                    prev_hand_qpos = hand_cmd.copy()
                continue

            # ── Send ──
            # FAULT gate: do not send actions when system is in fault state.
            if shared.safety_state.value == SafetyState.FAULT:
                if recording_active:
                    _record_held(recorder, arm_state, prev_qpos_cmd, prev_hand_qpos, vr_frame, cam,
                                 hand_state=hand_state, hand_tactile=hand_tactile,
                                 arm_qpos_sent=prev_qpos_cmd.copy(),
                                 target_eef_pos=_last_target_eef_pos,
                                 target_eef_rot6d=_last_target_eef_rot6d)
                continue
            if not _safe_arm_queue_put(shared, {"qpos": arm_cmd.copy()}):
                logger.error("policy_loop: arm_action_q full on main send — arm unresponsive")
                shared.is_running.value = False
                break
            _write_hand_cmd(shared, hand_cmd)
            stage_timer.mark("send")

            prev_qpos_cmd = arm_cmd.copy()
            ema_prev_pos = target_pos.copy()
            ema_prev_quat = target_quat.copy()

            # ── Record ──
            if recording_active:
                hand_tactile = _read_hand_tactile(shared)
                # Track last valid IK target for held-frame continuity.
                _last_target_eef_pos = target_pos.copy()
                _last_target_eef_rot6d = quat_wxyz_to_rot6d(normalize_quat_wxyz(target_quat))
                # Hand retargeting fail but IK success → mark separately
                if not retarget_ok and hand_available:
                    _f_status = _FRAME_RETARGET_FAIL
                else:
                    _f_status = _FRAME_OK
                _record_frame(recorder, arm_state, hand_state, arm_cmd, hand_cmd,
                              target_pos, target_quat, vr_frame, cam, ik_solve_time_ms,
                              target_pos_before_clamp,
                              hand_tactile, retarget_ok=retarget_ok, frame_status=_f_status)
            stage_timer.mark("rec")

    finally:
        gc.enable()
        if recording_active:
            _stop_recording(recorder, True, save=False)
        recorder.join_stop(timeout=60)
        kb.stop()
        shared.is_running.value = False
        audio.play("end")
        time.sleep(2.0)
        logger.info("Policy: loop exited")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _safe_arm_queue_put(shared: SharedStorage, action, *, timeout: float = 0.5) -> bool:
    """Put an action onto arm_action_q with timeout protection.

    Returns True on success, False if the queue was full (arm_loop dead or
    severely backed up).  Uses a short timeout so policy_loop can detect
    the failure and trigger a clean shutdown instead of blocking forever.
    """
    from queue import Full

    try:
        shared.arm_action_q.put(action, block=True, timeout=timeout)
        return True
    except Full:
        logger.warning("policy_loop: arm_action_q full (timeout=%.1fs) — arm unresponsive", timeout)
        return False



def _read_vr_frame(shared: SharedStorage) -> dict | None:
    """Read latest VR frame from ring, return as dict or None."""
    result = shared.vr_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    rec = data[0]
    return {
        "wrist_pos": np.asarray(rec["wrist_pos"], dtype=np.float64),
        "wrist_quat_wxyz": np.asarray(rec["wrist_quat_wxyz"], dtype=np.float64),
        "landmarks": np.asarray(rec["landmarks"], dtype=np.float64),
        "head_pos": np.asarray(rec["head_pos"], dtype=np.float64),
        "head_quat_wxyz": np.asarray(rec["head_quat_wxyz"], dtype=np.float64),
        "recv_ts_ns": int(rec["recv_ts_ns"]),
        "source_ts_ns": int(rec["source_ts_ns"]),
        "sequence_id": int(rec["sequence_id"]),
        "source_frame_seq": int(rec["source_frame_seq"]),
        "local_recv_ns": int(rec["local_recv_ns"]),
        "side": int(rec["side"]),
    }


def _read_hand_tactile(shared: SharedStorage) -> np.ndarray | None:
    """Read latest hand tactile force from sparse ring."""
    result = shared.hand_tactile_ring.read_latest()
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def _read_camera_frame(shared: SharedStorage) -> dict | None:
    """Read latest camera frame. Returns None on failure."""
    try:
        result = shared.camera_ring.read_latest()
        if result is not None:
            return {"header": result[0], "rgb": result[1], "depth": result[2]}
    except Exception:
        logger.warning("policy_loop: camera ring read failed", exc_info=True)
    return None



def _try_reset_mapper(shared: SharedStorage, arm_mapper: ArmWristMapper) -> bool:
    """Re-establish wrist→EEF mapping on resume. Returns True on success."""
    vr_frame = _read_vr_frame(shared)
    if vr_frame is None:
        return False
    arm_state = _read_arm_state(shared)
    if arm_state is None:
        return False
    eef_pos = arm_state["eef_pos"][0].copy()
    eef_rot6d = arm_state["eef_rot6d"][0].copy()
    eef_quat_wxyz = rot6d_to_quat_wxyz(eef_rot6d)
    if not np.all(np.isfinite(eef_pos)) or not np.all(np.isfinite(eef_quat_wxyz)):
        return False
    assert vr_frame is not None and arm_state is not None  # checked above
    arm_mapper.reset(
        wrist_pos=vr_frame["wrist_pos"],
        wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
        eef_pos=eef_pos,
        eef_quat_wxyz=eef_quat_wxyz,
    )
    return True


def _stop_recording(recorder: EpisodeRecorder, was_active: bool, *, save: bool, shared: SharedStorage | None = None) -> None:
    """Stop recording if active. Non-blocking — poll completion in main loop."""
    if was_active:
        recorder.stop_episode(success=save)
        if shared is not None:
            shared.is_recording.value = False


def _record_held(
    recorder: EpisodeRecorder,
    arm_state: np.ndarray | None,
    hold_arm: np.ndarray,
    hold_hand: np.ndarray,
    vr_frame: dict | None,
    cam: dict | None,
    *,
    hand_state: np.ndarray | None = None,
    hand_tactile: np.ndarray | None = None,
    frame_status: int = _FRAME_HELD,
    retarget_ok: bool = False,
    arm_qpos_sent: np.ndarray | None = None,
    diagnostics: dict | None = None,
    target_eef_pos: np.ndarray | None = None,
    target_eef_rot6d: np.ndarray | None = None,
) -> None:
    """Record a held frame (no new action sent).

    Args:
        arm_qpos_sent: Last command actually queued to arm_action_q.
            Ensures ``--source=sent`` replay works for held frames.
        diagnostics: Per-frame diagnostics (tracking_error, ik_solve_time_ms, etc.).
        target_eef_pos/rot6d: Last valid IK target — prevents NaN gaps in
            ``action_arm_ee`` for replay.
    """
    if vr_frame is None:
        vr_frame = {
            "wrist_pos": np.full(3, np.nan),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "landmarks": np.full((21, 3), np.nan),
        }
    action = RobotAction(
        arm_qpos_cmd=hold_arm,
        hand_qpos_cmd=hold_hand,
        target_eef_pos=target_eef_pos.copy() if target_eef_pos is not None else None,
        target_eef_rot6d=target_eef_rot6d.copy() if target_eef_rot6d is not None else None,
    )
    state = _build_robot_state(arm_state, hand_state, hand_tactile)
    recorder.add_frame(
        state, action, vr_frame, camera_frame=cam,
        signals={"ik_ok": False, "ik_attempted": frame_status != _FRAME_HELD,
                 "retarget_ok": retarget_ok, "held": True,
                 "flag_safety_reject": frame_status == _FRAME_SAFETY_REJECT,
                 "frame_status": frame_status},
        arm_qpos_sent=arm_qpos_sent,
        diagnostics=diagnostics,
    )


def _record_frame(
    recorder: EpisodeRecorder,
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    arm_cmd: np.ndarray,
    hand_cmd: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    vr_frame: dict | None,
    cam: dict | None,
    ik_solve_time_ms: float,
    target_pos_before_clamp: np.ndarray,
    hand_tactile: np.ndarray | None = None,
    *,
    retarget_ok: bool = False,
    frame_status: int = _FRAME_OK,
) -> None:
    """Record a normal (active teleop) frame.

    Args:
        target_quat: EMA-smoothed IK target quaternion (wxyz), NOT raw VR wrist.
            This is what the IK solver actually tracked.
    """
    action = RobotAction(
        arm_qpos_cmd=arm_cmd,
        hand_qpos_cmd=hand_cmd,
        target_eef_pos=target_pos.copy(),
        target_eef_rot6d=quat_wxyz_to_rot6d(normalize_quat_wxyz(target_quat)),
    )
    state = _build_robot_state(arm_state, hand_state, hand_tactile)
    head_quat = vr_frame.get("head_quat_wxyz") if vr_frame is not None else None
    _vr = vr_frame if vr_frame is not None else {
        "wrist_pos": np.full(3, np.nan),
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.full((21, 3), np.nan),
    }
    recorder.add_frame(
        state, action, _vr, camera_frame=cam,
        signals={"ik_ok": True, "ik_attempted": True, "retarget_ok": retarget_ok,
                 "held": False, "flag_safety_reject": frame_status == _FRAME_SAFETY_REJECT,
                 "frame_status": frame_status},
        arm_qpos_sent=arm_cmd.copy(),
        diagnostics={
            "tracking_error": float(arm_state["tracking_err"][0]) if arm_state is not None and "tracking_err" in arm_state.dtype.names else 0.0,
            "ik_solve_time_ms": ik_solve_time_ms,
            "target_pos_before_clamp": target_pos_before_clamp,
            "head_quat_wxyz": head_quat if head_quat is not None else np.full(4, np.nan),
        },
    )


def _build_robot_state(arm_state: np.ndarray | None, hand_state: np.ndarray | None,
                       hand_tactile: np.ndarray | None = None) -> RobotState:
    """Build a RobotState from ring data for recording compatibility."""
    if arm_state is not None:
        r = arm_state[0]
        arm_qpos = np.asarray(r["qpos"], dtype=np.float64)
        arm_qvel = np.asarray(r["qvel"], dtype=np.float64)
        arm_tau = np.asarray(r["tau"], dtype=np.float64)
        eef_pos = np.asarray(r["eef_pos"], dtype=np.float64)
        eef_rot6d = np.asarray(r["eef_rot6d"], dtype=np.float64)
        arm_connected = bool(r["connected"])
    else:
        arm_qpos = nan_array(7)
        arm_qvel = nan_array(7)
        arm_tau = nan_array(7)
        eef_pos = nan_array(3)
        eef_rot6d = nan_array(6)
        arm_connected = False

    if hand_state is not None:
        h = hand_state[0]
        hand_qpos = np.asarray(h["qpos"], dtype=np.float64)
        hand_current = np.asarray(h["current"], dtype=np.float64)
        hand_tactile_sum = np.asarray(h["tactile_sum"], dtype=np.float64)
        hand_tactile_contact = np.asarray(h["tactile_contact"], dtype=bool)
        hand_connected = bool(h["connected"])
        hand_qpos_stale = bool(h["qpos_stale"]) if "qpos_stale" in h.dtype.names else False
    else:
        hand_qpos = nan_array(12)
        hand_current = nan_array(12)
        hand_tactile_sum = nan_array((5, 3))
        hand_tactile_contact = np.zeros(5, dtype=bool)
        hand_connected = False
        hand_qpos_stale = False

    # Tactile force from separate ring (Phase 2.8)
    if hand_tactile is not None:
        hand_tactile_force = np.asarray(hand_tactile[0]["tactile_force"], dtype=np.float64)
    else:
        hand_tactile_force = np.zeros((5, 120, 3), dtype=np.float64)

    return RobotState(
        arm_qpos=arm_qpos,
        arm_qvel=arm_qvel,
        arm_tau=arm_tau,
        eef_pos=eef_pos,
        eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        eef_rot6d=eef_rot6d,
        hand_qpos=hand_qpos,
        hand_current=hand_current,
        hand_tactile_sum=hand_tactile_sum,
        hand_tactile_force=hand_tactile_force,
        hand_tactile_contact=hand_tactile_contact,
        hand_tipboard_err=np.zeros(12, dtype=np.int32),
        hand_commboard_err=np.zeros(12, dtype=np.int32),
        hand_jointboard_err=np.zeros(12, dtype=np.int32),
        hand_qpos_stale=hand_qpos_stale,
        fingertip_pos=nan_array((5, 3)),
        arm_connected=arm_connected,
        hand_connected=hand_connected,
        timestamp=time.perf_counter(),
    )



# ═══════════════════════════════════════════════════════════════════
# Hand retargeting helpers (P4 Step 1 — ported from vr_teleop_hand_record.py)
# ═══════════════════════════════════════════════════════════════════

_retarget_fail_warn = ThrottledWarner()


def _compute_hand_command(
    retargeter: XHandRetargeter | None,
    vr_frame: dict | None,
    prev_hand_cmd: np.ndarray,
    hand_available: bool,
) -> tuple[np.ndarray, bool]:
    """Compute hand joint command from VR landmarks via DexPilot retargeting.

    Returns (hand_cmd, retarget_ok). On failure or hand unavailable,
    returns prev_hand_cmd unchanged with retarget_ok=False.
    """
    if not hand_available:
        return prev_hand_cmd.copy(), False

    if retargeter is None:
        return prev_hand_cmd.copy(), False

    landmarks = vr_frame.get("landmarks") if vr_frame is not None else None
    if landmarks is None:
        return prev_hand_cmd.copy(), False

    try:
        target = retargeter.retarget(landmarks)  # validates shape + finiteness internally
        if target is not None and len(target) == 12:
            return np.asarray(target, dtype=np.float64), True
        _retarget_fail_warn(
            "Hand retargeting: retargeter.retarget() returned %s",
            "None" if target is None else f"len={len(target)}",
        )
    except Exception:
        logger.warning("Hand retargeting failed — holding position", exc_info=True)

    return prev_hand_cmd.copy(), False


def _reset_hand_retargeter(
    retargeter: XHandRetargeter | None,
    hand_qpos: np.ndarray | None = None,
) -> None:
    """Reset hand retargeter state for a clean teleop start.

    Seeds SLSQP warm-start from actual hardware pose so the first
    retarget() call converges from near-optimum instead of the neutral midpoint.
    """
    if retargeter is not None:
        try:
            retargeter.reset(initial_qpos=hand_qpos)
        except Exception:
            pass


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
        eef_pos = np.round(arm_state["eef_pos"][0], 3)
    else:
        eef_pos = np.full(3, np.nan)
    vr_age = (
        (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) / 1e9
        if vr_frame is not None
        else -1
    )
    parts = [
        f"[f={loop_count}]",
        f"eef={eef_pos}m",
        f"teleop={'ON' if teleop_active else 'OFF'}",
        f"rec={'ON' if recording_active else 'OFF'}",
        f"vr_age={vr_age:.3f}s",
        f"err={error_count}",
    ]
    if arm_q_depth >= 0:
        parts.append(f"q={arm_q_depth}")
    if arm_state_age_s >= 0:
        parts.append(f"arm_age={arm_state_age_s:.2f}s")
    print("  ".join(parts), flush=True)
