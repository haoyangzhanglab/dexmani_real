"""Policy process: VR-to-IK pipeline, state machine, and recording via SharedStorage."""

from __future__ import annotations

import gc
import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real import ASSET_DIR
from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.defaults import arm, hand, policy
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.hand_kinematics import HandKinematics
from dexmani_real.planning.pose_utils import compose_pose, normalize_quat_wxyz, quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.policy.loop_timing import StageTimer
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.shm.shared_storage import HAND_CMD_DTYPE, SharedStorage, hand_home_converge, make_arm_action
from dexmani_real.shm.shared_storage import read_arm_state as _read_arm_state
from dexmani_real.shm.shared_storage import read_hand_state as _read_hand_state
from dexmani_real.shm.shared_storage import send_arm_home
from dexmani_real.shm.shared_storage import write_hand_cmd as _write_hand_cmd
from dexmani_real.teleop.arm_mapper import ArmWristMapper
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.hand_retarget import TAGHandRetargeter, XHandRetargeter
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.log import ThrottledWarner, get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.signal_utils import ema_smooth_pose

logger = get_logger(__name__)

_FRAME_OK = 0
_FRAME_HELD = 1
_FRAME_IK_FAIL = 2
_FRAME_SAFETY_REJECT = 3
_FRAME_RETARGET_FAIL = 4


@dataclass
class PolicyConfig:
    """Policy process configuration."""

    control_hz: float = field(default_factory=lambda: policy.control_hz)

    # Mode 6 firmware parameters (deg — matches CLI convention)
    joint_max_speed_deg_s: float = field(default_factory=lambda: arm.max_joint_velocity_deg_per_s)
    joint_max_acc_deg_s2: float = field(default_factory=lambda: arm.max_joint_acceleration_deg_per_s2)
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)

    vr_pos_scale: float = field(default_factory=lambda: policy.vr_mapping.pos_scale)
    vr_rot_scale: float = field(default_factory=lambda: policy.vr_mapping.rot_scale)
    vr_max_delta_rot_rad: float = field(default_factory=lambda: policy.vr_mapping.max_delta_rot_rad)
    vr_stale_threshold_s: float = field(default_factory=lambda: policy.vr_mapping.stale_threshold_s)
    # Workspace bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] (m)
    workspace_bounds: tuple = field(default_factory=lambda: policy.workspace.as_tuple())

    # Contact-stall resync. Table height is context only, never a pose limit.
    contact_stall_enabled: bool = field(default_factory=lambda: policy.contact_stall_enabled)
    contact_stall_table_z_surface_m: float = field(default_factory=lambda: arm.table_z_surface_m)
    contact_stall_table_context_height_m: float = field(
        default_factory=lambda: policy.contact_stall_table_context_height_m
    )
    contact_stall_min_downward_target_m: float = field(
        default_factory=lambda: policy.contact_stall_min_downward_target_m
    )
    contact_stall_tracking_error_rad: float = field(default_factory=lambda: policy.contact_stall_tracking_error_rad)
    contact_stall_max_closing_speed_rad_s: float = field(
        default_factory=lambda: policy.contact_stall_max_closing_speed_rad_s
    )

    # Cartesian EMA smoothing (tuned at 16Hz)
    ema_alpha_pos: float = field(default_factory=lambda: policy.ema.alpha_pos)
    ema_alpha_rot: float = field(default_factory=lambda: policy.ema.alpha_rot)

    max_record_seconds: float = field(default_factory=lambda: policy.max_record_duration_s)
    min_record_seconds: float = field(default_factory=lambda: policy.min_record_duration_s)
    episodes_dir: str = field(default_factory=lambda: policy.episodes_dir)
    task_label: str = ""
    operator: str = ""

    # Status print interval (in control ticks)
    status_every: int = field(default_factory=lambda: policy.status_print_interval)

    max_consecutive_errors: int = field(default_factory=lambda: policy.max_consecutive_errors)

    hand_enabled: bool = field(default_factory=lambda: policy.hand_enabled)
    hand_retargeting_type: str = field(default_factory=lambda: policy.hand_retargeting_type)
    hand_output_smoothing_alpha: float = field(default_factory=lambda: policy.hand_output_smoothing_alpha)
    hand_ramp_duration_s: float = field(default_factory=lambda: policy.hand_ramp_duration_s)
    begin_motion_gate_timeout_s: float = field(default_factory=lambda: policy.begin_motion_gate_timeout_s)
    hand_disconnect_timeout_s: float = field(default_factory=lambda: policy.hand_disconnect_timeout_s)

    # Hand FK (fingertip positions)
    hand_urdf_path: str = field(default_factory=lambda: str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"))
    fingertip_link_names: tuple[str, ...] = field(default_factory=lambda: hand.fingertip_link_names)
    T_eef_handbase_pos_xyz: tuple[float, float, float] = field(default_factory=lambda: hand.T_eef_handbase_pos_xyz)
    T_eef_handbase_quat_wxyz: tuple[float, float, float, float] = field(
        default_factory=lambda: hand.T_eef_handbase_quat_wxyz
    )

    # Joint-space hard limits — sourced from arm singleton via shared_storage.
    joint_limit_lower: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_lower)
    joint_limit_upper: tuple[float, ...] = field(default_factory=lambda: arm.joint_limit_upper)

    hand_home_qpos_deg: tuple[float, ...] = field(default_factory=lambda: hand.home_qpos_deg)
    hand_qpos_lower_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_min_rad)
    hand_qpos_upper_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_max_rad)
    hand_feedback_bound_tolerance_rad: float = field(default_factory=lambda: hand.feedback_bound_tolerance_rad)
    hand_max_delta_rad: float | None = field(default_factory=lambda: hand.max_delta_rad)

    # VR transform config path (relative to repo root)
    vr_transform_path: str = "dexmani_real/config/vr_transform.json"

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        if not (0.0 <= self.hand_output_smoothing_alpha <= 1.0):
            raise ValueError("hand_output_smoothing_alpha must be in [0, 1]")
        if not np.isfinite(self.hand_ramp_duration_s) or self.hand_ramp_duration_s < 0:
            raise ValueError("hand_ramp_duration_s must be finite and >= 0")
        if not np.isfinite(self.begin_motion_gate_timeout_s) or self.begin_motion_gate_timeout_s < 0:
            raise ValueError("begin_motion_gate_timeout_s must be finite and >= 0")


def _sanitize_hand_command(
    hand_cmd: np.ndarray,
    previous_hand_cmd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_delta_rad: float | None,
) -> np.ndarray:
    """Return the exact finite, joint-limited hand target sent to the worker."""
    command = np.asarray(hand_cmd, dtype=np.float64)
    previous = np.asarray(previous_hand_cmd, dtype=np.float64)
    if command.shape != (12,) or previous.shape != (12,):
        raise ValueError(f"hand commands must have shape (12,), got {command.shape} and {previous.shape}")
    if not np.all(np.isfinite(command)) or not np.all(np.isfinite(previous)):
        raise ValueError("hand command contains NaN or Inf")
    command = np.clip(command, lower, upper)
    if max_delta_rad is not None:
        if not np.isfinite(max_delta_rad) or max_delta_rad <= 0:
            raise ValueError("hand_max_delta_rad must be finite and > 0")
        command = previous + np.clip(command - previous, -max_delta_rad, max_delta_rad)
    return command


def _hand_ramp_frame_count(duration_s: float, control_hz: float) -> int:
    """Convert a ramp duration to policy frames without baking in 16 Hz."""
    if not np.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and >= 0")
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and > 0")
    return max(0, int(round(duration_s * control_hz)))


def _smoothstep_hand_ramp(
    start: np.ndarray,
    target: np.ndarray,
    step_index: int,
    total_steps: int,
) -> np.ndarray:
    """Blend one startup-ramp sample; the final configured step reaches target exactly."""
    start_arr = np.asarray(start, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if start_arr.shape != target_arr.shape:
        raise ValueError(f"ramp arrays must have matching shapes, got {start_arr.shape} and {target_arr.shape}")
    if total_steps <= 0:
        return target_arr.copy()
    if not 0 <= step_index < total_steps:
        raise ValueError(f"step_index={step_index} must be in [0, {total_steps})")
    progress = (step_index + 1) / total_steps
    smooth = progress * progress * (3.0 - 2.0 * progress)
    return start_arr + smooth * (target_arr - start_arr)


def _update_audio_motion_gate(
    *,
    audio_playing: bool,
    begin_deadline_s: float | None,
    ignore_begin_until_silent: bool,
    now_s: float,
) -> tuple[bool, float | None, bool]:
    """Advance the bounded begin-audio gate and return ``(hold, deadline, ignore)``."""
    begin_active = begin_deadline_s is not None and now_s < begin_deadline_s
    if begin_deadline_s is not None and not begin_active:
        # The begin cue may continue audibly, but no longer gates motion.
        ignore_begin_until_silent = audio_playing
        begin_deadline_s = None

    if ignore_begin_until_silent:
        if not audio_playing:
            ignore_begin_until_silent = False
        should_hold = False
    else:
        should_hold = begin_active or audio_playing
    return should_hold, begin_deadline_s, ignore_begin_until_silent


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
        logger.warning("policy_loop: arm-hand collision check failed", exc_info=True)
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
    if qpos.shape != (7,) or qvel.shape != (7,) or command.shape != (7,):
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
    prev_hand_qpos: np.ndarray,
    planner,
    audio,
    hand_home_qpos: np.ndarray,
    table_z_surface_m: float,
    arm_mapper=None,
    hand_retargeter=None,
    heartbeat: bool = True,
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
            heartbeat=heartbeat,
            check_is_running=True,
            verbose=True,
        )
        if final_qpos is not None:
            prev_hand_qpos = final_qpos
            planner.set_hand_qpos(prev_hand_qpos)
        if not hand_reached:
            logger.warning("arm home cancelled: hand did not reach a fresh, healthy home state")
            return prev_hand_qpos
    elif not hand_available:
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
        _state_age_s > 0.5
        or not bool(_arm_state["connected"][0])
        or int(_arm_state["error_code"][0]) != 0
        or not np.all(np.isfinite(_arm_state["qpos"][0]))
    ):
        logger.warning("arm home cancelled: arm state is stale or unhealthy (age=%.3fs)", _state_age_s)
        return prev_hand_qpos
    arm_qpos = np.asarray(_arm_state["qpos"][0], dtype=np.float64).copy()
    _home_qpos = np.array(arm.home_qpos, dtype=np.float64)
    _ok = send_arm_home(
        shared,
        _home_qpos,
        planner=planner,
        table_z_surface_m=table_z_surface_m,
        current_qpos=arm_qpos,
        heartbeat=heartbeat,
        converge_timeout_s=15.0,
        queue_timeout=0.2,
        verbose=True,
    )
    if _ok:
        audio.play("home_done")
        print("  arm: home reached", flush=True)
    else:
        logger.warning("arm home failed or was cancelled; see correlated HOME result above")

    return prev_hand_qpos


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
    HAND_QPOS_LO = np.asarray(cfg.hand_qpos_lower_rad, dtype=np.float64)
    HAND_QPOS_HI = np.asarray(cfg.hand_qpos_upper_rad, dtype=np.float64)

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
                max_pose_error_pos_m=0.02,
                max_pose_error_rot_rad=np.deg2rad(5.0),
                nullspace_step_size_deg=1.0 * (50.0 / cfg.control_hz),
            ),
            hand_dof=True,  # 19-DOF — hand geometry follows set_hand_qpos()
        )

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
            base_to_world_rot=np.eye(3, dtype=np.float64),
        )

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

    kb = KeyboardHandler()
    kb.start()
    audio = AudioFeedback()

    hand_retargeter: TAGHandRetargeter | XHandRetargeter | None = None
    hand_available = False
    _hand_disconnected_at: float | None = None  # monotonic timestamp of first bad frame
    _hand_ramp_start: np.ndarray | None = None
    _hand_ramp_step = 0
    _hand_ramp_total_frames = _hand_ramp_frame_count(cfg.hand_ramp_duration_s, cfg.control_hz)

    def _try_init_hand_retargeter() -> None:
        """Lazily initialize hand_retargeter if not already created."""
        nonlocal hand_retargeter, hand_available
        if hand_retargeter is not None:
            return
        try:
            if cfg.hand_retargeting_type == "tag":
                hand_retargeter = TAGHandRetargeter(
                    hand_type="right",
                    smoothing_alpha=cfg.hand_output_smoothing_alpha,
                    feedback_bound_tolerance_rad=cfg.hand_feedback_bound_tolerance_rad,
                )
            else:
                hand_retargeter = XHandRetargeter(
                    hand_type="right",
                    retargeting_type=cfg.hand_retargeting_type,
                    smoothing_alpha=cfg.hand_output_smoothing_alpha,
                )
            logger.info("Hand retargeter ready (type=%s)", cfg.hand_retargeting_type)
        except Exception as e:
            logger.warning("Hand retargeter init failed: %s — degraded to hold-position", e)
            hand_available = False
            hand_retargeter = None

    def _init_and_seed_hand_retargeter() -> np.ndarray | None:
        """Lazy-init retargeter and seed NLP warm-start from hardware qpos.

        Returns the seeded qpos (for updating ``prev_hand_qpos``) or None.
        """
        _try_init_hand_retargeter()
        hs = _read_hand_state(shared)
        qpos = hs["qpos"][0] if hs is not None else None
        return _seed_hand_retargeter(hand_retargeter, qpos)

    _hand_fk: HandKinematics | None = None
    _T_eef_handbase_pos = np.array(cfg.T_eef_handbase_pos_xyz, dtype=np.float64)
    _T_eef_handbase_quat_wxyz = np.array(cfg.T_eef_handbase_quat_wxyz, dtype=np.float64)
    if cfg.hand_urdf_path:
        try:
            _hand_fk = HandKinematics(cfg.hand_urdf_path, list(cfg.fingertip_link_names))
            if _hand_fk.is_ready():
                logger.info("Hand FK ready")
            else:
                logger.warning("Hand FK not ready — fingertips will be NaN")
        except Exception as e:
            logger.warning("Hand FK init failed: %s", e)

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

    # Hand connectivity check: MUST run after hand_ready — hand_loop publishes
    # initial state BEFORE setting the event, so the ring is guaranteed populated.
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
            _try_init_hand_retargeter()
        else:
            logger.info("Hand not connected — hold-position only")

    # Write heartbeat NOW — Main unblocks from vr_ready at the same moment and
    # enters the supervisor, which checks policy_heartbeat_s immediately.
    # Without this, the ~40 lines of init below race the first supervisor tick.
    shared.policy_heartbeat_s.value = time.monotonic()

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
        else HAND_HOME_QPOS.copy()
    )
    planner.set_hand_qpos(prev_hand_qpos)  # sync hand pose for collision checks

    teleop_active = False
    recording_active = False
    recording_paused = False
    _prev_audio_playing = False
    _begin_audio_gate_deadline_s: float | None = None
    _ignore_begin_audio_until_silent = False

    # Post-teleop quit prompt (two-stage Q: first Q stops teleop, second Q exits)
    quit_pending = False
    post_teleop_deadline = 0.0

    ema_prev_pos: np.ndarray | None = None
    ema_prev_quat: np.ndarray | None = None

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
    _last_tactile_data: np.ndarray | None = None  # forward-fill cache for held-frame tactile

    # Automatic cyclic GC is left enabled (2026-08-05 audit).  The hot path
    # allocates mostly numpy arrays (refcounted, not cyclic), so GC thresholds
    # are rarely reached.  gc.collect() at episode boundaries serves as explicit hints.

    # SIGTERM graceful shutdown
    # Policy is spawned via mp.Process; Main calls process.terminate() on
    # shutdown, which sends SIGTERM.  Default SIGTERM handler kills the
    # process immediately → daemon stop-thread killed mid-HDF5-write
    # → truncated HDF5.  We intercept SIGTERM and set a flag so the main
    # loop exits cleanly through the finally block, which flushes and
    # joins the recorder daemon.
    _sigterm_requested = False

    def _on_sigterm(signum: int, frame: object) -> None:
        nonlocal _sigterm_requested
        _sigterm_requested = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    logger.info("Policy: entering main loop @ %.0f Hz", cfg.control_hz)

    try:
        while shared.is_running.value and not _sigterm_requested:
            shared.policy_heartbeat_s.value = time.monotonic()
            stage_timer.tick()
            limiter.wait()
            stage_timer.mark("wait")

            _stop_result = recorder.poll_stop()
            if _stop_result.done and _stop_result.path is not None:
                if _stop_result.error:
                    print(f"  ⚠ 保存失败 ({_stop_result.error}): {_stop_result.path}")
                elif _stop_result.success:
                    print(f"  录制已保存: {_stop_result.path}  ({_stop_result.frame_count} 帧)")
                gc.collect()

            loop_count += 1

            # Entered after Q key stops teleop.  Policy loop stays alive with
            # heartbeats ticking so arm/hand/vr continue running — H (return_home)
            # can still queue HOME_SENTINEL and wait for convergence.
            if quit_pending:
                # Show prompt once per entry; re-shown after H completes.
                for _sig in kb.poll(timeout=0.1):
                    if _sig == ControlSignal.HOME:
                        print("  H: return_home")
                        prev_hand_qpos = _do_teleop_home(
                            shared,
                            hand_available=hand_available,
                            prev_hand_qpos=prev_hand_qpos,
                            planner=planner,
                            audio=audio,
                            hand_home_qpos=HAND_HOME_QPOS,
                            table_z_surface_m=arm.table_z_surface_m,
                            # arm_mapper/hand_retargeter=None: post-teleop, state already cleared
                        )
                        limiter.reset()
                        print("  [Q] quit", flush=True)

                    elif _sig in (ControlSignal.QUIT, ControlSignal.EMERGENCY_STOP):
                        if _sig == ControlSignal.EMERGENCY_STOP:
                            shared.estop_request.value = True
                        shared.is_running.value = False
                        break

                if not shared.is_running.value:
                    break

                if time.perf_counter() > post_teleop_deadline:
                    print("  timeout — auto exit")
                    shared.is_running.value = False
                    break

                continue  # stay in quit_pending, don't process normal teleop

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
                        audio.queue("quit_save_prompt")  # queue: plays after "quit" finishes
                        print("  [S] 保存并退出  [D] 丢弃并退出  [H] 保存并归位 (30s 超时默认丢弃)")

                        decision: bool | None = None
                        do_home: bool = False
                        deadline = time.perf_counter() + 30.0
                        while time.perf_counter() < deadline:
                            shared.policy_heartbeat_s.value = time.monotonic()
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
                            teleop_active = False
                            transition(shared, SafetyState.ARMED)
                            ema_prev_pos = ema_prev_quat = None
                            prev_hand_qpos = _do_teleop_home(
                                shared,
                                hand_available=hand_available,
                                prev_hand_qpos=prev_hand_qpos,
                                planner=planner,
                                audio=audio,
                                hand_home_qpos=HAND_HOME_QPOS,
                                table_z_surface_m=arm.table_z_surface_m,
                                arm_mapper=arm_mapper,
                                hand_retargeter=hand_retargeter,
                            )

                    # Enter post-teleop state (two-stage Q) instead of immediate exit.
                    # Policy loop stays alive with heartbeats ticking; arm/hand/vr
                    # continue running so H (return_home) still works.
                    teleop_active = False
                    transition(shared, SafetyState.ARMED)
                    shared.quit_requested.value = True
                    quit_pending = True
                    post_teleop_deadline = time.perf_counter() + 60.0
                    print("\n[H] return_home  [Q] quit  (60s timeout)", flush=True)
                    skip_rest = True
                    break  # break from for-sig loop, re-enter main loop in quit_pending state

                elif sig == ControlSignal.HOME:
                    print("\nH: return_home")
                    audio.play("home")
                    _stop_recording(recorder, recording_active, save=True, shared=shared)
                    recording_active = False
                    teleop_active = False
                    transition(shared, SafetyState.ARMED)
                    ema_prev_pos = ema_prev_quat = None
                    prev_hand_qpos = _do_teleop_home(
                        shared,
                        hand_available=hand_available,
                        prev_hand_qpos=prev_hand_qpos,
                        planner=planner,
                        audio=audio,
                        hand_home_qpos=HAND_HOME_QPOS,
                        table_z_surface_m=arm.table_z_surface_m,
                        arm_mapper=arm_mapper,
                        hand_retargeter=hand_retargeter,
                    )
                    limiter.reset()
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
                        arm_mapper_reset = _try_reset_mapper(shared)
                        if arm_mapper_reset:
                            # Warm-start hand retargeter NLP from current hardware
                            # pose so the first retarget() converges from near-optimum
                            # (matching B-press pattern).
                            _seeded = _init_and_seed_hand_retargeter()
                            if _seeded is not None:
                                prev_hand_qpos = _seeded
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
                    _depth_scale = (
                        float(shared.camera_depth_scale.value) if shared.camera_depth_scale.value != 0.0 else None
                    )
                    _cam_serial_bytes = shared.camera_serial.value.rstrip(b"\x00")
                    _cam_serial = _cam_serial_bytes.decode() if _cam_serial_bytes else None

                    # Resolve camera extrinsics from cameras.json by serial.
                    _calib = CameraCalib()
                    try:
                        _cam_name = _calib.resolve_name_by_serial(_cam_serial) if _cam_serial else None
                    except (KeyError, FileNotFoundError):
                        _cam_name = None
                        logger.warning(
                            "Camera serial %s not found in cameras.json — no extrinsics in /meta", _cam_serial
                        )

                    if not recorder.start_episode(
                        task_label=cfg.task_label,
                        operator=cfg.operator,
                        calib=_calib,
                        camera_K=_cam_K,
                        camera_name=_cam_name,
                        camera_serial=_cam_serial,
                        depth_scale=_depth_scale,
                        record_config={
                            "ema_alpha_pos": cfg.ema_alpha_pos,
                            "ema_alpha_rot": cfg.ema_alpha_rot,
                            "joint_max_acc": cfg.joint_max_acc_deg_s2,
                            "joint_max_speed": cfg.joint_max_speed_deg_s,
                            "arm_loop_hz": cfg.arm_loop_hz,
                            "contact_stall_enabled": cfg.contact_stall_enabled,
                            "contact_stall_table_z_surface_m": cfg.contact_stall_table_z_surface_m,
                            "contact_stall_table_context_height_m": cfg.contact_stall_table_context_height_m,
                            "contact_stall_min_downward_target_m": cfg.contact_stall_min_downward_target_m,
                            "contact_stall_tracking_error_rad": cfg.contact_stall_tracking_error_rad,
                            "contact_stall_max_closing_speed_rad_s": cfg.contact_stall_max_closing_speed_rad_s,
                            "hand_available": hand_available,
                            "hand_retargeting_type": cfg.hand_retargeting_type,
                            "hand_output_smoothing_alpha": cfg.hand_output_smoothing_alpha,
                            "hand_ramp_duration_s": cfg.hand_ramp_duration_s,
                            "begin_motion_gate_timeout_s": cfg.begin_motion_gate_timeout_s,
                            "hand_feedback_bound_tolerance_rad": cfg.hand_feedback_bound_tolerance_rad,
                        },
                    ):
                        print("  ⚠ 无法开始录制")
                        skip_rest = True
                        continue
                    recording_active = True
                    shared.is_recording.value = True
                    kb.drain_signal(ControlSignal.BEGIN)
                    teleop_active = True
                    transition(shared, SafetyState.RUNNING)
                    recording_paused = False
                    error_count = 0
                    _hand_disconnected_at = None
                    # Reset hand retargeter and seed with current hardware pose
                    # so the first retarget() frame converges from near-optimum.
                    _seeded = _init_and_seed_hand_retargeter()
                    if _seeded is not None:
                        prev_hand_qpos = _seeded
                    audio.play("begin")
                    # The gate is evaluated only on policy ticks.  Subtract one
                    # nominal tick so the first tick after expiry stays within
                    # the configured wall-time budget (0.35 s by default).
                    _begin_audio_gate_deadline_s = time.monotonic() + max(
                        0.0, cfg.begin_motion_gate_timeout_s - ctrl_dt
                    )
                    _ignore_begin_audio_until_silent = False
                    # Treat the cue as a hold transition even if the audio thread
                    # has not yet spawned its player process on the next tick.
                    _prev_audio_playing = True
                    print(f"\nB: 遥操作+录制开始  episode={recorder.frame_count}")
                    # Episode setup, GC, and retargeter seeding are intentional
                    # boundary work, not an active control-loop overrun.
                    limiter.reset()
                    skip_rest = True

            if not shared.is_running.value:
                break
            if skip_rest:
                continue

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

            vr_frame = _read_vr_frame(shared)
            vr_stale = vr_frame is None or (
                (time.monotonic_ns() - vr_frame.get("local_recv_ns", 0)) > cfg.vr_stale_threshold_s * 1e9
            )
            stage_timer.mark("vr")

            cam = _read_camera_frame(shared)
            stage_timer.mark("cam")

            hand_state = _read_hand_state(shared)
            hand_tactile = _read_hand_tactile(shared)
            # Forward-fill: hand_loop publishes hand_tactile_ring sparsely
            # (contact-only writes — saves ~14.4 KB/tick @ 30 Hz).  Cache
            # the last non-None read so held and active frames reuse it when
            # the ring has no new data.  Avoids allocating zeros(5,120,3)
            # (~14 KB) per frame.
            if hand_tactile is not None:
                _last_tactile_data = hand_tactile
            elif _last_tactile_data is not None:
                hand_tactile = _last_tactile_data

            # Single-frame RS485/EtherCAT glitches → retargeter keeps running (commands
            # land on a silent bus — harmless).  Only degrade to hold-position mode after
            # a persistent disconnect exceeding hand_disconnect_timeout_s.
            _hand_connected = bool(hand_state["connected"][0]) if hand_state is not None else False
            if _hand_connected:
                if not hand_available:
                    # Recovery: hand was unavailable but is now connected.
                    # Two scenarios: (a) was True→False→True (runtime glitch,
                    # _hand_disconnected_at is set), or (b) was False from init
                    # (_hand_disconnected_at is None — cold-start connect).
                    if _hand_disconnected_at is not None:
                        _dt = time.monotonic() - _hand_disconnected_at
                        logger.info("Hand recovered after %.1fs — re-enabling hand control", _dt)
                    else:
                        logger.info("Hand connected — enabling hand control (cold-start)")
                    hand_available = True
                    _seeded = _init_and_seed_hand_retargeter()
                    if _seeded is not None:
                        prev_hand_qpos = _seeded
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

            if not teleop_active or vr_stale:
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
                    )
                prev_qpos_cmd = arm_qpos.copy()
                ema_prev_pos = ema_prev_quat = None
                continue

            # Hold during state-transition voice prompts.  The begin cue is
            # special: it may block motion only for a bounded interval, then it
            # continues playing in the background.  Other safety/state cues keep
            # their existing full-duration gate.
            _audio_playing = audio.is_playing
            _hold_for_audio, _begin_audio_gate_deadline_s, _ignore_begin_audio_until_silent = _update_audio_motion_gate(
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
                _prev_audio_playing = True
                continue

            if _prev_audio_playing:
                # The audio motion gate just ended — re-establish wrist→EEF
                # mapping from the current pose to prevent a jump.  Resetting
                # captures any VR-hand motion during the gate, so the first map()
                # produces a near-zero delta (same pattern as C-resume handler).
                if arm_state is not None and vr_frame is not None:
                    _eef_pos = arm_state["eef_pos"][0].copy()
                    _eef_rot6d = arm_state["eef_rot6d"][0].copy()
                    if np.all(np.isfinite(_eef_pos)) and np.all(np.isfinite(_eef_rot6d)):
                        # Dynamic heading calibration: set vr_to_base_rot from the
                        # operator's current head orientation so position mapping
                        # adapts to where the operator is actually facing.
                        _head_q = vr_frame.get("head_quat_wxyz")
                        if _head_q is not None and np.all(np.isfinite(_head_q)) and not np.allclose(_head_q, 0):
                            arm_mapper.set_heading(np.asarray(_head_q, dtype=np.float64))
                        arm_mapper.reset(
                            wrist_pos=vr_frame["wrist_pos"],
                            wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
                            eef_pos=_eef_pos,
                            eef_quat_wxyz=rot6d_to_quat_wxyz(_eef_rot6d),
                        )
                # Clear EMA state: the mapper was just reset above so
                # the first post-audio map() produces a near-zero delta.
                # Letting EMA start from None causes the first frame to
                # skip smoothing (no prior to interpolate against), which
                # is correct — the re-reset mapper output IS the current
                # arm pose in world frame.
                # IMPORTANT: do NOT seed from arm_state["eef_pos"] — that
                # value is in base frame (Pinocchio FK), while mapper
                # output is in world frame (identity transform: base=world).
                # For consistency, always use the mapper's own output.
                ema_prev_pos = ema_prev_quat = None
                _hand_ramp_start = prev_hand_qpos.copy() if hand_available else None
                _hand_ramp_step = 0
                # Re-seed hand retargeter NLP warm-start from current hardware
                # pose (mirrors the arm_mapper reset above).  Without this, the
                # optimizer initial guess remains the qpos captured at B press.
                # Smoothstep ramp + one-frame SLSQP self-correction make this
                # harmless in practice, but re-seeding is symmetric and clear.
                _reset_hand_retargeter(hand_retargeter, prev_hand_qpos.copy() if hand_available else None)
                _prev_audio_playing = False

            if vr_frame is None:
                logger.warning("policy_loop: vr_frame is None after vr_stale check — holding")
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
                    )
                continue
            _policy_compute_t0 = time.perf_counter()
            _map_t0 = time.perf_counter()
            mapped = arm_mapper.map(vr_frame["wrist_pos"], vr_frame["wrist_quat_wxyz"])
            if mapped is None:
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
                    )
                if not _safe_arm_queue_put(shared, {"qpos": prev_qpos_cmd.copy(), "is_hold": True}):
                    shared.is_running.value = False
                    break
                continue

            target_pos_raw = np.asarray(mapped["pos"], dtype=np.float64).copy()
            target_quat_raw = np.asarray(mapped["quat_wxyz"], dtype=np.float64).copy()
            target_pos = target_pos_raw.copy()
            target_quat = target_quat_raw.copy()

            if ema_prev_pos is not None:
                if ema_prev_quat is None:
                    logger.warning("policy_loop: ema_prev_quat is None but ema_prev_pos is set — skipping EMA")
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
                cfg.contact_stall_enabled
                and arm_state is not None
                and _contact_stall_detected(
                    arm_qpos,
                    arm_state["qvel"][0],
                    prev_qpos_cmd,
                    arm_state["eef_pos"][0],
                    target_pos,
                    table_z_surface_m=cfg.contact_stall_table_z_surface_m,
                    table_context_height_m=cfg.contact_stall_table_context_height_m,
                    min_downward_target_m=cfg.contact_stall_min_downward_target_m,
                    tracking_error_rad=cfg.contact_stall_tracking_error_rad,
                    max_closing_speed_rad_s=cfg.contact_stall_max_closing_speed_rad_s,
                )
            ):
                command_error = prev_qpos_cmd - arm_qpos
                closing_speed = float(
                    np.dot(arm_state["qvel"][0], command_error) / max(np.linalg.norm(command_error), 1e-12)
                )
                logger.warning(
                    "policy_loop: downward contact stall — resync measured pose "
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
                    )
                if not _safe_arm_queue_put(shared, {"qpos": hold_qpos, "is_hold": True}):
                    shared.is_running.value = False
                    break
                continue

            planner.set_hand_qpos(prev_hand_qpos)  # sync hand pose for collision checks
            target_pose = Pose(p=target_pos, q=target_quat)
            _ik_t0 = time.perf_counter()
            ik_result = planner.solve_teleop_ik(target_pose, arm_qpos, prev_qpos_cmd)
            ik_solve_time_ms = (time.perf_counter() - _ik_t0) * 1000.0
            stage_timer.mark("ik")

            # Compute hand retargeting before IK outcome handling. Hand-only motion
            # remains allowed on IK failure only after arm↔hand transition validation.
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

            # G1: detect hand driver board lockout (hand_loop → qpos_stale in state ring).
            # When the hand stops executing commands despite being connected, hold the
            # last known-good command to prevent a gap jump on recovery.
            if hand_state is not None and bool(hand_state["qpos_stale"][0]) and hand_available:
                _hand_stale_warn("policy_loop: hand qpos stale — holding position")
                hand_cmd = prev_hand_qpos.copy()
                retarget_ok = False

            hand_start_qpos = prev_hand_qpos.copy()
            if hand_state is not None and np.all(np.isfinite(hand_state["qpos"][0])):
                hand_start_qpos = np.asarray(hand_state["qpos"][0], dtype=np.float64).copy()
            hand_cmd_valid = True
            try:
                hand_cmd = _sanitize_hand_command(
                    hand_cmd, prev_hand_qpos, HAND_QPOS_LO, HAND_QPOS_HI, cfg.hand_max_delta_rad
                )
            except ValueError:
                _validate_warn("policy_loop: invalid hand command — holding")
                hand_cmd = prev_hand_qpos.copy()
                hand_cmd_valid = False
                retarget_ok = False

            if not ik_result.success or ik_result.qpos is None:
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
                    )
                if not _safe_arm_queue_put(shared, {"qpos": prev_qpos_cmd.copy(), "is_hold": True}):
                    shared.is_running.value = False
                    break
                # The arm is held, but hand motion still needs arm↔hand collision validation.
                hand_safe = hand_cmd_valid and _transition_collision_free(
                    planner, arm_qpos, prev_qpos_cmd, hand_start_qpos, hand_cmd
                )
                if hand_available:
                    safe_hand_cmd = hand_cmd if hand_safe else prev_hand_qpos
                    if not hand_safe:
                        _validate_warn("policy_loop: hand-only transition rejected — holding")
                    _write_hand_cmd(shared, safe_hand_cmd)
                    prev_hand_qpos = safe_hand_cmd.copy()
                continue

            # IK delta clamp
            arm_cmd = np.asarray(ik_result.qpos, dtype=np.float64)
            arm_cmd = prev_qpos_cmd + np.clip(arm_cmd - prev_qpos_cmd, -arm_cmd_max_step_rad, arm_cmd_max_step_rad)

            # Joint-space hard limit clip (P4 — prevents C9/C31 cascade when
            # IK drifts beyond firmware-accepted bounds).
            arm_cmd = np.clip(arm_cmd, JOINT_LO, JOINT_HI)

            # Validate cheap boundaries before entering Pinocchio/hpp-fcl.
            _transition_check_t0 = time.perf_counter()
            _arm_ok = arm_state is not None and bool(arm_state["connected"][0])
            _reject = False
            _reject_reason = ""
            if not np.all(np.isfinite(arm_cmd)):
                _reject = True
                _reject_reason = "arm_cmd NaN/Inf"
            elif not hand_cmd_valid:
                _reject = True
                _reject_reason = "hand_cmd NaN/Inf"
            elif not _arm_ok:
                _reject = True
                _reject_reason = "arm disconnected"
            elif not planner.is_workspace_segment_safe(arm_qpos, arm_cmd):
                _reject = True
                _reject_reason = "final arm transition leaves workspace"
            elif not _transition_collision_free(planner, arm_qpos, arm_cmd, hand_start_qpos, hand_cmd):
                _reject = True
                _reject_reason = "final arm-hand transition collision"
            transition_check_time_ms = (time.perf_counter() - _transition_check_t0) * 1000.0
            if _reject:
                _validate_warn("policy_loop: action rejected — %s", _reject_reason)
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
                    )
                if not _safe_arm_queue_put(shared, {"qpos": prev_qpos_cmd.copy(), "is_hold": True}):
                    shared.is_running.value = False
                    break
                # Any joint transition rejection holds both independently driven workers.
                if hand_available:
                    _write_hand_cmd(shared, prev_hand_qpos.copy())
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
                    )
                continue
            if not _safe_arm_queue_put(shared, {"qpos": arm_cmd.copy()}):
                logger.error("policy_loop: arm_action_q full on main send — arm unresponsive")
                shared.is_running.value = False
                break
            _write_hand_cmd(shared, hand_cmd)
            stage_timer.mark("send")

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
                    policy_map_time_ms=policy_map_time_ms,
                    hand_retarget_time_ms=hand_retarget_time_ms,
                    transition_check_time_ms=transition_check_time_ms,
                    policy_compute_time_ms=policy_compute_time_ms,
                    hand_fk=_hand_fk,
                    T_eef_handbase_pos=_T_eef_handbase_pos,
                    T_eef_handbase_quat_wxyz=_T_eef_handbase_quat_wxyz,
                )
            stage_timer.mark("rec")

    finally:
        if recording_active:
            _stop_recording(recorder, True, save=False)
        recorder.join_stop(timeout=60)
        kb.stop()
        shared.is_running.value = False
        audio.play("end")
        time.sleep(2.0)
        logger.info("Policy: loop exited")


def _safe_arm_queue_put(shared: SharedStorage, action, *, timeout: float = 0.2) -> bool:
    """Stamp and put an action onto arm_action_q with timeout protection.

    Returns True on success, False if the queue was full (arm_loop dead or
    severely backed up).  The default 0.2s timeout (~3 policy frames @16Hz)
    balances fast fault detection against bounded arm_loop C24 recovery
    (~200ms worst case); C22/C31 transition directly to FAULT.  A false-positive timeout triggers clean shutdown,
    not FAULT.
    """
    from queue import Full

    try:
        stamped_action = make_arm_action(
            shared,
            np.asarray(action["qpos"], dtype=np.float64),
            is_hold=bool(action.get("is_hold", False)),
        )
        shared.arm_action_q.put(stamped_action, block=True, timeout=timeout)
        return True
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("policy_loop: rejected invalid arm action: %s", exc)
        return False
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
            return {"header": result[0], "rgb": result[1], "depth": result[2], "pointcloud": result[3]}
    except Exception:
        logger.warning("policy_loop: camera ring read failed", exc_info=True)
    return None


def _try_reset_mapper(shared: SharedStorage) -> bool:
    """Validate that arm and VR state are available for resume-calibrate initiation. Returns True on success."""
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
    return True


def _stop_recording(
    recorder: EpisodeRecorder, was_active: bool, *, save: bool, shared: SharedStorage | None = None
) -> None:
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
    hand_fk=None,
    T_eef_handbase_pos: np.ndarray | None = None,
    T_eef_handbase_quat_wxyz: np.ndarray | None = None,
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
    state = _build_robot_state(
        arm_state,
        hand_state,
        hand_tactile,
        hk=hand_fk,
        T_eef_handbase_pos=T_eef_handbase_pos,
        T_eef_handbase_quat_wxyz=T_eef_handbase_quat_wxyz,
    )
    recorder.add_frame(
        state,
        action,
        vr_frame,
        camera_frame=cam,
        signals={
            "ik_ok": False,
            "ik_attempted": frame_status != _FRAME_HELD,
            "retarget_ok": retarget_ok,
            "held": True,
            "flag_safety_reject": frame_status == _FRAME_SAFETY_REJECT,
            "frame_status": frame_status,
        },
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
    target_eef_pos_raw: np.ndarray | None = None,
    target_eef_rot6d_raw: np.ndarray | None = None,
    action_hand_joint_raw: np.ndarray | None = None,
    policy_map_time_ms: float = np.nan,
    hand_retarget_time_ms: float = np.nan,
    transition_check_time_ms: float = np.nan,
    policy_compute_time_ms: float = np.nan,
    hand_fk=None,
    T_eef_handbase_pos: np.ndarray | None = None,
    T_eef_handbase_quat_wxyz: np.ndarray | None = None,
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
    state = _build_robot_state(
        arm_state,
        hand_state,
        hand_tactile,
        hk=hand_fk,
        T_eef_handbase_pos=T_eef_handbase_pos,
        T_eef_handbase_quat_wxyz=T_eef_handbase_quat_wxyz,
    )
    head_quat = vr_frame.get("head_quat_wxyz") if vr_frame is not None else None
    _vr = (
        vr_frame
        if vr_frame is not None
        else {
            "wrist_pos": np.full(3, np.nan),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "landmarks": np.full((21, 3), np.nan),
        }
    )
    recorder.add_frame(
        state,
        action,
        _vr,
        camera_frame=cam,
        signals={
            "ik_ok": True,
            "ik_attempted": True,
            "retarget_ok": retarget_ok,
            "held": False,
            "flag_safety_reject": frame_status == _FRAME_SAFETY_REJECT,
            "frame_status": frame_status,
        },
        arm_qpos_sent=arm_cmd.copy(),
        diagnostics={
            "tracking_error": (
                float(arm_state["tracking_err"][0])
                if arm_state is not None and "tracking_err" in arm_state.dtype.names
                else 0.0
            ),
            "ik_solve_time_ms": ik_solve_time_ms,
            "target_pos_before_clamp": target_pos_before_clamp,
            "head_quat_wxyz": head_quat if head_quat is not None else np.full(4, np.nan),
            "target_eef_pos_raw": (
                np.asarray(target_eef_pos_raw, dtype=np.float64)
                if target_eef_pos_raw is not None
                else np.full(3, np.nan)
            ),
            "target_eef_rot6d_raw": (
                np.asarray(target_eef_rot6d_raw, dtype=np.float64)
                if target_eef_rot6d_raw is not None
                else np.full(6, np.nan)
            ),
            "action_hand_joint_raw": (
                np.asarray(action_hand_joint_raw, dtype=np.float64)
                if action_hand_joint_raw is not None
                else hand_cmd.copy()
            ),
            "policy_map_time_ms": policy_map_time_ms,
            "hand_retarget_time_ms": hand_retarget_time_ms,
            "transition_check_time_ms": transition_check_time_ms,
            "policy_compute_time_ms": policy_compute_time_ms,
        },
    )


def _build_robot_state(
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    hand_tactile: np.ndarray | None = None,
    hk: HandKinematics | None = None,
    T_eef_handbase_pos: np.ndarray | None = None,
    T_eef_handbase_quat_wxyz: np.ndarray | None = None,
) -> RobotState:
    """Build a RobotState from ring data for recording.

    Reads arm_state_ring + hand_state_ring + hand_tactile_ring and assembles
    a complete RobotState.  Computes world-frame fingertip positions via hand FK
    chain (handbase -> fingertip -> world via EEF transform).

    Hand health flags (qpos_stale, error_state) are read from HAND_STATE_DTYPE
    and forwarded to RobotState for recording.
    """
    if arm_state is not None:
        r = arm_state[0]
        arm_qpos = np.asarray(r["qpos"], dtype=np.float64)
        arm_qvel = np.asarray(r["qvel"], dtype=np.float64)
        arm_tau = np.asarray(r["tau"], dtype=np.float64)
        eef_pos = np.asarray(r["eef_pos"], dtype=np.float64)
        eef_rot6d = np.asarray(r["eef_rot6d"], dtype=np.float64)
        arm_connected = bool(r["connected"])
        arm_last_cmd_seq = int(r["last_cmd_seq"]) if "last_cmd_seq" in r.dtype.names else 0
        arm_last_cmd_queue_latency_s = (
            float(r["last_cmd_queue_latency_s"]) if "last_cmd_queue_latency_s" in r.dtype.names else 0.0
        )
        arm_last_cmd_apply_latency_s = (
            float(r["last_cmd_apply_latency_s"]) if "last_cmd_apply_latency_s" in r.dtype.names else 0.0
        )
        arm_last_cmd_sdk_duration_s = (
            float(r["last_cmd_sdk_duration_s"]) if "last_cmd_sdk_duration_s" in r.dtype.names else 0.0
        )
        arm_last_cmd_is_hold = bool(r["last_cmd_is_hold"]) if "last_cmd_is_hold" in r.dtype.names else False
    else:
        arm_qpos = nan_array(7)
        arm_qvel = nan_array(7)
        arm_tau = nan_array(7)
        eef_pos = nan_array(3)
        eef_rot6d = nan_array(6)
        arm_connected = False
        arm_last_cmd_seq = 0
        arm_last_cmd_queue_latency_s = 0.0
        arm_last_cmd_apply_latency_s = 0.0
        arm_last_cmd_sdk_duration_s = 0.0
        arm_last_cmd_is_hold = False

    if hand_state is not None:
        h = hand_state[0]
        hand_qpos = np.asarray(h["qpos"], dtype=np.float64)
        hand_current = np.asarray(h["current"], dtype=np.float64)
        hand_tactile_sum = np.asarray(h["tactile_sum"], dtype=np.float64)
        hand_tactile_contact = np.asarray(h["tactile_contact"], dtype=bool)
        hand_connected = bool(h["connected"])
        hand_qpos_stale = bool(h["qpos_stale"]) if "qpos_stale" in h.dtype.names else False
        hand_error_state = bool(h["error_state"]) if "error_state" in h.dtype.names else False
        # Board error registers — per-joint hardware fault indicators.
        hand_commboard_err = (
            np.asarray(h["commboard_err"], dtype=np.int32)
            if "commboard_err" in h.dtype.names
            else np.zeros(12, dtype=np.int32)
        )
        hand_jointboard_err = (
            np.asarray(h["jointboard_err"], dtype=np.int32)
            if "jointboard_err" in h.dtype.names
            else np.zeros(12, dtype=np.int32)
        )
        hand_tipboard_err = (
            np.asarray(h["tipboard_err"], dtype=np.int32)
            if "tipboard_err" in h.dtype.names
            else np.zeros(12, dtype=np.int32)
        )
    else:
        hand_qpos = nan_array(12)
        hand_current = nan_array(12)
        hand_tactile_sum = nan_array((5, 3))
        hand_tactile_contact = np.zeros(5, dtype=bool)
        hand_connected = False
        hand_qpos_stale = False
        hand_error_state = False
        hand_commboard_err = np.zeros(12, dtype=np.int32)
        hand_jointboard_err = np.zeros(12, dtype=np.int32)
        hand_tipboard_err = np.zeros(12, dtype=np.int32)

    # Tactile force from separate ring (Phase 2.8)
    if hand_tactile is not None:
        hand_tactile_force = np.asarray(hand_tactile[0]["tactile_force"], dtype=np.float64)
    else:
        hand_tactile_force = np.zeros((5, 120, 3), dtype=np.float64)

    _eef_finite = np.all(np.isfinite(eef_rot6d))
    eef_quat_wxyz = rot6d_to_quat_wxyz(eef_rot6d) if _eef_finite else np.array([1.0, 0.0, 0.0, 0.0])

    # Compute world-frame fingertip positions via hand FK.
    fingertip_pos = nan_array((5, 3))
    if hk is not None and hk.is_ready() and hand_connected and np.all(np.isfinite(hand_qpos)):
        tips_in_handbase = hk.compute_tip_positions_in_handbase(hand_qpos)
        if np.all(np.isfinite(tips_in_handbase)) and _eef_finite and np.all(np.isfinite(eef_pos)):
            T_world_eef = Pose(p=eef_pos, q=eef_quat_wxyz)
            T_eef_handbase = Pose(
                p=T_eef_handbase_pos if T_eef_handbase_pos is not None else np.zeros(3),
                q=T_eef_handbase_quat_wxyz if T_eef_handbase_quat_wxyz is not None else np.array([1.0, 0.0, 0.0, 0.0]),
            )
            T_world_handbase = compose_pose(T_world_eef, T_eef_handbase)
            tips_world = np.zeros((5, 3), dtype=np.float64)
            _id_quat = np.array([1.0, 0.0, 0.0, 0.0])
            for i in range(5):
                T_world_tip = compose_pose(T_world_handbase, Pose(p=tips_in_handbase[i], q=_id_quat))
                tips_world[i] = T_world_tip.p
            fingertip_pos = tips_world

    return RobotState(
        arm_qpos=arm_qpos,
        arm_qvel=arm_qvel,
        arm_tau=arm_tau,
        eef_pos=eef_pos,
        eef_quat_wxyz=eef_quat_wxyz,
        eef_rot6d=eef_rot6d,
        hand_qpos=hand_qpos,
        hand_current=hand_current,
        hand_tactile_sum=hand_tactile_sum,
        hand_tactile_force=hand_tactile_force,
        hand_tactile_contact=hand_tactile_contact,
        hand_tipboard_err=hand_tipboard_err,
        hand_commboard_err=hand_commboard_err,
        hand_jointboard_err=hand_jointboard_err,
        hand_qpos_stale=hand_qpos_stale,
        hand_error_state=hand_error_state,
        arm_last_cmd_seq=arm_last_cmd_seq,
        arm_last_cmd_queue_latency_s=arm_last_cmd_queue_latency_s,
        arm_last_cmd_apply_latency_s=arm_last_cmd_apply_latency_s,
        arm_last_cmd_sdk_duration_s=arm_last_cmd_sdk_duration_s,
        arm_last_cmd_is_hold=arm_last_cmd_is_hold,
        fingertip_pos=fingertip_pos,
        arm_connected=arm_connected,
        hand_connected=hand_connected,
        timestamp=time.perf_counter(),
    )


_retarget_fail_warn = ThrottledWarner()


def _get_raw_hand_command(
    retargeter: TAGHandRetargeter | XHandRetargeter | None,
    filtered_command: np.ndarray,
    retarget_ok: bool,
) -> np.ndarray:
    """Return pre-EMA TAG output when available, otherwise the filtered command."""
    fallback = np.asarray(filtered_command, dtype=np.float64).copy()
    if not retarget_ok or retargeter is None:
        return fallback
    raw = getattr(retargeter, "last_raw_qpos", None)
    if raw is None:
        return fallback
    raw_arr = np.asarray(raw, dtype=np.float64)
    return raw_arr.copy() if raw_arr.shape == (12,) and np.all(np.isfinite(raw_arr)) else fallback


def _compute_hand_command(
    retargeter: TAGHandRetargeter | XHandRetargeter | None,
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
    retargeter: TAGHandRetargeter | XHandRetargeter | None,
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
            logger.warning("Hand retargeter reset failed — previous optimizer state retained", exc_info=True)


def _seed_hand_retargeter(
    retargeter: TAGHandRetargeter | XHandRetargeter | None,
    qpos: np.ndarray | None,
) -> np.ndarray | None:
    """Reset hand retargeter NLP warm-start from *qpos*.

    Returns a copy of *qpos* if valid (for seeding ``prev_hand_qpos``),
    else ``None``.
    """
    if qpos is not None and np.all(np.isfinite(qpos)):
        _reset_hand_retargeter(retargeter, qpos.copy())
        return qpos.copy()
    _reset_hand_retargeter(retargeter, None)
    return None


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
