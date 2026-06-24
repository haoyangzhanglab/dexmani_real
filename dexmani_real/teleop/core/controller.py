"""Teleoperation controller: state machine, _tick(), EMA smoothing, quality gating.

State machine:
    IDLE --T--> TELEOP --R--> RECORDING --S--> TELEOP
      |        |   S->IDLE      |   H->IDLE
      H        H                |
      v        v                v
  return_to_home          EMERGENCY_STOP (ESC / timeout)

Data sources:
    tracker: tracker.get_latest() directly
    dry-run: dummy state, no hardware
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.teleop.core.pipeline import TeleopPipeline
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.control import safety
from dexmani_real.teleop.control.safety import SlidingWindowMonitor
from dexmani_real.planning.pose_utils import quat_wxyz_to_rotmat
from dexmani_real.recording.collection_config import CollectionConfig
from dexmani_real.recording.collection_loop import CollectionLoop
from dexmani_real.robot.interface import (
    RobotAction,
    RobotInterface,
    RobotInterfaceConfig,
    RobotState,
)
from dexmani_real.utils.rate_limiter import RateLimiter
from dexmani_real.utils.rate_manager import RateManager, StreamStats

if TYPE_CHECKING:
    from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
    from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
    from dexmani_real.planning.planner import XArm7MotionPlanner
    from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker
    from dexmani_real.recording.episode_recorder import EpisodeRecorder
    from dexmani_real.sensor.multi_camera_manager import MultiCameraManager

logger = get_logger(__name__)


class ControllerState(Enum):
    IDLE = "IDLE"
    TELEOP = "TELEOP"  # recording controlled by bool flag
    PAUSED = "PAUSED"
    SAVE_PROMPT = "SAVE_PROMPT"
    EMERGENCY_STOP = "EMERGENCY_STOP"


from dataclasses import dataclass


@dataclass
class TeleopControllerConfig:
    """Configuration for TeleopController runtime behavior.

    Collapses 8 scattered keyword arguments into a single cfg parameter.
    """

    target_hz: float = 50.0
    ema_alpha_arm: float = 0.75  # Light EMA smoothing for hand tremor filtering.
    # 1.0 = no smoothing; 0.75 = ~3-frame time constant at 50Hz (~60ms lag).
    # Smoothing hand tremor (~2-3mm) without perceptible latency for dexterous teleop.
    dry_run: bool = False
    use_zmq_vr: bool = False
    zmq_vr_port: int = 5555
    use_precise_wait: bool = False  # True → RateManager busy-wait; False → RateLimiter sleep

    # Collection config
    collection_config: CollectionConfig | None = None  # None → defaults

    # Multi-camera config
    multi_camera_configs: list | None = None  # None → single-camera (backward compat)
    multi_camera_auto_restart: bool = True

    # ── Tracking safety ──
    # Max single-joint deviation between commanded and actual position (rad).
    # When |q_actual[i] - q_command[i]| exceeds this threshold for any joint,
    # _consecutive_divergence increments.  Three consecutive divergences
    # trigger emergency stop.  Default 5.0 rad (~286°) — intentionally high:
    # only triggers on gross mechanical failure / encoder fault.
    # Ref: T-Rex arm_hand_control.py TRACKING_SAFETY_THRESHOLD.
    tracking_divergence_threshold_rad: float = 5.0

    # ── Velocity-limited step smoothing ──
    # Per-frame position-delta bottleneck between pipeline output and send_action.
    # Uses bottleneck scaling (same algorithm as XArm7._limit_joint_step):
    # when any joint exceeds its per-step velocity limit, ALL joints are
    # scaled proportionally to preserve the trajectory shape.
    #
    # This is NOT redundant with the PID inner loop's _clip_arm_velocity:
    # - PID _clip_arm_velocity limits VELOCITY at 250 Hz (per inner tick)
    # - This limits POSITION DELTA at 50 Hz (per controller tick)
    # Without this, VR jitter / IK noise can cause frame-to-frame position
    # jumps that, while individually within PID velocity limits, produce
    # perceptibly less smooth motion — the PID tracks each jump aggressively.
    use_velocity_limited_smooth: bool = True


class TeleopController:
    """Main teleoperation controller.

    Owns the control loop: reads VR, runs IK+retarget, applies EMA smoothing (arm only),
    enforces safety clamps, manages recording lifecycle.
    Hand retargeting smoothing is handled by dex-retargeting's built-in low_pass_alpha.

    The controller operates on RobotInterface (not XArm7/XHand directly).
    """

    # Retargeter auto-reload: after this many consecutive None returns,
    # the retargeter is automatically reloaded to recover from optimizer
    # divergence / memory corruption.
    _RETARGET_AUTO_RELOAD_THRESHOLD = 5

    # ── VR tracking thresholds (inlined from TrackingQuality/FrameDropPolicy) ──
    # Ref: BunnyVisionPro FrameAge gate — three-tier staleness classification.
    _VR_SOFT_STALE_S: float = 0.1  # age > 0.1s → soft deceleration
    _VR_HARD_STALE_S: float = 0.5  # age > 0.5s → hold position (tracking lost)
    _VR_EMERGENCY_S: float = 1.0  # age > 1.0s → emergency stop

    def __init__(
        self,
        robot: RobotInterface,
        arm_mapper: ArmWristMapper,
        retargeter: XHandRetargeter,
        planner: XArm7MotionPlanner,
        cfg: TeleopControllerConfig | None = None,
        *,
        tracker: QuestHandTracker | None = None,
        keyboard_queue: object | None = None,
        # Backward compat individual kwargs (deprecated; prefer cfg)
        target_hz: float = 50.0,
        ema_alpha_arm: float = 1.0,
        dry_run: bool = False,
        recorder: EpisodeRecorder | None = None,
        use_zmq_vr: bool = False,
        zmq_vr_port: int = 5555,
        camera_process: object | None = None,
    ) -> None:
        if cfg is None:
            cfg = TeleopControllerConfig(
                target_hz=target_hz,
                ema_alpha_arm=ema_alpha_arm,
                dry_run=dry_run,
                use_zmq_vr=use_zmq_vr,
                zmq_vr_port=zmq_vr_port,
            )

        self.robot = robot
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self.tracker = tracker
        self.dry_run = cfg.dry_run

        # Recording: prefer CollectionLoop (new), fallback to legacy EpisodeRecorder
        if recorder is not None:
            from dexmani_real.recording.collection_config import CollectionConfig
            from dexmani_real.recording.collection_loop import CollectionLoop

            coll_cfg = cfg.collection_config or CollectionConfig()
            self._collection_loop = CollectionLoop(recorder, coll_cfg)
            self.recorder = recorder  # backward compat
        else:
            self._collection_loop = None
            self.recorder = None

        self.limiter = RateLimiter(cfg.target_hz)
        self.rate_manager = RateManager(cfg.target_hz) if cfg.use_precise_wait else None
        self.ema_alpha_arm = float(cfg.ema_alpha_arm)

        # Stream statistics
        self.vr_stats = StreamStats(name="vr_track", target_hz=120.0)
        self.camera_stats = StreamStats(name="camera", target_hz=30.0)

        # Camera: MultiCameraManager (new) or legacy single CameraProcess
        self._camera_process = camera_process
        self._multi_camera: MultiCameraManager | None = None
        if cfg.multi_camera_configs is not None and len(cfg.multi_camera_configs) > 0:
            from dexmani_real.sensor.multi_camera_manager import (
                MultiCameraConfig,
                MultiCameraManager,
            )

            mc_cfg = MultiCameraConfig(auto_restart=cfg.multi_camera_auto_restart)
            self._multi_camera = MultiCameraManager(cfg.multi_camera_configs, mc_cfg)
            logger.info(
                "MultiCameraManager created: %d camera(s)",
                self._multi_camera.n_cameras,
            )
            # Start camera processes if not already managed externally
            if camera_process is None:
                self._multi_camera.start_all()

        # ZMQ VR subscriber (optional, disabled by default).
        self._vr_subscriber: VRFrameSubscriber | None = None
        if cfg.use_zmq_vr:
            from dexmani_real.teleop.vr.vr_publisher import VRFrameSubscriber

            self._vr_subscriber = VRFrameSubscriber(sub_port=cfg.zmq_vr_port)
            self._vr_subscriber.connect()

        # ── VR tracking state (inlined from TrackingQuality) ──
        self._vr_lost_since: float | None = None

        # ── Hold-on-failure state (inlined from TeleopErrorHandler) ──
        # Ref: BunnyVisionPro — per-frame hold, no separate error handler class.
        # Last known-good positions; any pipeline failure returns a hold action.
        self._last_good_arm: np.ndarray | None = None
        self._last_good_hand: np.ndarray | None = None

        # State
        self.state = ControllerState.IDLE
        self.recording = False
        self.running = False
        self._last_arm_cmd: np.ndarray | None = None
        self._last_hand_cmd: np.ndarray | None = None

        # ── Tracking safety: command-vs-actual deviation monitoring ──
        # Ref: T-Rex arm_hand_control.py TRACKING_SAFETY_THRESHOLD
        self._consecutive_divergence: int = 0
        self._tracking_divergence_threshold = cfg.tracking_divergence_threshold_rad

        # ── Velocity-limited step smoothing ──
        self._use_vel_limited_smooth = cfg.use_velocity_limited_smooth

        # ── IK miss counter (Phase 2.2 — rate decoupling) ──
        self._ik_miss_count: int = 0
        self._ik_miss_total: int = 0
        self._ik_miss_max_consecutive: int = 0
        self._camera_frame_count: int = 0

        # ── Retargeter auto-reload (P1.4) ──
        self._retarget_consecutive_none: int = 0

        # ── Fingertip desk safety (P1.2) ──
        self._desk_safety_check_enabled: bool = True

        # ── Episode telemetry (P2.1) ──
        self._episode_start_time: float | None = None

        # ── Sliding window monitors (P3.3) ──
        self._hand_temp_monitor = SlidingWindowMonitor(
            window_size=50,
            warn_threshold=55.0,
            warn_fraction=0.6,
        )
        self._hand_current_monitor = SlidingWindowMonitor(
            window_size=50,
            warn_threshold=400.0,
            warn_fraction=0.6,
        )

        # Pipeline — shared action computation (extracted from controller)
        self.pipeline = TeleopPipeline(
            arm_mapper,
            retargeter,
            planner,
            ema_alpha_arm=self.ema_alpha_arm,
        )

        # Keyboard
        self.keyboard = KeyboardHandler(keyboard_queue) if keyboard_queue is not None else None

        # Status
        self.frame_count: int = 0
        self.ik_success_count: int = 0
        self.ik_fail_count: int = 0
        self.retarget_success_count: int = 0
        self.retarget_fail_count: int = 0
        self.last_status_ts: float = 0.0
        self.status_interval: float = 2.0

        # Cancel event for return_to_home
        self._cancel_event = threading.Event()

    # Lifecycle

    def start(self) -> None:
        if self.keyboard is not None:
            self.keyboard.start()
        self.running = True

    def stop(self) -> None:
        self.running = False
        if self.keyboard is not None:
            self.keyboard.stop()

    def run(self) -> None:
        """Main control loop."""
        self.start()
        if not self.dry_run and not self.robot.is_connected():
            logger.info("Robot not connected. Attempting connect...")
            result = self.robot.connect()
            logger.info("connect result: %s", result)

        logger.info("Entering main loop at %.0f Hz", self.limiter.target_hz)
        logger.info("  Mode: %s", "dry-run" if self.dry_run else "hardware")
        logger.info("  VR: direct")
        logger.info("  EMA: arm_alpha=%s (hand uses dex-retargeting low_pass_alpha)", self.ema_alpha_arm)
        logger.info("  Controls: B=begin S=stop C=pause H=home Q=quit ESC=emergency")

        self.last_status_ts = time.monotonic()

        try:
            while self.running:
                self._handle_keyboard()
                self._tick()
                if self.rate_manager is not None:
                    self.rate_manager.wait()
                else:
                    self.limiter.wait()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping.")
        except (RuntimeError, ConnectionError, ValueError) as e:
            logger.exception("Unhandled exception in main loop: %s", e)
        finally:
            self._shutdown()

    # Soft deceleration

    def _apply_soft_deceleration(self, lost_duration_s: float) -> None:
        """Decelerate during VR stale window (0.2s < age < 1.0s).

        Velocity mode (Phase 2.4 None-sentinel): clears the PID target so
        the inner loop sends zero velocity → natural deceleration.
        Servo mode: holds current physical position.
        """
        if self._last_arm_cmd is None:
            return

        arm_cfg = self.robot.arm.config
        if not arm_cfg.use_servo_control:
            # Velocity mode: let PID decelerate naturally via None-sentinel
            self.robot.arm.clear_target()
            return

        # Servo mode: hold current position
        state = self._dummy_state() if self.dry_run else self.robot.get_state()
        hand_cmd = self._last_hand_cmd if self._last_hand_cmd is not None else state.hand_qpos
        arm_cmd, hand_cmd = TeleopPipeline.soft_deceleration(
            state.arm_qpos,
            state.hand_qpos,
        )
        action = RobotAction(
            arm_qpos_cmd=arm_cmd,
            hand_qpos_cmd=hand_cmd,
        )
        if not self.dry_run and self.state != ControllerState.IDLE:
            if self.robot.is_error():
                self._escalate_to_emergency("Robot error state detected during soft deceleration")
                return
            self.robot.send_action(action)

    # Velocity-limited step (per-frame command smoothing)

    def _apply_velocity_limited_step(self, action: RobotAction, state: RobotState) -> RobotAction:
        """Per-frame position-delta bottleneck between pipeline and send_action.

        Bottleneck-scales the joint delta to max_qvel * dt, preserving the
        trajectory shape (all joints scaled by the same worst-case ratio).

        This complements, but is NOT redundant with, the PID inner loop's
        _clip_arm_velocity (250 Hz velocity clipping):
        - PID _clip_arm_velocity: per-250Hz-tick VELOCITY ceiling
        - This method: per-50Hz-tick POSITION DELTA ceiling
        Without this layer, VR jitter / IK noise can cause frame-to-frame
        position jumps that the PID tracks aggressively, producing
        perceptibly less smooth motion even though each jump is within
        individual PID velocity limits.
        """
        if self._last_arm_cmd is None:
            return action

        dt = 1.0 / self.limiter.target_hz
        arm_cfg = self.robot.arm.config
        max_step = arm_cfg.max_qvel * dt  # rad per tick

        arm_target = np.asarray(action.arm_qpos_cmd, dtype=np.float64)
        prev_cmd = np.asarray(self._last_arm_cmd, dtype=np.float64)
        delta = arm_target - prev_cmd

        normalized = np.abs(delta) / max_step
        max_ratio = np.max(normalized)
        if max_ratio > 1.0:
            delta = delta / max_ratio
            action.arm_qpos_cmd = prev_cmd + delta
            logger.debug(
                "Velocity-limited step: max_ratio=%.2f joint=%d",
                max_ratio,
                int(np.argmax(normalized)) + 1,
            )

        return action

    # State machine tick

    def _tick(self) -> None:
        # ESTOP guard: freeze all motion, only keyboard polling continues.
        # The main loop stays alive so QUIT can set self.running = False.
        if self.state == ControllerState.EMERGENCY_STOP:
            return

        tick_start = time.perf_counter()
        self.frame_count += 1

        # PAUSED: decelerate naturally (velocity mode) or hold position (servo mode).
        # No VR reading or pipeline computation.
        if self.state == ControllerState.PAUSED:
            if not self.dry_run and self._last_arm_cmd is not None:
                # None-sentinel (Phase 2.4): for velocity control mode,
                # clear the PID target so the inner loop sends zero velocity
                # → natural deceleration.  For servo mode, hold position as before.
                arm_cfg = self.robot.arm.config
                if not arm_cfg.use_servo_control:
                    # Velocity mode: let PID decelerate naturally
                    self.robot.arm.clear_target()
                else:
                    # Servo mode: hold last commanded position.
                    # Only call get_state() when _last_hand_cmd is None
                    # (first PAUSED tick after startup without a prior command).
                    if self._last_hand_cmd is None:
                        state = self.robot.get_state()
                        hand_cmd = state.hand_qpos
                    else:
                        hand_cmd = self._last_hand_cmd
                    hold_action = RobotAction(
                        arm_qpos_cmd=self._last_arm_cmd,
                        hand_qpos_cmd=hand_cmd,
                    )
                    self.robot.send_action(hold_action)
            return

        # IDLE / SAVE_PROMPT: no pipeline (waiting for user input)
        if self.state in (ControllerState.IDLE, ControllerState.SAVE_PROMPT):
            return

        # 1. Get VR frame
        vr_frame = self._read_vr_frame()

        # 2. VR tracking quality gate (inlined from TrackingQuality)
        #    Ref: BunnyVisionPro FrameAge gate — three-tier staleness.
        if vr_frame is None:
            age_s = float("inf")
        else:
            age_s = self._frame_age(vr_frame)
            self.vr_stats.record_consume(age_s)

        if age_s > self._VR_SOFT_STALE_S:
            if self._vr_lost_since is None:
                self._vr_lost_since = time.perf_counter()
            lost_duration = time.perf_counter() - self._vr_lost_since

            if age_s > self._VR_EMERGENCY_S or lost_duration > self._VR_EMERGENCY_S:
                self._escalate_to_emergency(f"VR tracking lost for {lost_duration:.1f}s (age={age_s:.3f}s)")
                return

            if age_s > self._VR_HARD_STALE_S or lost_duration > self._VR_HARD_STALE_S:
                self._apply_soft_deceleration(lost_duration)
            return
        else:
            self._vr_lost_since = None  # frame is fresh

        # 3. Read robot state
        if self.dry_run:
            state = self._dummy_state()
        else:
            state = self.robot.get_state()

        # 3b. Tracking safety: command-vs-actual deviation check
        #     Ref: T-Rex arm_hand_control.py TRACKING_SAFETY_THRESHOLD (10 rad).
        #     Detects mechanical jams / encoder faults that may not trigger
        #     hardware-layer torque/current thresholds.
        if not self.dry_run and self._last_arm_cmd is not None:
            err = np.abs(state.arm_qpos - self._last_arm_cmd)
            if np.any(err > self._tracking_divergence_threshold):
                self._consecutive_divergence += 1
                logger.warning(
                    "Tracking divergence: max_err=%.2f rad frame=%s consecutive=%d",
                    float(np.max(err)),
                    self.frame_count,
                    self._consecutive_divergence,
                )
                if self._consecutive_divergence >= 3:
                    self._escalate_to_emergency(
                        f"Tracking divergence > {self._tracking_divergence_threshold} rad "
                        f"for 3 consecutive frames (max_err={np.max(err):.2f} rad)"
                    )
                    return
            else:
                self._consecutive_divergence = 0

        # 4. Compute action
        action, status = self._compute_action(vr_frame, state)

        # 4b. IK miss tracking (Phase 2.2 — rate decoupling)
        #     Increment counter on IK failure; reset on success.
        #     The hold-on-failure strategy (send last good target) ensures
        #     the PID inner loop continues to receive commands at 50 Hz
        #     even when IK temporarily fails — isolating IK jitter from
        #     command sending.
        if status["ik_ok"]:
            if self._ik_miss_count > 0:
                self._ik_miss_max_consecutive = max(self._ik_miss_max_consecutive, self._ik_miss_count)
            self._ik_miss_count = 0
        else:
            self._ik_miss_count += 1
            self._ik_miss_total += 1
            if self._ik_miss_count >= 3:
                logger.warning(
                    "IK missed %d consecutive frames — holding position",
                    self._ik_miss_count,
                )

        # 4c. Velocity-limited step — per-frame position-delta smoothing.
        #     Bottleneck-scales the pipeline output to prevent VR jitter / IK
        #     noise from producing perceptibly jerky motion between 50 Hz ticks.
        #     Complements (not redundant with) PID _clip_arm_velocity at 250 Hz.
        if self._use_vel_limited_smooth and self._last_arm_cmd is not None:
            action = self._apply_velocity_limited_step(action, state)

        # Update EMA/motion reference AFTER velocity-limited step so the
        # inter-frame delta represents the actual sent command.  Updating
        # before this point causes _apply_velocity_limited_step to always
        # see delta=0 (no-op bug).
        if status.get("ik_ok"):
            self._last_arm_cmd = action.arm_qpos_cmd.copy()
        if status.get("retarget_ok"):
            self._last_hand_cmd = action.hand_qpos_cmd.copy()

        # 4d. Sliding window cyclic limit monitors (P3.3)
        #     Tracks hand temp and current over a 50-tick (~1s) window to
        #     detect gradual degradation before the hard limit triggers.
        #     Complements per-tick threshold checks in validate_action() below.
        #     Reads only temperature + current from hand state (full=False for
        #     minimal overhead — skips tactile force sum computation).
        if not self.dry_run and self.state != ControllerState.IDLE:
            try:
                hand_state = self.robot.hand.get_state(full=False)
                if hand_state is not None:
                    hand_temp = hand_state.get("temperature")
                    hand_current = hand_state.get("current")
                    if hand_temp is not None and np.all(np.isfinite(hand_temp)):
                        _, temp_warn = self._hand_temp_monitor.update(hand_temp)
                        if temp_warn and self.frame_count % 50 == 0:
                            logger.warning(
                                "Hand temp elevated: mean=%.1f°C max=%.1f°C over %d ticks",
                                self._hand_temp_monitor.window_mean,
                                self._hand_temp_monitor.window_max,
                                self._hand_temp_monitor.window_size,
                            )
                    if hand_current is not None and np.all(np.isfinite(hand_current)):
                        _, curr_warn = self._hand_current_monitor.update(hand_current)
                        if curr_warn and self.frame_count % 50 == 0:
                            logger.warning(
                                "Hand current elevated: mean=%.1fmA max=%.1fmA over %d ticks",
                                self._hand_current_monitor.window_mean,
                                self._hand_current_monitor.window_max,
                                self._hand_current_monitor.window_size,
                            )
            except (ValueError, RuntimeError, KeyError, AttributeError) as e:
                logger.debug("Sliding window monitor update failed: %s", e)

        # 5. Record frame (before safety gate — captures pre-hold action)
        camera_frame = None
        camera_frames: dict[str, dict] | None = None

        # ── Multi-camera path (new) ──
        if self._multi_camera is not None:
            self._ensure_multi_camera_running()
            try:
                camera_frames = self._multi_camera.read_all_latest()
                has_camera = False
                for cam_name, cam_frame in camera_frames.items():
                    if cam_frame is not None:
                        has_camera = True
                        ts = cam_frame.get("timestamp", 0.0)
                        if ts > 0:
                            self.camera_stats.record_consume(time.perf_counter() - ts)
                        break  # stats for first camera only
                if has_camera:
                    self._camera_frame_count += 1
            except (ValueError, RuntimeError, KeyError) as e:
                logger.debug("Multi-camera poll failed: %s", e)

        # ── Single-camera path (backward compat) ──
        elif self._camera_process is not None:
            self._ensure_camera_running()
            try:
                camera_frame = self._camera_process.poll_latest_frame()
            except (ValueError, RuntimeError, KeyError):
                logger.debug("Camera poll failed — continuing without camera frame.")

            if camera_frame is not None:
                self._camera_frame_count += 1
                self.camera_stats.record_consume(
                    time.perf_counter() - camera_frame.get("timestamp", time.perf_counter())
                )

        # ── Recording (delegated to CollectionLoop if available) ──
        T_base_eef = self._compute_T_base_eef(state)

        # Pre-record buffer (Phase 3.1): always buffer frames so that
        # pressing R captures the last N seconds before the keypress.
        # Flushed to HDF5 on start_episode() before normal recording.
        if self._collection_loop is not None:
            try:
                self._collection_loop.add_pre_frame(
                    state=state,
                    action=action,
                    vr_frame=vr_frame,
                    camera_frame=camera_frame,
                    camera_frames=camera_frames,
                    T_base_eef=T_base_eef,
                )
            except (ValueError, OSError) as e:
                logger.debug("collection_loop add_pre_frame failed: %s", e)

        if self.recording:
            if self._collection_loop is not None:
                try:
                    self._collection_loop.record_frame(
                        state=state,
                        action=action,
                        vr_frame=vr_frame,
                        camera_frame=camera_frame,
                        camera_frames=camera_frames,
                        T_base_eef=T_base_eef,
                    )
                except (ValueError, OSError) as e:
                    logger.exception("collection_loop record_frame failed: %s", e)

        # 7. Pre-send safety gate — centralized validate_action (ref: ManiUniCon).
        #    Joint limits enforced by XArm7/XHand drivers (error latch → is_error).
        if not self.dry_run:
            action_valid, fail_reason = self.robot.validate_action(action)
            if not action_valid:
                if "error state" in fail_reason or "not connected" in fail_reason:
                    self._escalate_to_emergency(f"Robot error before send_action: {fail_reason}")
                    return
                logger.warning("Pre-send safety: %s — holding", fail_reason)
                hold = self._hold_action()
                action = RobotAction(
                    arm_qpos_cmd=hold.arm_qpos_cmd,
                    hand_qpos_cmd=hold.hand_qpos_cmd,
                )

            # ── Fingertip desk safety (P1.2) — real-time check before send_action ──
            # Checks whether the commanded arm qpos would place fingertips below
            # the table surface. Previously only checked during path planning
            # (planner.py:553-565), leaving the teleop hot path unprotected.
            desk_safe = True
            if self._desk_safety_check_enabled and self.planner.desk_safety is not None:
                desk_safe, min_z, _name = self.planner.desk_safety.check_hand_desk_clearance(action.arm_qpos_cmd)
                if not desk_safe:
                    logger.warning(
                        "Fingertip desk violation: min_z=%.4fm threshold=%.4fm — holding position",
                        min_z,
                        self.planner.desk_safety._threshold,
                    )
                    # Hold position instead of sending dangerous command
                    hold = self._hold_action()
                    action = RobotAction(
                        arm_qpos_cmd=hold.arm_qpos_cmd,
                        hand_qpos_cmd=hold.hand_qpos_cmd,
                    )

            if self.state != ControllerState.IDLE:
                result = self.robot.send_action(action)
                arm_ok = result.get("arm_ok", False)
                hand_ok = result.get("hand_ok", False)
                if not arm_ok or not hand_ok:
                    logger.warning("send_action: arm_ok=%s hand_ok=%s", arm_ok, hand_ok)

        # 8. Periodic status
        now = time.monotonic()
        if now - self.last_status_ts >= self.status_interval:
            self.last_status_ts = now
            self._print_status(vr_frame, now)

        # 9. Control loop overrun detection (ref: BunnyVisionPro wait_until_next_control_signal).
        #    Warns when a tick exceeds 150% of the target period — flags IK slowdowns,
        #    GC pauses, or system contention before they cause visible stutter.
        tick_elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
        target_ms = self.limiter.period * 1000.0
        if tick_elapsed_ms > target_ms * 1.5:
            logger.warning(
                "Loop overrun: tick=%.1fms target=%.1fms frame=%s",
                tick_elapsed_ms,
                target_ms,
                self.frame_count,
            )

    # ── VR tracking helpers (inlined from TrackingQuality) ──

    @staticmethod
    def _frame_age(frame: dict) -> float:
        """Seconds since VR frame was received on this machine.

        Falls back to frame["timestamp"] when local_recv_ns is missing
        (e.g. DummyTracker).  Only returns inf when both are absent.
        """
        local_recv = frame.get("local_recv_ns")
        if local_recv is not None and np.isfinite(local_recv):
            return (time.monotonic_ns() - local_recv) * 1e-9
        # Fallback: use frame's own timestamp (less precise but prevents
        # false EMERGENCY classification when tracker doesn't set local_recv_ns).
        ts = frame.get("timestamp")
        if ts is not None and np.isfinite(ts):
            return max(0.0, time.perf_counter() - ts)
        return float("inf")

    def _reset_vr_tracking(self) -> None:
        self._vr_lost_since = None

    # ── Hold-on-failure helpers (inlined from TeleopErrorHandler) ──

    def _init_fallback(self, arm_qpos: np.ndarray, hand_qpos: np.ndarray) -> None:
        """Init fallback positions from current state. Idempotent."""
        if self._last_good_arm is None:
            self._last_good_arm = np.asarray(arm_qpos, dtype=np.float64).copy()
        if self._last_good_hand is None:
            self._last_good_hand = np.asarray(hand_qpos, dtype=np.float64).copy()

    def _hold_action(self) -> RobotAction:
        """Return a hold-in-place action."""
        arm = self._last_good_arm.copy() if self._last_good_arm is not None else np.zeros(7, dtype=np.float64)
        hand = self._last_good_hand.copy() if self._last_good_hand is not None else np.zeros(12, dtype=np.float64)
        return RobotAction(arm_qpos_cmd=arm, hand_qpos_cmd=hand)

    # Action computation (split into sub-methods per Phase 3.2)

    def _compute_action(self, vr_frame: dict, state: RobotState) -> tuple[RobotAction, dict[str, bool]]:
        """Compute action via TeleopPipeline (shared with sim controller)."""
        current_arm_qpos = state.arm_qpos
        current_hand_qpos = state.hand_qpos

        prev_arm_cmd = self._last_arm_cmd if self._last_arm_cmd is not None else current_arm_qpos
        prev_hand_cmd = self._last_hand_cmd if self._last_hand_cmd is not None else current_hand_qpos

        # Load last_good for hold-on-failure
        self._init_fallback(current_arm_qpos, current_hand_qpos)

        # Delegate to shared pipeline
        action, status = self.pipeline.compute_action(
            vr_frame=vr_frame,
            current_arm_qpos=current_arm_qpos,
            current_hand_qpos=current_hand_qpos,
            prev_arm_cmd=prev_arm_cmd,
            prev_hand_cmd=prev_hand_cmd,
            check_workspace=self.robot.check_workspace,
            clamp_workspace_pos=self.robot.clamp_workspace_pos,
            last_arm_cmd=self._last_arm_cmd,
        )

        ik_ok = status["ik_ok"]
        retarget_ok = status["retarget_ok"]

        # Check retarget validity with hardware safety
        retarget_valid = safety.check_retarget_valid(action.hand_qpos_cmd)
        if not retarget_valid:
            retarget_ok = False

        # Update last-good positions for hold-on-failure
        if ik_ok and retarget_ok:
            self._last_good_arm = np.asarray(action.arm_qpos_cmd, dtype=np.float64).copy()
            self._last_good_hand = np.asarray(action.hand_qpos_cmd, dtype=np.float64).copy()

        # NOTE: _last_arm_cmd / _last_hand_cmd are NOT updated here.
        # They serve as the reference for velocity-limited step smoothing
        # and EMA in the pipeline.  They are updated in _tick() AFTER the
        # velocity-limited step so the delta between ticks reflects the
        # actual sent command, not the raw IK output.  Otherwise the
        # velocity-limited step would always see delta=0 and be a no-op.

        # Update counters
        if ik_ok:
            self.ik_success_count += 1
        else:
            self.ik_fail_count += 1
        if retarget_ok:
            self.retarget_success_count += 1
            self._retarget_consecutive_none = 0
        else:
            self.retarget_fail_count += 1
            # ── Retargeter auto-reload (P1.4) ──
            # When retarget() returns None for consecutive frames, the
            # optimizer may have diverged or the internal state may be
            # corrupted. Auto-reload the retargeter to recover without
            # requiring a full controller restart.
            self._retarget_consecutive_none += 1
            if self._retarget_consecutive_none >= self._RETARGET_AUTO_RELOAD_THRESHOLD:
                logger.warning(
                    "Retargeter returned None %d consecutive times — auto-reloading",
                    self._retarget_consecutive_none,
                )
                try:
                    self.retargeter.load_retargeter()
                    self._retarget_consecutive_none = 0
                    logger.info("Retargeter reloaded successfully.")
                except (ValueError, RuntimeError, OSError) as e:
                    logger.error("Retargeter auto-reload failed: %s", e)
                    # Don't reset counter — will retry on next failure

        return action, {"ik_ok": ik_ok, "retarget_ok": retarget_ok}

    # State machine transitions

    def _handle_keyboard(self) -> None:
        if self.keyboard is None:
            return
        signals = self.keyboard.poll()
        for sig in signals:
            self._transition(sig)

    def _transition(self, signal: ControlSignal) -> None:
        # ── EMERGENCY_STOP: any state ──
        if signal == ControlSignal.EMERGENCY_STOP:
            self._escalate_to_emergency("Keyboard ESC")
            return

        # ── ESTOP guard: only QUIT accepted ──
        if self.state == ControllerState.EMERGENCY_STOP:
            if signal == ControlSignal.QUIT:
                logger.info("ESTOP → exit")
                self.running = False
            return

        # ── QUIT: context-dependent ──
        if signal == ControlSignal.QUIT:
            if self.state == ControllerState.IDLE:
                logger.info("QUIT — shutting down.")
                self.running = False
            elif self.state == ControllerState.SAVE_PROMPT:
                # Discard episode → IDLE
                self._discard_episode()
                self.state = ControllerState.IDLE
                logger.info("SAVE_PROMPT → IDLE (discarded)")
            else:
                # TELEOP / PAUSED → stop recording → SAVE_PROMPT
                self._stop_recording()
                self.state = ControllerState.SAVE_PROMPT
                logger.info("Recording stopped → SAVE_PROMPT")
            return

        # ── BEGIN (B): IDLE → TELEOP + recording ──
        if signal == ControlSignal.BEGIN:
            if self.state == ControllerState.IDLE:
                self._reset_mapper()
                if not self.dry_run:
                    self.robot.reset_soft_start()
                self._start_recording()
                self.state = ControllerState.TELEOP
                self.recording = True
                logger.info("IDLE → TELEOP (recording started)")
            elif self.state == ControllerState.SAVE_PROMPT:
                logger.info("BEGIN ignored: in SAVE_PROMPT (press S to save, Q to discard)")
            return

        # ── STOP (S): context-dependent ──
        if signal == ControlSignal.STOP:
            if self.state == ControllerState.SAVE_PROMPT:
                # Confirm save
                self.state = ControllerState.IDLE
                self.recording = False
                logger.info("SAVE_PROMPT → IDLE (saved)")
            elif self.state in (ControllerState.TELEOP, ControllerState.PAUSED):
                # Stop recording → SAVE_PROMPT
                self._stop_recording()
                self.state = ControllerState.SAVE_PROMPT
                logger.info("Recording stopped → SAVE_PROMPT")
            return

        # ── PAUSE (C): TELEOP ⇄ PAUSED ──
        if signal == ControlSignal.PAUSE:
            if self.state == ControllerState.TELEOP:
                self.state = ControllerState.PAUSED
                logger.info("TELEOP → PAUSED (recording=%s)", self.recording)
            elif self.state == ControllerState.PAUSED:
                # Re-anchor mapper to compensate for drift during pause
                if self._reset_mapper():
                    self.state = ControllerState.TELEOP
                    logger.info("PAUSED → TELEOP (recording=%s, mapper re-anchored)", self.recording)
                else:
                    logger.warning("Cannot resume: no VR frame for mapper re-anchor")
            elif self.state == ControllerState.SAVE_PROMPT:
                logger.info("PAUSE ignored: in SAVE_PROMPT (press S to save, Q to discard)")
            return

        # ── HOME (H): any non-ESTOP / non-SAVE_PROMPT state ──
        if signal == ControlSignal.HOME:
            if self.state == ControllerState.SAVE_PROMPT:
                logger.info("HOME ignored: in SAVE_PROMPT (press S to save, Q to discard)")
            elif self.state != ControllerState.EMERGENCY_STOP:
                self._do_home()
            return

    def _do_home(self) -> None:
        logger.info("Returning to home...")
        self._last_arm_cmd = None
        self._last_hand_cmd = None

        # Stop recording if active (discard current episode)
        if self.recording:
            if self._collection_loop is not None and self._collection_loop.is_recording:
                try:
                    self._collection_loop.stop_episode(success=False, reason="home")
                except (ValueError, OSError):
                    pass

        if not self.dry_run:
            self.robot.return_to_home(use_planning=True, cancel_event=self._cancel_event)
        else:
            logger.info("  [dry-run] home (no hardware)")

        self.state = ControllerState.IDLE
        self.recording = False
        self._last_good_arm = None
        self._last_good_hand = None
        self._reset_vr_tracking()
        self._consecutive_divergence = 0
        self._ik_miss_count = 0
        self._ik_miss_total = 0
        self._ik_miss_max_consecutive = 0
        self._camera_frame_count = 0
        self._retarget_consecutive_none = 0
        self._episode_start_time = None
        self._hand_temp_monitor.reset()
        self._hand_current_monitor.reset()

        # Reset rate limiter — return_to_home blocks the main loop for seconds,
        # so the next limiter.wait() would otherwise see a huge elapsed time
        # and emit a spurious "Control loop over budget" warning.
        self.limiter.reset()
        if self.rate_manager is not None:
            self.rate_manager.reset()

        logger.info("Home complete.")

    def _reset_mapper(self) -> bool:
        """Re-anchor VR reference from current VR frame + robot state.

        Returns True on success, False if no VR frame available.
        Called on IDLE→TELEOP and TELEOP→RECORDING transitions.
        """
        vr_frame = self._read_vr_frame()
        if vr_frame is None:
            logger.warning("No VR frame available, cannot reset mapper.")
            return False

        if self.dry_run:
            state = self._dummy_state()
        else:
            state = self.robot.get_state()

        self.arm_mapper.reset(
            wrist_pos=vr_frame["wrist_pos"],
            wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
            eef_pos=state.eef_pos,
            eef_quat_wxyz=state.eef_quat_wxyz,
        )

        return True

    def _start_recording(self) -> None:
        """Start episode recording. Does NOT change controller state."""
        logger.info("Starting episode recording...")

        # Re-anchor VR reference
        if not self._reset_mapper():
            logger.error("Cannot start recording without VR frame.")
            return

        # Reset per-episode telemetry
        self._ik_miss_total = 0
        self._ik_miss_max_consecutive = 0
        self._camera_frame_count = 0
        self._retarget_consecutive_none = 0
        self._episode_start_time = time.perf_counter()

        if self._collection_loop is not None:
            try:
                self._collection_loop.start_episode(
                    task_label="teleop",
                    operator="",
                )
            except (ValueError, OSError) as e:
                logger.exception("collection_loop start_episode failed: %s", e)

        self._last_good_arm = None
        self._last_good_hand = None
        logger.info("Episode recording started.")

    def _stop_recording(self, classification: str = "success") -> None:
        """Stop episode recording. Does NOT change controller state.

        Args:
            classification: "success", "failure", or "partial" —
                used for sidecar JSON metadata and file directory routing.
        """
        logger.info("Stopping episode. frames=%s classification=%s", self.frame_count, classification)
        path = None
        if self._collection_loop is not None and self._collection_loop.is_recording:
            try:
                # Compute camera frame rate for sidecar JSON
                cam_fps: float | None = None
                if self._camera_frame_count > 0 and self._episode_start_time is not None:
                    dur = time.perf_counter() - self._episode_start_time
                    if dur > 0:
                        cam_fps = self._camera_frame_count / dur

                # Flush IK miss max on final tick
                if self._ik_miss_count > 0:
                    self._ik_miss_max_consecutive = max(self._ik_miss_max_consecutive, self._ik_miss_count)

                path = self._collection_loop.stop_episode(
                    success=(classification == "success"),
                    reason="manual",
                    classification=classification,
                    ik_success_rate=self._compute_ik_success_rate(),
                    vr_drop_rate=self._compute_vr_drop_rate(),
                    ik_miss_count=self._ik_miss_total,
                    ik_miss_max_consecutive=self._ik_miss_max_consecutive,
                    camera_frame_rate=cam_fps,
                )
                if path:
                    logger.info("  Saved to %s", path)
            except (ValueError, OSError) as e:
                logger.exception("collection_loop stop_episode failed: %s", e)
        self.recording = False
        logger.info("Episode stopped.")

    def _discard_episode(self) -> None:
        """Discard the current episode file (SAVE_PROMPT → discard)."""
        if self._collection_loop is not None:
            self._collection_loop.discard_episode()

    def _escalate_to_emergency(self, reason: str) -> None:
        logger.error("EMERGENCY_STOP: %s", reason)
        self.state = ControllerState.EMERGENCY_STOP
        if not self.dry_run:
            self.robot.emergency_stop()


    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        if self.recording:
            if self._collection_loop is not None and self._collection_loop.is_recording:
                try:
                    self._collection_loop.stop_episode(success=False, reason="shutdown")
                except (ValueError, RuntimeError, KeyError):
                    pass  # shutdown must never fail

        # Stop multi-camera if active
        if self._multi_camera is not None:
            try:
                self._multi_camera.stop_all()
            except (ValueError, RuntimeError):
                pass

        if self.keyboard is not None:
            self.keyboard.stop()
        logger.info("  Frames: %s", self.frame_count)
        logger.info("  IK: ok=%s fail=%s", self.ik_success_count, self.ik_fail_count)
        logger.info("  Retarget: ok=%s fail=%s", self.retarget_success_count, self.retarget_fail_count)

    # VR data source

    def _read_vr_frame(self) -> dict | None:
        if self._vr_subscriber is not None:
            return self._vr_subscriber.recv_latest()
        if self.tracker is not None:
            return self.tracker.get_latest()
        return None

    # Camera health

    def _ensure_camera_running(self) -> None:
        """Periodic camera health check with auto-restart.

        Restarts the camera daemon process if it has crashed, preventing
        silent data loss where an entire session runs without camera frames.
        Ref: ManiUniCon Camera Daemon Process crash isolation.
        """
        if self._camera_process is None:
            return
        if getattr(self._camera_process, "crashed", False):
            logger.warning("Camera process crashed — restarting...")
            try:
                self._camera_process.stop()
            except (ValueError, RuntimeError):
                pass
            time.sleep(0.5)
            self._camera_process.start()
            logger.info("Camera process restarted.")

    def _ensure_multi_camera_running(self) -> None:
        """Periodic multi-camera health check with auto-restart.

        Checks all camera processes and restarts any that have crashed.
        """
        if self._multi_camera is None:
            return
        now = time.perf_counter()
        # Throttle health checks to avoid overhead (every 5s)
        if now - getattr(self, "_last_mc_health_check", 0.0) < 5.0:
            return
        self._last_mc_health_check = now
        health = self._multi_camera.check_health()
        unhealthy = [name for name, ok in health.items() if not ok]
        if unhealthy:
            logger.warning("Unhealthy cameras: %s", unhealthy)

    # EEF pose utilities

    def _compute_T_base_eef(self, state: RobotState) -> np.ndarray | None:
        """Compute 4x4 T_base_eef from EEF pose for camera extrinsics."""
        if not np.all(np.isfinite(state.eef_pos)):
            return None
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = state.eef_pos
        T[:3, :3] = quat_wxyz_to_rotmat(state.eef_quat_wxyz)
        return T

    @staticmethod
    def _dummy_state() -> RobotState:
        return RobotState(
            arm_qpos=np.zeros(7, dtype=np.float64),
            arm_qvel=np.zeros(7, dtype=np.float64),
            arm_tau=np.zeros(7, dtype=np.float64),
            eef_pos=np.array([0.4, 0.0, 0.3], dtype=np.float64),
            eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            eef_rot6d=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            hand_qpos=np.zeros(12, dtype=np.float64),
            hand_tactile_sum=np.zeros((5, 3), dtype=np.float64),
            fingertip_pos=np.zeros((5, 3), dtype=np.float64),
            arm_connected=True,
            hand_connected=True,
            timestamp=time.perf_counter(),
        )

    # Status

    def _compute_ik_success_rate(self) -> float:
        total = self.ik_success_count + self.ik_fail_count
        return self.ik_success_count / total if total > 0 else 1.0

    def _compute_vr_drop_rate(self) -> float:
        """VR frame drop rate: fraction of ticks that had no valid VR frame.

        Uses (total_ticks - vr_consumed) / total_ticks as a conservative
        estimate — frames consumed count is incremented only when a valid
        VR frame passes the tracking quality gate.
        """
        total = max(self.frame_count, 1)
        return max(0.0, (total - self.vr_stats.consumed) / total)

    def _print_status(self, vr_frame: dict | None, now: float) -> None:
        age_s = self._frame_age(vr_frame) if vr_frame is not None else float("inf")
        seq = vr_frame.get("sequence_id", "?") if vr_frame else "?"
        rec = "REC" if self.recording else "   "
        vr_mean_age = self.vr_stats.mean_age_s * 1000
        cam_mean_age = self.camera_stats.mean_age_s * 1000
        logger.info(
            "[t=%.1f] frames=%s state=%s %s vr_seq=%s age=%sms "
            "vr_ema=%.0fms cam_ema=%.0fms "
            "ik=%s/%s retarget=%s/%s",
            now,
            self.frame_count,
            self.state.value,
            rec,
            seq,
            f"{age_s*1000:.0f}",
            vr_mean_age,
            cam_mean_age,
            self.ik_success_count,
            self.ik_fail_count,
            self.retarget_success_count,
            self.retarget_fail_count,
        )
