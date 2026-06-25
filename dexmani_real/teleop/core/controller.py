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
    ema_alpha_arm: float = 0.95  # Fixed EMA smoothing for arm joint commands.
    # 1.0 = no smoothing; 0.95 = light smoothing (~1-frame time constant at 50Hz, ~2ms lag).
    # Smoothing hand tremor (~2-3mm) with minimal perceptible latency for dexterous teleop.
    # Lower values (e.g. 0.5) = heavier smoothing, higher values (e.g. 0.95) = lighter.

    dry_run: bool = False
    use_zmq_vr: bool = False
    zmq_vr_port: int = 5555
    use_precise_wait: bool = False  # True → RateManager busy-wait; False → RateLimiter sleep

    # Collection config
    collection_config: CollectionConfig | None = None  # None → defaults

    # Multi-camera config
    multi_camera_configs: list | None = None  # None → single-camera (backward compat)
    multi_camera_auto_restart: bool = True

    # Velocity-limited step smoothing removed — PID inner loop at 250 Hz
    # provides sufficient per-tick velocity clipping (108°/s).  Aligned with
    # BunnyVisionPro which has no outer-loop position-delta limiting.


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
        ema_alpha_arm: float = 0.95,
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

        # Pipeline — shared action computation (extracted from controller)
        self.pipeline = TeleopPipeline(
            arm_mapper,
            retargeter,
            planner,
            ema_alpha_arm=self.ema_alpha_arm,
        )

        # Keyboard
        self.keyboard = KeyboardHandler(keyboard_queue) if keyboard_queue is not None else None

        # ── Idle quit double-tap confirmation (Opt 2) ──
        self._idle_quit_pending_ts: float | None = None

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

        # H3: PID inner thread alive monitor.
        # The PID daemon thread can die silently from an unhandled
        # exception (IndexError, TypeError, ValueError).  Check every
        # 50 frames (~1 s) so the controller can escalate to E-Stop
        # before the arm drifts on a stale velocity command.
        if self.frame_count % 50 == 0 and not self.dry_run:
            arm = self.robot.arm
            if (
                not arm.config.use_servo_control
                and arm._arm_thread is not None
                and not arm._arm_thread.is_alive()
            ):
                self._escalate_to_emergency("PID inner thread died")
                return

        # 4. Compute action
        action, status = self._compute_action(vr_frame, state)

        # Update motion reference for next frame's EMA and prev_cmd fallback.
        if status.get("ik_ok"):
            self._last_arm_cmd = action.arm_qpos_cmd.copy()
        if status.get("retarget_ok"):
            self._last_hand_cmd = action.hand_qpos_cmd.copy()

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

        # Delegate to shared pipeline.
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
        # Reset idle quit pending on any non-QUIT signal (Opt 2)
        if signal != ControlSignal.QUIT:
            self._idle_quit_pending_ts = None

        # ── EMERGENCY_STOP: any state ──
        if signal == ControlSignal.EMERGENCY_STOP:
            self._escalate_to_emergency("Keyboard ESC")
            return

        # ── ESTOP guard: Q=quit, H=recover+home ──
        if self.state == ControllerState.EMERGENCY_STOP:
            if signal == ControlSignal.QUIT:
                logger.info("=== STATE: ESTOP → EXIT ===")
                self.running = False
            elif signal == ControlSignal.HOME:
                # M1: recovery path from E-Stop — clear errors and
                # return home so the operator doesn't have to restart
                # the entire process after a spurious E-Stop.
                logger.info("=== STATE: ESTOP → recovery → HOME ===")
                if not self.dry_run:
                    self.robot.clear_error()
                self.state = ControllerState.IDLE
                self._do_home()
            else:
                logger.info(
                    "ESTOP active — Q=quit, H=recover+home (got %s)",
                    signal.value,
                )
            return

        # ── QUIT: context-dependent ──
        if signal == ControlSignal.QUIT:
            if self.state == ControllerState.IDLE:
                now = time.perf_counter()
                if self._idle_quit_pending_ts is not None and (now - self._idle_quit_pending_ts) < 2.0:
                    logger.info("=== STATE: IDLE → EXIT (double-tap confirmed) ===")
                    self.running = False
                else:
                    self._idle_quit_pending_ts = now
                    logger.info("QUIT pending — press Q again within 2s to confirm exit (any other key cancels)")
            elif self.state == ControllerState.SAVE_PROMPT:
                # Discard episode → IDLE
                self._discard_episode()
                self.state = ControllerState.IDLE
                logger.info("=== STATE: SAVE_PROMPT → IDLE (discarded) ===")
            else:
                # TELEOP / PAUSED → stop recording → SAVE_PROMPT
                old_state = self.state.value
                self._stop_recording()
                self.state = ControllerState.SAVE_PROMPT
                logger.info("=== STATE: %s → SAVE_PROMPT ===", old_state)
            return

        # ── BEGIN (B): IDLE → TELEOP + recording; SAVE_PROMPT → save+new episode ──
        if signal == ControlSignal.BEGIN:
            if self.state == ControllerState.IDLE:
                self._reset_mapper()
                if not self.dry_run:
                    self.robot.reset_soft_start()
                self._start_recording()
                self.state = ControllerState.TELEOP
                self.recording = True
                logger.info("=== STATE: IDLE → TELEOP (recording started) ===")
            elif self.state == ControllerState.SAVE_PROMPT:
                # Save current episode + start new episode directly (skip IDLE roundtrip)
                # Episode is already persisted by _stop_recording() — just start fresh.
                self._reset_mapper()
                if not self.dry_run:
                    self.robot.reset_soft_start()
                self._start_recording()
                self.state = ControllerState.TELEOP
                self.recording = True
                logger.info("=== STATE: SAVE_PROMPT → TELEOP (saved + new episode) ===")
            return

        # ── STOP (S): context-dependent ──
        if signal == ControlSignal.STOP:
            if self.state == ControllerState.SAVE_PROMPT:
                # Confirm save
                self.state = ControllerState.IDLE
                self.recording = False
                logger.info("=== STATE: SAVE_PROMPT → IDLE (saved) ===")
            elif self.state in (ControllerState.TELEOP, ControllerState.PAUSED):
                # Stop recording → SAVE_PROMPT
                old_state = self.state.value
                self._stop_recording()
                self.state = ControllerState.SAVE_PROMPT
                logger.info("=== STATE: %s → SAVE_PROMPT ===", old_state)
            return

        # ── PAUSE (C): TELEOP ⇄ PAUSED ──
        if signal == ControlSignal.PAUSE:
            if self.state == ControllerState.TELEOP:
                self.state = ControllerState.PAUSED
                logger.info("=== STATE: TELEOP → PAUSED (recording=%s) ===", self.recording)
            elif self.state == ControllerState.PAUSED:
                # Re-anchor mapper to compensate for drift during pause
                if self._reset_mapper():
                    self.state = ControllerState.TELEOP
                    # M2: reset soft-start so the first few frames after
                    # resume are speed-limited (30%%).  Without this, the
                    # PID has converged (vel_ramp_start=None) and resumes
                    # at full speed, risking a snap if the operator's hand
                    # has drifted during the pause.
                    if not self.dry_run:
                        self.robot.arm.reset_soft_start()
                    logger.info("=== STATE: PAUSED → TELEOP (recording=%s, mapper re-anchored) ===", self.recording)
                else:
                    lost_msg = ""
                    if self._vr_lost_since is not None:
                        lost_dur = time.perf_counter() - self._vr_lost_since
                        lost_msg = f" (VR lost for {lost_dur:.1f}s)"
                    logger.warning(
                        "Cannot resume: no VR frame for mapper re-anchor.%s "
                        "Check headset connection / HTS SDK. Use H to return home, Q to stop.",
                        lost_msg,
                    )
            elif self.state == ControllerState.SAVE_PROMPT:
                logger.info("PAUSE ignored: in SAVE_PROMPT (press S to save, Q to discard, B to save+continue)")
            return

        # ── HOME (H): any non-ESTOP / non-SAVE_PROMPT state ──
        if signal == ControlSignal.HOME:
            if self.state == ControllerState.SAVE_PROMPT:
                logger.info("HOME ignored: in SAVE_PROMPT (press S to save, Q to discard, B to save+continue)")
            elif self.state != ControllerState.EMERGENCY_STOP:
                logger.info("=== STATE: %s → HOME ===", self.state.value)
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
        self._ik_miss_count = 0
        self._ik_miss_total = 0
        self._ik_miss_max_consecutive = 0
        self._camera_frame_count = 0
        self._retarget_consecutive_none = 0
        self._episode_start_time = None

        # Reset rate limiter — return_to_home blocks the main loop for seconds,
        # so the next limiter.wait() would otherwise see a huge elapsed time
        # and emit a spurious "Control loop over budget" warning.
        self.limiter.reset()
        if self.rate_manager is not None:
            self.rate_manager.reset()

        logger.info("=== STATE: HOME → IDLE ===")

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
        logger.error("=== STATE: → EMERGENCY_STOP: %s ===", reason)
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
        # Episode elapsed time + actual fps
        ep_elapsed = ""
        actual_fps = ""
        if self.recording and self._episode_start_time is not None:
            ep_dur = now - self._episode_start_time
            ep_elapsed = f"ep_t={ep_dur:.1f}s "
            actual_fps = f"fps={self.frame_count / max(ep_dur, 0.001):.1f} "
        logger.info(
            "[t=%.1f] frames=%s %s%sstate=%s %s vr_seq=%s age=%sms "
            "vr_ema=%.0fms cam_ema=%.0fms "
            "ik=%s/%s retarget=%s/%s",
            now,
            self.frame_count,
            ep_elapsed,
            actual_fps,
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
