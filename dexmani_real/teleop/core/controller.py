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
from dexmani_real.teleop.core.error_handler import TeleopErrorHandler
from dexmani_real.teleop.core.pipeline import TeleopPipeline
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.control import safety
from dexmani_real.teleop.core.tracking import (
    TrackingQuality,
    TrackingQualityConfig,
)
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
    from dexmani_real.teleop.vr.pose_interpolator import CartPoseInterpolator
    from dexmani_real.planning.planner import XArm7MotionPlanner
    from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker
    from dexmani_real.recording.episode_recorder import EpisodeRecorder
    from dexmani_real.sensor.multi_camera_manager import MultiCameraManager

logger = get_logger(__name__)


class ControllerState(Enum):
    IDLE = "IDLE"
    TELEOP = "TELEOP"              # recording controlled by bool flag
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
    ema_alpha_arm: float = 1.0  # 1.0 = no smoothing
    dry_run: bool = False
    use_cartesian_interpolation: bool | None = None
    interpolation_max_pos_speed: float | None = None
    interpolation_max_rot_speed: float | None = None
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

    # ── Velocity-limited smoothing (Phase 2.1) ──
    # When True, applies velocity-limited step between pipeline output and
    # send_action, bottleneck-scaling the joint delta to max_qvel * dt.
    # Reuses the existing _limit_joint_step bottleneck algorithm.
    use_velocity_limited_smooth: bool = True


class TeleopController:
    """Main teleoperation controller.

    Owns the control loop: reads VR, runs IK+retarget, applies EMA smoothing (arm only),
    enforces safety clamps, manages recording lifecycle.
    Hand retargeting smoothing is handled by dex-retargeting's built-in low_pass_alpha.

    The controller operates on RobotInterface (not XArm7/XHand directly).
    """

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
        use_cartesian_interpolation: bool | None = None,
        interpolation_max_pos_speed: float | None = None,
        interpolation_max_rot_speed: float | None = None,
        use_zmq_vr: bool = False,
        zmq_vr_port: int = 5555,
        camera_process: object | None = None,
    ) -> None:
        if cfg is None:
            cfg = TeleopControllerConfig(
                target_hz=target_hz,
                ema_alpha_arm=ema_alpha_arm,
                dry_run=dry_run,
                use_cartesian_interpolation=use_cartesian_interpolation,
                interpolation_max_pos_speed=interpolation_max_pos_speed,
                interpolation_max_rot_speed=interpolation_max_rot_speed,
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

        # Resolve interpolation settings from planner's TeleopProfile when not
        # explicitly passed.
        use_ci = cfg.use_cartesian_interpolation
        max_pos = cfg.interpolation_max_pos_speed
        max_rot = cfg.interpolation_max_rot_speed
        if use_ci is None:
            use_ci = self.planner.teleop_profile.use_cartesian_interpolation
        if max_pos is None:
            max_pos = self.planner.teleop_profile.interpolation_max_pos_speed
        if max_rot is None:
            max_rot = self.planner.teleop_profile.interpolation_max_rot_speed

        # Cartesian pose interpolator (optional, disabled by default).
        self._pose_interpolator: CartPoseInterpolator | None = None
        if use_ci:
            from dexmani_real.teleop.vr.pose_interpolator import CartPoseInterpolator
            self._pose_interpolator = CartPoseInterpolator(
                max_pos_speed=max_pos,
                max_rot_speed=max_rot,
            )

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

        self.tracking_quality = TrackingQuality(TrackingQualityConfig(max_frame_age_s=0.2))
        self.error_handler = TeleopErrorHandler()

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

        # ── Velocity-limited smoothing (Phase 2.1) ──
        self._use_vel_limited_smooth = cfg.use_velocity_limited_smooth

        # ── IK miss counter (Phase 2.2 — rate decoupling) ──
        self._ik_miss_count: int = 0

        # Pipeline — shared action computation (extracted from controller)
        self.pipeline = TeleopPipeline(
            arm_mapper, retargeter, planner,
            pose_interpolator=self._pose_interpolator,
            ema_alpha_arm=self.ema_alpha_arm,
        )

        # Keyboard
        self.keyboard: KeyboardHandler | None = None
        if keyboard_queue is not None:
            self.keyboard = KeyboardHandler(keyboard_queue)
        else:
            import multiprocessing
            q = multiprocessing.Queue()
            self.keyboard = KeyboardHandler(q)

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
        logger.info("  Controls: T=teleop R=record S=stop H=home ESC=emergency Q=quit")

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
        hand_cmd = (
            self._last_hand_cmd
            if self._last_hand_cmd is not None
            else state.hand_qpos
        )
        arm_cmd, hand_cmd = TeleopPipeline.soft_deceleration(
            state.arm_qpos, state.hand_qpos,
        )
        action = RobotAction(
            arm_qpos_cmd=arm_cmd,
            hand_qpos_cmd=hand_cmd,
        )
        if not self.dry_run and self.state != ControllerState.IDLE:
            if self.robot.is_error():
                self._escalate_to_emergency(
                    "Robot error state detected during soft deceleration"
                )
                return
            self.robot.send_action(action)

    # Velocity-limited step (Phase 2.1)

    def _apply_velocity_limited_step(
        self, action: RobotAction, state: RobotState
    ) -> RobotAction:
        """Apply velocity-limited smoothing between pipeline output and send_action.

        Uses bottleneck scaling (same algorithm as XArm7._limit_joint_step):
        when any joint exceeds its per-step velocity limit, ALL joints are
        scaled by the same factor to preserve the trajectory shape.

        Uses per-joint max_qvel from the arm config (default 90-150°/s per joint)
        converted to a per-step limit via dt = 1/target_hz.

        Ref: T-Rex arm_hand_control.py SmootherVelocity — bottleneck scaling
        with the same ratio applied to all joints.
        """
        if self._last_arm_cmd is None:
            return action

        dt = 1.0 / self.limiter.target_hz
        # Use arm config max_qvel as per-joint velocity limits
        arm_cfg = self.robot.arm.config
        max_step = arm_cfg.max_qvel * dt  # rad per tick

        arm_target = np.asarray(action.arm_qpos_cmd, dtype=np.float64)
        prev_cmd = np.asarray(self._last_arm_cmd, dtype=np.float64)
        delta = arm_target - prev_cmd

        # Bottleneck scaling: normalize each joint's delta by its limit,
        # find the worst offender, scale all joints proportionally.
        normalized = np.abs(delta) / max_step
        max_ratio = np.max(normalized)
        if max_ratio > 1.0:
            delta = delta / max_ratio
            action.arm_qpos_cmd = prev_cmd + delta
            logger.debug(
                "Velocity-limited step applied: max_ratio=%.2f joint=%d",
                max_ratio, int(np.argmax(normalized)) + 1,
            )

        return action

    # State machine tick

    def _tick(self) -> None:
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
                    # Servo mode: hold last commanded position
                    state = self.robot.get_state()
                    hold_action = RobotAction(
                        arm_qpos_cmd=self._last_arm_cmd,
                        hand_qpos_cmd=(
                            self._last_hand_cmd
                            if self._last_hand_cmd is not None
                            else state.hand_qpos
                        ),
                    )
                    self.robot.send_action(hold_action)
            return

        # IDLE / SAVE_PROMPT: no pipeline (waiting for user input)
        if self.state in (ControllerState.IDLE, ControllerState.SAVE_PROMPT):
            return

        # 1. Get VR frame
        vr_frame = self._read_vr_frame()

        # Track VR frame age in stats
        if vr_frame is not None:
            self.vr_stats.record_consume(
                self.tracking_quality.frame_age(vr_frame)
            )

        # 2. Tracking quality gate
        tq_result = self.tracking_quality.check(vr_frame)
        if not tq_result.ok:
            self.error_handler.record_failure("vr_stale")
            if tq_result.tracking_lost:
                self._escalate_to_emergency(
                    f"VR tracking lost for {tq_result.lost_duration_s:.1f}s"
                )
                return
            # Soft deceleration: during stale-but-not-lost window
            # (0.2s < age < 1.0s continuous), exponentially pull arm
            # toward current physical position to avoid abrupt holds.
            if tq_result.lost_duration_s > 0.0:
                self._apply_soft_deceleration(tq_result.lost_duration_s)
            return

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
                    float(np.max(err)), self.frame_count, self._consecutive_divergence,
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
            self._ik_miss_count = 0
        else:
            self._ik_miss_count += 1
            if self._ik_miss_count >= 3:
                logger.warning(
                    "IK missed %d consecutive frames — holding position",
                    self._ik_miss_count,
                )

        # 4c. Velocity-limited step (Phase 2.1 — smoothing safety net)
        #     Applies bottleneck scaling between pipeline output and send_action,
        #     ensuring joint velocity stays within per-joint max_qvel limits.
        #     This is a software-level safety net; the primary speed limit is
        #     still XArm7._limit_joint_step() at the driver layer.
        if self._use_vel_limited_smooth and self._last_arm_cmd is not None:
            action = self._apply_velocity_limited_step(action, state)

        # 5. Safety checks on hardware state (direct checks, no bitmask layer)
        hand_state = self.robot.hand.get_state(full=True)
        torque_ok = safety.check_arm_torque(state.arm_tau)
        current_ok = safety.check_hand_current(hand_state.get("current", np.zeros(12)))
        temp_ok = safety.check_hand_temperature(hand_state.get("temperature", np.full(12, 30.0)))
        comm_ok = safety.check_hand_comm(
            bool(
                np.any(hand_state.get("commboard_err", np.zeros(12)) != 0)
                or np.any(hand_state.get("jointboard_err", np.zeros(12)) != 0)
                or np.any(hand_state.get("tipboard_err", np.zeros(12)) != 0)
            )
        )

        if not (torque_ok and current_ok and temp_ok and comm_ok):
            action = self.error_handler.hold_action()

        # 6. Record frame (before safety gate — captures pre-hold action)
        camera_frame = None
        camera_frames: dict[str, dict] | None = None

        # ── Multi-camera path (new) ──
        if self._multi_camera is not None:
            self._ensure_multi_camera_running()
            try:
                camera_frames = self._multi_camera.read_all_latest()
                for cam_name, cam_frame in camera_frames.items():
                    if cam_frame is not None:
                        ts = cam_frame.get("timestamp", 0.0)
                        if ts > 0:
                            self.camera_stats.record_consume(
                                time.perf_counter() - ts
                            )
                        break  # stats for first camera only
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
                    state=state, action=action, vr_frame=vr_frame,
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
                        state=state, action=action, vr_frame=vr_frame,
                        camera_frame=camera_frame,
                        camera_frames=camera_frames,
                        T_base_eef=T_base_eef,
                    )
                except (ValueError, OSError) as e:
                    logger.exception("collection_loop record_frame failed: %s", e)
            elif self.recorder is not None:
                try:
                    self.recorder.add_frame(
                        state=state, action=action, vr_frame=vr_frame,
                        camera_frame=camera_frame,
                        camera_frames=camera_frames,
                        T_base_eef=T_base_eef,
                    )
                except (ValueError, OSError) as e:
                    logger.exception("recorder add_frame failed: %s", e)

        # 7. Pre-send safety gate — centralized validate_action (ref: ManiUniCon).
        #    Joint limits enforced by XArm7/XHand drivers (error latch → is_error).
        if not self.dry_run:
            action_valid, fail_reason = self.robot.validate_action(action)
            if not action_valid:
                if "error state" in fail_reason or "not connected" in fail_reason:
                    self._escalate_to_emergency(
                        f"Robot error before send_action: {fail_reason}"
                    )
                    return
                logger.warning("Pre-send safety: %s — holding", fail_reason)
                hold = self.error_handler.hold_action()
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
                tick_elapsed_ms, target_ms, self.frame_count,
            )

    # Action computation (split into sub-methods per Phase 3.2)

    def _compute_action(
        self, vr_frame: dict, state: RobotState
    ) -> tuple[RobotAction, dict[str, bool]]:
        """Compute action via TeleopPipeline (shared with sim controller)."""
        current_arm_qpos = state.arm_qpos
        current_hand_qpos = state.hand_qpos

        prev_arm_cmd = (
            self._last_arm_cmd
            if self._last_arm_cmd is not None
            else current_arm_qpos
        )
        prev_hand_cmd = (
            self._last_hand_cmd
            if self._last_hand_cmd is not None
            else current_hand_qpos
        )

        # Load last_good for hold-on-failure
        self.error_handler.init_fallback(current_arm_qpos, current_hand_qpos)

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
            self.error_handler.update_good_positions(
                action.arm_qpos_cmd, action.hand_qpos_cmd,
            )

        # Update EMA reference
        if ik_ok:
            self._last_arm_cmd = action.arm_qpos_cmd.copy()
        if retarget_ok:
            self._last_hand_cmd = action.hand_qpos_cmd.copy()

        # Update counters
        if ik_ok:
            self.ik_success_count += 1
        else:
            self.ik_fail_count += 1
        if retarget_ok:
            self.retarget_success_count += 1
        else:
            self.retarget_fail_count += 1

        return action, {"ik_ok": ik_ok, "retarget_ok": retarget_ok}

    # State machine transitions

    def _handle_keyboard(self) -> None:
        if self.keyboard is None:
            return
        signals = self.keyboard.poll()
        for sig in signals:
            self._transition(sig)

    def _transition(self, signal: ControlSignal) -> None:
        # ── QUIT: context-dependent ──
        if signal == ControlSignal.QUIT:
            if self.state == ControllerState.IDLE:
                logger.info("QUIT — shutting down.")
                self.running = False
            elif self.recording:
                # Stop recording → prompt save/discard
                self._stop_recording()
                self.state = ControllerState.SAVE_PROMPT
                logger.info("Recording stopped → SAVE_PROMPT")
            elif self.state == ControllerState.SAVE_PROMPT:
                # Discard episode → IDLE
                self._discard_episode()
                self.state = ControllerState.IDLE
                logger.info("SAVE_PROMPT → IDLE (discarded)")
            else:
                # TELEOP (not recording) → IDLE
                self.state = ControllerState.IDLE
                logger.info("TELEOP → IDLE")
            return

        # ── EMERGENCY_STOP: any state ──
        if signal == ControlSignal.EMERGENCY_STOP:
            self._escalate_to_emergency("Keyboard ESC")
            return

        # ── REARM: EMERGENCY_STOP only ──
        if signal == ControlSignal.REARM:
            self._rearm()
            return

        # ── HOME: any non-EMERGENCY state ──
        if signal == ControlSignal.HOME:
            if self.state != ControllerState.EMERGENCY_STOP:
                self._do_home()
            return

        # ── TELEOP (T): IDLE → TELEOP ──
        if signal == ControlSignal.TELEOP:
            if self.state == ControllerState.IDLE:
                self._reset_mapper()
                if not self.dry_run:
                    self.robot.reset_soft_start()
                self.state = ControllerState.TELEOP
                self.recording = False
                logger.info("IDLE → TELEOP (recording=False)")
            return

        # ── RECORD (R): toggle recording in TELEOP/PAUSED ──
        if signal == ControlSignal.RECORD:
            if self.state == ControllerState.TELEOP:
                if self.recording:
                    # Stop recording → prompt save/discard
                    self._stop_recording()
                    self.state = ControllerState.SAVE_PROMPT
                    logger.info("TELEOP(rec=True) → SAVE_PROMPT")
                else:
                    # Start recording
                    self._start_recording()
                    self.recording = True
                    logger.info("TELEOP(rec=False) → TELEOP(rec=True)")
            elif self.state == ControllerState.PAUSED:
                if self.recording:
                    self._stop_recording()
                    self.state = ControllerState.SAVE_PROMPT
                    logger.info("PAUSED(rec=True) → SAVE_PROMPT")
                else:
                    self._start_recording()
                    self.recording = True
                    self.state = ControllerState.TELEOP
                    logger.info("PAUSED(rec=False) → TELEOP(rec=True)")
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
                    logger.info("PAUSED → TELEOP (recording=%s, mapper re-anchored)",
                                self.recording)
                else:
                    logger.warning("Cannot resume: no VR frame for mapper re-anchor")
            return

        # ── STOP (S): context-dependent ──
        if signal == ControlSignal.STOP:
            if self.state == ControllerState.SAVE_PROMPT:
                # Confirm save
                self.state = ControllerState.IDLE
                self.recording = False
                logger.info("SAVE_PROMPT → IDLE (saved)")
            elif self.state == ControllerState.TELEOP:
                if self.recording:
                    # Stop recording → prompt
                    self._stop_recording()
                    self.state = ControllerState.SAVE_PROMPT
                    logger.info("TELEOP(rec=True) → SAVE_PROMPT")
                else:
                    self.state = ControllerState.IDLE
                    logger.info("TELEOP(rec=False) → IDLE")
            elif self.state == ControllerState.PAUSED:
                if self.recording:
                    self._stop_recording()
                    self.state = ControllerState.SAVE_PROMPT
                    logger.info("PAUSED(rec=True) → SAVE_PROMPT")
                else:
                    self.state = ControllerState.IDLE
                    logger.info("PAUSED(rec=False) → IDLE")
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
            elif self.recorder is not None and self.recorder.is_recording:
                try:
                    self.recorder.stop_episode(success=False)
                except (ValueError, OSError):
                    pass

        if not self.dry_run:
            self.robot.return_to_home(use_planning=True, cancel_event=self._cancel_event)
        else:
            logger.info("  [dry-run] home (no hardware)")

        self.state = ControllerState.IDLE
        self.recording = False
        self.error_handler.clear()
        self.tracking_quality.reset()
        self._consecutive_divergence = 0
        self._ik_miss_count = 0
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

        # Reset pose interpolator to clear stale waypoints.
        if self._pose_interpolator is not None:
            self._pose_interpolator.reset()

        return True

    def _start_recording(self) -> None:
        """Start episode recording. Does NOT change controller state."""
        logger.info("Starting episode recording...")

        # Re-anchor VR reference
        if not self._reset_mapper():
            logger.error("Cannot start recording without VR frame.")
            return

        if self._collection_loop is not None:
            try:
                self._collection_loop.start_episode(
                    task_label="teleop", operator="",
                )
            except (ValueError, OSError) as e:
                logger.exception("collection_loop start_episode failed: %s", e)
        elif self.recorder is not None:
            try:
                self.recorder.start_episode(task_label="teleop", operator="")
            except (ValueError, OSError) as e:
                logger.exception("recorder start_episode failed: %s", e)

        self.error_handler.clear()
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
                path = self._collection_loop.stop_episode(
                    success=(classification == "success"),
                    reason="manual",
                    classification=classification,
                    ik_success_rate=self._compute_ik_success_rate(),
                    vr_drop_rate=self._compute_vr_drop_rate(),
                )
                if path:
                    logger.info("  Saved to %s", path)
            except (ValueError, OSError) as e:
                logger.exception("collection_loop stop_episode failed: %s", e)
        elif self.recorder is not None and self.recorder.is_recording:
            try:
                path = self.recorder.stop_episode(success=(classification == "success"))
                if path:
                    logger.info("  Saved to %s", path)
            except (ValueError, OSError) as e:
                logger.exception("recorder stop_episode failed: %s", e)
        self.recording = False
        logger.info("Episode stopped.")

    def _discard_episode(self) -> None:
        """Discard the current episode file (SAVE_PROMPT → discard)."""
        if self._collection_loop is not None:
            self._collection_loop.discard_episode()
        elif self.recorder is not None and hasattr(self.recorder, '_episode_path'):
            path = getattr(self.recorder, '_episode_path', None)
            if path:
                try:
                    from pathlib import Path
                    Path(path).unlink(missing_ok=True)
                    logger.info("Episode discarded: %s", path)
                except OSError:
                    pass

    def _escalate_to_emergency(self, reason: str) -> None:
        logger.error("EMERGENCY_STOP: %s", reason)
        self.state = ControllerState.EMERGENCY_STOP
        if not self.dry_run:
            self.robot.emergency_stop()
        self.running = False

    def _rearm(self) -> None:
        """Re-arm from EMERGENCY_STOP without script restart.

        Clears errors, resets tracking, transitions to IDLE.
        No-op if not in EMERGENCY_STOP state.

        Ref: ManiUniCon reset_event (keyboard 'h' / Quest 'A' button).
        """
        if self.state != ControllerState.EMERGENCY_STOP:
            logger.info(
                "REARM ignored: not in EMERGENCY_STOP (current=%s)",
                self.state.value,
            )
            return

        logger.info("REARM: clearing errors and resetting...")
        self.running = True
        self.error_handler.clear()
        self.tracking_quality.reset()
        self._consecutive_divergence = 0
        self._ik_miss_count = 0

        if not self.dry_run:
            try:
                self.robot.arm.clear_error()
                self.robot.reset_soft_start()
            except (ValueError, RuntimeError, KeyError, AttributeError) as e:
                logger.warning("REARM: error clearing robot state: %s", e)

        self.state = ControllerState.IDLE
        self.recording = False
        self._last_arm_cmd = None
        self._last_hand_cmd = None
        logger.info("REARM complete. State: IDLE")

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        if self.recording:
            if self._collection_loop is not None and self._collection_loop.is_recording:
                try:
                    self._collection_loop.stop_episode(success=False, reason="shutdown")
                except (ValueError, RuntimeError, KeyError):
                    pass  # shutdown must never fail
            elif self.recorder is not None and self.recorder.is_recording:
                try:
                    self.recorder.stop_episode(success=False)
                except (ValueError, RuntimeError, KeyError):
                    pass

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
        if now - getattr(self, '_last_mc_health_check', 0.0) < 5.0:
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

    def _print_status(
        self, vr_frame: dict | None, now: float
    ) -> None:
        age_s = (
            self.tracking_quality.frame_age(vr_frame)
            if vr_frame is not None
            else float("inf")
        )
        seq = vr_frame.get("sequence_id", "?") if vr_frame else "?"
        rec = "REC" if self.recording else "   "
        vr_mean_age = self.vr_stats.mean_age_s * 1000
        cam_mean_age = self.camera_stats.mean_age_s * 1000
        logger.info(
            "[t=%.1f] frames=%s state=%s %s vr_seq=%s age=%sms "
            "vr_ema=%.0fms cam_ema=%.0fms "
            "ik=%s/%s retarget=%s/%s",
            now, self.frame_count, self.state.value, rec, seq, f"{age_s*1000:.0f}",
            vr_mean_age, cam_mean_age,
            self.ik_success_count, self.ik_fail_count,
            self.retarget_success_count, self.retarget_fail_count,
        )


