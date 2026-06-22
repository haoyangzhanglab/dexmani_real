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
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.control import safety
from dexmani_real.teleop.core.tracking import (
    TrackingQuality,
    TrackingQualityConfig,
)
from dexmani_real.planning.pose_utils import quat_wxyz_to_rotmat
from dexmani_real.planning.types import IKResult, Pose
from dexmani_real.recording.quality_flags import (
    ARM_TORQUE_OK,
    HAND_COMM_OK,
    HAND_CURRENT_OK,
    HAND_TEMP_OK,
    IK_SUCCESS,
    IN_WORKSPACE,
    JOINT_JUMP_OK,
    RETARGET_OK,
    RETARGET_VALID,
    TRACKING_OK,
    QualityFlags,
)
from dexmani_real.robot.interface import (
    RobotAction,
    RobotInterface,
    RobotInterfaceConfig,
    RobotState,
)
from dexmani_real.utils.hand_utils import (
    OPERATOR2MANO_RIGHT,
    estimate_frame_from_hand_points,
)
from dexmani_real.utils.rate_limiter import RateLimiter
from dexmani_real.utils.signal_utils import ema_smooth

if TYPE_CHECKING:
    from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
    from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
    from dexmani_real.teleop.vr.pose_interpolator import CartPoseInterpolator
    from dexmani_real.planning.planner import XArm7MotionPlanner
    from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker
    from dexmani_real.recording.episode_recorder import EpisodeRecorder

logger = get_logger(__name__)


class ControllerState(Enum):
    IDLE = "IDLE"
    TELEOP = "TELEOP"
    RECORDING = "RECORDING"
    EMERGENCY_STOP = "EMERGENCY_STOP"


# Per-step joint jump limits (rad).
# NOTE: These are IK-anomaly defenses, NOT routine speed limits.
# Routine speed limiting is handled by XArm7._limit_joint_step() at the driver
# layer (bottleneck scaling to max_qvel, ~1.8-3.0°/frame @ 50 Hz).
# These thresholds (5°/frame = 250°/s) are deliberately higher than max_qvel
# (~90-150°/s) — they only trigger when IK produces an abnormally large jump,
# e.g. a discontinuous solution from a poor seed or singularity crossing.
_ARM_JUMP_LIMIT_RAD = np.deg2rad(5.0)
_HAND_JUMP_LIMIT_RAD = np.deg2rad(10.0)


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
        *,
        tracker: QuestHandTracker | None = None,
        keyboard_queue: object | None = None,
        target_hz: float = 50.0,
        ema_alpha_arm: float = 1.0,  # 1.0 = no smoothing (disabled)
        dry_run: bool = False,
        recorder: EpisodeRecorder | None = None,
        use_cartesian_interpolation: bool = False,
        interpolation_max_pos_speed: float = 0.25,
        interpolation_max_rot_speed: float = 0.5,
        use_zmq_vr: bool = False,
        zmq_vr_port: int = 5555,
        camera_process: object | None = None,
    ) -> None:
        self.robot = robot
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self.tracker = tracker
        self.dry_run = dry_run
        self.recorder = recorder

        self.limiter = RateLimiter(target_hz)
        self.ema_alpha_arm = float(ema_alpha_arm)

        # Cartesian pose interpolator (optional, disabled by default).
        # Ref: ManiUniCon PoseTrajectoryInterpolator.
        self._pose_interpolator: CartPoseInterpolator | None = None
        if use_cartesian_interpolation:
            from dexmani_real.teleop.vr.pose_interpolator import CartPoseInterpolator
            self._pose_interpolator = CartPoseInterpolator(
                max_pos_speed=interpolation_max_pos_speed,
                max_rot_speed=interpolation_max_rot_speed,
            )

        # Camera daemon process (optional — crash-isolated frame capture).
        # Ref: ManiUniCon Camera Process.
        self._camera_process = camera_process

        # ZMQ VR subscriber (optional, disabled by default).
        # Ref: Open-Teach multi-process ZMQ PUB/SUB pattern.
        self._vr_subscriber: VRFrameSubscriber | None = None
        if use_zmq_vr:
            from dexmani_real.teleop.vr.vr_publisher import VRFrameSubscriber
            self._vr_subscriber = VRFrameSubscriber(sub_port=zmq_vr_port)
            self._vr_subscriber.connect()

        self.tracking_quality = TrackingQuality(TrackingQualityConfig(max_frame_age_s=0.2))
        self.error_handler = TeleopErrorHandler()

        # State
        self.state = ControllerState.IDLE
        self.running = False
        self._last_arm_cmd: np.ndarray | None = None
        self._last_hand_cmd: np.ndarray | None = None

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
                self.limiter.wait()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping.")
        except (RuntimeError, ConnectionError, ValueError) as e:
            logger.exception("Unhandled exception in main loop: %s", e)
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # State machine tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        self.frame_count += 1

        # 1. Get VR frame
        vr_frame = self._read_vr_frame()

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
            # Ref: BVPro clip_arm_velocity() soft-start pattern.
            if tq_result.lost_duration_s > 0.0 and self._last_arm_cmd is not None:
                if not self.dry_run:
                    state = self.robot.get_state()
                else:
                    state = self._dummy_state()
                decay = float(np.exp(-tq_result.lost_duration_s * 3.0))
                arm_interp = (
                    decay * self._last_arm_cmd
                    + (1.0 - decay) * state.arm_qpos
                )
                action = RobotAction(
                    arm_qpos_cmd=arm_interp,
                    hand_qpos_cmd=(
                        self._last_hand_cmd
                        if self._last_hand_cmd is not None
                        else state.hand_qpos
                    ),
                )
                if not self.dry_run and self.state != ControllerState.IDLE:
                    self.robot.send_action(action)
            return

        # 3. Read robot state
        if self.dry_run:
            state = self._dummy_state()
        else:
            state = self.robot.get_state()

        # 4. Compute action
        action, quality = self._compute_action(vr_frame, state)

        # 5. Safety checks on state (arm torque, hand current, hand temp, hand comm)
        quality.set(ARM_TORQUE_OK, safety.check_arm_torque(state))
        quality.set(HAND_CURRENT_OK, safety.check_hand_current(state))
        quality.set(HAND_TEMP_OK, safety.check_hand_temperature(state))
        quality.set(HAND_COMM_OK, safety.check_hand_comm(state))

        # Hard safety: joint limits (trigger E-Stop, not just quality flags)
        if not self.dry_run:
            if not safety.check_arm_joint_limits(
                state,
                self.robot.arm.config.qpos_min,
                self.robot.arm.config.qpos_max,
            ):
                self._escalate_to_emergency(
                    f"Arm joint out of limits: {state.arm_qpos}"
                )
                return
            if not safety.check_hand_joint_limits(
                state,
                self.robot.hand.config.qpos_min,
                self.robot.hand.config.qpos_max,
            ):
                # NOTE: Hand joint limit violations log a warning (not E-Stop),
                # unlike arm violations which trigger an immediate E-Stop.
                # Rationale: XHand has its own internal commboard-level error
                # protection that will fault on out-of-range commands before
                # mechanical damage occurs. If this assumption proves false
                # in testing, elevate to E-Stop.
                logger.warning(
                    "Hand joint out of limits: %s", state.hand_qpos
                )

        flags = quality.get()

        # 6. Execute action based on state

        # Poll camera frame from daemon process (crash-isolated).
        # A crashed camera process does NOT block the control loop.
        camera_frame = None
        if self._camera_process is not None:
            try:
                camera_frame = self._camera_process.poll_latest_frame()
                if getattr(self._camera_process, "crashed", False):
                    logger.warning("Camera daemon crashed — continuing without camera data.")
            except (ValueError, RuntimeError, KeyError):
                logger.debug("Camera poll failed — continuing without camera frame.")

        if self.state == ControllerState.RECORDING and self.recorder is not None:
            try:
                self.recorder.add_frame(
                    state=state, action=action, vr_frame=vr_frame,
                    quality_flags=flags,
                    camera_frame=camera_frame,
                    T_base_eef=self._compute_T_base_eef(state),
                )
            except (ValueError, OSError) as e:
                logger.exception("recorder add_frame failed: %s", e)

        if not self.dry_run:
            if self.robot.is_error():
                self._escalate_to_emergency("Robot error state detected before send_action")
                return
            # Pre-send safety: torque/current hold (was validate_action).
            # workspace position/orientation is enforced at L2
            # (_compute_action IN_WORKSPACE flag → hold).
            # arm_joint_limits is enforced at L4 (E-Stop, stricter than hold).
            # hand_joint_limits is warn-only per XHand internal protection design.
            if not (flags & ARM_TORQUE_OK) or not (flags & HAND_CURRENT_OK):
                logger.warning(
                    "Pre-send safety: torque=%s current=%s — holding",
                    not (flags & ARM_TORQUE_OK),
                    not (flags & HAND_CURRENT_OK),
                )
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

        # Note: cumulative E-Stop escalation removed per error_handler design.
        # Persistent failures are caught by robot.is_error() at driver level.

        # 7. Periodic status
        now = time.monotonic()
        if now - self.last_status_ts >= self.status_interval:
            self.last_status_ts = now
            self._print_status(vr_frame, flags, now)

    # ------------------------------------------------------------------
    # Action computation (split into sub-methods per Phase 3.2)
    # ------------------------------------------------------------------

    def _compute_action(
        self, vr_frame: dict, state: RobotState
    ) -> tuple[RobotAction, QualityFlags]:
        quality = QualityFlags()
        quality.set(TRACKING_OK, True)

        current_arm_qpos = state.arm_qpos.copy()
        current_hand_qpos = state.hand_qpos.copy()

        prev_arm_cmd = (
            self._last_arm_cmd.copy()
            if self._last_arm_cmd is not None
            else current_arm_qpos
        )
        prev_hand_cmd = (
            self._last_hand_cmd.copy()
            if self._last_hand_cmd is not None
            else current_hand_qpos
        )

        # Load last_good for hold-on-failure
        self.error_handler.init_fallback(current_arm_qpos, current_hand_qpos)

        # ── Arm IK ──
        arm_cmd, ik_ok, target_eef_pos = self._compute_arm_command(
            vr_frame, state, prev_arm_cmd, quality
        )

        # Workspace check on computed arm command (position + orientation)
        arm_eef_pose = self.planner.compute_eef_pose_world(arm_cmd)
        in_workspace = self.robot.check_workspace(arm_eef_pose.p)
        ori_ok = self.robot.check_workspace_orientation(arm_eef_pose.q)
        quality.set(IN_WORKSPACE, in_workspace and ori_ok)
        if not in_workspace or not ori_ok:
            hold = self.error_handler.hold_action()
            arm_cmd = hold.arm_qpos_cmd
            hand_cmd = hold.hand_qpos_cmd
        else:
            # ── Hand retarget ──
            hand_cmd, retarget_ok = self._compute_hand_command(
                vr_frame, prev_hand_cmd, quality
            )

        # ── Joint jump clamp ──
        arm_cmd, hand_cmd, jump_ok = self._apply_jump_clamp(
            arm_cmd, hand_cmd, prev_arm_cmd, prev_hand_cmd, quality
        )

        # Update last-good positions for hold-on-failure
        if ik_ok and retarget_ok:
            self.error_handler.update_good_positions(arm_cmd, hand_cmd)

        action = RobotAction(
            arm_qpos_cmd=arm_cmd,
            hand_qpos_cmd=hand_cmd,
            target_eef_pos=target_eef_pos,
        )
        return action, quality

    def _compute_arm_command(
        self,
        vr_frame: dict,
        state: RobotState,
        prev_arm_cmd: np.ndarray,
        quality: QualityFlags,
    ) -> tuple[np.ndarray, bool, np.ndarray | None]:
        """Compute arm IK command from VR wrist pose.

        Returns:
            (arm_cmd, ik_ok, target_eef_pos).
        """
        wrist_pos = vr_frame["wrist_pos"]
        wrist_quat_wxyz = vr_frame["wrist_quat_wxyz"]

        arm_cmd = prev_arm_cmd.copy()
        ik_ok = False
        target_eef_pos = None

        if self.arm_mapper.is_ready():
            mapped = self.arm_mapper.map(wrist_pos, wrist_quat_wxyz)
            if mapped is not None:
                target_eef_pos = mapped["pos"]
                target_eef_quat = mapped["quat_wxyz"]

                # Cartesian pose interpolation (optional).
                # Ref: ManiUniCon PoseTrajectoryInterpolator — linear pos +
                # SLERP rot between VR frames, eliminating stale re-use.
                if self._pose_interpolator is not None:
                    self._pose_interpolator.push_target_pose(
                        target_eef_pos, target_eef_quat,
                    )
                    result = self._pose_interpolator.get_interpolated_pose()
                    if result is not None:
                        target_eef_pos, target_eef_quat = result

                target_pose = Pose(p=target_eef_pos, q=target_eef_quat)
                ik_result: IKResult = self.planner.solve_teleop_ik(
                    target_pose, state.arm_qpos.copy(), prev_arm_cmd
                )
                if ik_result.success and ik_result.qpos is not None:
                    ik_ok = True
                    raw_arm = np.asarray(ik_result.qpos, dtype=np.float64)
                    arm_cmd = ema_smooth(raw_arm, self._last_arm_cmd, self.ema_alpha_arm)
                    self._last_arm_cmd = arm_cmd
                    self.ik_success_count += 1
                else:
                    self.ik_fail_count += 1
                    self.error_handler.record_failure(
                        "ik", ik_result.reason
                    )
                    arm_cmd = prev_arm_cmd.copy()
            else:
                self.error_handler.record_failure("wrist_map", "mapper returned None")
        # else: arm_mapper not ready yet (not reset), hold in place

        quality.set(IK_SUCCESS, ik_ok)
        return arm_cmd, ik_ok, target_eef_pos

    def _compute_hand_command(
        self,
        vr_frame: dict,
        prev_hand_cmd: np.ndarray,
        quality: QualityFlags,
    ) -> tuple[np.ndarray, bool]:
        """Compute hand retargeting command from VR landmarks.

        Returns:
            (hand_cmd, retarget_ok).
        """
        landmarks = vr_frame["landmarks"]
        hand_cmd = prev_hand_cmd.copy()
        retarget_ok = False

        try:
            wrist_rot = estimate_frame_from_hand_points(landmarks)
            mano_landmarks = landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT
            target_hand = self.retargeter.retarget(mano_landmarks)
            if target_hand is not None and len(target_hand) == 12:
                retarget_ok = True
                raw_hand = np.asarray(target_hand, dtype=np.float64)
                hand_cmd = raw_hand.copy()  # no EMA — dex_retargeting has built-in low_pass_alpha
                self._last_hand_cmd = hand_cmd.copy()
                self.retarget_success_count += 1
            else:
                self.retarget_fail_count += 1
                self.error_handler.record_failure("retarget", "retarget returned None")
                hand_cmd = prev_hand_cmd.copy()
        except (ValueError, TypeError) as e:
            self.retarget_fail_count += 1
            self.error_handler.record_failure("retarget", f"retarget threw exception: {e}")
            hand_cmd = prev_hand_cmd.copy()

        quality.set(RETARGET_OK, retarget_ok)
        quality.set(RETARGET_VALID, safety.check_retarget_valid(hand_cmd))
        return hand_cmd, retarget_ok

    def _apply_jump_clamp(
        self,
        arm_cmd: np.ndarray,
        hand_cmd: np.ndarray,
        prev_arm_cmd: np.ndarray,
        prev_hand_cmd: np.ndarray,
        quality: QualityFlags,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Clamp per-step joint deltas to prevent command jumps.

        Returns:
            (arm_cmd, hand_cmd, jump_ok).
        """
        jump_ok = True
        if prev_arm_cmd is not None:
            arm_delta = np.max(np.abs(arm_cmd - prev_arm_cmd))
            if arm_delta > _ARM_JUMP_LIMIT_RAD:
                jump_ok = False
                arm_cmd = prev_arm_cmd + np.clip(
                    arm_cmd - prev_arm_cmd, -_ARM_JUMP_LIMIT_RAD, _ARM_JUMP_LIMIT_RAD
                )
        if prev_hand_cmd is not None:
            hand_delta = np.max(np.abs(hand_cmd - prev_hand_cmd))
            if hand_delta > _HAND_JUMP_LIMIT_RAD:
                jump_ok = False
                hand_cmd = prev_hand_cmd + np.clip(
                    hand_cmd - prev_hand_cmd, -_HAND_JUMP_LIMIT_RAD, _HAND_JUMP_LIMIT_RAD
                )
        quality.set(JOINT_JUMP_OK, jump_ok)

        if not jump_ok:
            hold = self.error_handler.record_failure("joint_jump")
            arm_cmd = hold.arm_qpos_cmd
            hand_cmd = hold.hand_qpos_cmd

        return arm_cmd, hand_cmd, jump_ok

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _handle_keyboard(self) -> None:
        if self.keyboard is None:
            return
        signals = self.keyboard.poll()
        for sig in signals:
            self._transition(sig)

    def _transition(self, signal: ControlSignal) -> None:
        if signal == ControlSignal.QUIT:
            logger.info("QUIT — shutting down.")
            self.running = False
            return

        if signal == ControlSignal.EMERGENCY_STOP:
            self._escalate_to_emergency("Keyboard ESC")
            return

        if signal == ControlSignal.REARM:
            self._rearm()
            return

        if signal == ControlSignal.HOME:
            self._do_home()
            return

        if signal == ControlSignal.TELEOP:
            if self.state == ControllerState.IDLE:
                # Re-anchor VR reference so arm starts tracking immediately
                self._reset_mapper()
                # Reset soft-start ramp — ensures speed limit protection
                # even if robot was idle for minutes after connect()
                if not self.dry_run:
                    self.robot.reset_soft_start()
                self.state = ControllerState.TELEOP
                logger.info("IDLE → TELEOP")
            # else: already in TELEOP or RECORDING, no-op

        elif signal == ControlSignal.RECORD:
            if self.state == ControllerState.TELEOP:
                self._start_recording()
            elif self.state == ControllerState.RECORDING:
                logger.info("Already RECORDING, press S to stop.")

        elif signal == ControlSignal.STOP:
            if self.state == ControllerState.RECORDING:
                self._stop_recording()
                self.state = ControllerState.TELEOP
                logger.info("RECORDING → TELEOP")
            elif self.state == ControllerState.TELEOP:
                self.state = ControllerState.IDLE
                logger.info("TELEOP → IDLE")

    def _do_home(self) -> None:
        logger.info("Returning to home...")
        self._last_arm_cmd = None
        self._last_hand_cmd = None

        if not self.dry_run:
            self.robot.return_to_home(use_planning=True, cancel_event=self._cancel_event)
        else:
            logger.info("  [dry-run] home (no hardware)")

        self.state = ControllerState.IDLE
        self.error_handler.clear()
        self.tracking_quality.reset()
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
        logger.info("Starting episode recording...")

        # Re-anchor VR reference
        if not self._reset_mapper():
            logger.error("Cannot start recording without VR frame.")
            return

        if self.recorder is not None:
            try:
                self.recorder.start_episode(task_label="teleop", operator="")
            except (ValueError, OSError) as e:
                logger.exception("recorder start_episode failed: %s", e)

        self.state = ControllerState.RECORDING
        self.error_handler.clear()
        logger.info("TELEOP → RECORDING")

    def _stop_recording(self) -> None:
        logger.info("Stopping episode. frames=%s", self.frame_count)
        if self.recorder is not None and self.recorder.is_recording:
            try:
                path = self.recorder.stop_episode(success=True)
                if path:
                    logger.info("  Saved to %s", path)
            except (ValueError, OSError) as e:
                logger.exception("recorder stop_episode failed: %s", e)
        logger.info("Episode stopped.")

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

        if not self.dry_run:
            try:
                self.robot.arm.clear_error()
                self.robot.reset_soft_start()
            except (ValueError, RuntimeError, KeyError, AttributeError) as e:
                logger.warning("REARM: error clearing robot state: %s", e)

        self.state = ControllerState.IDLE
        self._last_arm_cmd = None
        self._last_hand_cmd = None
        logger.info("REARM complete. State: IDLE")

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        if self.recorder is not None and self.recorder.is_recording:
            try:
                self.recorder.stop_episode(success=False)
            except (ValueError, RuntimeError, KeyError):
                pass  # shutdown must never fail
        if self.keyboard is not None:
            self.keyboard.stop()
        logger.info("  Frames: %s", self.frame_count)
        logger.info("  IK: ok=%s fail=%s", self.ik_success_count, self.ik_fail_count)
        logger.info("  Retarget: ok=%s fail=%s", self.retarget_success_count, self.retarget_fail_count)

    # ------------------------------------------------------------------
    # VR data source
    # ------------------------------------------------------------------

    def _read_vr_frame(self) -> dict | None:
        if self._vr_subscriber is not None:
            return self._vr_subscriber.recv_latest()
        if self.tracker is not None:
            return self.tracker.get_latest()
        return None

    # ------------------------------------------------------------------
    # EEF pose utilities
    # ------------------------------------------------------------------

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
            hand_current=np.zeros(12, dtype=np.float64),
            hand_tactile_sum=np.zeros((5, 3), dtype=np.float64),
            hand_tactile_raw=np.zeros((5, 120, 3), dtype=np.float64),
            hand_temperature=np.full(12, 30.0, dtype=np.float64),
            fingertip_pos=np.zeros((5, 3), dtype=np.float64),
            arm_connected=True,
            hand_connected=True,
            hand_error=False,
            timestamp=time.perf_counter(),
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _print_status(
        self, vr_frame: dict | None, quality_flags: int, now: float
    ) -> None:
        failed = QualityFlags.describe(quality_flags)
        flags_str = ",".join(failed) if failed else "ALL_OK"
        age_s = (
            self.tracking_quality.frame_age(vr_frame)
            if vr_frame is not None
            else float("inf")
        )
        seq = vr_frame.get("sequence_id", "?") if vr_frame else "?"
        logger.info(
            "[t=%.1f] frames=%s state=%s vr_seq=%s age=%sms "
            "ik=%s/%s retarget=%s/%s [%s]",
            now, self.frame_count, self.state.value, seq, f"{age_s*1000:.0f}",
            self.ik_success_count, self.ik_fail_count,
            self.retarget_success_count, self.retarget_fail_count,
            flags_str,
        )


