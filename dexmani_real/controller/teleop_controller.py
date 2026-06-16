"""Teleoperation controller: state machine, _tick(), EMA smoothing, quality gating.

State machine:
    IDLE --T--> TELEOP --R--> RECORDING --S--> IDLE
      |        |   S->IDLE      |   H->IDLE
      H        H                |
      v        v                v
  return_to_home          EMERGENCY_STOP (ESC / timeout)

Data sources:
    --no-ipc (default): tracker.get_latest() directly
    --ipc:               ipc_buffer.read() → pickle.loads()
    dry-run:             dummy state, no hardware
"""

from __future__ import annotations

import threading
import time
import traceback
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.controller.error_handler import TeleopErrorHandler
from dexmani_real.controller.keyboard_handler import ControlSignal, KeyboardHandler
from dexmani_real.controller.tracking_quality import (
    TrackingQuality,
    TrackingQualityConfig,
)
from dexmani_real.planner.planner_types import IKResult, Pose
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
from dexmani_real.robot.robot_interface import (
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


class ControllerState(Enum):
    IDLE = "IDLE"
    TELEOP = "TELEOP"
    RECORDING = "RECORDING"
    EMERGENCY_STOP = "EMERGENCY_STOP"


_ARM_JUMP_LIMIT_RAD = np.deg2rad(5.0)
_HAND_JUMP_LIMIT_RAD = np.deg2rad(10.0)

# Safety thresholds
_ARM_TORQUE_LIMIT_NM = 50.0       # N·m
_HAND_CURRENT_LIMIT_MA = 500.0    # mA
_HAND_TEMP_LIMIT_C = 70.0         # °C


class TeleopController:
    """Main teleoperation controller.

    Owns the control loop: reads VR, runs IK+retarget, applies EMA smoothing,
    enforces safety clamps, manages recording lifecycle.

    The controller operates on RobotInterface (not XArm7/XHand directly).
    """

    def __init__(
        self,
        robot: RobotInterface,
        arm_mapper: Any,             # ArmWristMapper
        retargeter: Any,             # XHandRetargeter
        planner: Any,                # XArm7MotionPlanner
        *,
        tracker: Any | None = None,  # QuestHandTracker (None = dry-run or IPC)
        ipc_buffer: Any | None = None,  # SharedRingBuffer
        keyboard_queue: Any | None = None,
        target_hz: float = 50.0,
        ema_alpha_arm: float = 0.3,
        ema_alpha_hand: float = 0.3,
        dry_run: bool = False,
        recorder: Any | None = None,
    ) -> None:
        self.robot = robot
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self.tracker = tracker
        self.ipc_buffer = ipc_buffer
        self.dry_run = dry_run
        self.recorder = recorder

        self.limiter = RateLimiter(target_hz)
        self.ema_alpha_arm = float(ema_alpha_arm)
        self.ema_alpha_hand = float(ema_alpha_hand)

        self.tracking_quality = TrackingQuality(TrackingQualityConfig(max_frame_age_s=0.2))
        self.error_handler = TeleopErrorHandler()

        # State
        self.state = ControllerState.IDLE
        self.running = False
        self._ema_arm_qpos: np.ndarray | None = None
        self._ema_hand_qpos: np.ndarray | None = None

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
            print("[TeleopController] Robot not connected. Attempting connect...")
            result = self.robot.connect()
            print(f"  connect result: {result}")

        print(f"[TeleopController] Entering main loop at {self.limiter.target_hz:.0f} Hz")
        print(f"  Mode: {'dry-run' if self.dry_run else 'hardware'}")
        print(f"  VR: {'IPC' if self.ipc_buffer is not None else 'direct'}")
        print(f"  EMA: arm_alpha={self.ema_alpha_arm}, hand_alpha={self.ema_alpha_hand}")
        print(f"  Controls: T=teleop R=record S=stop H=home ESC=emergency Q=quit")

        self.last_status_ts = time.monotonic()

        try:
            while self.running:
                self._handle_keyboard()
                self._tick()
                self.limiter.wait()
        except KeyboardInterrupt:
            print("\n[TeleopController] KeyboardInterrupt — stopping.")
        except Exception:
            traceback.print_exc()
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
            if tq_result.tracking_lost or self.error_handler.should_emergency_stop:
                self._escalate_to_emergency(f"VR tracking lost for {tq_result.lost_duration_s:.1f}s")
            return

        # 3. Read robot state
        if self.dry_run:
            state = self._dummy_state()
        else:
            state = self.robot.get_state()

        # 4. Compute action
        action, quality = self._compute_action(vr_frame, state)

        # 5. Safety checks on state (arm torque, hand current, hand temp, hand comm)
        quality.set(ARM_TORQUE_OK, self._check_arm_torque(state))
        quality.set(HAND_CURRENT_OK, self._check_hand_current(state))
        quality.set(HAND_TEMP_OK, self._check_hand_temperature(state))
        quality.set(HAND_COMM_OK, not state.hand_error)

        flags = quality.get()

        # 6. Execute action based on state
        if self.state == ControllerState.RECORDING and self.recorder is not None:
            try:
                self.recorder.add_frame(
                    state=state, action=action, vr_frame=vr_frame,
                    quality_flags=flags,
                    T_base_eef=self._compute_T_base_eef(state),
                )
            except Exception:
                traceback.print_exc()

        if not self.dry_run:
            if self.robot.is_error():
                self._escalate_to_emergency("Robot error state detected before send_action")
                return
            result = self.robot.send_action(action)
            arm_ok = result.get("arm_ok", False)
            hand_ok = result.get("hand_ok", False)
            if not arm_ok or not hand_ok:
                print(f"[WARN] send_action: arm_ok={arm_ok} hand_ok={hand_ok}")

        if self.error_handler.should_emergency_stop:
            self._escalate_to_emergency(self.error_handler.summary())

        # 7. Periodic status
        now = time.monotonic()
        if now - self.last_status_ts >= self.status_interval:
            self.last_status_ts = now
            self._print_status(vr_frame, flags, now)

    def _compute_action(
        self, vr_frame: dict[str, Any], state: RobotState
    ) -> tuple[RobotAction, QualityFlags]:
        quality = QualityFlags()
        quality.set(TRACKING_OK, True)

        wrist_pos = vr_frame["wrist_pos"]
        wrist_quat_wxyz = vr_frame["wrist_quat_wxyz"]
        landmarks = vr_frame["landmarks"]

        current_arm_qpos = state.arm_qpos.copy()
        current_hand_qpos = state.hand_qpos.copy()

        prev_arm_cmd = (
            self._ema_arm_qpos.copy()
            if self._ema_arm_qpos is not None
            else current_arm_qpos
        )
        prev_hand_cmd = (
            self._ema_hand_qpos.copy()
            if self._ema_hand_qpos is not None
            else current_hand_qpos
        )

        # Load last_good for hold-on-failure
        self.error_handler.init_fallback(current_arm_qpos, current_hand_qpos)

        # ── Arm IK ──
        arm_cmd = prev_arm_cmd.copy()
        ik_ok = False
        target_eef_pos = None
        target_eef_rot6d = None

        if self.arm_mapper.is_ready():
            mapped = self.arm_mapper.map(wrist_pos, wrist_quat_wxyz)
            if mapped is not None:
                target_eef_pos = mapped["pos"]
                target_eef_quat = mapped["quat_wxyz"]
                target_pose = Pose(p=target_eef_pos, q=target_eef_quat)
                ik_result: IKResult = self.planner.solve_teleop_ik(
                    target_pose, current_arm_qpos, prev_arm_cmd
                )
                if ik_result.success and ik_result.qpos is not None:
                    ik_ok = True
                    raw_arm = np.asarray(ik_result.qpos, dtype=np.float64)
                    arm_cmd = self._ema_smooth(raw_arm, self._ema_arm_qpos, self.ema_alpha_arm)
                    self._ema_arm_qpos = arm_cmd
                    self.ik_success_count += 1
                else:
                    self.ik_fail_count += 1
                    self.error_handler.record_failure(
                        "ik", ik_result.reason
                    )
                    arm_cmd = prev_arm_cmd.copy()
            else:
                self.error_handler.record_failure("wrist_map", "mapper returned None")
        else:
            pass  # arm_mapper not ready yet (not reset), hold in place

        quality.set(IK_SUCCESS, ik_ok)
        # Check workspace by computing EEF FK for the arm command
        arm_eef_pos = self.planner.compute_eef_pose_world(arm_cmd).p
        in_workspace = self.robot.check_workspace(arm_eef_pos)
        quality.set(IN_WORKSPACE, in_workspace)
        if not in_workspace:
            arm_cmd = self.error_handler.hold_action().arm_qpos_cmd

        # ── Hand retarget ──
        hand_cmd = prev_hand_cmd.copy()
        retarget_ok = False

        try:
            wrist_rot = estimate_frame_from_hand_points(landmarks)
            mano_landmarks = landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT
            target_hand = self.retargeter.retarget(mano_landmarks)
            if target_hand is not None and len(target_hand) == 12:
                retarget_ok = True
                raw_hand = np.asarray(target_hand, dtype=np.float64)
                hand_cmd = self._ema_smooth(raw_hand, self._ema_hand_qpos, self.ema_alpha_hand)
                self._ema_hand_qpos = hand_cmd
                self.retarget_success_count += 1
            else:
                self.retarget_fail_count += 1
                self.error_handler.record_failure("retarget", "retarget returned None")
                hand_cmd = prev_hand_cmd.copy()
        except Exception:
            self.retarget_fail_count += 1
            self.error_handler.record_failure("retarget", "retarget threw exception")
            hand_cmd = prev_hand_cmd.copy()

        quality.set(RETARGET_OK, retarget_ok)
        quality.set(RETARGET_VALID, self._check_retarget_valid(hand_cmd))

        # ── Joint jump clamp ──
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

        # Record success for hold-on-failure
        if ik_ok and retarget_ok and jump_ok:
            self.error_handler.record_success(arm_cmd, hand_cmd)

        action = RobotAction(
            arm_qpos_cmd=arm_cmd,
            hand_qpos_cmd=hand_cmd,
            target_eef_pos=target_eef_pos,
        )
        return action, quality

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
            print("[Controller] QUIT — shutting down.")
            self.running = False
            return

        if signal == ControlSignal.EMERGENCY_STOP:
            self._escalate_to_emergency("Keyboard ESC")
            return

        if signal == ControlSignal.HOME:
            self._do_home()
            return

        if signal == ControlSignal.TELEOP:
            if self.state == ControllerState.IDLE:
                self.state = ControllerState.TELEOP
                print("[Controller] IDLE → TELEOP")
            # else: already in TELEOP or RECORDING, no-op

        elif signal == ControlSignal.RECORD:
            if self.state == ControllerState.TELEOP:
                self._start_recording()
            elif self.state == ControllerState.RECORDING:
                print("[Controller] Already RECORDING, press S to stop.")

        elif signal == ControlSignal.STOP:
            if self.state == ControllerState.RECORDING:
                self._stop_recording()
                self.state = ControllerState.IDLE
                print("[Controller] RECORDING → IDLE")
            elif self.state == ControllerState.TELEOP:
                self.state = ControllerState.IDLE
                print("[Controller] TELEOP → IDLE")

    def _do_home(self) -> None:
        print("[Controller] Returning to home...")
        self.state = ControllerState.IDLE
        self._ema_arm_qpos = None
        self._ema_hand_qpos = None

        if not self.dry_run:
            self.robot.return_to_home(use_planning=True, cancel_event=self._cancel_event)
        else:
            print("  [dry-run] home (no hardware)")

        self.error_handler.clear()
        self.tracking_quality.reset()
        print("[Controller] Home complete.")

    def _start_recording(self) -> None:
        print("[Controller] Starting episode recording...")

        # Re-anchor VR reference
        vr_frame = self._read_vr_frame()
        if vr_frame is None:
            print("  [ERROR] No VR frame available, cannot reset mapper.")
            return

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

        if self.recorder is not None:
            try:
                self.recorder.start_episode(task_label="teleop", operator="")
            except Exception:
                traceback.print_exc()

        self.state = ControllerState.RECORDING
        self.error_handler.clear()
        print("[Controller] TELEOP → RECORDING")

    def _stop_recording(self) -> None:
        print(f"[Controller] Stopping episode. frames={self.frame_count}")
        if self.recorder is not None and self.recorder.is_recording:
            try:
                path = self.recorder.stop_episode(success=True)
                if path:
                    print(f"  Saved to {path}")
            except Exception:
                traceback.print_exc()
        print("[Controller] RECORDING → IDLE")

    def _escalate_to_emergency(self, reason: str) -> None:
        print(f"[EMERGENCY_STOP] {reason}")
        self.state = ControllerState.EMERGENCY_STOP
        if not self.dry_run:
            self.robot.emergency_stop()
        self.running = False

    def _shutdown(self) -> None:
        print("[TeleopController] Shutting down...")
        if self.recorder is not None and self.recorder.is_recording:
            try:
                self.recorder.stop_episode(success=False)
            except Exception:
                pass
        if self.keyboard is not None:
            self.keyboard.stop()
        print(f"  Frames: {self.frame_count}")
        print(f"  IK: ok={self.ik_success_count} fail={self.ik_fail_count}")
        print(f"  Retarget: ok={self.retarget_success_count} fail={self.retarget_fail_count}")

    # ------------------------------------------------------------------
    # VR data source
    # ------------------------------------------------------------------

    def _read_vr_frame(self) -> dict[str, Any] | None:
        if self.ipc_buffer is not None:
            return self._read_vr_ipc()
        if self.tracker is not None:
            return self.tracker.get_latest()
        return None

    def _read_vr_ipc(self) -> dict[str, Any] | None:
        import pickle

        data, _ = self.ipc_buffer.read(last_seq=-1)
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except Exception as e:
            print(f"[WARN] VR IPC deserialization failed: {e}")
            return None

    # ------------------------------------------------------------------
    # EMA smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_smooth(
        new_val: np.ndarray, prev_val: np.ndarray | None, alpha: float
    ) -> np.ndarray:
        if prev_val is None:
            return np.asarray(new_val, dtype=np.float64).copy()
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * np.asarray(new_val, dtype=np.float64) + (1.0 - alpha) * prev_val

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_arm_torque(state: RobotState) -> bool:
        tau = state.arm_tau
        if not np.all(np.isfinite(tau)):
            return False
        return float(np.max(np.abs(tau))) < _ARM_TORQUE_LIMIT_NM

    @staticmethod
    def _check_hand_current(state: RobotState) -> bool:
        cur = state.hand_current
        if not np.all(np.isfinite(cur)):
            return False
        return float(np.max(cur)) < _HAND_CURRENT_LIMIT_MA

    @staticmethod
    def _check_hand_temperature(state: RobotState) -> bool:
        temp = state.hand_temperature
        if not np.all(np.isfinite(temp)):
            return False
        return float(np.max(temp)) < _HAND_TEMP_LIMIT_C

    @staticmethod
    def _check_retarget_valid(hand_qpos: np.ndarray) -> bool:
        """Check that retarget result looks physiologically plausible."""
        if not np.all(np.isfinite(hand_qpos)):
            return False
        if np.any(hand_qpos < -0.5) or np.any(hand_qpos > 2.5):
            return False
        return True

    def _compute_T_base_eef(self, state: RobotState) -> np.ndarray | None:
        """Compute 4x4 T_base_eef from EEF pose for camera extrinsics."""
        if not np.all(np.isfinite(state.eef_pos)):
            return None
        from dexmani_real.planner.pose_utils import quat_wxyz_to_mat

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = state.eef_pos
        T[:3, :3] = quat_wxyz_to_mat(state.eef_quat_wxyz)
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
        self, vr_frame: dict[str, Any] | None, quality_flags: int, now: float
    ) -> None:
        failed = QualityFlags.describe(quality_flags)
        flags_str = ",".join(failed) if failed else "ALL_OK"
        age_s = (
            self.tracking_quality._frame_age(vr_frame)
            if vr_frame is not None
            else float("inf")
        )
        seq = vr_frame.get("sequence_id", "?") if vr_frame else "?"
        print(
            f"[t={now:.1f}] frames={self.frame_count} "
            f"state={self.state.value} "
            f"vr_seq={seq} age={age_s*1000:.0f}ms "
            f"ik={self.ik_success_count}/{self.ik_fail_count} "
            f"retarget={self.retarget_success_count}/{self.retarget_fail_count} "
            f"[{flags_str}]"
        )


def example() -> None:
    """Dry-run example: no hardware required, tests full control logic."""
    import multiprocessing

    from dexmani_real.planner.arm_planner import XArm7MotionPlanner
    from dexmani_real import ASSET_DIR
    from dexmani_real.planner.planner_types import (
        PlanningProfile,
        TeleopProfile,
        XArm7PlannerConfig,
    )
    from dexmani_real.robot.robot_interface import RobotInterfaceConfig
    from dexmani_real.teleop.arm_wrist_mapper import ArmWristMapper
    from dexmani_real.teleop.hand_retarget import XHandRetargeter

    q = multiprocessing.Queue()

    # Planner
    urdf_path = str(ASSET_DIR / "robots" / "xarm7" / "xarm7_glb.urdf")
    srdf_path = str(ASSET_DIR / "robots" / "xarm7" / "xarm7_glb_mplib.srdf")
    planner_config = XArm7PlannerConfig(
        urdf_path=urdf_path,
        srdf_path=srdf_path,
        eef_link_name="custom_link_eef",
    )
    planner = XArm7MotionPlanner(
        config=planner_config,
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(),
    )

    robot_config = RobotInterfaceConfig()
    robot = RobotInterface(
        config=robot_config,
        kinematics=planner.kin,
        planner=planner,
    )

    arm_mapper = ArmWristMapper()
    retargeter = XHandRetargeter()

    controller = TeleopController(
        robot=robot,
        arm_mapper=arm_mapper,
        retargeter=retargeter,
        planner=planner,
        keyboard_queue=q,
        dry_run=True,
        target_hz=50.0,
    )
    print("Starting dry-run TeleopController. Press Ctrl+C to stop.")
    print("Insert dummy VR frames for 5 s to simulate tracking...")

    # Inject a dummy VR frame so the arm mapper can be reset
    import time

    dummy_frame = {
        "wrist_pos": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "landmarks": np.zeros((21, 3), dtype=np.float64),
        "sequence_id": 0,
        "local_recv_ns": time.monotonic_ns(),
    }
    class _DummyTracker:
        def get_latest(self, max_age_s=None):
            dummy_frame["local_recv_ns"] = time.monotonic_ns()
            dummy_frame["sequence_id"] = getattr(self, "_seq", 0)
            self._seq = getattr(self, "_seq", 0) + 1
            return dummy_frame.copy()

    controller.tracker = _DummyTracker()

    # Reset mapper so IK works
    arm_mapper.reset(
        wrist_pos=np.zeros(3),
        wrist_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        eef_pos=np.array([0.4, 0.0, 0.3]),
        eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # Press T to enter TELEOP
    q.put(ControlSignal.TELEOP)

    # Run for 3 seconds in background
    import threading

    t = threading.Thread(target=controller.run, daemon=True)
    t.start()
    time.sleep(3.0)
    q.put(ControlSignal.HOME)
    time.sleep(1.0)
    controller.stop()
    t.join(timeout=3.0)
    print("Dry-run complete.")


if __name__ == "__main__":
    example()
