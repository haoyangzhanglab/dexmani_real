"""Teleoperation controller — simplified state machine with in-process inner loop.

State machine:
    IDLE ──T(teleop)──→ TELEOP ──R(record)──→ RECORDING
      ↑       │   S(stop)→IDLE      │   H(home)→IDLE
      ├───H(home)─────┘              │
      └──ESC / timeout: EMERGENCY_STOP

Arm position servo is handled by ArmInnerLoop (daemon thread, same process).
Controller sends target qpos via inner.set_target(), reads state via inner.get_state().
Ref: BunnyVisionPro _internal_control_arm_qpos() thread pattern.
"""

from __future__ import annotations

import queue
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.planning.pose_utils import quat_wxyz_to_rotmat
from dexmani_real.recording.collection_config import CollectionConfig
from dexmani_real.recording.collection_loop import CollectionLoop
from dexmani_real.robot.inner_loop import ArmInnerLoop, ArmInnerLoopConfig
from dexmani_real.robot.interface import RobotAction, RobotInterface, RobotInterfaceConfig, RobotState
from dexmani_real.robot.validate import validate_action
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.teleop.core.pipeline import TeleopPipeline
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_limiter import RateLimiter

if TYPE_CHECKING:
    from dexmani_real.planning.planner import XArm7MotionPlanner
    from dexmani_real.recording.episode_recorder import EpisodeRecorder
    from dexmani_real.sensor.multi_camera_manager import MultiCameraManager
    from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
    from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
    from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker

logger = get_logger(__name__)


class ControllerState(Enum):
    IDLE = "IDLE"
    TELEOP = "TELEOP"
    PAUSED = "PAUSED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


from dataclasses import dataclass


@dataclass
class TeleopControllerConfig:
    target_hz: float = 50.0
    ema_alpha_arm: float = 0.95
    dry_run: bool = False
    use_zmq_vr: bool = False
    zmq_vr_port: int = 5555
    use_shm_vr: bool = False
    inner_loop_cfg: ArmInnerLoopConfig | None = None
    use_precise_wait: bool = False
    collection_config: CollectionConfig | None = None
    multi_camera_configs: list | None = None
    multi_camera_auto_restart: bool = True


class TeleopController:
    """Main teleoperation controller with PID process isolation.

    Owns the control loop: reads VR, runs IK+retarget, applies robust EMA smoothing,
    enforces safety checks, manages recording lifecycle.
    """

    _VR_STALE_THRESHOLD_S: float = 0.5  # single threshold for VR loss → deceleration

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
        target_hz: float = 50.0,
        ema_alpha_arm: float = 0.95,
        dry_run: bool = False,
        recorder: EpisodeRecorder | None = None,
        use_zmq_vr: bool = False,
        zmq_vr_port: int = 5555,
        use_shm_vr: bool = False,
        camera_process: object | None = None,
    ) -> None:
        if cfg is None:
            cfg = TeleopControllerConfig(
                target_hz=target_hz,
                ema_alpha_arm=ema_alpha_arm,
                dry_run=dry_run,
                use_zmq_vr=use_zmq_vr,
                zmq_vr_port=zmq_vr_port,
                use_shm_vr=use_shm_vr,
            )

        self.robot = robot
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self.tracker = tracker
        self.dry_run = cfg.dry_run

        # ── Arm inner loop (in-process thread, 250Hz) ──
        self._arm_inner: ArmInnerLoop | None = None
        if not self.dry_run:
            inner_cfg = cfg.inner_loop_cfg if cfg.inner_loop_cfg is not None else ArmInnerLoopConfig()
            self._arm_inner = ArmInnerLoop(cfg=inner_cfg)
            self._arm_inner.start()
            mode_label = {4: "velocity control + PID", 1: "position servo"}.get(
                inner_cfg.control_mode, f"mode {inner_cfg.control_mode}"
            )
            logger.info("ArmInnerLoop started (mode %d, %s, 250Hz)", inner_cfg.control_mode, mode_label)

        # Recording with async writer thread (offloads HDF5 I/O from hot path)
        if recorder is not None:
            coll_cfg = cfg.collection_config or CollectionConfig()
            self._collection_loop = CollectionLoop(recorder, coll_cfg)
            self.recorder = recorder
            self._recording_queue: queue.Queue[dict | None] = queue.Queue(maxsize=500)
            self._recording_thread = threading.Thread(
                target=self._recording_writer, name="recording_writer", daemon=True
            )
            self._recording_thread.start()
        else:
            self._collection_loop = None
            self.recorder = None
            self._recording_queue = None
            self._recording_thread = None

        self.limiter = RateLimiter(cfg.target_hz)
        self.ema_alpha_arm = float(cfg.ema_alpha_arm)

        # Camera
        self._camera_process = camera_process
        self._multi_camera: MultiCameraManager | None = None
        if cfg.multi_camera_configs is not None and len(cfg.multi_camera_configs) > 0:
            from dexmani_real.sensor.multi_camera_manager import MultiCameraConfig, MultiCameraManager

            mc_cfg = MultiCameraConfig(auto_restart=cfg.multi_camera_auto_restart)
            self._multi_camera = MultiCameraManager(cfg.multi_camera_configs, mc_cfg)
            if camera_process is None:
                self._multi_camera.start_all()

        # VR data sources (priority: SHM > ZMQ > Tracker)
        self._vr_shm: object | None = None  # SharedMemoryFrameManager
        self._vr_subscriber = None

        if cfg.use_shm_vr:
            from dexmani_real.shm.frame_manager import SharedMemoryFrameManager

            try:
                self._vr_shm = SharedMemoryFrameManager(create=False)
                logger.info("VR source: SharedMemory (attached, zero-copy)")
            except FileNotFoundError:
                logger.warning("VR SHM not found — falling back to ZMQ/tracker")
                self._vr_shm = None
        elif cfg.use_zmq_vr:
            from dexmani_real.teleop.vr.vr_publisher import VRFrameSubscriber

            self._vr_subscriber = VRFrameSubscriber(sub_port=cfg.zmq_vr_port)
            self._vr_subscriber.connect()
            logger.info("VR source: ZMQ SUB (tcp://127.0.0.1:%d)", cfg.zmq_vr_port)

        # State
        self.state = ControllerState.IDLE
        self.recording = False
        self.running = False
        self._last_arm_cmd: np.ndarray | None = None
        self._last_hand_cmd: np.ndarray | None = None
        self._last_good_arm: np.ndarray | None = None
        self._last_good_hand: np.ndarray | None = None

        # Pipeline
        self.pipeline = TeleopPipeline(arm_mapper, retargeter, planner, ema_alpha_arm=self.ema_alpha_arm)

        # Keyboard (accepts queue=None for backward compatibility;
        # KeyboardHandler now uses termios cbreak + select internally)
        self.keyboard = KeyboardHandler(keyboard_queue)

        # Status
        self.frame_count: int = 0
        self.ik_success_count: int = 0
        self.ik_fail_count: int = 0
        self.retarget_success_count: int = 0
        self.retarget_fail_count: int = 0
        self.last_status_ts: float = 0.0
        self.status_interval: float = 2.0

    # ── Lifecycle ──

    def start(self) -> None:
        if self.keyboard is not None:
            self.keyboard.start()
        self.running = True

    def stop(self) -> None:
        self.running = False
        if self.keyboard is not None:
            self.keyboard.stop()

    def run(self) -> None:
        self.start()
        if not self.dry_run and not self.robot.is_connected():
            logger.info("Robot not connected. Attempting connect...")
            result = self.robot.connect()
            logger.info("connect result: %s", result)

        logger.info("Entering main loop at %.0f Hz", self.limiter.target_hz)
        logger.info("  Controls: B=begin S=stop C=pause H=home Q=quit ESC=emergency")

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

    # ── Main tick ──

    def _tick(self) -> None:
        if self.state == ControllerState.EMERGENCY_STOP:
            return

        tick_start = time.perf_counter()
        self.frame_count += 1

        # PAUSED: send None-sentinel → inner loop holds position
        if self.state == ControllerState.PAUSED:
            if not self.dry_run and self._arm_inner is not None:
                self._arm_inner.set_target(None)
            return

        # IDLE: no pipeline
        if self.state == ControllerState.IDLE:
            return

        # ── 1. Get VR frame ──
        vr_frame = self._read_vr_frame()
        age_s = self._frame_age(vr_frame) if vr_frame is not None else float("inf")

        # ── 2. VR staleness check (single threshold) ──
        if age_s > self._VR_STALE_THRESHOLD_S or vr_frame is None:
            if not self.dry_run and self._arm_inner is not None:
                self._arm_inner.set_target(None)  # hold position
            return

        # ── 3. Read arm state from PID process ──
        if self.dry_run:
            state = self._dummy_state()
            arm_qpos = state.arm_qpos
        else:
            arm_qpos, error_state, _inner_ts = self._arm_inner.get_state()
            now = time.perf_counter()
            if error_state:
                self._escalate_to_emergency("Arm inner loop error")
                return
            state = self.robot.get_state(arm_qpos=arm_qpos)

        # ── 4. Compute action ──
        action, status = self._compute_action(vr_frame, state)

        if status.get("ik_ok"):
            self._last_arm_cmd = action.arm_qpos_cmd.copy()
        if status.get("retarget_ok"):
            self._last_hand_cmd = action.hand_qpos_cmd.copy()

        # ── 5. Record frame ──
        camera_frame = None
        camera_frames: dict[str, dict] | None = None

        if self._multi_camera is not None:
            try:
                camera_frames = self._multi_camera.read_all_latest()
            except (ValueError, RuntimeError, KeyError):
                pass
        elif self._camera_process is not None:
            try:
                camera_frame = self._camera_process.poll_latest_frame()
            except (ValueError, RuntimeError, KeyError):
                pass

        T_base_eef = self._compute_T_base_eef(state)

        if self.recording and self._collection_loop is not None and self._recording_queue is not None:
            try:
                self._recording_queue.put_nowait(
                    dict(
                        state=state,
                        action=action,
                        vr_frame=vr_frame,
                        camera_frame=camera_frame,
                        camera_frames=camera_frames,
                        T_base_eef=T_base_eef,
                    )
                )
            except queue.Full:
                logger.warning("Recording queue full — dropping frame")

        # ── 6. Pre-send validation ──
        if not self.dry_run:
            action_valid, fail_reason = validate_action(self.robot, action, actual_arm_qpos=arm_qpos)
            if not action_valid:
                if "error state" in fail_reason or "not connected" in fail_reason:
                    self._escalate_to_emergency(f"Robot error before send: {fail_reason}")
                    return
                logger.warning("Pre-send safety: %s — holding", fail_reason)
                hold = self._hold_action()
                action = RobotAction(arm_qpos_cmd=hold.arm_qpos_cmd, hand_qpos_cmd=hold.hand_qpos_cmd)

            # Send arm target to inner loop
            self._arm_inner.set_target(action.arm_qpos_cmd)
            # Send hand directly
            self.robot.send_action(action)

        # ── 7. Periodic status ──
        now = time.monotonic()
        if now - self.last_status_ts >= self.status_interval:
            self.last_status_ts = now
            self._print_status(vr_frame, now)

        # ── 8. Overrun detection ──
        tick_elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
        target_ms = self.limiter.period * 1000.0
        if tick_elapsed_ms > target_ms * 1.5:
            logger.warning("Loop overrun: tick=%.1fms target=%.1fms", tick_elapsed_ms, target_ms)

    # ── Action computation ──

    def _compute_action(self, vr_frame: dict, state: RobotState) -> tuple[RobotAction, dict[str, bool]]:
        current_arm_qpos = state.arm_qpos
        current_hand_qpos = state.hand_qpos
        prev_arm_cmd = self._last_arm_cmd if self._last_arm_cmd is not None else current_arm_qpos
        prev_hand_cmd = self._last_hand_cmd if self._last_hand_cmd is not None else current_hand_qpos

        self._init_fallback(current_arm_qpos, current_hand_qpos)

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

        # Retarget bounds check — uses XHandConfig per-joint limits
        if retarget_ok:
            hq = action.hand_qpos_cmd
            hand_min = self.robot.hand.config.qpos_min
            hand_max = self.robot.hand.config.qpos_max
            if not np.all(np.isfinite(hq)) or np.any(hq < hand_min) or np.any(hq > hand_max):
                retarget_ok = False

        if ik_ok and retarget_ok:
            self._last_good_arm = np.asarray(action.arm_qpos_cmd, dtype=np.float64).copy()
            self._last_good_hand = np.asarray(action.hand_qpos_cmd, dtype=np.float64).copy()

        if ik_ok:
            self.ik_success_count += 1
        else:
            self.ik_fail_count += 1
        if retarget_ok:
            self.retarget_success_count += 1
        else:
            self.retarget_fail_count += 1

        return action, {"ik_ok": ik_ok, "retarget_ok": retarget_ok}

    # ── State machine transitions ──

    def _handle_keyboard(self) -> None:
        if self.keyboard is None:
            return
        for sig in self.keyboard.poll():
            self._transition(sig)

    def _transition(self, signal: ControlSignal) -> None:
        if signal == ControlSignal.EMERGENCY_STOP:
            self._escalate_to_emergency("Keyboard ESC")
            return

        if self.state == ControllerState.EMERGENCY_STOP:
            if signal == ControlSignal.QUIT:
                self.running = False
            elif signal == ControlSignal.HOME:
                if not self.dry_run:
                    self.robot.clear_error()
                self.state = ControllerState.IDLE
                self._do_home()
            return

        if signal == ControlSignal.QUIT:
            if self.state == ControllerState.IDLE:
                self.running = False
            else:
                self._stop_recording()
                self.state = ControllerState.IDLE
                logger.info("=== STATE: → IDLE (quit) ===")
            return

        if signal == ControlSignal.BEGIN:
            if self.state == ControllerState.IDLE:
                if not self.dry_run:
                    self._ensure_inner_running()
                self._reset_mapper()
                self._start_recording()
                self.state = ControllerState.TELEOP
                self.recording = True
                logger.info("=== STATE: IDLE → TELEOP ===")
            return

        if signal == ControlSignal.STOP:
            if self.state in (ControllerState.TELEOP, ControllerState.PAUSED):
                self._stop_recording()
                self.state = ControllerState.IDLE
                logger.info("=== STATE: → IDLE (stopped) ===")
            return

        if signal == ControlSignal.PAUSE:
            if self.state == ControllerState.TELEOP:
                self.state = ControllerState.PAUSED
                logger.info("=== STATE: TELEOP → PAUSED ===")
            elif self.state == ControllerState.PAUSED:
                if self._reset_mapper():
                    self.state = ControllerState.TELEOP
                    logger.info("=== STATE: PAUSED → TELEOP ===")
            return

        if signal == ControlSignal.HOME:
            if self.state != ControllerState.EMERGENCY_STOP:
                logger.info("=== STATE: %s → HOME ===", self.state.value)
                self._do_home()
            return

    def _do_home(self) -> None:
        logger.info("Returning to home...")
        self._last_arm_cmd = None
        self._last_hand_cmd = None

        if self.recording and self._collection_loop is not None and self._collection_loop.is_recording:
            try:
                self._collection_loop.stop_episode(success=False, reason="home")
            except (ValueError, OSError):
                pass

        # Stop inner loop before return_to_home to avoid dual-connection conflicts
        # (RobotInterface.return_to_home() uses its own XArmAPI connection)
        if not self.dry_run and self._arm_inner is not None and self._arm_inner.is_alive:
            self._arm_inner.set_target(None)  # signal hold
            self._arm_inner.stop()
            logger.info("Arm inner loop stopped for return-to-home")

        if not self.dry_run:
            self.robot.return_to_home()

        self.state = ControllerState.IDLE
        self.recording = False
        self._last_good_arm = None
        self._last_good_hand = None
        self.limiter.reset()
        logger.info("=== STATE: HOME → IDLE ===")

    def _ensure_inner_running(self) -> None:
        """Restart the inner loop thread if it has been stopped (e.g. after return-to-home)."""
        if self._arm_inner is None:
            return
        if self._arm_inner.is_alive:
            return
        logger.info("Restarting arm inner loop...")
        # Preserve the same inner loop config (mode, PID gains, etc.)
        inner_cfg = getattr(self._arm_inner, "_cfg", ArmInnerLoopConfig())
        self._arm_inner = ArmInnerLoop(cfg=inner_cfg)
        self._arm_inner.start()
        logger.info("Arm inner loop restarted")

    def _escalate_to_emergency(self, reason: str) -> None:
        logger.error("=== STATE: → EMERGENCY_STOP: %s ===", reason)
        self.state = ControllerState.EMERGENCY_STOP
        if not self.dry_run:
            # Stop inner loop first to prevent SDK error spam
            if self._arm_inner is not None and self._arm_inner.is_alive:
                self._arm_inner.set_target(None)
                self._arm_inner.stop()
            self.robot.emergency_stop()

    # ── Recording ──

    def _start_recording(self) -> None:
        logger.info("Starting episode recording...")
        if not self._reset_mapper():
            logger.error("Cannot start recording without VR frame.")
            return
        if self._collection_loop is not None:
            try:
                self._collection_loop.start_episode(task_label="teleop", operator="")
            except (ValueError, OSError) as e:
                logger.exception("start_episode failed: %s", e)
        self._last_good_arm = None
        self._last_good_hand = None
        logger.info("Episode recording started.")

    def _stop_recording(self) -> None:
        logger.info("Stopping episode. frames=%s", self.frame_count)
        if self._collection_loop is not None and self._collection_loop.is_recording:
            try:
                path = self._collection_loop.stop_episode(success=True, reason="manual")
                if path:
                    logger.info("  Saved to %s", path)
            except (ValueError, OSError) as e:
                logger.exception("stop_episode failed: %s", e)
        self.recording = False
        logger.info("Episode stopped.")

    # ── VR ──

    def _read_vr_frame(self) -> dict | None:
        # Priority: SharedMemory (zero-copy, ~1μs) > ZMQ (TCP, ~1-2ms) > Tracker (direct SDK)
        if self._vr_shm is not None:
            return self._vr_shm.read_latest_vr()
        if self._vr_subscriber is not None:
            return self._vr_subscriber.recv_latest()
        if self.tracker is not None:
            return self.tracker.get_latest()
        return None

    @staticmethod
    def _frame_age(frame: dict) -> float:
        local_recv = frame.get("local_recv_ns")
        if local_recv is not None and np.isfinite(local_recv):
            return (time.monotonic_ns() - local_recv) * 1e-9
        ts = frame.get("timestamp")
        if ts is not None and np.isfinite(ts):
            return max(0.0, time.perf_counter() - ts)
        return float("inf")

    def _reset_mapper(self) -> bool:
        vr_frame = self._read_vr_frame()
        if vr_frame is None:
            logger.warning("No VR frame available, cannot reset mapper.")
            return False
        if self.dry_run:
            state = self._dummy_state()
        else:
            if self._arm_inner is not None:
                arm_qpos, _, _ = self._arm_inner.get_state()
                state = self.robot.get_state(arm_qpos=arm_qpos)
            else:
                state = self.robot.get_state()
        self.arm_mapper.reset(
            wrist_pos=vr_frame["wrist_pos"],
            wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
            eef_pos=state.eef_pos,
            eef_quat_wxyz=state.eef_quat_wxyz,
        )
        return True

    # ── Hold-on-failure ──

    def _init_fallback(self, arm_qpos: np.ndarray, hand_qpos: np.ndarray) -> None:
        if self._last_good_arm is None:
            self._last_good_arm = np.asarray(arm_qpos, dtype=np.float64).copy()
        if self._last_good_hand is None:
            self._last_good_hand = np.asarray(hand_qpos, dtype=np.float64).copy()

    def _hold_action(self) -> RobotAction:
        arm = self._last_good_arm.copy() if self._last_good_arm is not None else np.zeros(7, dtype=np.float64)
        hand = self._last_good_hand.copy() if self._last_good_hand is not None else np.zeros(12, dtype=np.float64)
        return RobotAction(arm_qpos_cmd=arm, hand_qpos_cmd=hand)

    # ── EEF ──

    def _compute_T_base_eef(self, state: RobotState) -> np.ndarray | None:
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
            hand_tactile_force=np.zeros((5, 120, 3), dtype=np.float64),
            fingertip_pos=np.zeros((5, 3), dtype=np.float64),
            arm_connected=True,
            hand_connected=True,
            timestamp=time.perf_counter(),
        )

    # ── Status ──

    def _print_status(self, vr_frame: dict | None, now: float) -> None:
        age_s = self._frame_age(vr_frame) if vr_frame is not None else float("inf")
        seq = vr_frame.get("sequence_id", "?") if vr_frame else "?"
        rec = "REC" if self.recording else "   "
        logger.info(
            "[t=%.1f] frames=%s state=%s %s vr_seq=%s age=%sms ik=%s/%s retarget=%s/%s",
            now,
            self.frame_count,
            self.state.value,
            rec,
            seq,
            f"{age_s*1000:.0f}",
            self.ik_success_count,
            self.ik_fail_count,
            self.retarget_success_count,
            self.retarget_fail_count,
        )

    # ── Recording writer thread (offloads HDF5 I/O from 50Hz hot path) ──

    def _recording_writer(self) -> None:
        """Daemon thread: consume recording frames from queue, write to HDF5."""
        while True:
            item = self._recording_queue.get()
            if item is None:  # sentinel — stop
                break
            try:
                self._collection_loop.record_frame(**item)
            except (ValueError, OSError) as e:
                logger.exception("record_frame failed: %s", e)

    # ── Shutdown ──

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        if self.recording and self._collection_loop is not None and self._collection_loop.is_recording:
            try:
                self._collection_loop.stop_episode(success=False, reason="shutdown")
            except (ValueError, RuntimeError, KeyError):
                pass

        # Stop recording writer thread
        if self._recording_queue is not None and self._recording_thread is not None:
            self._recording_queue.put(None)  # sentinel
            self._recording_thread.join(timeout=5.0)

        # Stop VR shared memory access (consumer-side only — no cleanup needed)
        self._vr_shm = None

        # Stop arm inner loop
        if self._arm_inner is not None:
            self._arm_inner.stop()

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
