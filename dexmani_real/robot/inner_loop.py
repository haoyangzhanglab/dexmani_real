"""Arm servo loop — Mode 6 joint online trajectory planning for xArm7.

Primary entry point: ``arm_loop(shared)`` — mp.Process target, reads
SharedStorage.arm_action_q, writes arm_state_ring. Communicates exclusively
through SharedStorage (no direct SDK access from other processes).

``ArmInnerLoop`` (class) is a legacy threading-based implementation retained
only for deprecated entry points (vr_teleop_arm_only*.py).

Mode 6: firmware performs online trajectory replanning with configurable
speed/acceleration limits. No inner-loop interpolation — commands forwarded
directly and firmware handles all trajectory smoothing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.throttle import ThrottledWarner

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ArmInnerLoopConfig:
    """Configuration for ArmInnerLoop (Mode 6: joint online trajectory planning).

    Attributes:
        joint_max_speed: Max joint speed (rad/s). Respected by firmware
                         trajectory planner. Default 120°/s.
        joint_max_acc: Max joint acceleration (rad/s²). Respected by firmware
                       trajectory planner. Default 900°/s².
        loop_period: Inner loop period in seconds. Default 1/30 (30Hz).
        target_timeout_s: Max age of target before auto-hold (0.2s).
    """

    joint_max_speed: float = 2.0944  # 120°/s in rad/s
    joint_max_acc: float = 15.708  # 900°/s² in rad/s²
    loop_period: float = 1.0 / 30.0  # 30Hz
    target_timeout_s: float = 0.2

    # Absolute joint limit clip — mirrors xarm7 URDF joint limits exactly.
    # URDF source: assets/robots/xhand/xarm7_xhand_collision.urdf
    joint_limit_lower: tuple[float, ...] = (-6.28318530718, -2.059, -6.28318530718, -0.19198, -6.28318530718, -1.69297, -6.28318530718)
    joint_limit_upper: tuple[float, ...] = (6.28318530718, 2.0944, 6.28318530718, 3.927, 6.28318530718, 3.14159265359, 6.28318530718)

    # Tracking error warning threshold (rad). Diagnostic only — does not
    # alter commands or trigger error state.
    tracking_error_warn_rad: float = 0.35

    # Arm connection (single source of truth for IP).
    arm_ip: str = "192.168.1.215"

    # Home position — single source of truth for all homing paths.
    home_qpos: tuple[float, ...] = (0.0, -0.349, 0.0, 1.571, 0.0, 1.047, 0.0)

    # Collision sensitivity level (0-5, 1 = most sensitive).
    collision_sensitivity: int = 1

    # Homing parameters for _simple_homing.
    homing_convergence_rad: float = 0.0174533  # ~1 degree
    homing_steps: int = 50
    homing_step_interval_s: float = 0.04

    # Arm loop control rate (Hz). State published at this rate regardless of
    # action arrival cadence. Non-blocking queue reads ensure timely state updates.
    arm_loop_hz: float = 30.0


# Controller errors that indicate a problematic target rather than a hardware fault.
_RECOVERABLE_ERRORS: frozenset[int] = frozenset({22, 24, 31})


# ═══════════════════════════════════════════════════════════════════
# ArmInnerLoop (legacy — for replay_traj.py)
# ═══════════════════════════════════════════════════════════════════


class ArmInnerLoop:
    """30Hz inner loop thread — owns the XArmAPI connection, runs Mode 6.

    Runs in the same process as the controller. Communicates via
    Lock-protected shared variables (no SHM/IPC overhead).
    """

    def __init__(self, ip: str = "192.168.1.111", cfg: ArmInnerLoopConfig | None = None) -> None:
        self._ip = ip
        self._cfg = cfg or ArmInnerLoopConfig()

        self._lock = threading.Lock()
        self._arm_target: np.ndarray | None = None
        self._target_ts: float = 0.0
        self._arm_qpos: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        self._arm_qvel: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        self._arm_tau: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        self._error_state: bool = False
        self._last_sent_target: np.ndarray | None = None
        self._last_sent_cmd: np.ndarray = np.zeros(7, dtype=np.float64)
        self._arm = None
        self._tracking_error: float = 0.0
        self._mode_warn = ThrottledWarner(interval_s=2.0)
        self._tracking_warn = ThrottledWarner(interval_s=5.0)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()

    # ── Public API ──

    def set_target(self, target: np.ndarray | None) -> None:
        with self._lock:
            if target is not None:
                self._arm_target = np.asarray(target, dtype=np.float64).ravel()[:7].copy()
            else:
                self._arm_target = None
            self._target_ts = time.perf_counter()

    def get_state(self) -> tuple[np.ndarray, bool, float]:
        with self._lock:
            return self._arm_qpos.copy(), self._error_state, self._target_ts

    def get_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._arm_qvel.copy(), self._arm_tau.copy()

    def get_state_and_dynamics(self) -> tuple[np.ndarray, bool, float, np.ndarray, np.ndarray]:
        with self._lock:
            return (
                self._arm_qpos.copy(), self._error_state, self._target_ts,
                self._arm_qvel.copy(), self._arm_tau.copy(),
            )

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def ensure_running(self) -> bool:
        if not self.is_alive:
            self.start()
        return self.wait_ready(timeout=30.0)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def tracking_error(self) -> float:
        with self._lock:
            return self._tracking_error

    @property
    def last_sent_cmd(self) -> np.ndarray:
        with self._lock:
            return self._last_sent_cmd.copy()

    @property
    def mode(self) -> int:
        arm = self._arm
        if arm is not None:
            try:
                return int(getattr(arm, "mode", -1) or -1)
            except Exception:
                return -1
        return -1

    @property
    def connected(self) -> bool:
        arm = self._arm
        if arm is not None:
            try:
                return bool(arm.connected)
            except Exception:
                return False
        return False

    # ── Lifecycle ──

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run, name="arm-inner-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        if not self.is_alive:
            return
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("ArmInnerLoop: stop timed out after %.1fs", timeout)
        self._thread = None

    def emergency_stop(self, settle_timeout: float | None = None) -> bool:
        """Issue set_state(4) on the live connection, then stop the thread."""
        self._stop_event.set()
        arm = self._arm
        if arm is not None:
            try:
                arm.set_state(4)
            except Exception:
                pass
        if self.is_alive:
            budget = 3.0 * self._cfg.loop_period if settle_timeout is None else float(settle_timeout)
            thread = self._thread
            if thread is not None:
                thread.join(timeout=max(budget, 0.05))
        self._thread = None
        return True

    # ── Initialization ──

    @staticmethod
    def _connect_arm(ip: str):
        from xarm.wrapper import XArmAPI
        try:
            return XArmAPI(ip, is_radian=True)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.error("ArmInnerLoop: XArmAPI init failed: %s", e)
            return None

    def _init_mode_6(self, arm) -> bool:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_mode(6)
        arm.set_state(0)
        arm.set_collision_sensitivity(self._cfg.collision_sensitivity)
        arm.set_joint_maxacc(self._cfg.joint_max_acc, is_radian=True)
        actual_mode = getattr(arm, "mode", -1)
        if actual_mode != 6:
            logger.error("ArmInnerLoop: failed to set mode 6 (arm mode=%d)", actual_mode)
            return False
        return True

    def _bootstrap_state(self, arm) -> np.ndarray | None:
        code, states = arm.get_joint_states(is_radian=True, num=3)
        if code != 0 or len(states) == 0:
            logger.error("ArmInnerLoop: failed to read initial state (code=%d)", code)
            return None
        current_qpos = np.asarray(states[0], dtype=np.float64)[:7].copy()
        if not np.all(np.isfinite(current_qpos)):
            logger.error("ArmInnerLoop: initial qpos contains NaN/Inf")
            return None
        with self._lock:
            self._arm_qpos = current_qpos.copy()
            self._error_state = False
            self._last_sent_cmd = current_qpos.copy()
            self._last_sent_target = current_qpos.copy()
        if len(states) > 1:
            self._arm_qvel = np.asarray(states[1], dtype=np.float64)[:7].copy()
        if len(states) > 2:
            self._arm_tau = np.asarray(states[2], dtype=np.float64)[:7].copy()
        return current_qpos

    # ── Error handling ──

    def _handle_arm_error(self, arm, arm_error: int) -> str:
        """Return 'break' for non-recoverable, 'continue' for recoverable, 'ok' otherwise."""
        if arm_error == 0:
            return "ok"
        if arm_error in _RECOVERABLE_ERRORS:
            logger.warning("ArmInnerLoop: recoverable error C%d — clearing", arm_error)
            arm.clean_error()
            arm.set_mode(6)
            arm.set_state(0)
            return "continue"
        logger.error("ArmInnerLoop: non-recoverable error C%d", arm_error)
        with self._lock:
            self._error_state = True
        return "break"

    def _read_and_update_state(self, arm) -> np.ndarray | None:
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
        except (RuntimeError, OSError) as e:
            logger.error("ArmInnerLoop: get_joint_states failed: %s", e)
            return None
        if code != 0 or len(states) == 0:
            return None
        current_qpos = np.asarray(states[0], dtype=np.float64)[:7]
        if not np.all(np.isfinite(current_qpos)):
            return None
        with self._lock:
            self._arm_qpos = current_qpos.copy()
            if len(states) > 1:
                self._arm_qvel = np.asarray(states[1], dtype=np.float64)[:7].copy()
            if len(states) > 2:
                self._arm_tau = np.asarray(states[2], dtype=np.float64)[:7].copy()
        return current_qpos

    def _recover_mode(self, arm) -> bool:
        """Clear error latch and re-init Mode 6 after a recoverable error."""
        arm.clean_error()
        arm.clean_warn()
        arm.set_state(0)
        arm.set_mode(6)
        arm.set_state(0)
        actual_mode = getattr(arm, "mode", -1)
        if actual_mode == 6:
            logger.info("ArmInnerLoop: Mode 6 re-initialised")
            return True
        logger.warning("ArmInnerLoop: set_mode(6) returned but mode=%d", actual_mode)
        return False

    # ── Monitor + Send ──

    def _monitor(self, arm, current_qpos: np.ndarray) -> None:
        """Tracking error + mode drift monitor (passive, diagnostic only)."""
        err = 0.0
        if self._last_sent_target is not None:
            err = float(np.max(np.abs(self._last_sent_target - current_qpos[:7])))
        with self._lock:
            self._tracking_error = err
        if err > self._cfg.tracking_error_warn_rad:
            self._tracking_warn(
                "ArmInnerLoop: tracking error %.3f rad > threshold %.3f rad",
                err, self._cfg.tracking_error_warn_rad,
            )
        if getattr(arm, "mode", 6) != 6:
            self._mode_warn("ArmInnerLoop: arm mode=%s (expected 6) — re-initialising",
                            getattr(arm, "mode", "?"))
            self._recover_mode(arm)

    def _send_target(self, arm, target: np.ndarray) -> None:
        """Forward target → set_servo_angle(wait=False). Firmware handles smoothing."""
        clamped = target[:7].copy()
        np.clip(clamped, self._cfg.joint_limit_lower, self._cfg.joint_limit_upper, out=clamped)
        try:
            code = arm.set_servo_angle(
                angle=clamped.tolist(), is_radian=True,
                speed=self._cfg.joint_max_speed, mvacc=self._cfg.joint_max_acc, wait=False,
            )
        except (RuntimeError, OSError) as e:
            logger.error("ArmInnerLoop: set_servo_angle failed: %s", e)
            with self._lock:
                self._error_state = True
            return
        if code != 0:
            if hasattr(arm, 'get_err_warn_code'):
                try:
                    ret, err_warn = arm.get_err_warn_code()
                    err_code = int(err_warn[0]) if len(err_warn) >= 1 else -1
                except Exception:
                    err_code = -1
            else:
                err_code = -1
            if err_code in _RECOVERABLE_ERRORS:
                logger.warning("ArmInnerLoop: set_servo_angle code=%d err=%d — recoverable, re-initing", code, err_code)
                with self._lock:
                    self._arm_target = None
                self._recover_mode(arm)
                return
            elif code == 9 and err_code <= 0:
                self._mode_warn("ArmInnerLoop: code=%d err=%d — mode drop, re-initing", code, err_code)
                with self._lock:
                    self._arm_target = None
                self._recover_mode(arm)
                return
            else:
                logger.error("ArmInnerLoop: set_servo_angle code=%d err=%d", code, err_code)
                with self._lock:
                    self._error_state = True
        else:
            self._last_sent_target = clamped.copy()
            self._last_sent_cmd = clamped.copy()

    # ── Main loop ──

    def _run(self) -> None:
        arm = self._connect_arm(self._ip)
        if arm is None:
            with self._lock:
                self._error_state = True
            return
        self._arm = arm

        try:
            if not self._init_mode_6(arm):
                with self._lock:
                    self._error_state = True
                return

            current_qpos = self._bootstrap_state(arm)
            if current_qpos is None:
                with self._lock:
                    self._error_state = True
                return

            last_target_ts: float = 0.0
            last_valid_qpos: np.ndarray = current_qpos.copy()
            inner_dt = self._cfg.loop_period
            freq_hz = int(round(1.0 / inner_dt))
            limiter = RateManager(float(freq_hz))

            logger.info("ArmInnerLoop: %dHz Mode 6 (speed=%.0f°/s, acc=%.0f°/s²)",
                        freq_hz, float(np.degrees(self._cfg.joint_max_speed)),
                        float(np.degrees(self._cfg.joint_max_acc)))
            self._ready_event.set()

            while not self._stop_event.is_set():
                limiter.wait()

                # Read target
                with self._lock:
                    target = self._arm_target.copy() if self._arm_target is not None else None
                    target_ts = self._target_ts

                now = time.perf_counter()

                # Timeout → hold
                no_target_yet = last_target_ts == 0.0
                if not no_target_yet and (
                    target is None or (now - max(target_ts, last_target_ts) > self._cfg.target_timeout_s)
                ):
                    continue
                if target is None:
                    continue

                last_target_ts = target_ts

                # NaN guard
                if not np.all(np.isfinite(target)):
                    try:
                        arm.set_servo_angle(angle=last_valid_qpos.tolist(), is_radian=True,
                                           speed=self._cfg.joint_max_speed,
                                           mvacc=self._cfg.joint_max_acc, wait=False)
                    except (RuntimeError, OSError):
                        logger.warning("ArmInnerLoop: NaN-recovery set_servo_angle failed", exc_info=True)
                    continue

                last_valid_qpos = target[:7].copy()

                # Read hardware state
                current_qpos = self._read_and_update_state(arm)
                if current_qpos is None:
                    continue

                # Error check
                arm_error = getattr(arm, "error_code", 0)
                action = self._handle_arm_error(arm, arm_error)
                if action == "break":
                    break
                elif action == "continue":
                    continue

                # Monitor + send
                self._monitor(arm, current_qpos)
                self._send_target(arm, target)

        except Exception:
            logger.exception("ArmInnerLoop: fatal error in main loop")
            with self._lock:
                self._error_state = True
        finally:
            self._ready_event.clear()
            try:
                arm.disconnect()
            except Exception:
                pass
            self._arm = None
            logger.info("ArmInnerLoop: stopped")


# ═══════════════════════════════════════════════════════════════════
# New architecture: arm_loop (mp.Process target)
# ═══════════════════════════════════════════════════════════════════


def arm_loop(shared, config: ArmInnerLoopConfig | None = None) -> None:
    """Arm process entry point — reads SharedStorage.arm_action_q, servos arm.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).
    """
    from queue import Empty

    from scipy.spatial.transform import Rotation

    from dexmani_real.shm.shared_storage import HOME_SENTINEL, ARM_STATE_DTYPE, new_frame
    from dexmani_real.robot.safety import SafetyState, transition

    _tracking_warn = ThrottledWarner(interval_s=5.0)
    _fk_warn = ThrottledWarner(interval_s=5.0)
    cfg = config or ArmInnerLoopConfig()

    HOME_QPOS = np.array(cfg.home_qpos, dtype=np.float64)

    try:
        from xarm.wrapper import XArmAPI
        arm = XArmAPI(cfg.arm_ip, is_radian=True)
    except Exception as e:
        logger.error("arm_loop: connect failed: %s", e)
        return

    try:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_mode(6)
        arm.set_state(0)
        arm.set_collision_sensitivity(cfg.collision_sensitivity)
        arm.set_joint_maxacc(cfg.joint_max_acc, is_radian=True)
        if getattr(arm, "mode", -1) != 6:
            logger.error("arm_loop: failed to set mode 6")
            _disconnect_arm(arm)
            return
    except Exception as e:
        logger.error("arm_loop: init failed: %s", e)
        _disconnect_arm(arm)
        return

    # Seed last_qpos — FAIL if initial state unreadable (safety: never cmd HOME_QPOS blind).
    try:
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code == 0 and len(states) > 0:
            last_qpos = np.asarray(states[0], dtype=np.float64)[:7].copy()
        else:
            logger.error("arm_loop: cannot read initial joint states (code=%d)", code)
            _disconnect_arm(arm)
            return
    except Exception as e:
        logger.error("arm_loop: joint states read failed: %s", e)
        _disconnect_arm(arm)
        return
    last_target = last_qpos.copy()

    shared.arm_ready.set()
    logger.info("arm_loop: ready (Mode 6, ip=%s, hz=%.0f)", cfg.arm_ip, cfg.arm_loop_hz)

    _interval = 1.0 / cfg.arm_loop_hz
    _last_tick = time.monotonic()
    while shared.is_running.value:
        # Heartbeat — written even when holding position (proves we're alive)
        shared.arm_heartbeat_s.value = time.monotonic()

        if shared.estop_request.value:
            try:
                arm.set_state(4)
            except Exception:
                logger.warning("arm_loop: estop set_state(4) failed", exc_info=True)
            break

        # Safety state gate — only process commands in ARMED or RUNNING.
        # When gated (DISARMED or FAULT), skip action read + servo but continue
        # to publish state (for monitoring) and rate-limit normally.
        _safety = shared.safety_state.value
        if _safety in (SafetyState.ARMED, SafetyState.RUNNING):

            # Read action from queue (non-blocking — rate limiter controls cadence)
            try:
                action = shared.arm_action_q.get(timeout=0.0)
            except Empty:
                action = None

            # HOME sentinel → homing
            if action == HOME_SENTINEL:
                logger.info("arm_loop: HOME sentinel — executing homing")
                _simple_homing(arm, HOME_QPOS, cfg, shared=shared)
                last_qpos = HOME_QPOS.copy()
                last_target = HOME_QPOS.copy()
                continue

            # Servo
            if action is not None and isinstance(action, dict):
                target = np.asarray(action.get("qpos", last_target), dtype=np.float64).ravel()[:7]
                if np.all(np.isfinite(target)):
                    last_target = target.copy()

            try:
                code = arm.set_servo_angle(angle=last_target, is_radian=True,
                                           speed=cfg.joint_max_speed, mvacc=cfg.joint_max_acc, wait=False)
                if code != 0:
                    err_code = getattr(arm, "error_code", 0)
                    if err_code in (22, 24, 31):
                        arm.clean_error()
                        arm.set_mode(6)
                        arm.set_state(0)
                    elif err_code != 0:
                        logger.error("arm_loop: set_servo_angle code=%d err=%d", code, err_code)
            except Exception:
                logger.warning("arm_loop: set_servo_angle failed", exc_info=True)

        # Read state
        arm_connected = True
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
            if code == 0 and len(states) > 0:
                qpos = np.asarray(states[0], dtype=np.float64)[:7]
                qvel = np.asarray(states[1], dtype=np.float64)[:7] if len(states) > 1 else np.zeros(7)
                tau = np.asarray(states[2], dtype=np.float64)[:7] if len(states) > 2 else np.zeros(7)
                last_qpos = qpos.copy()
            else:
                qpos, qvel, tau = last_qpos.copy(), np.zeros(7), np.zeros(7)
                arm_connected = False
        except Exception:
            logger.warning("arm_loop: get_joint_states failed", exc_info=True)
            qpos, qvel, tau = last_qpos.copy(), np.zeros(7), np.zeros(7)
            arm_connected = False

        # FK: read actual EEF pose from arm controller
        try:
            fk_code, fk_pose = arm.get_position_aa(is_radian=True)
            if fk_code == 0 and len(fk_pose) >= 6:
                eef_pos = np.asarray(fk_pose[:3], dtype=np.float64) / 1000.0  # mm → m
                rx, ry, rz = float(fk_pose[3]), float(fk_pose[4]), float(fk_pose[5])
                R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
                eef_rot6d = np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)
            else:
                _fk_warn("arm_loop: FK failed code=%d — publishing zero EEF", fk_code)
                eef_pos = np.zeros(3, dtype=np.float64)
                eef_rot6d = np.zeros(6, dtype=np.float64)
        except Exception:
            _fk_warn("arm_loop: FK failed — publishing zero EEF")
            eef_pos = np.zeros(3, dtype=np.float64)
            eef_rot6d = np.zeros(6, dtype=np.float64)

        # Compute tracking error
        tracking_err = float(np.max(np.abs(qpos - last_target)))

        if tracking_err > cfg.tracking_error_warn_rad:
            _tracking_warn("arm_loop: tracking error %.3f rad > threshold %.3f rad", tracking_err, cfg.tracking_error_warn_rad)

        # Error handling
        try:
            error_code = arm.error_code
        except Exception:
            error_code = 0
            arm_connected = False

        if error_code in (22, 24, 31):
            try:
                arm.clean_error()
                arm.set_mode(6)
                arm.set_state(0)
            except Exception:
                pass
        elif error_code != 0:
            shared.error_state.value = True
            transition(shared, SafetyState.FAULT)
            break

        # Publish state
        frame = new_frame(ARM_STATE_DTYPE)
        frame["qpos"][0] = qpos
        frame["qvel"][0] = qvel
        frame["tau"][0] = tau
        frame["eef_pos"][0] = eef_pos
        frame["eef_rot6d"][0] = eef_rot6d
        frame["error_code"][0] = int(error_code)
        frame["connected"][0] = 1 if arm_connected else 0
        frame["mode"][0] = 6
        frame["tracking_err"][0] = tracking_err
        frame["timestamp"][0] = time.monotonic()
        shared.arm_state_ring.write(frame)

        # Rate limit
        _elapsed = time.monotonic() - _last_tick
        _sleep = _interval - _elapsed
        if _sleep > 0:
            time.sleep(_sleep)
        _last_tick = time.monotonic()

    # Cleanup
    try:
        arm.set_state(4)
        arm.disconnect()
    except Exception:
        logger.warning("arm_loop: cleanup failed", exc_info=True)
    logger.info("arm_loop: exited")


def _disconnect_arm(arm) -> None:
    """Disconnect arm safely, ignoring errors."""
    try:
        arm.disconnect()
    except Exception:
        pass


def _simple_homing(arm, home_qpos: np.ndarray, cfg: ArmInnerLoopConfig | None = None, *, shared=None) -> None:
    """Simple joint-space linear interpolation to home.

    Writes heartbeat to ``shared.arm_heartbeat_s`` during execution so that
    the 2 s homing sequence does not trigger a false FAULT timeout (1 s).
    """
    _cfg = cfg or ArmInnerLoopConfig()

    try:
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code == 0 and len(states) > 0:
            current = np.asarray(states[0], dtype=np.float64)[:7]
        else:
            return
    except Exception:
        return

    if np.max(np.abs(current - home_qpos)) < _cfg.homing_convergence_rad:
        return

    steps = _cfg.homing_steps
    for i in range(1, steps + 1):
        if shared is not None:
            shared.arm_heartbeat_s.value = time.monotonic()
        wp = current + (i / steps) * (home_qpos - current)
        try:
            arm.set_servo_angle(angle=wp, is_radian=True, wait=False)
        except Exception:
            break
        time.sleep(_cfg.homing_step_interval_s)

    try:
        arm.set_servo_angle(angle=home_qpos, is_radian=True, wait=False)
    except Exception:
        pass
