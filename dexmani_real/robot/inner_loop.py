"""ArmInnerLoop — in-process inner loop thread (50Hz) for xArm7 control.

Default and only control mode is Mode 6 (joint online trajectory planning). The firmware
performs online trajectory replanning with configurable speed and acceleration limits that
ARE respected. No inner-loop interpolation is needed — the target is forwarded directly and
the firmware handles all trajectory smoothing. Effectively dissolves the inner/outer loop
distinction.

Mode 6: Joint online trajectory planning — forwards targets directly to
  arm.set_servo_angle(wait=False) at 50Hz. Firmware respects speed/accel limits
  (default: 90°/s, 500°/s²). Smooth motion, no desk vibration. Requires firmware >= 1.10.0.

Architecture:
    Main Thread (50Hz)                     Inner Loop Thread (50Hz)
    ──────────────────                     ───────────────────────
    inner.set_target(cmd)    ──Lock──→     self._arm_target
    qpos, err, ts = get_state()  ←──Lock── self._arm_qpos, _error_state
                                           passthrough → set_servo_angle(wait=False)
                                           firmware handles all trajectory planning
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexmani_real.shm.sync_primitives import SharedSyncPrimitives

import numpy as np

from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_limiter import RateLimiter

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ArmInnerLoopConfig:
    """Configuration for ArmInnerLoop (Mode 6: joint online trajectory planning).

    Attributes:
        joint_max_speed: Max joint speed (rad/s). Respected by firmware trajectory
                         planner. Default 90°/s (≈1.57 rad/s).
        joint_max_acc: Max joint acceleration (rad/s²). Respected by firmware
                       trajectory planner. Default 500°/s² (≈8.73 rad/s²).
        loop_period: Inner loop period in seconds. Default 0.02 (50Hz).
        target_timeout_s: Max age of target before auto-hold (0.2s).
        synchronized: Two-phase handshake for policy inference (default False).
    """

    # Mode 6 parameters (speed/accel ARE respected by firmware trajectory planner)
    joint_max_speed: float = 1.5708   # 90°/s in rad/s
    joint_max_acc: float = 8.7266     # 500°/s² in rad/s²
    loop_period: float = 0.02         # 50Hz
    # Shared
    target_timeout_s: float = 0.2

    # Two-phase handshake: when True, ArmInnerLoop sets robot_ready after each
    # hardware write and waits for policy_ready before the next target dispatch.
    synchronized: bool = False


# ═══════════════════════════════════════════════════════════════════
# ArmInnerLoop
# ═══════════════════════════════════════════════════════════════════


class ArmInnerLoop:
    """50Hz inner loop thread — owns the XArmAPI connection, runs Mode 6.

    Runs in the same process as the controller. Communicates via
    Lock-protected shared variables (no SHM/IPC overhead).

    Mode 6 (joint online trajectory planning): the firmware performs online
    trajectory replanning with configurable speed/accel limits. The inner loop
    forwards targets directly — no interpolation, no PID.

    Parameters:
        ip: XArm controller IP address.
        dt: (deprecated) Inner loop period — unused in Mode 6; kept for API compat.
        cfg: Inner loop configuration (speed/accel limits, loop period, timeout).
    """

    def __init__(
        self,
        ip: str = "192.168.1.111",
        dt: float = 1.0 / 125.0,
        cfg: ArmInnerLoopConfig | None = None,
        sync: SharedSyncPrimitives | None = None,
    ) -> None:
        self._ip = ip
        self._dt = float(dt)  # kept for API compat; unused in Mode 6
        self._cfg = cfg or ArmInnerLoopConfig()
        self._sync = sync

        # ── Shared state (protected by _lock) ──
        self._lock = threading.Lock()
        self._arm_target: np.ndarray | None = None
        self._target_ts: float = 0.0
        self._arm_qpos: np.ndarray = np.zeros(7, dtype=np.float64)
        self._error_state: bool = False

        # ── Lifecycle ──
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

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Lifecycle ──

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ArmInnerLoop already running")
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run, name="arm_inner_loop", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("ArmInnerLoop thread did not exit within %.0fs", timeout)

    # ── Sync handshake ──

    def _signal_ready_and_sync(self) -> None:
        """Signal robot_ready and wait for policy_ready (two-phase handshake)."""
        if self._sync is None:
            return
        self._sync.robot_ready.set()
        self._sync.policy_ready.wait()
        self._sync.policy_ready.clear()

    def _signal_ready_only(self) -> None:
        """Signal robot_ready without waiting (used in timeout/hold paths)."""
        if self._sync is None:
            return
        self._sync.robot_ready.set()

    # ── Inner loop ──

    def _run(self) -> None:
        from xarm.wrapper import XArmAPI

        try:
            arm = XArmAPI(self._ip, is_radian=True)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.error("ArmInnerLoop: XArmAPI init failed: %s", e)
            with self._lock:
                self._error_state = True
            return

        try:
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            self._init_mode(arm)
            arm.set_collision_sensitivity(1)

            # Verify arm entered mode 6
            actual_mode = getattr(arm, 'mode', -1)
            if actual_mode != 6:
                logger.warning(
                    "ArmInnerLoop: arm mode=%d but expected 6 — re-initializing", actual_mode,
                )
                self._init_mode(arm)
                actual_mode = getattr(arm, 'mode', -1)
                if actual_mode != 6:
                    logger.error("ArmInnerLoop: failed to set mode 6 (arm mode=%d)", actual_mode)
                    with self._lock:
                        self._error_state = True
                    return

            # Read initial position
            code, states = arm.get_joint_states(is_radian=True, num=1)
            if code == 0 and len(states) > 0:
                current_qpos = np.asarray(states[0], dtype=np.float64)[:7].copy()
                if not np.all(np.isfinite(current_qpos)):
                    current_qpos = np.zeros(7, dtype=np.float64)
            else:
                current_qpos = np.zeros(7, dtype=np.float64)

            with self._lock:
                self._arm_qpos = current_qpos.copy()
                self._error_state = False

            last_target_ts: float = 0.0
            last_valid_qpos: np.ndarray = current_qpos.copy()

            inner_dt = self._cfg.loop_period
            freq_hz = int(round(1.0 / inner_dt))
            limiter = RateLimiter(float(freq_hz))

            logger.info(
                "ArmInnerLoop: %dHz started (mode 6: online trajectory planning, "
                "speed=%.0f°/s, acc=%.0f°/s², no interp — firmware handles planning)",
                freq_hz,
                float(np.degrees(self._cfg.joint_max_speed)),
                float(np.degrees(self._cfg.joint_max_acc)),
            )
            self._ready_event.set()

            while not self._stop_event.is_set():
                limiter.wait()

                # ── 1. Read target ──
                with self._lock:
                    target = self._arm_target.copy() if self._arm_target is not None else None
                    target_ts = self._target_ts

                now = time.perf_counter()

                # ── 2. Timeout → hold ──
                # Skip timeout during startup: if no target has ever been received
                # (last_target_ts==0), the main thread is still initializing.
                no_target_yet = (last_target_ts == 0.0)
                if not no_target_yet and (target is None or (now - max(target_ts, last_target_ts) > self._cfg.target_timeout_s)):
                    self._hold_position(arm)
                    self._signal_ready_only()
                    continue

                if target is None:
                    continue  # startup: no target yet, silently skip

                last_target_ts = target_ts

                # ── 3. NaN guard — hold last valid position ──
                if not np.all(np.isfinite(target)):
                    try:
                        arm.set_servo_angle(
                            angle=last_valid_qpos.tolist(), is_radian=True,
                            speed=self._cfg.joint_max_speed,
                            mvacc=self._cfg.joint_max_acc,
                            wait=False,
                        )
                    except (RuntimeError, OSError):
                        pass
                    continue

                last_valid_qpos = target[:7].copy()

                # ── 4. Read current joint state ──
                try:
                    code, states = arm.get_joint_states(is_radian=True, num=1)
                except (RuntimeError, OSError) as e:
                    logger.error("ArmInnerLoop: get_joint_states failed: %s", e)
                    with self._lock:
                        self._error_state = True
                    continue

                if code != 0:
                    logger.error("ArmInnerLoop: arm error code=%d — stopping inner loop", code)
                    with self._lock:
                        self._error_state = True
                    break

                # Also check arm error flag (C31 collision sets error_code without
                # necessarily failing get_joint_states)
                arm_error = getattr(arm, 'error_code', 0)
                if arm_error != 0:
                    logger.error("ArmInnerLoop: arm error_code=%d — stopping inner loop", arm_error)
                    with self._lock:
                        self._error_state = True
                    break

                if len(states) > 0:
                    q = np.asarray(states[0], dtype=np.float64)
                    if q.shape[0] >= 7 and np.all(np.isfinite(q[:7])):
                        current_qpos = q[:7].copy()
                        with self._lock:
                            self._arm_qpos = current_qpos
                            self._error_state = False

                # ── 5. Forward target → firmware trajectory planner ──
                self._send_target(arm, target)

                # ── 6. Sync handshake (after hardware write) ──
                self._signal_ready_and_sync()

        except Exception:
            logger.exception("ArmInnerLoop: fatal error in main loop")
            with self._lock:
                self._error_state = True
        finally:
            self._ready_event.clear()
            # Mode 6: firmware holds last position on disconnect, no explicit stop needed.
            try:
                arm.disconnect()
            except Exception:
                pass
            logger.info("ArmInnerLoop: stopped")

    # ── Command dispatch ──

    def _send_target(self, arm, target: np.ndarray) -> None:
        """Forward target position → set_servo_angle(wait=False).

        Joint online trajectory planning (Mode 6). The firmware performs online
        trajectory replanning from the current state when each new command arrives.
        Speed and acceleration limits ARE respected by the firmware trajectory planner.

        No inner-loop interpolation — the target is forwarded directly and the
        firmware handles all trajectory smoothing.
        """
        try:
            code = arm.set_servo_angle(
                angle=target[:7].tolist(), is_radian=True,
                speed=self._cfg.joint_max_speed,
                mvacc=self._cfg.joint_max_acc,
                wait=False,
            )
        except (RuntimeError, OSError) as e:
            logger.error("ArmInnerLoop: set_servo_angle failed: %s", e)
            with self._lock:
                self._error_state = True
            return

        if code != 0:
            err_code = -1
            warn_code = -1
            try:
                ret, err_warn = arm.get_err_warn_code()
                if len(err_warn) >= 2:
                    err_code = int(err_warn[0])
                    warn_code = int(err_warn[1])
            except (RuntimeError, OSError, ValueError, IndexError):
                pass
            logger.error(
                "ArmInnerLoop: set_servo_angle code=%d, controller error=%d, warn=%d",
                code, err_code, warn_code,
            )
            with self._lock:
                self._error_state = True

    # ── Helpers ──

    def _hold_position(self, arm) -> None:
        """Read current position and re-send as hold command via set_servo_angle."""
        try:
            code, states = arm.get_joint_states(is_radian=True, num=1)
            if code == 0 and len(states) > 0:
                hold = np.asarray(states[0], dtype=np.float64)[:7]
                if np.all(np.isfinite(hold)):
                    arm.set_servo_angle(
                        angle=hold.tolist(), is_radian=True,
                        speed=self._cfg.joint_max_speed,
                        mvacc=self._cfg.joint_max_acc,
                        wait=False,
                    )
        except (RuntimeError, OSError):
            pass

    def _init_mode(self, arm) -> None:
        """Transition arm to Mode 6 (joint online trajectory planning).

        Requires firmware >= 1.10.0. Logs a warning if the firmware version
        cannot be parsed or is below the minimum.
        """
        # Check firmware version (Mode 6 requires >= 1.10.0)
        try:
            code, ver_str = arm.get_version()
            if code == 0 and ver_str:
                # Parse "v1.18.4" or "1.18.4" format
                ver_clean = ver_str.lstrip("vV")
                parts = ver_clean.split(".")
                if len(parts) >= 2:
                    major, minor = int(parts[0]), int(parts[1])
                    if major < 1 or (major == 1 and minor < 10):
                        logger.warning(
                            "ArmInnerLoop: firmware %s is below Mode 6 minimum (1.10.0). "
                            "Mode 6 may not work correctly.",
                            ver_str,
                        )
                    else:
                        logger.info("ArmInnerLoop: firmware %s OK (>= 1.10.0 required)", ver_str)
        except Exception:
            logger.warning("ArmInnerLoop: could not check firmware version — assuming >= 1.10.0")

        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.05)
        arm.set_mode(6)
        arm.set_state(0)
        time.sleep(0.05)
        arm.set_state(0)
        # Set firmware-level joint acceleration limit (respected by Mode 6
        # trajectory planner).
        arm.set_joint_maxacc(self._cfg.joint_max_acc, is_radian=True)
        logger.info(
            "ArmInnerLoop: set_joint_maxacc=%s rad/s² (%.0f°/s²)",
            self._cfg.joint_max_acc, round(float(np.degrees(self._cfg.joint_max_acc))),
        )
