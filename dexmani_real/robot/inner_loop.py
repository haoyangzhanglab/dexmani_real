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

from dexmani_real.robot.xarm7.error_codes import decode_error, decode_warn
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

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
        loop_period: Inner loop period in seconds. Default 0.04 (25Hz).
        target_timeout_s: Max age of target before auto-hold (0.2s).
        max_joint_delta: Per-step L∞ joint delta clamp (rad). Default 0.3 rad per
                         inner-loop step (~17°, ~15 rad/s ceiling at 50Hz). Mirrors
                         XHand E3. Set 0 to disable. Headroom over a normal target
                         step depends on the OUTER loop rate (see max_joint_delta
                         comment below).
        speed_ramp_frames: Number of frames for soft-start speed ramp. Default 20
                           frames (0.4s at 50Hz). Speed linearly ramps from
                           speed_ramp_min → joint_max_speed. Set 0 to disable.
        speed_ramp_min: Initial speed during soft-start ramp (rad/s). Default 0.2
                        rad/s (~11°/s). Avoids abrupt motion on teleop engagement.
        synchronized: Two-phase handshake for policy inference (default False).
    """

    # Mode 6 parameters (speed/accel ARE respected by firmware trajectory planner)
    joint_max_speed: float = 1.5708  # 90°/s in rad/s
    joint_max_acc: float = 8.7266  # 500°/s² in rad/s²
    loop_period: float = 0.04  # 25Hz (was 50Hz) — halves GIL contention from SDK socket I/O; Mode 6 firmware handles interpolation between commands
    # Shared
    target_timeout_s: float = 0.2

    # Per-step delta clamp — safety ceiling against IK solver anomalies.
    # Mirrors XHand E3 (XHandConfig.max_delta_rad).  A normal target step is
    # joint_max_speed / outer_loop_hz: ≈0.03 rad @50Hz outer (~10x headroom),
    # ≈0.098 rad @16Hz outer (~3x headroom).  Note the inner loop re-sends the
    # same target every 20ms, so a single anomalous target is chased at up to
    # 0.3 rad per inner step until it times out (target_timeout_s).
    max_joint_delta: float = 0.3

    # Soft-start speed ramp — prevents abrupt motion on teleop engagement.
    # Speed ramps linearly from speed_ramp_min → joint_max_speed over the first
    # speed_ramp_frames.  Set speed_ramp_frames=0 to disable.
    speed_ramp_frames: int = 20
    speed_ramp_min: float = 0.2  # ~11°/s

    # Passive tracking-error monitor — warns when |target - current| exceeds this
    # on any joint (soft saturation / follow error that arm error codes miss).
    # Does NOT trigger an error state or alter commands.  Set 0 to disable.
    tracking_error_warn_rad: float = 0.35

    # Two-phase handshake: when True, ArmInnerLoop sets robot_ready after each
    # hardware write and waits for policy_ready before the next target dispatch.
    synchronized: bool = False


# Controller errors that indicate a problematic target rather than a hardware fault.
# When the firmware rejects a target with one of these codes, we clear the latch
# immediately so the inner loop can continue holding position, and the outer loop's
# validate_action() does not see a stale error and trigger an unnecessary emergency
# stop.  Errors NOT in this set are treated as hard faults that stop the inner loop.
_RECOVERABLE_ERRORS: frozenset[int] = frozenset({
    22,  # Self-Collision Error — IK solver produced unsafe joint angles
    24,  # Speed Exceeds Limit — commanded motion too fast
})


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
        # Dynamics from the 50Hz readback (NaN until first valid read) — consumed
        # by the outer loop for recording + torque/temperature pre-send gates.
        self._arm_qvel: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        self._arm_tau: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        self._arm_temps: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        self._error_state: bool = False
        self._last_sent_target: np.ndarray | None = None  # for per-step delta clamp
        self._ramp_step: int = 0  # for soft-start speed ramp
        self._qvel_inf: float = 0.0  # last |qvel| L∞ — inner-loop thread only
        self._tracking_error: float = 0.0  # last |target-current| L∞ (passive monitor)
        self._track_warn_throttle: int = 0  # throttle counter for tracking-error warnings
        self._mode_warn_throttle: int = 0  # throttle counter for mode-drift warnings (independent)

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

    def get_dynamics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Latest (qvel, tau, temps) from the 50Hz readback — NaN until first valid read.

        The inner loop owns the sole XArmAPI connection, so the outer loop must
        source dynamics here instead of calling the SDK (avoids connection contention).
        """
        with self._lock:
            return self._arm_qvel.copy(), self._arm_tau.copy(), self._arm_temps.copy()

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def tracking_error(self) -> float:
        """Last |target - current| L∞ joint error (rad). Passive monitor."""
        with self._lock:
            return self._tracking_error

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
        # Unblock a loop thread stuck in the sync handshake (policy_ready.wait()
        # has no timeout) — it wakes, sees _stop_event, and exits cleanly.
        if self._sync is not None:
            self._sync.policy_ready.set()
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
            actual_mode = getattr(arm, "mode", -1)
            if actual_mode != 6:
                logger.warning(
                    "ArmInnerLoop: arm mode=%d but expected 6 — re-initializing",
                    actual_mode,
                )
                self._init_mode(arm)
                actual_mode = getattr(arm, "mode", -1)
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
            limiter = RateManager(float(freq_hz))

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
                no_target_yet = last_target_ts == 0.0
                if not no_target_yet and (
                    target is None or (now - max(target_ts, last_target_ts) > self._cfg.target_timeout_s)
                ):
                    # Re-arm the soft-start ramp: any hold (PAUSED, idle, VR loss,
                    # main-thread stall) counts as disengagement, so the next
                    # engagement ramps up again as the docstring promises.
                    self._ramp_step = 0
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
                            angle=last_valid_qpos.tolist(),
                            is_radian=True,
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
                    code, states = arm.get_joint_states(is_radian=True, num=3)
                except (RuntimeError, OSError) as e:
                    logger.error("ArmInnerLoop: get_joint_states failed: %s", e)
                    with self._lock:
                        self._error_state = True
                    continue
                time.sleep(0)  # explicit GIL yield — unblocks main-thread numpy ops

                if code != 0:
                    logger.error("ArmInnerLoop: arm error code=%d — stopping inner loop", code)
                    with self._lock:
                        self._error_state = True
                    break

                # Also check arm error flag (C31 collision sets error_code without
                # necessarily failing get_joint_states)
                arm_error = getattr(arm, "error_code", 0)
                if arm_error != 0:
                    if arm_error in _RECOVERABLE_ERRORS:
                        # Recoverable — clear the latch, re-init Mode 6 (firmware
                        # drops to Mode 0 on reject), then hold position and
                        # re-arm the soft-start ramp for a clean re-engagement.
                        logger.warning(
                            "ArmInnerLoop: arm error_code=%d (%s) — recoverable, re-initialising mode",
                            arm_error,
                            decode_error(arm_error),
                        )
                        with self._lock:
                            self._arm_target = None
                        self._recover_mode(arm)
                        self._ramp_step = 0  # re-arm soft-start ramp
                        self._hold_position(arm)
                        self._signal_ready_only()
                        continue
                    logger.error(
                        "ArmInnerLoop: arm error_code=%d (%s) — stopping inner loop",
                        arm_error,
                        decode_error(arm_error),
                    )
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
                if len(states) > 1:
                    v = np.asarray(states[1], dtype=np.float64)
                    if v.shape[0] >= 7 and np.all(np.isfinite(v[:7])):
                        self._qvel_inf = float(np.max(np.abs(v[:7])))
                        with self._lock:
                            self._arm_qvel = v[:7].copy()
                if len(states) > 2:
                    tau = np.asarray(states[2], dtype=np.float64)
                    if tau.shape[0] >= 7 and np.all(np.isfinite(tau[:7])):
                        with self._lock:
                            self._arm_tau = tau[:7].copy()
                temps = getattr(arm, "temperatures", None)
                if temps is not None:
                    t = np.asarray(temps, dtype=np.float64).ravel()
                    if t.shape[0] >= 7 and np.all(np.isfinite(t[:7])):
                        with self._lock:
                            self._arm_temps = t[:7].copy()

                # ── 4b. Passive tracking-error + mode monitor (no command change) ──
                self._monitor(arm, current_qpos)

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
            # Wake a controller blocked on robot_ready.wait() so it can observe
            # the error/exit state instead of hanging (error exits never reach
            # the normal per-frame handshake).
            self._signal_ready_only()
            # Mode 6: firmware holds last position on disconnect, no explicit stop needed.
            try:
                arm.disconnect()
            except Exception:
                pass
            logger.info("ArmInnerLoop: stopped")

    # ── Command dispatch ──

    def _monitor(self, arm, current_qpos: np.ndarray) -> None:
        """Passive health monitor — tracking error (A4) + mode drift (A5).

        Computes |last_sent_target - current| L∞ and, together with a per-frame
        mode==6 recheck, emits a throttled warning.  Never mutates commands or
        the error state — soft saturation is not a hard fault.
        """
        cfg = self._cfg
        err = 0.0
        if self._last_sent_target is not None:
            err = float(np.max(np.abs(self._last_sent_target - current_qpos[:7])))
        with self._lock:
            self._tracking_error = err

        mode_bad = getattr(arm, "mode", 6) != 6
        track_bad = cfg.tracking_error_warn_rad > 0 and err > cfg.tracking_error_warn_rad

        # Independent throttles: a persistent tracking-error warning must not
        # suppress a transient mode-drift warning (A5), and vice versa.
        if self._track_warn_throttle > 0:
            self._track_warn_throttle -= 1
        elif track_bad:
            logger.warning(
                "ArmInnerLoop: tracking error %.3f rad exceeds %.3f (soft saturation / follow error)",
                err,
                cfg.tracking_error_warn_rad,
            )
            self._track_warn_throttle = 50  # ~1s at 50Hz

        if self._mode_warn_throttle > 0:
            self._mode_warn_throttle -= 1
        elif mode_bad:
            logger.warning(
                "ArmInnerLoop: arm mode=%s (expected 6) — trajectory planning may be degraded",
                getattr(arm, "mode", "?"),
            )
            self._mode_warn_throttle = 50

    def _send_target(self, arm, target: np.ndarray) -> None:
        """Forward target position → set_servo_angle(wait=False).

        Joint online trajectory planning (Mode 6). The firmware performs online
        trajectory replanning from the current state when each new command arrives.
        Speed and acceleration limits ARE respected by the firmware trajectory planner.

        No inner-loop interpolation — the target is forwarded directly and the
        firmware handles all trajectory smoothing.
        """
        # ── Per-step joint delta clamp (mirrors XHand E3) ──
        # Safety ceiling against IK solver anomalies.  A normal target step is
        # joint_max_speed ÷ outer_loop_hz (≈0.03 rad @50Hz outer, ≈0.098 rad
        # @16Hz outer); the 0.3 rad default gives ~10x / ~3x headroom respectively.
        clamped = target[:7].copy()
        if self._cfg.max_joint_delta > 0 and self._last_sent_target is not None:
            delta = clamped - self._last_sent_target
            delta = np.clip(delta, -self._cfg.max_joint_delta, self._cfg.max_joint_delta)
            clamped = self._last_sent_target + delta

        # ── Soft-start speed ramp ──
        # Linearly ramp speed_ramp_min → joint_max_speed over first N frames.
        # Prevents abrupt motion when teleop engages (e.g. after idle/home).
        if self._cfg.speed_ramp_frames > 0 and self._ramp_step < self._cfg.speed_ramp_frames:
            t = self._ramp_step / self._cfg.speed_ramp_frames
            speed = self._cfg.speed_ramp_min + (self._cfg.joint_max_speed - self._cfg.speed_ramp_min) * t
            # Ramp re-armed mid-motion (main-thread stall → hold → fresh target) must
            # never command a speed cap below the arm's actual joint speed — Mode 6
            # firmware rejects that with C24 (Speed Exceeds Limit).
            speed = min(self._cfg.joint_max_speed, max(speed, 1.25 * self._qvel_inf))
        else:
            speed = self._cfg.joint_max_speed

        try:
            code = arm.set_servo_angle(
                angle=clamped.tolist(),
                is_radian=True,
                speed=speed,
                mvacc=self._cfg.joint_max_acc,
                wait=False,
            )
            time.sleep(0)  # explicit GIL yield — unblocks main-thread numpy ops
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

            if err_code in _RECOVERABLE_ERRORS:
                # Target rejected by firmware (e.g. self-collision, overspeed).
                # This is NOT a hardware fault — clear the latch, re-init Mode 6
                # (firmware drops to Mode 0 on reject), and wait for the next
                # valid command from the outer loop.  Do NOT set error_state=True,
                # do NOT update _last_sent_target (delta clamp keeps using the
                # last good target).
                logger.warning(
                    "ArmInnerLoop: set_servo_angle code=%d, controller error=%d (%s) — "
                    "target skipped, re-initialising mode",
                    code,
                    err_code,
                    decode_error(err_code),
                )
                with self._lock:
                    self._arm_target = None
                self._recover_mode(arm)
                self._ramp_step = 0  # re-arm soft-start ramp
                return
            else:
                logger.error(
                    "ArmInnerLoop: set_servo_angle code=%d, controller error=%d (%s), warn=%d (%s)",
                    code,
                    err_code,
                    decode_error(err_code),
                    warn_code,
                    decode_warn(warn_code),
                )
                with self._lock:
                    self._error_state = True
        else:
            self._last_sent_target = clamped.copy()
            self._ramp_step += 1

    # ── Helpers ──

    def _recover_mode(self, arm) -> bool:
        """Clear error latch and re-init Mode 6 after a recoverable error.

        After firmware rejects a command (e.g. error 22 self-collision), the arm
        drops from Mode 6 to Mode 0.  ``clean_error`` + ``set_state(0)`` clears
        the latch but does NOT restore the control mode — without Mode 6,
        subsequent ``set_servo_angle`` calls use the wrong protocol and the arm
        is effectively dead (observed as ``set_mode(1)`` returning code 10
        repeatedly during return_home).

        Returns ``True`` on success, ``False`` if all re-init attempts fail.
        """
        arm.clean_error()
        arm.clean_warn()
        arm.set_state(0)

        # Re-init Mode 6 with retries (firmware may reject the transition
        # while internal error state is still settling).
        for attempt in range(3):
            try:
                arm.set_mode(0)
                arm.set_state(0)
                time.sleep(0.05)
                arm.set_mode(6)
                arm.set_state(0)
                time.sleep(0.05)
                arm.set_state(0)
                actual_mode = getattr(arm, "mode", -1)
                if actual_mode == 6:
                    logger.info("ArmInnerLoop: Mode 6 re-initialised (attempt %d)", attempt + 1)
                    return True
                logger.warning(
                    "ArmInnerLoop: set_mode(6) returned but mode=%d (attempt %d/3)",
                    actual_mode, attempt + 1,
                )
            except (RuntimeError, OSError) as e:
                logger.warning("ArmInnerLoop: Mode 6 re-init attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.1)

        logger.error("ArmInnerLoop: failed to re-init Mode 6 after 3 attempts — arm may be in degraded mode")
        return False

    def _hold_position(self, arm) -> None:
        """Read current position and re-send as hold command via set_servo_angle.

        Updates ``_last_sent_target`` so the per-step delta clamp and tracking
        error monitor use a meaningful baseline when targets resume after a hold.
        """
        try:
            code, states = arm.get_joint_states(is_radian=True, num=1)
            if code == 0 and len(states) > 0:
                hold = np.asarray(states[0], dtype=np.float64)[:7]
                if np.all(np.isfinite(hold)):
                    arm.set_servo_angle(
                        angle=hold.tolist(),
                        is_radian=True,
                        speed=self._cfg.joint_max_speed,
                        mvacc=self._cfg.joint_max_acc,
                        wait=False,
                    )
                    self._last_sent_target = hold.copy()
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
            self._cfg.joint_max_acc,
            round(float(np.degrees(self._cfg.joint_max_acc))),
        )
