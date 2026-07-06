"""ArmInnerLoop — 250Hz inner loop thread running in the same process as the controller.

Supports two control modes (configurable):

  Mode 4 (default): Velocity control — user-space PID converts position error → velocity,
    then sends to arm.vc_set_joint_velocity(). Full control over PID gains,
    jerk/accel limiting, anti-windup. Ref: BunnyVisionPro _internal_control_arm_qpos().

  Mode 1: Position servo — forwards target to arm.set_servo_angle_j().
    Arm firmware handles PID, smoothing, velocity limiting internally (kHz level).
    Simplest fallback option.

Architecture:
    Main Thread (50Hz)                     Inner Loop Thread (250Hz)
    ──────────────────                     ─────────────────────────
    inner.set_target(cmd)    ──Lock──→     self._arm_target
    qpos, err, ts = get_state()  ←──Lock── self._arm_qpos, _error_state
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexmani_real.shm.sync_primitives import SharedSyncPrimitives

import numpy as np

from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_limiter import RateLimiter
from dexmani_real.utils.signal_utils import limit_jerk

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# PID Controller (ref: BunnyVisionPro xarm7_ability.py:11-36)
# ═══════════════════════════════════════════════════════════════════


class PIDController:
    """Per-joint independent PID controller with anti-windup.

    Gains are (7,) arrays — one per joint. Integral term is clamped
    to ``windup_limit`` × max_velocity to prevent unbounded accumulation
    when velocity output is saturated.

    Ref: BunnyVisionPro PIDController (no anti-windup in original; added here).
    """

    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        windup_limit: float = 0.3,
    ) -> None:
        self.kp = np.asarray(kp, dtype=np.float64)
        self.ki = np.asarray(ki, dtype=np.float64) if ki is not None else np.zeros_like(self.kp)
        self.kd = np.asarray(kd, dtype=np.float64) if kd is not None else np.zeros_like(self.kp)
        self._windup_limit = float(windup_limit)
        self._prev_err: np.ndarray | None = None
        self._cum_err: np.ndarray = np.zeros_like(self.kp)

    def reset(self) -> None:
        self._prev_err = None
        self._cum_err = np.zeros_like(self.kp)

    def control(self, err: np.ndarray, dt: float, max_output: np.ndarray | None = None) -> np.ndarray:
        """Compute PID output from position error.

        Args:
            err: (7,) position error (target - current) in radians.
            dt: Time step in seconds.
            max_output: (7,) per-joint max output magnitude (for anti-windup clamping).
                        If None, no clamping is applied.

        Returns:
            (7,) velocity command in rad/s.
        """
        err = np.asarray(err, dtype=np.float64)
        if self._prev_err is None:
            self._prev_err = err.copy()

        # Proportional + derivative (derivative-on-error)
        value = (
            self.kp * err
            + self.kd * (err - self._prev_err) / dt
            + self.ki * self._cum_err
        )

        # Anti-windup: clamp integral when output is near saturation
        self._cum_err += dt * err
        if max_output is not None:
            windup_bound = self._windup_limit * max_output
            self._cum_err = np.clip(self._cum_err, -windup_bound, windup_bound)

        self._prev_err = err.copy()
        return value


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ArmInnerLoopConfig:
    """Configuration for ArmInnerLoop.

    Attributes:
        control_mode: 1 = position servo (set_servo_angle_j), 4 = velocity control
                      (vc_set_joint_velocity + user-space PID).
        kp, ki, kd: Symmetric PID gains (7,) per-joint. All joints share the same
                   kp/kd for coordinated Cartesian tracking. Only used in mode 4.
        max_velocity: Per-joint max velocity (rad/s). Only used in mode 4 for
                      output clipping + anti-windup bound.
        max_accel: Per-joint max acceleration (rad/s²). If set, velocity output
                   is additionally clamped by accel-limited rate-of-change.
                   Only used in mode 4.
        max_jerk: Per-joint max jerk (rad/s³). If set, acceleration rate-of-change
                  is limited via proportional scaling (ref: limit_jerk in signal_utils).
                  Only used in mode 4. Default: None (disabled).
        smooth_position_alpha: If > 0, applies EMA smoothing to the target position
                               before sending (mode 1 only; mode 4 already has PID +
                               vel/accel/jerk limiting). Reduces raw-target jitter at
                               the cost of ~1 frame latency. Default: 0.0 (disabled).
        target_timeout_s: Max age of target before auto-stop (0.2s).
    """

    control_mode: int = 4
    kp: np.ndarray = field(default_factory=lambda: np.full(7, 10.0))
    ki: np.ndarray = field(default_factory=lambda: np.zeros(7))
    kd: np.ndarray = field(default_factory=lambda: np.full(7, 0.04))
    max_velocity: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.0, 1.2, 1.0, 1.5, 1.0, 1.5]))
    max_accel: np.ndarray | None = None
    max_jerk: np.ndarray | None = None
    smooth_position_alpha: float = 0.0
    target_timeout_s: float = 0.2

    # Two-phase handshake: when True, ArmInnerLoop sets robot_ready after each
    # hardware write and waits for policy_ready before the next target dispatch.
    synchronized: bool = False

    def __post_init__(self):
        self.kp = np.asarray(self.kp, dtype=np.float64).ravel()[:7]
        self.ki = np.asarray(self.ki, dtype=np.float64).ravel()[:7]
        self.kd = np.asarray(self.kd, dtype=np.float64).ravel()[:7]
        self.max_velocity = np.asarray(self.max_velocity, dtype=np.float64).ravel()[:7]
        if self.max_accel is not None:
            self.max_accel = np.asarray(self.max_accel, dtype=np.float64).ravel()[:7]
        if self.max_jerk is not None:
            self.max_jerk = np.asarray(self.max_jerk, dtype=np.float64).ravel()[:7]
        self.smooth_position_alpha = float(np.clip(self.smooth_position_alpha, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════
# ArmInnerLoop
# ═══════════════════════════════════════════════════════════════════


class ArmInnerLoop:
    """250Hz inner loop thread — owns the XArmAPI connection.

    Runs in the same process as the controller. Communicates via
    Lock-protected shared variables (no SHM/IPC overhead).

    Parameters:
        ip: XArm controller IP address.
        dt: Inner loop period in seconds (default 1/250 = 4ms).
        cfg: Inner loop configuration (control mode, PID gains, velocity limits).
    """

    def __init__(
        self,
        ip: str = "192.168.1.111",
        dt: float = 1.0 / 250.0,
        cfg: ArmInnerLoopConfig | None = None,
        sync: SharedSyncPrimitives | None = None,
    ) -> None:
        self._ip = ip
        self._dt = float(dt)
        self._cfg = cfg or ArmInnerLoopConfig()
        self._sync = sync

        # ── Shared state (protected by _lock) ──
        self._lock = threading.Lock()
        self._arm_target: np.ndarray | None = None
        self._target_ts: float = 0.0
        self._arm_qpos: np.ndarray = np.zeros(7, dtype=np.float64)
        self._error_state: bool = False

        # ── Mode 4 PID controller (created on start) ──
        self._pid: PIDController | None = None

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

    @property
    def control_mode(self) -> int:
        return self._cfg.control_mode

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

        mode = self._cfg.control_mode
        try:
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            self._init_mode(arm, mode)
            arm.set_collision_sensitivity(1)

            # Verify arm entered the requested mode
            actual_mode = getattr(arm, 'mode', -1)
            if actual_mode != mode:
                logger.warning(
                    "ArmInnerLoop: arm mode=%d but expected %d — re-initializing",
                    actual_mode, mode,
                )
                self._init_mode(arm, mode)
                actual_mode = getattr(arm, 'mode', -1)
                if actual_mode != mode:
                    logger.error("ArmInnerLoop: failed to set mode %d (arm mode=%d)", mode, actual_mode)
                    with self._lock:
                        self._error_state = True
                    return

            # Init PID for mode 4
            if mode == 4:
                self._pid = PIDController(
                    kp=self._cfg.kp, ki=self._cfg.ki, kd=self._cfg.kd,
                )

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
            limiter = RateLimiter(1.0 / self._dt)

            # For mode 4: track previous velocity + acceleration for limiting
            prev_qvel: np.ndarray | None = np.zeros(7, dtype=np.float64) if mode == 4 else None
            prev_qacc: np.ndarray | None = np.zeros(7, dtype=np.float64) if (mode == 4 and self._cfg.max_jerk is not None) else None
            # For mode 1 position EMA smoothing (mode 4 has PID + vel/accel/jerk limiting)
            self._mode1_smoothed: np.ndarray | None = None

            # Build mode label with smoothing details
            parts = [{1: "position servo", 4: "velocity control + PID"}.get(mode, f"mode {mode}")]
            if mode == 1 and self._cfg.smooth_position_alpha > 0:
                parts.append(f"pos EMA α={self._cfg.smooth_position_alpha}")
            if mode == 4:
                if self._cfg.max_accel is not None:
                    parts.append("accel limit")
                if self._cfg.max_jerk is not None:
                    parts.append("jerk limit")
            mode_label = ", ".join(parts)
            logger.info("ArmInnerLoop: 250Hz started (mode %d: %s)", mode, mode_label)
            self._ready_event.set()

            while not self._stop_event.is_set():
                limiter.wait()

                # ── 1. Read target ──
                with self._lock:
                    target = self._arm_target.copy() if self._arm_target is not None else None
                    target_ts = self._target_ts

                now = time.perf_counter()

                # ── 2. Timeout → stop (both modes) ──
                # Skip timeout during startup: if no target has ever been received
                # (last_target_ts==0), the main thread is still initializing.
                # Sending zero velocity every 4ms here would spam the SDK with
                # code=1 errors before the first real target arrives.
                no_target_yet = (last_target_ts == 0.0)
                if not no_target_yet and (target is None or (now - max(target_ts, last_target_ts) > self._cfg.target_timeout_s)):
                    if mode == 4:
                        self._send_zero_velocity(arm)
                    else:
                        self._hold_position(arm)
                    # Reset mode-1 smoothing state on timeout
                    self._mode1_smoothed = None
                    self._signal_ready_only()  # avoid deadlock: no new target expected
                    continue

                if target is None:
                    continue  # startup: no target yet, silently skip

                last_target_ts = target_ts

                # ── 3. NaN guard ──
                if not np.all(np.isfinite(target)):
                    if mode == 4:
                        self._send_zero_velocity(arm)
                    else:
                        try:
                            arm.set_servo_angle_j(angles=last_valid_qpos.tolist(), is_radian=True)
                        except (RuntimeError, OSError):
                            pass
                    self._mode1_smoothed = None
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

                # ── 5. Send command ──
                if mode == 4:
                    self._tick_mode4(arm, target, current_qpos, prev_qvel, prev_qacc)
                else:
                    self._tick_mode1(arm, target)

                # ── 6. Sync handshake (after hardware write) ──
                self._signal_ready_and_sync()

        except Exception:
            logger.exception("ArmInnerLoop: fatal error in main loop")
            with self._lock:
                self._error_state = True
        finally:
            self._ready_event.clear()
            # Send zero velocity before disconnecting (mode 4 safety).
            # Skip if arm is in error state — vc_set_joint_velocity will be rejected
            # anyway, and repeatedly calling it floods the SDK log.
            try:
                if mode == 4 and getattr(arm, 'error_code', 0) == 0:
                    arm.vc_set_joint_velocity(np.zeros(7, dtype=np.float64).tolist(), is_radian=True)
            except Exception:
                pass
            try:
                arm.disconnect()
            except Exception:
                pass
            logger.info("ArmInnerLoop: stopped")

    # ── Mode 1: Position Servo ──

    def _tick_mode1(self, arm, target: np.ndarray) -> None:
        """Mode 1: forward target position → set_servo_angle_j().

        Position EMA smoothing is applied here (not in the common path) because
        mode 4 already has PID + velocity/accel/jerk limiting for implicit
        smoothing.  Mode 1 has no such downstream filtering, so the optional EMA
        compensates for that.
        """
        # ── Position EMA smoothing (mode 1 only) ──
        if self._cfg.smooth_position_alpha > 0:
            if self._mode1_smoothed is None:
                self._mode1_smoothed = target[:7].copy()
            else:
                alpha = self._cfg.smooth_position_alpha
                self._mode1_smoothed = alpha * target[:7] + (1.0 - alpha) * self._mode1_smoothed
            target = self._mode1_smoothed

        try:
            code = arm.set_servo_angle_j(angles=target[:7].tolist(), is_radian=True)
        except (RuntimeError, OSError) as e:
            logger.error("ArmInnerLoop: set_servo_angle_j failed: %s", e)
            with self._lock:
                self._error_state = True
            return

        if code != 0:
            logger.error("ArmInnerLoop: set_servo_angle_j code=%d", code)
            with self._lock:
                self._error_state = True

    # ── Mode 4: Velocity Control + User-Space PID ──

    def _tick_mode4(
        self, arm, target: np.ndarray, current_qpos: np.ndarray, prev_qvel: np.ndarray | None, prev_qacc: np.ndarray | None = None
    ) -> None:
        """Mode 4: PID(position error) → velocity → vc_set_joint_velocity().

        Multi-stage output limiting (in order):
          1. PID: position error → raw velocity
          2. Velocity clipping (per-joint max)
          3. Acceleration limiting (rate-of-change, optional)
          4. Jerk limiting (rate-of-change of accel, optional, ref: limit_jerk)

        Ref: BunnyVisionPro _internal_control_arm_qpos() lines 223-228.
             LeFranX Ruckig jerk-limited OTG concept.
        """
        # 1. PID: position error → velocity
        error = target - current_qpos
        qvel = self._pid.control(error, self._dt, max_output=self._cfg.max_velocity)

        # 2. Clip to max velocity
        qvel = self._clip_velocity(qvel, self._cfg.max_velocity)

        # 3. Optional accel limiting
        if self._cfg.max_accel is not None and prev_qvel is not None:
            max_delta = self._cfg.max_accel * self._dt
            delta = qvel - prev_qvel
            overshot = np.abs(delta) / max_delta
            max_overshot = np.max(overshot)
            if max_overshot > 1.0:
                qvel = prev_qvel + delta / max_overshot

        # 4. Optional jerk limiting (ref: signal_utils.limit_jerk)
        if self._cfg.max_jerk is not None and prev_qvel is not None:
            # Use the non-None prev_qacc tracked by the caller
            qvel, updated_acc = limit_jerk(
                qvel, prev_qvel, prev_qacc, self._dt,
                max_jerk=float(np.min(self._cfg.max_jerk)),
            )
            if prev_qacc is not None:
                prev_qacc[:] = updated_acc

        # Track previous velocity (after all limiting)
        if prev_qvel is not None:
            prev_qvel[:] = qvel

        # Send velocity command
        try:
            code = arm.vc_set_joint_velocity(qvel.tolist(), is_radian=True)
        except (RuntimeError, OSError) as e:
            logger.error("ArmInnerLoop: vc_set_joint_velocity failed: %s", e)
            with self._lock:
                self._error_state = True
            return

        if code != 0:
            # code=1 on all-zero speeds is benign (arm already stopped, SDK rejects no-op).
            # code=1 on non-zero speeds suggests a mode mismatch or transient error.
            is_zero_vel = bool(np.all(np.abs(qvel) < 1e-9))
            if code == 1 and is_zero_vel:
                logger.debug("ArmInnerLoop: vc_set_joint_velocity code=1 (zero vel, benign)")
            elif code == 1:
                logger.warning("ArmInnerLoop: vc_set_joint_velocity code=1 (non-zero vel, mode issue?)")
            else:
                logger.error("ArmInnerLoop: vc_set_joint_velocity code=%d", code)
                with self._lock:
                    self._error_state = True

    # ── Helpers ──

    def _hold_position(self, arm) -> None:
        """Read current position and re-send as hold command (mode 1)."""
        try:
            code, states = arm.get_joint_states(is_radian=True, num=1)
            if code == 0 and len(states) > 0:
                hold = np.asarray(states[0], dtype=np.float64)[:7]
                if np.all(np.isfinite(hold)):
                    arm.set_servo_angle_j(angles=hold.tolist(), is_radian=True)
        except (RuntimeError, OSError):
            pass

    @staticmethod
    def _send_zero_velocity(arm) -> None:
        """Send zero velocity to stop arm (mode 4)."""
        try:
            arm.vc_set_joint_velocity(np.zeros(7, dtype=np.float64).tolist(), is_radian=True)
        except (RuntimeError, OSError):
            pass

    @staticmethod
    def _clip_velocity(qvel: np.ndarray, max_vel: np.ndarray) -> np.ndarray:
        """Clip per-joint velocity to max limits, preserving direction.

        Logs the bottleneck joint when clipping occurs (ref: BunnyVisionPro
        xarm7_ability.py:185-194 — prints which joint limited the velocity).
        """
        overshot = np.abs(qvel) / max_vel
        max_overshot = np.max(overshot)
        if max_overshot > 1.0 + 1e-4:
            bottleneck = int(np.argmax(overshot))
            logger.debug(
                "Vel clip: joint-%d overshot %.2fx (%.3f / %.3f rad/s)",
                bottleneck + 1,
                max_overshot,
                float(qvel[bottleneck]),
                float(max_vel[bottleneck]),
            )
            return qvel / max_overshot
        return qvel

    @staticmethod
    def _init_mode(arm, mode: int) -> None:
        """Transition arm to target control mode via idle intermediate state."""
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.05)
        arm.set_mode(mode)
        arm.set_state(0)
        time.sleep(0.05)
        arm.set_state(0)
