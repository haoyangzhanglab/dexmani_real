"""ArmInnerLoop — in-process inner loop thread (30Hz) for xArm7 control.

Default and only control mode is Mode 6 (joint online trajectory planning). The firmware
performs online trajectory replanning with configurable speed and acceleration limits that
ARE respected. No inner-loop interpolation is needed — the target is forwarded directly and
the firmware handles all trajectory smoothing. Effectively dissolves the inner/outer loop
distinction.

Mode 6: Joint online trajectory planning — forwards targets directly to
  arm.set_servo_angle(wait=False) at 30Hz. Firmware respects speed/accel limits
  (default: 120°/s, 500°/s²). Smooth motion, no desk vibration. Requires firmware >= 1.10.0.

Architecture:
    Main Thread (16Hz)                     Inner Loop Thread (30Hz)
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

# Additive noise floor for adaptive tracking-error threshold (rad).
# Unmodeled sources (encoder quantization, L∞ max across 7 joints, Mode 6
# smoothing trade-off) do not scale with joint speed — they are constant offsets.
# This bias is added to the physics-based model (steady-state lag + reversal
# distance) so the threshold doesn't collapse to an over-sensitive constant at
# low speeds where the physics term alone is too small.
_TRACKING_NOISE_RAD: float = 0.07


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ArmInnerLoopConfig:
    """Configuration for ArmInnerLoop (Mode 6: joint online trajectory planning).

    Attributes:
        joint_max_speed: Max joint speed (rad/s). Respected by firmware trajectory
                         planner. Default 120°/s (≈2.09 rad/s).
        joint_max_acc: Max joint acceleration (rad/s²). Respected by firmware
                       trajectory planner. Default 500°/s² (≈8.73 rad/s²).
        loop_period: Inner loop period in seconds. Default 1/30 (30Hz).
        target_timeout_s: Max age of target before auto-hold (0.2s).
        max_joint_delta: Per-step L∞ joint delta clamp (rad). Default 0.3 rad per
                         inner-loop step (~17°, ~9 rad/s ceiling at 30Hz). Mirrors
                         XHand E3. Set 0 to disable. Headroom over a normal target
                         step depends on the OUTER loop rate (see max_joint_delta
                         comment below).
        synchronized: Two-phase handshake for policy inference (default False).
    """

    # Mode 6 parameters (speed/accel ARE respected by firmware trajectory planner)
    joint_max_speed: float = 2.0944  # 120°/s in rad/s
    joint_max_acc: float = 15.708  # 900°/s² in rad/s²
    loop_period: float = (
        1.0 / 30.0  # 30Hz — Mode 6 firmware handles interpolation
    )
    # Shared
    target_timeout_s: float = 0.2

    # Per-step delta clamp — safety ceiling against IK solver anomalies.
    # Mirrors XHand E3 (XHandConfig.max_delta_rad).  A normal target step is
    # joint_max_speed / outer_loop_hz: ≈0.131 rad @16Hz outer (~2.3x headroom).
    # Note the inner loop re-sends the
    # same target every 20ms, so a single anomalous target is chased at up to
    # 0.3 rad per inner step until it times out (target_timeout_s).
    max_joint_delta: float = 0.3

    # Absolute joint limit clip — hardware safety bounds applied before the
    # per-step delta clamp (mirrors xArm7 physical limits in radians).
    # J1=±360°, J2=-118°/+120°, J3=±360°, J4=-11°/+225°, J5=±360°,
    # J6=-97°/+180°, J7=±360°.
    joint_limit_lower: tuple[float, ...] = (-6.283, -2.059, -6.283, -0.192, -6.283, -1.693, -6.283)
    joint_limit_upper: tuple[float, ...] = (6.283, 2.094, 6.283, 3.927, 6.283, 3.142, 6.283)

    # Passive tracking-error monitor — enables velocity-adaptive thresholding
    # when > 0 (recommended: 0.35).  The actual warn threshold scales with
    # commanded joint speed: tighter at rest (~0.15 rad), relaxed at max speed
    # (~0.38 rad @120°/s).  Anomalous (>2× adaptive or > anomaly_cap) always warns.
    # Set 0 to disable.  Does NOT alter commands or trigger error state.
    tracking_error_warn_rad: float = 0.35

    # Absolute ceiling for anomalous tracking error (rad).  Hard cap that flags
    # errors as anomalous regardless of the 2×-adaptive multiplier.  Calibrated
    # for joint_max_acc=500°/s²; retune when changing joint_max_acc.  Formula:
    # anomaly_cap should be k * adaptive_max where k ∈ [1.0, 1.5].
    tracking_error_anomaly_cap_rad: float = 0.50

    # Upper clamp for the velocity-adaptive threshold (rad).  Prevents the
    # physics formula from producing thresholds that would make the anomalous
    # cap trivially unreachable at high speed.
    tracking_error_adaptive_max_rad: float = 0.60

    # Two-phase handshake: when True, ArmInnerLoop sets robot_ready after each
    # hardware write and waits for policy_ready before the next target dispatch.
    synchronized: bool = False


# Controller errors that indicate a problematic target rather than a hardware fault.
# When the firmware rejects a target with one of these codes, we clear the latch
# immediately so the inner loop can continue holding position, and the outer loop's
# validate_action() does not see a stale error and trigger an unnecessary emergency
# stop.  Errors NOT in this set are treated as hard faults that stop the inner loop.
_RECOVERABLE_ERRORS: frozenset[int] = frozenset(
    {
        22,  # Self-Collision Error — IK solver produced unsafe joint angles
        24,  # Speed Exceeds Limit — commanded motion too fast
    }
)


# ═══════════════════════════════════════════════════════════════════
# ArmInnerLoop
# ═══════════════════════════════════════════════════════════════════


class ArmInnerLoop:
    """30Hz inner loop thread — owns the XArmAPI connection, runs Mode 6.

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
        cfg: ArmInnerLoopConfig | None = None,
        sync: SharedSyncPrimitives | None = None,
    ) -> None:
        self._ip = ip
        self._cfg = cfg or ArmInnerLoopConfig()
        self._sync = sync

        # ── Shared state (protected by _lock) ──
        self._lock = threading.Lock()
        self._arm_target: np.ndarray | None = None
        self._target_ts: float = 0.0
        self._arm_qpos: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        # Dynamics from the 30Hz readback (NaN until first valid read) — consumed
        # by the outer loop for recording + torque pre-send gates.
        self._arm_qvel: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        self._arm_tau: np.ndarray = np.full(7, np.nan, dtype=np.float64)
        self._error_state: bool = False
        self._last_sent_target: np.ndarray | None = None  # for per-step delta clamp
        # The delta-clamped value actually forwarded to the SDK each tick
        # (hold position during holds) — the inner-loop "sent" stream (plan §4.9).
        self._last_sent_cmd: np.ndarray = np.zeros(7, dtype=np.float64)
        self._arm = None  # live XArmAPI handle of the loop thread (mode/connected queries)
        self._tracking_error: float = 0.0  # last |target-current| L∞ (passive monitor)
        self._track_warn_throttle: int = 0  # throttle counter for anomalous tracking-error warnings
        self._track_info_throttle: int = 0  # throttle counter for elevated tracking-error info logs
        self._mode_warn_throttle: int = 0  # throttle counter for mode-drift warnings (independent)
        # Adaptive tracking-error threshold: peak-hold envelope of commanded joint
        # velocity (rad/s) — instant attack, exponential decay for noise rejection.
        self._cmd_vel_env: float = 0.0
        self._prev_monitor_target: np.ndarray | None = None

        # ── Lifecycle ──
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        # Emergency-stop request flag (plan §4.8 fast path): honored by the
        # loop thread at the top of each tick — set_state(4) on its OWN live
        # connection (no reconnect, ≤1 tick) — and by exec_macro's finally to
        # suppress the Mode 6 reconstruction after an emergency stop. Nothing
        # in the pre-existing control paths ever sets it.
        self._emergency_event = threading.Event()

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
        """Latest (qvel, tau) from the 30Hz readback — NaN until first valid read.

        The inner loop owns the sole XArmAPI connection, so the outer loop must
        source dynamics here instead of calling the SDK (avoids connection contention).
        """
        with self._lock:
            return self._arm_qvel.copy(), self._arm_tau.copy()

    def get_state_and_dynamics(self) -> tuple[np.ndarray, bool, float, np.ndarray, np.ndarray]:
        """Atomic read of (qpos, error_state, target_ts, qvel, tau) in one lock.

        Single lock acquisition guarantees all five fields are from the same
        inner-loop tick — eliminates the one-tick (≈33ms) temporal skew between
        ``get_state()`` + ``get_dynamics()`` in the outer loop.
        """
        with self._lock:
            return (
                self._arm_qpos.copy(),
                self._error_state,
                self._target_ts,
                self._arm_qvel.copy(),
                self._arm_tau.copy(),
            )

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def ensure_running(self) -> bool:
        """In-process threads can't crash independently — always running."""
        return True

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def tracking_error(self) -> float:
        """Last |target - current| L∞ joint error (rad). Passive monitor."""
        with self._lock:
            return self._tracking_error

    @property
    def last_sent_cmd(self) -> np.ndarray:
        """The delta-clamped target actually forwarded to the SDK last tick (7,).

        Updated at the two hardware write sites only: ``_send_target`` (normal
        dispatch, post delta-clamp) and ``_hold_position`` (hold position).
        During holds this equals the held position (plan §4.9 "sent" stream).
        """
        with self._lock:
            return self._last_sent_cmd.copy()

    @property
    def mode(self) -> int:
        """Current xArm control mode via the live connection (6 during teleop).

        Returns -1 when the loop thread has no connection (not started / exited).
        """
        arm = self._arm
        if arm is None:
            return -1
        try:
            return int(getattr(arm, "mode", -1))
        except Exception:
            return -1

    @property
    def connected(self) -> bool:
        """Whether the inner loop's XArmAPI connection reports connected."""
        arm = self._arm
        if arm is None:
            return False
        try:
            return bool(getattr(arm, "connected", False))
        except Exception:
            return False

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

    def emergency_stop(self, settle_timeout: float | None = None) -> bool:
        """Fast ``set_state(4)`` emergency stop (plan §4.8 ≤1 tick, A5 <60ms @30Hz).

        Two paths, neither requires the macro lock:

        * Loop thread alive → flag it (``_emergency_event``) to issue
          ``set_state(4)`` on its OWN live connection at the top of its next
          tick and exit (one SDK call, no disconnect/reconnect round-trip —
          unlike the ``exec_macro(ARM_CMD_EMERGENCY_STOP)`` path). Waits up
          to ~3 loop periods for the thread to honor the request.
        * Loop thread stopped (a mode-changing RPC macro is in flight and
          owns the controller) → ``set_state(4)`` via a short-lived XArm7
          connection. The controller honors it immediately regardless of the
          in-flight Mode 1 waypoint stream; the macro's next
          ``set_servo_angle_j`` / blocking ``wait`` then fails and the macro
          unwinds without reconstructing Mode 6 (its finally checks
          ``_emergency_event``).

        Returns True if ``set_state(4)`` was issued (loop honored the request
        or the fallback connection reported success). Never raises.
        """
        self._emergency_event.set()
        self.set_target(None)  # drop any live target immediately
        if self.is_alive:
            budget = 3.0 * self._cfg.loop_period if settle_timeout is None else float(settle_timeout)
            thread = self._thread
            if thread is not None:
                thread.join(timeout=max(budget, 0.05))
            if not self.is_alive:
                return True
            logger.warning("ArmInnerLoop.emergency_stop: loop thread did not exit in budget — reconnect fallback")
        # Loop stopped or unresponsive: set_state(4) on a short-lived connection.
        try:
            macro_arm = self._connect_macro_arm()
        except Exception as e:
            logger.warning("ArmInnerLoop.emergency_stop: fallback connection failed: %s", e)
            return False
        if macro_arm is None:
            return False
        try:
            return bool(macro_arm.stop())  # XArm7.stop → set_state(4)
        except Exception as e:
            logger.warning("ArmInnerLoop.emergency_stop: set_state(4) failed: %s", e)
            return False
        finally:
            try:
                macro_arm.disconnect()
            except Exception:
                pass

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

    # ── Initialisation sub-steps (extracted from _run for testability) ──

    @staticmethod
    def _connect_arm(ip: str):
        """Create XArmAPI connection. Returns (arm, None) on success, (None, None) on failure."""
        from xarm.wrapper import XArmAPI

        try:
            return XArmAPI(ip, is_radian=True)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.error("ArmInnerLoop: XArmAPI init failed: %s", e)
            return None

    def _init_mode_6(self, arm) -> bool:
        """Initialise Mode 6 (online trajectory planning). Returns True on success."""
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        self._init_mode(arm)
        arm.set_collision_sensitivity(1)

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
                return False
        return True

    def _bootstrap_state(self, arm) -> np.ndarray | None:
        """Read initial joint position and seed shared state. Returns current_qpos or None."""
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code != 0 or len(states) == 0:
            logger.error("ArmInnerLoop: failed to read initial qpos (code=%d)", code)
            return None

        current_qpos = np.asarray(states[0], dtype=np.float64)[:7].copy()
        if not np.all(np.isfinite(current_qpos)):
            logger.error("ArmInnerLoop: initial qpos contains NaN/Inf — refusing to start")
            return None

        with self._lock:
            self._arm_qpos = current_qpos.copy()
            self._error_state = False
            self._last_sent_cmd = current_qpos.copy()
        return current_qpos

    def _init_dynamics(self, arm) -> None:
        """One-shot velocity/torque read to avoid NaN window at teleop start.

        Best-effort: if hardware read fails, dynamics stay NaN (safe —
        validate_action rejects NaN).
        """
        try:
            _dyn_code, _dyn_states = arm.get_joint_states(is_radian=True, num=3)
            if _dyn_code == 0:
                if len(_dyn_states) > 1:
                    _v = np.asarray(_dyn_states[1], dtype=np.float64)
                    if _v.shape[0] >= 7 and np.all(np.isfinite(_v[:7])):
                        self._arm_qvel = _v[:7].copy()
                if len(_dyn_states) > 2:
                    _tau = np.asarray(_dyn_states[2], dtype=np.float64)
                    if _tau.shape[0] >= 7 and np.all(np.isfinite(_tau[:7])):
                        self._arm_tau = _tau[:7].copy()
        except (RuntimeError, OSError):
            pass

    def _handle_arm_error(self, arm, arm_error: int) -> str:
        """Respond to a non-zero arm error code.

        Returns one of ``"continue"``, ``"break"``, or ``"ok"`` (no error).
        ``"continue"`` means the caller should hold and re-arm — the error was
        recoverable.
        """
        if arm_error == 0:
            return "ok"

        if arm_error in _RECOVERABLE_ERRORS:
            logger.warning(
                "ArmInnerLoop: arm error_code=%d (%s) — recoverable, re-initialising mode",
                arm_error,
                decode_error(arm_error),
            )
            with self._lock:
                self._arm_target = None
            self._recover_mode(arm)
            self._hold_position(arm)
            self._signal_ready_only()
            return "continue"

        logger.error(
            "ArmInnerLoop: arm error_code=%d (%s) — stopping inner loop",
            arm_error,
            decode_error(arm_error),
        )
        with self._lock:
            self._error_state = True
        return "break"

    def _read_and_update_state(self, arm) -> np.ndarray | None:
        """Read joint states from hardware and update shared state under lock.

        Returns current_qpos on success, or ``None`` if the read failed
        (caller should break or continue based on context).
        """
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
        except (RuntimeError, OSError) as e:
            logger.error("ArmInnerLoop: get_joint_states failed: %s", e)
            with self._lock:
                self._error_state = True
            return None
        time.sleep(0)  # explicit GIL yield — unblocks main-thread numpy ops

        if code != 0:
            logger.error("ArmInnerLoop: arm error code=%d — stopping inner loop", code)
            with self._lock:
                self._error_state = True
            return None

        # Parse state arrays (best-effort per field)
        current_qpos = None
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
                with self._lock:
                    self._arm_qvel = v[:7].copy()
        if len(states) > 2:
            tau = np.asarray(states[2], dtype=np.float64)
            if tau.shape[0] >= 7 and np.all(np.isfinite(tau[:7])):
                with self._lock:
                    self._arm_tau = tau[:7].copy()

        return current_qpos

    # ── Main loop ──

    def _run(self) -> None:
        # ── Phase 1: Connect ──
        arm = self._connect_arm(self._ip)
        if arm is None:
            with self._lock:
                self._error_state = True
            return
        self._arm = arm

        try:
            # ── Phase 2: Mode 6 initialisation ──
            if not self._init_mode_6(arm):
                with self._lock:
                    self._error_state = True
                return

            # ── Phase 3: Bootstrap state ──
            current_qpos = self._bootstrap_state(arm)
            if current_qpos is None:
                with self._lock:
                    self._error_state = True
                return

            # ── Phase 4: Init dynamics ──
            self._init_dynamics(arm)

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

            # ── Phase 5: Main loop ──
            while not self._stop_event.is_set():
                limiter.wait()

                # Emergency stop
                if self._emergency_event.is_set():
                    try:
                        arm.set_state(4)
                    except (RuntimeError, OSError):
                        pass
                    with self._lock:
                        self._error_state = True
                    logger.info("ArmInnerLoop: emergency stop — set_state(4) issued, loop exiting")
                    break

                # Read target under lock
                with self._lock:
                    target = self._arm_target.copy() if self._arm_target is not None else None
                    target_ts = self._target_ts

                now = time.perf_counter()

                # Timeout → hold (skip during startup)
                no_target_yet = last_target_ts == 0.0
                if not no_target_yet and (
                    target is None or (now - max(target_ts, last_target_ts) > self._cfg.target_timeout_s)
                ):
                    self._hold_position(arm)
                    self._signal_ready_only()
                    continue

                if target is None:
                    continue  # startup: no target yet

                last_target_ts = target_ts

                # NaN guard — hold last valid position
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

                # Read hardware state
                current_qpos = self._read_and_update_state(arm)
                if current_qpos is None:
                    continue

                # Check arm error flag (C31 collision: error_code set without
                # necessarily failing get_joint_states)
                arm_error = getattr(arm, "error_code", 0)
                action = self._handle_arm_error(arm, arm_error)
                if action == "break":
                    break
                if action == "continue":
                    continue

                # Passive tracking-error + mode monitor
                self._monitor(arm, current_qpos)

                # Forward target → firmware trajectory planner
                self._send_target(arm, target)

                # Sync handshake (after hardware write)
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
            self._arm = None  # connection closed — mode/connected queries report unknown
            logger.info("ArmInnerLoop: stopped")

    # ── Command dispatch ──

    def _monitor(self, arm, current_qpos: np.ndarray) -> None:
        """Passive health monitor — tracking error (A4) + mode drift (A5).

        Tracking error threshold is **velocity-adaptive**: the expected
        tracking error grows with commanded joint speed because the firmware
        acceleration limit (joint_max_acc) physcally limits how fast the arm
        can respond to target changes, especially during direction reversals.

        * ``logger.info`` — elevated but commensurate with current speed
          (arm at its limit, not a fault).
        * ``logger.warning`` — anomalously high for the current speed
          (possible firmware degradation, increased friction, or collision).

        Never mutates commands or the error state.
        """
        cfg = self._cfg
        err = 0.0
        if self._last_sent_target is not None:
            err = float(np.max(np.abs(self._last_sent_target - current_qpos[:7])))
        with self._lock:
            self._tracking_error = err

        # ── Estimate commanded joint velocity (peak-hold envelope) ──
        # Peak-hold (instant attack, exponential decay) replaces the old EMA
        # (α=0.15, τ≈0.22s) which caused ~0.2s threshold lag during transients.
        # The envelope responds instantly to velocity increases and decays slowly
        # (α=0.15) for noise rejection during steady state.
        cmd_vel = 0.0
        if self._last_sent_target is not None and self._prev_monitor_target is not None:
            raw_delta = float(np.max(np.abs(self._last_sent_target - self._prev_monitor_target)))
            raw_vel = raw_delta / cfg.loop_period
            # Clamp to firmware speed limit: prevents idle→teleop first-tick spike
            # (delta accumulated during idle can produce raw_vel > 225°/s).
            raw_vel = min(raw_vel, cfg.joint_max_speed)
            self._cmd_vel_env = max(raw_vel, 0.85 * self._cmd_vel_env)
            cmd_vel = self._cmd_vel_env
        if self._last_sent_target is not None:
            self._prev_monitor_target = self._last_sent_target.copy()

        # ── Adaptive threshold (physics-based) ──
        #   expected = steady_state_lag + reversal_distance + noise_bias
        #   steady_state_lag ≈ cmd_vel / inner_loop_rate
        #   reversal_distance ≈ cmd_vel² / joint_max_acc  (full decel+accel, v²/a)
        #   noise_bias = additive (not multiplicative) — encoder quant, L∞ max,
        #                Mode 6 smoothing artefacts don't scale with speed.
        # Clamped to [0.18, 0.60] rad — 0.18 floor covers unmodeled noise at
        # low speed (was 0.15 with old multiplicative 1.25×, which collapsed to
        # a constant below ~66°/s and produced false positives).
        adaptive: float = 0.0
        if cfg.tracking_error_warn_rad > 0 and cfg.joint_max_acc > 0:
            inner_rate = 1.0 / cfg.loop_period  # 30 Hz
            steady = cmd_vel / inner_rate
            # Full reversal distance: decelerate to zero + accelerate to cmd_vel
            accel = cmd_vel * cmd_vel / cfg.joint_max_acc
            expected = steady + accel + _TRACKING_NOISE_RAD
            adaptive = float(np.clip(expected, 0.18, cfg.tracking_error_adaptive_max_rad))

        mode_bad = getattr(arm, "mode", 6) != 6
        track_elevated = adaptive > 0 and err > adaptive
        track_anomalous = adaptive > 0 and (err > 2.0 * adaptive or err >= cfg.tracking_error_anomaly_cap_rad)

        # ── Independent throttles ──
        # Anomalous warning: genuinely unexpected → always warn (throttled).
        if self._track_warn_throttle > 0:
            self._track_warn_throttle -= 1
        elif track_anomalous:
            logger.warning(
                "ArmInnerLoop: anomalous tracking error %.3f rad "
                "(adaptive threshold %.3f rad, cmd_vel=%.0f°/s) — "
                "possible degradation",
                err,
                adaptive,
                np.degrees(cmd_vel),
            )
            self._track_warn_throttle = 50  # ~1.67s at 30Hz

        # Elevated info: expected at current speed → throttled info log.
        if self._track_info_throttle > 0:
            self._track_info_throttle -= 1
        elif track_elevated and not track_anomalous:
            logger.info(
                "ArmInnerLoop: elevated tracking error %.3f rad "
                "(adaptive threshold %.3f rad, cmd_vel=%.0f°/s)",
                err,
                adaptive,
                np.degrees(cmd_vel),
            )
            self._track_info_throttle = 150  # ~5s at 30Hz

        # Mode-drift check (independent throttle from tracking error).
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
        # joint_max_speed ÷ outer_loop_hz: ≈0.098 rad @16Hz outer (~3x headroom).
        clamped = target[:7].copy()
        # Absolute joint limit clip (before per-step delta clamp, per validate.py §5)
        np.clip(clamped, self._cfg.joint_limit_lower, self._cfg.joint_limit_upper, out=clamped)
        if self._cfg.max_joint_delta > 0 and self._last_sent_target is not None:
            delta = clamped - self._last_sent_target
            delta = np.clip(delta, -self._cfg.max_joint_delta, self._cfg.max_joint_delta)
            clamped = self._last_sent_target + delta

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
            self._last_sent_cmd = clamped.copy()  # sent stream (plan §4.9)

    # ── Helpers ──

    def _recover_mode(self, arm) -> bool:
        """Clear error latch and re-init Mode 6 after a recoverable error.

        After firmware rejects a command (e.g. error 22 self-collision), the arm
        drops from Mode 6 to Mode 0.  ``clean_error`` + ``set_state(0)`` clears
        the latch but does NOT restore the control mode — without Mode 6,
        subsequent ``set_servo_angle`` calls use the wrong protocol and the arm
        is effectively dead (observed as ``set_mode(1)`` returning code 10
        repeatedly during return_home).

        Returns ``True`` on success, ``False`` if the re-init attempt fails.
        """
        arm.clean_error()
        arm.clean_warn()
        arm.set_state(0)

        # Re-init Mode 6 — single attempt.  The XArm SDK's own ``_set_mode``
        # uses a single attempt with no retry loop.  The 3-retry loop added
        # ~600 ms worst-case blocking (~18 inner-loop ticks at 30Hz) without
        # evidence that retries improve firmware recovery.
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
                logger.info("ArmInnerLoop: Mode 6 re-initialised")
                return True
            logger.warning(
                "ArmInnerLoop: set_mode(6) returned but mode=%d",
                actual_mode,
            )
        except (RuntimeError, OSError) as e:
            logger.warning("ArmInnerLoop: Mode 6 re-init failed: %s", e)

        logger.error("ArmInnerLoop: failed to re-init Mode 6 — arm may be in degraded mode")
        return False

    def _hold_position(self, arm) -> None:
        """Read current position and re-send as hold command via set_servo_angle.

        Updates ``_last_sent_target`` so the per-step delta clamp and tracking
        error monitor use a meaningful baseline when targets resume after a hold.
        """
        try:
            code, states = arm.get_joint_states(is_radian=True, num=3)
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
            # Refresh dynamics during hold so torque gates are meaningful
            # when teleop resumes (num=3 above returns velocity + torque alongside position).
            if code == 0:
                if len(states) > 1:
                    v = np.asarray(states[1], dtype=np.float64)
                    if v.shape[0] >= 7 and np.all(np.isfinite(v[:7])):
                        with self._lock:
                            self._arm_qvel = v[:7].copy()
                if len(states) > 2:
                    tau = np.asarray(states[2], dtype=np.float64)
                    if tau.shape[0] >= 7 and np.all(np.isfinite(tau[:7])):
                        with self._lock:
                            self._arm_tau = tau[:7].copy()
            # Lock-protected: last_sent_cmd is read from another thread via last_sent_cmd property.
            if code == 0 and len(states) > 0 and np.all(np.isfinite(hold)):
                with self._lock:
                    self._last_sent_cmd = hold.copy()  # sent stream = hold position (plan §4.9)
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

    # ── Macro commands (RPC executor — plan §4.3) ──

    def exec_macro(self, code: int, fields: dict) -> dict:
        """Execute a blocking macro command against the controller.

        Mirrors ``RobotInterface.return_to_home``'s exact call sequence: the
        mode-changing macros (EXEC_WAYPOINTS / RESET_BLOCKING / REINIT_MODE6)
        first stop the inner-loop thread — the sole Mode 6 command source, so
        no concurrent SDK traffic fights the macro on the controller — then run
        the move through a short-lived ``XArm7`` connection:

          * EXEC_WAYPOINTS — Mode 1 ``set_servo_angle_j`` per waypoint with a
            ``dt`` sleep, exactly as ``XArm7.send_action`` inside
            ``RobotInterface._execute_waypoints`` (caller segments >2048 points).
          * RESET_BLOCKING — Mode 0 ``set_servo_angle(wait=True)``, exactly as
            ``XArm7.reset`` (``speed``/``acc`` fields override the config
            reset defaults when > 0).
          * CLEAR_ERROR — ``XArm7.clear_error`` semantics; does NOT change the
            control mode, safe while the inner loop is running.
          * EMERGENCY_STOP — ``set_state(4)`` (``XArm7.stop``); the inner loop
            stays stopped afterwards.
          * REINIT_MODE6 — stop + ``XArm7.connect`` (robot_init: clean_error +
            motion_enable + collision/TCP params + reduced limits) + restart.

        After EXEC_WAYPOINTS / RESET_BLOCKING / REINIT_MODE6 the inner loop is
        restarted (its ``_run`` re-connects, verifies Mode 6 and re-arms the
        ramp) — Mode 6 is automatically reconstructed on completion (plan §4.3).

        Returns ``{"ok": bool, "arm_err": int, "sdk_ret": int,
        "final_qpos": ndarray(7,)}``.  Never raises — failures land in the
        result record so the RPC server can always answer.
        """
        from dexmani_real.shm.robot_layouts import (
            ARM_CMD_CLEAR_ERROR,
            ARM_CMD_EMERGENCY_STOP,
            ARM_CMD_EXEC_WAYPOINTS,
            ARM_CMD_REINIT_MODE6,
            ARM_CMD_RESET_BLOCKING,
        )

        result: dict = {
            "ok": False,
            "arm_err": 0,
            "sdk_ret": -1,
            "final_qpos": np.zeros(7, dtype=np.float64),
        }
        macro_arm = None
        restart_after = False
        try:
            if code == ARM_CMD_CLEAR_ERROR:
                # Mode-safe: no mode switch, inner loop may keep running.
                macro_arm = self._connect_macro_arm()
                if macro_arm is not None:
                    result["ok"] = macro_arm.clear_error()

            elif code == ARM_CMD_EMERGENCY_STOP:
                # Also raise the emergency flag so any concurrently finishing
                # macro skips its Mode 6 reconstruction (the arm must stay in
                # state 4 until a deliberate REINIT_MODE6 clears it).
                self._emergency_event.set()
                if self.is_alive:
                    self.stop()
                macro_arm = self._connect_macro_arm()
                if macro_arm is not None:
                    result["ok"] = macro_arm.stop()  # set_state(4); stays stopped

            elif code == ARM_CMD_EXEC_WAYPOINTS:
                if self.is_alive:
                    self.stop()  # silence Mode 6 dispatch before Mode 1 moves
                macro_arm = self._connect_macro_arm()
                if macro_arm is not None:
                    waypoints = np.asarray(
                        fields.get("waypoints", np.zeros((0, 7), dtype=np.float64)),
                        dtype=np.float64,
                    ).reshape(-1, 7)
                    dt = float(fields.get("dt") or macro_arm.config.dt)
                    ok = True
                    for wp in waypoints:
                        if macro_arm.is_error() or not macro_arm.send_action(wp):
                            ok = False
                            break
                        time.sleep(dt)
                    result["ok"] = ok
                restart_after = True  # auto-reconstruct Mode 6 (plan §4.3)

            elif code == ARM_CMD_RESET_BLOCKING:
                if self.is_alive:
                    self.stop()
                macro_arm = self._connect_macro_arm()
                if macro_arm is not None:
                    target = fields.get("target", None)
                    qpos = None if target is None else np.asarray(target, dtype=np.float64).reshape(7)
                    speed = float(fields.get("speed") or 0.0)
                    acc = float(fields.get("acc") or 0.0)
                    if speed > 0:
                        macro_arm.config.reset_speed = speed
                    if acc > 0:
                        macro_arm.config.reset_acc = acc
                    result["ok"] = macro_arm.reset(qpos)
                restart_after = True

            elif code == ARM_CMD_REINIT_MODE6:
                # Deliberate re-enable after an emergency stop: clear the
                # emergency flag so the finally-block reconstruction runs.
                self._emergency_event.clear()
                if self.is_alive:
                    self.stop()
                macro_arm = self._connect_macro_arm()  # robot_init clears errors + re-enables
                if macro_arm is not None:
                    result["ok"] = not macro_arm.error_state
                restart_after = True

            else:
                logger.warning("ArmInnerLoop.exec_macro: unknown code=%d", code)

            if macro_arm is not None:
                sdk_ret = getattr(macro_arm, "last_action_code", None)
                result["sdk_ret"] = -1 if sdk_ret is None else int(sdk_ret)
                try:
                    result["arm_err"] = int(getattr(macro_arm.arm, "error_code", 0) or 0)
                except Exception:
                    result["arm_err"] = 0
                try:
                    final = np.asarray(macro_arm.get_state()["qpos"], dtype=np.float64)
                    if final.shape == (7,) and np.all(np.isfinite(final)):
                        result["final_qpos"] = final
                except Exception:
                    pass
        except Exception as e:
            logger.warning("ArmInnerLoop.exec_macro(code=%d) exception: %s", code, e)
            result["ok"] = False
        finally:
            if macro_arm is not None:
                try:
                    macro_arm.disconnect()
                except Exception:
                    pass
            if restart_after and not self._emergency_event.is_set():
                # Mode 6 reconstruction: _run re-connects, re-inits and verifies
                # Mode 6 before setting the ready event. Skipped when an
                # emergency stop was raised mid-macro — the arm must stay in
                # state 4 until a deliberate REINIT_MODE6 (plan §4.8).
                self.start()
                if not self.wait_ready(timeout=10.0):
                    logger.error(
                        "ArmInnerLoop.exec_macro(code=%d): Mode 6 not restored within 10s",
                        code,
                    )
                    result["ok"] = False
        return result

    def _connect_macro_arm(self):
        """Short-lived XArm7 connection for macro commands (plan A2 lazy import).

        Separate socket from the inner-loop connection, mirroring today's
        RobotInterface.XArm7 ↔ ArmInnerLoop split; only ever used while the
        inner-loop thread is stopped (or for mode-safe CLEAR_ERROR).
        """
        from dexmani_real.robot.xarm7.xarm7 import XArm7, XArm7Config

        arm = XArm7(XArm7Config(ip=self._ip))
        if not arm.connect():
            logger.error(
                "ArmInnerLoop.exec_macro: XArm7 macro connection failed: %s",
                arm.last_error_message,
            )
            return None
        return arm
