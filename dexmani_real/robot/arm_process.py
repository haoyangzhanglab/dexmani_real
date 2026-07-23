"""Arm control process + SHM façade — crash-isolated xArm7 Mode 6 teleop (plan §4-5).

Runs ArmInnerLoop in a separate fork process that owns the sole XArmAPI
connection; the main process (and future policy processes) talk to it through
seqlock SHM rings:

    Main process (16Hz)                      Arm control child (30Hz)
    ───────────────────                      ────────────────────────
    ArmSHMFaçade.set_target(cmd)  ──arm_target──►  inner.set_target(cmd|None)
    ArmSHMFaçade.get_state()      ◄──arm_state───  publish every child tick
                                                     (incl. inner.last_sent_cmd)
    ArmSHMFaçade.rpc(EXEC_WAYPOINTS|RESET_BLOCKING|CLEAR_ERROR|
                     EMERGENCY_STOP|REINIT_MODE6)
                                  ◄─arm_cmd / arm_cmd_result─►  inner.exec_macro()

Safety semantics (plan §4.7-4.9, §5):
  * estop_event is checked FIRST each child tick → hold + set_state(4) once,
    via ``ArmInnerLoop.emergency_stop()``: the loop thread issues set_state(4)
    on its OWN live connection (≤1 tick, no reconnect — plan §4.8/A5), and
    when a macro owns the controller (loop stopped) a short-lived connection
    issues set_state(4) WITHOUT waiting on macro_lock — the controller honors
    it immediately and the in-flight macro unwinds on its next failed move.
  * Target staleness (slot monotonic ts via is_fresh, target_timeout_s=0.2) or
    is_hold → inner.set_target(None) — the inner loop re-arms its soft-start
    ramp on hold.
  * producer_id nonzero mismatch → throttled warning + ignore (D9).
  * Façade get_state freshness gate (age > state_stale_mult/loop_hz) returns a
    fabricated error record so validate_action trips; range sanity vs
    arm_joint_bounds treats implausible qpos as torn (last-good fallback).
  * SIGINT (Ctrl-C reaches the whole process group) → hold last position, exit.
  * The arm child is daemon=True: if the main process dies, Mode 6 firmware
    holds the last position on disconnect (inner_loop.py behaviour).

SDK imports are lazy inside the child run (plan assumption A2); the main
process never imports xarm.wrapper through this module.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

from dexmani_real.robot.isolation import arm_isolation_enabled
from dexmani_real.shm.robot_layouts import (
    ARM_CMD_CLEAR_ERROR,
    ARM_CMD_DTYPE,
    ARM_CMD_EMERGENCY_STOP,
    ARM_CMD_EXEC_WAYPOINTS,
    ARM_CMD_REINIT_MODE6,
    ARM_CMD_RESET_BLOCKING,
    ARM_CMD_RESULT_DTYPE,
    ARM_STATE_DTYPE,
    ARM_TARGET_DTYPE,
    MAX_ARM_WAYPOINTS,
    PRODUCER_TELEOP,
    new_frame,
)
from dexmani_real.shm.robot_ring import SeqlockRingBuffer, is_fresh
from dexmani_real.shm.robot_rpc import RpcClient, RpcServer
from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

if TYPE_CHECKING:
    from dexmani_real.robot.inner_loop import ArmInnerLoop, ArmInnerLoopConfig

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ArmProcessConfig:
    """Configuration for ArmControlProcess / ArmSHMFaçade.

    Ring names are derived from ``shm_prefix``: ``f"{shm_prefix}_state"``
    (maxlen=3), ``_target`` (maxlen=2), ``_cmd`` (maxlen=2), ``_cmd_result``
    (maxlen=2).

    Attributes:
        loop_hz: Child tick rate (Hz). Also the freshness-gate denominator.
        shm_prefix: Prefix for the four SHM ring names.
        target_timeout_s: Max age of an arm_target frame before the child
                          holds (slot monotonic ts; replaces perf_counter).
        state_stale_mult: Façade freshness gate: arm_state older than
                          state_stale_mult / loop_hz seconds → fabricated
                          error record (plan §4.7).
        ready_timeout_s: Child-side inner-loop wait_ready timeout.
        rpc_timeout_s: RpcClient call timeout for macro commands. Default
                       60 s so a worst-case blocking home move cannot spuriously
                       time out: RESET_BLOCKING runs set_servo_angle(wait=True)
                       at reset_speed=20°/s — ~9 s for a 180° single-joint move
                       from init pose, ~18 s from near a ±357.5° soft limit,
                       plus Mode 6 reconstruction. NOTE: a timeout means the
                       RESULT was not observed in time — the command is NOT
                       aborted and may still be running (see rpc()).
        expected_producer_id: Reject arm_target frames whose nonzero
                              producer_id differs (D9).
        inner_kwargs: Extra kwargs forwarded to the inner-loop factory
                      (e.g. {"ip": ..., "cfg": ArmInnerLoopConfig(...)}).
    """

    loop_hz: float = 50.0
    shm_prefix: str = "dexmani_arm"
    target_timeout_s: float = 0.2
    state_stale_mult: float = 3.0
    ready_timeout_s: float = 30.0
    rpc_timeout_s: float = 60.0
    expected_producer_id: int = PRODUCER_TELEOP
    inner_kwargs: dict = field(default_factory=dict)


# Default joint-range sanity window (rad) when no real soft limits are wired.
# ±2π·2 = ±4π per joint — catches torn/garbage qpos; P1 wiring passes the real
# XArm7 qpos_min_soft/qpos_max_soft via the arm_joint_bounds ctor argument.
_DEFAULT_JOINT_BOUND = 2.0 * np.pi * 2.0

# Façade warning throttle (s) — matches SeqlockRingBuffer's <=1/5s cadence.
_WARN_THROTTLE_S = 5.0


# ═══════════════════════════════════════════════════════════════════
# Inner-loop factory
# ═══════════════════════════════════════════════════════════════════


def _default_inner_factory(config: ArmProcessConfig) -> Any:
    """Build the real ArmInnerLoop inside the child (lazy SDK import, plan A2).

    Imports XArm7 too so a missing/broken xArm SDK fails fast at child startup
    (XArm7 is what ArmInnerLoop.exec_macro uses for macro commands).
    """
    from dexmani_real.robot.inner_loop import ArmInnerLoop
    from dexmani_real.robot.xarm7.xarm7 import XArm7, XArm7Config  # noqa: F401  (fail-fast SDK check)

    kwargs = dict(config.inner_kwargs or {})
    kwargs.setdefault("ip", XArm7Config().ip)
    return ArmInnerLoop(**kwargs)


# ═══════════════════════════════════════════════════════════════════
# Child process
# ═══════════════════════════════════════════════════════════════════


def _publish_arm_state(state_ring: SeqlockRingBuffer, inner: Any) -> None:
    """Publish one ARM_STATE record from the inner loop (never raises).

    ``last_sent`` carries inner.last_sent_cmd — the delta-clamped value
    actually forwarded to the SDK (hold position during holds), feeding the
    recording "sent" stream (plan §4.9).
    """
    try:
        qpos, error_state, _target_ts = inner.get_state()
        qvel, tau, temps = inner.get_dynamics()
        frame = new_frame(ARM_STATE_DTYPE)
        frame["qpos"][0] = np.asarray(qpos, dtype=np.float64)[:7]
        frame["qvel"][0] = np.asarray(qvel, dtype=np.float64)[:7]
        frame["tau"][0] = np.asarray(tau, dtype=np.float64)[:7]
        frame["temps"][0] = np.asarray(temps, dtype=np.float64)[:7]
        frame["error_state"][0] = 1 if error_state else 0
        frame["connected"][0] = 1 if inner.connected else 0
        frame["mode"][0] = int(inner.mode)
        frame["tracking_err"][0] = float(inner.tracking_error)
        frame["last_sent"][0] = np.asarray(inner.last_sent_cmd, dtype=np.float64)[:7]
        state_ring.write(frame)
    except Exception:
        logger.exception("ArmControlProcess: arm_state publish failed")


def _arm_child_main(
    config: ArmProcessConfig,
    estop_event: Any,
    stop_event: Any,
    ready_event: Any,
    crashed_event: Any,
    inner_factory: Callable[[ArmProcessConfig], Any] | None,
) -> None:
    """Arm control child entry point (module-level for fork+pickling).

    Sole XArmAPI connection owner. Each tick, in priority order:
      1. estop_event FIRST → hold + set_state(4) once via inner.emergency_stop()
         (fast path, does NOT take macro_lock — preempts in-flight RPC macros);
      2. SIGINT received → hold + clean exit;
      3. arm_target ring → is_hold / stale (target_timeout_s) → set_target(None)
         (inner loop re-arms the soft-start ramp on hold); nonzero producer_id
         mismatch → throttled warn + ignore; else set_target(qpos);
      4. publish ARM_STATE every tick;
      5. RPC macros served on a dedicated thread (state ring keeps flowing
         during long macros — plan §4.3) via inner.exec_macro.
    """
    # ── SIGINT: Ctrl-C reaches the whole process group; hold, then exit ──
    sigint_received = threading.Event()

    def _on_sigint(signum: int, frame: Any) -> None:
        sigint_received.set()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        pass  # not in the main thread — should not happen for the fork child

    inner: Any = None
    state_ring: SeqlockRingBuffer | None = None
    target_ring: SeqlockRingBuffer | None = None
    cmd_ring: SeqlockRingBuffer | None = None
    result_ring: SeqlockRingBuffer | None = None
    try:
        factory = inner_factory if inner_factory is not None else _default_inner_factory
        inner = factory(config)

        # Attach the rings the parent created (create=False).
        prefix = config.shm_prefix
        state_ring = SeqlockRingBuffer(f"{prefix}_state", ARM_STATE_DTYPE, maxlen=3, create=False, stale_cleanup=False)
        target_ring = SeqlockRingBuffer(
            f"{prefix}_target", ARM_TARGET_DTYPE, maxlen=2, create=False, stale_cleanup=False
        )
        cmd_ring = SeqlockRingBuffer(f"{prefix}_cmd", ARM_CMD_DTYPE, maxlen=2, create=False, stale_cleanup=False)
        result_ring = SeqlockRingBuffer(
            f"{prefix}_cmd_result", ARM_CMD_RESULT_DTYPE, maxlen=2, create=False, stale_cleanup=False
        )

        inner.start()
        if not inner.wait_ready(timeout=config.ready_timeout_s):
            logger.error(
                "ArmControlProcess: inner loop not ready within %.0fs (arm connection?)",
                config.ready_timeout_s,
            )
            _publish_arm_state(state_ring, inner)  # best-effort error record for the façade
            crashed_event.set()
            return
        ready_event.set()

        # ── RPC executor on its own thread: long macros (waypoints / blocking
        # reset) must not stop the tick loop publishing arm_state, or the
        # façade freshness gate would fabricate a false emergency (plan §4.3).
        macro_lock = threading.Lock()  # serialize tick-loop estop vs RPC macros

        def _handle_macro(request: np.ndarray, seq: int) -> np.ndarray:
            result = new_frame(ARM_CMD_RESULT_DTYPE)
            try:
                code = int(request["cmd"][0])
                n_wp = int(request["n_waypoints"][0])
                fields = {
                    "waypoints": np.array(request["waypoints"][0][:n_wp], dtype=np.float64),
                    "dt": float(request["dt"][0]),
                    "target": np.array(request["target"][0], dtype=np.float64),
                    "speed": float(request["speed"][0]),
                    "acc": float(request["acc"][0]),
                }
                with macro_lock:
                    out = inner.exec_macro(code, fields)
                result["ok"][0] = 1 if out.get("ok") else 0
                result["arm_err"][0] = int(out.get("arm_err", 0))
                result["sdk_ret"][0] = int(out.get("sdk_ret", -1))
                result["final_qpos"][0] = np.asarray(out.get("final_qpos", np.zeros(7)), dtype=np.float64)[:7]
            except Exception as e:
                logger.warning("ArmControlProcess: macro handler failed: %s", e)
                result["ok"][0] = 0
                result["sdk_ret"][0] = -1
            return result

        server = RpcServer(cmd_ring, result_ring, _handle_macro)

        def _rpc_loop() -> None:
            while not stop_event.is_set():
                try:
                    if not server.handle_pending():
                        time.sleep(0.005)
                except Exception:
                    logger.exception("ArmControlProcess: RPC loop iteration failed")
                    time.sleep(0.05)

        rpc_thread = threading.Thread(target=_rpc_loop, name="arm_rpc", daemon=True)
        rpc_thread.start()

        limiter = RateManager(config.loop_hz)
        estop_done = False
        producer_throttle = 0
        logger.info("ArmControlProcess: child loop started @ %.0f Hz", config.loop_hz)

        while not stop_event.is_set():
            limiter.wait()

            # ── 1. estop FIRST ──
            if estop_event.is_set():
                inner.set_target(None)
                if not estop_done:
                    estop_done = True
                    try:
                        fast_estop = getattr(inner, "emergency_stop", None)
                        if callable(fast_estop):
                            # Fast path (plan §4.8 / A5): set_state(4) on the
                            # loop's own connection (≤1 tick) — or, when a
                            # macro has stopped the loop, a short-lived
                            # connection WITHOUT macro_lock, so an in-flight
                            # EXEC_WAYPOINTS/RESET_BLOCKING cannot defer the
                            # emergency stop (the controller halts at once
                            # and the macro unwinds on its next failed move).
                            fast_estop()
                        else:
                            # Inner-loop impl without the fast path (e.g. test
                            # fakes): the reconnect-based EMERGENCY_STOP macro.
                            with macro_lock:
                                inner.exec_macro(ARM_CMD_EMERGENCY_STOP, {})
                    except Exception as e:
                        logger.warning("ArmControlProcess: estop failed: %s", e)
                _publish_arm_state(state_ring, inner)
                continue

            # ── 2. SIGINT → hold + exit (Mode 6 firmware holds on disconnect) ──
            if sigint_received.is_set():
                logger.info("ArmControlProcess: SIGINT received — holding last position and exiting")
                break

            # ── 3. Target ring → inner loop ──
            latest = target_ring.read_latest()
            if latest is None:
                inner.set_target(None)
            else:
                data, ts_ns, _seq = latest
                if bool(data["is_hold"][0]) or not is_fresh(ts_ns, config.target_timeout_s):
                    # Hold sentinel or stale target → hold; the inner loop
                    # re-arms its soft-start speed ramp on every hold.
                    inner.set_target(None)
                else:
                    pid = int(data["producer_id"][0])
                    if pid != 0 and pid != config.expected_producer_id:
                        if producer_throttle > 0:
                            producer_throttle -= 1
                        else:
                            logger.warning(
                                "ArmControlProcess: producer_id=%d != expected %d — ignoring target",
                                pid,
                                config.expected_producer_id,
                            )
                            producer_throttle = max(int(config.loop_hz), 1) * 5  # ~5s
                    else:
                        inner.set_target(np.asarray(data["target"][0], dtype=np.float64))

            # ── 4. Publish state every tick ──
            _publish_arm_state(state_ring, inner)

    except Exception:
        logger.exception("ArmControlProcess: fatal error in child")
        crashed_event.set()
    finally:
        if inner is not None:
            try:
                inner.set_target(None)  # explicit hold command ...
                time.sleep(0.2)  # ... given one beat to execute before teardown
            except Exception:
                pass
            try:
                inner.stop(timeout=3.0)  # disconnect → firmware holds last position
            except Exception:
                logger.exception("ArmControlProcess: inner stop failed")
        # Close (never unlink — the parent owns the blocks) our ring handles.
        for ring in (state_ring, target_ring, cmd_ring, result_ring):
            if ring is not None:
                try:
                    ring.close()
                except Exception:
                    pass
        logger.info("ArmControlProcess: child exited")


# ═══════════════════════════════════════════════════════════════════
# ArmControlProcess — process wrapper
# ═══════════════════════════════════════════════════════════════════


class ArmControlProcess:
    """Wraps the fork child running the arm inner loop (daemon=True).

    Mirrors CameraProcess lifecycle semantics: stop_event for clean shutdown,
    crashed event for failure detection, daemon=True so a SIGKILLed main
    process takes the child with it (Mode 6 firmware then holds position).
    """

    def __init__(
        self,
        config: ArmProcessConfig,
        estop_event: Any,
        inner_factory: Callable[[ArmProcessConfig], Any] | None = None,
    ) -> None:
        self._config = config
        self._estop_event = estop_event
        self._inner_factory = inner_factory
        # Plan A2: fork explicitly for robot control processes.
        self._ctx = mp.get_context("fork")
        self._stop_event = self._ctx.Event()
        self._ready_event = self._ctx.Event()
        self._crashed = self._ctx.Event()
        self._process: BaseProcess | None = None  # ForkContext.Process is a BaseProcess

    # ── Lifecycle ──

    def start(self) -> None:
        """Start the child process (rings must already exist)."""
        if self.running:
            logger.warning("ArmControlProcess already running.")
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._crashed.clear()
        self._process = self._ctx.Process(
            target=_arm_child_main,
            args=(
                self._config,
                self._estop_event,
                self._stop_event,
                self._ready_event,
                self._crashed,
                self._inner_factory,
            ),
            name=f"arm-control-{self._config.shm_prefix}",
            daemon=True,
        )
        self._process.start()
        logger.info(
            "ArmControlProcess started (prefix=%s, loop_hz=%.0f)",
            self._config.shm_prefix,
            self._config.loop_hz,
        )

    def stop(self, timeout: float = 3.0) -> None:
        """Signal stop, join, terminate as a last resort."""
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("ArmControlProcess did not exit within %.1fs, terminating.", timeout)
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._process = None

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait until the child's inner loop has verified Mode 6 + first qpos read."""
        return self._ready_event.wait(timeout=timeout)

    # ── Status ──

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def crashed(self) -> bool:
        """Whether the child died unexpectedly (or reported init/ready failure)."""
        if self._process is not None and not self._process.is_alive() and not self._stop_event.is_set():
            self._crashed.set()
        return self._crashed.is_set()


# ═══════════════════════════════════════════════════════════════════
# ArmSHMFaçade — main-process side
# ═══════════════════════════════════════════════════════════════════


class ArmSHMFaçade:
    """Main-process façade over the arm control child (plan §4.7 guards).

    Drop-in surface for the arm half of RobotInterface: set_target /
    get_state / blocking-move macros via RPC. All reads return
    ``(record, age_ns)`` — latency-compensation input for policy deployment
    (plan §10.6).
    """

    def __init__(
        self,
        config: ArmProcessConfig,
        estop_event: Any,
        inner_factory: Callable[[ArmProcessConfig], Any] | None = None,
        arm_joint_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self._config = config
        self._proc = ArmControlProcess(config, estop_event, inner_factory)
        # Range sanity window = soft limits ± 0.05 rad (plan §4.7); P1 wiring
        # passes the real XArm7 qpos_min_soft/qpos_max_soft here.
        if arm_joint_bounds is None:
            lo = np.full(7, -_DEFAULT_JOINT_BOUND, dtype=np.float64)
            hi = np.full(7, _DEFAULT_JOINT_BOUND, dtype=np.float64)
        else:
            lo = np.asarray(arm_joint_bounds[0], dtype=np.float64).reshape(7)
            hi = np.asarray(arm_joint_bounds[1], dtype=np.float64).reshape(7)
        self._joint_lo = lo - 0.05
        self._joint_hi = hi + 0.05
        self._state_ring: SeqlockRingBuffer | None = None
        self._target_ring: SeqlockRingBuffer | None = None
        self._cmd_ring: SeqlockRingBuffer | None = None
        self._result_ring: SeqlockRingBuffer | None = None
        self._rpc_client: RpcClient | None = None
        self._last_good: np.ndarray | None = None  # last range-valid state record
        self._last_warn_monotonic: float = 0.0  # warning throttle (<=1/5s)
        self._startup_grace_remaining: int = 0  # ticks to suppress no-frame error after restart

    # ── Lifecycle ──

    def start(self) -> bool:
        """Create the SHM rings (stale cleanup) and start the child."""
        if self._proc.running:
            logger.warning("ArmSHMFaçade already running.")
            return False
        try:
            self._create_rings()
        except Exception:
            logger.exception("ArmSHMFaçade: ring creation failed — not starting child")
            return False
        self._proc.start()
        # Suppress no-frame fabricated error for the first few get_state()
        # calls — the child may need a few ticks to publish its first state
        # frame after the inner loop thread starts (plan §11.3).
        self._startup_grace_remaining = 5
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the child and unlink all rings."""
        self._proc.stop(timeout=timeout)
        self._unlink_rings()
        self._last_good = None
        logger.info("ArmSHMFaçade stopped.")

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait for the child's inner loop to be ready (default ready_timeout_s)."""
        return self._proc.wait_ready(timeout if timeout is not None else self._config.ready_timeout_s)

    def ensure_running(self) -> bool:
        """Restart a dead child — mirrors TeleopController._ensure_inner_running.

        Recreates the rings (a leftover arm_target from the dead child would
        otherwise be chased as fresh — plan D5), clears the stale two-phase
        handshake, restarts the child and waits 10s for readiness. Also covers
        a child still connecting when the caller asks early (B-early-press
        protection: refuse TELEOP until the first real qpos readback).
        """
        if self._proc.running:
            if not self.wait_ready(timeout=10.0):
                logger.error("ArmSHMFaçade: child not ready within 10s (arm connection?)")
                return False
            return True
        logger.info("ArmSHMFaçade: child not running — recreating rings and restarting")
        self._unlink_rings()
        self._last_good = None
        try:
            self._create_rings()
        except Exception:
            logger.exception("ArmSHMFaçade: ring recreation failed")
            return False
        self._proc.start()
        if not self.wait_ready(timeout=10.0):
            logger.error("ArmSHMFaçade: child not ready within 10s (arm connection?)")
            return False
        # Suppress no-frame fabricated error for a few ticks after restart:
        # wait_ready() only guarantees the child is alive, not that the inner
        # loop has published its first state frame.  Without this grace period
        # the teleop loop sees error_state=True and may emergency-stop on a
        # transient timing race (plan §4.7 / §11.3).
        self._startup_grace_remaining = 5  # ~312ms at 16Hz — ample for 30Hz poller
        return True

    # ── State (freshness gate + range sanity, plan §4.7) ──

    def get_state(self) -> tuple[np.ndarray | None, int]:
        """Latest arm_state record with guards. Returns (ARM_STATE copy, age_ns).

        * age > state_stale_mult/loop_hz → fabricated error record
          (error_state=1, connected=0, qpos=last-good or zeros) + throttled
          warning, so validate_action trips instead of gating on stale
          tau/temps (the [[l515-midrun-stream-stall]] lesson).
        * non-finite qpos or outside [soft_min-0.05, soft_max+0.05] is treated
          as a torn read → last-good record (or None if none).
        """
        if self._state_ring is None:
            return None, -1
        latest = self._state_ring.read_latest()
        if latest is None:
            if self._startup_grace_remaining > 0:
                self._startup_grace_remaining -= 1
                # Startup grace period: child is alive but hasn't published its
                # first state frame yet.  Return last-good or None instead of
                # fabricating an error — prevents the teleop loop from seeing a
                # false error_state=True and potentially emergency-stopping
                # (plan §11.3).
                if self._last_good is not None:
                    return self._last_good.copy(), -1
                # Fresh instance (e.g. after return_home): no last-good yet,
                # but we must still suppress the error_state=True that the
                # adapter would derive from a None return.  Fabricate a safe
                # empty record so the teleop loop sees error_state=False
                # while the child publishes its first real frame.
                return new_frame(ARM_STATE_DTYPE), -1
            self._throttled_warn("ArmSHMFaçade: no arm_state frame yet — fabricated error record")
            return self._fabricate_error_state(), -1
        self._startup_grace_remaining = 0  # first real frame → end grace period
        data, ts_ns, _seq = latest
        age_ns = max(0, time.monotonic_ns() - ts_ns)

        stale_ns = int(self._config.state_stale_mult / self._config.loop_hz * 1e9)
        if age_ns > stale_ns:
            self._throttled_warn(
                "ArmSHMFaçade: arm_state stale (%.0fms > %.0fms) — fabricated error record " "so validate_action trips",
                age_ns / 1e6,
                stale_ns / 1e6,
            )
            return self._fabricate_error_state(data), age_ns

        qpos = np.asarray(data["qpos"][0], dtype=np.float64)
        if not np.all(np.isfinite(qpos)) or np.any(qpos < self._joint_lo) or np.any(qpos > self._joint_hi):
            self._throttled_warn("ArmSHMFaçade: implausible arm_state qpos (torn read?) — falling back to last-good")
            if self._last_good is not None:
                return self._last_good.copy(), age_ns
            return None, age_ns

        self._last_good = data.copy()
        return data, age_ns

    # ── Target (main → child) ──

    def set_target(self, qpos: np.ndarray | None, producer_id: int = PRODUCER_TELEOP) -> None:
        """Write an arm_target frame. ``None`` → is_hold=1 (hold sentinel)."""
        if self._target_ring is None:
            logger.warning("ArmSHMFaçade.set_target: not started — dropping target")
            return
        frame = new_frame(ARM_TARGET_DTYPE)
        if qpos is None:
            frame["is_hold"][0] = 1
        else:
            frame["target"][0] = np.asarray(qpos, dtype=np.float64).ravel()[:7]
        frame["producer_id"][0] = int(producer_id)
        self._target_ring.write(frame)

    # ── RPC macros (main → child, blocking) ──

    def rpc(self, code: int, **fields: Any) -> np.ndarray:
        """Build an ARM_CMD frame and wait for its ARM_CMD_RESULT.

        Fields: ``waypoints`` (N,7, ≤MAX_ARM_WAYPOINTS — caller segments),
        ``dt``, ``target`` (7,), ``speed``, ``acc`` (0 → child-side XArm7Config
        reset defaults for RESET_BLOCKING).

        Timeout semantics: RpcTimeoutError means the RESULT was not observed
        within ``config.rpc_timeout_s`` — the command is NOT aborted and may
        still be running child-side (a late result is simply discarded).
        Callers must treat a timed-out move as IN FLIGHT: verify arm qpos
        against the intended target before retrying, or a retry dispatches
        the move a second time (the RPC server serves any seq newer than its
        last-served one).
        """
        if self._rpc_client is None:
            raise RuntimeError("ArmSHMFaçade not started")
        frame = new_frame(ARM_CMD_DTYPE)
        frame["cmd"][0] = int(code)
        waypoints = fields.get("waypoints")
        if waypoints is not None:
            wp = np.asarray(waypoints, dtype=np.float64).reshape(-1, 7)
            if wp.shape[0] > MAX_ARM_WAYPOINTS:
                raise ValueError(
                    f"waypoints={wp.shape[0]} exceeds MAX_ARM_WAYPOINTS={MAX_ARM_WAYPOINTS}; "
                    "caller must segment (plan §4.3)"
                )
            frame["n_waypoints"][0] = wp.shape[0]
            frame["waypoints"][0][: wp.shape[0]] = wp
        for key in ("dt", "speed", "acc"):
            if key in fields:
                frame[key][0] = float(fields[key])
        if "target" in fields:
            frame["target"][0] = np.asarray(fields["target"], dtype=np.float64).ravel()[:7]
        return self._rpc_client.call(frame)

    def exec_waypoints(self, waypoints: np.ndarray, dt: float) -> np.ndarray:
        """EXEC_WAYPOINTS — Mode 1 set_servo_angle_j per waypoint (plan §4.3).

        Mode 6 is automatically reconstructed on completion. Blocks until the
        move finishes — a dense 2048-point path at dt=20 ms alone is ~41 s, so
        the default rpc_timeout_s (60 s) barely covers one full segment;
        raise it for slower/longer paths. On RpcTimeoutError the move is NOT
        aborted and may still be running (see rpc()).
        """
        return self.rpc(ARM_CMD_EXEC_WAYPOINTS, waypoints=waypoints, dt=dt)

    def reset_blocking(self, target: np.ndarray, speed: float = 0.0, acc: float = 0.0) -> np.ndarray:
        """RESET_BLOCKING — Mode 0 set_servo_angle(wait=True) (XArm7.reset).

        speed/acc = 0 → XArm7Config reset defaults (child-side: reset_speed
        20°/s). The blocking move can take ~18 s from near a joint soft
        limit; the default rpc_timeout_s (60 s) covers that with margin. On
        RpcTimeoutError the move is NOT aborted — the arm may still be
        moving home; check qpos vs ``target`` before retrying (see rpc()).
        """
        return self.rpc(ARM_CMD_RESET_BLOCKING, target=target, speed=speed, acc=acc)

    def clear_error(self) -> np.ndarray:
        """CLEAR_ERROR — clean_error + motion_enable, no mode change."""
        return self.rpc(ARM_CMD_CLEAR_ERROR)

    def emergency_stop_rpc(self) -> np.ndarray:
        """EMERGENCY_STOP — set_state(4); inner loop stays stopped."""
        return self.rpc(ARM_CMD_EMERGENCY_STOP)

    def reinit_mode6(self) -> np.ndarray:
        """REINIT_MODE6 — clear errors, restart the inner loop in Mode 6."""
        return self.rpc(ARM_CMD_REINIT_MODE6)

    # ── Status ──

    @property
    def running(self) -> bool:
        return self._proc.running

    @property
    def crashed(self) -> bool:
        return self._proc.crashed

    @property
    def config(self) -> ArmProcessConfig:
        return self._config

    # ── Internal helpers ──

    def _create_rings(self) -> None:
        prefix = self._config.shm_prefix
        # stale_cleanup=True: a leftover arm_target from a dead run would be
        # chased as a fresh target on restart — far more dangerous than stale
        # camera frames (camera_process.py:204 pattern, plan §5.1 / D5).
        self._state_ring = SeqlockRingBuffer(
            f"{prefix}_state", ARM_STATE_DTYPE, maxlen=3, create=True, stale_cleanup=True
        )
        self._target_ring = SeqlockRingBuffer(
            f"{prefix}_target", ARM_TARGET_DTYPE, maxlen=2, create=True, stale_cleanup=True
        )
        self._cmd_ring = SeqlockRingBuffer(f"{prefix}_cmd", ARM_CMD_DTYPE, maxlen=2, create=True, stale_cleanup=True)
        self._result_ring = SeqlockRingBuffer(
            f"{prefix}_cmd_result", ARM_CMD_RESULT_DTYPE, maxlen=2, create=True, stale_cleanup=True
        )
        self._rpc_client = RpcClient(self._cmd_ring, self._result_ring, timeout_s=self._config.rpc_timeout_s)

    def _unlink_rings(self) -> None:
        for ring in (self._state_ring, self._target_ring, self._cmd_ring, self._result_ring):
            if ring is None:
                continue
            try:
                ring.close()
            except Exception:
                pass
            try:
                ring.unlink()
            except Exception:
                pass
        self._state_ring = None
        self._target_ring = None
        self._cmd_ring = None
        self._result_ring = None
        self._rpc_client = None

    def _fabricate_error_state(self, donor: np.ndarray | None = None) -> np.ndarray:
        """Fabricated record so validate_action trips: error_state=1, connected=0.

        qpos (and the rest) come from the donor frame or the last-good cache;
        zeros if neither exists.
        """
        frame = new_frame(ARM_STATE_DTYPE)
        src = donor if donor is not None else self._last_good
        if src is not None:
            frame[0] = src[0]
        frame["error_state"][0] = 1
        frame["connected"][0] = 0
        return frame

    def _throttled_warn(self, msg: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last_warn_monotonic >= _WARN_THROTTLE_S:
            self._last_warn_monotonic = now
            logger.warning(msg, *args)


# ═══════════════════════════════════════════════════════════════════
# P1 wiring — ArmInnerLoop-compatible surface over the SHM façade
# ═══════════════════════════════════════════════════════════════════


class ArmServo(Protocol):
    """Structural surface entry points drive the arm servo through (plan §6 P1).

    Satisfied by both the in-process ``ArmInnerLoop`` (today's path) and
    ``ArmInnerLoopSHMAdapter`` (crash-isolated subprocess path), so an entry
    point swaps only its construction site (``make_arm_servo``) — the teleop /
    replay hot loop is byte-identical on both paths.
    """

    def set_target(self, target: np.ndarray | None) -> None: ...
    def get_state(self) -> tuple[np.ndarray, bool, float]: ...
    def get_dynamics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def start(self) -> None: ...
    def stop(self, timeout: float = 3.0) -> None: ...
    def wait_ready(self, timeout: float = 30.0) -> bool: ...
    def emergency_stop(self, settle_timeout: float | None = None) -> bool: ...

    @property
    def is_alive(self) -> bool: ...
    @property
    def last_sent_cmd(self) -> np.ndarray: ...
    @property
    def mode(self) -> int: ...
    @property
    def tracking_error(self) -> float: ...
    @property
    def connected(self) -> bool: ...


class ArmInnerLoopSHMAdapter:
    """Present the ``ArmInnerLoop`` surface over an ``ArmSHMFaçade`` (plan §6 P1).

    Unpacks the façade's policy-shaped ``get_state() -> (record, age_ns)`` into
    ``ArmInnerLoop``'s ``get_state() -> (qpos, error, ts)`` and
    ``get_dynamics() -> (qvel, tau, temps)`` shapes that the entry points
    destructure. ``emergency_stop`` uses the shared estop event (fast path:
    child issues ``set_state(4)`` on its own live connection within ≤1 tick —
    plan §4.8/A5), mirroring ``ArmInnerLoop.emergency_stop``.
    """

    def __init__(self, facade: ArmSHMFaçade, estop_event: Any) -> None:
        self._facade = facade
        self._estop_event = estop_event

    # ── Lifecycle (mirror ArmInnerLoop) ──

    def start(self) -> None:
        self._facade.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._facade.stop(timeout=timeout)

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._facade.wait_ready(timeout)

    def ensure_running(self) -> bool:
        return self._facade.ensure_running()

    @property
    def is_alive(self) -> bool:
        return self._facade.running

    # ── Command (main → child) ──

    def set_target(self, target: np.ndarray | None) -> None:
        self._facade.set_target(target)

    def emergency_stop(self, settle_timeout: float | None = None) -> bool:
        try:
            self._estop_event.set()
            return True
        except Exception:
            logger.exception("ArmInnerLoopSHMAdapter: estop_event.set failed")
            return False

    # ── State readback (unpack policy record → ArmInnerLoop shapes) ──

    def get_state(self) -> tuple[np.ndarray, bool, float]:
        rec, _age_ns = self._facade.get_state()
        if rec is None:
            return nan_array(7), True, time.perf_counter()
        return (
            np.asarray(rec["qpos"][0], dtype=np.float64).copy(),
            bool(rec["error_state"][0]),
            time.perf_counter(),
        )

    def get_dynamics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rec, _age_ns = self._facade.get_state()
        if rec is None:
            return nan_array(7), nan_array(7), nan_array(7)
        return (
            np.asarray(rec["qvel"][0], dtype=np.float64).copy(),
            np.asarray(rec["tau"][0], dtype=np.float64).copy(),
            np.asarray(rec["temps"][0], dtype=np.float64).copy(),
        )

    @property
    def last_sent_cmd(self) -> np.ndarray:
        rec, _ = self._facade.get_state()
        if rec is None:
            return np.zeros(7, dtype=np.float64)
        return np.asarray(rec["last_sent"][0], dtype=np.float64).copy()

    @property
    def mode(self) -> int:
        rec, _ = self._facade.get_state()
        return int(rec["mode"][0]) if rec is not None else 0

    @property
    def tracking_error(self) -> float:
        rec, _ = self._facade.get_state()
        return float(rec["tracking_err"][0]) if rec is not None else 0.0

    @property
    def connected(self) -> bool:
        rec, _ = self._facade.get_state()
        return bool(rec["connected"][0]) if rec is not None else False


def make_arm_servo(
    cfg: ArmInnerLoopConfig,
    ip: str | None = None,
    *,
    use_arm_isolation: bool = False,
    arm_joint_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    process_config: ArmProcessConfig | None = None,
    inner_factory: Callable[[ArmProcessConfig], Any] | None = None,
) -> ArmServo:
    """Build the arm servo: in-process ``ArmInnerLoop`` or isolated subprocess.

    Behind the arm transition flag (``use_arm_isolation`` or env
    ``DEXMANI_PROCESS_ISOLATION=1`` / ``DEXMANI_ARM_PROCESS_ISOLATION=1``)
    returns an ``ArmInnerLoopSHMAdapter`` over an ``ArmSHMFaçade`` (arm inner
    loop runs in a fork child owning the sole XArmAPI connection); otherwise
    the proven in-process ``ArmInnerLoop``. Both satisfy ``ArmServo``, so
    callers swap only this construction site.

    Args:
        cfg: Inner-loop config (forwarded to the child's ArmInnerLoop).
        ip: Arm IP; ``None`` → ArmInnerLoop/XArm7Config default.
        use_arm_isolation: Config half of the arm transition flag (env overrides).
        arm_joint_bounds: (qpos_min_soft, qpos_max_soft) for the façade's
            range-sanity gate; ``None`` → wide ±4π torn-read window.
        process_config: Full override for the subprocess config; when ``None``
            one is built from ``cfg``/``ip`` via ``inner_kwargs``.
        inner_factory: Test injection for the child's inner loop (no hardware).
    """
    if not arm_isolation_enabled(use_arm_isolation):
        from dexmani_real.robot.inner_loop import ArmInnerLoop

        if ip is not None:
            return ArmInnerLoop(cfg=cfg, ip=ip)
        return ArmInnerLoop(cfg=cfg)

    if process_config is None:
        inner_kwargs: dict[str, Any] = {"cfg": cfg}
        if ip is not None:
            inner_kwargs["ip"] = ip
        # Propagate the inner loop's target timeout to the child's target-ring
        # freshness gate — a SECOND independent gate that would otherwise silently
        # override it at the 0.2 s default (slow replays <~5 Hz would then have
        # every arm target dropped as stale and freeze on the subprocess path).
        target_timeout_s = max(float(getattr(cfg, "target_timeout_s", 0.2)), 0.2)
        process_config = ArmProcessConfig(target_timeout_s=target_timeout_s, inner_kwargs=inner_kwargs)
    estop_event = mp.get_context("fork").Event()
    facade = ArmSHMFaçade(
        process_config,
        estop_event=estop_event,
        inner_factory=inner_factory,
        arm_joint_bounds=arm_joint_bounds,
    )
    return ArmInnerLoopSHMAdapter(facade, estop_event)
