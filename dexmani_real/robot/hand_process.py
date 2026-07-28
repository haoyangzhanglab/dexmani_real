"""Hand control process + SHM façade (F1 design — plan §4.4-4.7, §5.2).

Architecture (F1: the clip state machine lives in the FAÇADE; the child is
a stateless executor):

    ┌─────────── main process (16 Hz) ───────────┐
    │ HandSHMFaçade.send_action(qpos_cmd)        │
    │   limit → E3 delta  ←─ state               │   hand_cmd ring
    │   (lifted VERBATIM from XHand.send_action; │ ────────────────► ┌─────────────────────┐
    │    _last_qpos_cmd lives HERE)              │                   │ HandControlProcess  │
    │   → returns (ok, expected_cmd) with ZERO   │   hand_state ring │ 30 Hz, sole XHand   │
    │     wait — record-what-was-sent, zero race │ ◄──────────────── │ connection;         │
    │ HandSHMFaçade.check_echo()                 │  (echo: seq +     │ stateless:          │
    │   seq gap / value mismatch → warn + resync │   value sent)     │ joint-limit net     │
    └────────────────────────────────────────────┘                   │ → hardware send     │
                                                                     │ → state/tactile/echo│
                                             macro RPC (reset/stop/  │ + macro executor    │
                                             clear_error/trajectory) │   (interpolator     │
                                             ◄──────────────────────►│    child-side)      │
                                                                     └─────────────────────┘

Child E3 no-op selection (verified against xhand.py — NO modification made):
    The child's XHand is built with ``dataclasses.replace(hand_config,
    max_delta_rad=0.0)``. The delta clip is a clean no-op:
      - E3: the delta clip is guarded by ``if np.any(limit > 0) and
        self.last_qpos_cmd is not None:`` — max_delta_rad=0.0 skips the clip
        entirely (XHandConfig docstring: "0.0 = disabled").
    The joint-limit ``np.clip`` (``_limit_joint_range``) stays ENABLED in the
    child as the safety net; it echoes the value actually sent. Since xhand.py
    already has the required no-op semantics, it was NOT modified.

Lifecycle (plan §5.2, D4 — worst-case assumption A4: mode 3 + kp=80 position
servo, firmware holds position without command refresh):
    - ``daemon=False`` — the child survives main-process death; firmware holds.
    - cmd ring stale > ``cmd_stale_hold_s`` (main dead) → child stops sending
      (hold position); NEVER zeros torque.
    - orphan exit: ``orphan_exit_s`` with zero new cmd seqs → the main process
      died or exited without calling ``stop()`` (multiprocessing's atexit
      would join this non-daemon child forever) — hold position and exit
      cleanly; firmware holds (A4). NEVER zeros torque.
    - SIGINT (Ctrl-C reaches the process group) → hold and exit; NEVER
      ``tor_max=0`` (``XHand.stop()`` is only called via the estop_event per
      plan §4.8 or an explicit STOP macro — the deliberate detorque actions).
    - estop preemption: macro RPC runs on a dedicated thread under a lock (as
      in the arm child), so a long macro can never defer the tick-loop estop
      check; SEND_TRAJECTORY checks the estop event between steps and aborts
      at the next step boundary (plan §4.8).
    - watchdog: consecutive send errors → ``XHand.reset_connection()``.

Rings (``SeqlockRingBuffer``, names from ``HandProcessConfig.shm_prefix``):
    {prefix}_state (maxlen=3, child→main), {prefix}_cmd (maxlen=2, main→child),
    {prefix}_macro_cmd / {prefix}_macro_result (maxlen=2 each, RPC).

Contract deviations (kept minimal, documented per convention):
    - Macro STOP is exposed as ``HandSHMFaçade.stop_rpc()`` — ``stop(timeout)``
      is the lifecycle method (same precedent as ArmSHMFaçade.emergency_stop_rpc).
      ``rpc(HAND_MACRO_STOP)`` also works.
    - ``HAND_MACRO_CMD_DTYPE`` / ``HAND_MACRO_RESULT_DTYPE`` are defined here
      (plan §4.6 specifies the fields; the shared robot_layouts contract covers
      §4.1-4.5 + §4.10 only).
    - ``get_state()`` never returns None: a stale/missing state yields a
      fabricated ``connected=0`` record (degraded mode, NO escalation — §4.7).


Ref: docs/arm-hand-process-isolation-plan.md §4.4-4.7 (SHM layouts, F1/F2),
     §4.6 (macros), §5.2 (D4 lifecycle), §7 D1 (clip/EMA in main process).
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any, Callable

import numpy as np


from dexmani_real.shm.robot_layouts import (
    HAND_CMD_DTYPE,
    HAND_MACRO_CLEAR_ERROR,
    HAND_MACRO_RESET,
    HAND_MACRO_SEND_TRAJECTORY,
    HAND_MACRO_STOP,
    HAND_STATE_DTYPE,
    MAX_HAND_WAYPOINTS,
    PRODUCER_TELEOP,
    new_frame,
)
from dexmani_real.shm.robot_ring import SeqlockRingBuffer, is_fresh
from dexmani_real.shm.robot_rpc import RpcClient
from dexmani_real.utils.array_utils import nan_array, safe_resize
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.robot.xhand.xhand import XHand, XHandConfig

logger = get_logger(__name__)


# ── hand macro RPC frames (plan §4.6) ──
# RESET(qpos[12]) / STOP / CLEAR_ERROR / SEND_TRAJECTORY(waypoints[256,12],
# duration_s, max_speed); result echoes {cmd_seq, ok, err_code}.
HAND_MACRO_CMD_DTYPE = np.dtype(
    [
        ("cmd", "u4"),  # HAND_MACRO_* codes (robot_layouts)
        ("n_waypoints", "u4"),
        ("waypoints", "<f8", (MAX_HAND_WAYPOINTS, 12)),
        ("qpos", "<f8", (12,)),  # RESET target
        ("duration_s", "<f8"),
        ("max_speed", "<f8"),  # <=0 ⇒ None child-side (XHand default max_qvel.min())
    ]
)

HAND_MACRO_RESULT_DTYPE = np.dtype(
    [
        ("cmd_seq", "u8"),  # RpcServer stamps the cmd ring sequence
        ("ok", "u1"),
        ("err_code", "i8"),
    ]
)

# ── ring geometry (plan §4.4-4.6) ──
_STATE_MAXLEN = 3
_CMD_MAXLEN = 2
_MACRO_MAXLEN = 2
# Echo mismatch tolerance (rad) — the façade's joint-limit clip and the
# child's safety-net clip are numerically identical (same np.clip), but
# floating-point round-trip through the SHM ring produces ~1e-4 rad
# differences.  1e-3 rad ≈ 0.057° is well below the hand's encoder
# resolution and filters out false positives without masking real anomalies.
_ECHO_MISMATCH_TOLERANCE_RAD: float = 1e-3

# Plan §5.1: hand ready wait 15 s; failure → degraded mode (connected=False).
_READY_TIMEOUT_S = 15.0
# Throttled-warning interval (same cadence as SeqlockRingBuffer torn-read warns).
_WARN_INTERVAL_S = 5.0
# Baseline seeding / post-macro resync poll budgets.
_SEED_TIMEOUT_S = 0.5
_MACRO_RESYNC_TIMEOUT_S = 0.5
# Watchdog (plan §5.2): persistent send errors → three-stage reconnect.
_WATCHDOG_RECONNECT_AFTER = 30


@dataclass
class HandProcessConfig:
    """Configuration for HandControlProcess / HandSHMFaçade."""

    hz: float = 30.0  # child loop rate (XHand control rate)
    shm_prefix: str = "dexmani_hand"  # rings: {prefix}_state/_cmd/_macro_cmd/_macro_result
    cmd_stale_hold_s: float = 0.5  # child: cmd ring stale → hold position, NEVER detorque (§5.2)
    state_stale_s: float = 0.1  # façade freshness gate: 3 × hand_dt (§4.7); stale → degraded, no escalation
    rpc_timeout_s: float = 10.0  # macro RPC client timeout
    expected_producer_id: int = PRODUCER_TELEOP  # nonzero mismatch on hand_cmd → reject + warn (D9)
    daemon: bool = False  # plan D4: non-daemon + watchdog + hold-never-detorque
    # Bounded autonomous exit for an orphaned child (daemon=False survives a
    # SIGKILLed main on purpose — but a main-process exit path that skips
    # stop() would otherwise hang interpreter shutdown forever, since
    # multiprocessing's atexit joins non-daemon children): after this many
    # seconds with ZERO new hand_cmd seqs (main dead/exited), the child holds
    # position (firmware holds, A4) and exits cleanly — never detorques.
    # 0 disables the budget (hold forever).
    orphan_exit_s: float = 60.0


class _ThrottledWarn:
    """Logger.warning throttled to at most once per interval (≤1 / 5 s pattern)."""

    def __init__(self, interval_s: float = _WARN_INTERVAL_S) -> None:
        self._interval_ns = int(interval_s * 1e9)
        self._last_ns = 0

    def __call__(self, msg: str, *args: Any) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_ns < self._interval_ns:
            return
        self._last_ns = now_ns
        logger.warning(msg, *args)


class HandControlProcess:
    """Wraps the hand control child (module-level ``_hand_child_main``, fork).

    Owns the SHM rings (created with stale cleanup on ``start()``, unlinked on
    ``stop()``) and the process lifecycle. ``HandSHMFaçade`` composes this
    class; it can also be driven directly.

    The child is a stateless executor at ``config.hz``: on a NEW hand_cmd seq
    only → joint-limit safety-net clip → hardware send → publish HAND_STATE
    with echo (processed ring seq + value actually sent) and full-bandwidth
    tactile; stale cmd ring → hold (never detorque); macros via RpcServer.
    """

    def __init__(
        self,
        config: HandProcessConfig,
        sync: Any,
        estop_event: Any,
        hand_config: "XHandConfig",
        hand_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self._config = config
        self._sync = sync  # API symmetry with the arm process; unused by the hand child
        self._estop_event = estop_event
        self._hand_config = hand_config
        self._hand_factory = hand_factory

        self._ctx = mp.get_context("fork")  # plan A2: SDKs lazily imported inside child run()
        self._process: BaseProcess | None = None  # ForkContext.Process is a BaseProcess
        self._stop_event = self._ctx.Event()
        self._ready_event = self._ctx.Event()
        self._crashed = self._ctx.Event()

        self._state_ring: SeqlockRingBuffer | None = None
        self._cmd_ring: SeqlockRingBuffer | None = None
        self._macro_cmd_ring: SeqlockRingBuffer | None = None
        self._macro_result_ring: SeqlockRingBuffer | None = None
        self._rpc_client: RpcClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Create the rings (stale cleanup, D5) and spawn the child.

        Returns True on success.
        """
        if self.running:
            logger.warning("HandControlProcess already running.")
            return False

        self._stop_event.clear()
        self._ready_event.clear()
        self._crashed.clear()

        prefix = self._config.shm_prefix
        self._state_ring = SeqlockRingBuffer(f"{prefix}_state", HAND_STATE_DTYPE, maxlen=_STATE_MAXLEN, create=True)
        self._cmd_ring = SeqlockRingBuffer(f"{prefix}_cmd", HAND_CMD_DTYPE, maxlen=_CMD_MAXLEN, create=True)
        self._macro_cmd_ring = SeqlockRingBuffer(
            f"{prefix}_macro_cmd", HAND_MACRO_CMD_DTYPE, maxlen=_MACRO_MAXLEN, create=True
        )
        self._macro_result_ring = SeqlockRingBuffer(
            f"{prefix}_macro_result", HAND_MACRO_RESULT_DTYPE, maxlen=_MACRO_MAXLEN, create=True
        )
        self._rpc_client = RpcClient(
            self._macro_cmd_ring, self._macro_result_ring, timeout_s=self._config.rpc_timeout_s
        )

        proc = self._ctx.Process(
            target=_hand_child_main,
            name=f"hand-{prefix}",
            daemon=self._config.daemon,
            args=(
                self._config,
                self._sync,
                self._estop_event,
                self._hand_config,
                self._stop_event,
                self._ready_event,
                self._crashed,
                self._hand_factory,
            ),
        )
        self._process = proc
        proc.start()
        logger.info(
            "HandControlProcess started (prefix=%s, hz=%.0f, daemon=%s).",
            prefix,
            self._config.hz,
            self._config.daemon,
        )
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Signal stop, join (terminate as fallback), and unlink all rings.

        Hold semantics: stopping never detorques — the child exits without
        touching torque and the firmware position servo holds (A4).
        """
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("HandControlProcess did not exit within %.1fs, terminating.", timeout)
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._process = None

        for ring in (self._state_ring, self._cmd_ring, self._macro_cmd_ring, self._macro_result_ring):
            if ring is None:
                continue
            try:
                ring.close()
                ring.unlink()
            except (FileNotFoundError, BufferError, OSError):
                pass
        self._state_ring = None
        self._cmd_ring = None
        self._macro_cmd_ring = None
        self._macro_result_ring = None
        self._rpc_client = None
        logger.info("HandControlProcess stopped.")

    def wait_ready(self, timeout: float) -> bool:
        """Wait for the child's ready event (XHand connect complete)."""
        return self._ready_event.wait(timeout)

    # ------------------------------------------------------------------
    # Observability + ring access (used by HandSHMFaçade)
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def crashed(self) -> bool:
        """True if the child died unexpectedly or reported a fatal failure."""
        if self._process is not None and not self._process.is_alive():
            if self._crashed.is_set() or (self._process.exitcode not in (0, None)):
                self._crashed.set()
        return self._crashed.is_set()

    @property
    def state_ring(self) -> SeqlockRingBuffer | None:
        return self._state_ring

    @property
    def cmd_ring(self) -> SeqlockRingBuffer | None:
        return self._cmd_ring

    @property
    def rpc_client(self) -> RpcClient | None:
        return self._rpc_client


class HandSHMFaçade:
    """Main-process façade for the hand control child (F1).

    Owns the E3 clip state machine lifted verbatim from
    ``XHand.send_action`` (``_last_qpos_cmd`` lives here).
    ``send_action()`` returns ``(ok, expected_cmd)`` with ZERO wait —
    ``expected_cmd`` is exactly what recording stores and what the hold
    baseline follows (record-what-was-sent, zero race). The child echoes
    ``(last_cmd_seq, value actually sent)``; ``check_echo()`` verifies it and
    resyncs the baseline on anomalies.
    """

    def __init__(
        self,
        config: HandProcessConfig,
        sync: Any,
        estop_event: Any,
        hand_config: "XHandConfig",
        hand_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self._config = config
        self._hand_config = hand_config
        self._proc = HandControlProcess(config, sync, estop_event, hand_config, hand_factory)

        # ── E3 state machine (verbatim from XHand.send_action) ──
        self._last_qpos_cmd: np.ndarray | None = None  # E3 delta baseline; hold baseline; record-what-was-sent
        self._last_joint_limit_clipped = False

        # ── echo verification (F1) ──
        self._last_acked_seq = 0
        self._expected_by_seq: dict[int, np.ndarray] = {}
        self._echo_warn = _ThrottledWarn()
        self._stale_warn = _ThrottledWarn()
        self._last_good_state: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Create rings (stale cleanup) + start the child. Returns True on success."""
        # Fresh session state so a restart never reuses a stale baseline/ack.
        self._last_qpos_cmd = None
        self._last_joint_limit_clipped = False
        self._last_acked_seq = 0
        self._expected_by_seq.clear()
        self._last_good_state = None
        return self._proc.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the child and unlink the rings (hold — never detorque)."""
        self._proc.stop(timeout)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait for the child to connect; on success seed the E3 baseline.

        Seeding mirrors ``XHand.connect`` → ``_init_hand_state``: the baseline
        anchors to the child's initial hardware read (fallback home_qpos) so
        the first delta clip is relative to the actual hand position.
        """
        ok = self._proc.wait_ready(timeout if timeout is not None else _READY_TIMEOUT_S)
        if ok:
            self._seed_baseline()
        return ok

    @property
    def running(self) -> bool:
        return self._proc.running

    @property
    def crashed(self) -> bool:
        return self._proc.crashed

    @property
    def config(self) -> HandProcessConfig:
        return self._config

    # ------------------------------------------------------------------
    # State (child → main)
    # ------------------------------------------------------------------

    def get_state(self) -> tuple[np.ndarray, int]:
        """Read the latest HAND_STATE record. Returns (record copy, age_ns).

        Freshness gate (plan §4.7): state older than ``state_stale_s`` (or
        never written) → fabricated ``connected=0`` record (qpos from the
        last-good frame if any) + throttled warning. Hand staleness degrades
        to arm-only — NO emergency-stop escalation.
        """
        ring = self._proc.state_ring
        if ring is None:
            return self._fabricate_disconnected(), -1
        result = ring.read_latest()
        if result is None:
            return self._fabricate_disconnected(), -1
        data, ts_ns, _seq = result
        now_ns = time.monotonic_ns()
        age_ns = now_ns - ts_ns if ts_ns > 0 else -1
        if not is_fresh(ts_ns, self._config.state_stale_s, now_ns):
            self._stale_warn(
                "HandSHMFaçade: state stale (%.0f ms > %.0f ms) — returning disconnected record (no escalation).",
                age_ns / 1e6,
                self._config.state_stale_s * 1e3,
            )
            return self._fabricate_disconnected(), age_ns
        self._last_good_state = data
        return data, age_ns

    def _fabricate_disconnected(self) -> np.ndarray:
        """Degraded-mode record: connected=0, error_state=1, last-good qpos."""
        rec = new_frame(HAND_STATE_DTYPE)
        if self._last_good_state is not None:
            rec["qpos"] = self._last_good_state["qpos"]
            rec["last_qpos_cmd"] = self._last_good_state["last_qpos_cmd"]
        rec["connected_flag"] = 0
        rec["error_state"] = 1
        return rec

    # ------------------------------------------------------------------
    # Command (main → child) — F1 core
    # ------------------------------------------------------------------

    def send_action(self, qpos_cmd: np.ndarray, producer_id: int = PRODUCER_TELEOP) -> tuple[bool, np.ndarray]:
        """Apply limit → E3 delta and publish to the hand_cmd ring.

        The clip state machine below is lifted VERBATIM from
        ``XHand.send_action`` (xhand.py), including first-command semantics:
        E3 skips while ``_last_qpos_cmd`` is None.
        Writes HAND_CMD(expected_cmd, producer_id) and
        returns ``(ok, expected_cmd)`` immediately — zero wait, zero race.
        ``expected_cmd`` is what recording and the hold baseline use
        (record-what-was-sent).
        """
        target_qpos = safe_resize(qpos_cmd, 12)
        expected_cmd = self._limit_joint_range(target_qpos)

        # ── E3: Delta jump limit ──
        # Hard safety gate: per-step change never exceeds max_delta_rad on any
        # joint.  Complements dex_retargetingʼs LPFilter.
        # Supports per-joint limits: pass a (12,) ndarray for joint-specific caps.
        limit = np.broadcast_to(np.asarray(self._hand_config.max_delta_rad), (12,))
        if np.any(limit > 0) and self._last_qpos_cmd is not None:
            delta = expected_cmd - self._last_qpos_cmd
            delta = np.clip(delta, -limit, limit)
            expected_cmd = self._last_qpos_cmd + delta

        ok = False
        ring = self._proc.cmd_ring
        if ring is not None and self._proc.running:
            try:
                frame = new_frame(HAND_CMD_DTYPE)
                frame["qpos_cmd"] = expected_cmd
                frame["producer_id"] = producer_id
                seq = ring.write(frame)
                self._expected_by_seq[seq] = expected_cmd.copy()
                # Keep the pending map bounded (FILO: only the latest few can be acked).
                stale = [s for s in self._expected_by_seq if s < seq - 16]
                for s in stale:
                    del self._expected_by_seq[s]
                ok = True
            except (OSError, ValueError):
                logger.warning("HandSHMFaçade: hand_cmd ring write failed.", exc_info=True)

        # Verbatim XHand semantics: last_qpos_cmd advances only on a successful
        # send; the EMA bookkeeping above advances regardless.
        if ok:
            self._last_qpos_cmd = expected_cmd.copy()
        return ok, expected_cmd

    @property
    def last_qpos_cmd(self) -> np.ndarray | None:
        """The clip/EMA-processed value last sent (hold baseline / record-what-was-sent)."""
        return None if self._last_qpos_cmd is None else self._last_qpos_cmd.copy()

    def _limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        """Verbatim lift of ``XHand._limit_joint_range`` (xhand.py)."""
        if not self._hand_config.clip_joint_limit:
            self._last_joint_limit_clipped = False
            return qpos

        clipped = np.clip(qpos, self._hand_config.qpos_min, self._hand_config.qpos_max)
        max_deviation = float(np.max(np.abs(qpos - clipped)))
        self._last_joint_limit_clipped = max_deviation > self._hand_config.clip_report_tolerance
        return clipped

    # ------------------------------------------------------------------
    # Echo verification (F1)
    # ------------------------------------------------------------------

    def check_echo(self) -> None:
        """Non-blocking echo verification against the child's HAND_STATE.

        Tracks ``last_acked_seq``; a child echo seq gap (FILO dropped commands)
        or a value mismatch (tolerance 1e-3 rad — the child re-clips with
        identical joint limits, but floating-point round-trip through the
        SHM ring produces ~1e-4 rad differences) → throttled warning +
        baseline resync
        (``_last_qpos_cmd`` ← echo value). All anomalies
        converge into a single detectable event.
        """
        ring = self._proc.state_ring
        if ring is None:
            return
        result = ring.read_latest()
        if result is None:
            return
        data, _ts_ns, _seq = result
        echo_seq = int(data["last_cmd_seq"][0])
        if echo_seq <= self._last_acked_seq:
            return  # no new echo (or ring recycled after a restart)

        echo_val = np.array(data["last_qpos_cmd"][0], dtype=np.float64)
        anomaly = False

        if self._last_acked_seq > 0 and echo_seq > self._last_acked_seq + 1:
            self._echo_warn(
                "HandSHMFaçade: echo seq gap %d → %d (FILO dropped %d cmd(s)) — resyncing baseline.",
                self._last_acked_seq,
                echo_seq,
                echo_seq - self._last_acked_seq - 1,
            )
            anomaly = True

        expected = self._expected_by_seq.get(echo_seq)
        if expected is not None:
            max_diff = float(np.max(np.abs(expected - echo_val)))
            if max_diff > _ECHO_MISMATCH_TOLERANCE_RAD:
                self._echo_warn(
                    "HandSHMFaçade: echo value mismatch at seq=%d (max diff %.3g > %.3g rad) — resyncing baseline.",
                    echo_seq,
                    max_diff,
                    _ECHO_MISMATCH_TOLERANCE_RAD,
                )
                anomaly = True

        if anomaly:
            self._last_qpos_cmd = echo_val.copy()

        self._last_acked_seq = echo_seq
        if self._expected_by_seq:
            for s in [s for s in self._expected_by_seq if s <= echo_seq]:
                del self._expected_by_seq[s]

    # ------------------------------------------------------------------
    # Baseline seeding / macro handover
    # ------------------------------------------------------------------

    def _seed_baseline(self) -> None:
        """Anchor the E3 baseline to the child's initial echo (plan §4.5).

        Mirrors ``XHand._init_hand_state``: poll briefly for the first
        HAND_STATE (published within one child tick of ready); fall back to
        home_qpos exactly like XHand does when no valid hardware read exists.
        ``_last_qpos_cmd`` is left untouched (None ⇒ delta clip skips the first command —
        verbatim XHand first-command semantics).
        """
        if self._last_qpos_cmd is not None:
            return
        ring = self._proc.state_ring
        if ring is not None:
            deadline = time.monotonic() + _SEED_TIMEOUT_S
            while time.monotonic() < deadline and self._proc.running:
                result = ring.read_latest()
                if result is not None:
                    echo_val = np.array(result[0]["last_qpos_cmd"][0], dtype=np.float64)
                    if np.all(np.isfinite(echo_val)):
                        self._last_qpos_cmd = echo_val.copy()
                        logger.info("HandSHMFaçade: E3 baseline seeded from child echo.")
                        return
                time.sleep(0.01)
        self._last_qpos_cmd = safe_resize(self._hand_config.home_qpos, 12).copy()
        logger.warning("HandSHMFaçade: no child echo at ready — E3 baseline seeded from home_qpos.")

    def _resync_baseline_after_macro(self) -> None:
        """State-machine handover at macro end (plan §4.5).

        A macro (reset/trajectory) moved the hand child-side, outside the
        façade's state machine; wait for the child's post-macro state
        publication and resync the E3 baseline so the next teleop delta
        clip anchors to the actual position.
        """
        ring = self._proc.state_ring
        if ring is None:
            return
        ts_floor = time.monotonic_ns()  # the RPC already returned ⇒ macro complete; next publish is post-macro
        deadline = time.monotonic() + _MACRO_RESYNC_TIMEOUT_S
        while time.monotonic() < deadline:
            result = ring.read_latest()
            if result is not None and result[1] >= ts_floor:
                echo_val = np.array(result[0]["last_qpos_cmd"][0], dtype=np.float64)
                if np.all(np.isfinite(echo_val)):
                    self._last_qpos_cmd = echo_val.copy()
                return
            time.sleep(0.005)
        logger.warning(
            "HandSHMFaçade: no post-macro state within %.1fs — baseline resync skipped.", _MACRO_RESYNC_TIMEOUT_S
        )

    # ------------------------------------------------------------------
    # Macro RPC (plan §4.6)
    # ------------------------------------------------------------------

    def rpc(self, code: int, **fields: Any) -> np.ndarray:
        """Build a HAND_MACRO_CMD frame and issue a blocking RpcClient.call.

        Fields: ``qpos`` (RESET target), ``waypoints`` ((N<=256, 12)),
        ``duration_s``, ``max_speed`` (None ⇒ <=0 sentinel ⇒ XHand default).
        Returns the HAND_MACRO_RESULT record; raises RpcTimeoutError on
        timeout, RuntimeError if not started.
        """
        client = self._proc.rpc_client
        if client is None:
            raise RuntimeError("HandSHMFaçade not started (no RPC client).")

        frame = new_frame(HAND_MACRO_CMD_DTYPE)
        frame["cmd"] = code
        qpos = fields.get("qpos")
        if qpos is not None:
            frame["qpos"] = safe_resize(qpos, 12)
        waypoints = fields.get("waypoints")
        if waypoints is not None:
            wps = np.asarray(waypoints, dtype=np.float64)
            if wps.ndim == 1:
                wps = wps.reshape(1, 12)
            if wps.ndim != 2 or wps.shape[1] != 12 or wps.shape[0] > MAX_HAND_WAYPOINTS:
                raise ValueError(f"send_trajectory waypoints must be (<= {MAX_HAND_WAYPOINTS}, 12), got {wps.shape}")
            frame["n_waypoints"] = wps.shape[0]
            frame["waypoints"][0, : wps.shape[0]] = wps
        if "duration_s" in fields:
            frame["duration_s"] = float(fields["duration_s"])
        if fields.get("max_speed") is not None:
            frame["max_speed"] = float(fields["max_speed"])
        return client.call(frame)

    def reset(self, qpos: np.ndarray) -> np.ndarray:
        """RESET macro — drive to ``qpos`` (child-side, then baseline resync)."""
        result = self.rpc(HAND_MACRO_RESET, qpos=qpos)
        self._resync_baseline_after_macro()
        return result

    def stop_rpc(self) -> np.ndarray:
        """STOP macro — the deliberate detorque (mirrors RobotInterface.stop).

        Named ``stop_rpc`` because ``stop(timeout)`` is the lifecycle method
        (same precedent as ArmSHMFaçade.emergency_stop_rpc).
        """
        return self.rpc(HAND_MACRO_STOP)

    def clear_error(self) -> np.ndarray:
        """CLEAR_ERROR macro."""
        return self.rpc(HAND_MACRO_CLEAR_ERROR)

    def send_trajectory(
        self,
        waypoints: np.ndarray,
        duration_s: float,
        max_speed: float | None = None,
    ) -> np.ndarray:
        """SEND_TRAJECTORY macro — interpolated child-side (MotorTrajectoryInterpolator)."""
        result = self.rpc(
            HAND_MACRO_SEND_TRAJECTORY,
            waypoints=waypoints,
            duration_s=duration_s,
            max_speed=max_speed,
        )
        self._resync_baseline_after_macro()
        return result


# ----------------------------------------------------------------------
# Child entry point (module-level for fork + pickling)
# ----------------------------------------------------------------------


def _publish_hand_state(hand: Any, frame: np.ndarray, state_ring: SeqlockRingBuffer, last_cmd_seq: int) -> None:
    """Publish one HAND_STATE frame: state + tactile + echo (child-side)."""
    try:
        st = hand.get_state(full=True, force_update=True)
    except Exception:
        logger.warning("hand child: get_state failed.", exc_info=True)
        st = None
    if st is not None:
        frame["qpos"][0] = np.asarray(st["qpos"], dtype=np.float64)
        frame["current"][0] = np.asarray(st["current"], dtype=np.float64)
        frame["temperature"][0] = np.full(12, np.nan, dtype=np.float64)
        frame["tactile_sum"][0] = np.asarray(st["tactile_force_sum"], dtype=np.float64)
        frame["tactile_force"][0] = np.asarray(st["tactile_force"], dtype=np.float64)
        frame["tactile_contact"][0] = np.asarray(st.get("tactile_contact", np.zeros(5, dtype=bool)), dtype=bool)
        frame["tipboard_err"][0] = np.asarray(st.get("tipboard_err", np.zeros(12, dtype=np.int32)), dtype=np.int32)

    lqc = hand.last_qpos_cmd
    if lqc is not None:
        frame["last_qpos_cmd"][0] = np.asarray(lqc, dtype=np.float64)
    frame["last_cmd_seq"][0] = last_cmd_seq
    frame["connected_flag"][0] = int(bool(hand.connected_flag))
    frame["error_state"][0] = int(bool(hand.error_state))
    frame["consecutive_errs"][0] = int(hand.consecutive_send_errors)
    frame["last_error_code"][0] = int(hand.last_error_code if hand.last_error_code is not None else 0)
    frame["limit_clipped"][0] = int(bool(hand.last_joint_limit_clipped))
    try:
        state_ring.write(frame)
    except (OSError, ValueError):
        logger.warning("hand child: state ring write failed.", exc_info=True)


def _hand_child_main(
    config: HandProcessConfig,
    sync: Any,  # noqa: ARG001 — API symmetry with the arm child; unused here
    estop_event: Any,
    hand_config: "XHandConfig",
    stop_event: Any,
    ready_event: Any,
    crashed_event: Any,
    hand_factory: Callable[[Any], Any] | None,
) -> None:
    """Hand control child main loop (runs in the forked process).

    Stateless executor (F1): on a NEW hand_cmd seq only → joint-limit
    safety-net clip (XHand._limit_joint_range; E3 disabled — see module
    docstring) → hardware send → publish HAND_STATE with echo
    (last_cmd_seq = ring seq processed, last_qpos_cmd = value actually sent,
    limit_clipped) + full-bandwidth tactile. Stale cmd ring → hold position,
    NEVER detorque. SIGINT → hold + exit, never tor_max=0. Macros execute via
    RpcServer on a DEDICATED thread (like the arm child) under ``macro_lock``
    — a long macro can never defer the tick loop's estop check;
    SEND_TRAJECTORY aborts at the next step boundary on estop (plan §4.8).
    """
    import inspect
    import signal
    import threading

    from dexmani_real.robot.xhand.xhand import XHand
    from dexmani_real.shm.robot_rpc import RpcServer

    # SIGINT (Ctrl-C reaches the whole process group): hold and exit. The loop
    # break ends the child without touching torque — NEVER detorque on Ctrl-C
    # (plan §5.2); the firmware position servo holds the hand (A4). A
    # threading.Event (RLock-based) is signal-handler safe — the mp semaphore
    # behind stop_event is not reentrant if the handler interrupts a blocked
    # is_set()/wait() on the same thread (same pattern as _arm_child_main).
    sigint_received = threading.Event()

    try:
        signal.signal(signal.SIGINT, lambda *_: sigint_received.set())
    except (ValueError, OSError):
        pass  # not in the main thread — should not happen for the fork child

    prefix = config.shm_prefix
    state_ring = cmd_ring = macro_cmd_ring = macro_result_ring = None
    rpc_thread = None
    try:
        state_ring = SeqlockRingBuffer(f"{prefix}_state", HAND_STATE_DTYPE, maxlen=_STATE_MAXLEN, create=False)
        cmd_ring = SeqlockRingBuffer(f"{prefix}_cmd", HAND_CMD_DTYPE, maxlen=_CMD_MAXLEN, create=False)
        macro_cmd_ring = SeqlockRingBuffer(
            f"{prefix}_macro_cmd", HAND_MACRO_CMD_DTYPE, maxlen=_MACRO_MAXLEN, create=False
        )
        macro_result_ring = SeqlockRingBuffer(
            f"{prefix}_macro_result", HAND_MACRO_RESULT_DTYPE, maxlen=_MACRO_MAXLEN, create=False
        )

        # Child XHand with E3 DISABLED — the façade owns the clip state
        # machine (F1). max_delta_rad=0.0 is a verified clean no-op in
        # XHand.send_action (module docstring); the joint-limit clip
        # stays enabled as the safety net.
        child_cfg = dataclasses.replace(hand_config, max_delta_rad=0.0)
        hand = hand_factory(child_cfg) if hand_factory is not None else XHand(child_cfg)

        if not hand.connect():
            logger.error("hand child: XHand connect failed — degraded mode (hand offline).")
            crashed_event.set()
            return

        # ── Move hand to home_qpos after connection ──
        # After connect(), the hand holds whatever position it was at.
        # Explicitly drive it to home_qpos and wait for it to settle before
        # signalling ready — no teleop commands should arrive while the hand
        # is still in transit from an arbitrary starting pose.
        #
        # Compare against last_qpos_cmd (the actual post-clip command sent
        # by send_action) rather than raw home_qpos: send_action applies
        # _limit_joint_range internally, so a future config change that puts
        # home_qpos outside joint limits won't silently break the settle
        # check with a false-positive timeout.
        logger.info("hand child: resetting to home_qpos...")
        if not hand.reset():
            logger.warning("hand child: reset() to home_qpos failed — proceeding anyway.")
        else:
            # reset() succeeded → last_qpos_cmd is the post-clip target.
            _home = hand.last_qpos_cmd.copy() if hand.last_qpos_cmd is not None else np.asarray(hand_config.home_qpos, dtype=np.float64)
            _settle_deadline = time.monotonic() + 3.0
            _settled = False
            _max_err = float("nan")
            while time.monotonic() < _settle_deadline:
                _st = hand.get_state(force_update=True)
                _qpos = np.asarray(_st.get("qpos", np.zeros(12)), dtype=np.float64)
                if np.all(np.isfinite(_qpos)):
                    _max_err = float(np.max(np.abs(_qpos - _home)))
                    if _max_err < 0.10:  # ~5.7° — close enough to home
                        logger.info("hand child: reached home_qpos (max_err=%.3f rad).", _max_err)
                        _settled = True
                        break
                time.sleep(0.05)
            if not _settled:
                logger.warning(
                    "hand child: home_qpos settle timeout (%.1f s, max_err=%.3f rad) — proceeding anyway.",
                    3.0,
                    _max_err,
                )

        # Serializes XHand access between the tick loop (send/stop/publish)
        # and the RPC thread (macros). A macro can never hold it for more
        # than ~one trajectory step after an estop: SEND_TRAJECTORY checks
        # estop_event between steps and aborts (plan §4.8).
        macro_lock = threading.Lock()
        # abort_event support in send_trajectory: the real XHand accepts it;
        # minimal test fakes may not — detect once, stay compatible with both.
        _traj_accepts_abort = "abort_event" in inspect.signature(hand.send_trajectory).parameters

        def handle_macro(request: np.ndarray, seq: int) -> np.ndarray:
            result = new_frame(HAND_MACRO_RESULT_DTYPE)
            code = int(request["cmd"][0])
            ok = False
            # Runs on the RPC thread; the lock makes the macro mutually
            # exclusive with the tick loop's cmd stream (§4.5 state-machine
            # handover at macro start/end).
            with macro_lock:
                if code == HAND_MACRO_RESET:
                    ok = hand.reset(np.array(request["qpos"][0], dtype=np.float64))
                elif code == HAND_MACRO_STOP:
                    ok = hand.stop()  # deliberate detorque (explicit macro only)
                elif code == HAND_MACRO_CLEAR_ERROR:
                    ok = hand.clear_error()
                elif code == HAND_MACRO_SEND_TRAJECTORY:
                    n = max(0, min(int(request["n_waypoints"][0]), MAX_HAND_WAYPOINTS))
                    duration_s = float(request["duration_s"][0])
                    max_speed = float(request["max_speed"][0])
                    if n > 0:
                        wps = np.array(request["waypoints"][0][:n], dtype=np.float64)
                        ms = max_speed if max_speed > 0 else None
                        if _traj_accepts_abort:
                            # Abort at step boundaries on estop (plan §4.8) —
                            # releases the lock so the tick loop can detorque.
                            ok = hand.send_trajectory(wps, duration_s, max_speed=ms, abort_event=estop_event)
                        else:
                            ok = hand.send_trajectory(wps, duration_s, max_speed=ms)
                else:
                    logger.warning("hand child: unknown macro cmd=%d (seq=%d).", code, seq)
            result["ok"][0] = int(bool(ok))
            result["err_code"][0] = int(hand.last_error_code if hand.last_error_code is not None else 0)
            return result

        rpc_server = RpcServer(macro_cmd_ring, macro_result_ring, handle_macro)

        interval = 1.0 / config.hz
        frame = new_frame(HAND_STATE_DTYPE)
        last_processed_seq = 0
        estopped = False
        stale_warn = _ThrottledWarn()
        producer_warn = _ThrottledWarn()
        watchdog_warn = _ThrottledWarn()
        stale_budget_ns = int(config.cmd_stale_hold_s * 1e9)
        last_ts = time.monotonic()
        # Orphan budget (F4): last time ANY new hand_cmd seq was observed.
        last_new_cmd_monotonic = time.monotonic()

        # RPC macros on their own thread (exactly like the arm child): a
        # blocking macro (SEND_TRAJECTORY runs for its full duration) must
        # not stop the tick loop checking estop / publishing state.
        def _rpc_loop() -> None:
            while not stop_event.is_set():
                try:
                    if rpc_server.handle_pending():
                        # Publish immediately so the façade can resync its
                        # baseline from the post-macro echo.
                        with macro_lock:
                            _publish_hand_state(hand, frame, state_ring, last_processed_seq)
                    else:
                        time.sleep(0.005)
                except Exception:
                    logger.warning("hand child: RPC loop iteration failed.", exc_info=True)
                    time.sleep(0.05)

        rpc_thread = threading.Thread(target=_rpc_loop, name="hand_rpc", daemon=True)
        rpc_thread.start()

        ready_event.set()
        logger.info("hand child ready @ %.0f Hz (prefix=%s).", config.hz, prefix)

        while not stop_event.is_set():
            # 1. E-stop checked FIRST (plan §4.8): hand → stop() once (the
            #    deliberate detorque), then keep publishing state. The RPC
            #    thread holds macro_lock for at most ~one trajectory step
            #    after estop (SEND_TRAJECTORY aborts between steps).
            if estop_event.is_set():
                with macro_lock:
                    if not estopped:
                        try:
                            hand.stop()
                        except Exception:
                            logger.warning("hand child: estop stop() failed.", exc_info=True)
                        estopped = True
                    _publish_hand_state(hand, frame, state_ring, last_processed_seq)
            else:
                estopped = False

                # 2. Stale cmd ring (main dead/stalled) → hold; NEVER detorque.
                if cmd_ring.latest_sequence > 0 and cmd_ring.frame_age_ns() > stale_budget_ns:
                    stale_warn(
                        "hand child: cmd ring stale (%.0f ms > %.0f ms) — holding position (never detorque).",
                        cmd_ring.frame_age_ns() / 1e6,
                        config.cmd_stale_hold_s * 1e3,
                    )

                # 3. hand_cmd — send on NEW seq only (position servo needs no
                #    refresh to hold).
                res = cmd_ring.read_latest()
                if res is not None:
                    data, ts_ns, seq = res
                    if seq != last_processed_seq:
                        # Any new seq proves the main process is alive —
                        # refresh the orphan budget (even if rejected below).
                        last_new_cmd_monotonic = time.monotonic()
                        if not is_fresh(ts_ns, config.cmd_stale_hold_s):
                            stale_warn("hand child: new cmd seq=%d is stale — holding position (never detorque).", seq)
                        else:
                            pid = int(data["producer_id"][0])
                            if pid != 0 and pid != config.expected_producer_id:
                                producer_warn(
                                    "hand child: rejecting cmd seq=%d from producer %d (expected %d).",
                                    seq,
                                    pid,
                                    config.expected_producer_id,
                                )
                            else:
                                qpos_cmd = np.array(data["qpos_cmd"][0], dtype=np.float64)
                                with macro_lock:
                                    try:
                                        # Joint-limit safety-net clip inside
                                        # send_action (E3 disabled — no-op).
                                        hand.send_action(qpos_cmd)
                                    except Exception:
                                        logger.warning("hand child: send_action failed.", exc_info=True)
                        last_processed_seq = seq

                # 4. Watchdog (plan §5.2): persistent send errors → three-stage
                #    reconnect (XHand.reset_connection).
                if hand.consecutive_send_errors >= _WATCHDOG_RECONNECT_AFTER:
                    watchdog_warn(
                        "hand child: %d consecutive send errors — resetting connection.",
                        hand.consecutive_send_errors,
                    )
                    with macro_lock:
                        try:
                            hand.reset_connection()
                        except Exception:
                            logger.warning("hand child: reset_connection failed.", exc_info=True)

                # 5. Publish state + echo + tactile every tick.
                with macro_lock:
                    _publish_hand_state(hand, frame, state_ring, last_processed_seq)

            # 6. Orphan exit (daemon=False, plan §5.2 / D4): no new cmd seqs
            #    for orphan_exit_s → the main process died or exited without
            #    calling stop(). multiprocessing's atexit joins non-daemon
            #    children, so without this the interpreter shutdown would hang
            #    forever on this loop. Hold position (firmware holds, A4) and
            #    exit cleanly — NEVER detorque.
            if config.orphan_exit_s > 0 and time.monotonic() - last_new_cmd_monotonic > config.orphan_exit_s:
                logger.info(
                    "hand child: no new cmd seqs for %.0f s — main process exited without stop(); "
                    "holding position (firmware, A4) and exiting cleanly (never detorque).",
                    config.orphan_exit_s,
                )
                break

            # SIGINT → hold last position and exit cleanly — NEVER detorque
            # (plan §5.2; the finally block does not call hand.stop()).
            if sigint_received.is_set():
                logger.info("hand child: SIGINT received — holding last position and exiting.")
                break

            # Maintain target rate.
            elapsed = time.monotonic() - last_ts
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_ts = time.monotonic()

    except Exception:
        logger.exception("hand child crashed.")
        crashed_event.set()
    finally:
        # Hold semantics: exit without detorque — hand.stop() is NEVER called
        # here (plan §5.2). Drain the RPC thread (bounded — a macro without
        # estop may still be running; it is a daemon and dies with us), then
        # close the ring handles; the parent unlinks.
        if rpc_thread is not None:
            rpc_thread.join(timeout=0.5)
        for ring in (state_ring, cmd_ring, macro_cmd_ring, macro_result_ring):
            if ring is None:
                continue
            try:
                ring.close()
            except (BufferError, OSError):
                pass
        logger.info("hand child exited (prefix=%s).", prefix)


# ----------------------------------------------------------------------
# P1/P2 wiring — XHand-compatible surface over the SHM façade
# ----------------------------------------------------------------------


class HandSHMAdapter:
    """Present the ``XHand`` duck-type over a ``HandSHMFaçade`` (plan §6 P1/P2).

    Lets ``RobotInterface`` swap in-process ``XHand`` for the crash-isolated
    hand subprocess without changing any hand call site: it exposes the exact
    surface ``RobotInterface`` + ``validate_action`` touch —
    ``connect/disconnect/is_connected/is_error/clear_error/stop(detorque)/
    reset(no-arg)/get_state(dict)/send_action(->bool)`` plus the
    ``connected_flag`` / ``error_state`` / ``last_qpos_cmd`` / ``config``
    attributes ``validate_action`` reads directly.

    Command-side E3 delta clip lives in the façade (main process); the
    child XHand is built stateless (``max_delta_rad=0``) and
    echoes what it actually sent. ``fingertip_pos`` is NOT carried by the SHM
    hand state — it stays main-side FK in ``RobotInterface`` (needs arm EEF pose).

    Semantics preserved vs ``XHand``:
      * ``stop()`` is the deliberate detorque (mirrors ``XHand.stop`` —
        de-energize + set ``error_state=True`` so ``is_connected()`` goes False
        until ``clear_error``); lifecycle shutdown is ``disconnect()`` (hold,
        never detorque).
      * ``get_state()`` returns the XHand dict shape (``qpos`` required key,
        ``tactile_force_sum`` / ``tactile_force`` optional); a stale/missing SHM
        record yields NaN qpos + zero tactile, matching XHand's read-error
        contract that ``RobotInterface.get_state`` wraps in try/except.
    """

    def __init__(self, facade: HandSHMFaçade, hand_config: XHandConfig, estop_event: Any) -> None:
        self._facade = facade
        self.config = hand_config  # validate_action reads config.qpos_min / qpos_max
        self._estop_event = estop_event
        self.last_qpos_cmd: np.ndarray | None = None
        # Live status attributes read directly by validate_action — refreshed
        # from the SHM record on every status query / get_state / send_action.
        self.connected_flag: bool = False
        self.error_state: bool = False

    # ── internal ──

    def _refresh_status(self, rec: np.ndarray | None) -> None:
        if rec is None:
            self.connected_flag = False
            self.error_state = True
            return
        self.connected_flag = bool(rec["connected_flag"][0])
        self.error_state = bool(rec["error_state"][0])
        self.last_qpos_cmd = np.asarray(rec["last_qpos_cmd"][0], dtype=np.float64).copy()

    # ── Lifecycle ──

    def connect(self) -> bool:
        try:
            self._estop_event.clear()  # fresh session — drop any prior detorque latch
            if not self._facade.start():
                return False
            ready = self._facade.wait_ready()
            rec, _age = self._facade.get_state()
            self._refresh_status(rec)
            return bool(ready)
        except Exception as e:
            logger.warning("HandSHMAdapter.connect exception: %s", e)
            return False

    def disconnect(self) -> None:
        try:
            self._facade.stop()  # lifecycle hold; never detorques (plan §5.2)
        except Exception as e:
            logger.warning("HandSHMAdapter.disconnect exception: %s", e)

    def is_connected(self) -> bool:
        # Return cached flags — they are refreshed by get_state() (every
        # control loop tick) and send_action().  Avoids a full SHM ring
        # read (~14 KB including tactile) on every status query.
        return self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        # Same rationale as is_connected() above.
        return (not self.connected_flag) or self.error_state

    def clear_error(self) -> bool:
        try:
            # Release the detorque latch FIRST — otherwise the child re-detorques
            # within ≤1 tick and undoes the clear (R1: stop()->clear_error()->resume).
            self._estop_event.clear()
            res = self._facade.clear_error()
            ok = bool(res["ok"][0])
        except Exception as e:
            logger.warning("HandSHMAdapter.clear_error exception: %s", e)
            return False
        rec, _age = self._facade.get_state()
        self._refresh_status(rec)
        return ok

    def stop(self) -> bool:
        """Deliberate detorque — mirrors ``XHand.stop()`` (de-energize + set
        ``error_state=True``). Sets the shared estop event: the child calls
        ``hand.stop()`` exactly once within ≤1 tick (plan §4.8).
        """
        try:
            self._estop_event.set()
            self.error_state = True
            self.connected_flag = False
            return True
        except Exception as e:
            logger.warning("HandSHMAdapter.stop exception: %s", e)
            return False

    def reset(self, qpos: np.ndarray | None = None) -> bool:
        target = (
            np.asarray(qpos, dtype=np.float64)
            if qpos is not None
            else np.asarray(self.config.home_qpos, dtype=np.float64)
        )
        try:
            res = self._facade.reset(target)
            ok = bool(res["ok"][0])
        except Exception as e:
            logger.warning("HandSHMAdapter.reset exception: %s", e)
            return False
        rec, _age = self._facade.get_state()
        self._refresh_status(rec)
        return ok

    # ── State (XHand dict shape) ──

    def get_state(self, full: bool = False, force_update: bool | None = None) -> dict:
        """Return hand state dict duck-typed to XHand.get_state().

        Parameters
        ----------
        full:
            When True, also includes connected_flag, error_state, and error
            fields available in the SHM ring.  Diagnostic fields that only
            exist in the real SDK (finger_ids, sensor_ids, raw_position,
            joint_names, etc.) are omitted — SHM carries a fixed subset.
        force_update:
            Ignored in SHM mode — the child process always reads hardware
            at 30 Hz regardless of this flag.
        """
        rec, _age = self._facade.get_state()
        self._refresh_status(rec)
        if rec is None or not np.all(np.isfinite(np.asarray(rec["qpos"][0], dtype=np.float64))):
            state: dict[str, Any] = {
                "qpos": nan_array(12),
                "current": np.zeros(12, dtype=np.float64),
                "timestamp": time.time(),
                "tactile_force": np.zeros((5, 120, 3), dtype=np.float64),
                "tactile_force_sum": np.zeros((5, 3), dtype=np.float64),
                "tactile_contact": np.zeros(5, dtype=bool),
                "tipboard_err": np.zeros(12, dtype=np.int32),
            }
        else:
            state = {
                "qpos": np.asarray(rec["qpos"][0], dtype=np.float64).copy(),
                "current": np.asarray(rec["current"][0], dtype=np.float64).copy(),
                "temperature": np.asarray(rec["temperature"][0], dtype=np.float64).copy(),
                "timestamp": time.time(),
                "tactile_force": np.asarray(rec["tactile_force"][0], dtype=np.float64).copy(),
                "tactile_force_sum": np.asarray(rec["tactile_sum"][0], dtype=np.float64).copy(),
                "tactile_contact": np.asarray(rec["tactile_contact"][0], dtype=bool).copy(),
                "tipboard_err": np.asarray(rec["tipboard_err"][0], dtype=np.int32).copy(),
            }
        if full:
            state.update({
                "connected_flag": self.connected_flag,
                "error_state": self.error_state,
                "consecutive_errors": int(rec["consecutive_errs"][0]) if rec is not None else 0,
                "last_error_code": int(rec["last_error_code"][0]) if rec is not None else 0,
                "limit_clipped": bool(rec["limit_clipped"][0]) if rec is not None else False,
            })
        return state

    # ── Command (main → child) ──

    def send_action(self, action: np.ndarray) -> bool:
        try:
            ok, expected_cmd = self._facade.send_action(action)
            self._facade.check_echo()
        except Exception as e:
            logger.warning("HandSHMAdapter.send_action exception: %s", e)
            return False
        if ok:
            self.last_qpos_cmd = np.asarray(expected_cmd, dtype=np.float64).copy()
        rec, _age = self._facade.get_state()
        self._refresh_status(rec)
        return bool(ok)


def make_hand_servo(
    hand_config: XHandConfig,
    *,
    process_config: HandProcessConfig | None = None,
    hand_factory: Callable[[XHandConfig], Any] | None = None,
) -> XHand | HandSHMAdapter:
    """Build the hand servo: crash-isolated subprocess via HandSHMAdapter.

    XHand runs in a fork child owning the sole hand SDK connection.
    Satisfies the XHand duck-type (connect/disconnect/is_connected/is_error/
    clear_error/stop/reset/get_state/send_action/last_qpos_cmd/config).

    Args:
        hand_config: XHand config (forwarded to the child's XHand).
        process_config: Full override for the subprocess config; when ``None``
            one is built with ``HandProcessConfig`` defaults.
        hand_factory: Test injection for the child's XHand (no hardware).
    """
    if process_config is None:
        process_config = HandProcessConfig()
    estop_event = mp.get_context("fork").Event()
    facade = HandSHMFaçade(
        process_config,
        sync=None,
        estop_event=estop_event,
        hand_config=hand_config,
        hand_factory=hand_factory,
    )
    return HandSHMAdapter(facade, hand_config, estop_event)
