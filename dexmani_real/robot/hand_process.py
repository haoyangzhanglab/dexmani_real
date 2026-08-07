"""Hand servo process — reads hand_cmd_ring, servos XHand, writes hand_state_ring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.config.defaults import hand
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import RetryCounter

logger = get_logger(__name__)


@dataclass
class HandProcessConfig:
    """Configuration for hand_loop."""

    loop_hz: float = field(default_factory=lambda: hand.loop_hz)

    # Homing convergence
    home_settle_timeout_s: float = field(default_factory=lambda: hand.home_settle_timeout_s)
    home_settle_tol_rad: float = field(default_factory=lambda: hand.home_settle_tol_rad)

    # Qpos freshness detection (driver board lockout guard)
    stale_qpos_frame_limit: int = field(default_factory=lambda: hand.stale.frame_count)
    stale_qpos_delta_rad: float = field(default_factory=lambda: hand.stale.qpos_delta_rad)

    # Send-error watchdog: auto clear_error() after N consecutive send failures
    send_err_watchdog_frames: int = field(default_factory=lambda: hand.send_err_watchdog_count)


def hand_loop(shared, config: HandProcessConfig | None = None) -> None:
    """Hand process entry point — reads shared.hand_cmd_ring, servos hand.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).
    """
    from dexmani_real.robot.safety import SafetyState, transition
    from dexmani_real.shm.shared_storage import HAND_STATE_DTYPE as _HS_STATE
    from dexmani_real.shm.shared_storage import HAND_TACTILE_DTYPE as _HS_TACTILE
    from dexmani_real.shm.shared_storage import new_frame as _nf

    cfg = config or HandProcessConfig()

    try:
        from dexmani_real.robot.xhand import XHand, XHandConfig

        # Per-joint tor_max: index abduction (J3) handles sideways load,
        # benefit from higher current limit (380 vs default 300 mA).
        _tor_max_pj = np.full(12, 300, dtype=np.int32)
        _tor_max_pj[3] = 380

        hand = XHand(XHandConfig(tor_max_per_joint=_tor_max_pj))
        if not hand.connect():
            logger.error("hand_loop: connect failed")
            shared.hand_ready.set()
            return
    except Exception as e:
        logger.error("hand_loop: init failed: %s", e)
        shared.hand_ready.set()
        return

    # Home — re-send in the polling loop so the hand PID keeps driving
    # toward home_qpos until the physical qpos converges.
    home_qpos = getattr(hand.config, "home_qpos", None)
    if home_qpos is not None and np.all(np.isfinite(home_qpos)):
        _home_deadline = time.monotonic() + cfg.home_settle_timeout_s
        _home_reached = False
        while time.monotonic() < _home_deadline:
            hand.send_action(home_qpos)
            try:
                st = hand.get_state()
                if st is not None:
                    current = np.asarray(st.get("qpos", np.zeros(12)), dtype=np.float64)
                    if float(np.max(np.abs(current - home_qpos))) < cfg.home_settle_tol_rad:
                        _home_reached = True
                        break
            except Exception:
                pass
            time.sleep(0.05)
        if not _home_reached:
            logger.error("hand_loop: home settle failed after %.1fs", cfg.home_settle_timeout_s)
            try:
                hand.stop()
                hand.disconnect()
            except Exception:
                pass
            shared.hand_ready.set()
            return

    # Publish initial state BEFORE hand_ready — consumers wait on hand_ready and
    # expect the ring to already contain a valid frame.  Without this, there is
    # a one-tick window where hand_ready is set but hand_state_ring is empty.
    # (Same pattern as arm_loop arm_ready.)
    try:
        st = hand.get_state()
        _init_qpos = np.asarray(st.get("qpos", np.zeros(12)), dtype=np.float64) if st is not None else np.zeros(12)
    except Exception:
        _init_qpos = np.zeros(12, dtype=np.float64)
    _frame0 = _nf(_HS_STATE)
    _frame0["qpos"][0] = _init_qpos
    _frame0["current"][0] = np.zeros(12, dtype=np.float64)
    _frame0["tactile_sum"][0] = np.zeros((5, 3), dtype=np.float64)
    _frame0["tactile_contact"][0] = np.zeros(5, dtype=bool)
    _frame0["error_state"][0] = 0
    _frame0["connected"][0] = 1
    _frame0["qpos_stale"][0] = 0
    _frame0["commboard_err"][0] = np.zeros(12, dtype=np.int32)
    _frame0["jointboard_err"][0] = np.zeros(12, dtype=np.int32)
    _frame0["tipboard_err"][0] = np.zeros(12, dtype=np.int32)
    _frame0["timestamp"][0] = time.monotonic()
    shared.hand_state_ring.write(_frame0)

    # Write heartbeat BEFORE ready signal — prevents false FAULT on startup
    # (same pattern as vr_loop).  Main's supervisor checks heartbeats immediately
    # after all ready events; if this process hasn't entered its main loop yet,
    # heartbeat=0 → age=inf → spurious FAULT.
    shared.hand_heartbeat_s.value = time.monotonic()
    shared.hand_ready.set()
    logger.info("hand_loop: ready")

    rate_mgr = RateManager(cfg.loop_hz)
    last_cmd_seq = 0
    _send_error_counter = RetryCounter(max_consecutive=cfg.send_err_watchdog_frames, label="hand_send")
    _error_state_counter = RetryCounter(max_consecutive=5, label="hand_error_state")
    _last_clear_error_s = 0.0

    # Qpos freshness detection (driver board lockout guard)
    _stale_frames = 0
    _last_fresh_qpos: np.ndarray | None = None
    last_known_qpos: np.ndarray = np.zeros(12, dtype=np.float64)

    _last_error_clear_s = 0.0

    while shared.is_running.value:
        # Heartbeat — written even when gated (proves we're alive)
        shared.hand_heartbeat_s.value = time.monotonic()

        if shared.estop_request.value:
            break

        # Safety state gate — only process commands in ARMED or RUNNING.
        _safety = shared.safety_state.value
        if _safety in (SafetyState.ARMED, SafetyState.RUNNING):

            # Read cmd ring (latest-wins)
            result = shared.hand_cmd_ring.read_latest()
            if result is not None:
                data, _ts, seq = result
                seq_int = int(seq) if isinstance(seq, (int, np.integer)) else 0
                if seq_int != last_cmd_seq:
                    cmd = np.asarray(data["qpos_cmd"][0], dtype=np.float64)
                    if np.all(np.isfinite(cmd)):
                        try:
                            hand.send_action(cmd)
                            _send_error_counter.reset()
                        except Exception:
                            _send_error_counter.inc()
                            logger.warning(
                                "hand_loop: send_action failed (consecutive=%d)", _send_error_counter.count, exc_info=True
                            )
                        last_cmd_seq = seq_int

            # Send-error watchdog: auto clear_error() after consecutive failures.
            if _send_error_counter.triggered:
                _now = time.monotonic()
                if _now - _last_clear_error_s > 2.0:
                    logger.warning("hand_loop: %d consecutive send errors — clear_error()", _send_error_counter.count)
                    try:
                        hand.clear_error()
                    except Exception:
                        logger.warning("hand_loop: clear_error() failed", exc_info=True)
                    _last_clear_error_s = _now

        # Read state (always — even when safety-gated)
        try:
            st = hand.get_state(full=True, force_update=True)
            # Use sentinel-based .get() to avoid eager fallback allocation.
            # dict.get(key, default) evaluates `default` unconditionally,
            # wasting ~14.5 KB per tick (30 Hz) on large tactile arrays.
            _raw_qpos = st.get("qpos")
            qpos = np.asarray(_raw_qpos if _raw_qpos is not None else np.zeros(12), dtype=np.float64)
            _raw_current = st.get("current")
            current = np.asarray(_raw_current if _raw_current is not None else np.zeros(12), dtype=np.float64)
            _raw_ts = st.get("tactile_force_sum")
            tactile_sum = np.asarray(_raw_ts if _raw_ts is not None else np.zeros((5, 3)), dtype=np.float64)
            _raw_tf = st.get("tactile_force")
            tactile_force = np.asarray(_raw_tf if _raw_tf is not None else np.zeros((5, 120, 3)), dtype=np.float64)
            _raw_tc = st.get("tactile_contact")
            tactile_contact = np.asarray(_raw_tc if _raw_tc is not None else np.zeros(5, dtype=bool), dtype=bool)
            connected = hand.connected_flag
            error_state = hand.error_state
            last_known_qpos = qpos.copy()
            # Board error registers (per-joint hardware fault indicators).
            _raw_cbe = st.get("commboard_err")
            commboard_err = np.asarray(_raw_cbe if _raw_cbe is not None else np.zeros(12, dtype=np.int32), dtype=np.int32)
            _raw_jbe = st.get("jointboard_err")
            jointboard_err = np.asarray(_raw_jbe if _raw_jbe is not None else np.zeros(12, dtype=np.int32), dtype=np.int32)
            _raw_tbe = st.get("tipboard_err")
            tipboard_err = np.asarray(_raw_tbe if _raw_tbe is not None else np.zeros(12, dtype=np.int32), dtype=np.int32)
        except Exception:
            logger.warning("hand_loop: get_state failed", exc_info=True)
            qpos = last_known_qpos.copy()
            current = np.zeros(12)
            tactile_sum = np.zeros((5, 3))
            tactile_force = np.zeros((5, 120, 3))
            tactile_contact = np.zeros(5, dtype=bool)
            connected = False
            error_state = False  # transient comm glitch — do NOT fabricate error_state
            commboard_err = np.zeros(12, dtype=np.int32)
            jointboard_err = np.zeros(12, dtype=np.int32)
            tipboard_err = np.zeros(12, dtype=np.int32)

        # Detect stale qpos (driver board lockout)
        if not connected:
            # Reset on disconnect — prevents false stale-positive on reconnect
            # when hand moved while limp (pre-disconnect qpos ≠ post-reconnect qpos).
            _last_fresh_qpos = None
            _stale_frames = 0
        elif connected and _last_fresh_qpos is not None:
            if np.max(np.abs(qpos - _last_fresh_qpos)) < cfg.stale_qpos_delta_rad:
                _stale_frames += 1
            else:
                _stale_frames = 0
                _last_fresh_qpos = qpos.copy()
        elif _last_fresh_qpos is None:
            _last_fresh_qpos = qpos.copy()
        qpos_stale = _stale_frames >= cfg.stale_qpos_frame_limit

        # Error-state retry: hand comm errors are frequently intermittent and
        # self-recovering.  Try clear_error() up to N times before escalating
        # to global error_state + FAULT.
        # (Same design rationale as send-error watchdog — ref: Hand Comm Error
        # Handling Policy memory.)
        if error_state and not shared.error_state.value:
            _error_state_counter.inc()
            _now_err = time.monotonic()
            if _now_err - _last_error_clear_s > 1.0:
                logger.warning(
                    "hand_loop: hand error_state — clear_error() (%d/%d consecutive)",
                    _error_state_counter.count,
                    _error_state_counter.max_consecutive,
                )
                try:
                    hand.clear_error()
                except Exception:
                    logger.warning("hand_loop: clear_error() failed", exc_info=True)
                _last_error_clear_s = _now_err
            if _error_state_counter.triggered:
                shared.error_state.value = True
                transition(shared, SafetyState.FAULT)
                logger.error(
                    "hand_loop: hand error_state persisted after %d retries — setting global error_state + FAULT",
                    _error_state_counter.max_consecutive,
                )
        elif not error_state:
            _error_state_counter.reset()

        # Publish state
        frame = _nf(_HS_STATE)
        frame["qpos"][0] = qpos
        frame["current"][0] = current
        frame["tactile_sum"][0] = tactile_sum
        frame["tactile_contact"][0] = tactile_contact
        frame["error_state"][0] = int(error_state)
        frame["connected"][0] = int(connected)
        frame["qpos_stale"][0] = int(qpos_stale)
        frame["commboard_err"][0] = commboard_err
        frame["jointboard_err"][0] = jointboard_err
        frame["tipboard_err"][0] = tipboard_err
        frame["timestamp"][0] = time.monotonic()
        shared.hand_state_ring.write(frame)

        # Publish tactile (sparse — only when contact detected)
        if np.any(tactile_contact):
            tf = _nf(_HS_TACTILE)
            tf["tactile_force"][0] = tactile_force
            shared.hand_tactile_ring.write(tf)

        # Rate limit (absolute-deadline scheduling, consistent with arm_loop/policy_loop)
        rate_mgr.wait()

    # ── Pre-disconnect home: drive hand to home_qpos before releasing the
    # EtherCAT bus.  If the hand driver board is in a degraded state
    # (qpos_stale / comm errors), a clean disconnect is more likely when
    # the board is idle at a known position rather than mid-grasp.
    _cleanup_home_qpos = home_qpos  # captured at init (line 79)
    if _cleanup_home_qpos is not None and np.all(np.isfinite(_cleanup_home_qpos)):
        try:
            for _ in range(40):  # ~2s at 30 Hz (generous settle window)
                hand.send_action(_cleanup_home_qpos)
                time.sleep(0.05)
        except Exception:
            pass

    try:
        hand.stop()
        hand.disconnect()
    except Exception:
        logger.warning("hand_loop: cleanup failed", exc_info=True)
    logger.info("hand_loop: exited")
