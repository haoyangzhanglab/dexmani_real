"""Hand process — SharedStorage architecture.

Architecture:
    Policy → hand_cmd_ring → hand_loop → XHand hardware
    XHand hardware → hand_state_ring / hand_tactile_ring → Policy

Single entry point: hand_loop(shared) — mp.Process target, uses SharedStorage rings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.defaults import hand
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.robot.xhand.xhand import XHand, XHandConfig

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class HandProcessConfig:
    """Configuration for hand_loop — defaults from hand singleton."""

    loop_hz: float = field(default_factory=lambda: hand.loop_hz)

    # Homing convergence
    home_settle_timeout_s: float = field(default_factory=lambda: hand.home_settle_timeout_s)
    home_settle_tol_rad: float = field(default_factory=lambda: hand.home_settle_tol_rad)

    # Qpos freshness detection (driver board lockout guard)
    stale_qpos_frame_limit: int = field(default_factory=lambda: hand.stale.frame_count)
    stale_qpos_cmd_gap_rad: float = field(default_factory=lambda: hand.stale.cmd_gap_rad)
    stale_qpos_delta_rad: float = field(default_factory=lambda: hand.stale.qpos_delta_rad)

    # Send-error watchdog: auto clear_error() after N consecutive send failures
    send_err_watchdog_frames: int = field(default_factory=lambda: hand.send_err_watchdog_count)


# ═══════════════════════════════════════════════════════════════════
# hand_loop — mp.Process target
# ═══════════════════════════════════════════════════════════════════


def hand_loop(shared, config: HandProcessConfig | None = None) -> None:
    """Hand process entry point — reads shared.hand_cmd_ring, servos hand.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).
    """
    from dexmani_real.shm.shared_storage import HAND_STATE_DTYPE as _HS_STATE
    from dexmani_real.shm.shared_storage import HAND_TACTILE_DTYPE as _HS_TACTILE, new_frame as _nf
    from dexmani_real.robot.safety import SafetyState, transition

    cfg = config or HandProcessConfig()

    try:
        from dexmani_real.robot.xhand.xhand import XHand, XHandConfig
        hand = XHand(XHandConfig())
        if not hand.connect():
            logger.error("hand_loop: connect failed")
            shared.hand_ready.set()
            shared.error_state.value = True
            return
    except Exception as e:
        logger.error("hand_loop: init failed: %s", e)
        shared.hand_ready.set()
        shared.error_state.value = True
        return

    # Home — re-send in the polling loop so the hand PID keeps driving
    # toward home_qpos until the physical qpos converges.
    home_qpos = getattr(hand.config, 'home_qpos', None)
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
            shared.error_state.value = True
            return

    # Write heartbeat BEFORE ready signal — prevents false FAULT on startup
    # (same pattern as vr_loop).  Main's supervisor checks heartbeats immediately
    # after all ready events; if this process hasn't entered its main loop yet,
    # heartbeat=0 → age=inf → spurious FAULT.
    shared.hand_heartbeat_s.value = time.monotonic()
    shared.hand_ready.set()
    logger.info("hand_loop: ready")

    interval = 1.0 / cfg.loop_hz
    last_ts = time.monotonic()
    last_cmd_seq = 0
    consecutive_send_errors = 0
    _last_clear_error_s = 0.0

    # Qpos freshness detection (driver board lockout guard)
    _stale_frames = 0
    _last_fresh_qpos: np.ndarray | None = None
    last_known_qpos: np.ndarray = np.zeros(12, dtype=np.float64)

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
                            consecutive_send_errors = 0
                        except Exception:
                            consecutive_send_errors += 1
                            logger.warning("hand_loop: send_action failed (consecutive=%d)", consecutive_send_errors, exc_info=True)
                        last_cmd_seq = seq_int

            # Send-error watchdog: auto clear_error() after consecutive failures.
            if consecutive_send_errors >= cfg.send_err_watchdog_frames:
                _now = time.monotonic()
                if _now - _last_clear_error_s > 2.0:
                    logger.warning("hand_loop: %d consecutive send errors — clear_error()", consecutive_send_errors)
                    try:
                        hand.clear_error()
                    except Exception:
                        logger.warning("hand_loop: clear_error() failed", exc_info=True)
                    _last_clear_error_s = _now

        # Read state (always — even when safety-gated)
        try:
            st = hand.get_state(full=True, force_update=True)
            qpos = np.asarray(st.get("qpos", np.zeros(12)), dtype=np.float64)
            current = np.asarray(st.get("current", np.zeros(12)), dtype=np.float64)
            tactile_sum = np.asarray(st.get("tactile_force_sum", np.zeros((5, 3))), dtype=np.float64)
            tactile_force = np.asarray(st.get("tactile_force", np.zeros((5, 120, 3))), dtype=np.float64)
            tactile_contact = np.asarray(st.get("tactile_contact", np.zeros(5, dtype=bool)), dtype=bool)
            connected = hand.connected_flag
            error_state = hand.error_state
            last_known_qpos = qpos.copy()
        except Exception:
            logger.warning("hand_loop: get_state failed", exc_info=True)
            qpos = last_known_qpos.copy()
            current = np.zeros(12)
            tactile_sum = np.zeros((5, 3))
            tactile_force = np.zeros((5, 120, 3))
            tactile_contact = np.zeros(5, dtype=bool)
            connected = False
            error_state = False  # transient comm glitch — do NOT fabricate error_state

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

        # Propagate hand error to global error_state (H6)
        if error_state and not shared.error_state.value:
            shared.error_state.value = True
            transition(shared, SafetyState.FAULT)
            logger.error("hand_loop: hand error state — setting global error_state + FAULT")

        # Publish state
        frame = _nf(_HS_STATE)
        frame["qpos"][0] = qpos
        frame["current"][0] = current
        frame["tactile_sum"][0] = tactile_sum
        frame["tactile_contact"][0] = tactile_contact
        frame["error_state"][0] = int(error_state)
        frame["connected"][0] = int(connected)
        frame["qpos_stale"][0] = int(qpos_stale)
        frame["timestamp"][0] = time.monotonic()
        shared.hand_state_ring.write(frame)

        # Publish tactile (sparse — only when contact detected)
        if np.any(tactile_contact):
            tf = _nf(_HS_TACTILE)
            tf["tactile_force"][0] = tactile_force
            shared.hand_tactile_ring.write(tf)

        # Rate limit
        elapsed = time.monotonic() - last_ts
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_ts = time.monotonic()

    try:
        hand.stop()
        hand.disconnect()
    except Exception:
        logger.warning("hand_loop: cleanup failed", exc_info=True)
    logger.info("hand_loop: exited")
