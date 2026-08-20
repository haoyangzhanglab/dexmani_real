"""Hand servo process — reads hand_cmd_ring, servos XHand, writes hand state and tactile rings.

Every device-read result publishes a tactile frame: complete sensor payloads
are fresh, while malformed/missing payloads immediately publish fresh=0 and
calibrated=0 so consumers never keep treating an older tactile frame as valid.
hand_state_ring publishes every tick and marks failed reads invalid.
Error recovery: three independent counters for send failures, board faults from
``XHandState``, and read exceptions — each latches global ``error_state`` only
after persistent failure.
"""

from __future__ import annotations

import json
import time

import numpy as np

from dexmani_real.config.defaults import HandParams
from dexmani_real.policy.safety import worker_validate_hand
from dexmani_real.utils.hand_health import XHAND_OVERCURRENT_ERROR_CODE
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import RetryCounter
from dexmani_real.utils.schema import (HAND_CONTACT_SHAPE, HAND_JOINT_SHAPE,
                                       HAND_STATE_DTYPE, HAND_TACTILE_DTYPE,
                                       HAND_TACTILE_FORCE_SHAPE,
                                       HAND_TACTILE_SUM_SHAPE)

logger = get_logger(__name__)


def _safe_disconnect(hand) -> bool:
    """Disconnect the hand driver, tolerating a never-connected instance.

    Mirrors the arm cleanup path (``XArm7.close`` in robot/xarm7.py): the
    single cleanup path for the hand worker, reached from every exit (startup
    failure, init exception, loop exit, or fault).  Returns True when there is
    nothing to disconnect or the disconnect succeeds; a raised disconnect is
    logged and reported as False.
    """
    if hand is None:
        return True
    try:
        hand.disconnect()
    except Exception:
        logger.warning("hand_loop: cleanup failed", exc_info=True)
        return False
    return True


def _log_board_error_transitions(
    previous: dict[str, np.ndarray], current: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Log per-joint board-error register appear/change/disappear transitions.

    Only joints whose register value changed since the previous sample are
    logged, so a steady-state error value never spams the log.  Returns a fresh
    dict of copies to use as the next ``previous`` (never aliases the driver's
    arrays).
    """
    for name in ("commboard_err", "jointboard_err", "tipboard_err"):
        prev = previous[name]
        cur = current[name]
        if prev.shape == cur.shape:
            for joint in range(int(cur.shape[0])):
                if prev[joint] != cur[joint]:
                    logger.info(
                        "%s[%d] 0x%08x -> 0x%08x",
                        name,
                        joint,
                        int(prev[joint]),
                        int(cur[joint]),
                    )
    return {name: current[name].copy() for name in previous}


def _build_tactile_frame(
    tactile_force: np.ndarray,
    *,
    source_monotonic_ns: int,
    valid: bool,
    calibrated: bool,
) -> np.ndarray:
    """Build one tactile publication, explicitly invalidating bad payloads."""
    frame = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
    if valid:
        force = np.asarray(tactile_force, dtype=np.float64)
        if force.shape != HAND_TACTILE_FORCE_SHAPE or not np.all(np.isfinite(force)):
            raise ValueError(
                "valid tactile_force must be finite with shape "
                f"{HAND_TACTILE_FORCE_SHAPE}"
            )
        frame["tactile_force"][0] = force
    frame["source_monotonic_ns"][0] = max(0, int(source_monotonic_ns))
    frame["fresh"][0] = int(valid)
    frame["calibrated"][0] = int(valid and calibrated)
    # SDK conversion provenance has not been independently established.
    frame["unit_code"][0] = 0
    return frame


def _publish_feedback(
    shared,
    *,
    qpos: np.ndarray,
    current_ma: np.ndarray,
    tactile_sum: np.ndarray,
    tactile_sum_valid: bool,
    tactile_contact: np.ndarray,
    tactile_force: np.ndarray,
    tactile_valid: bool,
    tactile_calibrated: bool,
    has_hardware_fault: bool,
    connected: bool,
    read_failed: bool,
    last_cmd_seq: int,
    last_cmd_qpos: np.ndarray,
    commboard_err: np.ndarray,
    jointboard_err: np.ndarray,
    tipboard_err: np.ndarray,
    source_monotonic_ns: int,
    send_healthy: bool,
    read_healthy: bool,
    read_error_count: int,
    overcurrent_error_count: int,
) -> None:
    """Serialize one feedback pair while preserving the existing SHM contract."""
    from dexmani_real.shm.shared_storage import new_frame

    source_ns = max(0, int(source_monotonic_ns))
    frame = new_frame(HAND_STATE_DTYPE)
    frame["qpos"][0] = qpos
    frame["current"][0] = current_ma
    frame["tactile_sum"][0] = tactile_sum
    frame["tactile_sum_valid"][0] = int(tactile_sum_valid)
    frame["tactile_contact"][0] = tactile_contact
    frame["error_state"][0] = int(has_hardware_fault)
    frame["connected"][0] = int(connected)
    frame["qpos_stale"][0] = int(read_failed)
    frame["last_cmd_seq"][0] = int(last_cmd_seq)
    frame["last_cmd_qpos"][0] = last_cmd_qpos
    frame["commboard_err"][0] = commboard_err
    frame["jointboard_err"][0] = jointboard_err
    frame["tipboard_err"][0] = tipboard_err
    frame["source_monotonic_ns"][0] = source_ns
    frame["publish_monotonic_ns"][0] = time.monotonic_ns()
    frame["state_valid"][0] = int(connected and not read_failed)
    frame["send_healthy"][0] = int(send_healthy)
    frame["read_healthy"][0] = int(read_healthy)
    frame["read_error_count"][0] = int(read_error_count)
    frame["overcurrent_error_count"][0] = int(overcurrent_error_count)
    frame["timestamp"][0] = source_ns / 1e9
    shared.hand_state_ring.write(frame)
    shared.hand_tactile_ring.write(
        _build_tactile_frame(
            tactile_force,
            source_monotonic_ns=source_ns,
            valid=bool(connected and tactile_valid),
            calibrated=tactile_calibrated,
        )
    )


def hand_loop(shared, config: HandParams) -> None:
    """Hand process entry point — reads shared.hand_cmd_ring, servos hand.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).
    """
    from dexmani_real.robot.safety import SafetyState
    logger.debug("hand_loop: LOADING")

    def _mark_startup_failure() -> None:
        logger.error("hand_loop: XHand startup failed; see process log")
        shared.error_state.value = True

    hand = None
    ready = False
    try:
        try:
            from dexmani_real.robot.xhand import XHand, XHandError

            hand = XHand(config)
            hand.connect()
            if hasattr(shared, "hand_device_identity"):
                identity = getattr(hand, "device_identity", {"backend": "unavailable"})
                encoded_identity = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
                shared.hand_device_identity.value = encoded_identity[:1023].ljust(1024, b"\x00")
        except Exception:
            logger.error("hand_loop: init failed", exc_info=True)
            _mark_startup_failure()
            return

        # Explicit tactile reset/bias, split out of connect(). The connection
        # itself only opens the device and seeds the command history.  Tactile
        # failure degrades to calibrated=False without blocking joint control —
        # never a startup failure.
        try:
            hand.calibrate_tactile()
        except Exception:
            logger.warning("hand_loop: tactile calibration raised", exc_info=True)

        # DISARMED startup is read-only.  Opening the bus and validating feedback
        # must never create a home motion; homing remains an explicit, correlated
        # policy action after Main transitions the system to ARMED.

        # Publish initial state BEFORE hand_ready — consumers wait on hand_ready and
        # expect the ring to already contain a valid frame.  Without this, there is
        # a one-tick window where hand_ready is set but hand_state_ring is empty.
        # (Same pattern as arm_loop arm_ready.)
        try:
            st = hand.get_state()
            _init_qpos = st.qpos
            _initial_values: dict[str, np.ndarray] = {
                "current": st.current_ma,
                "tactile_sum": st.tactile_sum,
                "tactile_contact": st.tactile_contact,
            }
            _initial_tactile_valid = bool(st.tactile_valid)
            _initial_tactile_sum_valid = bool(st.tactile_sum_valid)
            if not hand.is_connected:
                raise RuntimeError("initial hand feedback reports a disconnected device")
            if st.has_hardware_fault:
                raise RuntimeError("initial hand feedback reports a hardware error")
            _initial_board_errors: dict[str, np.ndarray] = {}
            for _name in ("commboard_err", "jointboard_err", "tipboard_err"):
                _value = getattr(st, _name)
                if np.any(_value != 0):
                    raise RuntimeError(f"initial hand feedback reports {_name}")
                _initial_board_errors[_name] = _value.copy()
        except Exception:
            logger.error("hand_loop: cannot publish a valid initial state", exc_info=True)
            _mark_startup_failure()
            return

        _initial_source_ns = time.monotonic_ns()
        _publish_feedback(
            shared,
            qpos=_init_qpos,
            current_ma=_initial_values["current"],
            tactile_sum=_initial_values["tactile_sum"],
            tactile_sum_valid=_initial_tactile_sum_valid,
            tactile_contact=_initial_values["tactile_contact"],
            tactile_force=st.tactile_force,
            tactile_valid=_initial_tactile_valid,
            tactile_calibrated=hand.tactile_calibrated,
            has_hardware_fault=st.has_hardware_fault,
            connected=hand.is_connected,
            read_failed=False,
            last_cmd_seq=0,
            last_cmd_qpos=np.asarray(hand.last_qpos_cmd, dtype=np.float64),
            commboard_err=_initial_board_errors["commboard_err"],
            jointboard_err=_initial_board_errors["jointboard_err"],
            tipboard_err=_initial_board_errors["tipboard_err"],
            source_monotonic_ns=_initial_source_ns,
            send_healthy=True,
            read_healthy=True,
            read_error_count=0,
            overcurrent_error_count=0,
        )

        # Write heartbeat BEFORE ready signal — prevents false FAULT on startup
        # (same pattern as vr_loop).  Main's supervisor checks heartbeats immediately
        # after all ready events; if this process hasn't entered its main loop yet,
        # heartbeat=0 → age=inf → spurious FAULT.
        shared.set_heartbeat("hand", time.monotonic())
        shared.set_ready("hand")
        ready = True
        logger.debug("hand_loop: READY")
        logger.info("hand_loop: ready")

        rate_mgr = RateManager(config.loop_hz, label="hand")
        last_consumed_ring_sequence = 0
        _send_error_counter = RetryCounter(max_consecutive=config.send_err_watchdog_count, label="hand_send")
        _error_state_counter = RetryCounter(max_consecutive=config.error_state_watchdog_frames, label="hand_error_state")
        _read_error_counter = RetryCounter(max_consecutive=config.error_state_watchdog_frames, label="hand_read_error")

        last_known_qpos = _init_qpos.copy()
        last_known_current = np.asarray(_initial_values["current"], dtype=np.float64).copy()
        _last_tactile_sum: np.ndarray = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        _last_tactile_force: np.ndarray = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        _last_state_source_ns = _initial_source_ns
        _read_error_count_total = 0
        _overcurrent_error_count_total = 0
        _last_read_error_code = 0
        last_applied_action_id = 0

        _prev_board_errs: dict[str, np.ndarray] = {
            _name: _initial_board_errors[_name].copy()
            for _name in ("commboard_err", "jointboard_err", "tipboard_err")
        }

        def _read_latest_command() -> np.ndarray | None:
            """Read one new latest-wins ring publication, if available.

            ``last_consumed_ring_sequence`` is the hand command ring cursor;
            it is unrelated to ``HAND_STATE_DTYPE.last_cmd_seq``, which exposes
            the last SDK-accepted ``action_id``. Claim before validation/send so
            a malformed or failed latest-wins publication is never replayed.
            """
            nonlocal last_consumed_ring_sequence
            result = shared.hand_cmd_ring.read_latest()
            if result is None:
                return None
            data, _ts, seq = result
            seq_int = int(seq) if isinstance(seq, (int, np.integer)) else 0
            if seq_int == last_consumed_ring_sequence:
                return None
            last_consumed_ring_sequence = seq_int
            if not worker_validate_hand(
                data,
                expected_run_generation=int(shared.run_generation.value),
                now_monotonic_ns=time.monotonic_ns(),
            ):
                logger.info(
                    "hand_loop: discarded malformed, stale-generation, or expired command"
                )
                return None
            return data.copy()

        while shared.is_running.value:
            # Heartbeat — written even when gated (proves we're alive)
            shared.set_heartbeat("hand", time.monotonic())

            if shared.estop_request.value:
                break

            # Safety state gate — only process commands in ARMED or RUNNING.
            _safety = shared.safety_state.value
            if _safety in (SafetyState.ARMED, SafetyState.RUNNING) and not shared.error_state.value:
                execute_action = _read_latest_command()
                if execute_action is not None:
                    cmd = np.asarray(execute_action["qpos_cmd"][0], dtype=np.float64)
                    try:
                        hand.send_action(cmd)
                    except (XHandError, ValueError):
                        logger.warning("hand_loop: send_action raised", exc_info=True)
                        _send_error_counter.inc()
                    else:
                        _send_error_counter.reset()
                        last_applied_action_id = int(execute_action["action_id"][0])

                # Send-error watchdog: persistent failed sends latch a global fault.
                # ``has_hardware_fault`` is derived from each fresh state, so the
                # send counter is the only send-recovery bookkeeping: successes reset
                # it and failed new commands accumulate.
                if _send_error_counter.triggered:
                    shared.error_state.value = True
                    logger.error("hand_loop: persistent send failures — latching global fault")

            # Read state (always — even when safety-gated)
            read_failed = False
            try:
                st = hand.get_state()
                qpos = st.qpos
                current = st.current_ma
                tactile_sum = st.tactile_sum
                tactile_force = st.tactile_force
                tactile_sum_valid = bool(st.tactile_sum_valid)
                tactile_contact = st.tactile_contact
                tactile_valid = bool(st.tactile_valid)
                connected = hand.is_connected
                has_hardware_fault = st.has_hardware_fault
                _last_state_source_ns = time.monotonic_ns()
                last_known_qpos = qpos.copy()
                last_known_current = current.copy()
                _last_tactile_sum = tactile_sum.copy()
                _last_tactile_force = tactile_force.copy()
                _read_error_counter.reset()
                # Board error registers (per-joint hardware fault indicators).
                commboard_err = st.commboard_err
                jointboard_err = st.jointboard_err
                tipboard_err = st.tipboard_err
                _prev_board_errs = _log_board_error_transitions(
                    _prev_board_errs,
                    {
                        "commboard_err": commboard_err,
                        "jointboard_err": jointboard_err,
                        "tipboard_err": tipboard_err,
                    },
                )
            except XHandError as exc:
                _last_read_error_code = exc.code
                is_overcurrent = _last_read_error_code == XHAND_OVERCURRENT_ERROR_CODE
                if is_overcurrent:
                    # Overcurrent is a recoverable firmware warning, not a read
                    # failure: the hand stays connected and joint feedback stays
                    # valid. Keep the last-known current so the stall load stays
                    # observable (firmware tor_max already bounds it) and record
                    # the event as an observation only — no pause, no fault.
                    read_failed = False
                    _overcurrent_error_count_total += 1
                    logger.warning(
                        "hand_loop: overcurrent context last_current_ma=%s tor_max_ma=%s "
                        "last_qpos_rad=%s last_cmd_qpos_rad=%s",
                        np.round(last_known_current, 1).tolist(),
                        list(config.tor_max_ma),
                        np.round(last_known_qpos, 4).tolist(),
                        np.round(
                            np.asarray(
                                hand.last_qpos_cmd
                                if hand.last_qpos_cmd is not None
                                else last_known_qpos
                            ),
                            4,
                        ).tolist(),
                    )
                else:
                    read_failed = True
                    _read_error_count_total += 1
                    logger.warning(
                        "hand_loop: get_state failed code=%d connected=%d action_id=%d",
                        _last_read_error_code,
                        int(bool(hand.connected_flag)),
                        last_applied_action_id,
                        exc_info=True,
                    )

                qpos = last_known_qpos.copy()
                current = (
                    last_known_current.copy()
                    if is_overcurrent
                    else np.zeros(HAND_JOINT_SHAPE)
                )
                tactile_sum = _last_tactile_sum.copy()
                tactile_force = _last_tactile_force.copy()
                tactile_sum_valid = False
                tactile_contact = np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
                tactile_valid = False
                connected = hand.is_connected
                # A transient read failure is not a board fault. The independent
                # read watchdog owns escalation to shared.error_state.
                has_hardware_fault = False
                commboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
                jointboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
                tipboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)

                if not is_overcurrent:
                    # Read-error escalation: persistent get_state exceptions (SDK crash,
                    # USB disconnect) bypass the board-fault path because no fresh
                    # board registers are available. A dedicated counter still latches
                    # global error_state for this silent-dead-hand scenario.
                    _read_error_counter.inc()
                    if _read_error_counter.triggered:
                        shared.error_state.value = True
                        logger.error(
                            "hand_loop: %d consecutive get_state exceptions — latching global error_state",
                            _read_error_counter.max_consecutive,
                        )

            # Board-fault escalation is independent of transient read failures.
            # Each successful fresh state decides the current board-fault value.
            if has_hardware_fault and not shared.error_state.value:
                _error_state_counter.inc()
                if _error_state_counter.triggered:
                    shared.error_state.value = True
                    logger.error(
                        "hand_loop: board fault persisted after %d retries — latching global error_state",
                        _error_state_counter.max_consecutive,
                    )
            elif not has_hardware_fault:
                _error_state_counter.reset()

            _publish_feedback(
                shared,
                qpos=qpos,
                current_ma=current,
                tactile_sum=tactile_sum,
                tactile_sum_valid=tactile_sum_valid,
                tactile_contact=tactile_contact,
                tactile_force=tactile_force,
                tactile_valid=tactile_valid,
                tactile_calibrated=hand.tactile_calibrated,
                has_hardware_fault=has_hardware_fault,
                connected=connected,
                read_failed=read_failed,
                last_cmd_seq=last_applied_action_id,
                last_cmd_qpos=np.asarray(hand.last_qpos_cmd, dtype=np.float64),
                commboard_err=commboard_err,
                jointboard_err=jointboard_err,
                tipboard_err=tipboard_err,
                source_monotonic_ns=_last_state_source_ns,
                send_healthy=not _send_error_counter.triggered,
                read_healthy=not read_failed and not _read_error_counter.triggered,
                read_error_count=_read_error_count_total,
                overcurrent_error_count=_overcurrent_error_count_total,
            )

            # Keep absolute-deadline scheduling.
            rate_mgr.wait()
    finally:
        # Shutdown never creates new motion. Homing is an explicit, correlated
        # policy operation; worker cleanup only closes the device and releases the
        # bus after the command loop has been gated. The hand is intentionally NOT
        # unforced (mode=0) on shutdown — it stays in its last commanded position,
        # matching examples/xhand_control_example.py.
        if not _safe_disconnect(hand):
            logger.error("hand_loop: XHand disconnect failed")
            shared.error_state.value = True
        elif ready:
            logger.debug("hand_loop: STOPPED")
        logger.info("hand_loop: exited")
