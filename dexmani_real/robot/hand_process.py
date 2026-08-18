"""Hand servo process — reads hand_cmd_ring, servos XHand, writes hand state and tactile rings.

Every device-read result publishes a tactile frame: complete sensor payloads
are fresh, while malformed/missing payloads immediately publish fresh=0 and
calibrated=0 so consumers never keep treating an older tactile frame as valid.
hand_state_ring publishes every tick and marks failed reads invalid.
Error recovery: three independent counters for send failures, board error states,
and read exceptions — each escalates to global error_state on persistent failure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.config.defaults import hand
from dexmani_real.policy.safety import worker_validate_hand
from dexmani_real.utils.hand_health import XHAND_OVERCURRENT_ERROR_CODE
from dexmani_real.utils.limits import validate_hand_limit_nesting
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import EventWindowCounter, RetryCounter
from dexmani_real.utils.schema import (HAND_CONTACT_SHAPE, HAND_JOINT_SHAPE,
                                       HAND_STATE_DTYPE, HAND_TACTILE_DTYPE,
                                       HAND_TACTILE_FORCE_SHAPE,
                                       HAND_TACTILE_SUM_SHAPE)

logger = get_logger(__name__)


@dataclass
class HandProcessConfig:
    """Configuration for hand_loop."""

    loop_hz: float = field(default_factory=lambda: hand.loop_hz)

    # Runtime entry points keep this fail-closed. The non-fatal mode is
    # retained only for explicit offline fault-injection tests.
    startup_failure_is_fatal: bool = True
    ethercat_slave_position: int = field(default_factory=lambda: hand.ethercat_slave_position)

    # Transport protocol + device identity, resolved from the immutable runtime
    # config (defaults + file/CLI overrides), not guessed by the driver.
    comm_type: str = field(default_factory=lambda: hand.comm_type)
    device_name: str | None = field(default_factory=lambda: hand.device_name)
    baudrate: int = field(default_factory=lambda: hand.baudrate)
    device_id: int = field(default_factory=lambda: hand.device_id)
    rs485_post_open_settle_s: float = field(default_factory=lambda: hand.rs485_post_open_settle_s)
    rs485_crc_retry_count: int = field(default_factory=lambda: hand.rs485_crc_retry_count)
    rs485_read_crc_retry_count: int = field(default_factory=lambda: hand.rs485_read_crc_retry_count)
    rs485_crc_retry_backoff_s: float = field(default_factory=lambda: hand.rs485_crc_retry_backoff_s)

    home_qpos_rad: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(value) for value in np.deg2rad(hand.home_qpos_deg))
    )
    qpos_lower_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_min_rad)
    qpos_upper_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_max_rad)
    mechanical_qpos_lower_rad: tuple[float, ...] = field(default_factory=lambda: hand.mechanical_qpos_min_rad)
    mechanical_qpos_upper_rad: tuple[float, ...] = field(default_factory=lambda: hand.mechanical_qpos_max_rad)

    # Servo gains (PID) and per-joint current limit, resolved from the
    # immutable runtime config (defaults + file/CLI overrides). ``kp`` and
    # ``tor_max_ma`` are per-joint; ``ki``/``kd`` are uniform.
    kp: tuple[int, ...] = field(default_factory=lambda: hand.kp)
    ki: int = field(default_factory=lambda: hand.ki)
    kd: int = field(default_factory=lambda: hand.kd)
    tor_max_ma: tuple[int, ...] = field(default_factory=lambda: hand.tor_max_ma)

    # Send-error watchdog: latch global error_state after N consecutive failed
    # *new* command sends (a frame-count threshold, not a wall-clock one — see
    # defaults.send_err_watchdog_count).
    send_err_watchdog_frames: int = field(default_factory=lambda: hand.send_err_watchdog_count)

    # Error-state watchdog: latch global error_state after N consecutive
    # error_state ticks (persistent board faults).  Shared with the
    # get_state-exception counter for consistent escalation behaviour.
    error_state_watchdog_frames: int = 5
    overcurrent_fault_count: int = field(default_factory=lambda: hand.overcurrent_fault_count)
    overcurrent_fault_window_s: float = field(default_factory=lambda: hand.overcurrent_fault_window_s)

    def __post_init__(self) -> None:
        if self.ethercat_slave_position < -1:
            raise ValueError("hand process EtherCAT slave position must be -1 or non-negative")
        if self.comm_type not in ("ethercat", "serial"):
            raise ValueError("hand process comm_type must be 'ethercat' or 'serial'")
        if self.device_name is not None and not isinstance(self.device_name, str):
            raise ValueError("hand process device_name must be a string or null")
        if not isinstance(self.baudrate, int) or self.baudrate <= 0:
            raise ValueError("hand process baudrate must be a positive integer")
        if not isinstance(self.device_id, int) or self.device_id < 0:
            raise ValueError("hand process device_id must be a non-negative integer")
        if not np.isfinite(self.rs485_post_open_settle_s) or self.rs485_post_open_settle_s < 0:
            raise ValueError("hand process rs485_post_open_settle_s must be finite and non-negative")
        if not isinstance(self.rs485_crc_retry_count, int) or self.rs485_crc_retry_count < 0:
            raise ValueError("hand process rs485_crc_retry_count must be a non-negative integer")
        if not isinstance(self.rs485_read_crc_retry_count, int) or self.rs485_read_crc_retry_count < 0:
            raise ValueError("hand process rs485_read_crc_retry_count must be a non-negative integer")
        if not np.isfinite(self.rs485_crc_retry_backoff_s) or self.rs485_crc_retry_backoff_s < 0:
            raise ValueError("hand process rs485_crc_retry_backoff_s must be finite and non-negative")
        home = np.asarray(self.home_qpos_rad, dtype=np.float64)
        lower = np.asarray(self.qpos_lower_rad, dtype=np.float64)
        upper = np.asarray(self.qpos_upper_rad, dtype=np.float64)
        mechanical_lower = np.asarray(self.mechanical_qpos_lower_rad, dtype=np.float64)
        mechanical_upper = np.asarray(self.mechanical_qpos_upper_rad, dtype=np.float64)
        rated_lower = np.asarray(hand.mechanical_qpos_min_rad, dtype=np.float64)
        rated_upper = np.asarray(hand.mechanical_qpos_max_rad, dtype=np.float64)
        vectors = (home, lower, upper, mechanical_lower, mechanical_upper)
        if any(value.shape != HAND_JOINT_SHAPE for value in vectors):
            raise ValueError(f"hand process home/limits must have shape {HAND_JOINT_SHAPE}")
        if not np.all(np.isfinite(np.concatenate(vectors))):
            raise ValueError("hand process home/limits must be finite")
        validate_hand_limit_nesting(
            lower,
            upper,
            mechanical_lower,
            mechanical_upper,
            rated_lower,
            rated_upper,
            label="hand process",
        )
        if np.any(home < lower - 1e-12) or np.any(home > upper + 1e-12):
            raise ValueError("hand process home must be inside command limits")
        if len(self.kp) != HAND_JOINT_SHAPE[0] or any(
            not isinstance(value, int) or value <= 0 for value in self.kp
        ):
            raise ValueError("hand process kp must contain twelve positive integer gains")
        if self.ki < 0 or self.kd < 0:
            raise ValueError("hand process ki/kd must be non-negative")
        if len(self.tor_max_ma) != HAND_JOINT_SHAPE[0] or any(
            not isinstance(value, int) or value <= 0 for value in self.tor_max_ma
        ):
            raise ValueError("hand process tor_max_ma must contain twelve positive integer mA limits")
        if not np.isfinite(self.loop_hz) or self.loop_hz <= 0:
            raise ValueError("hand process loop_hz must be finite and positive")
        if self.send_err_watchdog_frames <= 0 or self.error_state_watchdog_frames <= 0:
            raise ValueError("hand process watchdog thresholds must be positive")
        if not isinstance(self.overcurrent_fault_count, int) or self.overcurrent_fault_count <= 0:
            raise ValueError("hand process overcurrent_fault_count must be a positive integer")
        if (
            not np.isfinite(self.overcurrent_fault_window_s)
            or self.overcurrent_fault_window_s <= 0
        ):
            raise ValueError("hand process overcurrent_fault_window_s must be finite and positive")

    @classmethod
    def from_runtime(cls, runtime: object, *, startup_failure_is_fatal: bool = True) -> "HandProcessConfig":
        cfg = getattr(runtime, "hand")
        return cls(
            loop_hz=float(cfg.loop_hz),
            startup_failure_is_fatal=startup_failure_is_fatal,
            ethercat_slave_position=int(cfg.ethercat_slave_position),
            comm_type=str(cfg.comm_type),
            device_name=None if cfg.device_name is None else str(cfg.device_name),
            baudrate=int(cfg.baudrate),
            device_id=int(cfg.device_id),
            rs485_post_open_settle_s=float(cfg.rs485_post_open_settle_s),
            rs485_crc_retry_count=int(cfg.rs485_crc_retry_count),
            rs485_read_crc_retry_count=int(cfg.rs485_read_crc_retry_count),
            rs485_crc_retry_backoff_s=float(cfg.rs485_crc_retry_backoff_s),
            home_qpos_rad=tuple(float(value) for value in np.deg2rad(cfg.home_qpos_deg)),
            qpos_lower_rad=tuple(float(value) for value in cfg.qpos_min_rad),
            qpos_upper_rad=tuple(float(value) for value in cfg.qpos_max_rad),
            mechanical_qpos_lower_rad=tuple(float(value) for value in cfg.mechanical_qpos_min_rad),
            mechanical_qpos_upper_rad=tuple(float(value) for value in cfg.mechanical_qpos_max_rad),
            kp=tuple(int(value) for value in cfg.kp),
            ki=int(cfg.ki),
            kd=int(cfg.kd),
            tor_max_ma=tuple(int(value) for value in cfg.tor_max_ma),
            send_err_watchdog_frames=int(cfg.send_err_watchdog_count),
            overcurrent_fault_count=int(cfg.overcurrent_fault_count),
            overcurrent_fault_window_s=float(cfg.overcurrent_fault_window_s),
        )


def _safe_disconnect(hand) -> bool:
    """Disconnect the hand driver, tolerating a never-connected instance.

    Mirrors ``_disconnect_arm`` (arm_loop.py): the single cleanup path for the
    hand worker, reached from every exit (startup failure, init exception, loop
    exit, or fault).  Returns True when there is nothing to disconnect or the
    disconnect succeeds; a raised disconnect is logged and reported as False.
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


def hand_loop(shared, config: HandProcessConfig | None = None) -> None:
    """Hand process entry point — reads shared.hand_cmd_ring, servos hand.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).
    """
    from dexmani_real.robot.safety import SafetyState
    from dexmani_real.shm.shared_storage import new_frame as _nf

    cfg = config or HandProcessConfig()
    logger.debug("hand_loop: LOADING")

    def _mark_startup_failure() -> None:
        logger.error("hand_loop: XHand startup failed; see process log")
        if cfg.startup_failure_is_fatal:
            shared.error_state.value = True
        else:
            logger.warning("hand_loop: XHand unavailable in non-fatal test mode — leaving hand_ready unset")

    hand = None
    ready = False
    try:
        try:
            from dexmani_real.robot.xhand import XHand, XHandConfig, XHandReadError

            # Servo gains and per-joint current limit come from the resolved
            # runtime config (defaults + file/CLI overrides), not hardcoded here.
            hand = XHand(
                XHandConfig(
                    comm_type=cfg.comm_type,
                    device_name=cfg.device_name,
                    baudrate=cfg.baudrate,
                    device_id=cfg.device_id,
                    rs485_post_open_settle_s=cfg.rs485_post_open_settle_s,
                    rs485_crc_retry_count=cfg.rs485_crc_retry_count,
                    rs485_read_crc_retry_count=cfg.rs485_read_crc_retry_count,
                    rs485_crc_retry_backoff_s=cfg.rs485_crc_retry_backoff_s,
                    kp_per_joint=np.asarray(cfg.kp, dtype=np.int32),
                    ki=cfg.ki,
                    kd=cfg.kd,
                    tor_max_per_joint=np.asarray(cfg.tor_max_ma, dtype=np.int32),
                    ethercat_slave_position=cfg.ethercat_slave_position,
                    home_qpos=np.asarray(cfg.home_qpos_rad, dtype=np.float64),
                    qpos_min=np.asarray(cfg.qpos_lower_rad, dtype=np.float64),
                    qpos_max=np.asarray(cfg.qpos_upper_rad, dtype=np.float64),
                    mechanical_qpos_min=np.asarray(cfg.mechanical_qpos_lower_rad, dtype=np.float64),
                    mechanical_qpos_max=np.asarray(cfg.mechanical_qpos_upper_rad, dtype=np.float64),
                )
            )
            if not hand.connect():
                if cfg.startup_failure_is_fatal:
                    logger.error("hand_loop: connect failed")
                else:
                    logger.warning("hand_loop: XHand connect failed in non-fatal test mode")
                _mark_startup_failure()
                return
            if hasattr(shared, "hand_device_identity"):
                identity = getattr(hand, "device_identity", {"backend": "unavailable"})
                encoded_identity = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
                shared.hand_device_identity.value = encoded_identity[:1023].ljust(1024, b"\x00")
        except Exception:
            if cfg.startup_failure_is_fatal:
                logger.error("hand_loop: init failed", exc_info=True)
            else:
                logger.warning("hand_loop: XHand init failed in non-fatal test mode", exc_info=True)
            _mark_startup_failure()
            return

        # Explicit tactile reset/bias, split out of connect().  The connection
        # itself only opens the device and seeds the command history.  Tactile
        # failure degrades to calibrated=False without blocking joint control —
        # never a startup failure.
        try:
            hand.initialize_tactile()
        except Exception:
            hand.tactile_calibrated = False
            logger.warning("hand_loop: tactile initialization raised", exc_info=True)

        # DISARMED startup is read-only.  Opening the bus and validating feedback
        # must never create a home motion; homing remains an explicit, correlated
        # policy action after Main transitions the system to ARMED.

        # Publish initial state BEFORE hand_ready — consumers wait on hand_ready and
        # expect the ring to already contain a valid frame.  Without this, there is
        # a one-tick window where hand_ready is set but hand_state_ring is empty.
        # (Same pattern as arm_loop arm_ready.)
        try:
            st = hand.get_state(force_update=True)
            # shape + finite are already validated by XHandSample construction.
            _init_qpos = st.qpos
            _initial_values: dict[str, np.ndarray] = {
                "current": st.current,
                "tactile_sum": st.tactile_sum,
                "tactile_contact": st.tactile_contact,
            }
            _initial_tactile_valid = bool(st.tactile_valid)
            _initial_tactile_sum_valid = bool(st.tactile_sum_valid)
            if not bool(hand.connected_flag):
                raise RuntimeError("initial hand feedback reports a disconnected device")
            if bool(hand.error_state):
                raise RuntimeError("initial hand feedback reports a hardware error")
            _initial_board_errors: dict[str, np.ndarray] = {}
            for _name in ("commboard_err", "jointboard_err", "tipboard_err"):
                _value = getattr(st, _name)
                if np.any(_value != 0):
                    raise RuntimeError(f"initial hand feedback reports {_name}")
                _initial_board_errors[_name] = _value.copy()
        except Exception:
            _log = logger.error if cfg.startup_failure_is_fatal else logger.warning
            _log("hand_loop: cannot publish a valid initial state", exc_info=True)
            _mark_startup_failure()
            return

        _frame0 = _nf(HAND_STATE_DTYPE)
        _frame0["qpos"][0] = _init_qpos
        _frame0["current"][0] = _initial_values["current"]
        _frame0["tactile_sum"][0] = _initial_values["tactile_sum"]
        _frame0["tactile_sum_valid"][0] = int(_initial_tactile_sum_valid)
        _frame0["tactile_contact"][0] = _initial_values["tactile_contact"]
        _frame0["error_state"][0] = int(bool(hand.error_state))
        _frame0["connected"][0] = int(bool(hand.connected_flag))
        # Kept zero for schema compatibility. A stationary joint vector under
        # contact is valid feedback, not a stale-source condition.
        _frame0["qpos_stale"][0] = 0
        _frame0["last_cmd_seq"][0] = 0
        _frame0["last_cmd_qpos"][0] = np.asarray(hand.last_qpos_cmd, dtype=np.float64)
        for _name, _value in _initial_board_errors.items():
            _frame0[_name][0] = _value
        _initial_source_ns = time.monotonic_ns()
        _frame0["source_monotonic_ns"][0] = _initial_source_ns
        _frame0["publish_monotonic_ns"][0] = time.monotonic_ns()
        _frame0["state_valid"][0] = 1
        _frame0["send_healthy"][0] = 1
        _frame0["read_healthy"][0] = 1
        _frame0["read_error_count"][0] = 0
        _frame0["overcurrent_error_count"][0] = 0
        _frame0["last_read_error_code"][0] = 0
        _frame0["last_read_error_monotonic_ns"][0] = 0
        _frame0["timestamp"][0] = _initial_source_ns / 1e9
        shared.hand_state_ring.write(_frame0)
        shared.hand_tactile_ring.write(
            _build_tactile_frame(
                st.tactile_force,
                source_monotonic_ns=_initial_source_ns,
                valid=_initial_tactile_valid,
                calibrated=bool(hand.tactile_calibrated),
            )
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

        rate_mgr = RateManager(cfg.loop_hz, label="hand")
        last_consumed_ring_sequence = 0
        _send_error_counter = RetryCounter(max_consecutive=cfg.send_err_watchdog_frames, label="hand_send")
        _error_state_counter = RetryCounter(max_consecutive=cfg.error_state_watchdog_frames, label="hand_error_state")
        _read_error_counter = RetryCounter(max_consecutive=cfg.error_state_watchdog_frames, label="hand_read_error")
        _overcurrent_window = EventWindowCounter(
            max_events=cfg.overcurrent_fault_count,
            window_s=cfg.overcurrent_fault_window_s,
        )
        _overcurrent_fault_logged = False

        last_known_qpos = _init_qpos.copy()
        last_known_current = np.asarray(_initial_values["current"], dtype=np.float64).copy()
        _last_tactile_sum: np.ndarray = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        _last_tactile_force: np.ndarray = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        _last_state_source_ns = _initial_source_ns
        _read_error_count_total = 0
        _overcurrent_error_count_total = 0
        _last_read_error_code = 0
        _last_read_error_ns = 0
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
                qpos_lower_rad=np.asarray(cfg.qpos_lower_rad, dtype=np.float64),
                qpos_upper_rad=np.asarray(cfg.qpos_upper_rad, dtype=np.float64),
                mechanical_lower_rad=np.asarray(cfg.mechanical_qpos_lower_rad, dtype=np.float64),
                mechanical_upper_rad=np.asarray(cfg.mechanical_qpos_upper_rad, dtype=np.float64),
                expected_run_generation=int(shared.run_generation.value),
                now_monotonic_ns=time.monotonic_ns(),
            ):
                logger.info(
                    "hand_loop: discarded malformed, out-of-envelope, stale-generation, or expired command"
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
                        sent = hand.send_action(cmd)
                    except Exception:
                        sent = False
                        logger.warning("hand_loop: send_action raised", exc_info=True)
                    if sent:
                        _send_error_counter.reset()
                        last_applied_action_id = int(execute_action["action_id"][0])
                    else:
                        _send_error_counter.inc()

                # Send-error watchdog: persistent failed sends latch a global fault.
                # There is no local "clear" step — error_state is recomputed from the
                # board registers on every successful read, so the send counter is the
                # only recovery bookkeeping (success resets it, failures accumulate).
                if _send_error_counter.triggered:
                    shared.error_state.value = True
                    logger.error("hand_loop: persistent send failures — latching global fault")

            # Read state (always — even when safety-gated)
            read_failed = False
            try:
                st = hand.get_state(force_update=True)
                qpos = st.qpos
                current = st.current
                tactile_sum = st.tactile_sum
                tactile_force = st.tactile_force
                tactile_sum_valid = bool(st.tactile_sum_valid)
                tactile_contact = st.tactile_contact
                tactile_valid = bool(st.tactile_valid)
                connected = hand.connected_flag
                error_state = hand.error_state
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
            except Exception as exc:
                read_failed = True
                _read_error_count_total += 1
                _last_read_error_ns = time.monotonic_ns()
                _last_read_error_code = int(exc.code) if isinstance(exc, XHandReadError) else -1
                logger.warning(
                    "hand_loop: get_state failed code=%d connected=%d action_id=%d",
                    _last_read_error_code,
                    int(bool(hand.connected_flag)),
                    last_applied_action_id,
                    exc_info=True,
                )
                if _last_read_error_code == XHAND_OVERCURRENT_ERROR_CODE:
                    _overcurrent_error_count_total += 1
                    logger.warning(
                        "hand_loop: overcurrent context last_current_ma=%s tor_max_ma=%s "
                        "last_qpos_rad=%s last_cmd_qpos_rad=%s",
                        np.round(last_known_current, 1).tolist(),
                        list(cfg.tor_max_ma),
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
                    if _overcurrent_window.record(time.monotonic()):
                        shared.error_state.value = True
                        if not _overcurrent_fault_logged:
                            _overcurrent_fault_logged = True
                            logger.error(
                                "hand_loop: %d overcurrent events within %.1fs — "
                                "latching global error_state",
                                _overcurrent_window.count,
                                cfg.overcurrent_fault_window_s,
                            )
                qpos = last_known_qpos.copy()
                current = np.zeros(HAND_JOINT_SHAPE)
                tactile_sum = _last_tactile_sum.copy()
                tactile_force = _last_tactile_force.copy()
                tactile_sum_valid = False
                tactile_contact = np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
                tactile_valid = False
                connected = bool(hand.connected_flag)
                error_state = False  # transient comm glitch — do NOT fabricate error_state
                commboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
                jointboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
                tipboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)

                # Read-error escalation: persistent get_state exceptions (SDK crash,
                # USB disconnect) bypass the normal error_state retry path because
                # error_state is forced to False above.  A dedicated counter ensures
                # this silent-dead-hand scenario still escalates to global error_state.
                _read_error_counter.inc()
                if _read_error_counter.triggered:
                    shared.error_state.value = True
                    logger.error(
                        "hand_loop: %d consecutive get_state exceptions — latching global error_state",
                        _read_error_counter.max_consecutive,
                    )

            # Retry transient hand errors only for the configured bounded window.
            # ``error_state`` is recomputed from the board registers on every read,
            # so a single read/write result decides it — no separate clear step.
            if error_state and not shared.error_state.value:
                _error_state_counter.inc()
                if _error_state_counter.triggered:
                    shared.error_state.value = True
                    logger.error(
                        "hand_loop: hand error_state persisted after %d retries — latching global error_state",
                        _error_state_counter.max_consecutive,
                    )
            elif not error_state:
                _error_state_counter.reset()

            # Publish state
            frame = _nf(HAND_STATE_DTYPE)
            frame["qpos"][0] = qpos
            frame["current"][0] = current
            frame["tactile_sum"][0] = tactile_sum
            frame["tactile_sum_valid"][0] = int(tactile_sum_valid)
            frame["tactile_contact"][0] = tactile_contact
            frame["error_state"][0] = int(error_state)
            frame["connected"][0] = int(connected)
            frame["qpos_stale"][0] = int(read_failed)
            frame["last_cmd_seq"][0] = last_applied_action_id
            frame["last_cmd_qpos"][0] = np.asarray(hand.last_qpos_cmd, dtype=np.float64)
            frame["commboard_err"][0] = commboard_err
            frame["jointboard_err"][0] = jointboard_err
            frame["tipboard_err"][0] = tipboard_err
            frame["source_monotonic_ns"][0] = _last_state_source_ns
            frame["publish_monotonic_ns"][0] = time.monotonic_ns()
            frame["state_valid"][0] = int(connected and not read_failed)
            frame["send_healthy"][0] = int(not _send_error_counter.triggered)
            frame["read_healthy"][0] = int(not read_failed and not _read_error_counter.triggered)
            frame["read_error_count"][0] = _read_error_count_total
            frame["overcurrent_error_count"][0] = _overcurrent_error_count_total
            frame["last_read_error_code"][0] = _last_read_error_code
            frame["last_read_error_monotonic_ns"][0] = _last_read_error_ns
            frame["timestamp"][0] = _last_state_source_ns / 1e9
            shared.hand_state_ring.write(frame)

            # Explicitly invalidate malformed tactile or failed reads so an
            # older valid ring entry cannot masquerade as current data.
            shared.hand_tactile_ring.write(
                _build_tactile_frame(
                    tactile_force,
                    source_monotonic_ns=_last_state_source_ns,
                    valid=bool(connected and tactile_valid),
                    calibrated=bool(hand.tactile_calibrated),
                )
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
