"""Hand servo process — reads hand_cmd_ring, servos XHand, writes hand state and tactile rings.

Every successful device read publishes tactile data, including release/no-contact;
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
from dexmani_real.utils.limits import validate_hand_limit_nesting
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import RetryCounter
from dexmani_real.utils.schema import (HAND_CONTACT_SHAPE, HAND_JOINT_SHAPE,
                                       HAND_STATE_DTYPE, HAND_TACTILE_DTYPE,
                                       HAND_TACTILE_FORCE_SHAPE,
                                       HAND_TACTILE_SUM_SHAPE)

logger = get_logger(__name__)


@dataclass
class HandProcessConfig:
    """Configuration for hand_loop."""

    loop_hz: float = field(default_factory=lambda: hand.loop_hz)

    # Production entry points keep this fail-closed. The non-fatal mode is
    # retained only for explicit offline fault-injection tests.
    startup_failure_is_fatal: bool = True
    ethercat_slave_position: int = field(default_factory=lambda: hand.ethercat_slave_position)

    home_qpos_rad: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(value) for value in np.deg2rad(hand.home_qpos_deg))
    )
    qpos_lower_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_min_rad)
    qpos_upper_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_max_rad)
    mechanical_qpos_lower_rad: tuple[float, ...] = field(default_factory=lambda: hand.mechanical_qpos_min_rad)
    mechanical_qpos_upper_rad: tuple[float, ...] = field(default_factory=lambda: hand.mechanical_qpos_max_rad)
    max_command_delta_rad: float | None = field(default_factory=lambda: hand.max_delta_rad)

    # Servo gains (PID) and per-joint current limit, resolved from the
    # immutable runtime config (defaults + file/CLI overrides). ``kp`` and
    # ``tor_max_ma`` are per-joint; ``ki``/``kd`` are uniform.
    kp: tuple[int, ...] = field(default_factory=lambda: hand.kp)
    ki: int = field(default_factory=lambda: hand.ki)
    kd: int = field(default_factory=lambda: hand.kd)
    tor_max_ma: tuple[int, ...] = field(default_factory=lambda: hand.tor_max_ma)

    # Send-error watchdog: auto clear_local_error() after N consecutive send failures
    send_err_watchdog_frames: int = field(default_factory=lambda: hand.send_err_watchdog_count)

    # Error-state watchdog: latch global error_state after N consecutive
    # error_state ticks (persistent board faults).  Shared with the
    # get_state-exception counter for consistent escalation behaviour.
    error_state_watchdog_frames: int = 5

    def __post_init__(self) -> None:
        if self.ethercat_slave_position < -1:
            raise ValueError("hand process EtherCAT slave position must be -1 or non-negative")
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
        if self.max_command_delta_rad is not None and (
            not np.isfinite(self.max_command_delta_rad) or self.max_command_delta_rad <= 0.0
        ):
            raise ValueError("hand process max command delta must be finite and positive")
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

    @classmethod
    def from_runtime(cls, runtime: object, *, startup_failure_is_fatal: bool = True) -> "HandProcessConfig":
        cfg = getattr(runtime, "hand")
        return cls(
            loop_hz=float(cfg.loop_hz),
            startup_failure_is_fatal=startup_failure_is_fatal,
            ethercat_slave_position=int(cfg.ethercat_slave_position),
            home_qpos_rad=tuple(float(value) for value in np.deg2rad(cfg.home_qpos_deg)),
            qpos_lower_rad=tuple(float(value) for value in cfg.qpos_min_rad),
            qpos_upper_rad=tuple(float(value) for value in cfg.qpos_max_rad),
            mechanical_qpos_lower_rad=tuple(float(value) for value in cfg.mechanical_qpos_min_rad),
            mechanical_qpos_upper_rad=tuple(float(value) for value in cfg.mechanical_qpos_max_rad),
            max_command_delta_rad=None if cfg.max_delta_rad is None else float(cfg.max_delta_rad),
            kp=tuple(int(value) for value in cfg.kp),
            ki=int(cfg.ki),
            kd=int(cfg.kd),
            tor_max_ma=tuple(int(value) for value in cfg.tor_max_ma),
            send_err_watchdog_frames=int(cfg.send_err_watchdog_count),
        )


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

    try:
        from dexmani_real.robot.xhand import XHand, XHandConfig

        # Servo gains and per-joint current limit come from the resolved
        # runtime config (defaults + file/CLI overrides), not hardcoded here.
        hand = XHand(
            XHandConfig(
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
                max_delta_rad=cfg.max_command_delta_rad,
            )
        )
        if not hand.connect():
            if cfg.startup_failure_is_fatal:
                logger.error("hand_loop: connect failed")
            else:
                logger.warning("hand_loop: XHand connect failed in non-fatal test mode")
            try:
                hand.disconnect()
            except Exception:
                logger.warning("hand_loop: cleanup after connect failure failed", exc_info=True)
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

    # DISARMED startup is read-only.  Opening the bus and validating feedback
    # must never create a home motion; homing remains an explicit, correlated
    # policy action after Main transitions the system to ARMED.

    # Publish initial state BEFORE hand_ready — consumers wait on hand_ready and
    # expect the ring to already contain a valid frame.  Without this, there is
    # a one-tick window where hand_ready is set but hand_state_ring is empty.
    # (Same pattern as arm_loop arm_ready.)
    try:
        st = hand.get_state(full=True, force_update=True)
        if st is None:
            raise RuntimeError("initial hand state is unavailable")
        _init_qpos = np.asarray(st.get("qpos"), dtype=np.float64)
        if _init_qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(_init_qpos)):
            raise ValueError(f"invalid initial hand qpos shape/values: {_init_qpos.shape}")
        _initial_values: dict[str, np.ndarray] = {}
        for _name, _shape, _dtype in (
            ("current", HAND_JOINT_SHAPE, np.float64),
            ("tactile_force_sum", HAND_TACTILE_SUM_SHAPE, np.float64),
            ("tactile_contact", HAND_CONTACT_SHAPE, bool),
        ):
            _value = np.asarray(st.get(_name), dtype=_dtype)
            if _value.shape != _shape or (_name != "tactile_contact" and not np.all(np.isfinite(_value))):
                raise ValueError(f"invalid initial hand {_name} shape/values: {_value.shape}")
            _initial_values[_name] = _value.copy()
        if not bool(getattr(hand, "connected_flag", st.get("connected_flag", False))):
            raise RuntimeError("initial hand feedback reports a disconnected device")
        if bool(getattr(hand, "error_state", st.get("error_state", True))):
            raise RuntimeError("initial hand feedback reports a hardware error")
        _initial_board_errors: dict[str, np.ndarray] = {}
        for _name in ("commboard_err", "jointboard_err", "tipboard_err"):
            _value = np.asarray(st.get(_name), dtype=np.int32)
            if _value.shape != HAND_JOINT_SHAPE or np.any(_value != 0):
                raise RuntimeError(f"initial hand feedback reports {_name}")
            _initial_board_errors[_name] = _value.copy()
    except Exception:
        _log = logger.error if cfg.startup_failure_is_fatal else logger.warning
        _log("hand_loop: cannot publish a valid initial state", exc_info=True)
        try:
            hand.disconnect()
        except Exception:
            logger.warning("hand_loop: cleanup after initial-state failure failed", exc_info=True)
        _mark_startup_failure()
        return
    _frame0 = _nf(HAND_STATE_DTYPE)
    _frame0["qpos"][0] = _init_qpos
    _frame0["current"][0] = _initial_values["current"]
    _frame0["tactile_sum"][0] = _initial_values["tactile_force_sum"]
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
    _frame0["timestamp"][0] = _initial_source_ns / 1e9
    shared.hand_state_ring.write(_frame0)

    # Write heartbeat BEFORE ready signal — prevents false FAULT on startup
    # (same pattern as vr_loop).  Main's supervisor checks heartbeats immediately
    # after all ready events; if this process hasn't entered its main loop yet,
    # heartbeat=0 → age=inf → spurious FAULT.
    shared.set_heartbeat("hand", time.monotonic())
    shared.set_ready("hand")
    logger.debug("hand_loop: READY")
    logger.info("hand_loop: ready")

    rate_mgr = RateManager(cfg.loop_hz)
    last_cmd_seq = 0
    _send_error_counter = RetryCounter(max_consecutive=cfg.send_err_watchdog_frames, label="hand_send")
    _error_state_counter = RetryCounter(max_consecutive=cfg.error_state_watchdog_frames, label="hand_error_state")
    _read_error_counter = RetryCounter(max_consecutive=cfg.error_state_watchdog_frames, label="hand_read_error")
    _last_clear_error_s = 0.0

    last_known_qpos = _init_qpos.copy()
    _last_tactile_sum: np.ndarray = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
    _last_tactile_force: np.ndarray = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
    _last_state_source_ns = _initial_source_ns
    last_applied_action_id = 0

    _last_error_clear_s = 0.0

    def _read_latest_command() -> np.ndarray | None:
        """Read the latest-wins command if a new one is available."""
        nonlocal last_cmd_seq
        result = shared.hand_cmd_ring.read_latest()
        if result is None:
            return None
        data, _ts, seq = result
        seq_int = int(seq) if isinstance(seq, (int, np.integer)) else 0
        if seq_int == last_cmd_seq:
            return None
        last_cmd_seq = seq_int
        if not worker_validate_hand(
            data,
            qpos_lower_rad=np.asarray(cfg.qpos_lower_rad, dtype=np.float64),
            qpos_upper_rad=np.asarray(cfg.qpos_upper_rad, dtype=np.float64),
            mechanical_lower_rad=np.asarray(cfg.mechanical_qpos_lower_rad, dtype=np.float64),
            mechanical_upper_rad=np.asarray(cfg.mechanical_qpos_upper_rad, dtype=np.float64),
            previous_qpos_cmd=np.asarray(hand.last_qpos_cmd, dtype=np.float64),
            max_command_delta_rad=cfg.max_command_delta_rad,
            expected_run_generation=int(shared.run_generation.value),
            now_monotonic_ns=time.monotonic_ns(),
        ):
            logger.info(
                "hand_loop: discarded malformed, out-of-envelope, stale-generation, or expired command"
            )
            return None
        return data.copy()

    try:
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

                # Send-error watchdog: auto clear_local_error() after consecutive failures.
                if _send_error_counter.triggered:
                    shared.error_state.value = True
                    logger.error("hand_loop: persistent send failures — latching global fault")
                    _now = time.monotonic()
                    if _now - _last_clear_error_s > 2.0:
                        logger.warning("hand_loop: %d consecutive send errors — clear_local_error()", _send_error_counter.count)
                        try:
                            hand.clear_local_error()
                        except Exception:
                            logger.warning("hand_loop: clear_local_error() failed", exc_info=True)
                        _last_clear_error_s = _now

            # Read state (always — even when safety-gated)
            try:
                st = hand.get_state(full=True, force_update=True)
                # Use sentinel-based .get() to avoid eager fallback allocation.
                # dict.get(key, default) evaluates `default` unconditionally,
                # wasting ~14.5 KB per tick (30 Hz) on large tactile arrays.
                _raw_qpos = st.get("qpos")
                qpos = np.asarray(_raw_qpos if _raw_qpos is not None else np.zeros(HAND_JOINT_SHAPE), dtype=np.float64)
                _raw_current = st.get("current")
                current = np.asarray(
                    _raw_current if _raw_current is not None else np.zeros(HAND_JOINT_SHAPE), dtype=np.float64
                )
                _raw_ts = st.get("tactile_force_sum")
                tactile_sum = np.asarray(
                    _raw_ts if _raw_ts is not None else np.zeros(HAND_TACTILE_SUM_SHAPE), dtype=np.float64
                )
                _raw_tf = st.get("tactile_force")
                tactile_force = np.asarray(
                    _raw_tf if _raw_tf is not None else np.zeros(HAND_TACTILE_FORCE_SHAPE), dtype=np.float64
                )
                _raw_tc = st.get("tactile_contact")
                tactile_contact = np.asarray(
                    _raw_tc if _raw_tc is not None else np.zeros(HAND_CONTACT_SHAPE, dtype=bool), dtype=bool
                )
                expected_shapes = (
                    ("qpos", qpos, HAND_JOINT_SHAPE),
                    ("current", current, HAND_JOINT_SHAPE),
                    ("tactile_force_sum", tactile_sum, HAND_TACTILE_SUM_SHAPE),
                    ("tactile_force", tactile_force, HAND_TACTILE_FORCE_SHAPE),
                    ("tactile_contact", tactile_contact, HAND_CONTACT_SHAPE),
                )
                for field_name, value, expected_shape in expected_shapes:
                    if value.shape != expected_shape:
                        raise ValueError(f"invalid {field_name} shape {value.shape}, expected {expected_shape}")
                    if field_name != "tactile_contact" and not np.all(np.isfinite(value)):
                        raise ValueError(f"{field_name} contains NaN/Inf")
                connected = hand.connected_flag
                error_state = hand.error_state
                _last_state_source_ns = time.monotonic_ns()
                last_known_qpos = qpos.copy()
                _last_tactile_sum = tactile_sum.copy()
                _last_tactile_force = tactile_force.copy()
                _read_error_counter.reset()
                # Board error registers (per-joint hardware fault indicators).
                _raw_cbe = st.get("commboard_err")
                commboard_err = np.asarray(
                    _raw_cbe if _raw_cbe is not None else np.zeros(HAND_JOINT_SHAPE, dtype=np.int32), dtype=np.int32
                )
                _raw_jbe = st.get("jointboard_err")
                jointboard_err = np.asarray(
                    _raw_jbe if _raw_jbe is not None else np.zeros(HAND_JOINT_SHAPE, dtype=np.int32), dtype=np.int32
                )
                _raw_tbe = st.get("tipboard_err")
                tipboard_err = np.asarray(
                    _raw_tbe if _raw_tbe is not None else np.zeros(HAND_JOINT_SHAPE, dtype=np.int32), dtype=np.int32
                )
            except Exception:
                logger.warning("hand_loop: get_state failed", exc_info=True)
                qpos = last_known_qpos.copy()
                current = np.zeros(HAND_JOINT_SHAPE)
                tactile_sum = _last_tactile_sum.copy()
                tactile_force = _last_tactile_force.copy()
                tactile_contact = np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
                connected = False
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
            if error_state and not shared.error_state.value:
                _error_state_counter.inc()
                _now_err = time.monotonic()
                if _now_err - _last_error_clear_s > 1.0:
                    logger.warning(
                        "hand_loop: hand error_state — clear_local_error() (%d/%d consecutive)",
                        _error_state_counter.count,
                        _error_state_counter.max_consecutive,
                    )
                    try:
                        hand.clear_local_error()
                    except Exception:
                        logger.warning("hand_loop: clear_local_error() failed", exc_info=True)
                    _last_error_clear_s = _now_err
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
            frame["tactile_contact"][0] = tactile_contact
            frame["error_state"][0] = int(error_state)
            frame["connected"][0] = int(connected)
            frame["qpos_stale"][0] = 0
            frame["last_cmd_seq"][0] = last_applied_action_id
            frame["last_cmd_qpos"][0] = np.asarray(hand.last_qpos_cmd, dtype=np.float64)
            frame["commboard_err"][0] = commboard_err
            frame["jointboard_err"][0] = jointboard_err
            frame["tipboard_err"][0] = tipboard_err
            frame["source_monotonic_ns"][0] = _last_state_source_ns
            frame["publish_monotonic_ns"][0] = time.monotonic_ns()
            frame["state_valid"][0] = int(connected)
            frame["send_healthy"][0] = int(not _send_error_counter.triggered)
            frame["read_healthy"][0] = int(not _read_error_counter.triggered)
            frame["timestamp"][0] = _last_state_source_ns / 1e9
            shared.hand_state_ring.write(frame)

            # Every successful read is a source sample, including release/no-contact.
            if connected:
                tf = _nf(HAND_TACTILE_DTYPE)
                tf["tactile_force"][0] = tactile_force
                tf["source_monotonic_ns"][0] = _last_state_source_ns
                tf["fresh"][0] = 1
                tf["calibrated"][0] = int(hand.tactile_calibrated)
                # The SDK conversion provenance has not been independently
                # established on hardware.  Preserve the values, but label their
                # unit as unknown instead of guessing Newtons.
                tf["unit_code"][0] = 0
                shared.hand_tactile_ring.write(tf)

            # Rate limit (absolute-deadline scheduling, consistent with arm_loop/teleop_loop)
            rate_mgr.wait()
    finally:
        # Shutdown never creates new motion. Homing is an explicit, correlated
        # policy operation; worker cleanup only closes the device and releases the
        # bus after the command loop has been gated. The hand is intentionally NOT
        # unforced (mode=0) on shutdown — it stays in its last commanded position,
        # matching examples/xhand_control_example.py.
        stopped_cleanly = False
        try:
            hand.disconnect()
            stopped_cleanly = True
        except Exception:
            logger.warning("hand_loop: cleanup failed", exc_info=True)
            shared.error_state.value = True
        if stopped_cleanly:
            logger.debug("hand_loop: STOPPED")
        else:
            logger.error("hand_loop: XHand disconnect failed")
        logger.info("hand_loop: exited")
