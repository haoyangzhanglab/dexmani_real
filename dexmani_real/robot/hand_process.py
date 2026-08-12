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
from dexmani_real.utils.schema import (
    HAND_COMMAND_DTYPE,
    HAND_CONTACT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.policy.safety import worker_validate_hand
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import RetryCounter

logger = get_logger(__name__)


@dataclass
class HandProcessConfig:
    """Configuration for hand_loop."""

    loop_hz: float = field(default_factory=lambda: hand.loop_hz)

    # Production entry points keep this fail-closed. The non-fatal mode is
    # retained only for explicit offline fault-injection tests.
    startup_failure_is_fatal: bool = True
    ethercat_slave_position: int = field(default_factory=lambda: hand.ethercat_slave_position)

    # Homing convergence
    home_settle_timeout_s: float = field(default_factory=lambda: hand.home_settle_timeout_s)
    home_settle_tol_rad: float = field(default_factory=lambda: hand.home_settle_tol_rad)

    # Feedback-only diagnostic tolerance; strict XHand command limits are
    # configured separately in XHandConfig and remain unchanged.
    feedback_bound_tolerance_rad: float = field(default_factory=lambda: hand.feedback_bound_tolerance_rad)
    home_qpos_rad: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(value) for value in np.deg2rad(hand.home_qpos_deg))
    )
    qpos_lower_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_min_rad)
    qpos_upper_rad: tuple[float, ...] = field(default_factory=lambda: hand.qpos_max_rad)
    max_delta_rad: float | None = field(default_factory=lambda: hand.max_delta_rad)

    # Qpos freshness detection (driver board lockout guard)
    stale_qpos_frame_limit: int = field(default_factory=lambda: hand.stale.frame_count)
    stale_qpos_delta_rad: float = field(default_factory=lambda: hand.stale.qpos_delta_rad)

    # Send-error watchdog: auto clear_error() after N consecutive send failures
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
        if home.shape != HAND_JOINT_SHAPE or lower.shape != HAND_JOINT_SHAPE or upper.shape != HAND_JOINT_SHAPE:
            raise ValueError(f"hand process home/limits must have shape {HAND_JOINT_SHAPE}")
        if not np.all(np.isfinite(np.concatenate((home, lower, upper)))) or np.any(lower > upper):
            raise ValueError("hand process home/limits must be finite and ordered")
        if self.max_delta_rad is not None and (not np.isfinite(self.max_delta_rad) or self.max_delta_rad <= 0):
            raise ValueError("hand process max_delta_rad must be finite and positive")
        timing = (
            self.loop_hz,
            self.home_settle_timeout_s,
            self.home_settle_tol_rad,
            self.stale_qpos_delta_rad,
        )
        if not all(np.isfinite(value) and value > 0 for value in timing):
            raise ValueError("hand process rates/tolerances must be finite and positive")
        if not np.isfinite(self.feedback_bound_tolerance_rad) or self.feedback_bound_tolerance_rad < 0:
            raise ValueError("hand feedback tolerance must be finite and non-negative")
        if (
            self.stale_qpos_frame_limit <= 0
            or self.send_err_watchdog_frames <= 0
            or self.error_state_watchdog_frames <= 0
        ):
            raise ValueError("hand process rates/watchdog thresholds must be positive")

    @classmethod
    def from_runtime(cls, runtime: object, *, startup_failure_is_fatal: bool = True) -> "HandProcessConfig":
        cfg = getattr(runtime, "hand")
        return cls(
            loop_hz=float(cfg.loop_hz),
            startup_failure_is_fatal=startup_failure_is_fatal,
            ethercat_slave_position=int(cfg.ethercat_slave_position),
            home_settle_timeout_s=float(cfg.home_settle_timeout_s),
            home_settle_tol_rad=float(cfg.home_settle_tol_rad),
            feedback_bound_tolerance_rad=float(cfg.feedback_bound_tolerance_rad),
            home_qpos_rad=tuple(float(value) for value in np.deg2rad(cfg.home_qpos_deg)),
            qpos_lower_rad=tuple(float(value) for value in cfg.qpos_min_rad),
            qpos_upper_rad=tuple(float(value) for value in cfg.qpos_max_rad),
            max_delta_rad=None if cfg.max_delta_rad is None else float(cfg.max_delta_rad),
            stale_qpos_frame_limit=int(cfg.stale.frame_count),
            stale_qpos_delta_rad=float(cfg.stale.qpos_delta_rad),
            send_err_watchdog_frames=int(cfg.send_err_watchdog_count),
        )


def _update_tracking_stall(
    qpos: np.ndarray,
    target: np.ndarray | None,
    *,
    active: bool,
    previous_error_rad: float,
    stale_frames: int,
    progress_epsilon_rad: float,
) -> tuple[int, float, bool]:
    """Advance feedback-stall state without treating a settled hand as stale."""
    if not active or target is None:
        return 0, float("inf"), False
    error_rad = float(np.max(np.abs(qpos - target)))
    if error_rad <= progress_epsilon_rad:
        return 0, error_rad, False
    if np.isfinite(previous_error_rad) and previous_error_rad - error_rad > progress_epsilon_rad:
        stale_frames = 0
    else:
        stale_frames += 1
    return stale_frames, error_rad, True


def hand_loop(shared, config: HandProcessConfig | None = None) -> None:
    """Hand process entry point — reads shared.hand_cmd_ring, servos hand.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).
    """
    from dexmani_real.robot.safety import SafetyState
    from dexmani_real.runtime.status import ComponentPhase, FaultCode
    from dexmani_real.shm.shared_storage import new_frame as _nf
    from dexmani_real.shm.shared_storage import publish_component_status

    cfg = config or HandProcessConfig()
    publish_component_status(shared, "hand", ComponentPhase.LOADING)

    def _mark_startup_failure() -> None:
        publish_component_status(
            shared,
            "hand",
            ComponentPhase.FAULT,
            fault_code=FaultCode.STARTUP_FAILED,
            detail="XHand startup failed; see process log",
        )
        if cfg.startup_failure_is_fatal:
            shared.error_state.value = True
        else:
            logger.warning("hand_loop: XHand unavailable in non-fatal test mode — leaving hand_ready unset")

    try:
        from dexmani_real.robot.xhand import XHand, XHandConfig

        # Per-joint tor_max: index abduction (J3) handles sideways load,
        # benefit from higher current limit (380 vs default 300 mA).
        _tor_max_pj = np.full(HAND_JOINT_SHAPE, 300, dtype=np.int32)
        _tor_max_pj[3] = 380

        hand = XHand(
            XHandConfig(
                tor_max_per_joint=_tor_max_pj,
                ethercat_slave_position=cfg.ethercat_slave_position,
                feedback_bound_tolerance_rad=cfg.feedback_bound_tolerance_rad,
                home_qpos=np.asarray(cfg.home_qpos_rad, dtype=np.float64),
                qpos_min=np.asarray(cfg.qpos_lower_rad, dtype=np.float64),
                qpos_max=np.asarray(cfg.qpos_upper_rad, dtype=np.float64),
                max_delta_rad=cfg.max_delta_rad,
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
            hand.stop()
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
    _frame0["qpos_stale"][0] = 0
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
    shared.hand_heartbeat_s.value = time.monotonic()
    shared.hand_ready.set()
    publish_component_status(shared, "hand", ComponentPhase.READY)
    logger.info("hand_loop: ready")

    rate_mgr = RateManager(cfg.loop_hz)
    last_cmd_seq = 0
    _send_error_counter = RetryCounter(max_consecutive=cfg.send_err_watchdog_frames, label="hand_send")
    _error_state_counter = RetryCounter(max_consecutive=cfg.error_state_watchdog_frames, label="hand_error_state")
    _read_error_counter = RetryCounter(max_consecutive=cfg.error_state_watchdog_frames, label="hand_read_error")
    _last_clear_error_s = 0.0

    # Qpos freshness detection (driver board lockout guard)
    _stale_frames = 0
    _last_fresh_qpos: np.ndarray | None = None
    last_known_qpos = _init_qpos.copy()
    _last_tactile_sum: np.ndarray = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
    _last_tactile_force: np.ndarray = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
    _last_state_source_ns = _initial_source_ns
    _tracking_target: np.ndarray | None = None
    _tracking_prev_error = float("inf")
    _tracking_change_frames = 0

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
        if not worker_validate_hand(data):
            logger.warning("hand_loop: rejected malformed command")
            return None
        return data.copy()

    while shared.is_running.value:
        # Heartbeat — written even when gated (proves we're alive)
        shared.hand_heartbeat_s.value = time.monotonic()

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
                    target_changed = (
                        _tracking_target is None or np.max(np.abs(cmd - _tracking_target)) >= cfg.stale_qpos_delta_rad
                    )
                    target_unmet = np.max(np.abs(last_known_qpos - cmd)) >= cfg.stale_qpos_delta_rad
                    if target_changed or (_tracking_change_frames <= 0 and target_unmet):
                        _tracking_change_frames = 1
                        _tracking_prev_error = float(np.max(np.abs(last_known_qpos - cmd)))
                        _stale_frames = 0
                    _tracking_target = cmd.copy()
                else:
                    _send_error_counter.inc()

            # Send-error watchdog: auto clear_error() after consecutive failures.
            if _send_error_counter.triggered:
                shared.error_state.value = True
                logger.error("hand_loop: persistent send failures — latching global fault")
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

        # Tracking stall requires an unmet target and no feedback progress;
        # a stationary hand that has reached its target is healthy, not stale.
        _stale_frames, _tracking_prev_error, tracking_active = _update_tracking_stall(
            qpos,
            _tracking_target,
            active=connected and _tracking_change_frames > 0,
            previous_error_rad=_tracking_prev_error,
            stale_frames=_stale_frames,
            progress_epsilon_rad=cfg.stale_qpos_delta_rad,
        )
        _tracking_change_frames = int(tracking_active)
        qpos_stale = _stale_frames >= cfg.stale_qpos_frame_limit

        # Retry transient hand errors only for the configured bounded window.
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
        frame["qpos_stale"][0] = int(qpos_stale)
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
    # Shutdown never creates new motion. Homing is an explicit, correlated
    # policy operation; worker cleanup only stops the device and releases the
    # bus after the command loop has been gated.
    stopped_cleanly = False
    try:
        _feedback_stats = hand.feedback_bound_stats
        _feedback_checks = int(_feedback_stats["checks"])
        if _feedback_checks:
            logger.info(
                "hand_loop: feedback bounds checks=%d outside=%d over_tolerance=%d "
                "max=%.3fdeg tolerance=%.3fdeg per_joint_over=%s",
                _feedback_checks,
                int(_feedback_stats["outside_bounds_frames"]),
                int(_feedback_stats["over_tolerance_frames"]),
                float(np.rad2deg(float(_feedback_stats["max_violation_rad"]))),
                float(np.rad2deg(cfg.feedback_bound_tolerance_rad)),
                np.asarray(_feedback_stats["per_joint_over_tolerance_counts"], dtype=np.int64).tolist(),
            )
        if not hand.stop():
            raise RuntimeError(f"XHand stop failed with SDK code {hand.last_action_code!r}")
        hand.disconnect()
        stopped_cleanly = True
    except Exception:
        logger.warning("hand_loop: cleanup failed", exc_info=True)
        shared.error_state.value = True
    if stopped_cleanly:
        publish_component_status(shared, "hand", ComponentPhase.STOPPED)
    else:
        publish_component_status(
            shared,
            "hand",
            ComponentPhase.FAULT,
            fault_code=FaultCode.DEVICE_IO,
            detail="XHand stop/disconnect failed",
        )
    logger.info("hand_loop: exited")
