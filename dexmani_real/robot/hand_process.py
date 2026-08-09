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
from dexmani_real.policy.action_protocol import (
    HAND_COMMAND_DTYPE,
    AckStatus,
    RejectReason,
    command_matches_commit,
    make_ack,
    make_stopped_ack,
    validate_worker_command,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.retry import RetryCounter

logger = get_logger(__name__)


@dataclass
class HandProcessConfig:
    """Configuration for hand_loop."""

    loop_hz: float = field(default_factory=lambda: hand.loop_hz)

    # Some arm-only experimental entry points may probe for an optional XHand
    # and continue with an explicit open-hand collision-model assumption when
    # it is absent. Canonical data collection keeps the default fail-closed.
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
        if home.shape != (12,) or lower.shape != (12,) or upper.shape != (12,):
            raise ValueError("hand process home/limits must have shape (12,)")
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


def hand_loop(shared, config: HandProcessConfig | None = None) -> None:
    """Hand process entry point — reads shared.hand_cmd_ring, servos hand.

    Designed as an mp.Process target. Communicates exclusively through
    SharedStorage (no RPC, no side channels).
    """
    from dexmani_real.robot.safety import SafetyState
    from dexmani_real.runtime.status import ComponentPhase, FaultCode
    from dexmani_real.shm.shared_storage import HAND_STATE_DTYPE as _HS_STATE
    from dexmani_real.shm.shared_storage import HAND_TACTILE_DTYPE as _HS_TACTILE
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
            logger.warning("hand_loop: optional XHand unavailable — leaving hand_ready unset")

    try:
        from dexmani_real.robot.xhand import XHand, XHandConfig

        # Per-joint tor_max: index abduction (J3) handles sideways load,
        # benefit from higher current limit (380 vs default 300 mA).
        _tor_max_pj = np.full(12, 300, dtype=np.int32)
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
                logger.warning("hand_loop: optional XHand connect failed")
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
            logger.warning("hand_loop: optional XHand init failed", exc_info=True)
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
        st = hand.get_state()
        if st is None:
            raise RuntimeError("initial hand state is unavailable")
        _init_qpos = np.asarray(st.get("qpos"), dtype=np.float64)
        if _init_qpos.shape != (12,) or not np.all(np.isfinite(_init_qpos)):
            raise ValueError(f"invalid initial hand qpos shape/values: {_init_qpos.shape}")
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
    last_action_id = 0
    minimum_policy_epoch = int(shared.policy_epoch.value)
    pending_action: np.ndarray | None = None
    pending_committed = False
    _send_error_counter = RetryCounter(max_consecutive=cfg.send_err_watchdog_frames, label="hand_send")
    _error_state_counter = RetryCounter(max_consecutive=cfg.error_state_watchdog_frames, label="hand_error_state")
    _read_error_counter = RetryCounter(max_consecutive=cfg.error_state_watchdog_frames, label="hand_read_error")
    _last_clear_error_s = 0.0

    # Qpos freshness detection (driver board lockout guard)
    _stale_frames = 0
    _last_fresh_qpos: np.ndarray | None = None
    last_known_qpos: np.ndarray = np.zeros(12, dtype=np.float64)
    _last_tactile_sum: np.ndarray = np.zeros((5, 3), dtype=np.float64)
    _last_tactile_force: np.ndarray = np.zeros((5, 120, 3), dtype=np.float64)
    _last_state_source_ns = _initial_source_ns
    _tracking_target: np.ndarray | None = None
    _tracking_prev_error = float("inf")
    _tracking_change_frames = 0

    _last_error_clear_s = 0.0

    def _prepare_latest_command() -> np.ndarray | None:
        """Read and PREPARE a new latest-wins command, if one is available."""
        nonlocal last_action_id, last_cmd_seq, minimum_policy_epoch
        result = shared.hand_cmd_ring.read_latest()
        if result is None:
            return None
        data, _ts, seq = result
        seq_int = int(seq) if isinstance(seq, (int, np.integer)) else 0
        if seq_int == last_cmd_seq:
            return None
        received_ns = time.monotonic_ns()
        minimum_policy_epoch = max(minimum_policy_epoch, int(shared.policy_epoch.value))
        reason = validate_worker_command(
            data,
            dtype=HAND_COMMAND_DTYPE,
            expected_session_generation=int(shared.session_generation.value),
            minimum_policy_epoch=minimum_policy_epoch,
            last_action_id=last_action_id,
            now_monotonic_ns=received_ns,
            joint_lower_rad=np.asarray(hand.config.qpos_min, dtype=np.float64),
            joint_upper_rad=np.asarray(hand.config.qpos_max, dtype=np.float64),
        )
        last_cmd_seq = seq_int
        if reason is not RejectReason.NONE:
            shared.hand_ack_ring.write(
                make_ack(data, AckStatus.REJECTED, reject_reason=reason, received_monotonic_ns=received_ns)
            )
            return None
        shared.hand_ack_ring.write(make_ack(data, AckStatus.RECEIVED, received_monotonic_ns=received_ns))
        last_action_id = int(data["action_id"][0])
        prepared_ns = time.monotonic_ns()
        shared.hand_ack_ring.write(
            make_ack(
                data,
                AckStatus.PREPARED,
                received_monotonic_ns=received_ns,
                prepared_monotonic_ns=prepared_ns,
            )
        )
        return data.copy()

    while shared.is_running.value:
        # Heartbeat — written even when gated (proves we're alive)
        shared.hand_heartbeat_s.value = time.monotonic()

        if shared.estop_request.value:
            break

        # Safety state gate — only process commands in ARMED or RUNNING.
        _safety = shared.safety_state.value
        if _safety in (SafetyState.ARMED, SafetyState.RUNNING) and not shared.error_state.value:

            # Read prepare command ring (latest-wins) and ACK it without moving.
            if pending_action is not None and not pending_committed:
                commit_result = shared.action_commit_ring.read_latest()
                commit = commit_result[0] if commit_result is not None else None
                pending_committed = commit is not None and command_matches_commit(pending_action, commit)
            if pending_action is None or not pending_committed:
                prepared = _prepare_latest_command()
                if prepared is not None:
                    pending_action = prepared
                    pending_committed = False

            execute_action: np.ndarray | None = None
            if pending_action is not None:
                now_ns = time.monotonic_ns()
                pending_epoch_valid = int(pending_action["policy_epoch"][0]) == int(shared.policy_epoch.value)
                pending_session_valid = int(pending_action["session_generation"][0]) == int(
                    shared.session_generation.value
                )
                if not pending_epoch_valid or not pending_session_valid:
                    reason = RejectReason.OLD_EPOCH if not pending_epoch_valid else RejectReason.WRONG_SESSION
                    shared.hand_ack_ring.write(make_ack(pending_action, AckStatus.REJECTED, reject_reason=reason))
                    pending_action = None
                    pending_committed = False
                elif int(pending_action["valid_until_monotonic_ns"][0]) < now_ns:
                    shared.hand_ack_ring.write(
                        make_ack(pending_action, AckStatus.REJECTED, reject_reason=RejectReason.EXPIRED)
                    )
                    pending_action = None
                    pending_committed = False
                elif not pending_committed and int(pending_action["target_monotonic_ns"][0]) <= now_ns:
                    shared.hand_ack_ring.write(
                        make_ack(pending_action, AckStatus.REJECTED, reject_reason=RejectReason.NOT_COMMITTED)
                    )
                    pending_action = None
                else:
                    if pending_committed and now_ns >= int(pending_action["target_monotonic_ns"][0]):
                        execute_action = pending_action
                        pending_action = None
                        pending_committed = False

            if execute_action is not None:
                cmd = np.asarray(execute_action["qpos_cmd"][0], dtype=np.float64)
                try:
                    sent = hand.send_action(cmd)
                except Exception:
                    sent = False
                    logger.warning("hand_loop: committed send_action raised", exc_info=True)
                if sent:
                    _send_error_counter.reset()
                    shared.hand_ack_ring.write(
                        make_ack(execute_action, AckStatus.APPLIED, applied_monotonic_ns=time.monotonic_ns())
                    )
                    if _tracking_target is None or np.max(np.abs(cmd - _tracking_target)) >= cfg.stale_qpos_delta_rad:
                        _tracking_change_frames = cfg.stale_qpos_frame_limit
                    _tracking_target = cmd.copy()
                else:
                    _send_error_counter.inc()
                    shared.hand_ack_ring.write(
                        make_ack(
                            execute_action,
                            AckStatus.SDK_FAILED,
                            reject_reason=RejectReason.SDK_ERROR,
                            sdk_code=int(hand.last_action_code or -1),
                            applied_monotonic_ns=time.monotonic_ns(),
                        )
                    )

            # Freeing the committed slot and preparing the next latest-wins
            # command in the same worker tick avoids an extra 30 Hz delay that
            # can otherwise exceed the coordinator's prepare timeout.
            if pending_action is None:
                prepared = _prepare_latest_command()
                if prepared is not None:
                    pending_action = prepared
                    pending_committed = False

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
            qpos = np.asarray(_raw_qpos if _raw_qpos is not None else np.zeros(12), dtype=np.float64)
            _raw_current = st.get("current")
            current = np.asarray(_raw_current if _raw_current is not None else np.zeros(12), dtype=np.float64)
            _raw_ts = st.get("tactile_force_sum")
            tactile_sum = np.asarray(_raw_ts if _raw_ts is not None else np.zeros((5, 3)), dtype=np.float64)
            _raw_tf = st.get("tactile_force")
            tactile_force = np.asarray(_raw_tf if _raw_tf is not None else np.zeros((5, 120, 3)), dtype=np.float64)
            _raw_tc = st.get("tactile_contact")
            tactile_contact = np.asarray(_raw_tc if _raw_tc is not None else np.zeros(5, dtype=bool), dtype=bool)
            expected_shapes = (
                ("qpos", qpos, (12,)),
                ("current", current, (12,)),
                ("tactile_force_sum", tactile_sum, (5, 3)),
                ("tactile_force", tactile_force, (5, 120, 3)),
                ("tactile_contact", tactile_contact, (5,)),
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
                _raw_cbe if _raw_cbe is not None else np.zeros(12, dtype=np.int32), dtype=np.int32
            )
            _raw_jbe = st.get("jointboard_err")
            jointboard_err = np.asarray(
                _raw_jbe if _raw_jbe is not None else np.zeros(12, dtype=np.int32), dtype=np.int32
            )
            _raw_tbe = st.get("tipboard_err")
            tipboard_err = np.asarray(
                _raw_tbe if _raw_tbe is not None else np.zeros(12, dtype=np.int32), dtype=np.int32
            )
        except Exception:
            logger.warning("hand_loop: get_state failed", exc_info=True)
            qpos = last_known_qpos.copy()
            current = np.zeros(12)
            tactile_sum = _last_tactile_sum.copy()
            tactile_force = _last_tactile_force.copy()
            tactile_contact = np.zeros(5, dtype=bool)
            connected = False
            error_state = False  # transient comm glitch — do NOT fabricate error_state
            commboard_err = np.zeros(12, dtype=np.int32)
            jointboard_err = np.zeros(12, dtype=np.int32)
            tipboard_err = np.zeros(12, dtype=np.int32)

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

        # Tracking stall requires a changing target and no feedback progress;
        # a stationary hand is healthy, not stale.
        if not connected or _tracking_target is None or _tracking_change_frames <= 0:
            _stale_frames = 0
            _tracking_prev_error = float("inf")
        else:
            tracking_error = float(np.max(np.abs(qpos - _tracking_target)))
            if _tracking_prev_error - tracking_error > cfg.stale_qpos_delta_rad:
                _stale_frames = 0
            else:
                _stale_frames += 1
            _tracking_prev_error = tracking_error
            _tracking_change_frames -= 1
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
                logger.error(
                    "hand_loop: hand error_state persisted after %d retries — latching global error_state",
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
        frame["source_monotonic_ns"][0] = _last_state_source_ns
        frame["publish_monotonic_ns"][0] = time.monotonic_ns()
        frame["state_valid"][0] = int(connected)
        frame["send_healthy"][0] = int(not _send_error_counter.triggered)
        frame["read_healthy"][0] = int(not _read_error_counter.triggered)
        frame["timestamp"][0] = _last_state_source_ns / 1e9
        shared.hand_state_ring.write(frame)

        # Every successful read is a source sample, including release/no-contact.
        if connected:
            tf = _nf(_HS_TACTILE)
            tf["tactile_force"][0] = tactile_force
            tf["source_monotonic_ns"][0] = _last_state_source_ns
            tf["fresh"][0] = 1
            tf["calibrated"][0] = int(hand.tactile_calibrated)
            # The SDK conversion provenance has not been independently
            # established on hardware.  Preserve the values, but label their
            # unit as unknown instead of guessing Newtons.
            tf["unit_code"][0] = 0
            shared.hand_tactile_ring.write(tf)

        # Rate limit (absolute-deadline scheduling, consistent with arm_loop/policy_loop)
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
        shared.hand_ack_ring.write(make_stopped_ack())
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
