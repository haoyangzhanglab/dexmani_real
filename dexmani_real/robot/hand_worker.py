"""Single-owner XHand servo worker with ordered FIFO consumption.

Each tick reads one state, publishes it (or a clearly stale previous state),
then consumes at most one ordered command-FIFO record and sends at most one
rate-bounded setpoint. The endpoint cursor advances only on exact-endpoint
SDK acceptance: an intermediate slew setpoint or a CRC-unconfirmed send
leaves the record pending for the next tick without updating the acceptance
identity or the command-space reference. RUNNING motion advances from the
last SDK-accepted setpoint; ARMED homing remains bounded from measurement.
A CRC response keeps the target unacknowledged and the worker running; other
rejected SDK commands latch the shared fault so publishers cannot mistake
silence for successful application.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.config.defaults import HandParams
from dexmani_real.ipc.command_stream import CommandStreamConsumer
from dexmani_real.ipc.schema import (
    HAND_STATE_DTYPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.robot.command_validation import check_worker_hand_target
from dexmani_real.runtime.safety import (
    SafetyState,
    coupled_command_may_cross_sdk,
    read_motion_permit,
)
from dexmani_real.utils.limits import limit_hand_target_delta
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


class _HandWorkerFault(RuntimeError):
    """Unsafe hand target or SDK rejection; the loop latches the shared fault."""


@dataclass
class _HandCommandAck:
    """Exact-endpoint acceptance identity and command-space slew reference."""

    last_sdk_accepted_qpos: np.ndarray
    accepted_target_action_id: int = 0
    accepted_target_monotonic_ns: int = 0
    accepted_target_generation: int = 0
    accepted_target_sequence: int = 0
    last_sdk_setpoint_accepted_monotonic_ns: int = 0


def _limited_hand_setpoint(
    target_qpos: np.ndarray,
    *,
    measured_qpos: np.ndarray,
    last_sdk_accepted_qpos: np.ndarray,
    is_running: bool,
    max_delta_rad_per_tick: float | np.ndarray,
) -> np.ndarray:
    """Bound one SDK setpoint from command history or measured state.

    A RUNNING endpoint may be held by contact, so its setpoint slew is
    measured from the last SDK-accepted command rather than encoder tracking.
    ARMED homing stays measured-state bounded because home acceptance is not
    measured convergence.
    """
    reference_qpos = last_sdk_accepted_qpos if is_running else measured_qpos
    return limit_hand_target_delta(
        target_qpos,
        reference_qpos,
        max_delta_rad_per_tick,
    )


def _consume_one_hand_command(
    shared: Any,
    hand: Any,
    send_status_enum: Any,
    consumer: CommandStreamConsumer,
    permit: Any,
    measured_qpos: np.ndarray,
    ack: _HandCommandAck,
    *,
    mechanical_lower: np.ndarray,
    mechanical_upper: np.ndarray,
    max_delta_rad_per_tick: float | np.ndarray,
    phase_ms: dict[str, float] | None = None,
) -> None:
    """Consume at most one ordered FIFO record for the hand this tick.

    At most one SDK send happens per tick — the worker never drains the queue
    to catch up. The FIFO cursor advances only after the exact endpoint was
    ACCEPTED; an intermediate slew setpoint or a CRC-unconfirmed send retries
    the same record on the next tick. A blocked final fence also keeps the
    record pending (any real revocation advanced the generation and resyncs
    the cursor). Raises :class:`_HandWorkerFault` for an unsafe target or an
    SDK rejection; the caller latches the shared fault and exits fail-fast.
    """

    def _phase(name: str, started_ns: int) -> None:
        if phase_ms is not None:
            phase_ms[name] = (time.monotonic_ns() - started_ns) / 1e6

    read_started_ns = time.monotonic_ns()
    if consumer.resync_if_stale_generation(permit.run_generation):
        # A new epoch invalidates the queued backlog; re-anchor the
        # command-space slew reference to fresh measured state.
        ack.last_sdk_accepted_qpos = measured_qpos.copy()
    if consumer.generation != permit.run_generation:
        return  # Resync overtook this tick's permit; do not consume a new epoch.
    record = consumer.next_record()
    _phase("command_read", read_started_ns)
    if record is None:
        # EMPTY is a wait, never a fault.
        return
    command, sequence = record
    sequence_int = int(sequence)
    command_generation = int(command["run_generation"][0])
    if not bool(command["hand_present"][0]):
        # An absent actuator advances only its consumer; no SDK send and no
        # acceptance is implied.
        consumer.advance()
        return
    prepare_started_ns = time.monotonic_ns()
    action_id = int(command["action_id"][0])
    target = np.asarray(command["hand_qpos"][0], dtype=np.float64)
    issue = check_worker_hand_target(
        target,
        mechanical_lower_rad=mechanical_lower,
        mechanical_upper_rad=mechanical_upper,
    )
    bounded: np.ndarray | None = None
    if issue is None:
        bounded = _limited_hand_setpoint(
            target,
            measured_qpos=measured_qpos,
            last_sdk_accepted_qpos=ack.last_sdk_accepted_qpos,
            is_running=permit.state is SafetyState.RUNNING,
            max_delta_rad_per_tick=max_delta_rad_per_tick,
        )
        issue = check_worker_hand_target(
            bounded,
            mechanical_lower_rad=mechanical_lower,
            mechanical_upper_rad=mechanical_upper,
        )
    _phase("target_prepare", prepare_started_ns)
    # This remains the final authority check before the SDK call.
    fence_started_ns = time.monotonic_ns()
    allowed = coupled_command_may_cross_sdk(
        shared, run_generation=command_generation
    )
    _phase("command_fence", fence_started_ns)
    if not allowed:
        return
    if issue is not None:
        raise _HandWorkerFault(f"unsafe action_id={action_id}: {issue}")
    assert bounded is not None
    send_started_ns = time.monotonic_ns()
    send_status = hand.send_action(bounded)
    _phase("send_command", send_started_ns)
    ack_started_ns = time.monotonic_ns()
    if send_status is send_status_enum.ACCEPTED:
        accepted_now_ns = time.monotonic_ns()
        ack.last_sdk_accepted_qpos = bounded.copy()
        ack.last_sdk_setpoint_accepted_monotonic_ns = accepted_now_ns
        # ACK denotes SDK acceptance of the exact IPC endpoint, not
        # physical convergence or acceptance of an intermediate step.
        if np.array_equal(bounded, target):
            ack.accepted_target_action_id = action_id
            ack.accepted_target_monotonic_ns = accepted_now_ns
            ack.accepted_target_generation = command_generation
            ack.accepted_target_sequence = sequence_int
            # The exact endpoint was accepted: the record is fully processed
            # and its FIFO slot is released to the publisher's watermark.
            consumer.advance()
        # An intermediate slew setpoint does NOT advance the endpoint cursor;
        # the same record is retried next tick from the updated command-space
        # reference.
    elif send_status is send_status_enum.REJECTED:
        raise _HandWorkerFault(f"SDK rejected action_id={action_id}")
    # CRC_UNCONFIRMED deliberately leaves the exact target, the endpoint
    # cursor, and the command-space reference unacknowledged; the same
    # record is retried next tick.
    _phase("command_ack", ack_started_ns)


def _safe_disconnect(hand: Any) -> bool:
    """Disconnect the hand driver, tolerating a never-connected instance."""
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
    """Warn on nonzero registers and log clearing without decoding vendor bits.

    These diagnostics do not grant motion authority or replace feedback checks.
    """
    for name in ("commboard_err", "jointboard_err", "tipboard_err"):
        prev = previous[name]
        cur = current[name]
        if prev.shape == cur.shape:
            for joint in range(int(cur.shape[0])):
                if prev[joint] != cur[joint]:
                    log = logger.warning if int(cur[joint]) != 0 else logger.info
                    log(
                        "hand board register %s[%d] 0x%08x -> 0x%08x (%s)",
                        name,
                        joint,
                        int(prev[joint]),
                        int(cur[joint]),
                        "nonzero" if int(cur[joint]) != 0 else "cleared",
                    )
    return {name: current[name].copy() for name in previous}


def _publish_feedback(
    shared: Any,
    *,
    qpos: np.ndarray,
    current_ma: np.ndarray,
    tactile_aggregate: np.ndarray,
    tactile_aggregate_valid: bool,
    tactile_dense: np.ndarray,
    tactile_dense_valid: bool,
    tactile_calibrated: bool,
    connected: bool,
    read_failed: bool,
    ack: _HandCommandAck,
    commboard_err: np.ndarray,
    jointboard_err: np.ndarray,
    tipboard_err: np.ndarray,
    source_monotonic_ns: int,
) -> None:
    """Serialize one hand-state record carrying qpos/current and both tactile payloads."""
    from dexmani_real.ipc.channels import new_frame

    source_ns = max(0, int(source_monotonic_ns))
    frame = new_frame(HAND_STATE_DTYPE)
    frame["qpos"][0] = qpos
    frame["current"][0] = current_ma
    frame["tactile_aggregate"][0] = np.asarray(tactile_aggregate, dtype=np.float32)
    frame["tactile_dense"][0] = np.asarray(tactile_dense, dtype=np.float32)
    # Session software-bias readiness gates both tactile representations at the
    # worker boundary: a failed zeroing leaves them invalid while joint control
    # continues. This is the one-place combination of per-read aggregate/dense
    # validity with startup readiness.
    frame["tactile_aggregate_valid"][0] = int(
        tactile_calibrated and tactile_aggregate_valid
    )
    frame["tactile_dense_valid"][0] = int(tactile_calibrated and tactile_dense_valid)
    frame["connected"][0] = int(connected)
    frame["qpos_stale"][0] = int(read_failed)
    frame["accepted_target_action_id"][0] = int(ack.accepted_target_action_id)
    frame["accepted_target_monotonic_ns"][0] = int(ack.accepted_target_monotonic_ns)
    frame["accepted_target_generation"][0] = int(ack.accepted_target_generation)
    frame["accepted_target_sequence"][0] = int(ack.accepted_target_sequence)
    frame["last_sdk_setpoint_accepted_monotonic_ns"][0] = int(
        ack.last_sdk_setpoint_accepted_monotonic_ns
    )
    frame["commboard_err"][0] = commboard_err
    frame["jointboard_err"][0] = jointboard_err
    frame["tipboard_err"][0] = tipboard_err
    frame["source_monotonic_ns"][0] = source_ns
    frame["publish_monotonic_ns"][0] = time.monotonic_ns()
    frame["state_valid"][0] = int(connected and not read_failed)
    frame["timestamp"][0] = source_ns / 1e9
    shared.hand_state_ring.write(frame)


def hand_loop(
    shared: Any,
    config: HandParams,
    state_read_failure_timeout_s: float,
) -> None:
    """Run one XHand worker; all SDK objects remain in this process."""
    from dexmani_real.robot.drivers.xhand import XHand, XHandSendStatus

    logger.debug("hand_loop: LOADING")
    hand: XHand | None = None
    ready = False
    try:
        try:
            hand = XHand(config)
            hand.connect()
        except Exception:
            logger.error("hand_loop: init failed", exc_info=True)
            shared.error_state.value = True
            return

        try:
            hand.calibrate_tactile()
        except Exception:
            logger.warning("hand_loop: tactile calibration raised", exc_info=True)

        initial_state = hand.get_state()
        if initial_state is None:
            logger.error("hand_loop: cannot publish a valid initial state")
            shared.error_state.value = True
            return
        if not hand.is_connected:
            logger.error("hand_loop: initial state reports a disconnected hand")
            shared.error_state.value = True
            return

        last_state = initial_state
        last_source_ns = time.monotonic_ns()
        read_failure_started_s: float | None = None
        ack = _HandCommandAck(last_sdk_accepted_qpos=initial_state.qpos.copy())
        # Attach as the hand consumer of the ordered command FIFO before READY
        # so the publisher's capacity watermark sees this consumer from the
        # first committable command on.
        consumer = CommandStreamConsumer(shared, shared.hand_cmd_consumed_sequence)
        _publish_feedback(
            shared,
            qpos=initial_state.qpos,
            current_ma=initial_state.current_ma,
            tactile_aggregate=initial_state.tactile_aggregate,
            tactile_aggregate_valid=initial_state.tactile_aggregate_valid,
            tactile_dense=initial_state.tactile_dense,
            tactile_dense_valid=initial_state.tactile_dense_valid,
            tactile_calibrated=hand.tactile_calibrated,
            connected=True,
            read_failed=False,
            ack=ack,
            commboard_err=initial_state.commboard_err,
            jointboard_err=initial_state.jointboard_err,
            tipboard_err=initial_state.tipboard_err,
            source_monotonic_ns=last_source_ns,
        )

        previous_board_errors = {
            name: getattr(initial_state, name).copy()
            for name in ("commboard_err", "jointboard_err", "tipboard_err")
        }
        shared.set_heartbeat("hand", time.monotonic())
        shared.set_ready("hand")
        ready = True
        logger.info("hand_loop: ready")

        rate_mgr = LoopRate(config.loop_hz, label="hand")
        mechanical_lower = np.asarray(config.mechanical_qpos_min_rad, dtype=np.float64)
        mechanical_upper = np.asarray(config.mechanical_qpos_max_rad, dtype=np.float64)

        def wait_for_tick() -> None:
            # Measure host elapsed work, including any descheduling within a
            # phase. Residual time is uninstrumented work/measurement overhead,
            # not a separate estimate of OS scheduling delay.
            work_ms = (time.monotonic_ns() - tick_started_ns) / 1e6
            phase_ms["unattributed"] = work_ms - sum(phase_ms.values())
            phase_ms["work"] = work_ms
            rate_mgr.wait(phase_ms=phase_ms)

        while shared.is_running.value:
            tick_started_ns = time.monotonic_ns()
            phase_ms = dict.fromkeys(
                (
                    "tick_setup", "read_state", "board_error_log", "feedback_publish",
                    "command_read", "permit_read", "target_prepare", "command_fence",
                    "send_command", "command_ack",
                ),
                0.0,
            )
            shared.set_heartbeat("hand", time.monotonic())
            if shared.estop_request.value:
                break

            read_started_ns = time.monotonic_ns()
            phase_ms["tick_setup"] = (read_started_ns - tick_started_ns) / 1e6
            state = hand.get_state()
            phase_ms["read_state"] = (time.monotonic_ns() - read_started_ns) / 1e6
            if state is None:
                now_s = time.monotonic()
                if read_failure_started_s is None:
                    read_failure_started_s = now_s
                feedback_started_ns = time.monotonic_ns()
                _publish_feedback(
                    shared,
                    qpos=last_state.qpos,
                    current_ma=last_state.current_ma,
                    tactile_aggregate=np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64),
                    tactile_aggregate_valid=False,
                    tactile_dense=np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64),
                    tactile_dense_valid=False,
                    tactile_calibrated=hand.tactile_calibrated,
                    connected=hand.is_connected,
                    read_failed=True,
                    ack=ack,
                    commboard_err=last_state.commboard_err,
                    jointboard_err=last_state.jointboard_err,
                    tipboard_err=last_state.tipboard_err,
                    source_monotonic_ns=last_source_ns,
                )
                phase_ms["feedback_publish"] = (
                    time.monotonic_ns() - feedback_started_ns
                ) / 1e6
                if now_s - read_failure_started_s >= state_read_failure_timeout_s:
                    shared.error_state.value = True
                    raise RuntimeError(
                        "hand state reads failed for "
                        f"{now_s - read_failure_started_s:.3f}s"
                    )
                wait_for_tick()
                continue

            read_failure_started_s = None
            last_state = state
            last_source_ns = time.monotonic_ns()
            board_log_started_ns = time.monotonic_ns()
            previous_board_errors = _log_board_error_transitions(
                previous_board_errors,
                {
                    "commboard_err": state.commboard_err,
                    "jointboard_err": state.jointboard_err,
                    "tipboard_err": state.tipboard_err,
                },
            )
            feedback_started_ns = time.monotonic_ns()
            phase_ms["board_error_log"] = (
                feedback_started_ns - board_log_started_ns
            ) / 1e6
            _publish_feedback(
                shared,
                qpos=state.qpos,
                current_ma=state.current_ma,
                tactile_aggregate=state.tactile_aggregate,
                tactile_aggregate_valid=state.tactile_aggregate_valid,
                tactile_dense=state.tactile_dense,
                tactile_dense_valid=state.tactile_dense_valid,
                tactile_calibrated=hand.tactile_calibrated,
                connected=hand.is_connected,
                read_failed=False,
                ack=ack,
                commboard_err=state.commboard_err,
                jointboard_err=state.jointboard_err,
                tipboard_err=state.tipboard_err,
                source_monotonic_ns=last_source_ns,
            )
            phase_ms["feedback_publish"] = (
                time.monotonic_ns() - feedback_started_ns
            ) / 1e6

            permit_started_ns = time.monotonic_ns()
            permit = read_motion_permit(shared)
            phase_ms["permit_read"] = (time.monotonic_ns() - permit_started_ns) / 1e6
            if not permit.allows_motion or shared.error_state.value:
                wait_for_tick()
                continue
            try:
                _consume_one_hand_command(
                    shared,
                    hand,
                    XHandSendStatus,
                    consumer,
                    permit,
                    state.qpos,
                    ack,
                    mechanical_lower=mechanical_lower,
                    mechanical_upper=mechanical_upper,
                    max_delta_rad_per_tick=config.hand_max_delta_rad_per_tick,
                    phase_ms=phase_ms,
                )
            except _HandWorkerFault as exc:
                logger.error("hand_loop: %s; latching runtime fault", exc)
                shared.error_state.value = True
                return

            wait_for_tick()
    finally:
        if not _safe_disconnect(hand):
            logger.error("hand_loop: XHand disconnect failed")
            shared.error_state.value = True
        elif ready:
            logger.debug("hand_loop: STOPPED")
