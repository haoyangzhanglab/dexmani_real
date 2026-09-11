"""Single-owner XHand servo worker with latest-target, fixed-grid feedback.

Each tick reads one state, publishes it (or a clearly stale previous state),
then sends at most one rate-bounded target. RUNNING motion advances from the
last SDK-accepted setpoint; ARMED homing remains bounded from measurement. A
CRC response keeps the target unacknowledged and the worker running; other
rejected SDK commands latch the shared fault so publishers cannot mistake
silence for successful application.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.config.defaults import HandParams
from dexmani_real.ipc.schema import (
    HAND_STATE_DTYPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.robot.command_validation import check_worker_hand_target
from dexmani_real.utils.limits import limit_hand_target_delta
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


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
    """Log board-register transitions without assigning them safety meaning."""
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
    accepted_target_action_id: int,
    accepted_target_monotonic_ns: int,
    last_sdk_setpoint_accepted_monotonic_ns: int,
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
    # continues. This is the one-place combination of per-read validity with
    # startup readiness (see xhand_tactile_research_simplification_plan §5.4).
    frame["tactile_aggregate_valid"][0] = int(
        tactile_calibrated and tactile_aggregate_valid
    )
    frame["tactile_dense_valid"][0] = int(tactile_calibrated and tactile_dense_valid)
    frame["connected"][0] = int(connected)
    frame["qpos_stale"][0] = int(read_failed)
    frame["accepted_target_action_id"][0] = int(accepted_target_action_id)
    frame["accepted_target_monotonic_ns"][0] = int(
        accepted_target_monotonic_ns
    )
    frame["last_sdk_setpoint_accepted_monotonic_ns"][0] = int(
        last_sdk_setpoint_accepted_monotonic_ns
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
    from dexmani_real.runtime.safety import (
        CoupledCommandTicket,
        SafetyState,
        coupled_command_ticket_allows_execution,
        read_motion_permit,
    )

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
        last_sdk_accepted_qpos = initial_state.qpos.copy()
        accepted_target_action_id = 0
        accepted_target_monotonic_ns = 0
        last_sdk_setpoint_accepted_monotonic_ns = 0
        last_exact_target_sequence = 0
        command_generation: int | None = None
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
            accepted_target_action_id=accepted_target_action_id,
            accepted_target_monotonic_ns=accepted_target_monotonic_ns,
            last_sdk_setpoint_accepted_monotonic_ns=(
                last_sdk_setpoint_accepted_monotonic_ns
            ),
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

        while shared.is_running.value:
            shared.set_heartbeat("hand", time.monotonic())
            if shared.estop_request.value:
                break

            state = hand.get_state()
            if state is None:
                now_s = time.monotonic()
                if read_failure_started_s is None:
                    read_failure_started_s = now_s
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
                    accepted_target_action_id=accepted_target_action_id,
                    accepted_target_monotonic_ns=accepted_target_monotonic_ns,
                    last_sdk_setpoint_accepted_monotonic_ns=(
                        last_sdk_setpoint_accepted_monotonic_ns
                    ),
                    commboard_err=last_state.commboard_err,
                    jointboard_err=last_state.jointboard_err,
                    tipboard_err=last_state.tipboard_err,
                    source_monotonic_ns=last_source_ns,
                )
                if now_s - read_failure_started_s >= state_read_failure_timeout_s:
                    shared.error_state.value = True
                    raise RuntimeError(
                        "hand state reads failed for "
                        f"{now_s - read_failure_started_s:.3f}s"
                    )
                rate_mgr.wait()
                continue

            read_failure_started_s = None
            last_state = state
            last_source_ns = time.monotonic_ns()
            previous_board_errors = _log_board_error_transitions(
                previous_board_errors,
                {
                    "commboard_err": state.commboard_err,
                    "jointboard_err": state.jointboard_err,
                    "tipboard_err": state.tipboard_err,
                },
            )
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
                accepted_target_action_id=accepted_target_action_id,
                accepted_target_monotonic_ns=accepted_target_monotonic_ns,
                last_sdk_setpoint_accepted_monotonic_ns=(
                    last_sdk_setpoint_accepted_monotonic_ns
                ),
                commboard_err=state.commboard_err,
                jointboard_err=state.jointboard_err,
                tipboard_err=state.tipboard_err,
                source_monotonic_ns=last_source_ns,
            )

            result = shared.coupled_cmd_ring.read_latest()
            if result is None:
                rate_mgr.wait()
                continue
            command, _published_ns, sequence = result
            sequence_int = int(sequence)
            if not bool(command["hand_present"][0]):
                rate_mgr.wait()
                continue
            ticket = CoupledCommandTicket(
                run_generation=int(command["run_generation"][0]),
                ring_sequence=sequence_int,
                valid_until_monotonic_ns=int(command["valid_until_monotonic_ns"][0]),
            )
            permit = read_motion_permit(shared)
            if permit.run_generation != ticket.run_generation:
                rate_mgr.wait()
                continue
            if permit.run_generation != command_generation:
                last_exact_target_sequence = 0
                last_sdk_accepted_qpos = state.qpos.copy()
                command_generation = permit.run_generation
            if sequence_int == last_exact_target_sequence:
                # An accepted exact target is an endpoint event, not a
                # level-triggered command.  Retries remain allowed only until
                # the SDK has accepted the exact endpoint.
                rate_mgr.wait()
                continue
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
                    measured_qpos=state.qpos,
                    last_sdk_accepted_qpos=last_sdk_accepted_qpos,
                    is_running=permit.state is SafetyState.RUNNING,
                    max_delta_rad_per_tick=config.hand_max_delta_rad_per_tick,
                )
                issue = check_worker_hand_target(
                    bounded,
                    mechanical_lower_rad=mechanical_lower,
                    mechanical_upper_rad=mechanical_upper,
                )
            # This is the sole command-authority fence and the final operation
            # before an otherwise valid setpoint crosses the XHand SDK boundary.
            if not coupled_command_ticket_allows_execution(shared, ticket=ticket):
                rate_mgr.wait()
                continue
            if issue is not None:
                logger.error(
                    "hand_loop: unsafe action_id=%d: %s; latching runtime fault",
                    action_id,
                    issue,
                )
                shared.error_state.value = True
                return
            assert bounded is not None
            send_status = hand.send_action(bounded)
            if send_status is XHandSendStatus.ACCEPTED:
                accepted_now_ns = time.monotonic_ns()
                last_sdk_accepted_qpos = bounded.copy()
                last_sdk_setpoint_accepted_monotonic_ns = accepted_now_ns
                # ACK denotes SDK acceptance of the exact IPC endpoint, not
                # physical convergence or acceptance of an intermediate step.
                if np.array_equal(bounded, target):
                    accepted_target_action_id = action_id
                    accepted_target_monotonic_ns = accepted_now_ns
                    last_exact_target_sequence = sequence_int
            elif send_status is XHandSendStatus.REJECTED:
                logger.error(
                    "hand_loop: SDK rejected action_id=%d; latching runtime fault",
                    action_id,
                )
                shared.error_state.value = True
                return
            # CRC_UNCONFIRMED deliberately leaves both the action and its
            # command-space reference unacknowledged.

            rate_mgr.wait()
    finally:
        if not _safe_disconnect(hand):
            logger.error("hand_loop: XHand disconnect failed")
            shared.error_state.value = True
        elif ready:
            logger.debug("hand_loop: STOPPED")
