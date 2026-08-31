"""Single-owner XHand servo worker with latest-target, fixed-grid feedback.

Each tick reads one state, publishes it (or a clearly stale previous state),
then sends at most one measured-state-bounded target. A CRC response keeps the
target unacknowledged and the worker running; other rejected SDK commands latch
the shared fault so publishers cannot mistake silence for successful application.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from dexmani_real.config.defaults import HandParams
from dexmani_real.ipc.schema import (
    HAND_CONTACT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.robot.command_validation import check_worker_hand_command
from dexmani_real.utils.limits import limit_hand_target_delta
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


def expired_hand_command_diagnostics(
    command: np.ndarray,
    measured_qpos: np.ndarray,
    *,
    now_monotonic_ns: int,
    max_delta_rad_per_tick: float | np.ndarray,
) -> str:
    """Render bounded numeric context for one already-validated expired command."""
    target = np.asarray(command["hand_qpos"][0], dtype=np.float64)
    measured = np.asarray(measured_qpos, dtype=np.float64)
    max_delta = np.broadcast_to(
        np.asarray(max_delta_rad_per_tick, dtype=np.float64), HAND_JOINT_SHAPE
    )
    delta = np.abs(target - measured)
    worst_joint = int(np.argmax(delta))
    created_ns = int(command["created_monotonic_ns"][0])
    valid_until_ns = int(command["valid_until_monotonic_ns"][0])
    now_ns = int(now_monotonic_ns)
    return json.dumps(
        {
            "action_id": int(command["action_id"][0]),
            "age_ms": round((now_ns - created_ns) / 1e6, 3),
            "delivery_window_ms": round((valid_until_ns - created_ns) / 1e6, 3),
            "overdue_ms": round((now_ns - valid_until_ns) / 1e6, 3),
            "max_delta_rad_per_tick": float(max_delta[worst_joint]),
            "max_target_minus_measured_rad": round(float(delta[worst_joint]), 6),
            "minimum_ramp_ticks": int(
                np.ceil(delta[worst_joint] / max_delta[worst_joint])
            ),
            "worst_joint_index": worst_joint,
            "worst_joint_measured_rad": round(float(measured[worst_joint]), 6),
            "worst_joint_target_rad": round(float(target[worst_joint]), 6),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
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
    frame["unit_code"][0] = 0
    return frame


def _publish_feedback(
    shared: Any,
    *,
    qpos: np.ndarray,
    current_ma: np.ndarray,
    tactile_sum: np.ndarray,
    tactile_sum_valid: bool,
    tactile_contact: np.ndarray,
    tactile_force: np.ndarray,
    tactile_valid: bool,
    tactile_calibrated: bool,
    connected: bool,
    read_failed: bool,
    accepted_target_action_id: int,
    commboard_err: np.ndarray,
    jointboard_err: np.ndarray,
    tipboard_err: np.ndarray,
    source_monotonic_ns: int,
) -> None:
    """Serialize one hand-state and tactile feedback pair."""
    from dexmani_real.ipc.channels import new_frame

    source_ns = max(0, int(source_monotonic_ns))
    frame = new_frame(HAND_STATE_DTYPE)
    frame["qpos"][0] = qpos
    frame["current"][0] = current_ma
    frame["tactile_sum"][0] = tactile_sum
    frame["tactile_sum_valid"][0] = int(tactile_sum_valid)
    frame["tactile_contact"][0] = tactile_contact
    frame["connected"][0] = int(connected)
    frame["qpos_stale"][0] = int(read_failed)
    frame["accepted_target_action_id"][0] = int(accepted_target_action_id)
    frame["commboard_err"][0] = commboard_err
    frame["jointboard_err"][0] = jointboard_err
    frame["tipboard_err"][0] = tipboard_err
    frame["source_monotonic_ns"][0] = source_ns
    frame["publish_monotonic_ns"][0] = time.monotonic_ns()
    frame["state_valid"][0] = int(connected and not read_failed)
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


def hand_loop(shared: Any, config: HandParams) -> None:
    """Run one XHand worker; all SDK objects remain in this process."""
    from dexmani_real.robot.xhand import XHand, XHandSendStatus
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
            if hasattr(shared, "hand_device_identity"):
                identity = getattr(hand, "device_identity", {"backend": "unavailable"})
                encoded = json.dumps(
                    identity, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                shared.hand_device_identity.value = encoded[:1023].ljust(1024, b"\x00")
        except Exception:
            logger.error("hand_loop: init failed", exc_info=True)
            shared.error_state.value = True
            return

        try:
            hand.calibrate_tactile()
        except Exception:
            logger.warning("hand_loop: tactile calibration raised", exc_info=True)

        try:
            reset_status = hand.reset_home()
        except Exception:
            logger.error("hand_loop: startup reset-home command raised", exc_info=True)
            shared.error_state.value = True
            return
        if reset_status is XHandSendStatus.REJECTED:
            logger.error("hand_loop: startup reset-home command was rejected")
            shared.error_state.value = True
            return
        if reset_status is XHandSendStatus.CRC_UNCONFIRMED:
            logger.warning(
                "hand_loop: startup reset-home delivery is unconfirmed after CRC; "
                "continuing with live state"
            )
        else:
            logger.info("hand_loop: startup reset-home command accepted")

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
        accepted_target_action_id = 0
        sdk_send_attempts = 0
        exact_target_accepts = 0
        crc_unconfirmed = 0
        duplicate_skips = 0
        sdk_rejections = 0
        last_exact_target_sequence = 0
        stats_generation: int | None = None
        stats_generation_was_running = False
        last_running_stats: tuple[int, int, int, int, int] | None = None
        _publish_feedback(
            shared,
            qpos=initial_state.qpos,
            current_ma=initial_state.current_ma,
            tactile_sum=initial_state.tactile_sum,
            tactile_sum_valid=initial_state.tactile_sum_valid,
            tactile_contact=initial_state.tactile_contact,
            tactile_force=initial_state.tactile_force,
            tactile_valid=initial_state.tactile_valid,
            tactile_calibrated=hand.tactile_calibrated,
            connected=True,
            read_failed=False,
            accepted_target_action_id=accepted_target_action_id,
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
        operational_lower = np.asarray(config.qpos_min_rad, dtype=np.float64)
        operational_upper = np.asarray(config.qpos_max_rad, dtype=np.float64)
        mechanical_lower = np.asarray(config.mechanical_qpos_min_rad, dtype=np.float64)
        mechanical_upper = np.asarray(config.mechanical_qpos_max_rad, dtype=np.float64)
        last_rejected_ring_sequence = 0

        while shared.is_running.value:
            shared.set_heartbeat("hand", time.monotonic())
            if shared.estop_request.value:
                break

            state = hand.get_state()
            if state is None:
                _publish_feedback(
                    shared,
                    qpos=last_state.qpos,
                    current_ma=last_state.current_ma,
                    tactile_sum=np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64),
                    tactile_sum_valid=False,
                    tactile_contact=np.zeros(HAND_CONTACT_SHAPE, dtype=bool),
                    tactile_force=np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64),
                    tactile_valid=False,
                    tactile_calibrated=hand.tactile_calibrated,
                    connected=hand.is_connected,
                    read_failed=True,
                    accepted_target_action_id=accepted_target_action_id,
                    commboard_err=last_state.commboard_err,
                    jointboard_err=last_state.jointboard_err,
                    tipboard_err=last_state.tipboard_err,
                    source_monotonic_ns=last_source_ns,
                )
                rate_mgr.wait()
                continue

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
                tactile_sum=state.tactile_sum,
                tactile_sum_valid=state.tactile_sum_valid,
                tactile_contact=state.tactile_contact,
                tactile_force=state.tactile_force,
                tactile_valid=state.tactile_valid,
                tactile_calibrated=hand.tactile_calibrated,
                connected=hand.is_connected,
                read_failed=False,
                accepted_target_action_id=accepted_target_action_id,
                commboard_err=state.commboard_err,
                jointboard_err=state.jointboard_err,
                tipboard_err=state.tipboard_err,
                source_monotonic_ns=last_source_ns,
            )

            permit = read_motion_permit(shared)
            if permit.run_generation != stats_generation:
                # H4 uses an ARMED generation for the operator's physical-home
                # command, then a new RUNNING generation for the one policy
                # endpoint.  Preserve the completed RUNNING generation when
                # the coordinator revokes motion, while keeping the two
                # command classes distinct in the exit evidence.
                if stats_generation_was_running:
                    last_running_stats = (
                        sdk_send_attempts,
                        exact_target_accepts,
                        crc_unconfirmed,
                        duplicate_skips,
                        sdk_rejections,
                    )
                sdk_send_attempts = 0
                exact_target_accepts = 0
                crc_unconfirmed = 0
                duplicate_skips = 0
                sdk_rejections = 0
                last_exact_target_sequence = 0
                stats_generation = permit.run_generation
                stats_generation_was_running = permit.state is SafetyState.RUNNING
            if not permit.allows_motion:
                rate_mgr.wait()
                continue
            if shared.error_state.value:
                rate_mgr.wait()
                continue

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
            )
            command_generation = int(command["run_generation"][0])
            if command_generation != permit.run_generation or not (
                coupled_command_ticket_allows_execution(shared, ticket=ticket)
            ):
                rate_mgr.wait()
                continue
            if sequence_int == last_exact_target_sequence:
                # An accepted exact target is an endpoint event, not a
                # level-triggered command.  Retries remain allowed only until
                # the SDK has accepted the exact endpoint.
                duplicate_skips += 1
                rate_mgr.wait()
                continue
            issue = check_worker_hand_command(
                command,
                operational_lower_rad=operational_lower,
                operational_upper_rad=operational_upper,
                mechanical_lower_rad=mechanical_lower,
                mechanical_upper_rad=mechanical_upper,
                expected_run_generation=permit.run_generation,
                now_monotonic_ns=time.monotonic_ns(),
            )
            # A superseded/revoked snapshot has no authority to move hardware
            # or latch a fault, even if its contents fail validation.
            if not coupled_command_ticket_allows_execution(shared, ticket=ticket):
                rate_mgr.wait()
                continue
            if issue is not None:
                if issue.fault:
                    logger.error(
                        "hand_loop: unsafe action_id=%d: %s; latching runtime fault",
                        int(command["action_id"][0]),
                        issue.reason,
                    )
                    shared.error_state.value = True
                    return
                if sequence_int != last_rejected_ring_sequence:
                    if issue.reason == "expired command":
                        logger.info(
                            "hand_loop: discarded command: %s diagnostics=%s",
                            issue.reason,
                            expired_hand_command_diagnostics(
                                command,
                                state.qpos,
                                now_monotonic_ns=time.monotonic_ns(),
                                max_delta_rad_per_tick=config.hand_max_delta_rad_per_tick,
                            ),
                        )
                    else:
                        logger.info("hand_loop: discarded command: %s", issue.reason)
                    last_rejected_ring_sequence = sequence_int
                rate_mgr.wait()
                continue

            action_id = int(command["action_id"][0])
            target = np.asarray(command["hand_qpos"][0], dtype=np.float64)
            bounded = limit_hand_target_delta(
                target,
                state.qpos,
                config.hand_max_delta_rad_per_tick,
            )
            if not coupled_command_ticket_allows_execution(shared, ticket=ticket):
                rate_mgr.wait()
                continue
            if np.any(bounded < mechanical_lower - 1e-12) or np.any(
                bounded > mechanical_upper + 1e-12
            ):
                logger.error(
                    "hand_loop: measured-state-bounded action_id=%d violates mechanical limits",
                    action_id,
                )
                shared.error_state.value = True
                return
            sdk_send_attempts += 1
            send_status = hand.send_action(bounded)
            if send_status is XHandSendStatus.ACCEPTED:
                # ACK denotes SDK acceptance of the exact IPC endpoint, not
                # physical convergence or acceptance of an intermediate step.
                if np.array_equal(bounded, target):
                    accepted_target_action_id = action_id
                    exact_target_accepts += 1
                    last_exact_target_sequence = sequence_int
            elif send_status is XHandSendStatus.REJECTED:
                sdk_rejections += 1
                logger.error(
                    "hand_loop: SDK rejected action_id=%d; latching runtime fault",
                    action_id,
                )
                shared.error_state.value = True
                return
            # CRC_UNCONFIRMED deliberately leaves the action unacknowledged.
            # The next tick starts from fresh measured state and the latest
            # still-authorized command instead of latching a runtime fault.
            else:
                crc_unconfirmed += 1

            rate_mgr.wait()
    finally:
        if not _safe_disconnect(hand):
            logger.error("hand_loop: XHand disconnect failed")
            shared.error_state.value = True
        elif ready:
            logger.debug("hand_loop: STOPPED")
            exit_stats = last_running_stats or (
                sdk_send_attempts,
                exact_target_accepts,
                crc_unconfirmed,
                duplicate_skips,
                sdk_rejections,
            )
            logger.info(
                "hand_loop: exited (sdk_send_attempts=%d, "
                "exact_target_accepts=%d, crc_unconfirmed=%d, "
                "duplicate_skips=%d, sdk_rejections=%d)",
                *exit_stats,
            )
