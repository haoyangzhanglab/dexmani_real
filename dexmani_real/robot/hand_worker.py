"""Single-owner XHand servo worker with latest-target, fixed-grid feedback.

Each tick reads one state, publishes it (or a clearly stale previous state),
then sends at most one measured-state-bounded target. A rejected SDK command
latches the shared fault so fire-and-forget publishers cannot mistake silence
for successful application.
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
from dexmani_real.robot.command_validation import worker_validate_hand
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


def limit_hand_delta(
    target_qpos: np.ndarray,
    measured_qpos: np.ndarray,
    max_delta_rad_per_tick: float | np.ndarray,
) -> np.ndarray:
    """Bound one absolute target relative to fresh measured joint feedback."""
    target = np.asarray(target_qpos, dtype=np.float64)
    measured = np.asarray(measured_qpos, dtype=np.float64)
    if target.shape != HAND_JOINT_SHAPE or measured.shape != HAND_JOINT_SHAPE:
        raise ValueError("hand target and measured qpos must both have shape (12,)")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(measured)):
        raise ValueError("hand target and measured qpos must be finite")
    max_delta = np.broadcast_to(
        np.asarray(max_delta_rad_per_tick, dtype=np.float64), HAND_JOINT_SHAPE
    )
    if not np.all(np.isfinite(max_delta)) or np.any(max_delta <= 0.0):
        raise ValueError("hand max_delta_rad_per_tick must be finite and positive")
    return measured + np.clip(target - measured, -max_delta, max_delta)


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
    last_cmd_seq: int,
    last_cmd_qpos: np.ndarray,
    commboard_err: np.ndarray,
    jointboard_err: np.ndarray,
    tipboard_err: np.ndarray,
    source_monotonic_ns: int,
) -> None:
    """Serialize one feedback pair while preserving the existing SHM schema."""
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
    frame["last_cmd_seq"][0] = int(last_cmd_seq)
    frame["last_cmd_qpos"][0] = last_cmd_qpos
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


def _last_command_qpos(hand: Any, fallback_qpos: np.ndarray) -> np.ndarray:
    """Return the last SDK-accepted endpoint or a safe startup fallback."""
    last = getattr(hand, "last_qpos_cmd", None)
    if last is None:
        return np.asarray(fallback_qpos, dtype=np.float64).copy()
    value = np.asarray(last, dtype=np.float64)
    if value.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(value)):
        return np.asarray(fallback_qpos, dtype=np.float64).copy()
    return value.copy()


def hand_loop(shared: Any, config: HandParams) -> None:
    """Run one XHand worker; all SDK objects remain in this process."""
    from dexmani_real.robot.xhand import XHand
    from dexmani_real.runtime.safety import SafetyState

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
        last_applied_action_id = 0
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
            last_cmd_seq=last_applied_action_id,
            last_cmd_qpos=_last_command_qpos(hand, initial_state.qpos),
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
        last_consumed_ring_sequence = 0
        latest_command: np.ndarray | None = None
        latest_action_id = 0

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
                    last_cmd_seq=last_applied_action_id,
                    last_cmd_qpos=_last_command_qpos(hand, last_state.qpos),
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
                last_cmd_seq=last_applied_action_id,
                last_cmd_qpos=_last_command_qpos(hand, state.qpos),
                commboard_err=state.commboard_err,
                jointboard_err=state.jointboard_err,
                tipboard_err=state.tipboard_err,
                source_monotonic_ns=last_source_ns,
            )

            safety_state = shared.safety_state.value
            if safety_state not in (SafetyState.ARMED, SafetyState.RUNNING):
                latest_command = None
                latest_action_id = 0
                rate_mgr.wait()
                continue
            if shared.error_state.value:
                rate_mgr.wait()
                continue

            result = shared.hand_cmd_ring.read_latest()
            if result is not None:
                command, _published_ns, sequence = result
                sequence_int = (
                    int(sequence) if isinstance(sequence, (int, np.integer)) else 0
                )
                if sequence_int != last_consumed_ring_sequence:
                    last_consumed_ring_sequence = sequence_int
                    if worker_validate_hand(
                        command,
                        expected_run_generation=int(shared.run_generation.value),
                        now_monotonic_ns=time.monotonic_ns(),
                    ):
                        latest_command = command.copy()
                        latest_action_id = int(latest_command["action_id"][0])
                    else:
                        logger.info(
                            "hand_loop: discarded malformed, stale-generation, or expired command"
                        )
                        latest_command = None
                        latest_action_id = 0

            if latest_command is not None and not worker_validate_hand(
                latest_command,
                expected_run_generation=int(shared.run_generation.value),
                now_monotonic_ns=time.monotonic_ns(),
            ):
                latest_command = None
                latest_action_id = 0

            if latest_command is not None:
                target = np.asarray(latest_command["qpos_cmd"][0], dtype=np.float64)
                bounded = limit_hand_delta(
                    target,
                    state.qpos,
                    config.hand_max_delta_rad_per_tick,
                )
                if hand.send_action(bounded):
                    actual = _last_command_qpos(hand, state.qpos)
                    # ACK denotes acceptance of the original endpoint, not an
                    # intermediate max-delta step. Home/replay rely on this.
                    if np.array_equal(actual, target):
                        last_applied_action_id = latest_action_id
                else:
                    logger.error(
                        "hand_loop: SDK rejected action_id=%d; latching runtime fault",
                        latest_action_id,
                    )
                    shared.error_state.value = True
                    return

            rate_mgr.wait()
    finally:
        if not _safe_disconnect(hand):
            logger.error("hand_loop: XHand disconnect failed")
            shared.error_state.value = True
        elif ready:
            logger.debug("hand_loop: STOPPED")
        logger.info("hand_loop: exited")
