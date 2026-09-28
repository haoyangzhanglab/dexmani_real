"""XHand SDK owner: direct absolute targets and independent tactile validity."""

import time
from queue import Empty

import numpy as np

from dexmani_real.ipc.schema import HAND_STATE_DTYPE
from dexmani_real.robot.command_validation import check_worker_hand_target
from dexmani_real.robot.commands import read_robot_command
from dexmani_real.robot.home import HomeResult
from dexmani_real.runtime.observation import sample_is_fresh
from dexmani_real.runtime.safety import SafetyState, command_may_cross_sdk
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


def _publish_feedback(shared, state, tactile_calibrated):
    if any(
        np.shape(x) != (12,) or not np.isfinite(x).all() for x in (state.qpos, state.current_ma)
    ):
        raise RuntimeError("unusable hand joint/current feedback")
    frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
    frame["qpos"], frame["current"] = state.qpos, state.current_ma
    for name in ("aggregate", "dense"):
        valid = tactile_calibrated and getattr(state, f"tactile_{name}_valid")
        frame[f"tactile_{name}_valid"] = valid
        frame[f"tactile_{name}"] = getattr(state, f"tactile_{name}") if valid else np.nan
    frame["timestamp_ns"] = time.monotonic_ns()
    shared.hand_state_ring.write(frame)
    return int(frame["timestamp_ns"][0])


def _send_target(shared, hand, target, run_id, config, *, home=False):
    from dexmani_real.robot.drivers.xhand import XHandSendStatus

    issue = check_worker_hand_target(
        target,
        mechanical_lower_rad=np.asarray(config.mechanical_qpos_min_rad),
        mechanical_upper_rad=np.asarray(config.mechanical_qpos_max_rad),
    )
    if not command_may_cross_sdk(
        shared,
        run_id=run_id,
        required_safety_state=SafetyState.ARMED if home else SafetyState.RUNNING,
    ):
        return None
    if issue:
        raise RuntimeError(f"unsafe hand target: {issue}")
    status = hand.send_action(target)
    if status is XHandSendStatus.REJECTED:
        raise RuntimeError(f"XHand SDK send failed: {status}")
    return status


def _best_effort_passive(hand):
    from dexmani_real.robot.drivers.xhand import XHandSendStatus

    try:
        status = hand.set_passive()
        if status is not XHandSendStatus.ACCEPTED:
            logger.warning("XHand cleanup passive send: %s", status.value)
    except Exception:
        logger.warning("XHand cleanup passive send failed", exc_info=True)


def run_hand_worker(shared, config):
    from dexmani_real.robot.drivers.xhand import XHand, XHandSendStatus

    hand = XHand(config)
    last_sequence = 0
    failure_started = None
    previous_errors = None
    active_motion_authority: tuple[int, SafetyState] | None = None
    last_valid_qpos = None
    last_valid_qpos_timestamp_ns = None
    try:
        hand.connect()
        try:
            hand.calibrate_tactile()
        except Exception:
            logger.warning("hand tactile calibration failed", exc_info=True)
        rate = LoopRate(config.loop_hz, label="hand", busy_wait=False)
        while shared.is_running.value:
            if shared.estop_request.value:
                break
            state = hand.get_state()
            if shared.estop_request.value:
                break
            if state is None:
                failure_started = failure_started or time.monotonic()
            else:
                failure_started = None
                if not hand.is_connected:
                    raise RuntimeError("XHand disconnected")
                errors = tuple(
                    tuple(getattr(state, name))
                    for name in ("commboard_err", "jointboard_err", "tipboard_err")
                )
                if errors != previous_errors and any(any(row) for row in errors):
                    logger.warning("XHand board errors (comm, joint, tip): %s", errors)
                previous_errors = errors
                last_valid_qpos_timestamp_ns = _publish_feedback(
                    shared, state, hand.tactile_calibrated
                )
                last_valid_qpos = state.qpos.copy()
                shared.hand_ready.set()

            revoked = active_motion_authority is not None and not command_may_cross_sdk(
                shared,
                run_id=active_motion_authority[0],
                required_safety_state=active_motion_authority[1],
            )
            if revoked:
                # Brief read dropouts may use cached feedback, never an old endpoint.
                if last_valid_qpos is not None and sample_is_fresh(
                    last_valid_qpos_timestamp_ns, config.feedback_max_age_s
                ):
                    status = hand.hold_current(last_valid_qpos)
                else:
                    status = hand.set_passive()
                if status is XHandSendStatus.REJECTED:
                    raise RuntimeError("XHand authority revoke failed")
                if status is XHandSendStatus.ACCEPTED:
                    active_motion_authority = None
                # CRC uncertainty retains authority for a retry on the next tick.
            if (
                failure_started is not None
                and time.monotonic() - failure_started >= config.state_read_failure_timeout_s
            ):
                raise RuntimeError("hand joint feedback timed out")
            if revoked or state is None:
                rate.wait()
                continue
            try:
                request = shared.hand_home_q.get_nowait()
            except Empty:
                request = None
            if request is not None:
                target, run_id, expires_ns = request
                status = (
                    _send_target(shared, hand, target, run_id, config, home=True)
                    if time.monotonic_ns() < expires_ns
                    else None
                )
                ok = status is not None
                if ok:
                    active_motion_authority = (run_id, SafetyState.ARMED)
                shared.hand_home_result_q.put(
                    (run_id, HomeResult(ok, "" if ok else "home revoked or expired"))
                )
            else:
                latest = read_robot_command(shared)
                if latest is not None:
                    command, sequence = latest
                    if sequence != last_sequence:
                        last_sequence = sequence
                        if command.hand_qpos is not None:
                            status = _send_target(
                                shared,
                                hand,
                                command.hand_qpos,
                                command.run_id,
                                config,
                            )
                            if status is not None:
                                active_motion_authority = (command.run_id, SafetyState.RUNNING)
            rate.wait()
    except Exception:
        shared.error_state.value = True
        logger.exception("hand worker failed")
        raise
    finally:
        _best_effort_passive(hand)
        try:
            hand.disconnect()
        except Exception:
            shared.error_state.value = True
            logger.exception("XHand disconnect failed")
            raise
