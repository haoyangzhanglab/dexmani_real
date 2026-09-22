"""XHand SDK owner: direct absolute targets and independent tactile validity."""
from queue import Empty
import time
import numpy as np

from dexmani_real.ipc.schema import HAND_STATE_DTYPE
from dexmani_real.robot.commands import read_robot_command
from dexmani_real.robot.command_validation import check_worker_hand_target
from dexmani_real.robot.home import HomeResult
from dexmani_real.runtime.safety import SafetyState, command_may_cross_sdk
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


def _publish_feedback(shared, state, tactile_calibrated):
    if any(np.shape(x) != (12,) or not np.isfinite(x).all() for x in (state.qpos, state.current_ma)):
        raise RuntimeError("unusable hand joint/current feedback")
    frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
    frame["qpos"], frame["current"] = state.qpos, state.current_ma
    for name in ("aggregate", "dense"):
        valid = tactile_calibrated and getattr(state, f"tactile_{name}_valid")
        frame[f"tactile_{name}_valid"] = valid
        frame[f"tactile_{name}"] = getattr(state, f"tactile_{name}") if valid else np.nan
    frame["timestamp_ns"] = time.monotonic_ns()
    shared.hand_state_ring.write(frame)


def _send_target(shared, hand, target, run_id, config, accepted, *, home=False):
    issue = check_worker_hand_target(target,
        mechanical_lower_rad=np.asarray(config.mechanical_qpos_min_rad),
        mechanical_upper_rad=np.asarray(config.mechanical_qpos_max_rad))
    if not command_may_cross_sdk(shared, run_id=run_id,
            required_safety_state=SafetyState.ARMED if home else SafetyState.RUNNING):
        return False
    if issue:
        raise RuntimeError(f"unsafe hand target: {issue}")
    status = hand.send_action(target)
    if status is not accepted:
        raise RuntimeError(f"XHand SDK send failed: {status}")
    return True


def hand_loop(shared, config, state_read_failure_timeout_s):
    from dexmani_real.robot.drivers.xhand import XHand, XHandSendStatus
    hand = XHand(config)
    last_sequence = 0
    failure_started = None
    previous_errors = None
    try:
        hand.connect()
        try:
            hand.calibrate_tactile()
        except Exception:
            logger.warning("hand tactile calibration failed", exc_info=True)
        rate = LoopRate(config.loop_hz, label="hand")
        while shared.is_running.value:
            if shared.estop_request.value:
                break
            state = hand.get_state()
            if state is None:
                failure_started = failure_started or time.monotonic()
                if time.monotonic() - failure_started >= state_read_failure_timeout_s:
                    raise RuntimeError("hand joint feedback timed out")
                rate.wait()
                continue
            failure_started = None
            if not hand.is_connected:
                raise RuntimeError("XHand disconnected")
            errors = tuple(tuple(getattr(state, name)) for name in
                           ("commboard_err", "jointboard_err", "tipboard_err"))
            if errors != previous_errors and any(any(row) for row in errors):
                logger.warning("XHand board errors (comm, joint, tip): %s", errors)
            previous_errors = errors
            _publish_feedback(shared, state, hand.tactile_calibrated)
            shared.hand_ready.set()
            try:
                request = shared.hand_home_q.get_nowait()
            except Empty:
                request = None
            if request is not None:
                target, run_id, expires_ns = request
                ok = time.monotonic_ns() < expires_ns and _send_target(
                    shared, hand, target, run_id, config, XHandSendStatus.ACCEPTED, home=True)
                shared.hand_home_result_q.put((run_id, HomeResult(ok, "" if ok else "home revoked or expired")))
            else:
                latest = read_robot_command(shared)
                if latest is not None:
                    command, sequence = latest
                    if sequence != last_sequence:
                        last_sequence = sequence
                        if command.hand_qpos is not None:
                            _send_target(shared, hand, command.hand_qpos, command.run_id,
                                         config, XHandSendStatus.ACCEPTED)
            rate.wait()
    except Exception:
        shared.error_state.value = True
        logger.exception("hand worker failed")
        raise
    finally:
        try:
            hand.disconnect()
        except Exception:
            shared.error_state.value = True
            logger.exception("XHand disconnect failed")
            raise
