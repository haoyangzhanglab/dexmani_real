"""xArm SDK owner: Mode 6 absolute streaming and dedicated Mode 0 home."""
from queue import Empty
import time

import numpy as np

from dexmani_real.ipc.schema import ARM_STATE_DTYPE
from dexmani_real.robot.commands import read_robot_command
from dexmani_real.robot.command_validation import check_worker_arm_target
from dexmani_real.robot.drivers.xarm7 import HomeAborted, XArm7, describe_controller_error
from dexmani_real.robot.home import HomeResult
from dexmani_real.robot.model import XARM7_HARD_LOWER, XARM7_HARD_UPPER
from dexmani_real.runtime.safety import SafetyState, command_may_cross_sdk
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


def _publish_feedback(shared, qpos, qvel, effort):
    if any(np.shape(x) != (7,) or not np.isfinite(x).all() for x in (qpos, qvel, effort)):
        raise RuntimeError("unusable arm feedback")
    frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
    frame["qpos"], frame["qvel"], frame["effort"] = qpos, qvel, effort
    frame["timestamp_ns"] = time.monotonic_ns()
    shared.arm_state_ring.write(frame)


def _home(shared, arm, request):
    waypoints, final_qpos, run_id, expires_ns = request
    def aborted():
        return None if command_may_cross_sdk(shared, run_id=run_id,
            required_safety_state=SafetyState.ARMED) else "home authority revoked"
    if aborted() or time.monotonic_ns() >= expires_ns:
        shared.arm_home_result_q.put((run_id, HomeResult(False, "stale home request")))
        return False
    mode_ready = True
    try:
        arm.home(waypoints, final_qpos, abort_check=aborted,
                 feedback_callback=lambda q, v, e, target: _publish_feedback(shared, q, v, e))
    except HomeAborted as exc:
        arm.stop()
        mode_ready = bool(shared.is_running.value and not shared.estop_request.value and not shared.error_state.value)
        if mode_ready:
            arm.enter_mode6()
        result = HomeResult(False, str(exc))
    else:
        result = HomeResult(aborted() is None, aborted() or "")
    shared.arm_home_result_q.put((run_id, result))
    return mode_ready


def arm_loop(shared, config):
    arm = XArm7(config)
    last_sequence = 0
    streaming_epoch = None
    stopped = False
    try:
        arm.connect()
        _publish_feedback(shared, *arm.read())
        shared.arm_ready.set()
        rate = LoopRate(config.loop_hz, label="arm", busy_wait=False)
        while shared.is_running.value:
            if shared.estop_request.value:
                arm.emergency_stop()
                break
            if streaming_epoch is not None and not command_may_cross_sdk(shared, run_id=streaming_epoch):
                arm.stop()
                stopped = True
                streaming_epoch = None
            try:
                home = shared.arm_home_q.get_nowait()
            except Empty:
                home = None
            if home is not None:
                stopped = not _home(shared, arm, home)
                rate.reset()
            else:
                latest = read_robot_command(shared)
                if latest is not None:
                    command, sequence = latest
                    if sequence != last_sequence:
                        last_sequence = sequence
                        if command.arm_qpos is not None:
                            issue = check_worker_arm_target(command.arm_qpos,
                                joint_limit_lower_rad=np.asarray(XARM7_HARD_LOWER),
                                joint_limit_upper_rad=np.asarray(XARM7_HARD_UPPER))
                            if command_may_cross_sdk(shared, run_id=command.run_id):
                                if issue:
                                    raise RuntimeError(f"unsafe arm target: {issue}")
                                if stopped:
                                    arm.enter_mode6()
                                    stopped = False
                                # Mode changes can block; recheck the lifecycle at actual send.
                                if not command_may_cross_sdk(shared, run_id=command.run_id):
                                    continue
                                code = arm.servo(command.arm_qpos)
                                streaming_epoch = command.run_id
                                if code != 0:
                                    error = arm.read_live_error_code()
                                    raise RuntimeError(f"arm SDK send failed: {code}; {describe_controller_error(error)}")
            q, v, effort = arm.read()
            if arm.error_code:
                raise RuntimeError(f"arm controller error: {arm.error_code}")
            _publish_feedback(shared, q, v, effort)
            rate.wait()
    except Exception:
        shared.error_state.value = True
        logger.exception("arm worker failed")
        raise
    finally:
        arm.stop()
        arm.close()
