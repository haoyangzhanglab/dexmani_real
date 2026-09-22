"""Dedicated XHand home: success means SDK send success, not physical arrival."""
from queue import Full, Empty
import time
import numpy as np

from dexmani_real.robot.home import HomeResult, wait_home_result
from dexmani_real.runtime.safety import SafetyState, command_may_cross_sdk, revoke_motion
from dexmani_real.utils.limits import validate_hand_command_bounds


def home_hand(shared, runtime, *, abort_requested=None):
    if not runtime.policy.hand_enabled:
        return HomeResult(True)
    cfg = runtime.hand
    target = validate_hand_command_bounds(np.deg2rad(cfg.home_qpos_deg),
        np.asarray(cfg.qpos_min_rad), np.asarray(cfg.qpos_max_rad),
        np.asarray(cfg.mechanical_qpos_min_rad), np.asarray(cfg.mechanical_qpos_max_rad))
    if int(shared.safety_state.value) != int(SafetyState.ARMED):
        return HomeResult(False, "hand home requires ARMED")
    revoke_motion(shared)
    epoch = int(shared.run_id.value)
    while True:
        try:
            shared.hand_home_result_q.get_nowait()
        except Empty:
            break
    if not command_may_cross_sdk(shared, run_id=epoch, required_safety_state=SafetyState.ARMED):
        return HomeResult(False, "hand home requires ARMED authority")
    if abort_requested is not None and abort_requested():
        return HomeResult(False, "home aborted")
    deadline = time.monotonic_ns() + int(cfg.home_timeout_s * 1e9)
    try:
        shared.hand_home_q.put_nowait((target, epoch, deadline))
    except Full:
        return HomeResult(False, "hand home queue is full")
    return wait_home_result(shared, shared.hand_home_result_q, epoch, cfg.home_timeout_s, abort_requested)
