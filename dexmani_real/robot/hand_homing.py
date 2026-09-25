"""Dedicated XHand home with measured convergence under ARMED authority."""

import time
from queue import Empty, Full

import numpy as np

from dexmani_real.robot.home import HomeResult, wait_home_result
from dexmani_real.runtime.observation import sample_is_fresh
from dexmani_real.runtime.safety import (
    SafetyState,
    command_may_cross_sdk,
    revoke_motion,
    revoke_motion_if_run_id,
)
from dexmani_real.utils.limits import validate_hand_command_bounds

_HOME_CONSECUTIVE_SAMPLES = 3


def home_hand(shared, runtime, *, abort_requested=None):
    if not runtime.policy.hand_enabled:
        return HomeResult(True)
    cfg = runtime.hand
    target = validate_hand_command_bounds(
        np.deg2rad(cfg.home_qpos_deg),
        np.asarray(cfg.qpos_min_rad),
        np.asarray(cfg.qpos_max_rad),
        np.asarray(cfg.mechanical_qpos_min_rad),
        np.asarray(cfg.mechanical_qpos_max_rad),
    )
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
    result = wait_home_result(
        shared,
        shared.hand_home_result_q,
        epoch,
        max(0.0, (deadline - time.monotonic_ns()) / 1e9),
        abort_requested,
    )
    if not result.ok:
        revoke_motion_if_run_id(shared, epoch)
        return result

    # Only feedback newer than the submission result can establish arrival.
    last_sample_ns = time.monotonic_ns()
    consecutive = 0
    tolerance_rad = np.deg2rad(cfg.home_tolerance_deg)
    while time.monotonic_ns() < deadline:
        if abort_requested is not None and abort_requested():
            break
        if not command_may_cross_sdk(shared, run_id=epoch, required_safety_state=SafetyState.ARMED):
            break
        latest = shared.hand_state_ring.read_latest()
        if latest is not None:
            sample = latest[0][0]
            stamp = int(sample["timestamp_ns"])
            if stamp > last_sample_ns:
                last_sample_ns = stamp
                qpos = sample["qpos"]
                if (
                    sample_is_fresh(stamp, cfg.feedback_max_age_s)
                    and np.isfinite(qpos).all()
                    and np.max(np.abs(qpos - target)) <= tolerance_rad
                ):
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= _HOME_CONSECUTIVE_SAMPLES:
                    if (
                        time.monotonic_ns() < deadline
                        and not (abort_requested is not None and abort_requested())
                        and command_may_cross_sdk(
                            shared, run_id=epoch, required_safety_state=SafetyState.ARMED
                        )
                    ):
                        return HomeResult(True)
                    break
        time.sleep(0.01)
    revoke_motion_if_run_id(shared, epoch)
    return HomeResult(False, "hand home convergence aborted, interrupted or timed out")
