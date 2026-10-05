"""Dedicated XHand home with measured convergence under ARMED authority."""

import time

import numpy as np

from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.home import HomeResult
from dexmani_real.robot.robot import DispatchError, DispatchInterrupted
from dexmani_real.runtime.observation import sample_is_fresh
from dexmani_real.runtime.safety import (
    SafetyState,
    command_may_cross_sdk,
    revoke_motion,
    revoke_motion_if_run_id,
)
from dexmani_real.utils.limits import validate_hand_command_bounds
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_HOME_CONSECUTIVE_SAMPLES = 3


def home_hand(shared, runtime, *, robot, abort_requested=None):
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
    if not command_may_cross_sdk(shared, run_id=epoch, required_safety_state=SafetyState.ARMED):
        return HomeResult(False, "hand home requires ARMED authority", interrupted=True)
    if abort_requested is not None and abort_requested():
        return HomeResult(False, "home interrupted", interrupted=True)
    deadline = time.monotonic_ns() + int(cfg.home_timeout_s * 1e9)
    try:
        robot.send_hand_home(RobotCommand(epoch, hand_qpos=target), valid_until_ns=deadline)
    except DispatchInterrupted as exc:
        logger.warning("hand HOME cancelled: dispatch=%s", exc.result)
        raise
    except DispatchError as exc:
        if not exc.revoked:
            raise
        logger.info("hand HOME interrupted: dispatch=%s", exc.result)
        revoke_motion_if_run_id(shared, epoch)
        robot.stop()
        return HomeResult(False, "home interrupted", interrupted=True)

    # Only feedback newer than the submission result can establish arrival.
    last_sample_ns = time.monotonic_ns()
    consecutive = 0
    tolerance_rad = np.deg2rad(cfg.home_tolerance_deg)
    interrupted = False
    while time.monotonic_ns() < deadline:
        if abort_requested is not None and abort_requested():
            interrupted = True
            break
        if not command_may_cross_sdk(shared, run_id=epoch, required_safety_state=SafetyState.ARMED):
            interrupted = True
            break
        local = robot.read_state()
        if local.hand is not None:
            sample = local.hand[0]
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
                    interrupted = time.monotonic_ns() < deadline
                    break
        time.sleep(0.01)
    revoke_motion_if_run_id(shared, epoch)
    robot.stop()
    return HomeResult(
        False,
        "home interrupted" if interrupted else "hand home convergence timed out",
        interrupted=interrupted,
    )
