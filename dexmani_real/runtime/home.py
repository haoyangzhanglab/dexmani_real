"""机械臂和手部回零，以新鲜实测反馈确认收敛。"""

import logging
import time
from contextlib import contextmanager

import numpy as np

from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)
HOME_CONSECUTIVE_SAMPLES = 3


class HomeConfigurationError(ValueError):
    """HOME 配置无效，本次操作不能执行。"""


@contextmanager
def _hand_home_motion(state, robot):
    failure = None
    try:
        yield
    except BaseException as exc:
        failure = exc
        if isinstance(exc, KeyboardInterrupt):
            state.request_emergency_stop()
        raise
    finally:
        try:
            robot.stop()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                state.request_emergency_stop()
            if failure is None or isinstance(failure, OperationCancelled):
                raise
            logger.exception("HOME cleanup also failed")


def home_arm(state, robot, *, cancel_epoch, cancel_requested):
    """直接回配置关节位；操作者负责确认空间，不改变手部目标。"""
    try:
        robot.arm_config.validate_home()
    except (ValueError, TypeError) as exc:
        raise HomeConfigurationError(f"arm HOME configuration: {exc}") from exc
    deadline_ns = time.monotonic_ns() + int(robot.arm_config.homing.timeout_s * 1e9)

    def cancelled():
        return cancel_requested() or not (
            state.cancel_epoch == cancel_epoch and state.requests_allow_motion()
        )

    robot.home_arm(cancel_epoch, cancelled, deadline_ns=deadline_ns)


def home_hand(state, robot, *, cancel_epoch, cancel_requested):
    hand = robot.hand_config
    if not hand.enabled:
        return
    try:
        hand.validate_home()
    except (ValueError, TypeError) as exc:
        raise HomeConfigurationError(f"hand HOME configuration: {exc}") from exc
    target = np.deg2rad(hand.home_qpos_deg)
    deadline_ns = time.monotonic_ns() + int(hand.home_timeout_s * 1e9)
    tolerance = np.deg2rad(hand.home_tolerance_deg)

    def cancelled():
        return cancel_requested() or not (state.cancel_epoch == cancel_epoch and state.requests_allow_motion())

    if cancelled():
        raise OperationCancelled("home interrupted")
    with _hand_home_motion(state, robot):
        robot.stop()
        # 停止态允许等待关节反馈恢复；采样机会不代表获得了有效新帧。
        while True:
            if cancelled():
                raise OperationCancelled("home interrupted")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("hand HOME initial feedback timed out")
            robot.service_feedback()
            if cancelled():
                raise OperationCancelled("home interrupted")
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                raise TimeoutError("hand HOME initial feedback timed out")
            feedback = robot.hand_feedback.snapshot()
            if feedback is not None and sample_is_fresh(
                int(feedback["timestamp_ns"][0]), hand.feedback_max_age_s, now_ns,
            ):
                break
            time.sleep(min(0.01, (deadline_ns - now_ns) / 1e9))
        retry_pending = not robot.send_hand_home(target, deadline_ns=deadline_ns, cancel_requested=cancelled)

        # 只有提交完成之后的新样本才能确认到达。
        last_sample_ns = time.monotonic_ns()
        consecutive = 0
        while True:
            if cancelled():
                raise OperationCancelled("home interrupted")
            if time.monotonic_ns() >= deadline_ns:
                break
            sampled = robot.service_feedback()
            if cancelled():
                raise OperationCancelled("home interrupted")
            if time.monotonic_ns() >= deadline_ns:
                break
            if retry_pending and sampled:
                robot.send_hand_home(target, deadline_ns=deadline_ns, cancel_requested=cancelled)
                retry_pending = False
                last_sample_ns = time.monotonic_ns()
            feedback = robot.hand_feedback.snapshot()
            if feedback is not None:
                sample = feedback[0]
                stamp = int(sample["timestamp_ns"])
                if stamp > last_sample_ns:
                    last_sample_ns = stamp
                    if (
                        sample_is_fresh(stamp, hand.feedback_max_age_s, time.monotonic_ns())
                        and np.max(np.abs(sample["qpos"] - target)) <= tolerance
                    ):
                        consecutive += 1
                    else:
                        consecutive = 0
                    if consecutive >= HOME_CONSECUTIVE_SAMPLES:
                        if cancelled():
                            raise OperationCancelled("home interrupted")
                        if time.monotonic_ns() < deadline_ns:
                            return
                        break
            time.sleep(min(0.01, max(0, deadline_ns - time.monotonic_ns()) / 1e9))
        raise TimeoutError("hand home convergence timed out")
