"""机械臂和手部回零，以新鲜实测反馈确认收敛。"""

import logging
import time
from contextlib import contextmanager

import numpy as np

from dexmani_real.config.control import HomeConfigurationError
from dexmani_real.control.homing import (
    HomePathRejected,
    HomePlanner,
)
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.utils.table_plane import load_table_plane
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)
HOME_CONSECUTIVE_SAMPLES = 3


def build_home_planner(environment, arm, *, table_plane_abcd=None):
    if table_plane_abcd is None:
        try:
            table_plane_abcd = load_table_plane(environment.table_plane_path)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise HomeConfigurationError(
                f"HOME table calibration {environment.table_plane_path!r}: {exc}"
            ) from exc
    # 模型构建的技术异常不属于可恢复的输入拒绝。
    return HomePlanner(environment, arm, table_plane_abcd=table_plane_abcd)


@contextmanager
def _home_motion(state, robot):
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
            if failure is None:
                raise
            if isinstance(failure, (OperationCancelled, HomeConfigurationError, HomePathRejected)):
                raise exc
            logger.exception("HOME cleanup also failed")


def home_robot(state, robot, get_planner, *, cancel_epoch, cancel_requested):
    """父进程取消上下文贯穿回零；cancel_requested 同时服务故障与操作者输入。"""
    try:
        robot.arm_config.validate_home()
    except (ValueError, TypeError) as exc:
        raise HomeConfigurationError(f"arm HOME configuration: {exc}") from exc
    home_hand(state, robot, cancel_epoch=cancel_epoch, cancel_requested=cancel_requested)
    if cancel_requested() or not (state.cancel_epoch == cancel_epoch and state.requests_allow_motion()):
        raise OperationCancelled("HOME interrupted before planner initialization")
    deadline_ns = time.monotonic_ns() + int(robot.arm_config.homing.timeout_s * 1e9)
    robot.stop()
    planner = get_planner()
    home_arm(state, robot, planner, cancel_epoch=cancel_epoch, cancel_requested=cancel_requested, deadline_ns=deadline_ns)


def initialize_hand(state, robot, *, cancel_epoch, required_tactile, cancel_requested):
    """确认手部 HOME；只有实际消费者需要触觉时才 reset 并等待对应通道。"""
    if not robot.hand_config.enabled:
        return
    home_hand(state, robot, cancel_epoch=cancel_epoch, cancel_requested=cancel_requested)
    if not required_tactile:
        return
    logger.info("手部 HOME 已到位，正在请求触觉复位，请保持手部无接触。")

    def cancelled():
        return cancel_requested() or not (state.cancel_epoch == cancel_epoch and state.requests_allow_motion())

    robot.reset_tactile_sensors(
        cancel_epoch=cancel_epoch, cancel_requested=cancelled, required_tactile=required_tactile,
    )
    logger.info("触觉复位请求已完成，所需通道反馈已就绪。")


def home_arm(state, robot, planner, *, cancel_epoch, cancel_requested, deadline_ns):
    config, hand = robot.arm_config, robot.hand_config
    target = np.asarray(config.home_qpos, dtype=np.float64)

    def cancelled():
        return cancel_requested() or not (state.cancel_epoch == cancel_epoch and state.requests_allow_motion())

    with _home_motion(state, robot):
        deadline = deadline_ns / 1e9
        feedback_issue = "fresh stationary arm state unavailable"
        while time.monotonic() < deadline:
            if cancelled():
                raise OperationCancelled("home interrupted")
            robot.service_feedback()
            feedback = robot.read_state()
            arm_state = feedback.arm
            now_ns = time.monotonic_ns()
            feedback_issue = "fresh stationary arm state unavailable"
            if (
                sample_is_fresh(arm_state["timestamp_ns"][0], config.feedback_max_age_s, now_ns)
                and np.max(np.abs(arm_state["qvel"][0])) <= config.homing.velocity_convergence_rad_s
            ):
                if not hand.enabled:
                    # 禁用手部时，实验约定手已移除或固定在配置的 HOME 姿态。
                    hand_qpos = np.deg2rad(hand.home_qpos_deg)
                    break
                hand_state = feedback.hand
                if (
                    hand_state is not None
                    and sample_is_fresh(hand_state["timestamp_ns"][0], hand.feedback_max_age_s, now_ns)
                ):
                    hand_qpos = hand_state["qpos"][0]
                    break
                feedback_issue = "fresh hand state unavailable"
            time.sleep(0.01)
        else:
            raise TimeoutError(feedback_issue)

        planner.set_hand_qpos(hand_qpos)
        waypoints = planner.plan(arm_state["qpos"][0], target)
        if cancelled():
            raise OperationCancelled("home interrupted during planning")

        logger.debug("arm home: %d planned milestones", len(waypoints))
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("arm HOME budget exhausted during planning")
        robot.home_arm(
            waypoints, cancel_epoch, cancelled,
            deadline_ns=deadline_ns,
        )


def home_hand(state, robot, *, cancel_epoch, cancel_requested):
    hand = robot.hand_config
    if not hand.enabled:
        return
    try:
        hand.validate_home()
    except (ValueError, TypeError) as exc:
        raise HomeConfigurationError(f"hand HOME configuration: {exc}") from exc
    target = np.deg2rad(hand.home_qpos_deg)
    deadline = time.monotonic_ns() + int(hand.home_timeout_s * 1e9)
    tolerance = np.deg2rad(hand.home_tolerance_deg)

    def cancelled():
        return cancel_requested() or not (state.cancel_epoch == cancel_epoch and state.requests_allow_motion())

    if cancelled():
        raise OperationCancelled("home interrupted")
    with _home_motion(state, robot):
        robot.send_hand_home(RobotCommand(hand_qpos=target), expires_ns=deadline, cancel_epoch=cancel_epoch)

        # 只有提交完成之后的新样本才能确认到达。
        last_sample_ns = time.monotonic_ns()
        consecutive = 0
        while time.monotonic_ns() < deadline:
            if cancelled():
                raise OperationCancelled("home interrupted")
            robot.service_feedback()
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
                        if time.monotonic_ns() < deadline:
                            return
                        break
            time.sleep(0.01)
        raise TimeoutError("hand home convergence timed out")
