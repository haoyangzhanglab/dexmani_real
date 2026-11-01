"""机械臂和手部回零，以新鲜实测反馈确认收敛。"""

import logging
import time
from contextlib import contextmanager

import numpy as np

from dexmani_real.config.xarm7 import (
    EXECUTOR_TIMEOUT_S,
    HOME_CONVERGENCE_TIMEOUT_S,
)
from dexmani_real.control.homing import home_segment_timeout_s
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.runtime.state import SafetyState
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)
HOME_CONSECUTIVE_SAMPLES = 3


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
        if robot.stop_required:
            try:
                robot.stop()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    state.request_emergency_stop()
                if failure is None:
                    raise
                setattr(failure, "cleanup_errors", [
                    *getattr(failure, "cleanup_errors", ()), ("home_stop", exc),
                ])
                logger.exception("HOME cleanup also failed")


def home_robot(state, robot, planner, *, run_id, cancel_requested):
    # send_hand_home 自己完成整机停止边界，arm HOME 由 owner 停稳准备。
    home_hand(state, robot, run_id=run_id, cancel_requested=cancel_requested)
    home_arm(state, robot, planner, run_id=run_id, cancel_requested=cancel_requested)


def initialize_hand(state, robot, *, run_id, required_tactile, cancel_requested):
    """确认手部 HOME；只有实际消费者需要触觉时才 reset 并等待对应通道。"""
    if not robot.hand_config.enabled:
        return
    home_hand(state, robot, run_id=run_id, cancel_requested=cancel_requested)
    if not required_tactile:
        return
    logger.info("手部 HOME 已到位，正在请求触觉复位，请保持手部无接触。")

    def cancelled():
        return cancel_requested() or not state.can_move(
            run_id=run_id, required_safety_state=SafetyState.ARMED,
        )

    if cancelled():
        raise OperationCancelled("hand initialization interrupted")
    if not robot.reset_tactile_sensors(
        run_id=run_id, cancel_requested=cancelled, required_tactile=required_tactile,
    ):
        raise RobotError("tactile reset request failed")
    if cancelled():
        raise OperationCancelled("hand initialization interrupted")
    logger.info("触觉复位请求已完成，所需通道反馈已就绪。")


def home_arm(state, robot, planner, *, run_id, cancel_requested):
    config, hand = robot.arm_config, robot.hand_config
    target = np.asarray(config.home_qpos, dtype=np.float64)
    if state.safety_state != SafetyState.ARMED:
        raise OperationCancelled("home requires ARMED")
    boundary_ns = time.monotonic_ns()

    def cancelled():
        return cancel_requested() or not state.can_move(
            run_id=run_id, required_safety_state=SafetyState.ARMED,
        )

    with _home_motion(state, robot):
        deadline = time.monotonic() + HOME_CONVERGENCE_TIMEOUT_S
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
                int(arm_state["timestamp_ns"][0]) > boundary_ns
                and sample_is_fresh(arm_state["timestamp_ns"][0], config.feedback_max_age_s, now_ns)
                and np.max(np.abs(arm_state["qvel"][0])) <= config.homing.velocity_convergence_rad_s
            ):
                if not hand.enabled:
                    # 禁用手部时，实验约定手已移除或固定在配置的 HOME 姿态。
                    hand_qpos = np.deg2rad(hand.home_qpos_deg)
                    break
                hand_state = feedback.hand
                if (
                    hand_state is not None
                    and int(hand_state["timestamp_ns"][0]) > boundary_ns
                    and sample_is_fresh(hand_state["timestamp_ns"][0], hand.feedback_max_age_s, now_ns)
                ):
                    hand_qpos = hand_state["qpos"][0]
                    break
                feedback_issue = "fresh post-home hand state unavailable"
            time.sleep(0.01)
        else:
            raise TimeoutError(feedback_issue)

        planner.set_hand_qpos(hand_qpos)
        waypoints = planner.plan(arm_state["qpos"][0], target)
        if cancelled():
            raise OperationCancelled("home interrupted during planning")

        logger.debug("arm home: %d planned milestones", len(waypoints))
        # worker 先停稳，再查询起点/切换模式；每段允许计划起点的收敛误差。
        duration_s = HOME_CONVERGENCE_TIMEOUT_S + EXECUTOR_TIMEOUT_S + config.feedback_max_age_s
        duration_s += (
            sum(
                home_segment_timeout_s(float(distance) + config.homing.convergence_rad, config)
                for distance in np.max(np.abs(np.diff(waypoints, axis=0)), axis=1)
            )
            if len(waypoints) > 1 else HOME_CONVERGENCE_TIMEOUT_S
        )
        deadline_ns = time.monotonic_ns() + int(duration_s * 1e9)
        robot.home_arm(
            waypoints, target, run_id, cancelled,
            deadline_ns=deadline_ns,
        )


def home_hand(state, robot, *, run_id, cancel_requested):
    hand = robot.hand_config
    if not hand.enabled:
        return
    if state.safety_state != SafetyState.ARMED:
        raise OperationCancelled("hand home requires ARMED")
    target = np.deg2rad(hand.home_qpos_deg)
    deadline = time.monotonic_ns() + int(hand.home_timeout_s * 1e9)
    tolerance = np.deg2rad(hand.home_tolerance_deg)

    def cancelled():
        return cancel_requested() or not state.can_move(
            run_id=run_id, required_safety_state=SafetyState.ARMED,
        )

    if cancelled():
        raise OperationCancelled("home interrupted")
    with _home_motion(state, robot):
        robot.send_hand_home(RobotCommand(run_id, hand_qpos=target), expires_ns=deadline)

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
