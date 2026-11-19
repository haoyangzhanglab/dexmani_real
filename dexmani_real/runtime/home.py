"""Arm-first HOME；模型上的离散工程验证不覆盖异常制动或未知障碍。"""

import logging
import time
from math import ceil

import numpy as np

from dexmani_real.control.collision import CollisionModel
from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    HomeTimeout,
    OperationCancelled,
    RobotError,
    arm_reference_from_feedback,
    validate_hand_target,
)
from dexmani_real.robot.model import XHAND_RATED_QPOS_MAX_RAD, XHAND_RATED_QPOS_MIN_RAD
from dexmani_real.robot.xarm7.protocol import (
    HOME_POSITION_TOLERANCE_RAD,
    STANDSTILL_DWELL_S,
)
from dexmani_real.robot.xarm7.trajectory import (
    PositionLimitError,
    calculate_home_trajectory,
    validate_extrema,
)
from dexmani_real.robot.xhand.driver import (
    OVERCURRENT_BOARD_CODE,
    OVERCURRENT_WARNING_CODE,
    XHandError,
)
from dexmani_real.utils.table_plane import load_table_plane
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)
ARM_HOME_TIMEOUT_S = 60.0
HAND_HOME_TIMEOUT_S = 10.0
HAND_HOME_STEP_RAD = np.deg2rad(1.0)
HAND_HOLD_TOLERANCE_RAD = 0.02
HOME_CONSECUTIVE_SAMPLES = 3
HOME_OVERCURRENT_CONSECUTIVE_CYCLES = 3


class HomeConfigurationError(ValueError):
    """HOME 配置尚未具备准入条件。"""


class HomePathRejected(ValueError):
    """轨迹检查未通过，未提交 HOME 运动。"""


def _plan_home(cfg, q, v, hand_q, collision, check):
    """规划回零轨迹，并按位移上限沿同一曲线采样检查碰撞。"""
    check()
    max_speed = cfg.home_speed_rad_per_s
    if np.any(np.abs(v) > max_speed):
        raise HomePathRejected("HOME initial velocity exceeds its motion limit")
    home = np.asarray(cfg.home_qpos)
    trajectory = calculate_home_trajectory(cfg, q, v)
    check()
    if trajectory.duration > ARM_HOME_TIMEOUT_S:
        raise HomeTimeout("HOME trajectory exceeds motion budget")
    try:
        validate_extrema(
            trajectory, q, np.asarray(cfg.joint_limit_lower), np.asarray(cfg.joint_limit_upper),
        )
    except PositionLimitError as exc:
        raise HomePathRejected(str(exc)) from exc

    def check_collision(point):
        check()
        reason = collision.rejection(point, hand_q)
        if reason:
            raise HomePathRejected(reason)

    count = round(trajectory.duration * cfg.servo_hz)
    period = 1 / cfg.servo_hz
    states = [trajectory.at_time(i / cfg.servo_hz)[:2] for i in range(count + 1)]
    samples = np.array([state[0] for state in states])
    samples[-1] = home

    # 任一时刻距最近伺服采样至多半个周期，用加速度估计采样间的速度上界。
    peak_speed = np.max(np.abs([state[1] for state in states]), axis=0)
    speed_bound = np.minimum(max_speed, peak_speed + cfg.max_joint_acceleration_rad_per_s2 * period / 2)
    relative_travel = 2 * np.dot(collision.arm_radii, speed_bound) * period
    substeps = max(1, ceil(relative_travel / (cfg.homing.collision_margin_m / 2)))
    for i, point in enumerate(samples[:-1]):
        check_collision(point)
        for j in range(1, substeps):
            check_collision(trajectory.at_time((i + j / substeps) * period)[0])
    check_collision(home)
    logger.info("HOME 轨迹通过模型检查，共 %d 个伺服参考点。", len(samples))
    return samples


def _hand_sample(robot):
    frame = robot.hand_feedback.snapshot()
    if frame is None or not sample_is_fresh(
        int(frame["timestamp_ns"][0]),
        robot.hand_config.feedback_max_age_s,
        time.monotonic_ns(),
    ):
        raise FeedbackUnavailable("HOME hand feedback unavailable/stale")
    return frame[0]


def _home_hand(robot, check):
    cfg = robot.hand_config
    check()
    _hand_sample(robot)
    deadline = time.monotonic_ns() + int(HAND_HOME_TIMEOUT_S * 1e9)
    target = cfg.home_qpos_rad
    step = HAND_HOME_STEP_RAD
    tolerance = np.deg2rad(cfg.home_tolerance_deg)
    last = 0
    consecutive = 0
    overcurrent_consecutive = 0
    submitted = False
    while True:
        check()
        if time.monotonic_ns() >= deadline:
            raise HomeTimeout("hand HOME convergence timed out")
        robot.service_feedback()
        check()
        if time.monotonic_ns() >= deadline:
            raise HomeTimeout("hand HOME feedback exceeded deadline")
        sample = _hand_sample(robot)
        stamp = int(sample["timestamp_ns"])
        if stamp > last and not robot.hand_feedback_pending:
            warning_stamp, read_warning, joints = robot.hand.feedback_overcurrent
            if warning_stamp != stamp:
                raise FeedbackUnavailable("HOME hand overcurrent status does not match feedback")
            write_warning = False
            # 第三次读取告警不再发送；其余周期合并读写结果后只计一次。
            abort_before_send = (
                read_warning
                and overcurrent_consecutive + 1 >= HOME_OVERCURRENT_CONSECUTIVE_CYCLES
            )
            q = sample["qpos"]
            consecutive = (
                consecutive + 1
                if submitted and not read_warning and np.max(np.abs(q - target)) <= tolerance
                else 0
            )
            if consecutive >= HOME_CONSECUTIVE_SAMPLES:
                return
            if not abort_before_send:
                delta = target - q
                command = np.where(np.abs(delta) <= step, target, q + np.copysign(step, delta))
                write_warning = robot.send_hand_home(command, deadline_ns=deadline, check_abort=check)
            warned = read_warning or write_warning
            overcurrent_consecutive = overcurrent_consecutive + 1 if warned else 0
            if warned:
                consecutive = 0
            if overcurrent_consecutive >= HOME_OVERCURRENT_CONSECUTIVE_CYCLES:
                source = "read" if read_warning else "send"
                error = XHandError(
                    "HOME", OVERCURRENT_WARNING_CODE,
                    f"persistent overcurrent: source={source}, joints={list(joints)}, "
                    f"board_code={OVERCURRENT_BOARD_CODE if joints else 'unknown'}, consecutive={overcurrent_consecutive}",
                )
                logger.error("%s", error)
                raise error
            submitted = not write_warning
            last = time.monotonic_ns()
        time.sleep(0.005)


def home_robot(state, robot, *, environment, cancel_epoch, cancel_requested):
    """cancel_requested 须服务故障并读取取消；所有失败均确认停止后传播。"""

    def cancelled():
        return (
            cancel_requested()
            or state.cancel_epoch != cancel_epoch
            or not state.requests_allow_motion()
        )

    def check():
        # cancel_requested 服务运行故障并读取操作指令；此处只补充本次操作的取消上下文。
        if cancelled():
            raise OperationCancelled("HOME cancelled")

    try:
        robot.stop()
        check()
        cfg, hand_cfg = robot.arm_config, robot.hand_config
        hand_enabled = hand_cfg.enabled
        try:
            cfg.validate_home()
            if hand_enabled:
                hand_cfg.validate_home()
            else:
                # 禁用 XHand 通信时，手部 HOME 姿态仅用于组合碰撞模型。
                hand_anchor = validate_hand_target(
                    hand_cfg.home_qpos_rad, XHAND_RATED_QPOS_MIN_RAD, XHAND_RATED_QPOS_MAX_RAD,
                )
            for box in environment.static_boxes:
                box.validate()
        except (ValueError, TypeError) as exc:
            raise HomeConfigurationError(str(exc)) from exc
        deadline = time.monotonic_ns() + int(ARM_HOME_TIMEOUT_S * 1e9)
        if robot.home_collision_model is None:
            path = environment.table_plane_path
            try:
                table_plane = load_table_plane(path) if path is not None else None
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise HomeConfigurationError(f"无法使用桌面标定 {path}（{exc}）") from exc
            robot.home_collision_model = CollisionModel(
                static_boxes=environment.static_boxes,
                table_plane=table_plane,
                margin_m=cfg.homing.collision_margin_m,
            )
        model = robot.home_collision_model

        def check_arm():
            check()
            if time.monotonic_ns() >= deadline:
                raise HomeTimeout("arm HOME budget exhausted")
            robot.service_feedback()
            check()
            if time.monotonic_ns() >= deadline:
                raise HomeTimeout("arm HOME feedback exceeded deadline")

        check_arm()
        if hand_enabled:
            # 入口的 robot.stop() 已确认 XHand 位置保持。
            hand_anchor = _hand_sample(robot)["qpos"].copy()
            stable = time.monotonic_ns()
            while True:
                check_arm()
                sample = _hand_sample(robot)
                if (
                    np.max(np.abs(sample["qpos"] - hand_anchor))
                    > HAND_HOLD_TOLERANCE_RAD
                ):
                    hand_anchor = sample["qpos"].copy()
                    stable = time.monotonic_ns()
                elif int(sample["timestamp_ns"]) - stable >= int(STANDSTILL_DWELL_S * 1e9):
                    break
                time.sleep(0.005)
        row = robot.arm.snapshot()[0]
        q = arm_reference_from_feedback(
            row["qpos"], cfg.joint_limit_lower, cfg.joint_limit_upper
        )
        if not sample_is_fresh(
            int(row["timestamp_ns"]), cfg.feedback_max_age_s, time.monotonic_ns()
        ):
            raise FeedbackUnavailable("HOME arm feedback stale")
        if np.max(np.abs(row["qvel"])) > cfg.standstill_velocity_tolerance_rad_s:
            raise OperationCancelled("arm moved after standstill")

        def check_path():
            check_arm()
            if not hand_enabled:
                return
            sample = _hand_sample(robot)
            if (
                np.max(np.abs(sample["qpos"] - hand_anchor))
                > HAND_HOLD_TOLERANCE_RAD
            ):
                raise OperationCancelled("XHand moved from HOME planning posture")
            if not state.stop_signal.is_set():
                robot.arm.publish_home_hand_stamp(int(sample["timestamp_ns"]))

        check_path()
        home = np.asarray(cfg.home_qpos)
        if np.max(np.abs(q - home)) <= HOME_POSITION_TOLERANCE_RAD:
            reason = model.rejection(q, hand_anchor)
            if reason:
                raise HomePathRejected(reason)
        else:
            samples = _plan_home(cfg, q, row["qvel"], hand_anchor, model, check_path)

            def execution_service():
                # 取消收尾时不再检查保持偏差，仍服务设备故障。
                if state.stop_signal.is_set():
                    robot.service_feedback()
                else:
                    check_path()

            robot.arm.home(
                cancel_epoch,
                samples=samples,
                hand_max_age_s=hand_cfg.feedback_max_age_s if hand_enabled else None,
                service=execution_service,
                abort=cancelled,
                deadline_ns=deadline,
            )
        check_arm()
        current = robot.arm.snapshot()[0]
        check()
        now_ns = time.monotonic_ns()
        if now_ns >= deadline:
            raise HomeTimeout("arm HOME feedback exceeded deadline")
        if (
            not sample_is_fresh(
                int(current["timestamp_ns"]),
                cfg.feedback_max_age_s,
                now_ns,
            )
            or np.max(np.abs(current["qpos"] - home)) > HOME_POSITION_TOLERANCE_RAD
            or np.max(np.abs(current["qvel"])) > cfg.standstill_velocity_tolerance_rad_s
        ):
            raise FeedbackUnavailable("arm HOME position/standstill not confirmed")
        if hand_enabled:
            _home_hand(robot, check)
    except KeyboardInterrupt:
        state.request_emergency_stop()
        raise
    finally:
        try:
            robot.stop()
        except OperationCancelled as exc:
            raise RobotError("HOME stop could not be confirmed") from exc
