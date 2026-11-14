"""Arm-first HOME；模型上的离散工程验证不覆盖异常制动或未知障碍。"""

import logging
import time
from itertools import pairwise

import numpy as np

from dexmani_real.control.collision import CollisionModel
from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    HomeTimeout,
    OperationCancelled,
    arm_reference_from_feedback,
)
from dexmani_real.robot.xarm7.protocol import STANDSTILL_DWELL_S
from dexmani_real.robot.xarm7.trajectory import (
    PositionLimitError,
    calculate_home_segment,
    validate_extrema,
)
from dexmani_real.robot.xhand.driver import (
    OVERCURRENT_BOARD_CODE,
    OVERCURRENT_WARNING_CODE,
    XHandError,
)
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)
HOME_CONSECUTIVE_SAMPLES = 3
HOME_OVERCURRENT_CONSECUTIVE_CYCLES = 3


class HomeConfigurationError(ValueError):
    """HOME 配置尚未具备准入条件。"""


class HomePathRejected(ValueError):
    """所有配置候选均不可行；没有提交 HOME 运动。"""


def _plan_home(cfg, q, v, hand_q, collision, check):
    """接收 HOME 入口已校验的配置和反馈，尝试 Direct / Via 候选。"""
    check()
    if np.any(np.abs(v) > cfg.homing.max_speed_rad_per_s):
        raise ValueError("HOME initial velocity exceeds its motion limit")
    home = np.asarray(cfg.home_qpos)
    margin = cfg.homing.collision_margin_m
    reasons = []
    for name, targets in [("Direct", [home]), ("Via", [cfg.homing.via_qpos, home])]:
        if name == "Via" and cfg.homing.via_qpos is None:
            break
        current, velocity = q, v
        chunks: list[np.ndarray] = []
        duration = 0.0
        reason = None
        for target in targets:
            check()
            target = np.asarray(target)
            trajectory = calculate_home_segment(cfg, current, velocity, target)
            check()
            duration += trajectory.duration
            if duration > cfg.homing.timeout_s:
                raise HomeTimeout("HOME trajectory exceeds configured motion budget")
            try:
                validate_extrema(
                    trajectory,
                    current,
                    np.asarray(cfg.joint_limit_lower),
                    np.asarray(cfg.joint_limit_upper),
                )
            except PositionLimitError as exc:
                reason = str(exc)
                break
            count = round(trajectory.duration * cfg.servo_hz)
            times = np.arange(count + 1) / cfg.servo_hz
            samples = np.array([trajectory.at_time(t)[0] for t in times])
            samples[-1] = target
            for point in samples:
                check()
                reason = collision.rejection(point, hand_q)
                if reason:
                    break
            if reason:
                break
            # 几何半径传播给出位移上界估计；沿 Ruckig 曲线细分，不线性插值。
            intervals = list(pairwise(times))
            while intervals:
                check()
                a, b = intervals.pop()
                va = np.abs(trajectory.at_time(a)[1])
                vb = np.abs(trajectory.at_time(b)[1])
                speed = np.minimum(
                    cfg.homing.max_speed_rad_per_s,
                    np.maximum(va, vb)
                    + cfg.homing.max_acceleration_rad_per_s2 * (b - a) / 2,
                )
                if 2 * np.dot(collision.arm_radii, speed) * (b - a) <= margin / 2:
                    continue
                mid = (a + b) / 2
                reason = collision.rejection(
                    np.asarray(trajectory.at_time(mid)[0]), hand_q
                )
                if reason:
                    break
                intervals.extend(((a, mid), (mid, b)))
            if reason:
                break
            chunks.append(samples if not chunks else samples[1:])
            current, velocity = target, np.zeros(7)
        if reason is None:
            references = np.concatenate(chunks)
            logger.info(
                "HOME %s: %d references, margin=%g m, interval displacement estimate <= %g m; "
                "离散模型检查不保证实际跟踪或制动无碰撞",
                name,
                len(references),
                margin,
                margin / 2,
            )
            return references
        reasons.append(f"{name}: {reason}")
    raise HomePathRejected("; ".join(reasons))


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
    deadline = time.monotonic_ns() + int(cfg.home_timeout_s * 1e9)
    target = np.deg2rad(cfg.home_qpos_deg)
    step = np.deg2rad(cfg.home_max_step_deg)
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
        if stamp > last:
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
                command = q + np.clip(target - q, -step, step)
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
            submitted = True
            last = time.monotonic_ns()
        time.sleep(0.005)


def home_robot(state, robot, *, environment, cancel_epoch, cancel_requested):
    """所有失败均完成停止收尾；停止失败保留原异常链并优先传播。"""

    def cancelled():
        return (
            cancel_requested()
            or state.cancel_epoch != cancel_epoch
            or not state.requests_allow_motion()
        )

    def check():
        if cancelled():
            raise OperationCancelled("HOME cancelled")
        robot.check()

    try:
        robot.stop()
        check()
        try:
            robot.arm_config.validate_home()
            robot.hand_config.validate_home()
            environment.validate()
            if not environment.home_model_confirmed or not robot.hand_config.enabled:
                raise ValueError(
                    "HOME requires confirmed environment and measured XHand posture"
                )
        except (ValueError, TypeError) as exc:
            raise HomeConfigurationError(str(exc)) from exc
        cfg = robot.arm_config
        deadline = time.monotonic_ns() + int(cfg.homing.timeout_s * 1e9)

        def check_arm():
            check()
            if time.monotonic_ns() >= deadline:
                raise HomeTimeout("arm HOME budget exhausted")
            robot.service_feedback()
            check()
            if time.monotonic_ns() >= deadline:
                raise HomeTimeout("arm HOME feedback exceeded deadline")

        check_arm()
        _hand_sample(robot)
        if not robot.stop_hand():
            raise FeedbackUnavailable("HOME requires confirmed XHand position hold")
        robot.arm.settle(
            cancel_epoch,
            service=robot.service_feedback,
            abort=cancelled,
            deadline_ns=deadline,
        )
        anchor = _hand_sample(robot)["qpos"].copy()
        stable = time.monotonic_ns()
        while True:
            check_arm()
            sample = _hand_sample(robot)
            if (
                np.max(np.abs(sample["qpos"] - anchor))
                > cfg.homing.hand_hold_tolerance_rad
            ):
                anchor = sample["qpos"].copy()
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

        def held():
            check_arm()
            sample = _hand_sample(robot)
            if (
                np.max(np.abs(sample["qpos"] - anchor))
                > cfg.homing.hand_hold_tolerance_rad
            ):
                raise OperationCancelled("XHand moved from HOME planning posture")
            robot.arm.publish_home_hand_stamp(int(sample["timestamp_ns"]))

        if robot.home_collision_model is None:
            robot.home_collision_model = CollisionModel(
                static_boxes=environment.static_boxes,
                margin_m=cfg.homing.collision_margin_m,
            )
        model = robot.home_collision_model
        held()
        if (
            np.max(np.abs(q - np.asarray(cfg.home_qpos)))
            <= cfg.standstill_position_tolerance_rad
        ):
            reason = model.rejection(q, anchor)
            if reason:
                raise HomePathRejected(reason)
        else:
            samples = _plan_home(cfg, q, row["qvel"], anchor, model, held)

            def execution_service():
                # 取消收尾时不再检查保持偏差，仍服务设备故障。
                if state.stop_signal.is_set():
                    robot.service_feedback()
                else:
                    held()

            robot.arm.home(
                cancel_epoch,
                samples=samples,
                hand_max_age_s=robot.hand_config.feedback_max_age_s,
                service=execution_service,
                abort=cancelled,
                deadline_ns=deadline,
            )
        check()
        robot.service_feedback()
        current = robot.arm.snapshot()[0]
        if (
            not sample_is_fresh(
                int(current["timestamp_ns"]),
                cfg.feedback_max_age_s,
                time.monotonic_ns(),
            )
            or np.max(np.abs(current["qpos"] - np.asarray(cfg.home_qpos)))
            > cfg.standstill_position_tolerance_rad
            or np.max(np.abs(current["qvel"])) > cfg.standstill_velocity_tolerance_rad_s
        ):
            raise FeedbackUnavailable(
                "arm HOME position/standstill lost before hand HOME"
            )
        _home_hand(robot, check)
    except KeyboardInterrupt:
        state.request_emergency_stop()
        raise
    finally:
        robot.stop()
