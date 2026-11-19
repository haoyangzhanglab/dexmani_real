"""顺序回放 demo，结束后等待操作员回位或退出。"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.replay.trajectory import load_trajectory
from dexmani_real.robot.commands import (
    COMMAND_LEASE_NS,
    OperationCancelled,
    RobotCommand,
    RobotError,
)
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import home_robot
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.observation import check_device_feedback
from dexmani_real.runtime.state import SessionState
from dexmani_real.runtime.teleop_timing import TeleopTiming

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


def run_replay(
    episode_path: str | Path,
    runtime: RealConfig,
    *,
    segment: int | None = None,
    timing_output: str | Path | None = None,
) -> None:
    """起点读取一次姿态；播放和留驻通过公共设备服务处理 I/O 与停止。"""
    trajectory = load_trajectory(episode_path, runtime, segment=segment)
    period_ns = round(1e9 / trajectory.control_hz)
    timing = (
        TeleopTiming(
            timing_output, mode="replay", control_hz=trajectory.control_hz,
            arm=runtime.arm, motion=runtime.motion,
        ) if timing_output is not None else None
    )
    state = SessionState()
    epoch = state.cancel_epoch
    robot = keyboard = None
    sent_count = 0
    started = time.monotonic()

    def check_input(robot: Robot, keyboard: Keyboard, *, waiting_for_home=False) -> list[str]:
        robot.check()
        events, stopped = keyboard.poll(state, timeout=0.05 if waiting_for_home else 0)
        if state.estop_request or state.cancel_epoch != epoch or state.quit_requested:
            raise OperationCancelled("replay cancelled")
        if stopped and not waiting_for_home:
            raise RobotError("replay device stopped without operator input")
        return events

    def wait_until(deadline_ns: int, robot: Robot, keyboard: Keyboard) -> None:
        while True:
            robot.service_feedback()
            check_input(robot, keyboard)
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return
            time.sleep(min(0.005, remaining_ns / 1e9))

    try:
        robot = Robot(
            runtime.arm, runtime.hand, runtime_state=state,
            timing=timing, defer_hand_crc=False,
        )
        keyboard = Keyboard(
            estop_callback=state.request_emergency_stop,
            bindings={"s": state.request_stop, "q": state.request_quit},
        )
        keyboard.start()
        robot.check_services = keyboard.check_health
        check_input(robot, keyboard)
        robot.connect()
        check_input(robot, keyboard)
        robot.prepare_streaming(epoch)
        robot.service_feedback()
        check_input(robot, keyboard)
        current = robot.read_state()
        check_input(robot, keyboard)
        issues = check_device_feedback(current, runtime)
        if issues:
            raise RobotError("replay start feedback: " + "; ".join(issues))
        if any(
            np.max(np.abs(frame["qpos"][0] - target)) > np.deg2rad(10)
            for frame, target in (
                (current.arm, trajectory.action_arm_joint[0]),
                (current.hand, trajectory.action_hand_joint[0]),
            )
        ):
            raise ValueError(
                "start differs by more than 10 degrees; position the robot before replay"
            )
        if not (state.cancel_epoch == epoch and state.requests_allow_motion() and not state.stop_signal.is_set()):
            raise OperationCancelled("replay admission cancelled")
        for i, (arm, hand) in enumerate(
            zip(trajectory.action_arm_joint, trajectory.action_hand_joint)
        ):
            last_row = i == len(trajectory) - 1
            row_deadline_ns = time.monotonic_ns()
            if not last_row:
                row_deadline_ns += period_ns
            robot.service_feedback()
            check_input(robot, keyboard)
            submit_now_ns = time.monotonic_ns()
            arm_published_ns = None
            try:
                arm_published_ns = robot.send_action(
                    RobotCommand(arm, hand),
                    expires_ns=max(submit_now_ns, row_deadline_ns) + COMMAND_LEASE_NS,
                )
                sent_count += 1
                dwell_started_ns = time.monotonic_ns() if last_row else 0
            finally:
                if timing is not None:
                    timing.add(
                        "replay_row", timestamp_ns=submit_now_ns, finished_ns=time.monotonic_ns(),
                        raw_row=trajectory.raw_start_row + i, arm_published_ns=arm_published_ns,
                        accepted=arm_published_ns is not None,
                    )
            if not last_row:
                wait_until(row_deadline_ns, robot, keyboard)
                continue
            dwell_deadline_ns = dwell_started_ns + 500_000_000
            try:
                wait_until(dwell_deadline_ns, robot, keyboard)
            finally:
                if timing is not None:
                    # 从 Owner 已记录的反馈中离线读取终点，不增加 SDK 查询。
                    timing.add(
                        "replay_endpoint", timestamp_ns=dwell_started_ns, finished_ns=time.monotonic_ns(),
                        deadline_ns=dwell_deadline_ns, raw_row=trajectory.raw_start_row + i, qpos=arm,
                    )
        robot.stop()
        if timing is not None:
            # 先保存回放诊断，避免等待按键或 HOME 覆盖环缓冲中的端点反馈。
            playback_timing, timing = timing, None
            if robot.hand is not None:
                playback_timing.add("xhand_crc", counts=robot.hand.crc_counts.copy())
            playback_timing.write()
        keyboard.allow_start()
        logger.info("回放结束。H 回初始位后退出，S/Q 直接退出，Esc 急停。")
        while True:
            robot.service_idle()
            events = check_input(robot, keyboard, waiting_for_home=True)
            if "h" in events and keyboard.accept_operation(state) == epoch:
                break

        def home_cancel_requested() -> bool:
            robot.check()
            return False  # home_robot 合并急停、退出和本次操作的取消计数。

        logger.info("正在回初始位…")
        home_robot(
            state, robot, environment=runtime.environment, cancel_epoch=epoch,
            cancel_requested=home_cancel_requested,
        )
        logger.info("已回初始位。")
    except OperationCancelled:
        if robot is not None:
            robot.check()
        if state.estop_request or not (
            state.cancel_epoch != epoch or state.quit_requested
        ):
            raise
        logger.info("Replay stopped by operator")
    finally:
        primary = sys.exc_info()[1]
        if isinstance(primary, KeyboardInterrupt):
            state.request_emergency_stop()
        try:
            close_runtime(state, robot, keyboard=keyboard)
        except BaseException as exc:
            if primary is None:
                primary = exc
                raise
            logger.exception("replay close also failed")
        finally:
            logger.info(
                "Replay submitted %d/%d rows; session elapsed %.3f s",
                sent_count,
                len(trajectory),
                time.monotonic() - started,
            )
            if timing is not None:
                try:
                    timing.write()
                except Exception:
                    if primary is None and not state.estop_request:
                        raise
                    logger.exception("replay timing output also failed")
    if state.estop_request:
        raise OperationCancelled("replay emergency stop during shutdown")
