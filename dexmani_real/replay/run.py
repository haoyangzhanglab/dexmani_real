"""一次顺序 demo 播放，慢计算降低频率，统一关闭退出。"""

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
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.state import SessionState
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


def run_replay(episode_path: str | Path, runtime: RealConfig) -> None:
    """起点读取一次姿态；播放和留驻通过公共设备服务处理 I/O 与停止。"""
    trajectory = load_trajectory(episode_path, runtime)
    state = SessionState()
    epoch = state.cancel_epoch
    robot = keyboard = None
    sent_count = 0
    started = time.monotonic()

    def check_input(robot: Robot, keyboard: Keyboard) -> None:
        robot.check()
        _, stopped = keyboard.poll(state, timeout=0)
        if state.estop_request or state.cancel_epoch != epoch or state.quit_requested:
            raise OperationCancelled("replay cancelled")
        if stopped:
            raise RobotError("replay device stopped without operator input")

    def wait_until(deadline_ns: int, robot: Robot, keyboard: Keyboard) -> None:
        while True:
            robot.service_feedback()
            check_input(robot, keyboard)
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return
            time.sleep(min(0.005, remaining_ns / 1e9))

    try:
        robot = Robot(runtime.arm, runtime.hand, runtime_state=state)
        keyboard = Keyboard(
            estop_callback=state.request_emergency_stop,
            bindings={"s": state.request_stop, "q": state.request_quit},
        )
        keyboard.start()
        robot.check_services = keyboard.check_health
        check_input(robot, keyboard)
        robot.connect()
        check_input(robot, keyboard)
        if not (state.cancel_epoch == epoch and state.requests_allow_motion()):
            raise OperationCancelled("startup cancelled")
        robot.prepare_streaming(epoch)
        robot.service_feedback()
        check_input(robot, keyboard)
        current = robot.read_state()
        check_input(robot, keyboard)
        now = time.monotonic_ns()
        for name, frame, age in (
            ("arm", current.arm, runtime.arm.feedback_max_age_s),
            ("hand", current.hand, runtime.hand.feedback_max_age_s),
        ):
            if frame is None or not sample_is_fresh(int(frame["timestamp_ns"][0]), age, now):
                raise RobotError(f"replay {name} feedback stale at start")
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
                row_deadline_ns += round(1e9 / trajectory.control_hz)
            robot.service_feedback()
            check_input(robot, keyboard)
            submit_now_ns = time.monotonic_ns()
            robot.send_action(
                RobotCommand(arm, hand),
                expires_ns=max(submit_now_ns, row_deadline_ns) + COMMAND_LEASE_NS,
            )
            sent_count += 1
            wait_until(
                time.monotonic_ns() + 500_000_000 if last_row else row_deadline_ns,
                robot,
                keyboard,
            )
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
        except BaseException:
            if primary is None:
                raise
            logger.exception("replay close also failed")
        finally:
            logger.info(
                "Replay submitted %d/%d rows in %.3f s",
                sent_count,
                len(trajectory),
                time.monotonic() - started,
            )
    if state.estop_request:
        raise OperationCancelled("replay emergency stop during shutdown")
