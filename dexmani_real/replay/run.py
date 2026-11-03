"""一次顺序 demo 播放，慢计算降低频率，统一关闭退出。"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.deployment import COMMAND_LEASE_NS
from dexmani_real.replay.trajectory import load_trajectory
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.state import SessionState
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


def run_replay(episode_path: str | Path, runtime: RealConfig) -> None:
    trajectory = load_trajectory(episode_path, runtime)
    state = SessionState()
    epoch = state.cancel_epoch
    robot = keyboard = None
    sent_count = 0
    started = time.monotonic()
    stopped_by_user = False
    expires_ns: int | None = None

    def check_input(robot: Robot, keyboard: Keyboard) -> None:
        nonlocal stopped_by_user
        robot.check()
        events, stopped = keyboard.poll(state, timeout=0)
        stopped_by_user |= any(event.key in {"s", "q"} for event in events)
        if state.estop_request or state.cancel_epoch != epoch or state.quit_requested:
            raise OperationCancelled("replay cancelled")
        if stopped:
            raise RobotError("replay device stopped without operator input")

    def verify_feedback(robot: Robot):
        snapshot = robot.read_state()
        now = time.monotonic_ns()
        for name, frame, age in (
            ("arm", snapshot.arm, runtime.arm.feedback_max_age_s),
            ("hand", snapshot.hand, runtime.hand.feedback_max_age_s),
        ):
            if frame is None or not sample_is_fresh(
                int(frame["timestamp_ns"][0]), age, now
            ):
                raise RobotError(f"replay {name} feedback stale")
        if expires_ns is not None and now >= expires_ns:
            raise RobotError("replay target lease expired")
        return snapshot

    def feedback(robot: Robot, keyboard: Keyboard):
        robot.service_feedback()
        snapshot = verify_feedback(robot)
        check_input(robot, keyboard)
        return snapshot

    def wait_until(deadline: float, robot: Robot, keyboard: Keyboard) -> None:
        while True:
            feedback(robot, keyboard)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.005, remaining))

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
        state.arm(cancel_epoch=epoch)
        robot.prepare_streaming(epoch)
        current = feedback(robot, keyboard)
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
        if state.begin_motion(cancel_epoch=epoch) is None:
            raise OperationCancelled("replay admission cancelled")
        for i, (arm, hand) in enumerate(
            zip(trajectory.action_arm_joint, trajectory.action_hand_joint)
        ):
            iteration_start = time.monotonic()
            feedback(robot, keyboard)
            expires_ns = time.monotonic_ns() + COMMAND_LEASE_NS
            robot.send_action(RobotCommand(arm, hand), expires_ns=expires_ns)
            sent_count += 1
            wait_until(
                time.monotonic() + 0.5
                if i == len(trajectory) - 1
                else iteration_start + 1 / trajectory.control_hz,
                robot,
                keyboard,
            )
    except OperationCancelled:
        if robot is not None:
            robot.check()
            if robot.connected:
                verify_feedback(robot)
        if keyboard is not None:
            stopped_by_user |= any(
                event.key in {"s", "q"} for event in keyboard.poll(state, timeout=0)[0]
            )
        if state.estop_request or not stopped_by_user:
            raise
        logger.info("Replay stopped by operator")
    except KeyboardInterrupt:
        state.request_emergency_stop()
        raise
    finally:
        primary = sys.exc_info()[1]
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
