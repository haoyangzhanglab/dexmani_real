"""回放、人工回零与实验结果保存。"""

import logging
import math
from pathlib import Path

from dexmani_real.config.control import HomeConfigurationError
from dexmani_real.control.homing import HomePathRejected
from dexmani_real.replay.joint_replay import JointReplay, ReplayResult, ReplayStatus
from dexmani_real.replay.trajectory import save_replay_data, validate_trajectory
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import build_home_planner, home_robot
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
from dexmani_real.runtime.state import SessionState
from dexmani_real.utils.io import write_json
from dexmani_real.utils.log import configure_logging, error_detail

DEFAULT_OUTPUT_DIR = "replay_results"
logger = logging.getLogger(__name__)


def run_replay(
    trajectory,
    runtime,
    *,
    output_dir,
    hand_start_duration_s=0.5,
) -> ReplayResult:
    """运行一次回放，完成机器人收尾并保存控制结果和实测采样。"""
    configure_logging()
    if not math.isfinite(hand_start_duration_s) or hand_start_duration_s < 0:
        raise ValueError("hand_start_duration_s must be finite and >= 0")
    validate_robot_config(runtime)
    validate_control_rate(runtime, trajectory.fps)
    target = Path(output_dir)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("replay output must be missing or empty")
    validate_trajectory(trajectory, runtime)
    summary = {
        "episode_path": trajectory.episode_path,
        "status": "incomplete",
        "reason": "",
        "hand_start_duration_s": hand_start_duration_s,
        "home_results": [],
    }
    result_path = target / "session_result.json"
    write_json(summary, result_path, overwrite=False)
    runtime_state = SessionState()
    robot = keyboard = None
    trajectory_status = None
    outcome = ReplayResult(ReplayStatus.REJECTED)
    try:
        robot = Robot(
            runtime.arm,
            runtime.hand,
            runtime_state=runtime_state,
        )

        keyboard = Keyboard(
            estop_callback=runtime_state.request_emergency_stop,
            bindings={'q': runtime_state.request_quit, 's': runtime_state.request_stop},
        )
        keyboard.start()
        robot.check_services = lambda: keyboard.check_health()
        robot.connect()
        robot.check()
        cancel_epoch = runtime_state.arm()
        replay = JointReplay(runtime_state, runtime, keyboard, robot=robot)
        try:
            replay.run(trajectory, cancel_epoch=cancel_epoch, hand_start_duration_s=hand_start_duration_s)
        finally:
            outcome = replay.result
        trajectory_status = outcome.status.value
        if outcome.successful:
            wait_for_home(
                runtime_state, runtime, robot, keyboard, outcome, summary["home_results"],
            )
        robot.check()
    except OperationCancelled as exc:
        try:
            if robot is not None:
                robot.check()
        except Exception as failure:
            runtime_state.stop_signal.set()
            outcome.status = ReplayStatus.FAULT
            if not any(failure is error for error in outcome.errors):
                outcome.errors.append(failure)
                logger.exception("replay cancellation also failed")
        else:
            outcome.status = (
                ReplayStatus.ESTOP if runtime_state.estop_request else
                ReplayStatus.USER_QUIT if runtime_state.quit_requested else ReplayStatus.REJECTED
            )
            outcome.reason = str(exc)
    except KeyboardInterrupt:
        runtime_state.request_emergency_stop()
        outcome.status, outcome.reason = ReplayStatus.ESTOP, "KeyboardInterrupt"
    except Exception as exc:
        runtime_state.stop_signal.set()
        outcome.status = ReplayStatus.FAULT
        if not any(exc is error for error in outcome.errors):
            outcome.errors.append(exc)
            logger.exception("replay session failed")
    finally:
        shutdown_clean = False
        shutdown_failure = False
        try:
            close_runtime(runtime_state, robot, keyboard=keyboard)
            shutdown_clean = not runtime_state.estop_request
        except BaseException as exc:
            shutdown_failure = True
            if not any(exc is error for error in outcome.errors):
                outcome.errors.append(exc)
                logger.exception("replay shutdown failed")
            if isinstance(exc, KeyboardInterrupt):
                runtime_state.request_emergency_stop()
        if outcome.status == ReplayStatus.FAULT or shutdown_failure:
            outcome.status = ReplayStatus.FAULT
            if not outcome.reason and not outcome.errors:
                outcome.reason = "hardware/shutdown failure"
        elif runtime_state.estop_request:
            outcome.status = ReplayStatus.ESTOP
            outcome.reason = outcome.reason or "operator emergency stop"

        try:
            save_replay_data(outcome.replay_data, target)
        except Exception as exc:
            logger.exception("replay data save failed")
            outcome.status = ReplayStatus.FAULT
            outcome.errors.append(exc)
        finally:
            # 保留非异常原因，各真实错误只写一份简短结果。
            summary.update(
                status=outcome.status.value,
                reason=outcome.reason,
                errors=[error_detail("replay", error) for error in outcome.errors],
                shutdown_clean=shutdown_clean,
                trajectory_status=trajectory_status,
                replay_rows=len(outcome.replay_data["arm_qpos"]),
            )
            write_json(summary, result_path)
    return outcome


def wait_for_home(runtime_state, runtime, robot, keyboard, outcome, home_results):
    home_planner = None

    def get_planner():
        nonlocal home_planner
        if home_planner is None:
            home_planner = build_home_planner(runtime.environment, runtime.arm)
        return home_planner

    def home_cancelled():
        robot.check()
        return runtime_state.estop_request or runtime_state.quit_requested

    print("H: planned return_home; Q: exit", flush=True)
    while not runtime_state.quit_requested:
        robot.service_idle()
        if runtime_state.estop_request:
            break
        keyboard.allow_start()
        signals = keyboard.poll(timeout=0.1)
        keys = {event.key for event in signals}
        if runtime_state.quit_requested or "q" in keys:
            break
        if "s" in keys:
            continue
        home_event = next((event for event in signals if event.key == "h"), None)
        epoch = keyboard.accept_operation(runtime_state) if home_event is not None else None
        if epoch is not None:
            try:
                home_robot(
                    runtime_state,
                    robot,
                    get_planner,
                    cancel_epoch=epoch,
                    cancel_requested=home_cancelled,
                )
            except OperationCancelled as exc:
                robot.check()
                home_results.append(dict(ok=False, interrupted=True, reason=str(exc)))
                if not (runtime_state.quit_requested or runtime_state.estop_request):
                    continue
                outcome.status = (
                    ReplayStatus.ESTOP if runtime_state.estop_request else ReplayStatus.USER_QUIT
                )
                outcome.reason = outcome.reason or str(exc)
                break
            except (HomeConfigurationError, HomePathRejected) as exc:
                robot.check()
                home_results.append(dict(ok=False, interrupted=False, reason=str(exc)))
                logger.warning("本次 HOME 未完成（手部可能已回位）：%s", exc)
                continue
            except Exception as exc:
                home_results.append(dict(
                    ok=False, interrupted=False, reason=str(exc),
                ))
                outcome.status = ReplayStatus.FAULT
                outcome.reason = outcome.reason or "return_home incomplete"
                outcome.errors.append(exc)
                logger.exception("replay HOME failed")
                raise
            home_results.append(dict(ok=True, interrupted=False, reason=""))
            print("Return-home completed. H: planned return_home; Q: exit", flush=True)
