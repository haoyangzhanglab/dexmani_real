"""回放、人工回零与实验结果保存。"""

import logging
import math
from pathlib import Path

from dexmani_real.control.homing import HomePlanner
from dexmani_real.replay.joint_replay import JointReplay, ReplayResult, ReplayStatus
from dexmani_real.replay.trajectory import save_replay_data, validate_trajectory
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import home_robot
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
from dexmani_real.runtime.state import SessionState
from dexmani_real.utils.io import write_json
from dexmani_real.utils.log import configure_logging, error_detail
from dexmani_real.utils.table_plane import load_table_plane

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
    table_plane_abcd = load_table_plane(runtime.environment.table_plane_path)
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
            run_signal=runtime_state.run_signal,
            stop_signal=runtime_state.stop_signal,
            emergency_signal=runtime_state.emergency_signal,
        )

        keyboard = Keyboard(
            estop_callback=runtime_state.request_emergency_stop,
            bindings={'q': runtime_state.request_quit, 's': runtime_state.request_stop},
        )
        home_planner = HomePlanner(
            runtime.environment,
            runtime.arm,
            table_plane_abcd=table_plane_abcd,
        )
        keyboard.start()
        robot.check_services = lambda: keyboard.check_health()
        robot.connect()
        robot.check()
        runtime_state.arm()
        outcome = JointReplay(runtime_state, runtime, keyboard, robot=robot).run(
            trajectory, hand_start_duration_s=hand_start_duration_s,
        )
        trajectory_status = outcome.status.value
        if outcome.successful:
            wait_for_home(
                runtime_state, robot, keyboard, home_planner, outcome, summary["home_results"],
            )
    except OperationCancelled as exc:
        outcome.status = (
            ReplayStatus.ESTOP if runtime_state.estop_request else
            ReplayStatus.USER_QUIT if runtime_state.quit_requested else ReplayStatus.REJECTED
        )
        outcome.reason = str(exc)
    except KeyboardInterrupt:
        runtime_state.request_emergency_stop()
        outcome.status, outcome.reason = ReplayStatus.ESTOP, "KeyboardInterrupt"
    except Exception as exc:
        runtime_state.mark_error()
        logger.exception("replay session failed")
        outcome.status = ReplayStatus.FAULT
        if outcome.reason:
            outcome.cleanup_errors.append(exc)
        else:
            outcome.reason = f"{type(exc).__name__}: {exc}"
    finally:
        shutdown_clean = False
        try:
            close_runtime(runtime_state, robot, keyboard=keyboard)
            shutdown_clean = not runtime_state.error_state and not runtime_state.estop_request
        except BaseException as exc:
            outcome.cleanup_errors.append(exc)
            logger.exception("replay shutdown failed")
            if isinstance(exc, KeyboardInterrupt):
                runtime_state.request_emergency_stop()
        if runtime_state.error_state or (not shutdown_clean and not runtime_state.estop_request):
            outcome.status = ReplayStatus.FAULT
            if not outcome.reason and not outcome.cleanup_errors:
                outcome.reason = "hardware/shutdown failure"
        elif runtime_state.estop_request:
            outcome.status = ReplayStatus.ESTOP
            outcome.reason = outcome.reason or "operator emergency stop"

        try:
            save_replay_data(outcome.replay_data, target)
        except Exception as exc:
            logger.exception("replay data save failed")
            outcome.status = ReplayStatus.FAULT
            outcome.cleanup_errors.append(exc)
        finally:
            # 主要原因保持在前；实际清理错误只在现有 JSON 出口合并一次。
            outcome.reason = "; ".join(filter(None, (
                outcome.reason,
                *(f"{type(exc).__name__}: {error_detail('cleanup', exc)['message']}"
                  for exc in outcome.cleanup_errors),
            )))
            summary.update(
                status=outcome.status.value,
                reason=outcome.reason,
                shutdown_clean=shutdown_clean,
                trajectory_status=trajectory_status,
                replay_rows=len(outcome.replay_data["arm_qpos"]),
            )
            write_json(summary, result_path)
    return outcome


def wait_for_home(runtime_state, robot, keyboard, home_planner, outcome, home_results):
    keyboard.discard_events("h")
    print("H: planned return_home; Q: exit", flush=True)
    while runtime_state.running and not runtime_state.quit_requested:
        robot.service_idle()
        if not keyboard.check_health():
            runtime_state.request_emergency_stop()
        if runtime_state.error_state or runtime_state.estop_request:
            break
        signals = keyboard.poll(timeout=0.1)
        if runtime_state.quit_requested or "q" in signals:
            break
        if "s" in signals:
            continue
        if "h" in signals:
            try:
                home_robot(
                    runtime_state,
                    robot,
                    home_planner,
                    cancel_requested=lambda: (
                        bool(runtime_state.estop_request or runtime_state.quit_requested)
                        or not keyboard.check_health()
                    ),
                )
            except OperationCancelled as exc:
                home_results.append(dict(ok=False, interrupted=True, reason=str(exc)))
                if not (runtime_state.quit_requested or runtime_state.estop_request):
                    continue
                outcome.status = (
                    ReplayStatus.ESTOP if runtime_state.estop_request else ReplayStatus.USER_QUIT
                )
                outcome.reason = outcome.reason or str(exc)
                break
            except Exception as exc:
                home_results.append(dict(ok=False, interrupted=False, reason=str(exc)))
                outcome.status = ReplayStatus.FAULT
                outcome.reason = outcome.reason or f"return_home incomplete: {exc}"
                break
            finally:
                keyboard.discard_events("h")
            home_results.append(dict(ok=True, interrupted=False, reason=""))
            print("Return-home completed. H: planned return_home; Q: exit", flush=True)
