"""回放、人工回零与实验结果保存。"""

import logging
import math
from pathlib import Path

from dexmani_real.control.homing import HomePlanner
from dexmani_real.replay.joint_replay import JointReplay, ReplayResult, ReplayStatus
from dexmani_real.replay.metrics import compute_metrics
from dexmani_real.replay.trajectory import save_replay_data, validate_trajectory
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.system import Robot
from dexmani_real.runtime.home import home_robot
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
from dexmani_real.runtime.state import SessionState
from dexmani_real.utils.io import write_json
from dexmani_real.utils.log import configure_logging
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
    """运行一次回放，完成机器人收尾并保存评估结果。"""
    configure_logging()
    if not math.isfinite(hand_start_duration_s) or hand_start_duration_s < 0:
        raise ValueError("hand_start_duration_s must be finite and >= 0")
    validate_robot_config(runtime)
    validate_control_rate(runtime, trajectory.fps)
    table_plane_abcd = (
        load_table_plane(runtime.environment.table)
        if runtime.environment.table.enabled else None
    )
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
    outcome = ReplayResult(ReplayStatus.REJECTED, reason="startup failed")
    try:
        robot = Robot(
            runtime.arm,
            runtime.hand,
            run_signal=runtime_state.run_signal,
            stop_signal=runtime_state.stop_signal,
            emergency_signal=runtime_state.emergency_signal,
            signal_lock=runtime_state.motion_lock,
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
        outcome.reason = f"{outcome.reason}; {type(exc).__name__}: {exc}".lstrip("; ")
    finally:
        shutdown_clean = close_runtime(runtime_state, robot, keyboard=keyboard)
        if runtime_state.error_state or (not shutdown_clean and not runtime_state.estop_request):
            outcome.status = ReplayStatus.FAULT
            outcome.reason = f"{outcome.reason}; hardware/shutdown failure".lstrip("; ")
        elif runtime_state.estop_request:
            outcome.status = ReplayStatus.ESTOP
            outcome.reason = outcome.reason or "operator emergency stop"

        try:
            # 先保存采样，离线 FK 或指标计算失败后仍可重新评估。
            save_replay_data(outcome.replay_data, target)
            metrics = compute_metrics(trajectory, outcome.replay_data)
            write_json(metrics, target / "metrics.json")
            print(f"Replay: {metrics['matching_frames']}/{metrics['original_frames']} matching frames")
            for name in ("arm_joint", "hand_joint", "eef_position", "eef_orientation"):
                print(f"  {name}: {metrics[name]}")
        except Exception as exc:
            logger.exception("replay evaluation failed")
            outcome.status = ReplayStatus.FAULT
            outcome.reason = f"{outcome.reason}; evaluation: {exc}".lstrip("; ")
        finally:
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
        stopped_epoch = runtime_state.run_id
        robot.service_idle()
        if not keyboard.check_health():
            runtime_state.request_emergency_stop()
        if runtime_state.error_state or runtime_state.estop_request:
            break
        signals = keyboard.poll(timeout=0.1)
        if runtime_state.quit_requested or "q" in signals:
            break
        if runtime_state.stop_request or "s" in signals:
            if not robot.stop_required:
                runtime_state.acknowledge_stop(stopped_epoch)
            continue
        if "h" in signals:
            home_result = home_robot(
                runtime_state,
                robot,
                home_planner,
                cancel_requested=lambda: (
                    bool(runtime_state.estop_request or runtime_state.quit_requested)
                    or not keyboard.check_health()
                ),
            )
            keyboard.discard_events("h")
            home_results.append(
                dict(
                    ok=home_result.ok,
                    interrupted=home_result.interrupted,
                    reason=home_result.reason,
                )
            )
            if not home_result.ok:
                if (
                    home_result.interrupted
                    and runtime_state.stop_request
                    and not (runtime_state.quit_requested or runtime_state.estop_request)
                ):
                    continue
                if home_result.interrupted and runtime_state.quit_requested:
                    status = ReplayStatus.USER_QUIT
                elif runtime_state.estop_request:
                    status = ReplayStatus.ESTOP
                else:
                    status = ReplayStatus.FAULT
                outcome.status = status
                outcome.reason = f"return_home incomplete: {home_result.reason}"
                break
            print("Return-home completed. H: planned return_home; Q: exit", flush=True)
