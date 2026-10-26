"""Own local robot I/O, replay scheduling, return-home and replay evaluation."""

import logging
import math
from dataclasses import replace
from pathlib import Path

from dexmani_real.calibration.table_plane import resolve_table_plane
from dexmani_real.control.homing import HomePlanner
from dexmani_real.data.session_results import SessionResults, error_detail
from dexmani_real.replay.consistency import evaluate_replay
from dexmani_real.replay.replayer import ReplayOutcome, ReplayStatus, TrajectoryReplayer
from dexmani_real.replay.trajectory import verify_replay_preflight
from dexmani_real.robot.system import DexManiRobot
from dexmani_real.runtime.home import home_hand, home_robot
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.observation import service_feedback
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
from dexmani_real.runtime.state import SessionState
from dexmani_real.utils.log import configure_logging

DEFAULT_OUTPUT_DIR = "replay_results"
logger = logging.getLogger(__name__)


def run_replay(
    trajectory, runtime, *, output_dir, hand_start_duration_s=0.5,
) -> ReplayOutcome:
    """运行一次回放，完成机器人收尾并保存评估结果。"""
    configure_logging()
    if not math.isfinite(hand_start_duration_s) or hand_start_duration_s < 0:
        raise ValueError("hand_start_duration_s must be finite and >= 0")
    validate_robot_config(runtime)
    validate_control_rate(runtime, trajectory.fps)
    table_plane_abcd = resolve_table_plane(
        runtime.environment.table, required=runtime.environment.table.enabled,
    )
    target = Path(output_dir)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("replay output must be missing or empty")
    verify_replay_preflight(trajectory, runtime)
    results = SessionResults(target, "replay")
    runtime_state = robot = keyboard = None
    trajectory_outcome = None
    shutdown_clean = False
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="startup failed")
    try:
        runtime_state = SessionState()
        robot = DexManiRobot(
            runtime.arm, runtime.hand, run_signal=runtime_state.run_signal,
            stop_signal=runtime_state.stop_signal,
            emergency_signal=runtime_state.emergency_signal,
            signal_lock=runtime_state.motion_lock,
        )

        keyboard = Keyboard(
            quit_callback=runtime_state.request_quit,
            stop_callback=runtime_state.request_stop,
            estop_callback=runtime_state.request_emergency_stop,
        )
        home_planner = HomePlanner(
            runtime.environment, table_plane_abcd=table_plane_abcd,
        )
        keyboard.start()
        robot.check_services = lambda: keyboard.healthy
        robot.connect()
        robot.check()
        runtime_state.arm()
        home_result = home_hand(
            runtime_state, robot,
            cancel_requested=lambda: (
                bool(runtime_state.estop_request or runtime_state.quit_requested)
                or not keyboard.healthy
            ),
        )
        if home_result.ok:
            outcome = TrajectoryReplayer(
                runtime_state, runtime, keyboard, robot=robot
            ).run(
                trajectory,
                hand_start_duration_s=hand_start_duration_s,
                results=results,
            )
        else:
            if runtime_state.estop_request:
                status = ReplayStatus.ESTOP
            elif home_result.interrupted and runtime_state.quit_requested:
                status = ReplayStatus.USER_QUIT
            else:
                status = ReplayStatus.REJECTED
            outcome = ReplayOutcome(
                status,
                reason=f"startup hand home incomplete: {home_result.reason}",
            )
        trajectory_outcome = outcome
        if results.current is not None:
            results.finish_episode(
                outcome.status.value,
                details=outcome.reason,
                replay_rows=(
                    len(outcome.replay_data["arm_qpos"])
                    if outcome.replay_data is not None
                    else 0
                ),
            )
        if outcome.successful:
            keyboard.discard_events("h")
            print("H: planned return_home; Q: exit", flush=True)
            while runtime_state.running and not runtime_state.quit_requested:
                stopped_epoch = runtime_state.run_id
                robot.service_idle()
                if not keyboard.healthy:
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
                        runtime_state, robot, home_planner,
                        service=lambda: service_feedback(robot, runtime),
                        cancel_requested=lambda: (
                            bool(
                                runtime_state.estop_request
                                or runtime_state.quit_requested
                            )
                            or not keyboard.healthy
                        ),
                    )
                    keyboard.discard_events("h")
                    results.record_home_result(
                        ok=home_result.ok,
                        interrupted=home_result.interrupted,
                        reason=home_result.reason,
                    )
                    if not home_result.ok:
                        if home_result.interrupted and runtime_state.stop_request and not (
                            runtime_state.quit_requested or runtime_state.estop_request
                        ):
                            continue
                        if home_result.interrupted and runtime_state.quit_requested:
                            status = ReplayStatus.USER_QUIT
                        elif runtime_state.estop_request:
                            status = ReplayStatus.ESTOP
                        else:
                            status = ReplayStatus.FAULT
                        outcome = replace(
                            outcome,
                            status=status,
                            reason=f"return_home incomplete: {home_result.reason}",
                        )
                        break
                    print("Return-home completed. H: planned return_home; Q: exit", flush=True)
    except KeyboardInterrupt:
        if runtime_state is not None:
            runtime_state.request_emergency_stop()
        outcome = replace(outcome, status=ReplayStatus.ESTOP, reason="KeyboardInterrupt")
    except Exception as exc:
        if runtime_state is not None:
            runtime_state.mark_error()
        logger.exception("replay session failed")
        outcome = replace(
            outcome, status=ReplayStatus.FAULT, reason=f"{type(exc).__name__}: {exc}"
        )
    finally:
        try:
            if runtime_state is not None:
                shutdown_clean = close_runtime(
                    runtime_state,
                    robot,
                    keyboard=keyboard,
                )
        except Exception as exc:
            results.record_errors([error_detail("shutdown", exc)])
        if runtime_state is not None:
            if runtime_state.error_state or (
                not shutdown_clean and not runtime_state.estop_request
            ):
                outcome = replace(
                    outcome,
                    status=ReplayStatus.FAULT,
                    reason=f"{outcome.reason}; hardware/shutdown failure".lstrip("; "),
                )
            elif runtime_state.estop_request:
                outcome = replace(
                    outcome,
                    status=ReplayStatus.ESTOP,
                    reason=outcome.reason or "operator emergency stop",
                )
        try:
            if results.current is not None:
                results.finish_episode(outcome.status.value, details=outcome.reason)
            evaluate_replay(
                trajectory,
                outcome.replay_data,
                output_dir=output_dir,
                hand_start_duration_s=hand_start_duration_s,
            )
        except Exception as exc:
            results.record_errors([error_detail("evaluation_save", exc)])
            outcome = replace(
                outcome,
                status=ReplayStatus.FAULT,
                reason=f"{outcome.reason}; {exc}".lstrip("; "),
            )
        try:
            results.finish_session(
                outcome=outcome.status.value,
                reason=outcome.reason,
                shutdown_clean=shutdown_clean,
                trajectory_status=(
                    trajectory_outcome.status.value if trajectory_outcome else None
                ),
                trajectory_reason=trajectory_outcome.reason if trajectory_outcome else None,
            )
        except Exception as exc:
            logger.exception("replay final result publication failed")
            outcome = replace(
                outcome,
                status=ReplayStatus.FAULT,
                reason=f"{outcome.reason}; final result: {exc}".lstrip("; "),
            )
    return outcome
