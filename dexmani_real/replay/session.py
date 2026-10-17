"""Own local robot I/O, replay scheduling, return-home and replay evaluation."""

import logging
import math
from dataclasses import replace
from pathlib import Path

from dexmani_real.runtime.state import RuntimeState, RunEndReason
from dexmani_real.utils.episode_results import EpisodeResults, error_detail
from dexmani_real.replay.evaluation import evaluate_replay
from dexmani_real.replay.replayer import ReplayOutcome, ReplayStatus, replay_targets
from dexmani_real.replay.trajectory import verify_replay_preflight
from dexmani_real.robot.home import build_home_planner, home_robot, home_hand
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.utils.log import configure_logging


DEFAULT_OUTPUT_DIR = "replay_results"
logger = logging.getLogger(__name__)


def replay_episode(trajectory, runtime, *, output_dir, hand_start_duration_s=0.5):
    configure_logging()
    from dexmani_real.config.experiment import (
        resolve_runtime_table, validate_robot_config, validate_control_rate,
    )

    if not math.isfinite(hand_start_duration_s) or hand_start_duration_s < 0:
        raise ValueError("hand_start_duration_s must be finite and >= 0")
    validate_robot_config(runtime)
    validate_control_rate(runtime, trajectory.fps)
    runtime = resolve_runtime_table(runtime)
    target = Path(output_dir)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("replay output must be missing or empty")
    verify_replay_preflight(trajectory, runtime)
    results = EpisodeResults(target, "replay")
    shared = robot = keyboard = None
    trajectory_outcome = None
    shutdown_clean = False
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="startup failed")
    try:
        shared = RuntimeState()
        robot = DexManiRobot(shared, runtime)

        def request_quit():
            shared.revoke_motion(reason=RunEndReason.QUIT)
            shared.quit_requested = True

        keyboard = KeyboardInput(
            quit_callback=request_quit,
            estop_callback=shared.request_emergency_stop,
        )
        from dexmani_real.planning.kinematics.arm_fk import make_arm_fk

        make_arm_fk()
        home_planner = build_home_planner(runtime)
        robot.connect()
        shared.arm()
        keyboard.start()
        robot.check_services = lambda: keyboard.healthy
        home_result = home_hand(
            shared,
            runtime,
            robot=robot,
            abort_requested=lambda: (
                bool(shared.estop_request or shared.quit_requested)
                or not keyboard.healthy
            ),
        )
        if home_result.ok:
            outcome = replay_targets(
                shared,
                runtime,
                trajectory,
                keyboard,
                robot=robot,
                hand_start_duration_s=hand_start_duration_s,
                results=results,
            )
        else:
            outcome = ReplayOutcome(
                ReplayStatus.ESTOP
                if shared.estop_request
                else ReplayStatus.USER_QUIT
                if home_result.interrupted and shared.quit_requested
                else ReplayStatus.REJECTED,
                reason=f"startup hand home incomplete: {home_result.reason}",
            )
        trajectory_outcome = outcome
        if results.current is not None:
            results.finish_episode(
                outcome.status.value,
                details=outcome.reason,
                replay_rows=len(outcome.replay_data["arm_qpos"])
                if outcome.replay_data is not None
                else 0,
            )
        if outcome.successful:
            print("H: planned return_home; Q: exit", flush=True)
            while (
                shared.running and not shared.quit_requested
            ):
                robot.service_idle()
                if not keyboard.healthy:
                    shared.request_emergency_stop()
                if shared.error_state or shared.estop_request:
                    break
                signals = keyboard.poll(timeout=0.1)
                if OperatorCommand.QUIT in signals:
                    break
                if OperatorCommand.HOME in signals:
                    home_result = home_robot(
                        shared,
                        runtime,
                        home_planner,
                        robot=robot,
                        abort_requested=lambda: (
                            bool(shared.estop_request or shared.quit_requested)
                            or not keyboard.healthy
                        ),
                    )
                    results.session.setdefault("home_results", []).append(dict(
                        ok=home_result.ok, interrupted=home_result.interrupted,
                        reason=home_result.reason,
                    ))
                    if not home_result.ok:
                        outcome = replace(
                            outcome, status=ReplayStatus.USER_QUIT
                            if home_result.interrupted and shared.quit_requested
                            else ReplayStatus.ESTOP
                            if shared.estop_request
                            else ReplayStatus.FAULT,
                            reason=f"return_home incomplete: {home_result.reason}",
                        )
                        break
                    print("Return-home completed. H: planned return_home; Q: exit", flush=True)
    except KeyboardInterrupt:
        if shared is not None:
            shared.request_emergency_stop()
        outcome = replace(outcome, status=ReplayStatus.ESTOP, reason="KeyboardInterrupt")
    except Exception as exc:
        if shared is not None:
            shared.error_state = True
        logger.exception("replay session failed")
        outcome = replace(
            outcome, status=ReplayStatus.FAULT, reason=f"{type(exc).__name__}: {exc}"
        )
    finally:
        try:
            if shared is not None:
                shutdown_clean = shutdown_local_runtime(
                    shared, robot,
                    keyboard=keyboard,
                )
        except Exception as exc:
            results.session["errors"].append(error_detail("shutdown", exc))
        if shared is not None:
            if shared.error_state or (not shutdown_clean and not shared.estop_request):
                outcome = replace(
                    outcome, status=ReplayStatus.FAULT,
                    reason=f"{outcome.reason}; hardware/shutdown failure".lstrip("; "),
                )
            elif shared.estop_request:
                outcome = replace(
                    outcome, status=ReplayStatus.ESTOP,
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
            results.session["errors"].append(error_detail("evaluation_save", exc))
            outcome = replace(
                outcome, status=ReplayStatus.FAULT, reason=f"{outcome.reason}; {exc}".lstrip("; ")
            )
        try:
            results.finish_session(
                outcome=outcome.status.value,
                reason=outcome.reason,
                shutdown_clean=shutdown_clean,
                trajectory_status=trajectory_outcome.status.value if trajectory_outcome else None,
                trajectory_reason=trajectory_outcome.reason if trajectory_outcome else None,
            )
        except Exception as exc:
            logger.exception("replay final result publication failed")
            outcome = replace(
                outcome, status=ReplayStatus.FAULT,
                reason=f"{outcome.reason}; final result: {exc}".lstrip("; "),
            )
    return outcome
