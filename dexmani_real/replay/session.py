"""Own local robot I/O, replay scheduling, return-home and replay evaluation."""

import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.recording.results import SessionResults, error_detail
from dexmani_real.replay.evaluation import evaluate_replay
from dexmani_real.replay.replayer import ReplayOutcome, ReplayStatus, replay_targets
from dexmani_real.replay.trajectory import verify_replay_preflight
from dexmani_real.robot.arm_homing import build_home_planner, home_robot
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.safety import RunEndReason, SafetyState, require_transition, revoke_motion
from dexmani_real.runtime.supervisor import RuntimeSupervisor
from dexmani_real.utils.log import get_logger

DEFAULT_OUTPUT_DIR = "replay_results"
logger = get_logger(__name__)


@dataclass(frozen=True)
class EpisodeReplayConfig:
    output_dir: str
    evaluate_consistency: bool
    hand_start_duration_s: float = 0.5

    def __post_init__(self):
        if not math.isfinite(self.hand_start_duration_s) or self.hand_start_duration_s < 0:
            raise ValueError("hand_start_duration_s must be finite and >= 0")


def replay_episode(trajectory, runtime, config):
    from dexmani_real.config.experiment import resolve_runtime_table, validate_robot_config

    validate_robot_config(runtime)
    runtime = resolve_runtime_table(runtime)
    target = Path(config.output_dir)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("replay output must be missing or empty")
    verify_replay_preflight(trajectory, runtime)
    results = SessionResults(target, "replay")
    shared = supervisor = robot = keyboard = None
    trajectory_outcome = None
    shutdown_clean = False
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="startup failed")
    try:
        ctx = mp.get_context("spawn")
        shared = RuntimeChannels.create(
            prefix=f"replay_{os.getpid()}",
            config=RuntimeChannelsConfig.from_runtime(runtime),
            mp_context=ctx,
        )
        supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
        robot = DexManiRobot(shared, runtime, check_services=supervisor.check)

        def request_quit():
            revoke_motion(shared, reason=RunEndReason.QUIT)
            shared.quit_requested.value = True

        keyboard = KeyboardInput(
            quit_callback=request_quit,
            estop_callback=lambda: setattr(shared.estop_request, "value", True),
        )
        from dexmani_real.planning.kinematics.arm_fk import make_arm_fk

        make_arm_fk()
        home_planner = build_home_planner(runtime)
        robot.connect()
        require_transition(shared, SafetyState.ARMED)
        keyboard.start()
        robot.check_services = lambda: supervisor.check() and keyboard.healthy
        home_result = home_hand(
            shared,
            runtime,
            robot=robot,
            abort_requested=lambda: (
                bool(shared.estop_request.value or shared.quit_requested.value)
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
                hand_start_duration_s=config.hand_start_duration_s,
                results=results,
            )
        else:
            outcome = ReplayOutcome(
                ReplayStatus.ESTOP
                if shared.estop_request.value
                else ReplayStatus.USER_QUIT
                if home_result.interrupted and shared.quit_requested.value
                else ReplayStatus.REJECTED,
                reason=f"startup hand home incomplete: {home_result.reason}",
            )
        trajectory_outcome = outcome
        if results.attempt is not None:
            results.finish_attempt(
                outcome.status.value,
                details=outcome.reason,
                row_count=len(outcome.replay_data["arm_qpos"])
                if outcome.replay_data is not None
                else 0,
            )
        if outcome.successful:
            print("H: planned return_home; Q: exit", flush=True)
            while (
                shared.is_running.value and not shared.quit_requested.value and supervisor.check()
            ):
                robot.service_idle()
                if not keyboard.healthy:
                    shared.estop_request.value = True
                if shared.error_state.value or shared.estop_request.value:
                    break
                signals = keyboard.poll(timeout=0.1)
                if OperatorCommand.QUIT in signals:
                    break
                if OperatorCommand.HOME in signals:
                    ok = home_robot(
                        shared,
                        runtime,
                        home_planner,
                        robot=robot,
                        abort_requested=lambda: (
                            bool(shared.estop_request.value or shared.quit_requested.value)
                            or not keyboard.healthy
                        ),
                    )
                    if not ok:
                        outcome = ReplayOutcome(
                            ReplayStatus.FAULT, outcome.replay_data, "return_home failed"
                        )
                        break
                    print("Return-home completed. H: planned return_home; Q: exit", flush=True)
    except KeyboardInterrupt:
        if shared is not None:
            shared.estop_request.value = True
        outcome = ReplayOutcome(ReplayStatus.ESTOP, outcome.replay_data, "KeyboardInterrupt")
    except Exception as exc:
        if shared is not None:
            shared.error_state.value = True
        logger.exception("replay session failed")
        outcome = ReplayOutcome(
            ReplayStatus.FAULT, outcome.replay_data, f"{type(exc).__name__}: {exc}"
        )
    finally:
        try:
            if shared is not None:
                revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
            if supervisor is not None:
                shutdown_clean = shutdown_local_runtime(
                    robot,
                    supervisor,
                    keyboard=keyboard,
                    timeout_s=runtime.safety.shutdown_timeout_s,
                )
            elif shared is not None:
                shutdown_clean = bool(shared.close())
        except Exception as exc:
            results.session["artifact_errors"].append(error_detail("shutdown", exc))
        if shared is not None:
            if shared.error_state.value or (not shutdown_clean and not shared.estop_request.value):
                outcome = ReplayOutcome(
                    ReplayStatus.FAULT,
                    outcome.replay_data,
                    f"{outcome.reason}; hardware/shutdown failure".lstrip("; "),
                )
            elif shared.estop_request.value:
                outcome = ReplayOutcome(
                    ReplayStatus.ESTOP,
                    outcome.replay_data,
                    outcome.reason or "operator emergency stop",
                )
        try:
            if results.attempt is not None and not results.failed:
                results.finish_attempt(outcome.status.value, details=outcome.reason)
            evaluate_replay(
                trajectory,
                outcome.replay_data,
                evaluate_consistency=config.evaluate_consistency,
                output_dir=config.output_dir,
                hand_start_duration_s=config.hand_start_duration_s,
            )
        except Exception as exc:
            results.session["artifact_errors"].append(error_detail("evaluation_save", exc))
            outcome = ReplayOutcome(
                ReplayStatus.FAULT, outcome.replay_data, f"{outcome.reason}; {exc}".lstrip("; ")
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
            outcome = ReplayOutcome(
                ReplayStatus.FAULT,
                outcome.replay_data,
                f"{outcome.reason}; final result: {exc}".lstrip("; "),
            )
    return outcome
