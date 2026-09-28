"""Own local robot I/O, replay scheduling, return-home and replay evaluation."""

import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.replay.evaluation import evaluate_replay
from dexmani_real.replay.replayer import ReplayOutcome, ReplayStatus, replay_targets
from dexmani_real.replay.trajectory import verify_replay_preflight
from dexmani_real.robot.arm_homing import build_policy_home_planner, home_policy_robot
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.safety import SafetyState, require_transition, revoke_motion
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
    target = Path(config.output_dir)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("replay output must be missing or empty")
    verify_replay_preflight(trajectory, runtime)
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=f"replay_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
    robot = DexManiRobot(shared, runtime, check_services=supervisor.check)

    def request_quit():
        revoke_motion(shared)
        shared.quit_requested.value = True

    keyboard = KeyboardInput(
        quit_callback=request_quit,
        estop_callback=lambda: setattr(shared.estop_request, "value", True),
    )
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="startup failed")
    try:
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
            )
        else:
            outcome = ReplayOutcome(
                ReplayStatus.REJECTED, reason=f"startup hand home failed: {home_result.reason}"
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
                    ok = home_policy_robot(
                        shared,
                        runtime,
                        build_policy_home_planner(runtime),
                        robot=robot,
                        abort_requested=lambda: (
                            bool(shared.estop_request.value or shared.quit_requested.value)
                            or not keyboard.healthy
                        ),
                    )
                    if not ok:
                        outcome = ReplayOutcome(
                            ReplayStatus.REJECTED, outcome.replay_data, "return_home failed"
                        )
                        break
                    print("Return-home completed. H: planned return_home; Q: exit", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        shared.error_state.value = True
        logger.exception("replay session failed")
        outcome = ReplayOutcome(
            ReplayStatus.FAULT, outcome.replay_data, f"{type(exc).__name__}: {exc}"
        )
    finally:
        revoke_motion(shared)
        shutdown_clean = shutdown_local_runtime(
            robot, supervisor, keyboard=keyboard, timeout_s=runtime.safety.shutdown_timeout_s
        )
    if shared.error_state.value or shared.estop_request.value or not shutdown_clean:
        outcome = ReplayOutcome(
            ReplayStatus.FAULT,
            outcome.replay_data,
            f"{outcome.reason}; hardware/shutdown failure".lstrip("; "),
        )
    evaluate_replay(
        trajectory,
        outcome.replay_data,
        evaluate_consistency=config.evaluate_consistency,
        output_dir=config.output_dir,
        hand_start_duration_s=config.hand_start_duration_s,
    )
    return outcome
