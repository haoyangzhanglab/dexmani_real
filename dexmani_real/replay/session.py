"""Own robot processes, replay scheduling, optional return-home and diagnostics."""

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
from dexmani_real.robot.arm_worker import run_arm_worker
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.hand_worker import run_hand_worker
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.safety import SafetyState, require_transition
from dexmani_real.runtime.supervisor import RuntimeSupervisor

DEFAULT_OUTPUT_DIR = "replay_results"


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
    processes = [
        ctx.Process(name="arm", target=run_arm_worker, args=(shared, runtime.arm)),
        ctx.Process(name="hand", target=run_hand_worker, args=(shared, runtime.hand)),
    ]
    supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
    keyboard = KeyboardInput(estop_callback=lambda: setattr(shared.estop_request, "value", True))
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="startup failed")
    try:
        supervisor.start(processes)
        require_transition(shared, SafetyState.ARMED)
        keyboard.start()
        home_result = home_hand(
            shared,
            runtime,
            abort_requested=lambda: bool(shared.estop_request.value) or not keyboard.healthy,
        )
        if home_result.ok:
            outcome = replay_targets(
                shared,
                runtime,
                trajectory,
                keyboard,
                hand_start_duration_s=config.hand_start_duration_s,
            )
        else:
            outcome = ReplayOutcome(
                ReplayStatus.REJECTED, reason=f"startup hand home failed: {home_result.reason}"
            )
        if outcome.successful:
            print("H: planned return_home; Q: exit", flush=True)
            while shared.is_running.value and supervisor.check():
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
                        abort_requested=lambda: (
                            bool(shared.estop_request.value) or not keyboard.healthy
                        ),
                    )
                    if not ok:
                        outcome = ReplayOutcome(
                            ReplayStatus.REJECTED, outcome.replay_data, "return_home failed"
                        )
                        break
                    print("Return-home completed. H: planned return_home; Q: exit", flush=True)
    finally:
        keyboard.quiesce()
        shutdown_clean = supervisor.shutdown(
            graceful_timeout_s=runtime.safety.shutdown_timeout_s,
        )
        keyboard.stop()
    if shared.error_state.value or shared.estop_request.value or not shutdown_clean:
        outcome = ReplayOutcome(
            ReplayStatus.FAULT, outcome.replay_data, "hardware/shutdown failure"
        )
    evaluate_replay(
        trajectory,
        outcome.replay_data,
        evaluate_consistency=config.evaluate_consistency,
        output_dir=config.output_dir,
        hand_start_duration_s=config.hand_start_duration_s,
    )
    return outcome
