"""Own robot processes, replay scheduling, optional return-home and diagnostics."""

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.replay.evaluation import evaluate_replay
from dexmani_real.replay.replayer import ReplayOutcome, ReplayStatus, replay_targets
from dexmani_real.replay.trajectory import verify_replay_preflight
from dexmani_real.robot.arm_homing import build_policy_home_planner, home_policy_robot
from dexmani_real.robot.arm_worker import arm_loop
from dexmani_real.robot.hand_worker import hand_loop
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.processes import shutdown_processes_verified
from dexmani_real.runtime.safety import SafetyState, require_transition
from dexmani_real.runtime.supervisor import check_processes, start_processes

DEFAULT_OUTPUT_DIR = "replay_results"


@dataclass(frozen=True)
class EpisodeReplayConfig:
    output_dir: str
    evaluate_consistency: bool


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
        ctx.Process(name="arm", target=arm_loop, args=(shared, runtime.arm)),
        ctx.Process(name="hand", target=hand_loop, args=(shared, runtime.hand)),
    ]
    started = []
    keyboard = KeyboardInput(estop_callback=lambda: setattr(shared.estop_request, "value", True))
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="startup failed")
    try:
        start_processes(shared, processes, runtime.safety.readiness_timeouts_s, started)
        require_transition(shared, SafetyState.ARMED)
        keyboard.start()
        outcome = replay_targets(shared, runtime, trajectory, keyboard)
        if outcome.successful:
            print("H: planned return_home; Q: exit", flush=True)
            deadline = time.monotonic() + runtime.policy.post_teleop_timeout_s
            while time.monotonic() < deadline and check_processes(shared, started):
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
    finally:
        keyboard.quiesce()
        report = shutdown_processes_verified(
            shared,
            started,
            graceful_timeout_s=runtime.safety.shutdown_timeout_s,
            disarm_if_clean=outcome.successful,
        )
        keyboard.stop()
    if (
        shared.error_state.value
        or shared.estop_request.value
        or not report.shared_closed
        or any(x.exitcode != 0 or x.escalation != "graceful" for x in report.exits)
    ):
        outcome = ReplayOutcome(
            ReplayStatus.FAULT, outcome.replay_data, "hardware/shutdown failure"
        )
    evaluate_replay(
        trajectory,
        outcome.replay_data,
        evaluate_consistency=config.evaluate_consistency,
        output_dir=config.output_dir,
    )
    return outcome
