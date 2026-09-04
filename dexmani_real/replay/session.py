"""Physically replay one recorded DexMani episode on the real robot.

Spawn-only architecture: arm_loop + hand_loop processes with RuntimeChannels
and the SafetyState machine. Commands flow through coupled_cmd_ring; state is
read from arm_state_ring / hand_state_ring. No direct SDK access from
the main process.

Replay reruns dense geometry and provenance preflight, spawns arm/hand workers,
replays the exact submitted ``sent`` joint-command stream, captures measured
robot state, and evaluates joint, EEF, and tracking-lag consistency metrics.

This module owns replay lifecycle, worker topology, and terminal outcome.
``replay_controller`` owns safety-gated frame scheduling, while
``replay_capture`` owns the bounded in-memory measurement buffer.

Replay always replays the recorded hand command stream. If the episode was
recorded with non-default ``--acc``/``--speed``, pass the same values here so the
resolved-config provenance matches.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.config.runtime import ResolvedRuntimeConfig
from dexmani_real.control.arm_home import ArmHomeConfig, execute_arm_home
from dexmani_real.control.hand_home import publish_hand_home_and_wait_accepted
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.replay.controller import EpisodeReplayer, ReplayOutcome, ReplayStatus
from dexmani_real.replay.evaluation import evaluate_replay
from dexmani_real.replay.trajectory import TrajectoryData, verify_replay_preflight
from dexmani_real.robot.arm_worker import arm_loop as _arm_loop
from dexmani_real.robot.hand_worker import hand_loop as _hand_loop
from dexmani_real.runtime.safety import SafetyState, require_transition
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.runtime.workers import (
    ShutdownReport,
    WorkerSpec,
    build_processes,
    start_processes,
)
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "replay_results"


@dataclass(frozen=True)
class EpisodeReplayConfig:
    """Output and evaluation settings for one physical replay."""

    output_dir: str
    evaluate_consistency: bool
    config_sha256: str


def _latched_fault_status(shared: RuntimeChannels) -> ReplayStatus | None:
    """Classify sticky runtime state before any transition can mask it."""
    if shared.estop_request.value:
        return ReplayStatus.ESTOP
    if shared.error_state.value or int(shared.safety_state.value) == int(
        SafetyState.FAULT
    ):
        return ReplayStatus.FAULT
    return None


def _post_shutdown_outcome(
    shared: RuntimeChannels, outcome: ReplayOutcome, report: ShutdownReport
) -> ReplayOutcome:
    """Apply faults observed only after workers have reached a terminal state."""
    if bool(shared.estop_request.value):
        shutdown_reason = "e-stop latched during replay shutdown"
        reason = (
            f"{outcome.reason}; {shutdown_reason}"
            if outcome.reason
            else shutdown_reason
        )
        return ReplayOutcome(
            ReplayStatus.ESTOP,
            outcome.replay_data,
            reason,
        )
    abnormal_exits = tuple(
        item
        for item in report.exits
        if item.exitcode != 0 or item.escalation != "graceful"
    )
    if (
        not report.shared_closed
        or bool(shared.error_state.value)
        or int(shared.safety_state.value) == int(SafetyState.FAULT)
        or abnormal_exits
    ):
        failed = ", ".join(
            f"{item.name}={item.escalation}:{item.exitcode}"
            for item in abnormal_exits
        )
        shutdown_reason = (
            f"worker failed during replay shutdown: {failed}"
            if failed
            else "fault latched during replay shutdown"
        )
        reason = (
            f"{outcome.reason}; {shutdown_reason}"
            if outcome.reason
            else shutdown_reason
        )
        return ReplayOutcome(ReplayStatus.FAULT, outcome.replay_data, reason)
    return outcome


def _worker_health_issue(
    shared: RuntimeChannels,
    processes: list[Any],
    heartbeat_timeouts_s: dict[str, float],
    *,
    now_s: float | None = None,
) -> str | None:
    """Return the first arm/hand worker-health failure, if any."""
    for process in processes:
        if not process.is_alive():
            return f"worker {process.name!r} exited with code {process.exitcode}"

    heartbeat_by_name = {"arm", "hand"}
    for process in processes:
        if process.name not in heartbeat_by_name:
            continue
        last_s = shared.get_heartbeat(process.name)
        now = time.monotonic() if now_s is None else now_s
        timeout_s = float(heartbeat_timeouts_s[process.name])
        if (
            not np.isfinite(last_s)
            or last_s <= 0
            or last_s > now
            or now - last_s > timeout_s
        ):
            return f"worker {process.name!r} heartbeat timed out"
    return None


def _offer_return_home(
    shared: RuntimeChannels,
    replayer: EpisodeReplayer,
    runtime: ResolvedRuntimeConfig,
    *,
    hand_available: bool,
    health_check: Callable[[], str | None],
) -> tuple[ReplayStatus, str] | None:
    """Post-replay prompt: press H to return arm/hand to home, Q to exit."""
    print("\nPress H to return_home, or Q to exit...")
    keyboard = KeyboardHandler(
        estop_callback=lambda: setattr(shared.estop_request, "value", True)
    )
    keyboard.start()
    try:
        deadline = time.perf_counter() + float(runtime.policy.post_teleop_timeout_s)
        while time.perf_counter() < deadline:
            signals = set(keyboard.poll(timeout=0.1))
            if ControlSignal.EMERGENCY_STOP in signals or shared.estop_request.value:
                shared.estop_request.value = True
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                return (
                    ReplayStatus.ESTOP,
                    "operator emergency stop during return-home prompt",
                )
            if shared.error_state.value or int(shared.safety_state.value) == int(
                SafetyState.FAULT
            ):
                require_transition(shared, SafetyState.FAULT)
                return ReplayStatus.FAULT, "runtime fault during return-home prompt"
            health_issue = health_check()
            if health_issue:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                return ReplayStatus.FAULT, health_issue
            if ControlSignal.QUIT in signals:
                return None
            if ControlSignal.HOME not in signals:
                continue

            if hand_available:
                hand_home = np.deg2rad(
                    np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)
                )
                hand_accepted = publish_hand_home_and_wait_accepted(
                    shared,
                    hand_home,
                    command_lower_rad=np.asarray(
                        runtime.hand.qpos_min_rad, dtype=np.float64
                    ),
                    command_upper_rad=np.asarray(
                        runtime.hand.qpos_max_rad, dtype=np.float64
                    ),
                    mechanical_lower_rad=np.asarray(
                        runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
                    ),
                    mechanical_upper_rad=np.asarray(
                        runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
                    ),
                    hand_feedback_max_age_s=float(
                        runtime.safety.heartbeat_timeouts["hand"]
                    ),
                    timeout_s=float(runtime.hand.home_command_ack_timeout_s),
                    heartbeat=False,
                    check_is_running=False,
                    verbose=True,
                    abort_requested=lambda: keyboard.estop_latched
                    or not keyboard.healthy,
                )
                if not hand_accepted:
                    logger.warning(
                        "arm home cancelled because hand-home command was not accepted"
                    )
                    continue
                assert replayer.planner is not None
                replayer.planner.set_hand_qpos(hand_home)

            home_result = execute_arm_home(
                shared,
                np.asarray(runtime.arm.home_qpos, dtype=np.float64),
                planner=replayer.planner,
                config=ArmHomeConfig.from_runtime(
                    runtime,
                    publish_policy_heartbeat=False,
                ),
                table_z_surface_m=float(runtime.arm.table_z_surface_m),
                estop_requested=lambda: keyboard.estop_latched or not keyboard.healthy,
                progress=lambda message: print(f"  {message}", flush=True),
            )
            if not home_result.succeeded:
                if shared.estop_request.value:
                    shared.error_state.value = True
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.ESTOP, "e-stop requested during return home"
                if shared.error_state.value or int(shared.safety_state.value) == int(
                    SafetyState.FAULT
                ):
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.FAULT, "runtime fault during return home"
                health_issue = health_check()
                if health_issue:
                    shared.error_state.value = True
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.FAULT, health_issue
                return ReplayStatus.REJECTED, "return-home request was not completed"
            print("Press Q to exit...")
    finally:
        keyboard.stop()
    if shared.estop_request.value:
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.ESTOP, "e-stop requested when return-home prompt expired"
    if shared.error_state.value or int(shared.safety_state.value) == int(
        SafetyState.FAULT
    ):
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.FAULT, "runtime fault when return-home prompt expired"
    health_issue = health_check()
    if health_issue:
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.FAULT, health_issue
    return None


def replay_episode(
    trajectory: TrajectoryData,
    runtime: ResolvedRuntimeConfig,
    config: EpisodeReplayConfig,
) -> ReplayOutcome:
    """Start arm/hand workers, run one replay, then shut the session down."""
    try:
        verify_replay_preflight(
            trajectory,
            runtime,
            provenance_sha256=config.config_sha256,
        )
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        return ReplayOutcome(
            ReplayStatus.REJECTED,
            reason=f"physical replay preflight rejected: {exc}",
        )

    print("\n" + "=" * 60)
    print("Replay — RuntimeChannels architecture (arm_loop + hand_loop)")
    print("=" * 60)

    context = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=f"dexmani_replay_{os.getpid()}",
        config=RuntimeChannelsConfig.from_runtime(runtime),
        mp_context=context,
    )
    processes: list[Any] = []
    replayer: EpisodeReplayer | None = None
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="replay did not start")
    try:
        hand_available = trajectory.has_hand
        specs = [WorkerSpec("arm", _arm_loop, (shared, runtime.arm), ready_name="arm")]
        if hand_available:
            specs.append(
                WorkerSpec(
                    "hand", _hand_loop, (shared, runtime.hand), ready_name="hand"
                )
            )

        require_transition(shared, SafetyState.DISARMED)
        processes = build_processes(context, specs)
        start_processes(processes)

        timeouts = runtime.safety.readiness_timeouts_s
        ready_checks = [
            (spec.ready_name, float(timeouts[spec.ready_name]))
            for spec in specs
            if spec.ready_name
        ]
        workers_ready = wait_subsystem_ready(shared, ready_checks, processes)
        if not workers_ready:
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            outcome = ReplayOutcome(
                ReplayStatus.FAULT, reason="worker readiness failed"
            )
        else:
            heartbeat_timeouts = dict(runtime.safety.heartbeat_timeouts)

            def health_check() -> str | None:
                return _worker_health_issue(shared, processes, heartbeat_timeouts)

            replayer = EpisodeReplayer(
                trajectory,
                shared,
                runtime=runtime,
                health_check=health_check,
            )
            replayer.setup()
            latched_status = _latched_fault_status(shared)
            health_issue = None if latched_status is not None else health_check()
            if latched_status is not None:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(
                    latched_status, reason="fault latched before replay could arm"
                )
            elif health_issue:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(ReplayStatus.FAULT, reason=health_issue)
            else:
                require_transition(shared, SafetyState.ARMED)
                try:
                    outcome = replayer.run()
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    if replayer.can_offer_home:
                        shared.error_state.value = True
                        require_transition(shared, SafetyState.FAULT)
                        outcome = ReplayOutcome(
                            ReplayStatus.FAULT,
                            replayer.partial_data,
                            "replay interrupted before command quiescence could be established",
                        )
                    else:
                        outcome = ReplayOutcome(
                            ReplayStatus.USER_QUIT,
                            replayer.partial_data,
                            "KeyboardInterrupt",
                        )

                if (
                    outcome.status is ReplayStatus.COMPLETED
                    and not shared.error_state.value
                    and replayer.can_offer_home
                ):
                    home_outcome = _offer_return_home(
                        shared,
                        replayer,
                        runtime,
                        hand_available=hand_available,
                        health_check=health_check,
                    )
                    if home_outcome is not None:
                        status, reason = home_outcome
                        outcome = ReplayOutcome(status, outcome.replay_data, reason)
    except Exception:
        logger.error("physical replay session failed", exc_info=True)
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        raise
    finally:
        if replayer is not None:
            try:
                replayer.shutdown()
            except Exception:
                logger.error("replay controller shutdown failed", exc_info=True)
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(
                    ReplayStatus.FAULT,
                    outcome.replay_data,
                    outcome.reason or "replay controller shutdown failed",
                )

        if outcome.status is ReplayStatus.ESTOP:
            shared.estop_request.value = True
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
        elif outcome.status is ReplayStatus.FAULT:
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)

        started = [process for process in processes if process.pid is not None]
        try:
            shutdown_report = shutdown_processes(
                shared,
                started,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                disarm_if_clean=outcome.status
                not in (ReplayStatus.ESTOP, ReplayStatus.FAULT),
            )
        except RuntimeError:
            logger.critical(
                "child process remains alive; leaving RuntimeChannels linked",
                exc_info=True,
            )
            raise
        outcome = _post_shutdown_outcome(shared, outcome, shutdown_report)

    if replayer is not None:
        evaluate_replay(
            trajectory,
            outcome.replay_data,
            evaluate_consistency=config.evaluate_consistency,
            output_dir=config.output_dir,
        )
    return outcome
