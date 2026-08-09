"""Spawn-only process construction, supervision priority, and shutdown."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from dexmani_real.runtime.status import ExitReason
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def spawn_context() -> Any:
    """Return the repository's sole multiprocessing context."""
    return mp.get_context("spawn")


@dataclass(frozen=True)
class ProcessExit:
    name: str
    exitcode: int | None
    escalation: str


@dataclass(frozen=True)
class ShutdownReport:
    exits: tuple[ProcessExit, ...]
    shared_closed: bool

    @property
    def all_stopped(self) -> bool:
        return all(item.exitcode is not None for item in self.exits)


def supervisor_exit_reason(
    shared: Any,
    processes: Iterable[Any],
    heartbeat_ages_s: Mapping[str, float],
    heartbeat_timeouts_s: Mapping[str, float],
) -> ExitReason:
    """Apply the fixed safety-first supervisor priority."""
    if bool(shared.estop_request.value):
        return ExitReason.ESTOP
    if bool(shared.error_state.value):
        return ExitReason.STICKY_FAULT
    if any(process.exitcode is not None for process in processes):
        return ExitReason.WORKER_DEATH
    if any(heartbeat_ages_s.get(name, float("inf")) > timeout for name, timeout in heartbeat_timeouts_s.items()):
        return ExitReason.HEARTBEAT_TIMEOUT
    if bool(shared.quit_requested.value) or not bool(shared.is_running.value):
        return ExitReason.EXPLICIT_QUIT
    return ExitReason.NONE


def shutdown_processes_verified(
    shared: Any,
    processes: Iterable[Any],
    *,
    graceful_timeout_s: float = 5.0,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
) -> ShutdownReport:
    """Join, terminate, then kill stragglers; close IPC only after all exited."""
    procs = list(processes)
    shared.is_running.value = False
    exits: list[ProcessExit] = []
    deadline = time.monotonic() + graceful_timeout_s
    for process in procs:
        process.join(timeout=max(0.0, deadline - time.monotonic()))

    for process in procs:
        escalation = "graceful"
        if process.is_alive():
            escalation = "terminate"
            process.terminate()
            process.join(timeout=terminate_timeout_s)
        if process.is_alive():
            escalation = "kill"
            if not hasattr(process, "kill"):
                raise RuntimeError(f"process {process.name} ignored SIGTERM and kill() is unavailable")
            process.kill()
            process.join(timeout=kill_timeout_s)
        if process.is_alive() or process.exitcode is None:
            # Never unlink shared memory while a child may still access it.
            raise RuntimeError(f"process {process.name} could not be confirmed stopped; SharedStorage remains open")
        exits.append(ProcessExit(process.name, process.exitcode, escalation))

    shared.close()
    report = ShutdownReport(tuple(exits), shared_closed=True)
    logger.info("verified process shutdown: %s", report.exits)
    return report
