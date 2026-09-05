"""Spawn-only worker construction, supervision priority, and shutdown."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from dexmani_real.runtime.safety import SafetyState, transition
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProcessExit:
    name: str
    exitcode: int | None
    escalation: str


@dataclass(frozen=True)
class ProcessSpec:
    """One process to construct: name, target, args, readiness key.

    ``ready_name`` is optional because only asynchronous initialization belongs
    in readiness checks. Heartbeats are selected independently by the lifecycle.
    """

    name: str
    target: Callable[..., None]
    args: tuple[Any, ...]
    ready_name: str | None = None
    daemon: bool = False


def build_processes(context: Any, specs: Iterable[ProcessSpec]) -> list[Any]:
    """Construct (but do not start) one process per spec."""
    return [
        context.Process(
            target=spec.target, args=spec.args, name=spec.name, daemon=spec.daemon
        )
        for spec in specs
    ]


def start_processes(processes: Iterable[Any]) -> None:
    """Start each process. Callers keep require_transition(DISARMED) between
    build and start so worker processes never race the safety transition."""
    for process in processes:
        process.start()


@dataclass(frozen=True)
class ShutdownReport:
    exits: tuple[ProcessExit, ...]
    shared_closed: bool


def _shared_value(shared: Any, name: str) -> Any | None:
    field = getattr(shared, name, None)
    return None if field is None else getattr(field, "value", None)


def _finalize_shutdown_state(
    shared: Any,
    exits: tuple[ProcessExit, ...],
    *,
    disarm_if_clean: bool,
) -> None:
    """Latch post-join failures, or disarm only after a verified clean stop."""
    error_latched = bool(_shared_value(shared, "error_state"))
    estop_requested = bool(_shared_value(shared, "estop_request"))
    safety_value = _shared_value(shared, "safety_state")
    safety_state = None if safety_value is None else int(safety_value)
    worker_failed = any(
        item.exitcode != 0 or item.escalation != "graceful" for item in exits
    )
    faulted = (
        error_latched
        or estop_requested
        or safety_state == int(SafetyState.FAULT)
        or worker_failed
    )

    if faulted:
        error_field = getattr(shared, "error_state", None)
        if error_field is not None:
            error_field.value = True
        if safety_state is not None:
            transition(shared, SafetyState.FAULT)
    elif disarm_if_clean and safety_state is not None:
        if not transition(shared, SafetyState.DISARMED):
            error_field = getattr(shared, "error_state", None)
            if error_field is not None:
                error_field.value = True
            transition(shared, SafetyState.FAULT)


def _close_runtime_channels(shared: Any) -> bool:
    """Close IPC and record resource-cleanup errors without changing safety state."""
    try:
        return bool(shared.close())
    except Exception:
        logger.error("RuntimeChannels cleanup raised", exc_info=True)
        return False


def _latch_unverified_shutdown_fault(shared: Any) -> None:
    """Fail closed when a child might still access live IPC resources."""
    error_field = getattr(shared, "error_state", None)
    if error_field is not None:
        error_field.value = True
    if _shared_value(shared, "safety_state") is not None:
        transition(shared, SafetyState.FAULT)


def stop_processes_verified(
    shared: Any,
    processes: Iterable[Any],
    *,
    graceful_timeout_s: float = 5.0,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
) -> tuple[ProcessExit, ...]:
    """Stop every worker without closing IPC that another local thread may use."""
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
                _latch_unverified_shutdown_fault(shared)
                raise RuntimeError(
                    f"process {process.name} ignored SIGTERM and kill() is unavailable"
                )
            process.kill()
            process.join(timeout=kill_timeout_s)
        if process.is_alive() or process.exitcode is None:
            # Never unlink shared memory while a child may still access it.
            _latch_unverified_shutdown_fault(shared)
            raise RuntimeError(
                f"process {process.name} could not be confirmed stopped; RuntimeChannels remains open"
            )
        exits.append(ProcessExit(process.name, process.exitcode, escalation))

    frozen_exits = tuple(exits)
    logger.info("verified process stop: %s", frozen_exits)
    return frozen_exits


def shutdown_processes_verified(
    shared: Any,
    processes: Iterable[Any],
    *,
    graceful_timeout_s: float = 5.0,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
    disarm_if_clean: bool = False,
) -> ShutdownReport:
    """Stop workers, finalize physical safety, then close IPC after verification."""
    frozen_exits = stop_processes_verified(
        shared,
        processes,
        graceful_timeout_s=graceful_timeout_s,
        terminate_timeout_s=terminate_timeout_s,
        kill_timeout_s=kill_timeout_s,
    )

    _finalize_shutdown_state(
        shared,
        frozen_exits,
        disarm_if_clean=disarm_if_clean,
    )
    shared_closed = _close_runtime_channels(shared)
    report = ShutdownReport(frozen_exits, shared_closed=shared_closed)
    logger.info("verified process shutdown: %s", report.exits)
    return report
