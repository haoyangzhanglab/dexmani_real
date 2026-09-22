"""Bounded process shutdown; never unlink shared memory while a worker is alive."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Collection, Iterable

from dexmani_real.runtime.safety import RunEndReason, SafetyState, revoke_motion, transition
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProcessExit:
    name: str
    exitcode: int | None
    escalation: str


@dataclass(frozen=True)
class ShutdownReport:
    exits: tuple[ProcessExit, ...]
    shared_closed: bool


def _finalize_shutdown_state(
    shared: Any,
    exits: tuple[ProcessExit, ...],
    *,
    disarm_if_clean: bool,
    service_process_names: Collection[str] = (),
) -> None:
    """Latch post-join failures, or disarm only after a verified clean stop.

    A confirmed-stopped ``service_process_names`` member's nonzero/escalated
    exit fails the session (``workflow_failed``) without claiming a physical
    fault. Any other (critical) process failing the same way remains FAULT.
    This classification only applies after process termination is verified —
    a child that cannot be confirmed stopped still faults the runtime.
    """
    error_latched = bool(shared.error_state.value)
    estop_requested = bool(shared.estop_request.value)
    safety_state = int(shared.safety_state.value)
    critical_worker_failed = any(
        (item.exitcode != 0 or item.escalation != "graceful")
        and item.name not in service_process_names
        for item in exits
    )
    service_worker_failed = any(
        (item.exitcode != 0 or item.escalation != "graceful") and item.name in service_process_names
        for item in exits
    )
    faulted = (
        error_latched
        or estop_requested
        or safety_state == int(SafetyState.FAULT)
        or critical_worker_failed
    )

    if faulted:
        shared.error_state.value = True
        transition(shared, SafetyState.FAULT)
        return

    if service_worker_failed:
        shared.workflow_failed.value = True

    if disarm_if_clean:
        if not transition(shared, SafetyState.DISARMED):
            shared.error_state.value = True
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
    shared.error_state.value = True
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
    # Fence before any blocking join; record the software end if still RUNNING.
    if int(shared.safety_state.value) == int(SafetyState.RUNNING):
        revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
    if shared.is_recording.value and (
        shared.error_state.value or shared.estop_request.value or shared.workflow_failed.value
    ):
        from dexmani_real.recording.client import write_recording_failure

        write_recording_failure(shared, "invalid session interrupted recording")
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
    log_stop = (
        logger.warning
        if any(item.exitcode != 0 or item.escalation != "graceful" for item in frozen_exits)
        else logger.debug
    )
    log_stop("verified process stop: %s", frozen_exits)
    return frozen_exits


def shutdown_processes_verified(
    shared: Any,
    processes: Iterable[Any],
    *,
    graceful_timeout_s: float = 5.0,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
    disarm_if_clean: bool = False,
    service_process_names: Collection[str] = (),
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
        service_process_names=service_process_names,
    )
    shared_closed = _close_runtime_channels(shared)
    report = ShutdownReport(frozen_exits, shared_closed=shared_closed)
    logger.debug("verified process shutdown: %s", report.exits)
    return report
