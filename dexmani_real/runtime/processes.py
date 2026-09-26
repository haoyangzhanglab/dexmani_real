"""Bounded process shutdown; never unlink shared memory while a worker is alive."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from dexmani_real.runtime.safety import RunEndReason, SafetyState, _revoke_motion_locked, transition
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)
_PHYSICAL_PROCESS_NAMES = frozenset({"arm", "hand"})


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
    def clean(self) -> bool:
        return self.shared_closed and all(
            item.exitcode == 0 and item.escalation == "graceful" for item in self.exits
        )


def _finalize_shutdown_state(
    shared: Any,
    report: ShutdownReport,
) -> None:
    """Latch physical failures, or disarm after verified terminal worker stop."""
    error_latched = bool(shared.error_state.value)
    estop_requested = bool(shared.estop_request.value)
    safety_state = int(shared.safety_state.value)
    physical_worker_failed = any(
        (item.exitcode != 0 or item.escalation != "graceful")
        and item.name in _PHYSICAL_PROCESS_NAMES
        for item in report.exits
    )
    faulted = (
        error_latched
        or estop_requested
        or safety_state == int(SafetyState.FAULT)
        or physical_worker_failed
    )

    if faulted:
        shared.error_state.value = True
        transition(shared, SafetyState.FAULT)
        return

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
    with shared.motion_lock:
        shared.is_running.value = False
        if int(shared.safety_state.value) == int(SafetyState.RUNNING):
            _revoke_motion_locked(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
    exits: list[ProcessExit] = []
    try:
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
                    raise RuntimeError(
                        f"process {process.name} ignored SIGTERM and kill() is unavailable"
                    )
                process.kill()
                process.join(timeout=kill_timeout_s)
            if process.is_alive() or process.exitcode is None:
                # Never unlink shared memory while a child may still access it.
                raise RuntimeError(
                    f"process {process.name} could not be confirmed stopped; RuntimeChannels remains open"
                )
            exits.append(ProcessExit(process.name, process.exitcode, escalation))

    except Exception:
        shared.error_state.value = True
        transition(shared, SafetyState.FAULT)
        raise

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
) -> ShutdownReport:
    """Stop workers, close verified IPC, and finalize physical safety."""
    frozen_exits = stop_processes_verified(
        shared,
        processes,
        graceful_timeout_s=graceful_timeout_s,
        terminate_timeout_s=terminate_timeout_s,
        kill_timeout_s=kill_timeout_s,
    )

    shared_closed = _close_runtime_channels(shared)
    report = ShutdownReport(frozen_exits, shared_closed=shared_closed)
    _finalize_shutdown_state(shared, report)
    logger.debug("verified process shutdown: %s", report.exits)
    return report
