"""Spawn-only process construction, supervision priority, and shutdown."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.runtime.status import ExitReason
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProcessExit:
    name: str
    exitcode: int | None
    escalation: str


@dataclass(frozen=True)
class WorkerSpec:
    """One worker process to construct: name, target, args, readiness key.

    ``ready_name`` is the SharedStorage readiness/heartbeat key (it may differ
    from the OS process ``name``, e.g. a single "arm" worker named "arm-calib").
    """

    name: str
    target: Callable[..., None]
    args: tuple[Any, ...]
    ready_name: str | None = None
    daemon: bool = False


def build_processes(context: Any, specs: Iterable[WorkerSpec]) -> list[Any]:
    """Construct (but do not start) one process per spec."""
    return [
        context.Process(target=spec.target, args=spec.args, name=spec.name, daemon=spec.daemon)
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
    error_latched: bool = False
    estop_requested: bool = False
    safety_state: int | None = None

    @property
    def all_stopped(self) -> bool:
        return all(item.exitcode is not None for item in self.exits)

    @property
    def abnormal_exits(self) -> tuple[ProcessExit, ...]:
        return tuple(item for item in self.exits if item.exitcode != 0 or item.escalation != "graceful")

    @property
    def faulted(self) -> bool:
        return (
            self.error_latched
            or self.estop_requested
            or self.safety_state == int(SafetyState.FAULT)
            or bool(self.abnormal_exits)
        )

    @property
    def clean(self) -> bool:
        safety_is_clean = self.safety_state in (None, int(SafetyState.DISARMED))
        return self.all_stopped and self.shared_closed and safety_is_clean and not self.faulted


def _shared_value(shared: Any, name: str) -> Any | None:
    field = getattr(shared, name, None)
    return None if field is None else getattr(field, "value", None)


def _finalize_shutdown_state(
    shared: Any,
    exits: tuple[ProcessExit, ...],
    *,
    disarm_if_clean: bool,
) -> tuple[bool, bool, int | None]:
    """Latch post-join failures, or disarm only after a verified clean stop."""
    error_latched = bool(_shared_value(shared, "error_state"))
    estop_requested = bool(_shared_value(shared, "estop_request"))
    safety_value = _shared_value(shared, "safety_state")
    safety_state = None if safety_value is None else int(safety_value)
    worker_failed = any(item.exitcode != 0 or item.escalation != "graceful" for item in exits)
    faulted = error_latched or estop_requested or safety_state == int(SafetyState.FAULT) or worker_failed

    if faulted:
        error_field = getattr(shared, "error_state", None)
        if error_field is not None:
            error_field.value = True
            error_latched = True
        if safety_state is not None:
            transition(shared, SafetyState.FAULT)
    elif disarm_if_clean and safety_state is not None:
        if not transition(shared, SafetyState.DISARMED):
            error_field = getattr(shared, "error_state", None)
            if error_field is not None:
                error_field.value = True
                error_latched = True
            transition(shared, SafetyState.FAULT)

    final_safety_value = _shared_value(shared, "safety_state")
    final_safety_state = None if final_safety_value is None else int(final_safety_value)
    return error_latched, estop_requested, final_safety_state


def _close_shared_storage(shared: Any) -> bool:
    """Close IPC and convert cleanup exceptions into a failed shutdown report."""
    try:
        return bool(shared.close())
    except Exception:
        logger.error("SharedStorage cleanup raised", exc_info=True)
        return False


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
    stopped = [process for process in processes if process.exitcode is not None]
    explicit_quit = bool(shared.quit_requested.value) or not bool(shared.is_running.value)
    # Accept the policy's intentional zero exit without masking worker failures.
    if stopped and explicit_quit and all(int(process.exitcode) == 0 for process in stopped):
        return ExitReason.EXPLICIT_QUIT
    if stopped:
        return ExitReason.WORKER_DEATH
    for name, timeout in heartbeat_timeouts_s.items():
        age_s = float(heartbeat_ages_s.get(name, float("inf")))
        timeout_s = float(timeout)
        if (
            not math.isfinite(age_s)
            or age_s < 0.0
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
            or age_s > timeout_s
        ):
            return ExitReason.HEARTBEAT_TIMEOUT
    if explicit_quit:
        return ExitReason.EXPLICIT_QUIT
    return ExitReason.NONE


def shutdown_processes_verified(
    shared: Any,
    processes: Iterable[Any],
    *,
    graceful_timeout_s: float = 5.0,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
    disarm_if_clean: bool = False,
) -> ShutdownReport:
    """Join workers, finalize safety from their terminal state, then close IPC."""
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

    frozen_exits = tuple(exits)
    error_latched, estop_requested, safety_state = _finalize_shutdown_state(
        shared,
        frozen_exits,
        disarm_if_clean=disarm_if_clean,
    )
    shared_closed = _close_shared_storage(shared)
    if not shared_closed:
        error_field = getattr(shared, "error_state", None)
        if error_field is not None:
            error_field.value = True
            error_latched = True
        if safety_state is not None:
            transition(shared, SafetyState.FAULT)
            final_safety_value = _shared_value(shared, "safety_state")
            safety_state = None if final_safety_value is None else int(final_safety_value)
    report = ShutdownReport(
        frozen_exits,
        shared_closed=shared_closed,
        error_latched=error_latched,
        estop_requested=estop_requested,
        safety_state=safety_state,
    )
    logger.info("verified process shutdown: %s", report.exits)
    return report
