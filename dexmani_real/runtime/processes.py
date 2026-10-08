"""Bounded process shutdown; never unlink shared memory while a worker is alive."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable

from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    _revoke_motion_locked,
    revoke_motion,
    transition,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProcessExit:
    name: str
    exitcode: int | None
    escalation: str


def _finalize_shutdown_state(
    shared: Any,
    exits: tuple[_ProcessExit, ...],
) -> None:
    """Latch local/child failures; DISARMED describes software authority only."""
    error_latched = bool(shared.error_state)
    estop_requested = bool(shared.estop_request)
    safety_state = int(shared.safety_state)
    child_failed = any(item.exitcode != 0 or item.escalation != "graceful" for item in exits)
    faulted = (
        error_latched or estop_requested or safety_state == int(SafetyState.FAULT) or child_failed
    )

    if faulted:
        if (
            error_latched
            or child_failed
            or (safety_state == int(SafetyState.FAULT) and not estop_requested)
        ):
            shared.error_state = True
        transition(shared, SafetyState.FAULT)
        return

    if not transition(shared, SafetyState.DISARMED):
        shared.error_state = True
        transition(shared, SafetyState.FAULT)


def stop_processes_verified(
    shared: Any,
    processes: Iterable[Any],
    *,
    graceful_timeout_s: float = 5.0,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
) -> tuple[_ProcessExit, ...]:
    """Confirm all sensor children exited before their shared resources can be released."""
    procs = list(processes)
    # Fence before any blocking join; record the software end if still RUNNING.
    with shared.motion_lock:
        shared.sensors.is_running.value = False
        if int(shared.safety_state) == int(SafetyState.RUNNING):
            _revoke_motion_locked(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
    exits: list[_ProcessExit] = []
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
                    f"process {process.name} could not be confirmed stopped; sensor channels remain open"
                )
            exits.append(_ProcessExit(process.name, process.exitcode, escalation))

    except Exception:
        shared.error_state = True
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
) -> bool:
    """Stop sensors, close verified IPC, and finalize software safety state."""
    frozen_exits = stop_processes_verified(
        shared,
        processes,
        graceful_timeout_s=graceful_timeout_s,
        terminate_timeout_s=terminate_timeout_s,
        kill_timeout_s=kill_timeout_s,
    )

    try:
        shared_closed = bool(shared.sensors.close())
    except Exception:
        logger.error("sensor channel cleanup raised", exc_info=True)
        shared_closed = False
    if not shared_closed:
        shared.error_state = True
    _finalize_shutdown_state(shared, frozen_exits)
    logger.debug("verified process shutdown: %s", frozen_exits)
    return shared_closed and all(
        item.exitcode == 0 and item.escalation == "graceful" for item in frozen_exits
    )


def shutdown_local_runtime(robot, supervisor, *, model=None, keyboard=None, timeout_s=5.0):
    """Stop local devices before closing resources and verified sensor IPC release."""
    shared = supervisor.shared
    revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
    clean = True
    for close in (
        robot.close if robot is not None and robot._owner is not None else None,
        model.close if model is not None else None,
        keyboard.stop if keyboard is not None else None,
    ):
        if close is None:
            continue
        try:
            close()
        except Exception:
            clean = False
            shared.error_state = True
            logger.exception("local runtime cleanup failed")
    try:
        sensors_clean = supervisor.shutdown(graceful_timeout_s=timeout_s)
    except Exception:
        shared.error_state = True
        logger.exception("sensor shutdown failed; live resources remain linked")
        sensors_clean = False
    return (
        clean and sensors_clean and not shared.error_state and not shared.estop_request
    )
