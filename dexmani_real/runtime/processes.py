"""Bounded process shutdown; never unlink shared memory while a worker is alive."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    _revoke_motion_locked,
    revoke_motion,
    transition,
)


logger = logging.getLogger(__name__)


def shutdown_processes_verified(
    shared: Any,
    processes: Iterable[Any],
    *,
    graceful_timeout_s: float = 5.0,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
) -> bool:
    """Revoke, join, then close channels; return whether cleanup was graceful and complete."""
    procs = list(processes)
    # Fence before any blocking join; record the software end if still RUNNING.
    with shared.motion_lock:
        shared.end_session()
        if int(shared.safety_state) == int(SafetyState.RUNNING):
            _revoke_motion_locked(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
    clean = True
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
                process.kill()
                process.join(timeout=kill_timeout_s)
            if process.is_alive() or process.exitcode is None:
                # Never unlink shared memory while a child may still access it.
                raise RuntimeError(
                    f"process {process.name} could not be confirmed stopped; sensor channels remain open"
                )
            if process.exitcode != 0 or escalation != "graceful":
                clean = False
                logger.warning(
                    "worker %s stopped: exitcode=%s escalation=%s",
                    process.name, process.exitcode, escalation,
                )

    except Exception:
        shared.error_state = True
        transition(shared, SafetyState.FAULT)
        raise

    try:
        if shared.sensors is not None and not shared.sensors.close():
            clean = False
    except Exception:
        logger.error("sensor channel cleanup raised", exc_info=True)
        clean = False
    if not clean:
        shared.error_state = True
    # DISARMED describes software authority only, never physical stopping.
    with shared.motion_lock:
        faulted = (
            shared.error_state or shared.estop_request or shared.safety_state == SafetyState.FAULT
        )
        if faulted and not shared.estop_request:
            shared.error_state = True
        transition(shared, SafetyState.FAULT if faulted else SafetyState.DISARMED)
    return clean


def shutdown_local_runtime(robot, supervisor=None, *, shared=None, keyboard=None, timeout_s=5.0):
    """Close local devices; timeout_s bounds the sensor supervisor's graceful join."""
    shared = supervisor.shared if supervisor is not None else shared
    revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
    for close in (
        robot.close if robot is not None and robot._owner is not None else None,
        keyboard.stop if keyboard is not None else None,
    ):
        if close is None:
            continue
        try:
            close()
        except Exception:
            shared.error_state = True
            logger.exception("local runtime cleanup failed")
    try:
        sensors_clean = (
            shutdown_processes_verified(shared, (), graceful_timeout_s=timeout_s)
            if supervisor is None else supervisor.shutdown(graceful_timeout_s=timeout_s)
        )
    except Exception:
        shared.error_state = True
        logger.exception("sensor shutdown failed; live resources remain linked")
        sensors_clean = False
    return sensors_clean and not shared.error_state and not shared.estop_request
