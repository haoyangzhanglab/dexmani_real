"""Sensor process ownership and bounded shutdown of local runtime resources."""

import logging
import math
import time

from dexmani_real.runtime.state import RunEndReason, SafetyState

logger = logging.getLogger(__name__)


class SensorProcesses:
    """Own started sensor processes; workflows choose devices and start order."""

    def __init__(self, runtime_state):
        self.runtime_state = runtime_state
        self.started = []

    def start(self, process, *, ready, timeout_s=None, check_services=None) -> None:
        """Start ownership immediately; a timeout also requests a blocking readiness wait."""
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("readiness timeout must be finite and positive")

        def allowed():
            return (
                (check_services is None or check_services())
                and self.runtime_state.requests_allow_motion()
            )

        if not allowed():
            raise RuntimeError("sensor startup cancelled")
        process.start()
        self.started.append((process, ready))
        if timeout_s is not None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if not allowed() or not self.check():
                    break
                if ready.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic()))):
                    if allowed() and self.check():
                        return
                    break
            raise RuntimeError(f"{process.name} failed to become ready")

    def check(self) -> bool:
        dead = [(process, ready) for process, ready in self.started if not process.is_alive()]
        if not dead:
            return True
        for process, ready in dead:
            ready.clear()
            logger.error("required worker %s exited: %s", process.name, process.exitcode)
        self.runtime_state.end_session(reason=RunEndReason.HARDWARE_FAULT, fault=True)
        return False

    def shutdown(self, *, graceful_timeout_s=5.0, terminate_timeout_s=1.0, kill_timeout_s=1.0):
        """Verify every child exited; retain handles if any exit remains unresolved."""
        self.runtime_state.end_session()
        deadline = time.monotonic() + graceful_timeout_s
        clean = True
        errors = []
        for process, ready in self.started:
            try:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
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
                    raise RuntimeError(f"process {process.name} could not be confirmed stopped")
                ready.clear()
                if process.exitcode != 0 or escalation != "graceful":
                    clean = False
                    logger.warning(
                        "worker %s stopped: exitcode=%s escalation=%s",
                        process.name,
                        process.exitcode,
                        escalation,
                    )
            except Exception as exc:
                errors.append(exc)
                logger.exception("worker %s shutdown failed", process.name)
        if errors or not clean:
            self.runtime_state.end_session(fault=True)
        if errors:
            raise RuntimeError(f"sensor shutdown unresolved: {errors}") from errors[0]
        self.started.clear()
        return clean


def shutdown_local_runtime(
    runtime_state, robot=None, *, sensors=None, keyboard=None, timeout_s=5.0
):
    """Revoke, close devices and inputs, then stop sensors within their join budget."""
    with runtime_state.motion_lock:
        runtime_state.revoke_motion(
            (
                SafetyState.FAULT
                if runtime_state.safety_state == SafetyState.FAULT
                else SafetyState.DISARMED
            ),
            reason=RunEndReason.RUNTIME_SHUTDOWN,
        )
    for close in (
        robot.close if robot is not None else None,
        keyboard.stop if keyboard is not None else None,
    ):
        if close is None:
            continue
        try:
            close()
        except Exception:
            runtime_state.mark_error()
            logger.exception("local runtime cleanup failed")
    runtime_state.end_session()
    sensors_clean = True
    if sensors is not None:
        try:
            sensors_clean = sensors.shutdown(graceful_timeout_s=timeout_s)
        except Exception:
            runtime_state.mark_error()
            logger.exception("sensor shutdown failed; unresolved process handles retained")
            sensors_clean = False
    clean = (
        sensors_clean
        and not runtime_state.error_state
        and not runtime_state.estop_request
        and runtime_state.safety_state != SafetyState.FAULT
    )
    if not clean and not runtime_state.estop_request:
        runtime_state.mark_error()
    # DISARMED describes software authority only, never physical stopping.
    runtime_state.revoke_motion(SafetyState.DISARMED if clean else SafetyState.FAULT)
    return clean
