"""Process readiness and liveness; observation freshness belongs to control steps."""

import logging
import math
import time

from dexmani_real.runtime.processes import (
    shutdown_processes_verified,
)
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    _revoke_motion_locked,
)


logger = logging.getLogger(__name__)


def wait_subsystem_ready(shared, process, timeout_s, *, check=None):
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{process.name}: readiness timeout must be finite and positive")
    event = getattr(shared.sensors, f"{process.name}_ready")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (
            (check is not None and not check())
            or not process.is_alive()
            or shared.error_state
            or shared.estop_request
            or not shared.sensors.is_running.value
        ):
            return False
        if event.wait(timeout=0.05):
            return True
    return False


class RuntimeSupervisor:
    """Own started children; workflows construct processes and choose their start order."""

    def __init__(self, shared, readiness_timeouts):
        self.shared = shared
        self.readiness_timeouts = readiness_timeouts
        self.started_processes = []

    def start(self, processes, *, wait_ready=True) -> None:
        """Start sensors; the local caller handles readiness when wait_ready=False."""
        for process in processes:
            # Even deferred readiness (Teleop VR) requires an explicit budget.
            timeout = self.readiness_timeouts.get(process.name)
            if timeout is None or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError(
                    f"active process {process.name!r} requires a finite positive readiness timeout"
                )
            process.start()
            self.started_processes.append(process)
            if wait_ready and not wait_subsystem_ready(
                self.shared, process, timeout, check=self.check
            ):
                raise RuntimeError(f"{process.name} failed to become ready")

    def check(self) -> bool:
        dead = [process for process in self.started_processes if not process.is_alive()]
        for process in dead:
            getattr(self.shared.sensors, f"{process.name}_ready").clear()
        if not dead:
            return True
        for process in dead:
            logger.error("required worker %s exited: %s", process.name, process.exitcode)
        with self.shared.motion_lock:
            self.shared.sensors.is_running.value = False
            self.shared.error_state = True
            _revoke_motion_locked(
                self.shared, SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT
            )
        return False

    def shutdown(self, *, graceful_timeout_s=5.0) -> bool:
        return shutdown_processes_verified(
            self.shared, self.started_processes, graceful_timeout_s=graceful_timeout_s
        )
