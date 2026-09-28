"""Process readiness and liveness; observation freshness belongs to control steps."""

import math
import time

from dexmani_real.runtime.processes import (
    _PHYSICAL_PROCESS_NAMES,
    shutdown_processes_verified,
)
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    _revoke_motion_locked,
    revoke_motion,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def wait_subsystem_ready(shared, process, timeout_s, *, check=None):
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{process.name}: readiness timeout must be finite and positive")
    event = getattr(shared, f"{process.name}_ready")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (
            (check is not None and not check())
            or not process.is_alive()
            or shared.error_state.value
            or shared.estop_request.value
            or not shared.is_running.value
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
        """Own child startup; a dependent worker may handle readiness when wait_ready=False."""
        for process in processes:
            # Even deferred readiness (Teleop VR/policy) requires an explicit budget.
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
        unexpected = []
        for process in dead:
            getattr(self.shared, f"{process.name}_ready").clear()
            if (
                process.name == "policy"
                and self.shared.quit_requested.value
                and process.exitcode == 0
            ):
                continue
            unexpected.append(process)
        if not unexpected:
            return True
        for process in unexpected:
            logger.error("required worker %s exited: %s", process.name, process.exitcode)
        with self.shared.motion_lock:
            self.shared.is_running.value = False
            if any(process.name in _PHYSICAL_PROCESS_NAMES for process in unexpected):
                self.shared.error_state.value = True
                _revoke_motion_locked(self.shared, SafetyState.FAULT)
            elif int(self.shared.safety_state.value) == int(SafetyState.RUNNING):
                _revoke_motion_locked(self.shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
        return False

    def run(self) -> bool:
        while self.shared.is_running.value:
            if self.shared.estop_request.value or self.shared.error_state.value:
                revoke_motion(self.shared, SafetyState.FAULT)
                return False
            if not self.check():
                return False
            if self.shared.quit_requested.value:
                return True
            time.sleep(0.02)
        return False

    def shutdown(self, *, graceful_timeout_s=5.0) -> bool:
        deadline = time.monotonic() + graceful_timeout_s
        if (
            self.shared.quit_requested.value
            and self.shared.is_running.value
            and not self.shared.error_state.value
            and not self.shared.estop_request.value
        ):
            # A normal quit lets the control owner finish its local writer
            # before process shutdown can interrupt publication.
            if int(self.shared.safety_state.value) in (
                int(SafetyState.ARMED),
                int(SafetyState.RUNNING),
            ):
                revoke_motion(self.shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
            for process in self.started_processes:
                if process.name == "policy":
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
        return shutdown_processes_verified(
            self.shared,
            self.started_processes,
            graceful_timeout_s=max(0.0, deadline - time.monotonic()),
        )
