"""Process readiness and liveness; observation freshness belongs to control steps."""

import time

from dexmani_real.runtime.processes import (
    _PHYSICAL_PROCESS_NAMES,
    ShutdownReport,
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

    def start(self, processes) -> None:
        for process in processes:
            process.start()
            self.started_processes.append(process)
            if not wait_subsystem_ready(
                self.shared, process, self.readiness_timeouts[process.name], check=self.check
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

    def shutdown(self, *, graceful_timeout_s=5.0) -> ShutdownReport:
        deadline = time.monotonic() + graceful_timeout_s
        if (
            self.shared.quit_requested.value
            and self.shared.is_running.value
            and not self.shared.error_state.value
            and not self.shared.estop_request.value
        ):
            # A normal quit must let the control owner send its final STOP before
            # RecorderIO falls back to invalidating an interrupted capture.
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
