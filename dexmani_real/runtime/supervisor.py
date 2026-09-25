"""Process readiness and liveness; observation freshness belongs to control steps."""

import time

from dexmani_real.runtime.processes import ShutdownReport, shutdown_processes_verified
from dexmani_real.runtime.safety import SafetyState, revoke_motion


def wait_subsystem_ready(shared, process, timeout_s):
    event = getattr(shared, f"{process.name}_ready")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (
            not process.is_alive()
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
                self.shared, process, self.readiness_timeouts[process.name]
            ):
                raise RuntimeError(f"{process.name} failed to become ready")

    def check(self) -> bool:
        """Service failures revoke streaming but leave healthy hardware available."""
        for process in self.started_processes:
            if process.is_alive():
                continue
            getattr(self.shared, f"{process.name}_ready").clear()
            if process.name in {"arm", "hand"}:
                self.shared.error_state.value = True
            else:
                self.shared.workflow_failed.value = True
            if int(self.shared.safety_state.value) == int(SafetyState.RUNNING):
                revoke_motion(self.shared)
            if process.name in {"arm", "hand", "policy"}:
                return False
        return True

    def run(self) -> None:
        while self.shared.is_running.value and not self.shared.quit_requested.value:
            if self.shared.estop_request.value or self.shared.error_state.value:
                revoke_motion(self.shared, SafetyState.FAULT)
                break
            if not self.check():
                break
            time.sleep(0.02)

    def shutdown(
        self, *, graceful_timeout_s=5.0, disarm_if_clean=False, service_process_names=()
    ) -> ShutdownReport:
        return shutdown_processes_verified(
            self.shared,
            self.started_processes,
            graceful_timeout_s=graceful_timeout_s,
            disarm_if_clean=disarm_if_clean,
            service_process_names=service_process_names,
        )
