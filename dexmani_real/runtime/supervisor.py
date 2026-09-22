"""Process readiness and liveness; observation freshness belongs to control steps."""
import time
from dexmani_real.runtime.safety import SafetyState, revoke_motion
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def wait_subsystem_ready(shared, process, timeout_s):
    event = getattr(shared, f"{process.name}_ready")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not process.is_alive() or shared.error_state.value or shared.estop_request.value or not shared.is_running.value:
            return False
        if event.wait(timeout=0.05):
            return True
    return False


def start_processes(shared, processes, timeouts, started):
    for process in processes:
        process.start()
        started.append(process)
        if not wait_subsystem_ready(shared, process, timeouts[process.name]):
            raise RuntimeError(f"{process.name} failed to become ready")


def check_processes(shared, processes):
    """Sensor/recording failures revoke streaming but leave healthy hardware available."""
    for process in processes:
        if process.is_alive():
            continue
        getattr(shared, f"{process.name}_ready").clear()
        if process.name in {"arm", "hand"}:
            shared.error_state.value = True
        else:
            shared.workflow_failed.value = True
        if int(shared.safety_state.value) == int(SafetyState.RUNNING):
            revoke_motion(shared)
        if process.name in {"arm", "hand", "policy"}:
            return False
    return True


def run_supervisor(shared, processes):
    while shared.is_running.value and not shared.quit_requested.value:
        if shared.estop_request.value or shared.error_state.value:
            revoke_motion(shared, SafetyState.FAULT)
            break
        if not check_processes(shared, processes):
            break
        time.sleep(0.02)
