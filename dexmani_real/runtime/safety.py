"""Four motion states and a run epoch shared by controllers and SDK workers."""
import time
from enum import IntEnum

class SafetyState(IntEnum):
    """Values stored in ``RuntimeChannels.safety_state``."""

    DISARMED = 0
    ARMED = 1
    RUNNING = 2
    FAULT = 3


class StopRequest(IntEnum):
    """Operator stop request carried from Main to the control owner."""

    NONE = 0
    OPERATOR = 1


class RunEndReason(IntEnum):
    """First software RUNNING termination cause; never physical convergence."""

    NONE = 0
    EXECUTOR_BOUNDARY = 1
    OPERATOR = 2
    QUIT = 3
    TIMEOUT = 4
    POLICY_FAILURE = 5
    ESTOP = 6
    HARDWARE_FAULT = 7
    RUNTIME_SHUTDOWN = 8
    RECORDING_FAILURE = 9


def _begin_motion_locked(shared):
    if (int(shared.safety_state.value) != int(SafetyState.ARMED)
            or not shared.is_running.value or shared.error_state.value or shared.estop_request.value):
        return None
    shared.run_id.value += 1
    started = time.monotonic_ns()
    shared.run_started_monotonic_ns.value = started
    shared.run_ended_reason.value = int(RunEndReason.NONE)
    shared.safety_state.value = int(SafetyState.RUNNING)
    return int(shared.run_id.value), started


def begin_motion(shared):
    with shared.motion_lock:
        return _begin_motion_locked(shared) is not None


def command_may_cross_sdk(shared, *, run_id, required_safety_state=SafetyState.RUNNING):
    """Last software fence; no lock is held during the subsequent SDK call."""
    with shared.motion_lock:
        return (shared.is_running.value and not shared.error_state.value and not shared.estop_request.value
            and int(shared.run_id.value) == run_id
            and int(shared.safety_state.value) == int(required_safety_state)
            and required_safety_state in (SafetyState.ARMED, SafetyState.RUNNING))


def revoke_motion(shared, new_state=SafetyState.ARMED, *, reason=RunEndReason.EXECUTOR_BOUNDARY):
    if new_state not in (SafetyState.ARMED, SafetyState.DISARMED, SafetyState.FAULT):
        raise ValueError("revocation must leave streaming mode")
    with shared.motion_lock:
        current = SafetyState(int(shared.safety_state.value))
        if current == SafetyState.FAULT and new_state == SafetyState.ARMED:
            return False
        if current == SafetyState.RUNNING:
            if shared.estop_request.value:
                reason = RunEndReason.ESTOP
            elif shared.error_state.value or new_state == SafetyState.FAULT:
                reason = RunEndReason.HARDWARE_FAULT
            shared.run_ended_reason.value = int(reason)
        shared.run_id.value += 1
        shared.run_started_monotonic_ns.value = 0
        shared.safety_state.value = int(new_state)
    return True


def revoke_motion_if_run_id(shared, expected_run_id, new_state=SafetyState.ARMED, *, reason=RunEndReason.EXECUTOR_BOUNDARY):
    with shared.motion_lock:
        if int(shared.run_id.value) != expected_run_id:
            return False
        return revoke_motion(shared, new_state, reason=reason)


def transition(shared, new_state):
    return begin_motion(shared) if new_state == SafetyState.RUNNING else revoke_motion(shared, new_state)


def require_transition(shared, new_state):
    if not transition(shared, new_state):
        raise RuntimeError(f"safety transition to {new_state.name} rejected")


def request_policy_start(shared, *, require_physical_home):
    with shared.motion_lock:
        if (int(shared.safety_state.value) != int(SafetyState.ARMED) or not shared.is_running.value
                or shared.error_state.value or shared.estop_request.value or shared.workflow_failed.value
                or shared.stop_request.value or (require_physical_home and not shared.physical_home_completed.value)):
            return False
        shared.start_request.value = True
        return True


def begin_requested_motion(shared):
    with shared.motion_lock:
        if not shared.start_request.value or shared.stop_request.value:
            return None
        epoch = _begin_motion_locked(shared)
        if epoch is not None:
            shared.start_request.value = False
        return epoch


def request_policy_stop(shared, *, reason=RunEndReason.OPERATOR):
    with shared.motion_lock:
        shared.start_request.value = False
        shared.physical_home_completed.value = False
        shared.stop_request.value = int(StopRequest.OPERATOR)
        if int(shared.safety_state.value) in (int(SafetyState.ARMED), int(SafetyState.RUNNING)):
            return revoke_motion(shared, reason=reason)
        return True
