"""Main-process motion authority shared by the I/O owner and operator callbacks."""

import time
from enum import IntEnum


class SafetyState(IntEnum):
    DISARMED = 0
    ARMED = 1
    RUNNING = 2
    FAULT = 3


class StopRequest(IntEnum):
    """Operator stop request consumed by the local policy runner."""

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
    if (
        int(shared.safety_state) != int(SafetyState.ARMED)
        or not shared.running
        or shared.error_state
        or shared.estop_request
    ):
        return None
    shared.run_id += 1
    started = time.monotonic_ns()
    shared.run_ended_reason = int(RunEndReason.NONE)
    shared.safety_state = int(SafetyState.RUNNING)
    publish_arm_authority(shared)
    return int(shared.run_id), started


def begin_motion(shared):
    with shared.motion_lock:
        return _begin_motion_locked(shared) is not None


def _command_may_cross_sdk_locked(shared, *, run_id, required_safety_state=SafetyState.RUNNING):
    """Check motion authority while the caller holds motion_lock."""
    return (
        shared.running
        and not shared.error_state
        and not shared.estop_request
        and int(shared.run_id) == run_id
        and int(shared.safety_state) == int(required_safety_state)
        and required_safety_state in (SafetyState.ARMED, SafetyState.RUNNING)
    )


def command_may_cross_sdk(shared, *, run_id, required_safety_state=SafetyState.RUNNING):
    """Last software fence; no lock is held during the subsequent SDK call."""
    with shared.motion_lock:
        return _command_may_cross_sdk_locked(
            shared, run_id=run_id, required_safety_state=required_safety_state
        )


def _revoke_motion_locked(
    shared, new_state=SafetyState.ARMED, *, reason=RunEndReason.EXECUTOR_BOUNDARY
):
    """Revoke the current epoch while the caller holds motion_lock."""
    if new_state not in (SafetyState.ARMED, SafetyState.DISARMED, SafetyState.FAULT):
        raise ValueError("revocation must leave streaming mode")
    current = SafetyState(int(shared.safety_state))
    if current == SafetyState.FAULT and new_state == SafetyState.ARMED:
        return False
    if current == SafetyState.RUNNING:
        if shared.estop_request:
            reason = RunEndReason.ESTOP
        elif new_state == SafetyState.FAULT and reason == RunEndReason.EXECUTOR_BOUNDARY:
            reason = RunEndReason.HARDWARE_FAULT
        shared.run_ended_id = int(shared.run_id)
        shared.run_ended_reason = int(reason)
    shared.run_id += 1
    shared.safety_state = int(new_state)
    publish_arm_authority(shared)
    return True


def publish_arm_authority(shared):
    """Signed epoch: ARMED preparation < 0, RUNNING > 0, revoked = 0.

    RuntimeState remains local. The child never waits for a parent callback or
    reads the parent's sensor buffers to decide whether it may send a point.
    """
    signal = getattr(shared, 'arm_authority', None)
    if signal is None:
        return
    with shared.motion_lock:
        if (not shared.running or shared.error_state or shared.estop_request
                or shared.quit_requested or shared.stop_request):
            signal.value = 0
        elif int(shared.safety_state) == SafetyState.RUNNING:
            signal.value = int(shared.run_id)
        elif int(shared.safety_state) == SafetyState.ARMED:
            signal.value = -int(shared.run_id)
        else:
            signal.value = 0
        if shared.estop_request and shared.arm_emergency is not None:
            shared.arm_emergency.set()


def request_emergency_stop(shared):
    with shared.motion_lock:
        already_requested = shared.estop_request
        shared.estop_request = True
        publish_arm_authority(shared)
        if not already_requested:
            _revoke_motion_locked(shared, reason=RunEndReason.ESTOP)


def revoke_motion(shared, new_state=SafetyState.ARMED, *, reason=RunEndReason.EXECUTOR_BOUNDARY):
    with shared.motion_lock:
        return _revoke_motion_locked(shared, new_state, reason=reason)


def revoke_motion_if_run_id(
    shared,
    expected_run_id,
    new_state=SafetyState.ARMED,
    *,
    reason=RunEndReason.EXECUTOR_BOUNDARY,
):
    with shared.motion_lock:
        if int(shared.run_id) != expected_run_id:
            return False
        return _revoke_motion_locked(shared, new_state, reason=reason)


def transition(shared, new_state):
    return (
        begin_motion(shared)
        if new_state == SafetyState.RUNNING
        else revoke_motion(shared, new_state)
    )


def require_transition(shared, new_state):
    if not transition(shared, new_state):
        raise RuntimeError(f"safety transition to {new_state.name} rejected")


def request_policy_start(shared):
    with shared.motion_lock:
        if (
            int(shared.safety_state) != int(SafetyState.ARMED)
            or not shared.running
            or shared.error_state
            or shared.estop_request
            or shared.stop_request
        ):
            return False
        shared.start_request = True
        return True


def request_policy_stop(shared, *, reason=RunEndReason.OPERATOR):
    with shared.motion_lock:
        shared.start_request = False
        shared.stop_request = int(StopRequest.OPERATOR)
        if int(shared.safety_state) in (int(SafetyState.ARMED), int(SafetyState.RUNNING)):
            return _revoke_motion_locked(shared, reason=reason)
        return True
