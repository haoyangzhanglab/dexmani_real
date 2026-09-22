"""Validated transitions for the shared runtime safety state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

PUBLISH_REASON_RUNTIME_STOPPED = "runtime stopped"
PUBLISH_REASON_ESTOP = "e-stop requested"
PUBLISH_REASON_FAULT = "sticky fault"
PUBLISH_REASON_SAFETY_STATE = "safety state does not permit motion"
PUBLISH_REASON_RUN = "command run no longer owns motion"
PUBLISH_REASON_PENDING = "previous command adoption/accounting pending"


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


_ALLOWED_TRANSITIONS = frozenset(
    {
        (SafetyState.DISARMED, SafetyState.ARMED),
        (SafetyState.ARMED, SafetyState.DISARMED),
        (SafetyState.RUNNING, SafetyState.DISARMED),
        (SafetyState.ARMED, SafetyState.RUNNING),
        (SafetyState.RUNNING, SafetyState.ARMED),
        (SafetyState.DISARMED, SafetyState.FAULT),
        (SafetyState.ARMED, SafetyState.FAULT),
        (SafetyState.RUNNING, SafetyState.FAULT),
        (SafetyState.FAULT, SafetyState.DISARMED),
    }
)


@dataclass(frozen=True)
class MotionPermit:
    """One atomic snapshot of the software motion permission."""

    state: SafetyState
    run_id: int

    @property
    def allows_motion(self) -> bool:
        return self.state in (SafetyState.ARMED, SafetyState.RUNNING)


def _invalidate_coupled_commands_locked(shared: Any) -> int:
    """Fence prior commands and preserve the first pending accounting boundary."""
    if shared.pending_record_command_id.value and not shared.pending_record_revoked_ns.value:
        shared.pending_record_revoked_ns.value = time.monotonic_ns()
    shared.run_id.value += 1
    return int(shared.run_id.value)


def _read_motion_permit_locked(shared: Any) -> MotionPermit:
    """Read the permit while the caller owns ``motion_lock``."""
    try:
        state = SafetyState(int(shared.safety_state.value))
    except ValueError:
        state = SafetyState.FAULT
    return MotionPermit(state, int(shared.run_id.value))


def _begin_motion_locked(shared: Any) -> tuple[int, int] | None:
    """Enter RUNNING while the caller owns ``motion_lock``."""
    if (
        int(shared.safety_state.value) != int(SafetyState.ARMED)
        or bool(shared.pending_record_command_id.value)
        or not shared.is_running.value
        or shared.error_state.value
        or shared.estop_request.value
    ):
        return None
    epoch = _invalidate_coupled_commands_locked(shared)
    started_ns = time.monotonic_ns()
    shared.run_started_monotonic_ns.value = started_ns
    shared.run_started_id.value = epoch
    shared.safety_state.value = int(SafetyState.RUNNING)
    return epoch, started_ns


def _revoke_motion_locked(
    shared: Any, new_state: SafetyState, reason: RunEndReason = RunEndReason.EXECUTOR_BOUNDARY
) -> tuple[SafetyState, int] | None:
    """Revoke motion while the caller owns ``motion_lock``."""
    current_value = int(shared.safety_state.value)
    try:
        current = SafetyState(current_value)
    except ValueError:
        current = SafetyState.FAULT
        shared.safety_state.value = int(current)
    if current != new_state and (current, new_state) not in _ALLOWED_TRANSITIONS:
        logger.error(
            "safety: rejected revocation transition %s(%d) → %s(%d)",
            current.name,
            int(current),
            new_state.name,
            int(new_state),
        )
        return None
    if current is SafetyState.RUNNING and new_state is not SafetyState.RUNNING:
        # Only this actual state transition writes the terminal fact. Later
        # ARMED/home/cleanup revocations cannot replace it; command-only pause
        # invalidation never passes through this boundary.
        if shared.estop_request.value:
            reason = RunEndReason.ESTOP
        elif shared.error_state.value or new_state is SafetyState.FAULT:
            reason = RunEndReason.HARDWARE_FAULT
        shared.run_ended_id.value = int(shared.run_started_id.value)
        shared.run_ended_monotonic_ns.value = time.monotonic_ns()
        shared.run_ended_reason.value = int(reason)
    epoch = _invalidate_coupled_commands_locked(shared)
    shared.run_started_monotonic_ns.value = 0
    shared.safety_state.value = int(new_state)
    return current, epoch


def invalidate_coupled_commands(shared: Any) -> int:
    """Invalidate coupled commands while deliberately retaining lifecycle state.

    This narrow primitive is for an already-established pause boundary
    and for cancellation of a failed home command. Start/stop state changes
    must use :func:`begin_motion` or :func:`revoke_motion` instead.
    """
    with shared.motion_lock:
        return _invalidate_coupled_commands_locked(shared)


def cancel_coupled_command_if_current(
    shared: Any,
    *,
    command: Any,
) -> bool:
    """Revoke admission for *command* only while its run is current.

    An aborted wait fences its run, so a timed-out caller cannot revoke a
    newer run's command.
    """
    with shared.motion_lock:
        if not _committed_command_is_current_locked(shared, command):
            return False
        _invalidate_coupled_commands_locked(shared)
        return True


def read_motion_permit(shared: Any) -> MotionPermit:
    """Read state and epoch as one indivisible worker/send permit."""
    with shared.motion_lock:
        return _read_motion_permit_locked(shared)


def read_run_state(shared: Any) -> tuple[SafetyState, int, int, int]:
    """Read permission, cancellation epoch, run start and STOP atomically."""
    with shared.motion_lock:
        permit = _read_motion_permit_locked(shared)
        return (permit.state, permit.run_id,
                int(shared.run_started_monotonic_ns.value), int(shared.stop_request.value))


def read_run_end(shared: Any) -> tuple[int, int, RunEndReason]:
    """First terminal fact survives delayed inference and subsequent cleanup."""
    with shared.motion_lock:
        return (int(shared.run_ended_id.value),
                int(shared.run_ended_monotonic_ns.value),
                RunEndReason(int(shared.run_ended_reason.value)))


def _committed_command_is_current_locked(
    shared: Any,
    command: Any,
) -> bool:
    """Return whether *command*'s epoch still owns motion.

    This checks cancellation identity; the SDK fence separately checks health.
    """
    permit = _read_motion_permit_locked(shared)
    return bool(
        permit.allows_motion
        and permit.run_id == int(command.run_id)
    )


def coupled_command_is_current(
    shared: Any,
    *,
    command: Any,
) -> bool:
    """Return whether a committed command's epoch remains unrevoked."""
    with shared.motion_lock:
        return _committed_command_is_current_locked(shared, command)


def command_may_cross_sdk(
    shared: Any, *, run_id: int, required_safety_state: SafetyState | None = None,
) -> bool:
    """Last software admission fence; an admitted SDK call is not retractable."""
    with shared.motion_lock:
        permit = _read_motion_permit_locked(shared)
        return bool(
            permit.allows_motion and permit.run_id == int(run_id)
            and (required_safety_state is None or permit.state is required_safety_state)
            and shared.is_running.value and not shared.error_state.value
            and not shared.estop_request.value
        )


def begin_motion(shared: Any) -> bool:
    """Atomically enter RUNNING and advance the command epoch."""
    with shared.motion_lock:
        epoch = _begin_motion_locked(shared)
    if epoch is None:
        return False
    logger.info(
        "safety: ARMED(%d) → RUNNING(%d), epoch=%d epoch_ns=%d",
        1,
        2,
        epoch[0],
        epoch[1],
    )
    return True


def begin_requested_motion(shared: Any) -> tuple[int, int] | None:
    """Consume one B request and enter RUNNING unless a newer S is pending."""
    with shared.motion_lock:
        if not bool(shared.start_request.value) or int(
            shared.stop_request.value
        ) != int(StopRequest.NONE):
            return None
        epoch = _begin_motion_locked(shared)
        if epoch is None:
            return None
        shared.start_request.value = False
    logger.info(
        "safety: consumed B; ARMED(%d) → RUNNING(%d), epoch=%d epoch_ns=%d",
        1,
        2,
        epoch[0],
        epoch[1],
    )
    return epoch


def request_policy_start(shared: Any, *, require_physical_home: bool) -> bool:
    """Publish B only after the prior S has been acknowledged."""
    if not isinstance(require_physical_home, bool):
        raise TypeError("require_physical_home must be a boolean")
    with shared.motion_lock:
        raw_stop_request = int(shared.stop_request.value)
        if (
            int(shared.safety_state.value) != int(SafetyState.ARMED)
            or bool(shared.pending_record_command_id.value)
            or not shared.is_running.value
            or shared.error_state.value
            or shared.estop_request.value
            or raw_stop_request != int(StopRequest.NONE)
            or (
                require_physical_home and not bool(shared.physical_home_completed.value)
            )
        ):
            return False
        shared.start_request.value = True
        return True


def request_policy_stop(shared: Any, *, reason: RunEndReason = RunEndReason.OPERATOR) -> bool:
    """Publish S and revoke ARMED/RUNNING motion as one ordered operation."""
    with shared.motion_lock:
        already_requested = int(shared.stop_request.value) == int(
            StopRequest.OPERATOR
        ) and not bool(shared.start_request.value)
        shared.start_request.value = False
        shared.physical_home_completed.value = False
        shared.stop_request.value = int(StopRequest.OPERATOR)
        try:
            current = SafetyState(int(shared.safety_state.value))
        except ValueError:
            shared.error_state.value = True
            return False
        if current not in {SafetyState.ARMED, SafetyState.RUNNING}:
            return True
        if current is SafetyState.ARMED and already_requested:
            return True
        revoked = _revoke_motion_locked(shared, SafetyState.ARMED, reason)
    if revoked is None:
        return False
    previous, epoch = revoked
    logger.info(
        "safety: policy S revoked motion %s(%d) → ARMED(%d), epoch=%d",
        previous.name,
        int(previous),
        int(SafetyState.ARMED),
        epoch,
    )
    return True


def revoke_motion_if_run_id(
    shared: Any,
    expected_run_id: int,
    new_state: SafetyState = SafetyState.ARMED,
    *, reason: RunEndReason = RunEndReason.EXECUTOR_BOUNDARY,
) -> bool:
    """Revoke motion only while *expected_run_id* still owns it.

    Lets a bounded parent-side timeout (the run budget) fence a blocking
    policy predict without ever revoking a newer episode whose epoch already
    advanced: the epoch is re-verified inside the same critical section
    as the revocation, so an expired timeout is a no-op.
    """
    with shared.motion_lock:
        if int(shared.run_id.value) != int(expected_run_id):
            return False
        revoked = _revoke_motion_locked(shared, new_state, reason)
    if revoked is None:
        return False
    current, epoch = revoked
    logger.info(
        "safety: epoch-checked revocation %s(%d) → %s(%d), epoch=%d",
        current.name,
        int(current),
        new_state.name,
        int(new_state),
        epoch,
    )
    return True


def revoke_motion(shared: Any, new_state: SafetyState = SafetyState.ARMED, *,
                  reason: RunEndReason = RunEndReason.EXECUTOR_BOUNDARY) -> bool:
    """Atomically invalidate commands and leave the current motion state.

    This is the required path for a normal command pause boundary and fault
    escalation. The short critical section never includes hardware IO.
    """
    if new_state not in (SafetyState.ARMED, SafetyState.DISARMED, SafetyState.FAULT):
        raise ValueError("motion revocation must target ARMED, DISARMED, or FAULT")
    with shared.motion_lock:
        revoked = _revoke_motion_locked(shared, new_state, reason)
    if revoked is None:
        return False
    current, epoch = revoked
    logger.info(
        "safety: revoked motion %s(%d) → %s(%d), epoch=%d",
        current.name,
        int(current),
        new_state.name,
        int(new_state),
        epoch,
    )
    return True


def transition(shared: Any, new_state: SafetyState) -> bool:
    """Execute a fenced safety-state transition."""
    if new_state is SafetyState.RUNNING:
        return begin_motion(shared)
    return revoke_motion(shared, new_state)


def require_transition(shared: Any, new_state: SafetyState) -> None:
    """Perform a transition or raise when the state machine rejects it."""
    if not transition(shared, new_state):
        raise RuntimeError(f"safety transition to {new_state.name} was rejected")
