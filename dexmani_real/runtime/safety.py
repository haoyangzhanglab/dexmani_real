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
PUBLISH_REASON_GENERATION = "command generation no longer owns motion"
PUBLISH_REASON_EXPIRED = "command validity window closed"


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
    run_generation: int

    @property
    def allows_motion(self) -> bool:
        return self.state in (SafetyState.ARMED, SafetyState.RUNNING)


@dataclass(frozen=True)
class CoupledCommandTicket:
    """Ownership identity of one coherent record in the coupled ring."""

    run_generation: int
    ring_sequence: int
    valid_until_monotonic_ns: int
    published_monotonic_ns: int = 0


@dataclass(frozen=True)
class RunEpoch:
    """Atomic identity and start time of the current control run."""

    generation: int
    started_monotonic_ns: int


@dataclass(frozen=True)
class RunStateSnapshot:
    """Atomic safety state and observation epoch used by worker loops."""

    state: SafetyState
    generation: int
    started_monotonic_ns: int
    stop_request: int


def _advance_run_generation_locked(shared: Any) -> int:
    shared.run_generation.value = int(shared.run_generation.value) + 1
    return int(shared.run_generation.value)


def _read_motion_permit_locked(shared: Any) -> MotionPermit:
    """Read the permit while the caller owns ``motion_lock``."""
    try:
        state = SafetyState(int(shared.safety_state.value))
    except ValueError:
        state = SafetyState.FAULT
    return MotionPermit(state, int(shared.run_generation.value))


def _begin_motion_locked(shared: Any) -> RunEpoch | None:
    """Enter RUNNING while the caller owns ``motion_lock``."""
    if (
        int(shared.safety_state.value) != int(SafetyState.ARMED)
        or not shared.is_running.value
        or shared.error_state.value
        or shared.estop_request.value
    ):
        return None
    generation = _invalidate_coupled_commands_locked(shared)
    started_ns = time.monotonic_ns()
    shared.run_started_monotonic_ns.value = started_ns
    shared.safety_state.value = int(SafetyState.RUNNING)
    return RunEpoch(generation=generation, started_monotonic_ns=started_ns)


def _revoke_motion_locked(
    shared: Any, new_state: SafetyState
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
    generation = _invalidate_coupled_commands_locked(shared)
    shared.run_started_monotonic_ns.value = 0
    shared.safety_state.value = int(new_state)
    return current, generation


def _invalidate_coupled_commands_locked(shared: Any) -> int:
    """Advance the generation to invalidate every prior coupled command."""
    return _advance_run_generation_locked(shared)


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
    ticket: CoupledCommandTicket,
) -> bool:
    """Invalidate *ticket* only while it still owns the command slot.

    A timed-out caller must not revoke a newer publisher's command merely
    because the newer record has not reached actuator acknowledgement yet.
    """
    with shared.motion_lock:
        if not _ticket_is_current_locked(shared, ticket):
            return False
        _invalidate_coupled_commands_locked(shared)
        return True


def read_motion_permit(shared: Any) -> MotionPermit:
    """Read state and generation as one indivisible worker/send permit."""
    with shared.motion_lock:
        return _read_motion_permit_locked(shared)


def read_run_state_snapshot(shared: Any) -> RunStateSnapshot:
    """Read the run boundary state under one motion lock."""
    with shared.motion_lock:
        permit = _read_motion_permit_locked(shared)
        return RunStateSnapshot(
            state=permit.state,
            generation=permit.run_generation,
            started_monotonic_ns=int(shared.run_started_monotonic_ns.value),
            stop_request=int(shared.stop_request.value),
        )


def publish_coupled_command_if_motion_permitted(
    shared: Any,
    *,
    expected_run_generation: int,
    frame: Any,
    required_state: SafetyState | None = None,
    minimum_delivery_window_ns: int = 0,
) -> tuple[CoupledCommandTicket | None, str]:
    """Publish one frame or return its exact locked rejection reason.

    The ring sequence is recorded under the same lock as the motion permit.
    A newer publication atomically supersedes the prior ticket, so delayed
    workers cannot execute an overwritten frame.
    """
    if (
        isinstance(minimum_delivery_window_ns, bool)
        or not isinstance(minimum_delivery_window_ns, int)
        or minimum_delivery_window_ns < 0
    ):
        raise ValueError("minimum_delivery_window_ns must be a non-negative integer")
    names = getattr(getattr(frame, "dtype", None), "names", None) or ()
    required_fields = {
        "run_generation",
        "action_id",
        "arm_present",
        "hand_present",
        "valid_until_monotonic_ns",
    }
    if getattr(frame, "shape", None) != (1,) or not required_fields.issubset(names):
        raise ValueError("coupled command frame is malformed")
    frame_generation = int(frame["run_generation"][0])
    action_id = int(frame["action_id"][0])
    valid_until_monotonic_ns = int(frame["valid_until_monotonic_ns"][0])
    controls_actuator = bool(frame["arm_present"][0]) or bool(frame["hand_present"][0])
    if frame_generation != int(expected_run_generation) or action_id <= 0:
        raise ValueError(
            "coupled command identity does not match its publication permit"
        )
    if not controls_actuator:
        raise ValueError("coupled command must target at least one actuator")
    with shared.motion_lock:
        if bool(shared.estop_request.value):
            return None, PUBLISH_REASON_ESTOP
        if bool(shared.error_state.value):
            return None, PUBLISH_REASON_FAULT
        if not bool(shared.is_running.value):
            return None, PUBLISH_REASON_RUNTIME_STOPPED
        permit = _read_motion_permit_locked(shared)
        if not permit.allows_motion:
            return None, f"{PUBLISH_REASON_SAFETY_STATE}: {permit.state.name}"
        if required_state is not None and permit.state is not required_state:
            return (
                None,
                f"{PUBLISH_REASON_SAFETY_STATE}: expected {required_state.name}, "
                f"got {permit.state.name}",
            )
        if permit.run_generation != int(expected_run_generation):
            return None, PUBLISH_REASON_GENERATION
        if valid_until_monotonic_ns - time.monotonic_ns() <= minimum_delivery_window_ns:
            return None, PUBLISH_REASON_EXPIRED
        sequence = int(shared.coupled_cmd_ring.write(frame))
        if sequence <= 0:
            raise RuntimeError("coupled command ring returned an invalid sequence")
        published_monotonic_ns = time.monotonic_ns()
        return (
            CoupledCommandTicket(
                run_generation=permit.run_generation,
                ring_sequence=sequence,
                valid_until_monotonic_ns=valid_until_monotonic_ns,
                published_monotonic_ns=published_monotonic_ns,
            ),
            "",
        )


def _ticket_is_current_locked(
    shared: Any,
    ticket: CoupledCommandTicket,
) -> bool:
    """Return whether *ticket* still owns the active command slot."""
    permit = _read_motion_permit_locked(shared)
    return bool(
        permit.allows_motion
        and permit.run_generation == int(ticket.run_generation)
        and int(shared.coupled_cmd_ring.latest_sequence) == int(ticket.ring_sequence)
    )


def coupled_command_ticket_is_current(
    shared: Any,
    *,
    ticket: CoupledCommandTicket,
) -> bool:
    """Return whether a published ticket remains latest and unrevoked."""
    with shared.motion_lock:
        return _ticket_is_current_locked(shared, ticket)


def coupled_command_ticket_allows_execution(
    shared: Any,
    *,
    ticket: CoupledCommandTicket,
) -> bool:
    """Return whether *ticket* may still cross an actuator SDK boundary.

    This is the common final worker check: the record must still own the
    latest-wins slot, remain inside its delivery window, and the runtime must
    not be stopping or faulted. The lock is deliberately released before
    hardware IO, so workers call this immediately before their SDK method.
    """
    with shared.motion_lock:
        return bool(
            _ticket_is_current_locked(shared, ticket)
            and shared.is_running.value
            and not shared.error_state.value
            and not shared.estop_request.value
            and time.monotonic_ns() < int(ticket.valid_until_monotonic_ns)
        )


def begin_motion(shared: Any) -> bool:
    """Atomically enter RUNNING and advance the command generation."""
    with shared.motion_lock:
        epoch = _begin_motion_locked(shared)
    if epoch is None:
        return False
    logger.info(
        "safety: ARMED(%d) → RUNNING(%d), generation=%d epoch_ns=%d",
        1,
        2,
        epoch.generation,
        epoch.started_monotonic_ns,
    )
    return True


def begin_requested_motion(shared: Any) -> RunEpoch | None:
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
        "safety: consumed B; ARMED(%d) → RUNNING(%d), generation=%d epoch_ns=%d",
        1,
        2,
        epoch.generation,
        epoch.started_monotonic_ns,
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


def request_policy_stop(shared: Any) -> bool:
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
        revoked = _revoke_motion_locked(shared, SafetyState.ARMED)
    if revoked is None:
        return False
    previous, generation = revoked
    logger.info(
        "safety: policy S revoked motion %s(%d) → ARMED(%d), generation=%d",
        previous.name,
        int(previous),
        int(SafetyState.ARMED),
        generation,
    )
    return True


def revoke_motion(shared: Any, new_state: SafetyState = SafetyState.ARMED) -> bool:
    """Atomically invalidate commands and leave the current motion state.

    This is the required path for a normal command pause boundary and fault
    escalation. The short critical section never includes hardware IO.
    """
    if new_state not in (SafetyState.ARMED, SafetyState.DISARMED, SafetyState.FAULT):
        raise ValueError("motion revocation must target ARMED, DISARMED, or FAULT")
    with shared.motion_lock:
        revoked = _revoke_motion_locked(shared, new_state)
    if revoked is None:
        return False
    current, generation = revoked
    logger.info(
        "safety: revoked motion %s(%d) → %s(%d), generation=%d",
        current.name,
        int(current),
        new_state.name,
        int(new_state),
        generation,
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
