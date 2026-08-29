"""Validated transitions for the shared runtime safety state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class SafetyState(IntEnum):
    """Values stored in ``RuntimeChannels.safety_state``."""

    DISARMED = 0
    ARMED = 1
    RUNNING = 2
    FAULT = 3


class StopRequest(IntEnum):
    """Reason code carried by the Main-to-coordinator stop request."""

    NONE = 0
    OPERATOR = 1
    RUN_TIME_LIMIT = 2


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


@dataclass(frozen=True)
class RunEpoch:
    """Atomic identity and start time of the current control run."""

    generation: int
    started_monotonic_ns: int


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


def _clear_coupled_command_locked(shared: Any) -> None:
    """Clear the active coupled-command ticket under ``motion_lock``."""
    shared.active_coupled_command_sequence.value = 0


def _invalidate_coupled_commands_locked(shared: Any) -> int:
    """Advance the generation and clear the active ticket under ``motion_lock``."""
    generation = _advance_run_generation_locked(shared)
    _clear_coupled_command_locked(shared)
    return generation


def invalidate_coupled_commands(shared: Any) -> int:
    """Invalidate coupled commands while deliberately retaining lifecycle state.

    This narrow primitive is for an already-established quiescence boundary
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


def read_run_epoch(shared: Any) -> RunEpoch:
    """Read the run generation and observation boundary under one lock."""
    with shared.motion_lock:
        return RunEpoch(
            generation=int(shared.run_generation.value),
            started_monotonic_ns=int(shared.run_started_monotonic_ns.value),
        )


def publish_coupled_command_if_motion_permitted(
    shared: Any,
    *,
    expected_run_generation: int,
    frame: Any,
) -> CoupledCommandTicket | None:
    """Publish one coherent frame and return its unambiguous sequence ticket.

    The ring sequence is recorded under the same lock as the motion permit and
    active sequence. A newer publication atomically supersedes the prior
    ticket, so delayed workers cannot execute an overwritten frame.
    """
    names = getattr(getattr(frame, "dtype", None), "names", None) or ()
    required_fields = {
        "run_generation",
        "action_id",
        "arm_present",
        "hand_present",
    }
    if getattr(frame, "shape", None) != (1,) or not required_fields.issubset(names):
        raise ValueError("coupled command frame is malformed")
    frame_generation = int(frame["run_generation"][0])
    action_id = int(frame["action_id"][0])
    controls_actuator = bool(frame["arm_present"][0]) or bool(frame["hand_present"][0])
    if frame_generation != int(expected_run_generation) or action_id <= 0:
        raise ValueError(
            "coupled command identity does not match its publication permit"
        )
    if not controls_actuator:
        raise ValueError("coupled command must target at least one actuator")
    with shared.motion_lock:
        permit = _read_motion_permit_locked(shared)
        if not permit.allows_motion or permit.run_generation != int(
            expected_run_generation
        ):
            return None
        sequence = int(shared.coupled_cmd_ring.write(frame))
        if sequence <= 0:
            raise RuntimeError("coupled command ring returned an invalid sequence")
        shared.active_coupled_command_sequence.value = sequence
        return CoupledCommandTicket(
            run_generation=permit.run_generation,
            ring_sequence=sequence,
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
        and int(shared.active_coupled_command_sequence.value)
        == int(ticket.ring_sequence)
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
    latest-wins slot and the runtime must not be stopping or faulted.  The
    lock is deliberately released before hardware IO.
    """
    with shared.motion_lock:
        return bool(
            _ticket_is_current_locked(shared, ticket)
            and shared.is_running.value
            and not shared.error_state.value
            and not shared.estop_request.value
        )


def begin_motion(shared: Any) -> bool:
    """Atomically enter RUNNING and advance the command generation."""
    with shared.motion_lock:
        if (
            int(shared.safety_state.value) != int(SafetyState.ARMED)
            or not shared.is_running.value
            or shared.error_state.value
            or shared.estop_request.value
        ):
            return False
        generation = _invalidate_coupled_commands_locked(shared)
        started_ns = time.monotonic_ns()
        shared.run_started_monotonic_ns.value = started_ns
        shared.safety_state.value = int(SafetyState.RUNNING)
    logger.info(
        "safety: ARMED(%d) → RUNNING(%d), generation=%d epoch_ns=%d",
        1,
        2,
        generation,
        started_ns,
    )
    return True


def revoke_motion(shared: Any, new_state: SafetyState = SafetyState.ARMED) -> bool:
    """Atomically invalidate commands and leave the current motion state.

    This is the required path for normal command quiescence and fault
    escalation. The short critical section never includes hardware IO.
    """
    if new_state not in (SafetyState.ARMED, SafetyState.DISARMED, SafetyState.FAULT):
        raise ValueError("motion revocation must target ARMED, DISARMED, or FAULT")
    with shared.motion_lock:
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
            return False
        generation = _invalidate_coupled_commands_locked(shared)
        shared.run_started_monotonic_ns.value = 0
        shared.safety_state.value = int(new_state)
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
