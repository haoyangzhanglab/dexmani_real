"""Validated transitions for the shared runtime safety state."""

from __future__ import annotations

from contextlib import nullcontext
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


def advance_run_generation(shared: Any) -> int:
    """Invalidate robot commands prepared before the current control run."""
    lock_getter = getattr(shared.run_generation, "get_lock", None)
    lock = lock_getter() if callable(lock_getter) else nullcontext()
    with lock:
        shared.run_generation.value = int(shared.run_generation.value) + 1
        return int(shared.run_generation.value)


def transition(shared: Any, new_state: SafetyState) -> bool:
    """Validate and execute a safety state transition.

    The synchronized ``mp.Value`` lock covers read/validate/write as one
    operation.  Write ownership remains split between Main and Policy, while
    the lock prevents a concurrent FAULT transition from being overwritten.

    Args:
        shared: RuntimeChannels instance with ``safety_state`` (mp.Value('i')).
        new_state: Target SafetyState.

    Returns:
        Whether the transition succeeded.
    """
    lock_getter = getattr(shared.safety_state, "get_lock", None)
    lock = lock_getter() if callable(lock_getter) else nullcontext()
    with lock:
        current_int = shared.safety_state.value
        try:
            current = SafetyState(current_int)
        except ValueError:
            logger.error(
                "safety: unknown current state %d — forcing FAULT", current_int
            )
            shared.safety_state.value = int(SafetyState.FAULT)
            return False

        if current == new_state:
            return True

        if (current, new_state) not in _ALLOWED_TRANSITIONS:
            logger.error(
                "safety: rejected transition %s(%d) → %s(%d)",
                current.name,
                int(current),
                new_state.name,
                int(new_state),
            )
            return False

        if new_state is SafetyState.ARMED and current is SafetyState.DISARMED:
            # Ignore endpoints published before the latest motion-arm sequence.
            armed_at_seq = getattr(shared, "arm_armed_at_seq", None)
            command_seq = getattr(shared, "arm_command_seq", None)
            if armed_at_seq is not None and command_seq is not None:
                armed_at_seq.value = int(command_seq.value)

        shared.safety_state.value = int(new_state)
    logger.info(
        "safety: %s(%d) → %s(%d)",
        current.name,
        int(current),
        new_state.name,
        int(new_state),
    )
    return True


def require_transition(shared: Any, new_state: SafetyState) -> None:
    """Perform a transition or raise when the state machine rejects it."""
    if not transition(shared, new_state):
        raise RuntimeError(f"safety transition to {new_state.name} was rejected")
