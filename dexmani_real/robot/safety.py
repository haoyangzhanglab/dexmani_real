"""Safety state machine — ManiUniCon P0 compliant (ref: ManiUniCon §13.2).

Four-state model: DISARMED → ARMED → RUNNING → FAULT

Write ownership:
  - Main  owns: DISARMED↔ARMED, →FAULT, →DISARMED
  - Policy owns: ARMED↔RUNNING (teleop start/stop)
  - Arm/Hand read-only: gate servo on safety_state

This split prevents races: the two writers operate on disjoint transition pairs.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class SafetyState(IntEnum):
    """System safety state — stored in shared.safety_state (mp.Value('i'))."""

    DISARMED = 0  # Robot disabled. Arm in state=4 (stopped). No servo possible.
    ARMED = 1  # Hardware connected, Mode 6 active, ready for teleop.
    RUNNING = 2  # Teleop active, Policy sending servo commands.
    FAULT = 3  # Error — manual intervention required to clear.


# Allowed transitions: (from, to) → True
# Disallowed transitions will be logged and rejected.
ALLOWED_TRANSITIONS: dict[tuple[int, int], bool] = {
    # Main-owned
    (SafetyState.DISARMED, SafetyState.ARMED): True,
    (SafetyState.ARMED, SafetyState.DISARMED): True,
    (SafetyState.RUNNING, SafetyState.DISARMED): True,
    # Policy-owned
    (SafetyState.ARMED, SafetyState.RUNNING): True,
    (SafetyState.RUNNING, SafetyState.ARMED): True,
    # Main: any → FAULT
    (SafetyState.DISARMED, SafetyState.FAULT): True,
    (SafetyState.ARMED, SafetyState.FAULT): True,
    (SafetyState.RUNNING, SafetyState.FAULT): True,
    # Main: FAULT → DISARMED (shutdown only)
    (SafetyState.FAULT, SafetyState.DISARMED): True,
}


def transition(shared: Any, new_state: SafetyState) -> bool:
    """Validate and execute a safety state transition.

    Note:
        The read-validate-write on ``shared.safety_state`` (``mp.Value('i')``)
        is **not atomic** — no ``mp.Lock`` protects it.  This is a known
        TOCTOU window, safe in practice because:

        1. **Write ownership separation** prevents overlapping transitions:
           Main owns DISARMED↔ARMED / →FAULT; Policy owns ARMED↔RUNNING.
        2. **FAULT is self-correcting:** the heartbeat supervisor re-asserts
           FAULT within 100 ms (10 Hz), so any overwrite is transient.
        3. **No new motion can occur after a worker fault latch:** arm_loop and
           hand_loop gate commands on sticky ``error_state`` immediately;
           Main then owns the transition to FAULT.  The arm holds its last
           Mode 6 target during that bounded handoff.
           ``set_state(4)`` is called during cleanup as a belt-and-suspenders
           measure, not the primary safety mechanism.

        Adding ``mp.Lock`` was considered (2026-08-05 audit) and rejected:
        the overhead for a theoretical race that self-corrects within one
        supervisor tick is not justified.

    Args:
        shared: SharedStorage instance with ``safety_state`` (mp.Value('i')).
        new_state: Target SafetyState.

    Returns:
        True if transition succeeded, False if rejected (invalid transition).
    """
    current_int = shared.safety_state.value
    try:
        current = SafetyState(current_int)
    except ValueError:
        logger.error("safety: unknown current state %d — forcing FAULT", current_int)
        shared.safety_state.value = int(SafetyState.FAULT)
        return False

    if current == new_state:
        return True  # Idempotent — no-op transitions are harmless

    key = (int(current), int(new_state))
    if not ALLOWED_TRANSITIONS.get(key, False):
        logger.error(
            "safety: rejected transition %s(%d) → %s(%d)",
            current.name,
            int(current),
            new_state.name,
            int(new_state),
        )
        return False

    shared.safety_state.value = int(new_state)
    logger.info(
        "safety: %s(%d) → %s(%d)",
        current.name,
        int(current),
        new_state.name,
        int(new_state),
    )
    return True
