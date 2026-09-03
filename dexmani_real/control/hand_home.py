"""Exact fail-closed hand-home publication and acknowledgement sequence."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.control.publication import (
    build_action_candidate,
    check_runtime_gate,
    read_hand_feedback,
    send_command,
    validate_hand_command_bounds,
)
from dexmani_real.runtime.safety import (
    cancel_coupled_command_if_current,
    coupled_command_ticket_is_current,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

__all__ = ["publish_hand_home_and_wait_applied"]


def publish_hand_home_and_wait_applied(
    shared: Any,
    home_qpos: np.ndarray,
    *,
    command_lower_rad: np.ndarray,
    command_upper_rad: np.ndarray,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
    hand_feedback_max_age_s: float,
    timeout_s: float = 1.0,
    heartbeat: bool = False,
    check_is_running: bool = True,
    verbose: bool = True,
    abort_requested: Any = None,
) -> bool:
    """Publish exact hand-home and wait only for worker/SDK acceptance.

    The configured endpoint must lie inside both the operational command box
    and the rated mechanical box. Success means the worker accepted the exact
    home endpoint. Measured qpos must be healthy and fresh, but it is neither
    required to lie inside command bounds nor compared with the target because
    encoder zero offsets, contact, and steady-state position error are valid.
    """
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(
            "hand home command acknowledgement timeout must be finite and positive"
        )
    # Reject bound violations; never clip coupled hand commands here.
    target = validate_hand_command_bounds(
        home_qpos,
        command_lower_rad,
        command_upper_rad,
        mechanical_lower_rad,
        mechanical_upper_rad,
    )
    runtime_rejection = check_runtime_gate(
        shared,
        check_is_running=check_is_running,
    )
    if runtime_rejection is not None:
        logger.warning(
            "hand home rejected by runtime gate: %s", runtime_rejection.reason
        )
        return False
    deadline_s = time.monotonic() + timeout_s
    hand_feedback, feedback_rejection = read_hand_feedback(
        shared, None, hand_feedback_max_age_s=hand_feedback_max_age_s
    )
    if feedback_rejection is not None:
        logger.warning("hand home rejected: %s", feedback_rejection.reason)
        return False
    assert hand_feedback is not None
    # Feedback must be healthy, but its measured angle is not an outgoing
    # command.  Encoder zero offsets must not block a legal home target.

    runtime_rejection = check_runtime_gate(shared, check_is_running=check_is_running)
    if runtime_rejection is not None:
        logger.warning(
            "hand home stopped by runtime gate: %s", runtime_rejection.reason
        )
        return False
    candidate = build_action_candidate(
        shared,
        None,
        target,
        observation_id=None,
        action_validity_s=max(0.3, deadline_s - time.monotonic() + 0.1),
    )
    if candidate is None:
        logger.warning("hand home rejected: command delivery window is closed")
        return False
    publish_result = send_command(
        shared,
        candidate,
        check_is_running=check_is_running,
    )
    if not publish_result.succeeded:
        logger.warning("hand home publish failed: %s", publish_result.reason)
        return False
    if publish_result.ticket is None:
        logger.error("hand home published without a coupled-command ticket")
        return False

    action_id = int(candidate.action_id)
    ticket = publish_result.ticket

    while time.monotonic() < deadline_s:
        if abort_requested is not None and abort_requested():
            cancel_coupled_command_if_current(shared, ticket=ticket)
            return False
        if check_runtime_gate(shared, check_is_running=check_is_running) is not None:
            cancel_coupled_command_if_current(shared, ticket=ticket)
            return False
        if heartbeat:
            shared.set_heartbeat("policy", time.monotonic())

        hand_feedback, feedback_rejection = read_hand_feedback(
            shared, None, hand_feedback_max_age_s=hand_feedback_max_age_s
        )
        if feedback_rejection is not None:
            logger.warning(
                "hand home acknowledgement stopped: %s", feedback_rejection.reason
            )
            cancel_coupled_command_if_current(shared, ticket=ticket)
            return False
        assert hand_feedback is not None
        accepted_action_id = hand_feedback.accepted_target_action_id
        if accepted_action_id > action_id:
            logger.warning(
                "hand home action_id=%d was superseded by action_id=%d before acknowledgement",
                action_id,
                accepted_action_id,
            )
            cancel_coupled_command_if_current(shared, ticket=ticket)
            return False
        if accepted_action_id == action_id:
            if verbose:
                print(
                    f"  hand: home command accepted (action_id={action_id})", flush=True
                )
            return True
        if not coupled_command_ticket_is_current(shared, ticket=ticket):
            logger.warning(
                "hand home action_id=%d lost command ownership before acknowledgement",
                action_id,
            )
            return False
        time.sleep(0.01)

    logger.warning(
        "hand home action_id=%d was not acknowledged within %.3fs",
        action_id,
        timeout_s,
    )
    cancel_coupled_command_if_current(shared, ticket=ticket)
    return False
