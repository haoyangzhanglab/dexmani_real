"""Exact fail-closed hand-home publication and acknowledgement sequence."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.control.publication import (
    build_action_candidate,
    motion_rejection_reason,
    publish_command,
    read_hand_feedback,
    validate_hand_command_bounds,
    wait_command_accepted,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

__all__ = ["publish_hand_home_and_wait_accepted"]


def publish_hand_home_and_wait_accepted(
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
    runtime_rejection = motion_rejection_reason(
        shared,
        check_is_running=check_is_running,
    )
    if runtime_rejection:
        logger.warning("hand home rejected by runtime gate: %s", runtime_rejection)
        return False
    deadline_s = time.monotonic() + timeout_s
    hand_feedback, feedback_rejection, _ = read_hand_feedback(
        shared, max_age_s=hand_feedback_max_age_s
    )
    if hand_feedback is None:
        logger.warning("hand home rejected: %s", feedback_rejection)
        return False
    # Feedback must be healthy, but its measured angle is not an outgoing
    # command.  Encoder zero offsets must not block a legal home target.

    runtime_rejection = motion_rejection_reason(
        shared, check_is_running=check_is_running
    )
    if runtime_rejection:
        logger.warning("hand home stopped by runtime gate: %s", runtime_rejection)
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
    publish_result = publish_command(
        shared,
        candidate,
    )
    if not publish_result.published:
        logger.warning("hand home publish failed: %s", publish_result.reason)
        return False
    if publish_result.ticket is None:
        logger.error("hand home published without a coupled-command ticket")
        return False

    acceptance = wait_command_accepted(
        shared,
        ticket=publish_result.ticket,
        action_id=int(candidate.action_id),
        wait_for_arm=False,
        wait_for_hand=True,
        timeout_s=max(1e-6, deadline_s - time.monotonic()),
        arm_feedback_max_age_s=hand_feedback_max_age_s,
        hand_feedback_max_age_s=hand_feedback_max_age_s,
        check_is_running=check_is_running,
        abort_requested=abort_requested,
        heartbeat=(
            (lambda: shared.set_heartbeat("policy", time.monotonic()))
            if heartbeat
            else None
        ),
    )
    if not acceptance.accepted:
        logger.warning("hand home acknowledgement stopped: %s", acceptance.reason)
        return False
    if verbose:
        print(
            f"  hand: home command accepted (action_id={candidate.action_id})",
            flush=True,
        )
    return True
