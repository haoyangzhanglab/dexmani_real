"""Exact fail-closed hand-home publication and acknowledgement sequence."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dexmani_real.config.experiment import ExperimentConfig

import numpy as np

from dexmani_real.control.publication import (
    PUBLISH_REASON_FIFO_FULL,
    PublishWaitTracker,
    build_action_candidate,
    motion_rejection_reason,
    publish_command,
    read_hand_feedback,
    wait_command_accepted,
)
from dexmani_real.runtime.safety import SafetyState
from dexmani_real.utils.limits import validate_hand_command_bounds
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

__all__ = ["publish_hand_home_and_wait_accepted", "initialize_hand_home"]

_FULL_RETRY_POLL_S = 0.01


def initialize_hand_home(
    shared: Any,
    runtime: ExperimentConfig,
    *,
    heartbeat: bool = False,
    abort_requested: Any = None,
) -> bool:
    """Restore the configured hand pose before task commands or recording begin."""
    if not runtime.policy.hand_enabled:
        return True
    if abort_requested is not None and abort_requested():
        return False
    hand = runtime.hand
    return publish_hand_home_and_wait_accepted(
        shared,
        np.deg2rad(np.asarray(hand.home_qpos_deg, dtype=np.float64)),
        command_lower_rad=np.asarray(hand.qpos_min_rad, dtype=np.float64),
        command_upper_rad=np.asarray(hand.qpos_max_rad, dtype=np.float64),
        mechanical_lower_rad=np.asarray(hand.mechanical_qpos_min_rad, dtype=np.float64),
        mechanical_upper_rad=np.asarray(hand.mechanical_qpos_max_rad, dtype=np.float64),
        hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
        timeout_s=hand.home_command_ack_timeout_s,
        heartbeat=heartbeat,
        abort_requested=abort_requested,
    )


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
    candidate = build_action_candidate(shared, None, target)
    # FULL is recoverable backpressure: retry the identical home candidate
    # inside the original operation deadline; abort and the runtime gates
    # keep priority over the retry.
    fifo_wait = PublishWaitTracker("hand_home")
    publish_result = publish_command(
        shared,
        candidate,
        required_safety_state=SafetyState.ARMED,
    )
    while (
        not publish_result.published
        and publish_result.reason == PUBLISH_REASON_FIFO_FULL
        and time.monotonic() < deadline_s
    ):
        fifo_wait.note_full(publish_result.fifo_depth, candidate.action_id)
        if abort_requested is not None and abort_requested():
            break
        time.sleep(_FULL_RETRY_POLL_S)
        publish_result = publish_command(
            shared,
            candidate,
            required_safety_state=SafetyState.ARMED,
        )
    if not publish_result.published:
        fifo_wait.note_dropped("hand home publish stopped")
        logger.warning("hand home publish failed: %s", publish_result.reason)
        return False
    fifo_wait.note_committed()
    if publish_result.command is None:
        logger.error("hand home published without a committed-command receipt")
        return False

    acceptance = wait_command_accepted(
        shared,
        command=publish_result.command,
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
