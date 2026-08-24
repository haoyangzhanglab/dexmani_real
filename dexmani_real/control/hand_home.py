"""Exact fail-closed hand-home publication and acknowledgement sequence."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.control.publication import (
    check_runtime_gate,
    read_hand_feedback,
    validate_hand_command_bounds,
)
from dexmani_real.ipc.schema import HAND_COMMAND_DTYPE
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
    and the rated mechanical box. Success means every SDK send, including the
    exact final home endpoint, was acknowledged. Measured qpos is deliberately
    not compared with the target because contact and steady-state position
    error are valid.
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
    command_lower = np.asarray(command_lower_rad, dtype=np.float64)
    command_upper = np.asarray(command_upper_rad, dtype=np.float64)
    deadline_s = time.monotonic() + timeout_s
    hand_feedback, feedback_rejection = read_hand_feedback(
        shared, None, hand_feedback_max_age_s=hand_feedback_max_age_s
    )
    if feedback_rejection is not None:
        logger.warning("hand home rejected: %s", feedback_rejection.reason)
        return False
    assert hand_feedback is not None
    start = hand_feedback.last_cmd_qpos
    if np.any(start < command_lower - 1e-12) or np.any(start > command_upper + 1e-12):
        logger.warning(
            "hand home rejected: last accepted hand command violates operational limits"
        )
        return False

    # Publish the exact home endpoint as one command.
    milestone_count = 1

    last_action_id = 0
    acknowledged = False
    for milestone_index in range(1, milestone_count + 1):
        if time.monotonic() >= deadline_s:
            break
        runtime_rejection = check_runtime_gate(
            shared,
            check_is_running=check_is_running,
        )
        if runtime_rejection is not None:
            logger.warning(
                "hand home stopped by runtime gate: %s", runtime_rejection.reason
            )
            return False
        milestone = target.copy()

        with shared.arm_command_seq.get_lock():
            action_id = int(shared.arm_command_seq.value) + 1
            shared.arm_command_seq.value = action_id
        last_action_id = action_id
        now_ns = time.monotonic_ns()
        frame = np.zeros(1, dtype=HAND_COMMAND_DTYPE)
        frame["run_generation"][0] = int(shared.run_generation.value)
        frame["observation_id"][0] = action_id
        frame["action_id"][0] = action_id
        frame["created_monotonic_ns"][0] = now_ns
        frame["target_monotonic_ns"][0] = now_ns
        frame["valid_until_monotonic_ns"][0] = now_ns + int(
            max(0.3, deadline_s - time.monotonic() + 0.1) * 1e9
        )
        frame["is_hold"][0] = 0
        frame["qpos_cmd"][0] = milestone
        shared.hand_cmd_ring.write(frame)

        acknowledged = False
        while time.monotonic() < deadline_s:
            if abort_requested is not None and abort_requested():
                return False
            if (
                check_runtime_gate(
                    shared,
                    check_is_running=check_is_running,
                )
                is not None
            ):
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
                return False
            assert hand_feedback is not None
            applied_id = hand_feedback.last_cmd_seq
            if applied_id > action_id:
                logger.warning(
                    "hand home action_id=%d was superseded by action_id=%d before acknowledgement",
                    action_id,
                    applied_id,
                )
                return False
            if applied_id == action_id:
                acknowledged = True
                break
            time.sleep(0.01)
        if not acknowledged:
            break

    if last_action_id and milestone_index == milestone_count and acknowledged:
        if verbose:
            print(
                f"  hand: home command accepted (action_id={last_action_id})",
                flush=True,
            )
        return True

    logger.warning(
        "hand home action_id=%d was not acknowledged within %.3fs",
        last_action_id,
        timeout_s,
    )
    return False
