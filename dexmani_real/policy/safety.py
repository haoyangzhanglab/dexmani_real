"""Compatibility exports and the exact hand-home publication sequence.

New control code uses :mod:`dexmani_real.policy.safety_gate` for candidate
validation, :mod:`dexmani_real.policy.command_publication` for controller-side
runtime/feedback checks and IPC publication, and
:mod:`dexmani_real.robot.command_validation` at the worker boundary.  This
module preserves established imports while owning only the hand-home helper.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.policy.command_publication import (
    CommandPublishResult,
    CommandPublishStatus,
    _hand_feedback_snapshot,
    _publication_runtime_gate,
    build_action_candidate,
    publish_joint_targets,
    send_command,
    validate_and_send_candidate,
)
from dexmani_real.policy.safety_gate import (
    GateRejectCode,
    GateResult,
    SafetyGate,
    planner_action_safety_gate,
)
from dexmani_real.robot.command_validation import (
    worker_validate_arm,
    worker_validate_hand,
)
from dexmani_real.robot.safety import advance_run_generation
from dexmani_real.utils.limits import (
    validate_hand_command_bounds as _validate_hand_bounds,
)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import HAND_COMMAND_DTYPE

logger = get_logger(__name__)

__all__ = [
    "CommandPublishResult",
    "CommandPublishStatus",
    "GateRejectCode",
    "GateResult",
    "SafetyGate",
    "advance_run_generation",
    "build_action_candidate",
    "planner_action_safety_gate",
    "publish_hand_home_and_wait_applied",
    "publish_joint_targets",
    "send_command",
    "validate_and_send_candidate",
    "validate_hand_command_bounds",
    "worker_validate_arm",
    "worker_validate_hand",
]


def validate_hand_command_bounds(
    hand_cmd: np.ndarray,
    operational_lower: np.ndarray,
    operational_upper: np.ndarray,
    mechanical_lower: np.ndarray,
    mechanical_upper: np.ndarray,
) -> np.ndarray:
    """Validate one hand target against operational + rated mechanical bounds;
    reject-whole, never clip.

    Shared preflight for every coupled hand path (teleop, replay, return-home).
    Normal action producers reach it through ``validate_and_send_candidate``;
    hand-home also reuses it before publishing the exact home endpoint.  Raises
    ``ValueError`` on any violation and returns a copy otherwise.
    """
    rated_lower = np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
    rated_upper = np.asarray(hand_defaults.mechanical_qpos_max_rad, dtype=np.float64)
    return _validate_hand_bounds(
        hand_cmd,
        operational_lower,
        operational_upper,
        mechanical_lower,
        mechanical_upper,
        rated_lower,
        rated_upper,
    )


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
    runtime_rejection = _publication_runtime_gate(
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
    hand_feedback, feedback_rejection = _hand_feedback_snapshot(
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
        runtime_rejection = _publication_runtime_gate(
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
                _publication_runtime_gate(
                    shared,
                    check_is_running=check_is_running,
                )
                is not None
            ):
                return False
            if heartbeat:
                shared.set_heartbeat("policy", time.monotonic())

            hand_feedback, feedback_rejection = _hand_feedback_snapshot(
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
