"""Teleoperation feedback-health predicates."""

from __future__ import annotations

import time

import numpy as np

from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.utils.feedback import validate_arm_feedback, validate_hand_feedback


def arm_feedback_issue(
    state: np.ndarray | None,
    *,
    now_monotonic_ns: int,
    max_age_s: float,
) -> str | None:
    """Return why an arm state is unsafe to consume, including controller faults."""
    if state is None:
        return "arm feedback unavailable"
    issue = validate_arm_feedback(
        connected=bool(state["connected"][0]),
        error_code=int(state["error_code"][0]),
        state_valid=bool(state["state_valid"][0]),
        source_monotonic_ns=int(state["source_monotonic_ns"][0]),
        now_monotonic_ns=int(now_monotonic_ns),
        max_age_s=max_age_s,
        qpos=np.asarray(state["qpos"][0]),
        qvel=np.asarray(state["qvel"][0]),
    )
    if issue is not None:
        return issue
    return None


def advance_arm_feedback_error_count(
    current_count: int,
    issue: str | None,
    *,
    max_consecutive_errors: int,
) -> tuple[int, bool]:
    """Reset on valid feedback; fault at the configured invalid-frame limit."""
    if issue is None:
        return 0, False
    next_count = current_count + 1
    return next_count, next_count >= max_consecutive_errors


def hand_feedback_issue(
    cfg: TeleopConfig,
    state: np.ndarray | None,
) -> str | None:
    if not cfg.runtime.policy.hand_enabled:
        return None
    if state is None:
        return "hand feedback unavailable"
    return validate_hand_feedback(
        connected=bool(state["connected"][0]),
        state_valid=bool(state["state_valid"][0]),
        source_monotonic_ns=int(state["source_monotonic_ns"][0]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=float(cfg.runtime.safety.heartbeat_timeouts["hand"]),
        qpos=np.asarray(state["qpos"][0]),
    )
