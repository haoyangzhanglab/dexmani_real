"""Pure feedback-health predicates shared by teleop, replay, and deployment.

These are schema-shape and freshness checks over fixed NumPy state dtypes —
not vendor I/O and not policy disposition.  ``robot/`` keeps doing device I/O;
control and its callers keep owning what a rejection *means*.  Both arm and hand
predicates live here (rather than split across two packages) so the pair of
fail-closed predicates has a single home.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.robot_spec import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

__all__ = [
    "FeedbackIssue",
    "FeedbackIssueCode",
    "diagnose_arm_feedback",
    "diagnose_hand_feedback",
    "validate_arm_feedback",
    "validate_hand_feedback",
]


class FeedbackIssueCode(str, Enum):
    """Machine-readable feedback-health failure categories."""

    DISCONNECTED = "disconnected"
    CONTROLLER_ERROR = "controller_error"
    STATE_INVALID = "state_invalid"
    MISSING_TIMESTAMP = "missing_timestamp"
    FUTURE_TIMESTAMP = "future_timestamp"
    STALE = "stale"
    MALFORMED_SHAPE = "malformed_shape"
    NONFINITE = "nonfinite"


@dataclass(frozen=True)
class FeedbackIssue:
    """Typed feedback failure while retaining the existing diagnostic text."""

    code: FeedbackIssueCode
    detail: str

    def __str__(self) -> str:
        return self.detail


def diagnose_arm_feedback(
    *,
    connected: bool,
    error_code: int,
    state_valid: bool,
    source_monotonic_ns: int,
    now_monotonic_ns: int,
    max_age_s: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> FeedbackIssue | None:
    """Return the typed arm feedback failure, or ``None`` when healthy."""
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    fields = {
        "qpos": (np.asarray(qpos), ARM_JOINT_SHAPE),
        "qvel": (np.asarray(qvel), ARM_JOINT_SHAPE),
    }
    for name, (value, expected_shape) in fields.items():
        if value.shape != expected_shape:
            return FeedbackIssue(
                FeedbackIssueCode.MALFORMED_SHAPE,
                f"arm {name} has shape {value.shape}, expected {expected_shape}",
            )
        if not np.all(np.isfinite(value)):
            return FeedbackIssue(
                FeedbackIssueCode.NONFINITE,
                f"arm {name} is non-finite",
            )
    if not connected:
        return FeedbackIssue(FeedbackIssueCode.DISCONNECTED, "arm disconnected")
    if int(error_code) != 0:
        return FeedbackIssue(
            FeedbackIssueCode.CONTROLLER_ERROR,
            f"arm controller error C{int(error_code)}",
        )
    if not state_valid:
        return FeedbackIssue(
            FeedbackIssueCode.STATE_INVALID, "arm state marked invalid"
        )
    if source_monotonic_ns <= 0:
        return FeedbackIssue(
            FeedbackIssueCode.MISSING_TIMESTAMP,
            "arm state has no source timestamp",
        )
    age_s = (now_monotonic_ns - source_monotonic_ns) * 1e-9
    if age_s < 0.0:
        return FeedbackIssue(
            FeedbackIssueCode.FUTURE_TIMESTAMP,
            f"arm state timestamp is {abs(age_s):.3f}s in the future",
        )
    if age_s > max_age_s:
        return FeedbackIssue(FeedbackIssueCode.STALE, f"arm state stale ({age_s:.2f}s)")
    return None


def diagnose_hand_feedback(
    *,
    connected: bool,
    state_valid: bool,
    source_monotonic_ns: int,
    now_monotonic_ns: int,
    max_age_s: float,
    qpos: np.ndarray,
) -> FeedbackIssue | None:
    """Return the typed hand feedback failure, or ``None`` when healthy."""
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    value = np.asarray(qpos)
    if value.shape != HAND_JOINT_SHAPE:
        return FeedbackIssue(
            FeedbackIssueCode.MALFORMED_SHAPE,
            f"hand qpos has shape {value.shape}, expected {HAND_JOINT_SHAPE}",
        )
    if not np.all(np.isfinite(value)):
        return FeedbackIssue(FeedbackIssueCode.NONFINITE, "hand qpos is non-finite")
    if not connected:
        return FeedbackIssue(FeedbackIssueCode.DISCONNECTED, "hand disconnected")
    if not state_valid:
        return FeedbackIssue(
            FeedbackIssueCode.STATE_INVALID, "hand state marked invalid"
        )
    if source_monotonic_ns <= 0:
        return FeedbackIssue(
            FeedbackIssueCode.MISSING_TIMESTAMP,
            "hand state has no source timestamp",
        )
    age_s = (now_monotonic_ns - source_monotonic_ns) * 1e-9
    if age_s < 0.0:
        return FeedbackIssue(
            FeedbackIssueCode.FUTURE_TIMESTAMP,
            f"hand state timestamp is {abs(age_s):.3f}s in the future",
        )
    if age_s > max_age_s:
        return FeedbackIssue(
            FeedbackIssueCode.STALE, f"hand state stale ({age_s:.2f}s)"
        )
    return None


def validate_arm_feedback(
    *,
    connected: bool,
    error_code: int,
    state_valid: bool,
    source_monotonic_ns: int,
    now_monotonic_ns: int,
    max_age_s: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> str | None:
    """Return why required arm feedback is unusable, or ``None``."""
    issue = diagnose_arm_feedback(
        connected=connected,
        error_code=error_code,
        state_valid=state_valid,
        source_monotonic_ns=source_monotonic_ns,
        now_monotonic_ns=now_monotonic_ns,
        max_age_s=max_age_s,
        qpos=qpos,
        qvel=qvel,
    )
    return None if issue is None else issue.detail


def validate_hand_feedback(
    *,
    connected: bool,
    state_valid: bool,
    source_monotonic_ns: int,
    now_monotonic_ns: int,
    max_age_s: float,
    qpos: np.ndarray,
) -> str | None:
    """Return why measured XHand feedback is unusable, or ``None``."""
    issue = diagnose_hand_feedback(
        connected=connected,
        state_valid=state_valid,
        source_monotonic_ns=source_monotonic_ns,
        now_monotonic_ns=now_monotonic_ns,
        max_age_s=max_age_s,
        qpos=qpos,
    )
    return None if issue is None else issue.detail
