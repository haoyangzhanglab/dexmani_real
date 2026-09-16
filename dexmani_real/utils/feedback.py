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

from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

__all__ = [
    "FeedbackIssue",
    "FeedbackIssueCode",
    "diagnose_arm_feedback",
    "diagnose_hand_feedback",
    "diagnose_feedback_timestamp_order",
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
    TIMESTAMP_ORDER = "timestamp_order"
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


def diagnose_feedback_timestamp_order(
    *,
    source_monotonic_ns: int,
    ring_commit_monotonic_ns: int,
    validation_now_ns: int,
    max_age_s: float,
    modality: str,
) -> FeedbackIssue | None:
    """Validate ring-commit provenance and freshness against a post-selection now.

    This is the post-selection recheck, layered on the per-modality
    ``diagnose_arm_feedback``/``diagnose_hand_feedback`` pass that owns the
    ``FUTURE_TIMESTAMP`` (source vs. its own post-read now) check. One
    already-selected frame must satisfy ``0 < source <= ring_commit <=
    validation_now`` and ``validation_now - source <= max_age``. A producer or
    clock-invariant violation (``source > ring_commit`` or ``ring_commit >
    validation_now``) is a fatal timestamp-order issue, distinct from a plain
    ``STALE`` frame. No future/ordering tolerance is applied.
    """
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    if source_monotonic_ns <= 0:
        return FeedbackIssue(
            FeedbackIssueCode.MISSING_TIMESTAMP,
            f"{modality} state has no source timestamp",
        )
    if source_monotonic_ns > ring_commit_monotonic_ns:
        return FeedbackIssue(
            FeedbackIssueCode.TIMESTAMP_ORDER,
            f"{modality} source_ns={source_monotonic_ns} exceeds "
            f"ring_commit_ns={ring_commit_monotonic_ns} "
            f"(source_to_commit_ms="
            f"{(ring_commit_monotonic_ns - source_monotonic_ns) / 1e6:.3f})",
        )
    if ring_commit_monotonic_ns > validation_now_ns:
        return FeedbackIssue(
            FeedbackIssueCode.TIMESTAMP_ORDER,
            f"{modality} ring_commit_ns={ring_commit_monotonic_ns} exceeds "
            f"validation_now_ns={validation_now_ns} "
            f"(commit_to_now_ms="
            f"{(validation_now_ns - ring_commit_monotonic_ns) / 1e6:.3f})",
        )
    age_s = (validation_now_ns - source_monotonic_ns) * 1e-9
    if age_s > max_age_s:
        return FeedbackIssue(
            FeedbackIssueCode.STALE, f"{modality} state stale ({age_s:.2f}s)"
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
