"""Pure feedback-health predicates shared by teleop, replay, and the policy
publication boundary.

These are schema-shape and freshness checks over fixed NumPy state dtypes —
not vendor I/O and not policy disposition.  ``robot/`` keeps doing device I/O;
policy/teleop keep owning what a rejection *means*.  Both arm and hand
predicates live here (rather than split across two packages) so the pair of
fail-closed predicates has a single home.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

XHAND_OVERCURRENT_ERROR_CODE = 1_501_035

__all__ = [
    "HandOvercurrentGate",
    "XHAND_OVERCURRENT_ERROR_CODE",
    "validate_arm_feedback",
    "validate_hand_feedback",
]


@dataclass
class HandOvercurrentGate:
    """Latch an observed overcurrent until healthy feedback and operator ack."""

    recovery_frames: int
    last_event_count: int = 0
    resume_required: bool = False
    healthy_frames: int = 0

    def __post_init__(self) -> None:
        if self.recovery_frames <= 0:
            raise ValueError("overcurrent recovery_frames must be positive")

    @property
    def can_resume(self) -> bool:
        return self.resume_required and self.healthy_frames >= self.recovery_frames

    def observe(self, *, event_count: int, feedback_healthy: bool) -> bool:
        """Consume cumulative telemetry and return whether a new event arrived."""
        count = int(event_count)
        if count < 0:
            raise ValueError("overcurrent event_count must be non-negative")
        new_event = count > self.last_event_count
        self.last_event_count = max(self.last_event_count, count)
        if new_event:
            self.resume_required = True
            self.healthy_frames = 0
        if self.resume_required:
            if feedback_healthy:
                self.healthy_frames = min(self.recovery_frames, self.healthy_frames + 1)
            else:
                self.healthy_frames = 0
        return new_event

    def acknowledge_resume(self) -> bool:
        """Clear the latch only after the configured healthy window."""
        if not self.can_resume:
            return False
        self.resume_required = False
        self.healthy_frames = 0
        return True


def validate_arm_feedback(
    *,
    connected: bool,
    state_valid: bool,
    source_monotonic_ns: int,
    now_monotonic_ns: int,
    max_age_s: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    eef_pos: np.ndarray,
    eef_rot6d: np.ndarray,
) -> str | None:
    """Return why required arm feedback is unusable, or ``None``."""
    if not connected:
        return "arm disconnected"
    if not state_valid:
        return "arm state marked invalid"
    if source_monotonic_ns <= 0:
        return "arm state has no source timestamp"
    age_s = (now_monotonic_ns - source_monotonic_ns) * 1e-9
    if age_s < 0.0:
        return f"arm state timestamp is {abs(age_s):.3f}s in the future"
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    if age_s > max_age_s:
        return f"arm state stale ({age_s:.2f}s)"
    fields = {
        "qpos": (np.asarray(qpos), ARM_JOINT_SHAPE),
        "qvel": (np.asarray(qvel), ARM_JOINT_SHAPE),
        "eef_pos": (np.asarray(eef_pos), (3,)),
        "eef_rot6d": (np.asarray(eef_rot6d), (6,)),
    }
    for name, (value, expected_shape) in fields.items():
        if value.shape != expected_shape:
            return f"arm {name} has shape {value.shape}, expected {expected_shape}"
        if not np.all(np.isfinite(value)):
            return f"arm {name} is non-finite"
    return None


def validate_hand_feedback(
    *,
    connected: bool,
    error_state: bool,
    state_valid: bool,
    send_healthy: bool,
    read_healthy: bool,
    source_monotonic_ns: int,
    now_monotonic_ns: int,
    max_age_s: float,
    qpos: np.ndarray,
) -> str | None:
    """Return why measured XHand feedback is unusable, or ``None``."""
    if not connected:
        return "hand disconnected"
    if error_state:
        return "hand reported a hardware error"
    if not state_valid:
        return "hand state marked invalid"
    if not send_healthy or not read_healthy:
        return "hand command/state I/O is unhealthy"
    if source_monotonic_ns <= 0:
        return "hand state has no source timestamp"
    age_s = (now_monotonic_ns - source_monotonic_ns) * 1e-9
    if age_s < 0.0:
        return f"hand state timestamp is {abs(age_s):.3f}s in the future"
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    if age_s > max_age_s:
        return f"hand state stale ({age_s:.2f}s)"
    value = np.asarray(qpos)
    if value.shape != HAND_JOINT_SHAPE:
        return f"hand qpos has shape {value.shape}, expected {HAND_JOINT_SHAPE}"
    if not np.all(np.isfinite(value)):
        return "hand qpos is non-finite"
    return None
