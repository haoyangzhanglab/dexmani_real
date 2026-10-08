"""Strict current-writer fields; each published field keeps its meaning."""

from dataclasses import dataclass

import numpy as np

RAW_FORMAT = "dexmani.raw"


@dataclass(frozen=True)
class DatasetSpec:
    tail_shape: tuple[int, ...]
    dtype: np.dtype


def _spec(dtype, tail_shape=()):
    return DatasetSpec(tail_shape, np.dtype(dtype))


DATASET_SPECS = {
    "arm_qpos": _spec(np.float64, (7,)),
    "arm_qvel": _spec(np.float64, (7,)),
    # xArm SDK ``get_joint_states(num=3)`` effort telemetry. The SDK does not
    # establish an SI torque unit, so its native numeric semantics are preserved.
    "arm_effort": _spec(np.float64, (7,)),
    "hand_qpos": _spec(np.float64, (12,)),
    "hand_current": _spec(np.float64, (12,)),
    # XHand SDK-native, bias-corrected aggregate tactile values. Invalid
    # aggregate samples are represented by NaN.
    "hand_contact": _spec(np.float32, (5, 3)),
    # XHand SDK-native, bias-corrected dense tactile values. Invalid dense
    # samples are represented by NaN independently of aggregate tactile state.
    "hand_tactile_force": _spec(np.float32, (5, 120, 3)),
    "action_arm_joint_target": _spec(np.float64, (7,)),
    "action_hand_joint_target": _spec(np.float64, (12,)),
    # Seconds relative to the first observation; dispatch columns are arm, hand.
    "timestamp": _spec(np.float64),
    "dispatch_status": _spec(np.uint8, (2,)),
}


def validate_capture_rows(timestamp, dispatch_status) -> tuple[str, ...]:
    """Validate the episode clock and dispatch statuses, independently of observations."""
    errors = []
    if len(timestamp):
        if not np.isfinite(timestamp).all() or np.any(timestamp < 0):
            errors.append("timestamp must contain finite nonnegative relative seconds")
        if timestamp[0] != 0:
            errors.append("timestamp must start at zero")
        if np.any(np.diff(timestamp) <= 0):
            errors.append("timestamp must be strictly increasing within an episode")
    if np.any(dispatch_status > 4):
        errors.append("dispatch_status must contain status codes 0 through 4")
    return tuple(errors)
