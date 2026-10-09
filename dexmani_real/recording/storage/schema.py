"""Strict current-writer fields; each published field keeps its meaning."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.utils.control_clock import sampling_clock_ns

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


def validate_training_rows(timestamp, dispatch_status, control_hz) -> tuple[str, ...]:
    """Whole-episode admission only; failed evidence remains readable as Raw."""
    errors = validate_capture_rows(timestamp, dispatch_status)
    if errors:
        return errors
    dt_ns, epsilon_ns = sampling_clock_ns(control_hz)
    # Raw seconds originate from integer ns; round only representation error.
    stamps = np.rint(np.asarray(timestamp) * 1e9)
    intervals = np.diff(stamps)
    deviations = stamps - np.arange(len(stamps)) * dt_ns
    bad_interval = np.flatnonzero(np.abs(intervals - dt_ns) > epsilon_ns)
    bad_grid = np.flatnonzero(np.abs(deviations) > epsilon_ns)
    bad_rows = [int(bad_interval[0]) + 1] if len(bad_interval) else []
    if len(bad_grid):
        bad_rows.append(int(bad_grid[0]))
    if bad_rows:
        row = min(bad_rows)
        return (f"timing row {row}: interval_ns={intervals[row - 1]:.0f}, "
                f"nominal_ns={dt_ns}, grid_deviation_ns={deviations[row]:.0f}, "
                f"tolerance_ns={epsilon_ns}",)
    bad_dispatch = np.flatnonzero(np.any(np.asarray(dispatch_status) != 1, axis=1))
    if len(bad_dispatch):
        row = int(bad_dispatch[0])
        return (f"dispatch row {row}: arm/hand={list(dispatch_status[row])}; "
                "training requires [1, 1] (SDK ACCEPTED)",)
    return ()
