"""Strict core fields and optional source/pipeline timing evidence in data.h5.

``timing/rows`` aligns with core observation rows. ``timing/queries`` stores one
JSON record per (run_id, query_id), including the query's observation history
sources and inference start/completion. ``timing/termination`` is a separate
JSON boundary record, including a pending query and its deadline when present.
Missing row values are -1; unavailable query times are null. Completion after
Raw publication never rewrites these records. Source/pipeline clocks use host
monotonic ns. Camera SDK timestamps separately retain their per-channel domains
(0=hardware, 1=system, 2=global); global/system source times are mapped to monotonic,
hardware-only sources use first host receive time. These are not guaranteed
exposure-midpoint or achieved-motion clocks. Missing timing columns in older Raw
remain unavailable, without rewriting or filling historical evidence.
"""

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


# Optional evidence, independent of core Raw fields. Signed -1 means unavailable.
# Rows align with core observation rows; queries/termination have separate lengths.
# slot is the visited owner slot; query/index identify its candidate target, even
# if rejected. dispatch_status alone indicates which SDK calls actually occurred.
TIMING_REQUIRED_FIELDS = (
    "observation_ns", "control_ns", "arm_ns", "hand_ns", "camera_ns", "pointcloud_ns",
    "camera_sequence", "pointcloud_camera_sequence", "run_id", "slot", "query_id",
    "prediction_index", "submit_started_ns", "submit_completed_ns",
)
TIMING_ROW_DTYPE = np.dtype([(name, "<i8") for name in TIMING_REQUIRED_FIELDS + (
    "camera_received_ns", "camera_published_ns", "camera_depth_sdk_ns", "camera_color_sdk_ns",
    "camera_depth_timestamp_domain", "camera_color_timestamp_domain",
    "camera_depth_frame_number", "camera_color_frame_number",
    "pointcloud_camera_received_ns", "pointcloud_camera_published_ns",
    "pointcloud_started_ns", "pointcloud_published_ns",
)])
