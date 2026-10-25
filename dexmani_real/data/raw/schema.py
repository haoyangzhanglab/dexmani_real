"""Raw observations, final submission attempts and host timing."""

import numpy as np

from dexmani_real.utils.timing import sampling_clock_ns

ARM_ACTION_SEMANTICS = "q_goal_submission_attempt"
TACTILE_PROCESSING = "sdk_native_bounded_previous_valid_vector_hold"
TACTILE_UNITS = "sdk_native_unknown_physical_scale"
TACTILE_AXES = "sdk_sensor_axes_not_world_frame"
TACTILE_FINGER_ORDER = "thumb,index,middle,ring,pinky"


DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "arm_qpos": ((7,), np.dtype(np.float64)),
    "hand_qpos": ((12,), np.dtype(np.float64)),
    # Consumed tactile observation: SDK units, bounded previous-valid vector hold (see metadata).
    "hand_contact": ((5, 3), np.dtype(np.float32)),
    "hand_tactile_force": ((5, 120, 3), np.dtype(np.float32)),
    # Final absolute q_goal passed to submission, including failed attempts.
    "action_arm_joint_target": ((7,), np.dtype(np.float64)),
    "action_hand_joint_target": ((12,), np.dtype(np.float64)),
    # Seconds relative to the first observation.
    "timestamp": ((), np.dtype(np.float64)),
}


def validate_fixed_rate_rows(
    timestamp, control_hz, *, row_offset=0
) -> tuple[str, ...]:
    """Admit complete fixed-rate rows without compressing waits or missed slots."""
    timestamp = np.asarray(timestamp)
    if timestamp.ndim != 1:
        return ("timestamp must be a one-dimensional array",)
    if not np.isfinite(timestamp).all() or np.any(timestamp < 0):
        return ("timestamp must contain finite nonnegative relative seconds",)
    if len(timestamp) and row_offset == 0 and timestamp[0] != 0:
        return ("timestamp must start at zero",)
    if np.any(np.diff(timestamp) <= 0):
        return ("timestamp must be strictly increasing within an episode",)

    dt_ns, tolerance_ns = sampling_clock_ns(control_hz)
    stamps = np.rint(timestamp * 1e9)
    intervals = np.diff(stamps)
    deviations = stamps - np.arange(row_offset, row_offset + len(stamps)) * dt_ns
    bad_interval = np.flatnonzero(np.abs(intervals - dt_ns) > tolerance_ns)
    bad_grid = np.flatnonzero(np.abs(deviations) > tolerance_ns)
    bad_rows = [int(bad_interval[0]) + 1] if len(bad_interval) else []
    if len(bad_grid):
        bad_rows.append(int(bad_grid[0]))
    if bad_rows:
        row = min(bad_rows)
        interval = intervals[row - 1] if row else float("nan")
        return (
            f"timing row {row + row_offset}: interval_ns={interval:.0f}, "
            f"nominal_ns={dt_ns}, grid_deviation_ns={deviations[row]:.0f}, "
            f"tolerance_ns={tolerance_ns}",
        )
    return ()


# Host monotonic clocks; unavailable sources use -1. Tactile clocks retain the
# oldest original valid vector, rather than the later joint-feedback read time.
TIMING_ROW_DTYPE = np.dtype(
    [
        (name, "<i8")
        for name in (
            "observation_ns", "control_ns", "arm_ns", "hand_ns",
            "tactile_aggregate_ns", "tactile_dense_ns", "camera_ns", "camera_sequence",
        )
    ]
)


def encode_raw_row(row, command, timestamp):
    arm, hand, camera = (
        row.arm[0],
        None if row.hand is None else row.hand[0],
        row.camera,
    )
    if camera is None:
        raise ValueError("recording requires RGB-D from the current observation")
    values = {
        "timestamp": timestamp,
        "arm_qpos": arm["qpos"],
        "hand_qpos": hand["qpos"] if hand is not None else np.full(12, np.nan),
        "hand_contact": hand["tactile_aggregate"] if hand is not None else np.full((5, 3), np.nan),
        "hand_tactile_force": (
            hand["tactile_dense"] if hand is not None else np.full((5, 120, 3), np.nan)
        ),
        "action_arm_joint_target": (
            command.arm_qpos
            if command is not None
            and command.arm_qpos is not None
            else np.full(7, np.nan)
        ),
        "action_hand_joint_target": (
            command.hand_qpos
            if command is not None
            and command.hand_qpos is not None
            else np.full(12, np.nan)
        ),
    }
    for name, value in values.items():
        tail_shape, dtype = DATASET_SPECS[name]
        array = np.asarray(value, dtype=dtype)
        if array.shape != tail_shape:
            raise ValueError(f"Raw {name}: expected {tail_shape}, got {array.shape}")
        values[name] = array
    return values


def encode_timing_row(row):
    """Capture consumed source clocks without scheduling or SDK bookkeeping."""
    hand = row.hand
    return (
        int(row.observation_timestamp_ns),
        int(row.control_timestamp_ns),
        int(row.arm["timestamp_ns"][0]),
        int(hand["timestamp_ns"][0]) if hand is not None else -1,
        int(hand["tactile_aggregate_timestamp_ns"][0]) if hand is not None else -1,
        int(hand["tactile_dense_timestamp_ns"][0]) if hand is not None else -1,
        int(row.camera["timestamp_ns"]) if row.camera is not None else -1,
        int(row.camera["ring_sequence"]) if row.camera is not None else -1,
    )
