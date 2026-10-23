"""Raw observations, dispatch evidence and host timing."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import DispatchStatus
from dexmani_real.utils.timing import sampling_clock_ns

RAW_FORMAT = "dexmani.raw"


@dataclass(frozen=True)
class DatasetSpec:
    tail_shape: tuple[int, ...]
    dtype: np.dtype


def dataset_spec(dtype, tail_shape=()):
    return DatasetSpec(tail_shape, np.dtype(dtype))


DATASET_SPECS = {
    "arm_qpos": dataset_spec(np.float64, (7,)),
    "hand_qpos": dataset_spec(np.float64, (12,)),
    # Consumed tactile observation: SDK units, bounded previous-valid vector hold (see metadata).
    "hand_contact": dataset_spec(np.float32, (5, 3)),
    "hand_tactile_force": dataset_spec(np.float32, (5, 120, 3)),
    # Submitted absolute q_goal, not the next step's measured arrival.
    "action_arm_joint_target": dataset_spec(np.float64, (7,)),
    "action_hand_joint_target": dataset_spec(np.float64, (12,)),
    # Seconds relative to the first observation; dispatch columns are arm, hand.
    "timestamp": dataset_spec(np.float64),
    "dispatch_status": dataset_spec(np.uint8, (2,)),
}


def validate_fixed_rate_rows(
    timestamp, dispatch_status, control_hz, *, row_offset=0
) -> tuple[str, ...]:
    """Admit complete fixed-rate rows without compressing waits or missed slots."""
    timestamp = np.asarray(timestamp)
    dispatch = np.asarray(dispatch_status)
    if timestamp.ndim != 1 or dispatch.shape != (len(timestamp), 2):
        return ("timestamp and dispatch_status must be aligned (N,) and (N, 2) arrays",)
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
    # Host submission is not SDK acceptance or measured arrival.
    bad_dispatch = np.flatnonzero(
        (dispatch[:, 0] != DispatchStatus.SUBMITTED)
        | ~np.isin(dispatch[:, 1], (DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED))
    )
    if len(bad_dispatch):
        row = int(bad_dispatch[0])
        return (
            f"dispatch row {row + row_offset}: arm/hand={list(dispatch[row])}; "
            "requires arm SUBMITTED, hand ACCEPTED or CRC_UNCONFIRMED",
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


def encode_raw_row(row, command, result, timestamp):
    arm, hand, camera = (
        row.arm[0],
        None if row.hand is None else row.hand[0],
        row.camera,
    )
    if camera is None:
        raise ValueError("recording requires RGB-D from the current observation")
    values = {
        "timestamp": timestamp,
        "dispatch_status": (int(result.arm), int(result.hand)) if result is not None else (0, 0),
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
            and result is not None
            and result.arm != DispatchStatus.NOT_CALLED
            else np.full(7, np.nan)
        ),
        "action_hand_joint_target": (
            command.hand_qpos
            if command is not None
            and command.hand_qpos is not None
            and result is not None
            and result.hand != DispatchStatus.NOT_CALLED
            else np.full(12, np.nan)
        ),
    }
    if any(status not in range(6) for status in values["dispatch_status"]):
        raise ValueError("dispatch_status must contain status codes 0 through 5")
    for name, value in values.items():
        spec = DATASET_SPECS[name]
        array = np.asarray(value, dtype=spec.dtype)
        if array.shape != spec.tail_shape:
            raise ValueError(f"Raw {name}: expected {spec.tail_shape}, got {array.shape}")
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


def unwritten_dispatch_detail(observation_ns, dispatch_status, arm_target, hand_target):
    """A bounded failed-write fact, not an SDK execution audit or a replacement row."""
    arm_status, hand_status = map(int, dispatch_status)
    if not (arm_status or hand_status):
        return None
    return dict(
        stage="unwritten_dispatch", observation_ns=int(observation_ns),
        dispatch_status=[arm_status, hand_status],
        arm_target=np.asarray(arm_target).tolist() if arm_status and arm_target is not None else None,
        hand_target=np.asarray(hand_target).tolist() if hand_status and hand_target is not None else None,
    )
