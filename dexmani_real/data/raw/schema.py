"""Raw observations, dispatch evidence and host timing."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import DispatchStatus

RAW_FORMAT = "dexmani.raw"


@dataclass(frozen=True)
class DatasetSpec:
    tail_shape: tuple[int, ...]
    dtype: np.dtype


def dataset_spec(dtype, tail_shape=()):
    return DatasetSpec(tail_shape, np.dtype(dtype))


DATASET_SPECS = {
    "arm_qpos": dataset_spec(np.float64, (7,)),
    "arm_qvel": dataset_spec(np.float64, (7,)),
    # xArm get_joint_states(num=3) effort keeps SDK-native units; SI torque is not established.
    "arm_effort": dataset_spec(np.float64, (7,)),
    "hand_qpos": dataset_spec(np.float64, (12,)),
    "hand_current": dataset_spec(np.float64, (12,)),
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
    if np.any(dispatch_status > DispatchStatus.SUBMITTED):
        errors.append("dispatch_status must contain status codes 0 through 5")
    return tuple(errors)


def validate_dispatch_rows(dispatch_status) -> tuple[str, ...]:
    """Shared dispatch admission for training and replay; execution is checked separately."""
    dispatch = np.asarray(dispatch_status)
    if dispatch.ndim != 2 or dispatch.shape[1] != 2:
        return ("dispatch_status must have shape (N, 2)",)
    bad_dispatch = np.flatnonzero(
        (dispatch[:, 0] != DispatchStatus.SUBMITTED)
        | ~np.isin(dispatch[:, 1], (DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED))
    )
    if len(bad_dispatch):
        row = int(bad_dispatch[0])
        return (
            f"dispatch row {row}: arm/hand={list(dispatch_status[row])}; "
            "requires arm SUBMITTED, hand ACCEPTED or CRC_UNCONFIRMED",
        )
    return ()


def validate_execution_boundary(snapshot, *, stopped=False):
    """Only resolved execution can be published; a resolved fault remains Raw evidence."""
    if not isinstance(snapshot, dict):
        raise ValueError("arm execution snapshot missing")
    for name in (
        "submitted",
        "first_accepted",
        "sdk_attempted",
        "sdk_completed",
        "sdk_accepted",
        "cancelled_before_first",
        "active_sequence",
        "pending_sequence",
    ):
        value = snapshot[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid arm execution counter: {name}")
    call = snapshot["last_call"]
    if (
        snapshot["active_sequence"]
        or snapshot["pending_sequence"]
        or snapshot["operation_unresolved"]
        or snapshot["close_error"]
        or snapshot["sdk_attempted"] != snapshot["sdk_completed"]
        or (call["started_ns"] > 0 and call["completed_ns"] == 0)
    ):
        raise ValueError("arm execution/stop unresolved; retain staging")
    if stopped:
        if not snapshot["stopped"] or not snapshot["robot_stopped"]:
            raise ValueError("robot stop not confirmed; retain staging")
    elif not snapshot["connected"] or snapshot["fault"]:
        raise ValueError("capture start requires a healthy arm owner")


def validate_arm_execution(start, end, rows, dispatch_status):
    """Admit complete episodes; SDK acceptance counts once per target and does not imply arrival."""
    validate_execution_boundary(start)
    validate_execution_boundary(end, stopped=True)
    if end["fault"]:
        raise ValueError(f"arm execution fault: {end['fault']}")
    delta = {}
    for name in (
        "submitted",
        "first_accepted",
        "sdk_attempted",
        "sdk_completed",
        "sdk_accepted",
        "cancelled_before_first",
    ):
        delta[name] = end[name] - start[name]
        if delta[name] < 0:
            raise ValueError(f"arm execution counter regressed: {name}")
    dispatch = np.asarray(dispatch_status)
    if not len(rows) or dispatch.shape != (len(rows), 2):
        raise ValueError("arm execution requires nonempty aligned control rows")
    submitted = dispatch[:, 0] == DispatchStatus.SUBMITTED
    sequence = rows["command_seq"]
    if delta["submitted"] != int(submitted.sum()):
        raise ValueError("arm submissions/control-row count does not match")
    expected = np.arange(start["submitted"] + 1, end["submitted"] + 1, dtype=np.int64)
    if (
        not np.array_equal(sequence[submitted], expected)
        or np.any(sequence[~submitted] != -1)
        or np.any(rows["run_id"][submitted] <= 0)
    ):
        raise ValueError("arm submissions/control-row sequences do not match")
    if delta["cancelled_before_first"] or delta["first_accepted"] != delta["submitted"]:
        raise ValueError("arm target cancelled or never accepted by SDK")
    if not (
        delta["sdk_attempted"] == delta["sdk_completed"] == delta["sdk_accepted"]
        and delta["sdk_accepted"] >= delta["first_accepted"]
    ):
        raise ValueError("arm SDK calls incomplete or not accepted")


# Timing rows are required; individual unavailable times/identities use -1.
# Queries/termination are separate; visited candidates need not have crossed the SDK boundary.
# dispatch_status records host submission / hand SDK results; arm_execution summarizes arm SDK results.
TIMING_ROW_DTYPE = np.dtype(
    [
        (name, "<i8")
        for name in (
            "observation_ns",
            "control_ns",
            "arm_ns",
            "hand_ns",
            "camera_ns",
            "pointcloud_ns",
            "camera_sequence",
            "pointcloud_camera_sequence",
            "run_id",
            "slot",
            "query_id",
            "prediction_index",
            "submit_started_ns",
            "submit_completed_ns",
            "command_seq",
            "camera_received_ns",
            "camera_published_ns",
            "camera_depth_frame_number",
            "camera_color_frame_number",
            "pointcloud_camera_received_ns",
            "pointcloud_camera_published_ns",
            "pointcloud_started_ns",
            "pointcloud_published_ns",
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
        "arm_qvel": arm["qvel"],
        "arm_effort": arm["effort"],
        "hand_qpos": hand["qpos"] if hand is not None else np.full(12, np.nan),
        "hand_current": hand["current"] if hand is not None else np.full(12, np.nan),
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


def observation_timing(row):
    """Host observation and pipeline clocks in monotonic nanoseconds."""
    return dict(
        observation_ns=int(row.observation_timestamp_ns),
        control_ns=int(row.control_timestamp_ns),
        arm_ns=int(row.arm["timestamp_ns"][0]),
        hand_ns=int(row.hand["timestamp_ns"][0]) if row.hand is not None else -1,
        camera_ns=int(row.camera["timestamp_ns"]) if row.camera is not None else -1,
        pointcloud_ns=int(row.pointcloud_timestamp_ns) if row.pointcloud_timestamp_ns > 0 else -1,
        camera_sequence=int(row.camera["ring_sequence"]) if row.camera is not None else -1,
        pointcloud_camera_sequence=int(row.pointcloud_camera_sequence),
        **{
            f"camera_{name}": int(row.camera[name]) if row.camera is not None else -1
            for name in ("received_ns", "published_ns", "depth_frame_number", "color_frame_number")
        },
        pointcloud_camera_received_ns=int(row.pointcloud_camera_received_ns),
        pointcloud_camera_published_ns=int(row.pointcloud_camera_published_ns),
        pointcloud_started_ns=int(row.pointcloud_started_ns),
        pointcloud_published_ns=int(row.pointcloud_published_ns),
    )


def encode_timing_row(row, result, execution_timing):
    evidence = dict(observation_timing(row), **execution_timing)
    evidence["command_seq"] = -1
    if result is not None and result.arm == DispatchStatus.SUBMITTED:
        run_id, sequence = result.command_id
        if evidence.get("run_id", run_id) != run_id:
            raise ValueError("dispatch/timing run_id mismatch")
        evidence.update(run_id=run_id, command_seq=sequence)
    unknown = evidence.keys() - set(TIMING_ROW_DTYPE.names)
    if unknown:
        raise ValueError(f"Unknown timing row fields: {sorted(unknown)}")
    return tuple(evidence.get(name, -1) for name in TIMING_ROW_DTYPE.names)
