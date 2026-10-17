"""Raw core data, execution summaries and host-monotonic timing in data.h5.
Rows align with observations; queries key by (run_id,query_id), and termination is separate.
Unavailable row/query times are -1/null; published Raw is never rewritten.
RGB-D/cloud age starts at the oldest channel's first host receive, excluding pre-receive SDK latency.
"""

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import DispatchStatus
from dexmani_real.utils.control_clock import sampling_clock_ns

RAW_FORMAT = "dexmani.raw"
ARM_EXECUTION_FORMAT = "mode1_episode_summary_v1"


@dataclass(frozen=True)
class DatasetSpec:
    tail_shape: tuple[int, ...]
    dtype: np.dtype


def _spec(dtype, tail_shape=()):
    return DatasetSpec(tail_shape, np.dtype(dtype))


DATASET_SPECS = {
    "arm_qpos": _spec(np.float64, (7,)),
    "arm_qvel": _spec(np.float64, (7,)),
    # xArm get_joint_states(num=3) effort keeps SDK-native units; SI torque is not established.
    "arm_effort": _spec(np.float64, (7,)),
    "hand_qpos": _spec(np.float64, (12,)),
    "hand_current": _spec(np.float64, (12,)),
    # Consumed tactile observation: SDK units, bounded previous-valid vector hold (see metadata).
    "hand_contact": _spec(np.float32, (5, 3)),
    "hand_tactile_force": _spec(np.float32, (5, 120, 3)),
    # Submitted absolute q_goal, not the next step's measured arrival.
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
    if np.any(dispatch_status > DispatchStatus.SUBMITTED):
        errors.append("dispatch_status must contain status codes 0 through 5")
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
    return validate_dispatch_rows(dispatch_status)


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
        return (f"dispatch row {row}: arm/hand={list(dispatch_status[row])}; "
                "requires arm SUBMITTED, hand ACCEPTED or CRC_UNCONFIRMED",)
    return ()


def validate_execution_boundary(snapshot, *, stopped=False):
    """Only resolved execution can be published; a resolved fault remains Raw evidence."""
    if not isinstance(snapshot, dict):
        raise ValueError("arm execution snapshot missing")
    for name in ("submitted", "first_accepted", "sdk_attempted", "sdk_completed",
                 "sdk_accepted", "cancelled_before_first", "active_sequence", "pending_sequence"):
        value = snapshot[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid arm execution counter: {name}")
    call = snapshot["last_call"]
    if (snapshot["active_sequence"] or snapshot["pending_sequence"]
            or snapshot["operation_unresolved"] or snapshot["close_error"]
            or snapshot["sdk_attempted"] != snapshot["sdk_completed"]
            or (call["started_ns"] > 0 and call["completed_ns"] == 0)):
        raise ValueError("arm execution/stop unresolved; retain staging")
    if stopped:
        if not snapshot["stopped"] or not snapshot["robot_stopped"]:
            raise ValueError("robot stop not confirmed; retain staging")
    elif not snapshot["connected"] or snapshot["fault"]:
        raise ValueError("capture start requires a healthy arm owner")


def validate_arm_execution(start, end, rows, dispatch_status):
    """Admit a complete episode, not individual frames or servo references.

    Sequences account for every submission. The worker counts first SDK acceptance
    once per target, so accepted supersession is valid without requiring arrival.
    """
    validate_execution_boundary(start)
    validate_execution_boundary(end, stopped=True)
    if end["fault"]:
        raise ValueError(f"arm execution fault: {end['fault']}")
    delta = {}
    for name in ("submitted", "first_accepted", "sdk_attempted", "sdk_completed",
                 "sdk_accepted", "cancelled_before_first"):
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
    if (not np.array_equal(sequence[submitted], expected)
            or np.any(sequence[~submitted] != -1)
            or np.any(rows["run_id"][submitted] <= 0)):
        raise ValueError("arm submissions/control-row sequences do not match")
    if delta["cancelled_before_first"] or delta["first_accepted"] != delta["submitted"]:
        raise ValueError("arm target cancelled or never accepted by SDK")
    if not (delta["sdk_attempted"] == delta["sdk_completed"] == delta["sdk_accepted"]
            and delta["sdk_accepted"] >= delta["first_accepted"]):
        raise ValueError("arm SDK calls incomplete or not accepted")


# Timing rows are required; individual unavailable times/identities use -1.
# Queries/termination are separate; visited candidates need not have crossed the SDK boundary.
# dispatch_status records host submission / hand SDK results; arm_execution summarizes arm SDK results.
TIMING_ROW_DTYPE = np.dtype([(name, "<i8") for name in (
    "observation_ns", "control_ns", "arm_ns", "hand_ns", "camera_ns", "pointcloud_ns",
    "camera_sequence", "pointcloud_camera_sequence", "run_id", "slot", "query_id",
    "prediction_index", "submit_started_ns", "submit_completed_ns", "command_seq",
    "camera_received_ns", "camera_published_ns",
    "camera_depth_frame_number", "camera_color_frame_number",
    "pointcloud_camera_received_ns", "pointcloud_camera_published_ns",
    "pointcloud_started_ns", "pointcloud_published_ns",
)])
