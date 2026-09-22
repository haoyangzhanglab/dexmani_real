"""The only supported raw episode layout: physical source rows, schema v31."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.ipc.schema import RECORD_COMMAND_FIELDS

EPISODE_SCHEMA_VERSION = 31
ARM_SENT_DATASET = "action_arm_joint_sent"




@dataclass(frozen=True)
class DatasetSpec:
    tail_shape: tuple[int, ...]
    dtype: np.dtype


def _spec(dtype, tail_shape=()):
    return DatasetSpec(tail_shape, np.dtype(dtype))


DATASET_SPECS = {
    "timestamp": _spec(np.float64),
    "arm_qpos": _spec(np.float64, (7,)),
    "arm_qvel": _spec(np.float64, (7,)),
    "arm_tau": _spec(np.float64, (7,)),
    "hand_qpos": _spec(np.float64, (12,)),
    "hand_current": _spec(np.float64, (12,)),
    "hand_contact": _spec(np.float32, (5, 3)),
    # Tactile validity states only whether the bias-corrected payload of the
    # selected causal hand sample is usable; it never encodes freshness, units,
    # or contact state. Invalid payloads are persisted as NaN.
    "hand_contact_valid": _spec(np.bool_),
    "hand_tactile_force": _spec(np.float32, (5, 120, 3)),
    "hand_tactile_force_valid": _spec(np.bool_),
    "arm_connected": _spec(np.bool_),
    "hand_connected": _spec(np.bool_),
    "hand_qpos_stale": _spec(np.bool_),
    "tracking_error": _spec(np.float64),
    "action_arm_joint_sent": _spec(np.float64, (7,)),
    "action_hand_joint": _spec(np.float64, (12,)),
    "action_arm_ee": _spec(np.float64, (9,)),
    **{name: _spec(dtype) for name, dtype in RECORD_COMMAND_FIELDS},
    "flag_frame_status": _spec(np.uint8),
    "observation_anchor_monotonic_ns": _spec(np.uint64),
    "observation_valid": _spec(np.bool_),
    "arm_source_monotonic_ns": _spec(np.uint64),
    "hand_source_monotonic_ns": _spec(np.uint64),
    "vr_source_monotonic_ns": _spec(np.uint64),
    "camera_source_monotonic_ns": _spec(np.uint64),
    "flag_camera_fresh": _spec(np.bool_),
    # Persisted camera header health enum; ``flag_camera_fresh`` keeps its
    # runtime "new + healthy + recent" semantics and is retained for audit.
    "camera_health": _spec(np.uint8),
    "camera_depth_frame_number": _spec(np.uint64),
    "camera_color_frame_number": _spec(np.uint64),
    "vr_wrist_pos": _spec(np.float64, (3,)),
    "vr_wrist_rot6d": _spec(np.float64, (6,)),
    "vr_landmarks": _spec(np.float64, (21, 3)),
    "head_quat_wxyz": _spec(np.float64, (4,)),
}
SOURCE_FRAME_DATASET_NAMES = frozenset(DATASET_SPECS) - {
    "timestamp",
}


def validate_data_layout(shapes, dtypes, *, frame_count: int) -> tuple[str, ...]:
    """Check the storage boundary without replaying runtime admission proofs."""
    errors = []
    if frame_count < 0:
        errors.append("num_frames must be non-negative")
    for name, spec in DATASET_SPECS.items():
        if name not in shapes:
            errors.append(f"missing required data.h5 dataset: {name}")
            continue
        if tuple(shapes[name]) != (frame_count,) + spec.tail_shape:
            errors.append(f"wrong shape for {name}: {shapes[name]}")
        if name not in dtypes or np.dtype(dtypes[name]) != spec.dtype:
            errors.append(f"wrong dtype for {name}: expected {spec.dtype}")
    for name in set(shapes) - DATASET_SPECS.keys():
        errors.append(f"unexpected data.h5 dataset: {name}")
    return tuple(errors)


FRAME_OK = 0
FRAME_HELD = 1
FRAME_IK_FAIL = 2
FRAME_SAFETY_REJECT = 3
FRAME_RETARGET_FAIL = 4
FRAME_PARTIAL_ADOPTION = 5
FRAME_ADOPTION_UNKNOWN = 6
DIAGNOSTIC_STATUSES = (FRAME_PARTIAL_ADOPTION, FRAME_ADOPTION_UNKNOWN)


def validate_command_row(row) -> None:
    """Validate externally persisted command facts without inventing missing ACKs."""
    status = int(row["flag_frame_status"])
    if status not in range(7):
        raise ValueError("unknown frame status")
    cid = int(row["command_id"])
    issued = int(row["command_issued_monotonic_ns"])
    present = False
    complete = True
    for actuator in ("arm", "hand"):
        has_target = bool(row[f"command_{actuator}_present"])
        adopted = bool(row[f"{actuator}_command_adopted"])
        stamp = int(row[f"{actuator}_command_adopted_monotonic_ns"])
        present |= has_target
        complete &= not has_target or adopted
        if adopted != (stamp > 0) or (adopted and (not has_target or stamp < issued)):
            raise ValueError(f"inconsistent {actuator} adoption")
        if cid == 0 and (has_target or adopted or stamp):
            raise ValueError("unknown command cannot carry actuator facts")
    if cid == 0:
        if issued or int(row["command_run_id"]):
            raise ValueError("unknown command has identity/timing")
        if status == FRAME_OK:
            raise ValueError("normal action requires an adopted command")
    elif not issued or not int(row["command_run_id"]) or not present:
        raise ValueError("command identity/presence/timing incomplete")
    elif status not in DIAGNOSTIC_STATUSES and not complete:
        raise ValueError("normal/held action is not jointly adopted")
    if status == FRAME_PARTIAL_ADOPTION:
        adopted_count = sum(bool(row[f"{name}_command_adopted"]) for name in ("arm", "hand"))
        present_count = sum(bool(row[f"command_{name}_present"]) for name in ("arm", "hand"))
        if not 0 < adopted_count < present_count:
            raise ValueError("partial adoption requires a known adopted subset")


COMMAND_IDENTITY_FIELDS = tuple(name for name, _ in RECORD_COMMAND_FIELDS) + (
    "action_arm_joint_sent", "action_hand_joint", "action_arm_ee",
)


class CommandHistory:
    """Raw identity validation shared by the streaming writer and offline reader."""

    def __init__(self) -> None:
        self.highest_id = 0
        self.last_adopted = None

    def observe(self, row) -> bool:
        validate_command_row(row)
        cid = int(row["command_id"])
        status = int(row["flag_frame_status"])
        if status not in (FRAME_OK, *DIAGNOSTIC_STATUSES):
            # After partial adoption a held row still references the last JOINT
            # command; it does not claim to undo the diagnostic physical action.
            expected = 0 if self.last_adopted is None else int(self.last_adopted["command_id"])
            if cid != expected:
                raise ValueError("held/failure row must reference the last jointly adopted command")
            if cid:
                for name in COMMAND_IDENTITY_FIELDS:
                    if not np.array_equal(row[name], self.last_adopted[name], equal_nan=True):
                        raise ValueError(f"repeated command changed {name}")
            return False
        if cid <= self.highest_id:
            raise ValueError("new command IDs must increase")
        self.highest_id = cid
        if status == FRAME_OK:
            self.last_adopted = {name: np.copy(row[name]) for name in COMMAND_IDENTITY_FIELDS}
            return True
        return False


def command_send_mask(rows) -> np.ndarray:
    """Validate command history and select new jointly adopted commands."""
    mask = np.zeros(len(rows["command_id"]), dtype=bool)
    history = CommandHistory()
    for i in range(len(mask)):
        row = {name: rows[name][i] for name in (*COMMAND_IDENTITY_FIELDS, "flag_frame_status")}
        mask[i] = history.observe(row)
    return mask
