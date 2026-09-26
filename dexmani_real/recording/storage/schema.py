"""The only supported raw episode layout: research control rows, schema v33."""

from dataclasses import dataclass

import numpy as np

EPISODE_SCHEMA_VERSION = 33


@dataclass(frozen=True)
class DatasetSpec:
    tail_shape: tuple[int, ...]
    dtype: np.dtype


def _spec(dtype, tail_shape=()):
    return DatasetSpec(tail_shape, np.dtype(dtype))


DATASET_SPECS = {
    "timestamp": _spec(np.float64),
    "action_timestamp_ns": _spec(np.uint64),
    "arm_qpos": _spec(np.float64, (7,)),
    "arm_qvel": _spec(np.float64, (7,)),
    "arm_effort": _spec(np.float64, (7,)),
    "hand_qpos": _spec(np.float64, (12,)),
    "hand_current": _spec(np.float64, (12,)),
    "hand_contact": _spec(np.float32, (5, 3)),
    # Tactile validity states only whether the bias-corrected payload of the
    # selected hand sample is usable; it never encodes freshness, units,
    # or contact state. Invalid payloads are persisted as NaN.
    "hand_contact_valid": _spec(np.bool_),
    "hand_tactile_force": _spec(np.float32, (5, 120, 3)),
    "hand_tactile_force_valid": _spec(np.bool_),
    "action_arm_joint_target": _spec(np.float64, (7,)),
    "action_hand_joint_target": _spec(np.float64, (12,)),
    "arm_eef_intent": _spec(np.float64, (9,)),
    "flag_frame_status": _spec(np.uint8),
    "observation_timestamp_ns": _spec(np.uint64),
    "arm_timestamp_ns": _spec(np.uint64),
    "hand_timestamp_ns": _spec(np.uint64),
    "vr_timestamp_ns": _spec(np.uint64),
    "camera_timestamp_ns": _spec(np.uint64),
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
    """Check required raw arrays, row counts, shapes, and dtypes."""
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
FRAME_IK_FAIL = 1
FRAME_RETARGET_FAIL = 2
