"""The only supported raw episode layout: physical source rows, schema v28."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

EPISODE_SCHEMA_VERSION = 28
ARM_SENT_DATASET = "action_arm_joint_sent"


class FillReason(IntEnum):
    SOURCE = 0
    CAUSAL_HOLD_LAST = 1
    LEADING_PLACEHOLDER = 2


@dataclass(frozen=True)
class DatasetSpec:
    tail_shape: tuple[int, ...]
    dtype: np.dtype


def _spec(dtype, tail_shape=()):
    return DatasetSpec(tail_shape, np.dtype(dtype))


DATASET_SPECS = {
    "timestamp": _spec(np.float64),
    "source_sample_index": _spec(np.int64),
    "fill_reason": _spec(np.uint8),
    "flag_sample_valid": _spec(np.bool_),
    "arm_qpos": _spec(np.float64, (7,)),
    "arm_qvel": _spec(np.float64, (7,)),
    "arm_tau": _spec(np.float64, (7,)),
    "hand_qpos": _spec(np.float64, (12,)),
    "hand_current": _spec(np.float64, (12,)),
    "hand_contact": _spec(np.float64, (5, 3)),
    "hand_contact_source_monotonic_ns": _spec(np.uint64),
    "hand_tactile_force": _spec(np.float64, (5, 120, 3)),
    "arm_connected": _spec(np.bool_),
    "hand_connected": _spec(np.bool_),
    "hand_qpos_stale": _spec(np.bool_),
    "tracking_error": _spec(np.float64),
    "arm_last_cmd_seq": _spec(np.int64),
    "action_arm_joint_sent": _spec(np.float64, (7,)),
    "action_hand_joint": _spec(np.float64, (12,)),
    "action_arm_ee": _spec(np.float64, (9,)),
    "flag_action_queued": _spec(np.bool_),
    "flag_frame_status": _spec(np.uint8),
    "observation_anchor_monotonic_ns": _spec(np.uint64),
    "observation_valid": _spec(np.bool_),
    "arm_source_monotonic_ns": _spec(np.uint64),
    "hand_source_monotonic_ns": _spec(np.uint64),
    "tactile_source_monotonic_ns": _spec(np.uint64),
    "vr_source_monotonic_ns": _spec(np.uint64),
    "camera_source_monotonic_ns": _spec(np.uint64),
    # Aggregate contact can be valid but old; freshness is telemetry only.
    # Dense calibration/unit telemetry below describes its separate ring frame.
    "tactile_sum_fresh": _spec(np.bool_),
    "tactile_fresh": _spec(np.bool_),
    "tactile_calibrated": _spec(np.bool_),
    "tactile_unit_code": _spec(np.uint8),
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
    "source_sample_index",
    "fill_reason",
    "flag_sample_valid",
}


def required_dataset_names() -> frozenset[str]:
    return frozenset(DATASET_SPECS)


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
