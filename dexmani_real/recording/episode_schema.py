"""Pure schema-v17/v18 contracts shared by episode writers and readers.

The v17 HDF5 data file has 93 unconditional per-grid datasets; v18 drops three
retired arm-worker latency fields. The exact command submitted to the arm
worker is a conditional dataset controlled by the ``arm_sent_stream`` metadata
marker. Unknown datasets are tolerated for historical diagnostic extensions,
but they must remain aligned to the episode grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dexmani_real.utils.schema import (
    ARM_JOINT_SHAPE,
    HAND_CONTACT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)

EPISODE_SCHEMA_VERSION = 18
# Runtime writes v18; readers accept v17 and v18.
SUPPORTED_EPISODE_SCHEMA_VERSIONS: frozenset[int] = frozenset({17, 18})
ARM_WORKER_TELEMETRY_DATASETS_V17: frozenset[str] = frozenset(
    {
        "arm_last_cmd_queue_latency_s",
        "arm_last_cmd_apply_latency_s",
        "arm_last_cmd_sdk_duration_s",
    }
)
ARM_SENT_DATASET = "action_arm_joint_sent"
ARM_SENT_MARKER = "arm_sent_stream"

ARM_RAW_ACTION_VALIDITY_EXPRESSION = "flag_sample_valid & ~flag_held & flag_ik_ok"
HAND_RAW_ACTION_VALIDITY_EXPRESSION = (
    "flag_sample_valid & ~flag_held & flag_retarget_ok"
)


@dataclass(frozen=True)
class DatasetSpec:
    """Fixed tail shape and dtype for one per-grid ``data.h5`` dataset."""

    tail_shape: tuple[int, ...]
    dtype: np.dtype[Any]


def _spec(dtype: Any, tail_shape: tuple[int, ...] = ()) -> DatasetSpec:
    return DatasetSpec(tail_shape=tail_shape, dtype=np.dtype(dtype))


# Superset field specifications anchored to v17. ``required_dataset_names``
# Select the v17 or v18 required subset while preserving field grouping.
BASE_DATASET_SPECS_V17: dict[str, DatasetSpec] = {
    "timestamp": _spec(np.float64),
    "flag_sample_valid": _spec(np.bool_),
    "source_sample_index": _spec(np.int64),
    "source_timestamp": _spec(np.float64),
    "fill_reason": _spec(np.uint8),
    "arm_qpos": _spec(np.float64, ARM_JOINT_SHAPE),
    "arm_qvel": _spec(np.float64, ARM_JOINT_SHAPE),
    "arm_tau": _spec(np.float64, ARM_JOINT_SHAPE),
    "arm_ee": _spec(np.float64, (9,)),
    "arm_connected": _spec(np.bool_),
    "arm_last_cmd_seq": _spec(np.int64),
    "arm_last_cmd_queue_latency_s": _spec(np.float64),
    "arm_last_cmd_apply_latency_s": _spec(np.float64),
    "arm_last_cmd_sdk_duration_s": _spec(np.float64),
    "arm_last_cmd_is_hold": _spec(np.bool_),
    "hand_qpos": _spec(np.float64, HAND_JOINT_SHAPE),
    "hand_current": _spec(np.float64, HAND_JOINT_SHAPE),
    "hand_fingertip": _spec(np.float64, HAND_FINGERTIP_SHAPE),
    "hand_contact": _spec(np.float64, HAND_TACTILE_SUM_SHAPE),
    "hand_tactile_force": _spec(np.float64, HAND_TACTILE_FORCE_SHAPE),
    "hand_tactile_contact": _spec(np.bool_, HAND_CONTACT_SHAPE),
    "hand_tipboard_err": _spec(np.int32, HAND_JOINT_SHAPE),
    "hand_commboard_err": _spec(np.int32, HAND_JOINT_SHAPE),
    "hand_jointboard_err": _spec(np.int32, HAND_JOINT_SHAPE),
    "hand_connected": _spec(np.bool_),
    "hand_error_state": _spec(np.bool_),
    "hand_qpos_stale": _spec(np.bool_),
    "tactile_fresh": _spec(np.bool_),
    "tactile_source_monotonic_ns": _spec(np.int64),
    "tactile_calibrated": _spec(np.bool_),
    "tactile_unit_code": _spec(np.int64),
    "action_arm_joint_raw": _spec(np.float64, ARM_JOINT_SHAPE),
    "action_arm_joint": _spec(np.float64, ARM_JOINT_SHAPE),
    "action_hand_joint_raw": _spec(np.float64, HAND_JOINT_SHAPE),
    "action_hand_joint": _spec(np.float64, HAND_JOINT_SHAPE),
    "action_arm_ee": _spec(np.float64, (9,)),
    "target_eef_pos_raw": _spec(np.float64, (3,)),
    "target_eef_rot6d_raw": _spec(np.float64, (6,)),
    "target_pos_before_clamp": _spec(np.float64, (3,)),
    "observation_id": _spec(np.int64),
    "observation_anchor_monotonic_ns": _spec(np.int64),
    "arm_source_sequence": _spec(np.int64),
    "hand_source_sequence": _spec(np.int64),
    "vr_source_sequence": _spec(np.int64),
    "camera_source_sequence": _spec(np.int64),
    "arm_source_monotonic_ns": _spec(np.int64),
    "hand_source_monotonic_ns": _spec(np.int64),
    "vr_source_monotonic_ns": _spec(np.int64),
    "camera_source_monotonic_ns": _spec(np.int64),
    "arm_publish_monotonic_ns": _spec(np.int64),
    "hand_publish_monotonic_ns": _spec(np.int64),
    "vr_publish_monotonic_ns": _spec(np.int64),
    "camera_publish_monotonic_ns": _spec(np.int64),
    "observation_source_receive_monotonic_ns": _spec(np.uint64, (4,)),
    "observation_source_age_s": _spec(np.float64, (4,)),
    "observation_source_skew_s": _spec(np.float64, (4,)),
    "observation_history_valid_mask": _spec(np.bool_, (4, 1)),
    "observation_valid": _spec(np.bool_),
    "observation_skew_s": _spec(np.float64),
    "action_id": _spec(np.int64),
    "action_created_monotonic_ns": _spec(np.int64),
    "action_target_monotonic_ns": _spec(np.int64),
    "action_valid_until_monotonic_ns": _spec(np.int64),
    "flag_action_queued": _spec(np.bool_),
    "vr_wrist_pos": _spec(np.float64, (3,)),
    "vr_wrist_rot6d": _spec(np.float64, (6,)),
    "vr_landmarks": _spec(np.float64, (21, 3)),
    "head_quat_wxyz": _spec(np.float64, (4,)),
    "camera_health": _spec(np.int64),
    "flag_camera_fresh": _spec(np.bool_),
    "camera_frame_number": _spec(np.int64),
    "camera_ring_sequence": _spec(np.int64),
    "camera_device_timestamp_s": _spec(np.float64),
    "camera_capture_monotonic_s": _spec(np.float64),
    "camera_age_s": _spec(np.float64),
    "camera_generation": _spec(np.int64),
    "camera_clock_reset": _spec(np.bool_),
    "camera_duplicate": _spec(np.bool_),
    "camera_frame_gap": _spec(np.int64),
    "camera_backlog_s": _spec(np.float64),
    "pointcloud_valid_depth_ratio": _spec(np.float64),
    "flag_ik_ok": _spec(np.bool_),
    "flag_ik_attempted": _spec(np.bool_),
    "flag_retarget_ok": _spec(np.bool_),
    "flag_held": _spec(np.bool_),
    "flag_safety_reject": _spec(np.bool_),
    "flag_frame_status": _spec(np.int64),
    "tracking_error": _spec(np.float64),
    "ik_solve_time_ms": _spec(np.float64),
    "policy_map_time_ms": _spec(np.float64),
    "hand_retarget_time_ms": _spec(np.float64),
    "transition_check_time_ms": _spec(np.float64),
    "policy_compute_time_ms": _spec(np.float64),
}

CONDITIONAL_DATASET_SPECS_V17: dict[str, DatasetSpec] = {
    ARM_SENT_DATASET: _spec(np.float64, ARM_JOINT_SHAPE),
}

ALIGNMENT_DATASET_NAMES_V17 = frozenset(
    {
        "timestamp",
        "flag_sample_valid",
        "source_sample_index",
        "source_timestamp",
        "fill_reason",
    }
)
SOURCE_FRAME_DATASET_NAMES_V17 = (
    frozenset(BASE_DATASET_SPECS_V17)
    - ALIGNMENT_DATASET_NAMES_V17
    - ARM_WORKER_TELEMETRY_DATASETS_V17
)

# These values override existing fields; they do not extend the contract.
DIAGNOSTIC_TAIL_SHAPES_V17: dict[str, tuple[int, ...]] = {
    "tracking_error": (),
    "ik_solve_time_ms": (),
    "target_pos_before_clamp": (3,),
    "head_quat_wxyz": (4,),
    "target_eef_pos_raw": (3,),
    "target_eef_rot6d_raw": (6,),
    "action_hand_joint_raw": HAND_JOINT_SHAPE,
    "policy_map_time_ms": (),
    "hand_retarget_time_ms": (),
    "transition_check_time_ms": (),
    "policy_compute_time_ms": (),
}

# Additive metadata only: older schema-v17 readers may ignore every key here.
SEMANTIC_META_ATTRS_V17: dict[str, str | float | bool] = {
    "robot_world_frame": "xarm_base",
    "robot_world_equals_xarm_base": True,
    "arm_ee_frame": "xarm_base",
    "action_arm_ee_frame": "xarm_base",
    "hand_fingertip_frame": "xarm_base",
    "action_arm_joint_raw_validity_expression": ARM_RAW_ACTION_VALIDITY_EXPRESSION,
    "action_hand_joint_raw_validity_expression": HAND_RAW_ACTION_VALIDITY_EXPRESSION,
    "tactile_sdk_scale_factor": 0.1,
    "tactile_unit": "sdk_scaled_unknown_si",
    "tactile_si_unit_verified": False,
    "tactile_bias_semantics": "startup_software_bias_subtracted_when_available;see_tactile_calibrated",
    "tactile_contact_metric": "per_finger_l2_norm_hand_contact",
    "tactile_contact_threshold": 1.0,
    "raw_force_contact_threshold": 1.0,
    "tactile_contact_comparison": "strict_greater_than",
    "arm_tau_source": "xarm_get_joint_states_num_3_effort",
    "arm_tau_unit": "unknown",
    "arm_tau_si_unit_verified": False,
    "arm_tau_sensor_semantics": "current_estimated_effort_not_direct_torque_sensor",
}


def expected_source_frame_dataset_names_v17(*, arm_sent_stream: bool) -> frozenset[str]:
    """Return the exact keys accepted from one low-level recorder source frame."""

    if arm_sent_stream:
        return SOURCE_FRAME_DATASET_NAMES_V17 | frozenset({ARM_SENT_DATASET})
    return SOURCE_FRAME_DATASET_NAMES_V17


def validate_source_frame_keys_v17(
    keys: set[str], *, arm_sent_stream: bool
) -> tuple[str, ...]:
    """Validate exact writer-input keys before the timestamp buffer freezes them."""

    expected = expected_source_frame_dataset_names_v17(arm_sent_stream=arm_sent_stream)
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    errors: list[str] = []
    if missing:
        errors.append(f"missing source-frame datasets: {missing}")
    if unexpected:
        errors.append(f"unexpected source-frame datasets: {unexpected}")
    return tuple(errors)


def normalize_diagnostics_v17(
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Validate the diagnostic override set shared by schema v17 and v18."""

    if not diagnostics:
        return {}
    keys = set(diagnostics)
    allowed = set(DIAGNOSTIC_TAIL_SHAPES_V17)
    unexpected = keys - allowed
    if unexpected:
        reserved = sorted(
            unexpected
            & (set(BASE_DATASET_SPECS_V17) | set(CONDITIONAL_DATASET_SPECS_V17))
        )
        unknown = sorted(unexpected - set(reserved))
        details: list[str] = []
        if reserved:
            details.append(f"reserved dataset collisions={reserved}")
        if unknown:
            details.append(f"unsupported keys={unknown}")
        raise ValueError("episode diagnostics rejected: " + "; ".join(details))

    normalized: dict[str, np.ndarray] = {}
    for name in sorted(keys):
        try:
            value = np.asarray(diagnostics[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"episode diagnostic {name!r} must be float64-compatible"
            ) from exc
        expected_shape = DIAGNOSTIC_TAIL_SHAPES_V17[name]
        if value.shape != expected_shape:
            raise ValueError(
                f"episode diagnostic {name!r} has shape {value.shape}, expected {expected_shape}"
            )
        normalized[name] = value
    return normalized


def required_dataset_names(schema_version: int = EPISODE_SCHEMA_VERSION) -> frozenset[str]:
    """Return the datasets a ``data.h5`` of *schema_version* must contain.

    v18 drops the arm-worker telemetry datasets; v17 episodes keep them and
    stay readable (the extra datasets are accepted as known-but-optional).
    """
    if int(schema_version) <= 17:
        return frozenset(BASE_DATASET_SPECS_V17)
    return frozenset(BASE_DATASET_SPECS_V17) - ARM_WORKER_TELEMETRY_DATASETS_V17


def validate_data_layout_v17(
    dataset_shapes: Mapping[str, tuple[int, ...]],
    dataset_dtypes: Mapping[str, Any],
    *,
    frame_count: int,
    arm_sent_stream: bool,
    schema_version: int = EPISODE_SCHEMA_VERSION,
) -> tuple[str, ...]:
    """Return deterministic errors for one ``data.h5`` or in-memory layout.

    Unknown historical diagnostic datasets are accepted, but—like every
    writer-produced data stream—they must have a grid dimension equal to
    ``frame_count``.  Known fields additionally require their exact tail shape
    and dtype.  ``schema_version`` selects the required dataset set (v17 keeps
    the arm-worker telemetry datasets, v18 drops them).
    """

    errors: list[str] = []
    if frame_count < 0:
        errors.append(f"num_frames must be non-negative, got {frame_count}")

    names = set(dataset_shapes)
    missing = sorted(required_dataset_names(schema_version) - names)
    if missing:
        errors.append(f"missing required data.h5 datasets: {missing}")

    sent_present = ARM_SENT_DATASET in names
    if arm_sent_stream and not sent_present:
        errors.append(f"{ARM_SENT_MARKER}=True requires dataset {ARM_SENT_DATASET!r}")
    elif sent_present and not arm_sent_stream:
        errors.append(f"dataset {ARM_SENT_DATASET!r} requires {ARM_SENT_MARKER}=True")

    known_specs = dict(BASE_DATASET_SPECS_V17)
    known_specs.update(CONDITIONAL_DATASET_SPECS_V17)
    for name in sorted(names):
        shape = tuple(int(dim) for dim in dataset_shapes[name])
        if not shape:
            errors.append(
                f"data.h5 dataset {name!r} must be per-frame, got scalar shape"
            )
            continue
        if shape[0] != frame_count:
            errors.append(
                f"data.h5 dataset {name!r} length {shape[0]} != num_frames {frame_count}"
            )
        spec = known_specs.get(name)
        if spec is None:
            continue
        if shape[1:] != spec.tail_shape:
            errors.append(
                f"data.h5 dataset {name!r} tail shape {shape[1:]} != expected {spec.tail_shape}"
            )
        if name not in dataset_dtypes:
            errors.append(f"data.h5 dataset {name!r} has no dtype description")
            continue
        try:
            dtype = np.dtype(dataset_dtypes[name])
        except TypeError:
            errors.append(
                f"data.h5 dataset {name!r} has invalid dtype {dataset_dtypes[name]!r}"
            )
            continue
        if dtype != spec.dtype:
            errors.append(
                f"data.h5 dataset {name!r} dtype {dtype} != expected {spec.dtype}"
            )
    return tuple(errors)


__all__ = [
    "ALIGNMENT_DATASET_NAMES_V17",
    "ARM_RAW_ACTION_VALIDITY_EXPRESSION",
    "ARM_SENT_DATASET",
    "ARM_SENT_MARKER",
    "ARM_WORKER_TELEMETRY_DATASETS_V17",
    "SUPPORTED_EPISODE_SCHEMA_VERSIONS",
    "required_dataset_names",
    "BASE_DATASET_SPECS_V17",
    "CONDITIONAL_DATASET_SPECS_V17",
    "DIAGNOSTIC_TAIL_SHAPES_V17",
    "DatasetSpec",
    "EPISODE_SCHEMA_VERSION",
    "HAND_RAW_ACTION_VALIDITY_EXPRESSION",
    "SEMANTIC_META_ATTRS_V17",
    "SOURCE_FRAME_DATASET_NAMES_V17",
    "expected_source_frame_dataset_names_v17",
    "normalize_diagnostics_v17",
    "validate_data_layout_v17",
    "validate_source_frame_keys_v17",
]
