"""Schema-v21 contracts shared by episode writers and readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dexmani_real.ipc.schema import (
    ARM_JOINT_SHAPE,
    HAND_CONTACT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)

EPISODE_SCHEMA_VERSION = 21
# Only the current raw layout is supported at runtime.
SUPPORTED_EPISODE_SCHEMA_VERSIONS: frozenset[int] = frozenset({EPISODE_SCHEMA_VERSION})
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


DATASET_SPECS: dict[str, DatasetSpec] = {
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
    "camera_depth_frame_number": _spec(np.int64),
    "camera_color_frame_number": _spec(np.int64),
    "camera_ring_sequence": _spec(np.int64),
    "camera_depth_device_timestamp_s": _spec(np.float64),
    "camera_color_device_timestamp_s": _spec(np.float64),
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

CONDITIONAL_DATASET_SPECS: dict[str, DatasetSpec] = {
    ARM_SENT_DATASET: _spec(np.float64, ARM_JOINT_SHAPE),
}

CAMERA_TIMING_DATASET_SPECS: dict[str, DatasetSpec] = {
    "camera_wait_return_monotonic_ns": _spec(np.int64),
    "camera_payload_ready_monotonic_ns": _spec(np.int64),
    "camera_depth_timestamp_domain": _spec(np.int64),
    "camera_color_timestamp_domain": _spec(np.int64),
    "camera_delivery_delay_above_floor_s": _spec(np.float64),
}

ALIGNMENT_DATASET_NAMES = frozenset(
    {
        "timestamp",
        "flag_sample_valid",
        "source_sample_index",
        "source_timestamp",
        "fill_reason",
    }
)
SOURCE_FRAME_DATASET_NAMES = frozenset(
    DATASET_SPECS
) - ALIGNMENT_DATASET_NAMES | frozenset(CAMERA_TIMING_DATASET_SPECS)

DIAGNOSTIC_TAIL_SHAPES: dict[str, tuple[int, ...]] = {
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

SEMANTIC_META_ATTRS: dict[str, str | float | bool] = {
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
    "camera_payload_mode": "native_rgbd",
    "camera_pair_source_monotonic_ns_semantics": "minimum_of_depth_and_color_mapped_source_times",
    "camera_wait_return_monotonic_ns_semantics": "host_monotonic_immediately_after_frame_queue_wait_for_frame_return",
    "camera_payload_ready_monotonic_ns_semantics": "host_monotonic_after_owned_native_rgb_depth_copies",
    "camera_depth_device_timestamp_s_semantics": "native_depth_frame_device_timestamp",
    "camera_color_device_timestamp_s_semantics": "native_color_frame_device_timestamp_or_nan_when_absent",
    "camera_generation_semantics": "depth_stream_clock_mapper_generation",
    "camera_clock_reset_semantics": "depth_stream_clock_mapper_reset",
    "camera_duplicate_semantics": "depth_stream_duplicate_detection",
    "camera_frame_gap_semantics": "depth_stream_frame_number_gap",
    "camera_backlog_s_semantics": "host_wait_return_minus_pair_oldest_mapped_source_time",
}


def expected_source_frame_dataset_names(*, arm_sent_stream: bool) -> frozenset[str]:
    """Return the exact keys accepted from one recorder source frame."""
    return SOURCE_FRAME_DATASET_NAMES | (
        frozenset({ARM_SENT_DATASET}) if arm_sent_stream else frozenset()
    )


def compute_episode_quality_metrics(
    datasets: dict[str, Any],
    *,
    frame_count: int,
    control_hz: float,
) -> dict[str, int]:
    """Summarize persisted frame flags."""
    if frame_count < 0 or not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("invalid episode quality dimensions")
    if frame_count == 0:
        return {
            "ik_hold_frame_count": 0,
            "camera_invalid_frame_count": 0,
            "observation_invalid_frame_count": 0,
            "sample_invalid_frame_count": 0,
            "safety_reject_frame_count": 0,
            "command_quiescence_count": 0,
        }

    def read_bool_dataset(name: str) -> np.ndarray:
        if name not in datasets:
            raise KeyError(f"required quality dataset missing: {name}")
        return np.asarray(datasets[name][:frame_count], dtype=bool)

    held = read_bool_dataset("flag_held")
    ik_ok = read_bool_dataset("flag_ik_ok")
    observation_valid = read_bool_dataset("observation_valid")
    camera_fresh = read_bool_dataset("flag_camera_fresh")
    sample_valid = read_bool_dataset("flag_sample_valid")
    safety_reject = read_bool_dataset("flag_safety_reject")
    if "timestamp" not in datasets:
        raise KeyError("required quality dataset missing: timestamp")
    timestamps = np.asarray(datasets["timestamp"][:frame_count], dtype=np.float64)
    return {
        "ik_hold_frame_count": int(np.count_nonzero(held & ~ik_ok)),
        "camera_invalid_frame_count": int(np.count_nonzero(~camera_fresh)),
        "observation_invalid_frame_count": int(np.count_nonzero(~observation_valid)),
        "sample_invalid_frame_count": int(np.count_nonzero(~sample_valid)),
        "safety_reject_frame_count": int(np.count_nonzero(safety_reject)),
        "command_quiescence_count": int(
            np.count_nonzero(np.diff(timestamps) > (1.5 / control_hz))
        ),
    }


def validate_source_frame_keys(
    keys: set[str], *, arm_sent_stream: bool
) -> tuple[str, ...]:
    """Validate exact writer-input keys before timestamp alignment."""
    expected = expected_source_frame_dataset_names(arm_sent_stream=arm_sent_stream)
    missing, unexpected = sorted(expected - keys), sorted(keys - expected)
    return tuple(
        (["missing source-frame datasets: {}".format(missing)] if missing else [])
        + (
            ["unexpected source-frame datasets: {}".format(unexpected)]
            if unexpected
            else []
        )
    )


def normalize_diagnostics(
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Validate the diagnostic override set."""

    if not diagnostics:
        return {}
    keys = set(diagnostics)
    allowed = set(DIAGNOSTIC_TAIL_SHAPES)
    unexpected = keys - allowed
    if unexpected:
        reserved = sorted(
            unexpected & (set(DATASET_SPECS) | set(CONDITIONAL_DATASET_SPECS))
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
        expected_shape = DIAGNOSTIC_TAIL_SHAPES[name]
        if value.shape != expected_shape:
            raise ValueError(
                f"episode diagnostic {name!r} has shape {value.shape}, expected {expected_shape}"
            )
        normalized[name] = value
    return normalized


def required_dataset_names() -> frozenset[str]:
    """Return the datasets required in every raw v21 data.h5."""
    return frozenset(DATASET_SPECS) | frozenset(CAMERA_TIMING_DATASET_SPECS)


def validate_data_layout(
    dataset_shapes: Mapping[str, tuple[int, ...]],
    dataset_dtypes: Mapping[str, Any],
    *,
    frame_count: int,
    arm_sent_stream: bool,
) -> tuple[str, ...]:
    """Return deterministic validation errors for a raw v21 layout."""
    errors: list[str] = []
    if frame_count < 0:
        errors.append("num_frames must be non-negative, got {}".format(frame_count))
    names = set(dataset_shapes)
    missing = sorted(required_dataset_names() - names)
    if missing:
        errors.append("missing required data.h5 datasets: {}".format(missing))
    sent_present = ARM_SENT_DATASET in names
    if arm_sent_stream and not sent_present:
        errors.append(
            "{}=True requires dataset {!r}".format(ARM_SENT_MARKER, ARM_SENT_DATASET)
        )
    elif sent_present and not arm_sent_stream:
        errors.append(
            "dataset {!r} requires {}=True".format(ARM_SENT_DATASET, ARM_SENT_MARKER)
        )
    known_specs = {
        **DATASET_SPECS,
        **CAMERA_TIMING_DATASET_SPECS,
        **CONDITIONAL_DATASET_SPECS,
    }
    for name in sorted(names):
        shape = tuple(int(dim) for dim in dataset_shapes[name])
        if not shape:
            errors.append(
                "data.h5 dataset {!r} must be per-frame, got scalar shape".format(name)
            )
            continue
        if shape[0] != frame_count:
            errors.append(
                "data.h5 dataset {!r} length {} != num_frames {}".format(
                    name, shape[0], frame_count
                )
            )
        spec = known_specs.get(name)
        if spec is None:
            errors.append("unexpected data.h5 dataset: {!r}".format(name))
            continue
        if shape[1:] != spec.tail_shape:
            errors.append(
                "data.h5 dataset {!r} tail shape {} != expected {}".format(
                    name, shape[1:], spec.tail_shape
                )
            )
        if name not in dataset_dtypes:
            errors.append("data.h5 dataset {!r} has no dtype description".format(name))
            continue
        try:
            dtype = np.dtype(dataset_dtypes[name])
        except TypeError:
            errors.append(
                "data.h5 dataset {!r} has invalid dtype {!r}".format(
                    name, dataset_dtypes[name]
                )
            )
            continue
        if dtype != spec.dtype:
            errors.append(
                "data.h5 dataset {!r} dtype {} != expected {}".format(
                    name, dtype, spec.dtype
                )
            )
    return tuple(errors)


__all__ = [
    "ALIGNMENT_DATASET_NAMES",
    "ARM_RAW_ACTION_VALIDITY_EXPRESSION",
    "ARM_SENT_DATASET",
    "ARM_SENT_MARKER",
    "CAMERA_TIMING_DATASET_SPECS",
    "CONDITIONAL_DATASET_SPECS",
    "DATASET_SPECS",
    "DIAGNOSTIC_TAIL_SHAPES",
    "DatasetSpec",
    "EPISODE_SCHEMA_VERSION",
    "HAND_RAW_ACTION_VALIDITY_EXPRESSION",
    "SEMANTIC_META_ATTRS",
    "SOURCE_FRAME_DATASET_NAMES",
    "SUPPORTED_EPISODE_SCHEMA_VERSIONS",
    "compute_episode_quality_metrics",
    "expected_source_frame_dataset_names",
    "normalize_diagnostics",
    "required_dataset_names",
    "validate_data_layout",
    "validate_source_frame_keys",
]
