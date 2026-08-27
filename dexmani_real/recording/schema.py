"""Raw episode contracts shared by episode writers, finalizers, and readers."""

from __future__ import annotations

import json
import re
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

# v23 is retained as an explicitly opted-in legacy format.  New recordings
# carry the sidecar manifest and the stricter semantic checks below.
LEGACY_EPISODE_SCHEMA_VERSION = 23
EPISODE_SCHEMA_VERSION = 24
RAW_MANIFEST_VERSION = 1
RAW_MANIFEST_VERSION_ATTR = "raw_manifest_version"
RAW_DEPTH_SHA256_ATTR = "depth_sha256"
RAW_RGB_SHA256_ATTR = "rgb_sha256"
RAW_MEMBER_SHA256_JSON_ATTR = "raw_member_sha256_json"
_RAW_MEMBER_ATTRS = {
    "depth.h5": RAW_DEPTH_SHA256_ATTR,
    "rgb.mp4": RAW_RGB_SHA256_ATTR,
}
RAW_MEMBER_NAMES = tuple(_RAW_MEMBER_ATTRS)
CAMERA_HEALTH_TAXONOMY: dict[int, str] = {
    0: "OK",
    1: "CLOCK_RESET",
    2: "DUPLICATE",
    3: "FRAME_GAP",
    4: "DELIVERY_DELAY_ABOVE_FLOOR",
}
CAMERA_HEALTH_TAXONOMY_JSON = json.dumps(
    {str(key): value for key, value in CAMERA_HEALTH_TAXONOMY.items()},
    sort_keys=True,
    separators=(",", ":"),
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
    # These fields are the policy observation, not the latest control
    # feedback. They pair robot state with camera/point-cloud source time.
    "policy_observation_arm_qpos": _spec(np.float64, ARM_JOINT_SHAPE),
    "policy_observation_hand_qpos": _spec(np.float64, HAND_JOINT_SHAPE),
    "policy_observation_reference_monotonic_ns": _spec(np.int64),
    "policy_observation_arm_source_sequence": _spec(np.int64),
    "policy_observation_hand_source_sequence": _spec(np.int64),
    "policy_observation_arm_source_monotonic_ns": _spec(np.int64),
    "policy_observation_hand_source_monotonic_ns": _spec(np.int64),
    "policy_observation_arm_publish_monotonic_ns": _spec(np.int64),
    "policy_observation_hand_publish_monotonic_ns": _spec(np.int64),
    "policy_observation_valid": _spec(np.bool_),
    "policy_observation_skew_s": _spec(np.float64),
    # SDK acknowledgement of an exact hand endpoint, not physical
    # convergence. It makes endpoint application auditable without
    # relabelling action_hand_joint as a servo-tick command.
    "hand_accepted_target_action_id": _spec(np.int64),
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

_RAW_DATASET_SPECS = {**DATASET_SPECS, **CAMERA_TIMING_DATASET_SPECS}

ALIGNMENT_DATASET_NAMES = frozenset(
    {
        "timestamp",
        "flag_sample_valid",
        "source_sample_index",
        "source_timestamp",
        "fill_reason",
    }
)
SOURCE_FRAME_DATASET_NAMES = frozenset(_RAW_DATASET_SPECS) - ALIGNMENT_DATASET_NAMES

_DIAGNOSTIC_DATASET_NAMES = (
    "tracking_error",
    "ik_solve_time_ms",
    "target_pos_before_clamp",
    "head_quat_wxyz",
    "target_eef_pos_raw",
    "target_eef_rot6d_raw",
    "action_hand_joint_raw",
    "policy_map_time_ms",
    "hand_retarget_time_ms",
    "transition_check_time_ms",
    "policy_compute_time_ms",
)
DIAGNOSTIC_TAIL_SHAPES: dict[str, tuple[int, ...]] = {
    name: DATASET_SPECS[name].tail_shape for name in _DIAGNOSTIC_DATASET_NAMES
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
    "camera_payload_mode": "depth_to_color_aligned_rgbd",
    "camera_health_taxonomy_json": CAMERA_HEALTH_TAXONOMY_JSON,
    "camera_pair_source_monotonic_ns_semantics": "minimum_of_depth_and_color_mapped_source_times",
    "camera_wait_return_monotonic_ns_semantics": "host_monotonic_immediately_after_frame_queue_wait_for_frame_return",
    "camera_payload_ready_monotonic_ns_semantics": "host_monotonic_after_owned_native_rgb_depth_copies",
    "camera_depth_device_timestamp_s_semantics": "native_depth_frame_device_timestamp",
    "camera_color_device_timestamp_s_semantics": "native_color_frame_device_timestamp_or_nan_when_absent",
    "camera_generation_semantics": "depth_stream_clock_mapper_generation",
    "camera_clock_reset_semantics": "depth_stream_clock_mapper_reset",
    "camera_duplicate_semantics": "depth_stream_duplicate_detection",
    # The first attr defines only the numeric telemetry value.  Admission is
    # deliberately separate: a recovered current frame may remain usable.
    "camera_frame_gap_semantics": (
        "depth_stream_frame_number_gap=max(0,current_depth_frame_number-"
        "previous_depth_frame_number-1);reset_or_duplicate=>0"
    ),
    "camera_frame_gap_admission_policy": (
        "telemetry_only;recovered_current_frame_is_eligible_when_payload_and_"
        "timestamps_are_fresh"
    ),
    "source_timestamp_semantics": (
        "selected_recorder_source_row_logical_grid_anchor_s;SOURCE=timestamp;"
        "CAUSAL_HOLD_LAST=repeats_previous_source_anchor;NaN_on_LEADING_PLACEHOLDER"
    ),
    "source_sample_index_semantics": (
        "recorder_source_row_index;strictly_increasing_on_SOURCE;holds_repeat_"
        "the_latest_source;leading_placeholder=-1"
    ),
    "camera_backlog_s_semantics": "host_wait_return_minus_pair_oldest_mapped_source_time",
    "policy_observation_reference_semantics": "camera_source_monotonic_ns",
    "policy_observation_state_alignment": "newest_state_source_at_or_before_camera_source;state_publish_at_or_before_grid_anchor",
    "policy_observation_skew_semantics": "camera_source_monotonic_ns_minus_oldest_selected_robot_state_source_monotonic_ns",
    "action_hand_joint_semantics": "policy_grid_target_rate_limited_from_previous_published_endpoint_or_initial_feedback",
    "hand_accepted_target_action_id_semantics": "xhand_sdk_accepted_exact_target_action_id_not_physical_convergence",
}

# Metadata written by the recorder itself is a schema boundary.  Caller
# extensions are still supported, but may not replace a fixed value or a
# reserved provenance namespace.  Keep this set here so the writer and any
# future metadata ingress use the same ownership rule.
RAW_RESERVED_META_KEYS = frozenset(
    set(SEMANTIC_META_ATTRS)
    | {
        "schema_version",
        "task_label",
        "operator",
        "control_hz",
        "fps",
        "resolved_config_sha256",
        ARM_SENT_MARKER,
        "skip_initial_frames",
        "camera_serial",
        "camera_name",
        "camera_type",
        "camera_depth_intrinsics",
        "camera_depth_width",
        "camera_depth_height",
        "camera_depth_distortion_model",
        "camera_depth_distortion_coeffs",
        "camera_color_intrinsics",
        "camera_color_width",
        "camera_color_height",
        "camera_color_distortion_model",
        "camera_color_distortion_coeffs",
        "camera_T_color_from_depth",
        "camera_T_xarm_base_from_color",
        "camera_T_xarm_base_from_depth",
        "camera_T_eef_from_depth",
        "camera_calibration_source_optical_frame",
        "camera_geometry_frame_semantics",
        "depth_scale",
        "camera_writer_queue_size",
        "camera_encoding_codec",
        "camera_encoding_crf",
        "camera_encoding_preset",
        "camera_encoding_pixel_format",
        "camera_encoding_width",
        "camera_encoding_height",
        "camera_encoding_fps",
        "camera_depth_storage",
        "camera_depth_payload_semantics",
        "camera_stream_frames",
        "camera_writer_error",
        "camera_writer_queue_high_watermark",
        "camera_writer_queue_capacity",
        "camera_writer_close_s",
        "camera_encode_p50_s",
        "camera_encode_p95_s",
        "camera_encode_p99_s",
        "camera_encode_max_s",
        "camera_hdf5_p50_s",
        "camera_hdf5_p95_s",
        "camera_hdf5_p99_s",
        "camera_hdf5_max_s",
        "duration",
        "wall_duration_s",
        "grid_duration_s",
        "grid_dt_s",
        "non_sampled_duration_s",
        "wall_fps",
        "num_frames",
        "success",
        "truncated",
        "stop_reason",
        "min_frames_met",
        "has_camera",
        "has_timestamps",
        "ik_hold_frame_count",
        "camera_invalid_frame_count",
        "observation_invalid_frame_count",
        "sample_invalid_frame_count",
        "safety_reject_frame_count",
        "command_quiescence_count",
        RAW_MANIFEST_VERSION_ATTR,
        RAW_DEPTH_SHA256_ATTR,
        RAW_RGB_SHA256_ATTR,
        RAW_MEMBER_SHA256_JSON_ATTR,
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAMERA_HEALTH_VALUES = frozenset(CAMERA_HEALTH_TAXONOMY)
_FLOAT_ROW_SENTINEL_EXCEPTIONS = frozenset(
    {
        "timestamp",
        "source_timestamp",
        "pointcloud_valid_depth_ratio",
        "observation_source_age_s",
        "observation_source_skew_s",
        "observation_skew_s",
        "policy_observation_skew_s",
    }
)
_SOURCE_STREAM_NAMES = ("arm", "hand", "vr", "camera")


def validate_camera_metadata_keys(
    camera_metadata: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return errors for metadata that would overwrite recorder-owned attrs."""
    if camera_metadata is None:
        return ()
    if not isinstance(camera_metadata, Mapping):
        return ("camera_metadata must be a mapping",)
    errors: list[str] = []
    for key in sorted(camera_metadata, key=str):
        if not isinstance(key, str) or not key:
            errors.append("camera_metadata keys must be non-empty strings")
            continue
        if key in RAW_RESERVED_META_KEYS or key.startswith("provenance_"):
            errors.append(f"camera_metadata key {key!r} is recorder-reserved")
    return tuple(errors)


def _attr_scalar(value: Any) -> Any:
    """Unwrap NumPy/HDF5 scalar attributes without changing their type."""
    return value.item() if isinstance(value, np.generic) else value


def validate_semantic_meta_attrs(
    attrs: Mapping[str, Any], *, legacy: bool = False
) -> tuple[str, ...]:
    """Validate fixed semantic attrs against this schema's single definition.

    v23 had a different ``camera_frame_gap_semantics`` value and no explicit
    source timestamp/index definitions, so an explicitly opted-in legacy read
    intentionally skips these v24-only equality checks.
    """
    if legacy:
        return ()
    errors: list[str] = []
    for key, expected in SEMANTIC_META_ATTRS.items():
        if key not in attrs:
            errors.append(f"missing fixed semantic metadata attr: {key}")
            continue
        actual = _attr_scalar(attrs[key])
        if isinstance(expected, bool):
            if not isinstance(actual, (bool, np.bool_)) or bool(actual) != expected:
                errors.append(f"metadata attr {key!r} != fixed value {expected!r}")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
                    errors.append(f"metadata attr {key!r} != fixed value {expected!r}")
            except (TypeError, ValueError):
                errors.append(f"metadata attr {key!r} is not numeric")
        else:
            try:
                matches = bool(actual == expected)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                errors.append(f"metadata attr {key!r} != fixed value {expected!r}")
    return tuple(errors)


def validate_raw_member_hashes(
    attrs: Mapping[str, Any],
    member_sha256: Mapping[str, str],
) -> tuple[str, ...]:
    """Validate the v24 sidecar manifest against hashes computed by a caller."""
    errors: list[str] = []
    try:
        version = int(_attr_scalar(attrs.get(RAW_MANIFEST_VERSION_ATTR, -1)))
    except (TypeError, ValueError):
        version = -1
    if version != RAW_MANIFEST_VERSION:
        errors.append(
            f"{RAW_MANIFEST_VERSION_ATTR} must be {RAW_MANIFEST_VERSION}, got {version}"
        )
    expected_member_names = set(RAW_MEMBER_NAMES)
    parsed_json_manifest: dict[str, Any] | None = None
    if RAW_MEMBER_SHA256_JSON_ATTR not in attrs:
        errors.append(f"metadata attr {RAW_MEMBER_SHA256_JSON_ATTR!r} is missing")
    else:
        try:
            decoded = json.loads(_attr_scalar(attrs[RAW_MEMBER_SHA256_JSON_ATTR]))
            if not isinstance(decoded, dict):
                raise ValueError("manifest JSON must be an object")
            actual_member_names = set(decoded)
            if actual_member_names != expected_member_names:
                missing = sorted(expected_member_names - actual_member_names)
                extra = sorted(actual_member_names - expected_member_names)
                errors.append(
                    f"metadata attr {RAW_MEMBER_SHA256_JSON_ATTR!r} must contain "
                    f"exactly {sorted(expected_member_names)!r}; "
                    f"missing={missing!r}, extra={extra!r}"
                )
            else:
                parsed_json_manifest = decoded
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(
                f"metadata attr {RAW_MEMBER_SHA256_JSON_ATTR!r} is invalid: {exc}"
            )
    for member_name, attr_name in _RAW_MEMBER_ATTRS.items():
        if attr_name not in attrs:
            errors.append(f"metadata attr {attr_name!r} is missing")
            expected = ""
        else:
            expected = str(_attr_scalar(attrs[attr_name])).lower()
        if not _SHA256_RE.fullmatch(expected):
            errors.append(f"metadata attr {attr_name!r} is not a SHA-256 digest")
        actual = str(member_sha256.get(member_name, "")).lower()
        if not _SHA256_RE.fullmatch(actual):
            errors.append(f"computed hash for {member_name!r} is invalid")
        elif actual != expected:
            errors.append(
                f"{member_name} SHA-256 mismatch: metadata={expected}, computed={actual}"
            )
        if parsed_json_manifest is not None:
            json_expected = str(parsed_json_manifest.get(member_name, "")).lower()
            if not _SHA256_RE.fullmatch(json_expected):
                errors.append(
                    f"manifest JSON hash for {member_name!r} is not a SHA-256 digest"
                )
            elif json_expected != expected:
                errors.append(
                    f"manifest JSON hash for {member_name!r} disagrees with its fixed attr"
                )
    return tuple(errors)


def _read_semantic_dataset(
    datasets: Mapping[str, Any], name: str, frame_count: int
) -> np.ndarray | None:
    """Read one semantic dataset from HDF5 or an in-memory mapping."""
    if name not in datasets:
        return None
    value = datasets[name]
    try:
        result = np.asarray(value[:frame_count] if hasattr(value, "shape") else value)
    except (TypeError, ValueError, OSError):
        return None
    return result


def _row_all_finite(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.isfinite(values)
    return np.all(np.isfinite(values.reshape(values.shape[0], -1)), axis=1)


def _row_all_nan(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.isnan(values)
    return np.all(np.isnan(values.reshape(values.shape[0], -1)), axis=1)


def _describe_bad_rows(mask: np.ndarray, *, limit: int = 4) -> str:
    rows = np.flatnonzero(mask)[:limit].tolist()
    suffix = "..." if int(np.count_nonzero(mask)) > limit else ""
    return f"rows={rows}{suffix}"


def _stack_source_fields(
    values: Mapping[str, np.ndarray], suffix: str
) -> np.ndarray | None:
    """Stack one scalar provenance field in the fixed arm/hand/VR/camera order."""
    names = tuple(f"{source}_{suffix}" for source in _SOURCE_STREAM_NAMES)
    if not all(name in values for name in names):
        return None
    return np.column_stack(
        [values[name].astype(np.int64, copy=False) for name in names]
    )


def validate_raw_semantics(
    datasets: Mapping[str, Any],
    *,
    frame_count: int,
    attrs: Mapping[str, Any] | None = None,
    legacy: bool = False,
    source_timestamp_tolerance_s: float = 1e-7,
) -> tuple[str, ...]:
    """Return deterministic errors for persisted raw-v24 semantic values.

    This validator is intentionally shared by finalization and reading.  All
    floating-point rows are either fully finite or an explicit all-NaN
    sentinel; fields whose contract permits a mixed row (source history ages
    and skews) are checked against their validity mask instead.  Leading grid
    placeholders are the only legal ``source_*`` NaN rows; causal holds repeat
    the preceding source identity.
    """
    errors: list[str] = []
    if frame_count < 0:
        return (f"num_frames must be non-negative, got {frame_count}",)
    if (
        not np.isfinite(source_timestamp_tolerance_s)
        or source_timestamp_tolerance_s < 0
    ):
        return ("source timestamp tolerance must be finite and non-negative",)

    values: dict[str, np.ndarray] = {}
    semantic_names = set(_RAW_DATASET_SPECS)
    if ARM_SENT_DATASET in datasets:
        semantic_names.add(ARM_SENT_DATASET)
    for name in sorted(semantic_names):
        array = _read_semantic_dataset(datasets, name, frame_count)
        if array is None:
            errors.append(f"missing or unreadable semantic dataset: {name}")
            continue
        if array.ndim == 0 or array.shape[0] != frame_count:
            errors.append(
                f"semantic dataset {name!r} shape {array.shape} does not start with num_frames={frame_count}"
            )
            continue
        values[name] = array

    if errors:
        return tuple(errors)

    for name, spec in _RAW_DATASET_SPECS.items():
        if spec.dtype != np.dtype(np.bool_):
            continue
        if values[name].dtype != np.dtype(np.bool_):
            errors.append(f"{name} must use bool dtype")
    if errors:
        return tuple(errors)

    timestamps = values["timestamp"].astype(np.float64, copy=False)
    if not np.all(np.isfinite(timestamps)):
        errors.append("timestamp contains non-finite values")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
        errors.append("timestamp must be strictly increasing")

    sample_valid = values["flag_sample_valid"].astype(bool, copy=False)
    held = values["flag_held"].astype(bool, copy=False)
    if not np.any(sample_valid):
        errors.append("raw episode must contain at least one SOURCE row")
    source_indices = values["source_sample_index"].astype(np.int64, copy=False)
    source_timestamps = values["source_timestamp"].astype(np.float64, copy=False)
    fill_reasons = values["fill_reason"].astype(np.uint8, copy=False)
    is_source = fill_reasons == 0
    is_hold = fill_reasons == 1
    is_leading = fill_reasons == 2
    if np.any(~np.isin(fill_reasons, (0, 1, 2))):
        errors.append("fill_reason contains an unknown enum value")
    if np.any(is_source != sample_valid):
        errors.append("flag_sample_valid must equal fill_reason==SOURCE")
    source_defined = source_indices >= 0
    source_time_finite = np.isfinite(source_timestamps)
    if np.any((is_source | is_hold) != source_defined):
        errors.append("source_sample_index sentinel disagrees with fill_reason")
    if np.any((is_source | is_hold) != source_time_finite):
        errors.append("source_timestamp sentinel disagrees with fill_reason")
    if np.any(is_leading & (source_defined | source_time_finite | sample_valid)):
        errors.append(
            "leading placeholders must use source index=-1, source timestamp=NaN"
        )
    if np.any(
        source_defined & (source_timestamps > timestamps + source_timestamp_tolerance_s)
    ):
        errors.append("source_timestamp is later than its causal grid timestamp")
    if np.any(source_defined & (source_timestamps < 0.0)):
        errors.append("source_timestamp must be non-negative when defined")
    if not legacy and np.any(
        is_source
        & ~np.isclose(
            source_timestamps,
            timestamps,
            rtol=0.0,
            atol=source_timestamp_tolerance_s,
        )
    ):
        errors.append("SOURCE source_timestamp must equal its logical grid timestamp")

    source_rows = np.flatnonzero(is_source)
    if source_rows.size > 1:
        if np.any(np.diff(source_indices[source_rows]) <= 0):
            errors.append(
                "source_sample_index must strictly increase on SOURCE rows "
                + _describe_bad_rows(
                    np.r_[False, np.diff(source_indices[source_rows]) <= 0]
                )
            )
        if np.any(np.diff(source_timestamps[source_rows]) <= 0.0):
            errors.append("source_timestamp must strictly increase on SOURCE rows")
    previous_source_row = -1
    for row in range(frame_count):
        if is_source[row]:
            previous_source_row = row
        elif is_hold[row]:
            if previous_source_row < 0:
                errors.append(f"CAUSAL_HOLD_LAST row {row} has no preceding source")
            elif source_indices[row] != source_indices[previous_source_row]:
                errors.append(
                    f"CAUSAL_HOLD_LAST row {row} does not repeat its source index"
                )
            elif not np.isclose(
                source_timestamps[row],
                source_timestamps[previous_source_row],
                rtol=0.0,
                atol=source_timestamp_tolerance_s,
            ):
                errors.append(
                    f"CAUSAL_HOLD_LAST row {row} does not repeat its source timestamp"
                )

    # Float rows use all-finite or all-NaN sentinels.  Mixed rows are almost
    # always partial writes and are not an allowed representation.
    # These float fields have dedicated masks or independent sentinel rules.
    float_row_names = tuple(
        name
        for name, spec in _RAW_DATASET_SPECS.items()
        if np.issubdtype(spec.dtype, np.floating)
        and name not in _FLOAT_ROW_SENTINEL_EXCEPTIONS
    )
    for name in float_row_names:
        array = values[name].astype(np.float64, copy=False)
        finite = _row_all_finite(array)
        all_nan = _row_all_nan(array)
        if np.any(~finite & ~all_nan):
            errors.append(
                f"{name} has partial/non-finite rows "
                + _describe_bad_rows(~finite & ~all_nan)
            )
    if ARM_SENT_DATASET in datasets:
        arm_sent = values[ARM_SENT_DATASET]
        if arm_sent is None or arm_sent.shape[1:] != ARM_JOINT_SHAPE:
            errors.append(f"{ARM_SENT_DATASET} must have tail shape {ARM_JOINT_SHAPE}")
        else:
            arm_sent = arm_sent.astype(np.float64, copy=False)
            arm_sent_finite = _row_all_finite(arm_sent)
            arm_sent_nan = _row_all_nan(arm_sent)
            if np.any(~arm_sent_finite & ~arm_sent_nan):
                errors.append(
                    f"{ARM_SENT_DATASET} has partial/non-finite rows "
                    + _describe_bad_rows(~arm_sent_finite & ~arm_sent_nan)
                )
            if np.any(sample_valid & ~held & ~arm_sent_finite):
                errors.append(f"active non-held rows require finite {ARM_SENT_DATASET}")

    # Numeric camera telemetry is non-negative where present.  Fresh rows must
    # carry complete finite telemetry and a ratio in the closed unit interval.
    camera_health = values["camera_health"].astype(np.int64, copy=False)
    if np.any(~np.isin(camera_health, tuple(_CAMERA_HEALTH_VALUES))):
        errors.append("camera_health contains an unknown enum value")
    generation = values["camera_generation"].astype(np.int64, copy=False)
    if np.any(generation < 0):
        errors.append("camera_generation must be non-negative")
    # Zero is the explicit "no camera source" sentinel.  Ignore those rows
    # when checking the monotonic clock-mapper generation sequence.
    defined_generations = generation[generation > 0]
    if defined_generations.size > 1 and np.any(np.diff(defined_generations) < 0):
        errors.append("camera_generation must not move backwards")
    for name in (
        "camera_depth_frame_number",
        "camera_color_frame_number",
        "camera_ring_sequence",
        "camera_frame_gap",
    ):
        if np.any(values[name].astype(np.int64, copy=False) < 0):
            errors.append(f"{name} must be non-negative")
    for name in (
        "camera_wait_return_monotonic_ns",
        "camera_payload_ready_monotonic_ns",
    ):
        if np.any(values[name].astype(np.int64, copy=False) < 0):
            errors.append(f"{name} must be non-negative")
    for name in ("camera_depth_timestamp_domain", "camera_color_timestamp_domain"):
        domain = values[name].astype(np.int64, copy=False)
        if np.any((domain < 0) | (domain > 255)):
            errors.append(f"{name} must be in [0, 255]")
    camera_wait_ns = values["camera_wait_return_monotonic_ns"].astype(
        np.int64, copy=False
    )
    camera_ready_ns = values["camera_payload_ready_monotonic_ns"].astype(
        np.int64, copy=False
    )
    timing_present = (camera_wait_ns > 0) | (camera_ready_ns > 0)
    if np.any(timing_present & ((camera_wait_ns <= 0) | (camera_ready_ns <= 0))):
        errors.append("camera wait/payload timestamps must be both present or absent")
    if np.any(
        (camera_wait_ns > 0)
        & (camera_ready_ns > 0)
        & (camera_ready_ns < camera_wait_ns)
    ):
        errors.append("camera payload readiness cannot precede frame wait return")
    for name in (
        "camera_depth_device_timestamp_s",
        "camera_color_device_timestamp_s",
        "camera_age_s",
        "camera_backlog_s",
        "camera_delivery_delay_above_floor_s",
    ):
        array = values[name].astype(np.float64, copy=False)
        finite = np.isfinite(array)
        if np.any(finite & (array < 0.0)):
            errors.append(f"{name} must be non-negative when present")
    ratios = values["pointcloud_valid_depth_ratio"].astype(np.float64, copy=False)
    ratio_finite = np.isfinite(ratios)
    if np.any(ratio_finite & ((ratios < 0.0) | (ratios > 1.0))):
        errors.append("pointcloud_valid_depth_ratio must be in [0, 1]")
    camera_fresh = values["flag_camera_fresh"].astype(bool, copy=False)
    camera_required = camera_fresh & (
        ~np.isfinite(values["camera_depth_device_timestamp_s"])
        | ~np.isfinite(values["camera_age_s"])
        | ~np.isfinite(values["camera_backlog_s"])
        | ~np.isfinite(values["camera_delivery_delay_above_floor_s"])
        | ~ratio_finite
    )
    if np.any(camera_required):
        errors.append(
            "fresh camera rows require finite timestamps, latency, and depth ratio"
        )
    if np.any(camera_fresh & ((camera_wait_ns <= 0) | (camera_ready_ns <= 0))):
        errors.append("fresh camera rows require positive wait/payload timestamps")
    if np.any(camera_fresh & (generation <= 0)):
        errors.append("fresh camera rows require a positive camera_generation")
    if np.any(camera_fresh & (camera_health != 0)):
        errors.append("fresh camera rows require camera_health=OK")
    if np.any(
        camera_fresh & (values["camera_clock_reset"] | values["camera_duplicate"])
    ):
        errors.append("fresh camera rows cannot carry reset/duplicate flags")

    observation_ids = values["observation_id"].astype(np.int64, copy=False)
    action_ids = values["action_id"].astype(np.int64, copy=False)
    action_created = values["action_created_monotonic_ns"].astype(np.int64, copy=False)
    action_target = values["action_target_monotonic_ns"].astype(np.int64, copy=False)
    action_valid_until = values["action_valid_until_monotonic_ns"].astype(
        np.int64, copy=False
    )
    for name, array in (
        ("observation_id", observation_ids),
        ("action_id", action_ids),
        ("action_created_monotonic_ns", action_created),
        ("action_target_monotonic_ns", action_target),
        ("action_valid_until_monotonic_ns", action_valid_until),
    ):
        if np.any(array < 0):
            errors.append(f"{name} must be non-negative")
    queued = values["flag_action_queued"].astype(bool, copy=False)
    if np.any(sample_valid & (observation_ids <= 0)):
        errors.append("sample-valid rows require a positive observation_id")
    if np.any(sample_valid & ~held & (action_ids <= 0)):
        errors.append("active non-held rows require a positive action_id")
    action_timing_valid = (
        (action_created > 0)
        & (action_created <= action_target)
        & (action_target <= action_valid_until)
    )
    if np.any(queued & ((action_ids <= 0) | ~action_timing_valid)):
        errors.append("queued actions have invalid identity/timing metadata")
    if np.any(sample_valid & ~held & ~queued):
        errors.append("active non-held rows must carry flag_action_queued")

    # Source-history ages/skews intentionally allow per-source NaN sentinels;
    # recompute their values from the persisted monotonic source timestamps.
    anchors = values["observation_anchor_monotonic_ns"].astype(np.int64, copy=False)
    if anchors.shape != (frame_count,):
        errors.append("observation_anchor_monotonic_ns must have shape (N,)")
        anchors = np.zeros(frame_count, dtype=np.int64)
    source_ns = _stack_source_fields(values, "source_monotonic_ns")
    receive_ns = values["observation_source_receive_monotonic_ns"].astype(
        np.int64, copy=False
    )
    history = values["observation_history_valid_mask"].astype(bool, copy=False)
    if receive_ns.shape != (frame_count, 4):
        errors.append("observation_source_receive_monotonic_ns must have shape (N,4)")
        receive_ns = np.zeros((frame_count, 4), dtype=np.int64)
    if np.any(receive_ns < 0):
        errors.append("observation source receive timestamps must be non-negative")
    if history.shape[1:] != (4, 1):
        errors.append("observation_history_valid_mask must have shape (N,4,1)")
        valid_sources = np.zeros((frame_count, 4), dtype=bool)
    else:
        valid_sources = history[:, :, 0]
    source_sequences = _stack_source_fields(values, "source_sequence")
    if source_sequences is None or source_sequences.shape != (frame_count, 4):
        errors.append("observation source sequence datasets must have shape (N,)")
        source_sequences = np.zeros((frame_count, 4), dtype=np.int64)
    if np.any(source_sequences < 0):
        errors.append("observation source sequences must be non-negative")
    if np.any(valid_sources & (source_sequences <= 0)):
        errors.append("valid observation sources require positive source sequences")
    if source_ns is None:
        errors.append("source monotonic timestamp datasets are missing")
    else:
        publish_ns = _stack_source_fields(values, "publish_monotonic_ns")
        if publish_ns is None:
            errors.append("publish monotonic timestamp datasets are missing")
        else:
            if source_ns.shape != (frame_count, 4) or publish_ns.shape != (
                frame_count,
                4,
            ):
                errors.append(
                    "source/publish monotonic timestamp datasets must have shape (N,4)"
                )
                source_ns = np.zeros((frame_count, 4), dtype=np.int64)
                publish_ns = np.zeros((frame_count, 4), dtype=np.int64)
            if np.any(source_ns < 0) or np.any(publish_ns < 0):
                errors.append(
                    "observation source/publish timestamps must be non-negative"
                )
            if np.any(anchors <= 0):
                errors.append("observation_anchor_monotonic_ns must be positive")
            causal = (
                (source_ns > 0)
                & (receive_ns > 0)
                & (publish_ns > 0)
                & (source_ns <= receive_ns)
                & (receive_ns <= publish_ns)
                & (publish_ns <= anchors[:, None])
            )
            if np.any(camera_fresh & ~causal[:, 3]):
                errors.append("fresh camera rows require a valid causal camera source")
            if np.any(valid_sources & ~causal):
                errors.append(
                    "valid observation sources violate source→receive→publish causality"
                )
            age = values["observation_source_age_s"].astype(np.float64, copy=False)
            skew = values["observation_source_skew_s"].astype(np.float64, copy=False)
            if age.shape != (frame_count, 4):
                errors.append("observation_source_age_s must have shape (N,4)")
                age = np.full((frame_count, 4), np.nan, dtype=np.float64)
            if skew.shape != (frame_count, 4):
                errors.append("observation_source_skew_s must have shape (N,4)")
                skew = np.full((frame_count, 4), np.nan, dtype=np.float64)
            expected_age = np.full((frame_count, 4), np.nan, dtype=np.float64)
            anchor_matrix = np.broadcast_to(anchors[:, None], source_ns.shape)
            expected_age[valid_sources] = (
                anchor_matrix[valid_sources] - source_ns[valid_sources]
            ) / 1e9
            expected_skew = np.full((frame_count, 4), np.nan, dtype=np.float64)
            causal_sources = valid_sources & causal
            for row in range(frame_count):
                selected = causal_sources[row]
                if np.any(selected):
                    newest = int(np.max(source_ns[row, selected]))
                    expected_skew[row, selected] = (
                        newest - source_ns[row, selected]
                    ) / 1e9
            if np.any(
                valid_sources
                & (
                    ~np.isfinite(age)
                    | ~np.isclose(
                        age, expected_age, atol=1e-7, rtol=0.0, equal_nan=True
                    )
                )
            ):
                errors.append(
                    "observation_source_age_s is inconsistent with source timestamps"
                )
            if np.any(
                valid_sources
                & (
                    ~np.isfinite(skew)
                    | ~np.isclose(
                        skew, expected_skew, atol=1e-7, rtol=0.0, equal_nan=True
                    )
                )
            ):
                errors.append(
                    "observation_source_skew_s is inconsistent with source timestamps"
                )
            if np.any(~valid_sources & np.isfinite(age)):
                errors.append("invalid observation sources must use age NaN sentinels")
            if np.any(~valid_sources & np.isfinite(skew)):
                errors.append("invalid observation sources must use skew NaN sentinels")
            aggregate = values["observation_skew_s"].astype(np.float64, copy=False)
            expected_aggregate = np.zeros(frame_count, dtype=np.float64)
            for row in range(frame_count):
                selected = causal_sources[row]
                if np.any(selected):
                    expected_aggregate[row] = float(
                        np.max(expected_skew[row, selected])
                    )
            aggregate_ok = np.isfinite(aggregate) & np.isclose(
                aggregate, expected_aggregate, atol=1e-7, rtol=0.0
            )
            # An invalid observation may use NaN as its explicit aggregate
            # sentinel; the producer's historical no-source path also wrote 0.
            no_sources = ~np.any(causal_sources, axis=1)
            aggregate_ok |= no_sources & np.isnan(aggregate)
            if np.any(~aggregate_ok):
                errors.append(
                    "observation_skew_s is inconsistent with source timestamps"
                )

    # Policy observation skew has a separate reference contract.  Recompute
    # it rather than trusting the scalar persisted alongside the state pair.
    policy_valid = values["policy_observation_valid"].astype(bool, copy=False)
    policy_ref = values["policy_observation_reference_monotonic_ns"].astype(
        np.int64, copy=False
    )
    policy_arm_source = values["policy_observation_arm_source_monotonic_ns"].astype(
        np.int64, copy=False
    )
    policy_hand_source = values["policy_observation_hand_source_monotonic_ns"].astype(
        np.int64, copy=False
    )
    policy_arm_publish = values["policy_observation_arm_publish_monotonic_ns"].astype(
        np.int64, copy=False
    )
    policy_hand_publish = values["policy_observation_hand_publish_monotonic_ns"].astype(
        np.int64, copy=False
    )
    policy_arm_sequence = values["policy_observation_arm_source_sequence"].astype(
        np.int64, copy=False
    )
    policy_hand_sequence = values["policy_observation_hand_source_sequence"].astype(
        np.int64, copy=False
    )
    policy_skew = values["policy_observation_skew_s"].astype(np.float64, copy=False)
    policy_expected = np.full(frame_count, np.nan, dtype=np.float64)
    policy_expected[policy_valid] = (
        policy_ref[policy_valid]
        - np.minimum(policy_arm_source[policy_valid], policy_hand_source[policy_valid])
    ) / 1e9
    if np.any(
        policy_valid
        & (
            (policy_ref <= 0)
            | (policy_arm_source <= 0)
            | (policy_hand_source <= 0)
            | (policy_arm_publish <= 0)
            | (policy_hand_publish <= 0)
            | (policy_arm_sequence <= 0)
            | (policy_hand_sequence <= 0)
            | (policy_arm_source > policy_ref)
            | (policy_hand_source > policy_ref)
            | (policy_arm_source > policy_arm_publish)
            | (policy_hand_source > policy_hand_publish)
            | (policy_arm_publish > anchors)
            | (policy_hand_publish > anchors)
        )
    ):
        errors.append(
            "valid policy observations violate source/reference/publish causality"
        )
    if np.any(policy_valid & (policy_expected < 0.0)):
        errors.append("policy observation source timestamp is later than its reference")
    policy_ok = np.isfinite(policy_skew) & np.isclose(
        policy_skew, policy_expected, atol=1e-7, rtol=0.0
    )
    policy_ok |= ~policy_valid & np.isnan(policy_skew)
    if np.any(~policy_ok):
        errors.append(
            "policy_observation_skew_s is inconsistent with source timestamps"
        )
    if np.any(
        policy_valid
        & (
            policy_ref
            != values["camera_source_monotonic_ns"].astype(np.int64, copy=False)
        )
    ):
        errors.append(
            "valid policy observation reference must equal camera source timestamp"
        )
    for name in ("policy_observation_arm_qpos", "policy_observation_hand_qpos"):
        if np.any(
            policy_valid & ~_row_all_finite(values[name].astype(np.float64, copy=False))
        ):
            errors.append(
                f"{name} must be finite when policy_observation_valid is true"
            )

    # A finite EEF action is canonical rot6d; an all-NaN row is the explicit
    # held/placeholder sentinel.  Use the planning boundary rather than
    # implementing a second rotation algorithm in the storage layer.
    from dexmani_real.planning.poses import validate_canonical_rot6d

    def _canonical_rot6d(rot6d: np.ndarray) -> bool:
        if rot6d.shape != (6,) or not np.all(np.isfinite(rot6d)):
            return False
        try:
            validate_canonical_rot6d(rot6d)
        except (TypeError, ValueError, FloatingPointError):
            return False
        return True

    for name in ("arm_ee", "action_arm_ee"):
        array = values[name].astype(np.float64, copy=False)
        if array.shape[1:] != (9,):
            errors.append(f"{name} must have tail shape (9,)")
            continue
        finite_rows = _row_all_finite(array)
        for row in np.flatnonzero(finite_rows):
            if not _canonical_rot6d(array[row, 3:]):
                errors.append(f"{name} row {row} has non-canonical rot6d")
                break
    active = sample_valid & ~held
    action_arm_ee = values["action_arm_ee"].astype(np.float64, copy=False)
    if np.any(active & ~_row_all_finite(action_arm_ee)):
        errors.append("active non-held rows require finite action_arm_ee")

    # Connected/fresh/valid flags gate when finite payloads are required.  A
    # disconnected or stale source may use the documented all-NaN sentinel.
    arm_connected = values["arm_connected"].astype(bool, copy=False)
    hand_connected = values["hand_connected"].astype(bool, copy=False)
    for flag, names in (
        (arm_connected, ("arm_qpos", "arm_qvel", "arm_tau", "arm_ee")),
        (
            hand_connected,
            (
                "hand_qpos",
                "hand_current",
                "hand_fingertip",
                "hand_contact",
                "hand_tactile_force",
            ),
        ),
    ):
        for name in names:
            if np.any(
                flag & ~_row_all_finite(values[name].astype(np.float64, copy=False))
            ):
                errors.append(f"{name} must be finite when its connected flag is true")
    tactile_fresh = values["tactile_fresh"].astype(bool, copy=False)
    tactile_source = values["tactile_source_monotonic_ns"].astype(np.int64, copy=False)
    if np.any(tactile_source < 0):
        errors.append("tactile source timestamps must be non-negative")
    if np.any(tactile_fresh & ((tactile_source <= 0) | (tactile_source > anchors))):
        errors.append(
            "fresh tactile rows require a positive source timestamp at or before the anchor"
        )
    for name in ("hand_contact", "hand_tactile_force"):
        if np.any(
            tactile_fresh
            & ~_row_all_finite(values[name].astype(np.float64, copy=False))
        ):
            errors.append(f"{name} must be finite when tactile_fresh is true")
    if np.any(
        sample_valid
        & ~_row_all_finite(values["action_arm_joint"].astype(np.float64, copy=False))
    ):
        errors.append("sample-valid rows require finite action_arm_joint")
    if np.any(
        sample_valid
        & ~_row_all_finite(values["action_hand_joint"].astype(np.float64, copy=False))
    ):
        errors.append("sample-valid rows require finite action_hand_joint")
    raw_arm_valid = sample_valid & ~held & values["flag_ik_ok"].astype(bool)
    raw_hand_valid = sample_valid & ~held & values["flag_retarget_ok"].astype(bool)
    if np.any(
        raw_arm_valid
        & ~_row_all_finite(
            values["action_arm_joint_raw"].astype(np.float64, copy=False)
        )
    ):
        errors.append("valid raw arm actions require finite action_arm_joint_raw")
    if np.any(
        raw_hand_valid
        & ~_row_all_finite(
            values["action_hand_joint_raw"].astype(np.float64, copy=False)
        )
    ):
        errors.append("valid raw hand actions require finite action_hand_joint_raw")

    if attrs is not None:
        errors.extend(validate_semantic_meta_attrs(attrs, legacy=legacy))
    return tuple(errors)


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
    """Return the datasets required in one raw episode data.h5."""
    return frozenset(_RAW_DATASET_SPECS)


def validate_data_layout(
    dataset_shapes: Mapping[str, tuple[int, ...]],
    dataset_dtypes: Mapping[str, Any],
    *,
    frame_count: int,
    arm_sent_stream: bool,
) -> tuple[str, ...]:
    """Return deterministic validation errors for a raw episode layout."""
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
    known_specs = {**_RAW_DATASET_SPECS, **CONDITIONAL_DATASET_SPECS}
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
    "CAMERA_HEALTH_TAXONOMY",
    "CAMERA_HEALTH_TAXONOMY_JSON",
    "CAMERA_TIMING_DATASET_SPECS",
    "CONDITIONAL_DATASET_SPECS",
    "DATASET_SPECS",
    "DIAGNOSTIC_TAIL_SHAPES",
    "DatasetSpec",
    "EPISODE_SCHEMA_VERSION",
    "LEGACY_EPISODE_SCHEMA_VERSION",
    "HAND_RAW_ACTION_VALIDITY_EXPRESSION",
    "RAW_DEPTH_SHA256_ATTR",
    "RAW_MANIFEST_VERSION",
    "RAW_MANIFEST_VERSION_ATTR",
    "RAW_MEMBER_SHA256_JSON_ATTR",
    "RAW_MEMBER_NAMES",
    "RAW_RESERVED_META_KEYS",
    "RAW_RGB_SHA256_ATTR",
    "SEMANTIC_META_ATTRS",
    "SOURCE_FRAME_DATASET_NAMES",
    "compute_episode_quality_metrics",
    "expected_source_frame_dataset_names",
    "normalize_diagnostics",
    "required_dataset_names",
    "validate_camera_metadata_keys",
    "validate_data_layout",
    "validate_raw_member_hashes",
    "validate_raw_semantics",
    "validate_semantic_meta_attrs",
    "validate_source_frame_keys",
]
