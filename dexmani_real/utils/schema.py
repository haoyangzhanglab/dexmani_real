"""Cross-process NumPy dtype definitions, shape constants, and NaN-array construction.

This module deliberately imports only NumPy.  Shared-memory allocation,
policy logic, device workers, and recording serialization may depend on these
schemas; the schema layer must never depend on any of those implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RECORD_TASK_LABEL_BYTES = 128
RECORD_OPERATOR_BYTES = 128
RECORD_STOP_REASON_BYTES = 512
RECORD_STATUS_TEXT_BYTES = 2_048

ARM_DOF = 7
HAND_DOF = 12
HAND_FINGER_COUNT = 5
TACTILE_POINTS_PER_FINGER = 120
TACTILE_AXIS_COUNT = 3

# Canonical order for every cross-process XHand joint vector.  Retargeting,
# planning, recording, and the device boundary all consume this SDK order.
XHAND_SDK_JOINT_NAMES: tuple[str, ...] = (
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
)
if (
    len(XHAND_SDK_JOINT_NAMES) != HAND_DOF
    or len(set(XHAND_SDK_JOINT_NAMES)) != HAND_DOF
):
    raise RuntimeError("XHand SDK joint names must be unique and match HAND_DOF")

ARM_JOINT_SHAPE = (ARM_DOF,)
HAND_JOINT_SHAPE = (HAND_DOF,)
ARM_EE_SHAPE = (9,)
HAND_TACTILE_SUM_SHAPE = (HAND_FINGER_COUNT, TACTILE_AXIS_COUNT)
HAND_TACTILE_FORCE_SHAPE = (
    HAND_FINGER_COUNT,
    TACTILE_POINTS_PER_FINGER,
    TACTILE_AXIS_COUNT,
)
HAND_CONTACT_SHAPE = (HAND_FINGER_COUNT,)
HAND_FINGERTIP_SHAPE = (HAND_FINGER_COUNT, 3)

# Runtime IPC capacity for learned-policy plans; adapters must not truncate larger requests.
MAX_POLICY_CHUNK_STEPS = 32

# Realtime point-cloud transport is fixed-size per deployment. Keeping the
# supported sizes explicit prevents a model/SharedStorage shape mismatch from
# reaching the inference boundary.
POINT_CLOUD_FEATURE_DIM = 6
SUPPORTED_POINT_CLOUD_COUNTS = frozenset({1024, 2048, 4096, 8192})


def make_pointcloud_frame_dtype(num_points: int) -> np.dtype:
    """Return the latest-only realtime point-cloud IPC schema."""
    if (
        isinstance(num_points, bool)
        or int(num_points) not in SUPPORTED_POINT_CLOUD_COUNTS
    ):
        raise ValueError(
            "point-cloud count must be one of "
            f"{sorted(SUPPORTED_POINT_CLOUD_COUNTS)}, got {num_points!r}"
        )
    count = int(num_points)
    return np.dtype(
        [
            ("source_camera_sequence", "<u8"),
            ("source_monotonic_ns", "<u8"),
            ("camera_publish_monotonic_ns", "<u8"),
            ("publish_monotonic_ns", "<u8"),
            ("camera_generation", "<u8"),
            ("depth_frame_number", "<u8"),
            ("color_frame_number", "<u8"),
            ("point_cloud", "<f4", (count, POINT_CLOUD_FEATURE_DIM)),
        ],
        align=True,
    )


def validate_point_cloud_array(
    value: Any,
    *,
    num_points: int,
    label: str = "point_cloud",
) -> np.ndarray:
    """Validate the canonical finite ``float32[N,6]`` xyzrgb payload."""
    if isinstance(num_points, bool) or int(num_points) <= 0:
        raise ValueError("num_points must be a positive integer")
    array = np.asarray(value)
    expected_shape = (int(num_points), POINT_CLOUD_FEATURE_DIM)
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ValueError(
            f"{label} must be float32 {expected_shape}, "
            f"got shape={array.shape} dtype={array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN/Inf")
    if np.any(array[:, 3:] < 0.0) or np.any(array[:, 3:] > 1.0):
        raise ValueError(f"{label} RGB values must be in [0,1]")
    return array


_HAND_COMMAND_COMMON_FIELDS = [
    ("run_generation", "<u8"),
    ("observation_id", "<u8"),
    ("action_id", "<u8"),
    ("created_monotonic_ns", "<u8"),
    ("target_monotonic_ns", "<u8"),
    ("valid_until_monotonic_ns", "<u8"),
    ("is_hold", "<u1"),
]

# The arm ring is latest-wins; staleness uses ``action_id`` and command age.
ARM_COMMAND_DTYPE = np.dtype(
    [
        ("action_id", "<u8"),
        ("created_monotonic_ns", "<u8"),
        ("is_hold", "<u1"),
        ("qpos_cmd", "<f8", ARM_JOINT_SHAPE),
    ],
    align=True,
)
HAND_COMMAND_DTYPE = np.dtype(
    _HAND_COMMAND_COMMON_FIELDS + [("qpos_cmd", "<f8", HAND_JOINT_SHAPE)], align=True
)

# One payload is written per inference; the coordinator consumes only the latest.
POLICY_PLAN_DTYPE = np.dtype(
    [
        ("plan_id", "<u8"),
        ("run_generation", "<u8"),
        ("observation_id", "<u8"),
        ("observation_anchor_monotonic_ns", "<u8"),
        ("inference_started_monotonic_ns", "<u8"),
        ("inference_finished_monotonic_ns", "<u8"),
        ("num_steps", "<u4"),
        ("arm_present", "<u1"),
        ("hand_present", "<u1"),
        ("target_monotonic_ns", "<u8", (MAX_POLICY_CHUNK_STEPS,)),
        ("arm_qpos", "<f8", (MAX_POLICY_CHUNK_STEPS, ARM_DOF)),
        ("hand_qpos", "<f8", (MAX_POLICY_CHUNK_STEPS, HAND_DOF)),
        ("valid_mask", "<u1", (MAX_POLICY_CHUNK_STEPS,)),
    ],
    align=True,
)

ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", ARM_JOINT_SHAPE),
        ("qvel", "<f8", ARM_JOINT_SHAPE),
        ("tau", "<f8", ARM_JOINT_SHAPE),
        ("error_code", "<i4"),
        # Worker-alive indicator kept for the shared feedback-health predicate;
        # a disconnect now fails the worker, so this is truthful while alive.
        ("connected", "<u1"),
        ("tracking_err", "<f8"),
        ("last_cmd_seq", "<u8"),
        ("last_cmd_is_hold", "<u1"),
        ("source_monotonic_ns", "<u8"),
        # Load-bearing for causal consumer selection (shm/causal_reader.py,
        # deployment observation history, recording sample alignment).
        ("publish_monotonic_ns", "<u8"),
        ("state_valid", "<u1"),
    ]
)

HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", HAND_JOINT_SHAPE),
        ("current", "<f8", HAND_JOINT_SHAPE),
        ("tactile_sum", "<f8", HAND_TACTILE_SUM_SHAPE),
        # Combined tactile values are zero-filled on read failure; this bit
        # distinguishes an invalid sample from a valid zero-contact sample.
        ("tactile_sum_valid", "<u1"),
        ("tactile_contact", "<u1", HAND_CONTACT_SHAPE),
        ("error_state", "<u1"),
        ("connected", "<u1"),
        # Set when qpos is held from the last read after a single-frame failure.
        ("qpos_stale", "<u1"),
        # action_id of the last command accepted by XHand.send_action(),
        # including a configured-current overrun accepted as grasp contact;
        # this is not the hand command ring's internal sequence.
        ("last_cmd_seq", "<u8"),
        ("last_cmd_qpos", "<f8", HAND_JOINT_SHAPE),
        ("commboard_err", "<i4", HAND_JOINT_SHAPE),
        ("jointboard_err", "<i4", HAND_JOINT_SHAPE),
        ("tipboard_err", "<i4", HAND_JOINT_SHAPE),
        ("source_monotonic_ns", "<u8"),
        ("publish_monotonic_ns", "<u8"),
        ("state_valid", "<u1"),
        ("send_healthy", "<u1"),
        ("read_healthy", "<u1"),
        # Cumulative telemetry includes accepted grasp-contact overrun sends
        # and recoverable overrun reads. It is carried by the next successful
        # feedback frame when a read intentionally publishes no synthetic
        # state frame.
        ("read_error_count", "<u8"),
        ("overcurrent_error_count", "<u8"),
        ("timestamp", "<f8"),
    ]
)

HAND_TACTILE_DTYPE = np.dtype(
    [
        ("tactile_force", "<f8", HAND_TACTILE_FORCE_SHAPE),
        ("source_monotonic_ns", "<u8"),
        ("fresh", "<u1"),
        ("calibrated", "<u1"),
        ("unit_code", "<u1"),
    ]
)

# A ring publication is driven by a right-hand frame; ``head_*`` fields cache the latest HeadFrame.
VR_FRAME_DTYPE = np.dtype(
    [
        ("wrist_pos", "<f8", (3,)),
        ("wrist_quat_wxyz", "<f8", (4,)),
        ("landmarks", "<f8", (21, 3)),
        ("head_pos", "<f8", (3,)),
        ("head_quat_wxyz", "<f8", (4,)),
        ("recv_ts_ns", "<u8"),
        ("source_ts_ns", "<u8"),
        ("sequence_id", "<u8"),
        ("source_frame_seq", "<u8"),
        ("local_recv_ns", "<u8"),
        ("side", "<i4"),
        ("head_sequence_id", "<u8"),
        ("head_recv_ts_ns", "<u8"),
    ],
    align=True,
)

CAMERA_FRAME_HEADER_DTYPE = np.dtype(
    [
        ("depth_device_timestamp_s", "<f8"),
        ("color_device_timestamp_s", "<f8"),
        ("source_monotonic_ns", "<u8"),
        ("receive_monotonic_ns", "<u8"),
        # ``receive`` is immediately after wait_for_frames; payload readiness
        # is after owned native RGB/depth NumPy copies.
        ("payload_ready_monotonic_ns", "<u8"),
        ("depth_timestamp_domain", "<u1"),
        # 255 denotes that no color stream is present.
        ("color_timestamp_domain", "<u1"),
        ("publish_monotonic_ns", "<u8"),
        ("camera_generation", "<u8"),
        ("depth_frame_number", "<u8"),
        ("color_frame_number", "<u8"),
        ("frame_gap", "<u4"),
        ("clock_reset", "<u1"),
        ("duplicate", "<u1"),
        ("backlog_s", "<f8"),
        ("rgb_size", "<u8"),
        ("depth_size", "<u8"),
        ("rgb_shape_h", "<u4"),
        ("rgb_shape_w", "<u4"),
        ("rgb_shape_c", "<u4"),
        ("depth_shape_h", "<u4"),
        ("depth_shape_w", "<u4"),
        ("pc_valid_depth_ratio", "<f4"),
        ("camera_health", "<u1"),
        ("pad", "<u1", (4,)),
    ],
    align=True,
)

RECORD_CONTROL_DTYPE = np.dtype(
    [
        ("command", "<u1"),
        ("generation", "<u8"),
        ("save", "<u1"),
        ("created_monotonic_ns", "<u8"),
        ("task_label", f"S{RECORD_TASK_LABEL_BYTES}"),
        ("operator", f"S{RECORD_OPERATOR_BYTES}"),
        ("stop_reason", f"S{RECORD_STOP_REASON_BYTES}"),
    ],
    align=True,
)

RECORD_STATUS_DTYPE = np.dtype(
    [
        ("phase", "<u1"),
        ("saved", "<u1"),
        ("min_frames_met", "<u1"),
        ("generation", "<u8"),
        ("frame_count", "<u8"),
        ("failure_count", "<u8"),
        ("updated_monotonic_ns", "<u8"),
        ("reason_length", "<u4"),
        ("reason", f"S{RECORD_STOP_REASON_BYTES}"),
        ("error_length", "<u4"),
        ("error", f"S{RECORD_STATUS_TEXT_BYTES}"),
        ("path_length", "<u4"),
        ("path", f"S{RECORD_STATUS_TEXT_BYTES}"),
    ],
    align=True,
)


def make_record_sample_dtype(
    rgb_shape: tuple[int, int, int],
    depth_shape: tuple[int, int],
) -> np.dtype:
    """Return the fixed recorder sample schema for configured camera shapes."""
    return np.dtype(
        [
            ("generation", "<u8"),
            # Transient control-run identity. RecorderIO uses it to split the
            # time grid across command-silent pauses; it is not persisted.
            ("control_run_generation", "<u8"),
            ("sample_sequence", "<u8"),
            ("arm_qpos", "<f8", ARM_JOINT_SHAPE),
            ("arm_qvel", "<f8", ARM_JOINT_SHAPE),
            ("arm_tau", "<f8", ARM_JOINT_SHAPE),
            ("eef_pos", "<f8", (3,)),
            ("eef_quat_wxyz", "<f8", (4,)),
            ("eef_rot6d", "<f8", (6,)),
            ("hand_qpos", "<f8", HAND_JOINT_SHAPE),
            ("hand_current", "<f8", HAND_JOINT_SHAPE),
            ("hand_tactile_sum", "<f8", HAND_TACTILE_SUM_SHAPE),
            ("hand_tactile_force", "<f8", HAND_TACTILE_FORCE_SHAPE),
            ("hand_tactile_contact", "<u1", HAND_CONTACT_SHAPE),
            ("hand_tipboard_err", "<i4", HAND_JOINT_SHAPE),
            ("hand_commboard_err", "<i4", HAND_JOINT_SHAPE),
            ("hand_jointboard_err", "<i4", HAND_JOINT_SHAPE),
            ("hand_qpos_stale", "<u1"),
            ("fingertip_pos", "<f8", HAND_FINGERTIP_SHAPE),
            ("arm_connected", "<u1"),
            ("hand_connected", "<u1"),
            ("state_timestamp", "<f8"),
            ("hand_error_state", "<u1"),
            ("arm_last_cmd_seq", "<u8"),
            ("arm_last_cmd_is_hold", "<u1"),
            ("action_arm_qpos", "<f8", ARM_JOINT_SHAPE),
            ("action_hand_qpos", "<f8", HAND_JOINT_SHAPE),
            ("action_target_eef_pos", "<f8", (3,)),
            ("action_target_eef_rot6d", "<f8", (6,)),
            ("vr_wrist_pos", "<f8", (3,)),
            ("vr_wrist_quat_wxyz", "<f8", (4,)),
            ("vr_landmarks", "<f8", (21, 3)),
            ("vr_head_quat_wxyz", "<f8", (4,)),
            ("camera_present", "<u1"),
            ("camera_rgb", "<u1", rgb_shape),
            ("camera_depth", "<u2", depth_shape),
            # Typed per-grid metadata used by the recorder.
            ("arm_qpos_sent", "<f8", ARM_JOINT_SHAPE),
            ("observation_id", "<u8"),
            ("observation_anchor_monotonic_ns", "<u8"),
            ("arm_source_sequence", "<u8"),
            ("hand_source_sequence", "<u8"),
            ("vr_source_sequence", "<u8"),
            ("camera_source_sequence", "<u8"),
            ("arm_source_monotonic_ns", "<u8"),
            ("hand_source_monotonic_ns", "<u8"),
            ("vr_source_monotonic_ns", "<u8"),
            ("camera_source_monotonic_ns", "<u8"),
            ("arm_publish_monotonic_ns", "<u8"),
            ("hand_publish_monotonic_ns", "<u8"),
            ("vr_publish_monotonic_ns", "<u8"),
            ("camera_publish_monotonic_ns", "<u8"),
            ("observation_source_receive_monotonic_ns", "<u8", (4,)),
            ("observation_source_age_s", "<f8", (4,)),
            ("observation_source_skew_s", "<f8", (4,)),
            ("observation_history_valid_mask", "<u1", (4, 1)),
            ("observation_valid", "<u1"),
            ("observation_skew_s", "<f8"),
            ("action_id", "<u8"),
            ("action_created_monotonic_ns", "<u8"),
            ("action_target_monotonic_ns", "<u8"),
            ("action_valid_until_monotonic_ns", "<u8"),
            ("action_arm_joint_raw", "<f8", ARM_JOINT_SHAPE),
            ("action_hand_joint_raw", "<f8", HAND_JOINT_SHAPE),
            ("flag_action_queued", "<u1"),
            ("tactile_fresh", "<u1"),
            ("tactile_source_monotonic_ns", "<u8"),
            ("tactile_calibrated", "<u1"),
            ("tactile_unit_code", "<u1"),
            ("pointcloud_valid_depth_ratio", "<f8"),
            ("flag_ik_ok", "<u1"),
            ("flag_ik_attempted", "<u1"),
            ("flag_retarget_ok", "<u1"),
            ("flag_held", "<u1"),
            ("flag_safety_reject", "<u1"),
            ("flag_frame_status", "<u1"),
            ("camera_health", "<u1"),
            ("camera_fresh", "<u1"),
            ("camera_depth_frame_number", "<u8"),
            ("camera_color_frame_number", "<u8"),
            ("camera_ring_sequence", "<u8"),
            ("camera_depth_device_timestamp_s", "<f8"),
            ("camera_color_device_timestamp_s", "<f8"),
            ("camera_wait_return_monotonic_ns", "<u8"),
            ("camera_payload_ready_monotonic_ns", "<u8"),
            ("camera_depth_timestamp_domain", "<u1"),
            ("camera_color_timestamp_domain", "<u1"),
            ("camera_age_s", "<f8"),
            ("camera_generation", "<u8"),
            ("camera_clock_reset", "<u1"),
            ("camera_duplicate", "<u1"),
            ("camera_frame_gap", "<u4"),
            ("camera_backlog_s", "<f8"),
            ("camera_delivery_delay_above_floor_s", "<f8"),
            ("tracking_error", "<f8"),
            ("ik_solve_time_ms", "<f8"),
            ("target_pos_before_clamp", "<f8", (3,)),
            ("head_quat_wxyz", "<f8", (4,)),
            ("target_eef_pos_raw", "<f8", (3,)),
            ("target_eef_rot6d_raw", "<f8", (6,)),
            ("policy_map_time_ms", "<f8"),
            # Full backend solve for a new VR sequence, or cache-lookup time
            # when the causal control grid reuses an observation.
            ("hand_retarget_time_ms", "<f8"),
            ("transition_check_time_ms", "<f8"),  # reserved (always 0)
            ("policy_compute_time_ms", "<f8"),
        ],
        align=True,
    )


def nan_array(shape: int | tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
    """Create an array filled with NaN.

    Centralized factory for the ``np.full(shape, np.nan, dtype=np.float64)``
    pattern repeated across the codebase.  Ensures consistent dtype and NaN fill.
    """
    return np.full(shape, np.nan, dtype=dtype)


__all__ = [
    "ARM_COMMAND_DTYPE",
    "ARM_STATE_DTYPE",
    "CAMERA_FRAME_HEADER_DTYPE",
    "HAND_COMMAND_DTYPE",
    "HAND_STATE_DTYPE",
    "HAND_TACTILE_DTYPE",
    "MAX_POLICY_CHUNK_STEPS",
    "POINT_CLOUD_FEATURE_DIM",
    "POLICY_PLAN_DTYPE",
    "RECORD_CONTROL_DTYPE",
    "RECORD_OPERATOR_BYTES",
    "RECORD_STATUS_DTYPE",
    "RECORD_STOP_REASON_BYTES",
    "RECORD_TASK_LABEL_BYTES",
    "SUPPORTED_POINT_CLOUD_COUNTS",
    "VR_FRAME_DTYPE",
    "make_pointcloud_frame_dtype",
    "make_record_sample_dtype",
    "nan_array",
    "validate_point_cloud_array",
]
