"""Stable NumPy wire schemas for cross-process channels.

This module deliberately imports only NumPy.  Shared-memory allocation,
policy logic, device workers, and recording serialization may depend on these
schemas; the schema layer must never depend on any of those implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.robot_spec import (
    ARM_JOINT_SHAPE,
    HAND_CONTACT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)

RECORD_TASK_LABEL_BYTES = 128
RECORD_OPERATOR_BYTES = 128
RECORD_STOP_REASON_BYTES = 512
RECORD_STATUS_TEXT_BYTES = 2_048

# Runtime IPC capacity for flat learned-policy predictions. Adapters must not
# truncate larger requests, and the executor decodes only the validated 19-D
# joint or 21-D EE representations.
MAX_PREDICTION_STEPS = 32
MAX_POLICY_ACTION_DIM = 21

# Realtime point-cloud transport is fixed-size per deployment. Keeping the
# supported sizes explicit prevents a model/RuntimeChannels shape mismatch from
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


# Every servo publication is one coherent arm/hand record. The publisher marks
# its ring sequence active atomically; workers execute only while that exact
# latest-wins ticket is still active. Physical execution remains asynchronous.
_COMMON_COMMAND_FIELDS = [
    ("run_generation", "<u8"),
    ("observation_id", "<u8"),
    ("action_id", "<u8"),
    ("created_monotonic_ns", "<u8"),
    # Policy-grid target retained for provenance even when publication occurs
    # after that instant and the worker delivery target is immediate.
    ("scheduled_target_monotonic_ns", "<u8"),
    ("target_monotonic_ns", "<u8"),
    ("valid_until_monotonic_ns", "<u8"),
    ("is_hold", "<u1"),
]
COUPLED_COMMAND_DTYPE = np.dtype(
    _COMMON_COMMAND_FIELDS
    + [
        ("arm_present", "<u1"),
        ("hand_present", "<u1"),
        ("arm_qpos", "<f8", ARM_JOINT_SHAPE),
        ("hand_qpos", "<f8", HAND_JOINT_SHAPE),
    ],
    align=True,
)

# One flat policy prediction is written per inference; the executor consumes
# only the latest. ``num_steps`` and ``action_dim`` describe the populated
# prefix of the fixed-capacity payload for an independent IPC decoder.
PREDICTION_DTYPE = np.dtype(
    [
        ("run_generation", "<u8"),
        ("source_monotonic_ns", "<u8"),
        ("logical_step_monotonic_ns", "<u8"),
        ("num_steps", "<u4"),
        ("action_dim", "<u4"),
        ("actions", "<f8", (MAX_PREDICTION_STEPS, MAX_POLICY_ACTION_DIM)),
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
        # Monotonic time immediately after the arm SDK accepted last_cmd_seq.
        ("last_cmd_accepted_monotonic_ns", "<u8"),
        ("last_cmd_is_hold", "<u1"),
        ("source_monotonic_ns", "<u8"),
        # Load-bearing for causal consumer selection (ipc/causal.py,
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
        ("connected", "<u1"),
        # Set when qpos is held from the last read after a single-frame failure.
        ("qpos_stale", "<u1"),
        # action_id whose exact requested target was accepted by
        # XHand.send_action(), including a configured-current overrun accepted
        # as grasp contact. This is not physical convergence.
        ("accepted_target_action_id", "<u8"),
        # Monotonic time immediately after the XHand SDK accepted that target.
        ("accepted_target_monotonic_ns", "<u8"),
        ("commboard_err", "<i4", HAND_JOINT_SHAPE),
        ("jointboard_err", "<i4", HAND_JOINT_SHAPE),
        ("tipboard_err", "<i4", HAND_JOINT_SHAPE),
        ("source_monotonic_ns", "<u8"),
        ("publish_monotonic_ns", "<u8"),
        ("state_valid", "<u1"),
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
        # ``receive`` is immediately after frame_queue.wait_for_frame returns;
        # payload readiness is after owned RGB and depth-to-color Z16 copies.
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
            # Robot state selected for the camera/point-cloud source time.
            # It is distinct from the newest grid-cut feedback above.
            ("policy_observation_arm_qpos", "<f8", ARM_JOINT_SHAPE),
            ("policy_observation_hand_qpos", "<f8", HAND_JOINT_SHAPE),
            ("policy_observation_reference_monotonic_ns", "<u8"),
            ("policy_observation_arm_source_sequence", "<u8"),
            ("policy_observation_hand_source_sequence", "<u8"),
            ("policy_observation_arm_source_monotonic_ns", "<u8"),
            ("policy_observation_hand_source_monotonic_ns", "<u8"),
            ("policy_observation_arm_publish_monotonic_ns", "<u8"),
            ("policy_observation_hand_publish_monotonic_ns", "<u8"),
            ("policy_observation_valid", "<u1"),
            ("policy_observation_skew_s", "<f8"),
            ("hand_accepted_target_action_id", "<u8"),
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
    "ARM_STATE_DTYPE",
    "CAMERA_FRAME_HEADER_DTYPE",
    "COUPLED_COMMAND_DTYPE",
    "HAND_STATE_DTYPE",
    "HAND_TACTILE_DTYPE",
    "MAX_POLICY_ACTION_DIM",
    "MAX_PREDICTION_STEPS",
    "POINT_CLOUD_FEATURE_DIM",
    "PREDICTION_DTYPE",
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
