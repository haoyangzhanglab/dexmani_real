"""Cross-process NumPy dtype definitions and shape constants.

This module deliberately imports only NumPy.  Shared-memory allocation,
policy logic, device workers, and recording serialization may depend on these
schemas; the schema layer must never depend on any of those implementations.
"""

from __future__ import annotations

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

_COMMAND_COMMON_FIELDS = [
    ("run_generation", "<u8"),
    ("observation_id", "<u8"),
    ("action_id", "<u8"),
    ("created_monotonic_ns", "<u8"),
    ("target_monotonic_ns", "<u8"),
    ("valid_until_monotonic_ns", "<u8"),
    ("is_hold", "<u1"),
]

ARM_COMMAND_DTYPE = np.dtype(
    _COMMAND_COMMON_FIELDS + [("qpos_cmd", "<f8", ARM_JOINT_SHAPE)], align=True
)
HAND_COMMAND_DTYPE = np.dtype(
    _COMMAND_COMMON_FIELDS + [("qpos_cmd", "<f8", HAND_JOINT_SHAPE)], align=True
)

ARM_CONTROL_DTYPE = np.dtype(
    [
        ("kind", "<u1"),
        ("run_generation", "<u8"),
        ("action_id", "<u8"),
        ("created_monotonic_ns", "<u8"),
        ("valid_until_monotonic_ns", "<u8"),
    ],
    align=True,
)

INFERENCE_CANDIDATE_DTYPE = np.dtype(
    [
        ("observation_id", "<u8"),
        ("run_generation", "<u8"),
        ("created_monotonic_ns", "<u8"),
        ("target_monotonic_ns", "<u8"),
        ("valid_until_monotonic_ns", "<u8"),
        ("has_arm", "<u1"),
        ("has_hand", "<u1"),
        ("is_hold", "<u1"),
        ("arm_qpos", "<f8", ARM_JOINT_SHAPE),
        ("hand_qpos", "<f8", HAND_JOINT_SHAPE),
    ],
    align=True,
)

ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", ARM_JOINT_SHAPE),
        ("qvel", "<f8", ARM_JOINT_SHAPE),
        ("tau", "<f8", ARM_JOINT_SHAPE),
        ("eef_pos", "<f8", (3,)),
        ("eef_rot6d", "<f8", (6,)),
        ("error_code", "<i4"),
        ("connected", "<u1"),
        ("mode", "<i4"),
        ("tracking_err", "<f8"),
        ("last_cmd_seq", "<u8"),
        ("last_cmd_created_s", "<f8"),
        ("last_cmd_received_s", "<f8"),
        ("last_cmd_applied_s", "<f8"),
        ("last_cmd_queue_latency_s", "<f8"),
        ("last_cmd_apply_latency_s", "<f8"),
        ("last_cmd_sdk_duration_s", "<f8"),
        ("last_cmd_is_hold", "<u1"),
        ("source_monotonic_ns", "<u8"),
        ("publish_monotonic_ns", "<u8"),
        ("state_valid", "<u1"),
        ("timestamp", "<f8"),
    ]
)

HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", HAND_JOINT_SHAPE),
        ("current", "<f8", HAND_JOINT_SHAPE),
        ("tactile_sum", "<f8", HAND_TACTILE_SUM_SHAPE),
        ("tactile_contact", "<u1", HAND_CONTACT_SHAPE),
        ("error_state", "<u1"),
        ("connected", "<u1"),
        # Reserved compatibility bit: execution non-convergence is not a
        # feedback-freshness fault. Freshness comes from source timestamp and
        # read_healthy/state_valid.
        ("qpos_stale", "<u1"),
        # Last command for which XHand.send_action() returned success.
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

# A ring publication is driven by a right-hand frame. ``head_*`` pose fields
# are the latest cached HeadFrame, identified by its own sequence and receive time.
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
        ("timestamp", "<f8"),
        ("capture_monotonic_s", "<f8"),
        ("source_monotonic_ns", "<u8"),
        ("receive_monotonic_ns", "<u8"),
        ("publish_monotonic_ns", "<u8"),
        ("camera_generation", "<u8"),
        ("frame_number", "<u8"),
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
        ("pc_num_points", "<u4"),
        ("pc_source_point_count", "<u4"),
        ("pc_valid_depth_ratio", "<f4"),
        ("pc_padding_count", "<u4"),
        ("camera_health", "<u1"),
        ("pointcloud_valid", "<u1"),
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
        ("generation", "<u8"),
        ("frame_count", "<u8"),
        ("updated_monotonic_ns", "<u8"),
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
    pointcloud_shape: tuple[int, int],
) -> np.dtype:
    """Return the fixed recorder sample schema for configured camera shapes."""
    return np.dtype(
        [
            ("generation", "<u8"),
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
            ("arm_last_cmd_queue_latency_s", "<f8"),
            ("arm_last_cmd_apply_latency_s", "<f8"),
            ("arm_last_cmd_sdk_duration_s", "<f8"),
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
            ("camera_pointcloud", "<f4", pointcloud_shape),
            # Typed per-grid metadata required to reconstruct recorder input.
            # Legacy HDF5-only defaults stay inside RecorderIO rather than
            # becoming permanent cross-process fields.
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
            ("pointcloud_source_point_count", "<u4"),
            ("pointcloud_valid_depth_ratio", "<f8"),
            ("pointcloud_padding_count", "<u4"),
            ("flag_ik_ok", "<u1"),
            ("flag_ik_attempted", "<u1"),
            ("flag_retarget_ok", "<u1"),
            ("flag_held", "<u1"),
            ("flag_safety_reject", "<u1"),
            ("flag_frame_status", "<u1"),
            ("camera_health", "<u1"),
            ("camera_fresh", "<u1"),
            ("pointcloud_valid", "<u1"),
            ("camera_frame_number", "<u8"),
            ("camera_ring_sequence", "<u8"),
            ("camera_device_timestamp_s", "<f8"),
            ("camera_capture_monotonic_s", "<f8"),
            ("camera_age_s", "<f8"),
            ("camera_generation", "<u8"),
            ("camera_clock_reset", "<u1"),
            ("camera_duplicate", "<u1"),
            ("camera_frame_gap", "<u4"),
            ("camera_backlog_s", "<f8"),
            ("tracking_error", "<f8"),
            ("ik_solve_time_ms", "<f8"),
            ("target_pos_before_clamp", "<f8", (3,)),
            ("head_quat_wxyz", "<f8", (4,)),
            ("target_eef_pos_raw", "<f8", (3,)),
            ("target_eef_rot6d_raw", "<f8", (6,)),
            ("policy_map_time_ms", "<f8"),
            ("hand_retarget_time_ms", "<f8"),
            ("transition_check_time_ms", "<f8"),  # reserved (always 0; SafetyGate collision/transition removed 2026-08-12)
            ("policy_compute_time_ms", "<f8"),
        ],
        align=True,
    )


__all__ = [
    "ARM_COMMAND_DTYPE",
    "ARM_CONTROL_DTYPE",
    "ARM_STATE_DTYPE",
    "CAMERA_FRAME_HEADER_DTYPE",
    "HAND_COMMAND_DTYPE",
    "HAND_STATE_DTYPE",
    "HAND_TACTILE_DTYPE",
    "INFERENCE_CANDIDATE_DTYPE",
    "RECORD_CONTROL_DTYPE",
    "RECORD_OPERATOR_BYTES",
    "RECORD_STATUS_DTYPE",
    "RECORD_STOP_REASON_BYTES",
    "RECORD_TASK_LABEL_BYTES",
    "VR_FRAME_DTYPE",
    "make_record_sample_dtype",
]
