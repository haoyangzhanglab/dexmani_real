"""Dependency-neutral NumPy schemas for all cross-process payloads.

This module deliberately imports only NumPy.  Shared-memory allocation,
policy logic, device workers, and recording serialization may depend on these
schemas; the schema layer must never depend on any of those implementations.
"""

from __future__ import annotations

import numpy as np

RECORD_CONTROL_JSON_BYTES = 65_536
RECORD_SAMPLE_JSON_BYTES = 32_768
RECORD_STATUS_TEXT_BYTES = 2_048

_COMMAND_COMMON_FIELDS = [
    ("session_generation", "<u8"),
    ("policy_epoch", "<u8"),
    ("observation_id", "<u8"),
    ("action_id", "<u8"),
    ("chunk_id", "<u8"),
    ("step_index", "<u4"),
    ("created_monotonic_ns", "<u8"),
    ("target_monotonic_ns", "<u8"),
    ("valid_until_monotonic_ns", "<u8"),
    ("is_hold", "<u1"),
]

ARM_COMMAND_DTYPE = np.dtype(_COMMAND_COMMON_FIELDS + [("qpos_cmd", "<f8", (7,))], align=True)
HAND_COMMAND_DTYPE = np.dtype(_COMMAND_COMMON_FIELDS + [("qpos_cmd", "<f8", (12,))], align=True)
COMMIT_DTYPE = np.dtype(
    [
        ("session_generation", "<u8"),
        ("policy_epoch", "<u8"),
        ("observation_id", "<u8"),
        ("action_id", "<u8"),
        ("chunk_id", "<u8"),
        ("step_index", "<u4"),
        ("created_monotonic_ns", "<u8"),
        ("committed_monotonic_ns", "<u8"),
        ("target_monotonic_ns", "<u8"),
        ("valid_until_monotonic_ns", "<u8"),
        ("is_hold", "<u1"),
    ],
    align=True,
)
ACK_DTYPE = np.dtype(
    [
        ("session_generation", "<u8"),
        ("policy_epoch", "<u8"),
        ("observation_id", "<u8"),
        ("action_id", "<u8"),
        ("chunk_id", "<u8"),
        ("step_index", "<u4"),
        ("status", "<u1"),
        ("reject_reason", "<u2"),
        ("sdk_code", "<i4"),
        ("received_monotonic_ns", "<u8"),
        ("prepared_monotonic_ns", "<u8"),
        ("applied_monotonic_ns", "<u8"),
    ],
    align=True,
)

INFERENCE_CANDIDATE_DTYPE = np.dtype(
    [
        ("observation_id", "<u8"),
        ("session_generation", "<u8"),
        ("policy_epoch", "<u8"),
        ("action_id", "<u8"),
        ("chunk_id", "<u8"),
        ("step_index", "<u4"),
        ("chunk_length", "<u4"),
        ("created_monotonic_ns", "<u8"),
        ("target_monotonic_ns", "<u8"),
        ("valid_until_monotonic_ns", "<u8"),
        ("has_arm", "<u1"),
        ("has_hand", "<u1"),
        ("is_hold", "<u1"),
        ("arm_qpos", "<f8", (7,)),
        ("hand_qpos", "<f8", (12,)),
    ],
    align=True,
)

ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", (7,)),
        ("qvel", "<f8", (7,)),
        ("tau", "<f8", (7,)),
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
        ("qpos", "<f8", (12,)),
        ("current", "<f8", (12,)),
        ("tactile_sum", "<f8", (5, 3)),
        ("tactile_contact", "<u1", (5,)),
        ("error_state", "<u1"),
        ("connected", "<u1"),
        ("qpos_stale", "<u1"),
        ("commboard_err", "<i4", (12,)),
        ("jointboard_err", "<i4", (12,)),
        ("tipboard_err", "<i4", (12,)),
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
        ("tactile_force", "<f8", (5, 120, 3)),
        ("source_monotonic_ns", "<u8"),
        ("fresh", "<u1"),
        ("calibrated", "<u1"),
        ("unit_code", "<u1"),
    ]
)

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

COMPONENT_STATUS_DTYPE = np.dtype(
    [
        ("component", "S24"),
        ("phase", "<u1"),
        ("fault_code", "<u2"),
        ("exit_reason", "<u1"),
        ("generation", "<u8"),
        ("updated_monotonic_ns", "<u8"),
        ("detail", "S160"),
    ],
    align=True,
)

COMPONENT_METRICS_DTYPE = np.dtype(
    [
        ("component", "S16"),
        ("target_period_s", "<f8"),
        ("loop_count", "<u8"),
        ("last_work_duration_s", "<f8"),
        ("max_work_duration_s", "<f8"),
        ("deadline_overrun_count", "<u8"),
        ("missed_slot_count", "<u8"),
        ("long_block_reanchor_count", "<u8"),
        ("elapsed_s", "<f8"),
        ("actual_hz", "<f8"),
        ("updated_monotonic_ns", "<u8"),
    ],
    align=True,
)

RECORD_CONTROL_DTYPE = np.dtype(
    [
        ("command", "<u1"),
        ("generation", "<u8"),
        ("save", "<u1"),
        ("created_monotonic_ns", "<u8"),
        ("json_length", "<u4"),
        ("json_crc32", "<u4"),
        ("json_payload", f"S{RECORD_CONTROL_JSON_BYTES}"),
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
            ("arm_qpos", "<f8", (7,)),
            ("arm_qvel", "<f8", (7,)),
            ("arm_tau", "<f8", (7,)),
            ("eef_pos", "<f8", (3,)),
            ("eef_quat_wxyz", "<f8", (4,)),
            ("eef_rot6d", "<f8", (6,)),
            ("hand_qpos", "<f8", (12,)),
            ("hand_current", "<f8", (12,)),
            ("hand_tactile_sum", "<f8", (5, 3)),
            ("hand_tactile_force", "<f8", (5, 120, 3)),
            ("hand_tactile_contact", "<u1", (5,)),
            ("hand_tipboard_err", "<i4", (12,)),
            ("hand_commboard_err", "<i4", (12,)),
            ("hand_jointboard_err", "<i4", (12,)),
            ("hand_qpos_stale", "<u1"),
            ("fingertip_pos", "<f8", (5, 3)),
            ("arm_connected", "<u1"),
            ("hand_connected", "<u1"),
            ("state_timestamp", "<f8"),
            ("hand_error_state", "<u1"),
            ("arm_last_cmd_seq", "<u8"),
            ("arm_last_cmd_queue_latency_s", "<f8"),
            ("arm_last_cmd_apply_latency_s", "<f8"),
            ("arm_last_cmd_sdk_duration_s", "<f8"),
            ("arm_last_cmd_is_hold", "<u1"),
            ("action_arm_qpos", "<f8", (7,)),
            ("action_hand_qpos", "<f8", (12,)),
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
            ("json_length", "<u4"),
            ("json_crc32", "<u4"),
            ("json_payload", f"S{RECORD_SAMPLE_JSON_BYTES}"),
        ],
        align=True,
    )
