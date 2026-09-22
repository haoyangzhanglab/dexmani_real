"""Stable NumPy wire schemas for cross-process channels.

This module deliberately imports only NumPy.  Shared-memory allocation,
policy logic, device workers, and recording serialization may depend on these
schemas; the schema layer must never depend on any of those implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)

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


# One current-run single-inflight command; ring sequence is not command identity.
ROBOT_COMMAND_DTYPE = np.dtype([
    ("command_id", "<u8"), ("run_id", "<u8"),
    ("issued_monotonic_ns", "<u8"),
    ("arm_present", "u1"), ("hand_present", "u1"),
    ("arm_qpos", "<f8", ARM_JOINT_SHAPE),
    ("hand_qpos", "<f8", HAND_JOINT_SHAPE),
], align=True)

# Raw v31 command truth. Zero timestamps never stand in for an inferred adoption.
RECORD_COMMAND_FIELDS = [
    ("command_id", "<u8"), ("command_run_id", "<u8"),
    ("command_issued_monotonic_ns", "<u8"),
    ("command_arm_present", "?"), ("command_hand_present", "?"),
    ("arm_command_adopted", "?"), ("hand_command_adopted", "?"),
    ("arm_command_adopted_monotonic_ns", "<u8"),
    ("hand_command_adopted_monotonic_ns", "<u8"),
]

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
        # Monotonic time immediately after the arm SDK accepted the command.
        ("last_adopted_monotonic_ns", "<u8"),
        # Exact command identity, also used for raw adoption accounting.
        ("last_adopted_run_id", "<u8"),
        ("last_adopted_command_id", "<u8"),
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
        ("tactile_aggregate", "<f4", HAND_TACTILE_SUM_SHAPE),
        # Aggregate and dense tactile payloads are zero-filled on read failure;
        # these bits distinguish an invalid sample from a valid zero-contact
        # sample. Both are gated on session software-bias readiness by the
        # worker before publication.
        ("tactile_aggregate_valid", "<u1"),
        ("tactile_dense", "<f4", HAND_TACTILE_FORCE_SHAPE),
        ("tactile_dense_valid", "<u1"),
        ("connected", "<u1"),
        # Set when qpos is held from the last read after a single-frame failure.
        ("qpos_stale", "<u1"),
        # Adoption follows the first accepted bounded setpoint; reached follows
        # acceptance of the exact endpoint. Neither proves physical convergence.
        ("last_adopted_run_id", "<u8"),
        ("last_adopted_command_id", "<u8"),
        ("last_adopted_monotonic_ns", "<u8"),
        ("last_reached_monotonic_ns", "<u8"),
        # Exact endpoint identity for operations that require target reach.
        ("last_reached_run_id", "<u8"),
        ("last_reached_command_id", "<u8"),
        # Monotonic time after the most recent accepted SDK setpoint, including
        # intermediate slew setpoints that have not reached the exact target.
        ("last_sdk_setpoint_accepted_monotonic_ns", "<u8"),
        ("commboard_err", "<i4", HAND_JOINT_SHAPE),
        ("jointboard_err", "<i4", HAND_JOINT_SHAPE),
        ("tipboard_err", "<i4", HAND_JOINT_SHAPE),
        ("source_monotonic_ns", "<u8"),
        ("publish_monotonic_ns", "<u8"),
        ("state_valid", "<u1"),
        ("timestamp", "<f8"),
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
        ("side", "<i4"),
        ("head_sequence_id", "<u8"),
        ("head_recv_ts_ns", "<u8"),
    ],
    align=True,
)

CAMERA_FRAME_HEADER_DTYPE = np.dtype(
    [
        ("source_monotonic_ns", "<u8"),
        ("receive_monotonic_ns", "<u8"),
        ("publish_monotonic_ns", "<u8"),
        ("camera_generation", "<u8"),
        ("depth_frame_number", "<u8"),
        ("color_frame_number", "<u8"),
        ("rgb_size", "<u8"),
        ("depth_size", "<u8"),
        ("rgb_shape_h", "<u4"),
        ("rgb_shape_w", "<u4"),
        ("rgb_shape_c", "<u4"),
        ("depth_shape_h", "<u4"),
        ("depth_shape_w", "<u4"),
        ("camera_health", "<u1"),
    ],
    align=True,
)


def make_record_sample_dtype(
    rgb_shape: tuple[int, int, int],
    depth_shape: tuple[int, int],
) -> np.dtype:
    """Return the control-step source row and fixed camera payload for the sample ring."""
    return np.dtype(
        [
            ("timestamp", "<f8"),
            ("arm_qpos", "<f8", ARM_JOINT_SHAPE),
            ("arm_qvel", "<f8", ARM_JOINT_SHAPE),
            ("arm_tau", "<f8", ARM_JOINT_SHAPE),
            ("hand_qpos", "<f8", HAND_JOINT_SHAPE),
            ("hand_current", "<f8", HAND_JOINT_SHAPE),
            ("hand_contact", "<f4", HAND_TACTILE_SUM_SHAPE),
            ("hand_contact_valid", "<u1"),
            ("hand_tactile_force", "<f4", HAND_TACTILE_FORCE_SHAPE),
            ("hand_tactile_force_valid", "<u1"),
            ("hand_qpos_stale", "<u1"),
            ("arm_connected", "<u1"),
            ("hand_connected", "<u1"),
            ("tracking_error", "<f8"),
            ("action_arm_joint_sent", "<f8", ARM_JOINT_SHAPE),
            ("action_hand_joint", "<f8", HAND_JOINT_SHAPE),
            ("action_arm_ee", "<f8", (9,)),
            *RECORD_COMMAND_FIELDS,
            ("flag_frame_status", "<u1"),
            ("observation_anchor_monotonic_ns", "<u8"),
            ("observation_valid", "<u1"),
            ("arm_source_monotonic_ns", "<u8"),
            ("hand_source_monotonic_ns", "<u8"),
            ("vr_source_monotonic_ns", "<u8"),
            ("camera_source_monotonic_ns", "<u8"),
            ("flag_camera_fresh", "<u1"),
            ("camera_health", "<u1"),
            ("camera_depth_frame_number", "<u8"),
            ("camera_color_frame_number", "<u8"),
            ("vr_wrist_pos", "<f8", (3,)),
            ("vr_wrist_rot6d", "<f8", (6,)),
            ("vr_landmarks", "<f8", (21, 3)),
            ("head_quat_wxyz", "<f8", (4,)),
            ("camera_present", "<u1"),
            ("camera_rgb", "<u1", rgb_shape),
            ("camera_depth", "<u2", depth_shape),
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
    "ROBOT_COMMAND_DTYPE",
    "HAND_STATE_DTYPE",
    "POINT_CLOUD_FEATURE_DIM",
    "SUPPORTED_POINT_CLOUD_COUNTS",
    "VR_FRAME_DTYPE",
    "make_pointcloud_frame_dtype",
    "make_record_sample_dtype",
    "nan_array",
    "validate_point_cloud_array",
]
