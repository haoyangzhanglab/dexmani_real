"""NumPy cross-process schemas, with recording rows based on the raw layout."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)

# Realtime point-cloud transport is fixed-size per deployment.
POINT_CLOUD_FEATURE_DIM = 6


def make_pointcloud_frame_dtype(num_points: int) -> np.dtype:
    """Return the latest-only realtime point-cloud IPC schema."""
    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, (int, np.integer))
        or num_points <= 0
    ):
        raise ValueError(f"point-cloud count must be a positive integer, got {num_points!r}")
    count = int(num_points)
    return np.dtype(
        [
            ("source_camera_sequence", "<u8"),
            ("timestamp_ns", "<u8"),
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
    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, (int, np.integer))
        or num_points <= 0
    ):
        raise ValueError("num_points must be a positive integer")
    array = np.asarray(value)
    expected_shape = (int(num_points), POINT_CLOUD_FEATURE_DIM)
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ValueError(
            f"{label} must be float32 {expected_shape}, got shape={array.shape} dtype={array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN/Inf")
    if np.any(array[:, 3:] < 0.0) or np.any(array[:, 3:] > 1.0):
        raise ValueError(f"{label} RGB values must be in [0,1]")
    return array


# Latest absolute target. Presence bits support arm-only manual workflows.
ROBOT_COMMAND_DTYPE = np.dtype(
    [
        ("run_id", "<u8"),
        ("arm_present", "u1"),
        ("hand_present", "u1"),
        ("arm_qpos", "<f8", ARM_JOINT_SHAPE),
        ("hand_qpos", "<f8", HAND_JOINT_SHAPE),
    ],
    align=True,
)

ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", ARM_JOINT_SHAPE),
        ("qvel", "<f8", ARM_JOINT_SHAPE),
        ("effort", "<f8", ARM_JOINT_SHAPE),
        ("timestamp_ns", "<u8"),
    ]
)
HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", HAND_JOINT_SHAPE),
        ("current", "<f8", HAND_JOINT_SHAPE),
        ("tactile_aggregate", "<f4", HAND_TACTILE_SUM_SHAPE),
        ("tactile_aggregate_valid", "u1"),
        ("tactile_dense", "<f4", HAND_TACTILE_FORCE_SHAPE),
        ("tactile_dense_valid", "u1"),
        ("timestamp_ns", "<u8"),
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
        ("timestamp_ns", "<u8"),
        ("depth_frame_number", "<u8"),
        ("color_frame_number", "<u8"),
        ("rgb_size", "<u8"),
        ("depth_size", "<u8"),
        ("rgb_shape_h", "<u4"),
        ("rgb_shape_w", "<u4"),
        ("rgb_shape_c", "<u4"),
        ("depth_shape_h", "<u4"),
        ("depth_shape_w", "<u4"),
    ],
    align=True,
)


def make_record_sample_dtype(rgb_shape, depth_shape) -> np.dtype:
    """Transport a raw row and its RGB-D payload to RecorderIO."""
    from dexmani_real.recording.storage.schema import DATASET_SPECS

    return np.dtype(
        [(name, spec.dtype, spec.tail_shape) for name, spec in DATASET_SPECS.items()]
        + [
            ("camera_present", "u1"),
            ("camera_rgb", "u1", rgb_shape),
            ("camera_depth", "<u2", depth_shape),
        ],
        align=True,
    )


__all__ = [
    "ARM_STATE_DTYPE",
    "CAMERA_FRAME_HEADER_DTYPE",
    "ROBOT_COMMAND_DTYPE",
    "HAND_STATE_DTYPE",
    "POINT_CLOUD_FEATURE_DIM",
    "VR_FRAME_DTYPE",
    "make_pointcloud_frame_dtype",
    "make_record_sample_dtype",
    "validate_point_cloud_array",
]
