"""Numpy dtype definitions for shared memory data structures.

All dtypes use fixed-size fields — no objects, no variable-length strings —
so they are safe for placement in multiprocessing.shared_memory.SharedMemory.

Layout principles:
  - Little-endian (<) for x86_64 compatibility
  - All arrays are flat (no sub-dtypes with object refs)
  - Timestamps in uint64 nanoseconds
"""

from __future__ import annotations

import numpy as np

# ── VR Frame Layout ──
# Matches QuestHandTracker.convert_frame() output dict.
# Total size: ~600 bytes per frame

VR_FRAME_DTYPE = np.dtype(
    [
        # Wrist pose (FLU coordinate frame)
        ("wrist_pos", "<f8", (3,)),  # 3 × 8 = 24 bytes
        ("wrist_quat_wxyz", "<f8", (4,)),  # 4 × 8 = 32 bytes
        # Hand landmarks (21 points × 3 coordinates)
        ("landmarks", "<f8", (21, 3)),  # 21 × 3 × 8 = 504 bytes
        # Timestamps
        ("recv_ts_ns", "<u8"),  # HTS SDK receive timestamp (ns)
        ("source_ts_ns", "<u8"),  # HTS source timestamp (ns)
        ("sequence_id", "<u8"),  # Frame sequence ID
        ("source_frame_seq", "<u8"),  # Source frame sequence
        ("local_recv_ns", "<u8"),  # Local monotonic_ns at receive time
        # Metadata
        ("side", "<i4"),  # Hand side enum value
    ],
    align=True,
)

# ── Camera Frame Layout (per camera) ──
# RGB + Depth for a standard RealSense frame.
# RGB:  H×W×3 uint8
# Depth: H×W uint16
# Timestamps: float64
#
# Note: Camera frames are LARGE (~1.5MB each). The SharedMemoryRingBuffer
# stores N slots of raw camera bytes. We use a header struct for metadata
# and a raw byte array for the actual image data.

CAMERA_FRAME_HEADER_DTYPE = np.dtype(
    [
        ("timestamp", "<f8"),  # Camera frame timestamp (perf_counter)
        ("frame_number", "<u8"),  # Monotonic frame counter
        ("rgb_size", "<u8"),  # Number of bytes in RGB array
        ("depth_size", "<u8"),  # Number of bytes in Depth array
        ("rgb_shape_h", "<u4"),  # RGB height
        ("rgb_shape_w", "<u4"),  # RGB width
        ("rgb_shape_c", "<u4"),  # RGB channels (always 3)
        ("depth_shape_h", "<u4"),  # Depth height
        ("depth_shape_w", "<u4"),  # Depth width
        ("pad", "<u4", (3,)),  # Padding to 64-byte alignment
    ],
    align=True,
)

# ── Ring buffer slot layout ──
# Each slot in a ring buffer has a timestamp and the data payload.
# The write_idx (global atomic counter) is stored at offset 0 of the
# shared memory block.

RING_SLOT_HEADER_DTYPE = np.dtype(
    [
        ("timestamp_ns", "<u8"),  # Monotonic timestamp when written
        ("sequence", "<u8"),  # Monotonic sequence counter
    ],
    align=True,
)


def vr_frame_to_array(frame: dict) -> np.ndarray:
    """Convert a VR frame dict (from QuestHandTracker) to a structured array.

    Returns a 0-d array (scalar) of VR_FRAME_DTYPE.
    """
    arr = np.zeros((), dtype=VR_FRAME_DTYPE)
    arr["wrist_pos"] = np.asarray(frame["wrist_pos"], dtype=np.float64)
    arr["wrist_quat_wxyz"] = np.asarray(frame["wrist_quat_wxyz"], dtype=np.float64)
    arr["landmarks"] = np.asarray(frame["landmarks"], dtype=np.float64).reshape(21, 3)
    arr["recv_ts_ns"] = np.uint64(frame.get("recv_ts_ns", 0))
    arr["source_ts_ns"] = np.uint64(frame.get("source_ts_ns", 0))
    arr["sequence_id"] = np.uint64(frame.get("sequence_id", 0))
    arr["source_frame_seq"] = np.uint64(frame.get("source_frame_seq", 0))
    arr["local_recv_ns"] = np.uint64(frame.get("local_recv_ns", 0))
    arr["side"] = np.int32(frame.get("side", -1))
    return arr


def array_to_vr_frame(arr: np.ndarray) -> dict:
    """Convert a structured array back to a VR frame dict.

    Returns a dict matching QuestHandTracker.get_latest() output format
    (copying numpy arrays by value for thread safety).
    """
    return {
        "side": int(arr["side"]),
        "wrist_pos": arr["wrist_pos"].copy(),
        "wrist_quat_wxyz": arr["wrist_quat_wxyz"].copy(),
        "landmarks": arr["landmarks"].copy().reshape(21, 3),
        "recv_ts_ns": int(arr["recv_ts_ns"]),
        "source_ts_ns": int(arr["source_ts_ns"]),
        "sequence_id": int(arr["sequence_id"]),
        "source_frame_seq": int(arr["source_frame_seq"]),
        "coordinate_frame": "flu",  # default
        "local_recv_ns": int(arr["local_recv_ns"]),
    }


def camera_frame_to_bytes(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert a camera frame dict to header + raw bytes arrays.

    Returns (header_array, rgb_bytes, depth_bytes) where:
      - header_array: 1-d array of CAMERA_FRAME_HEADER_DTYPE (1 element)
      - rgb_bytes: raw uint8 array of RGB data
      - depth_bytes: raw uint16 array of depth data
    """
    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["timestamp"] = np.float64(frame.get("timestamp", 0.0))
    header["frame_number"] = np.uint64(frame.get("frame_id", 0))

    rgb = np.asarray(frame.get("rgb"), dtype=np.uint8)
    depth = np.asarray(frame.get("depth"), dtype=np.uint16)

    header["rgb_size"] = np.uint64(rgb.nbytes)
    header["depth_size"] = np.uint64(depth.nbytes)
    header["rgb_shape_h"] = np.uint32(rgb.shape[0])
    header["rgb_shape_w"] = np.uint32(rgb.shape[1])
    header["rgb_shape_c"] = np.uint32(rgb.shape[2])
    header["depth_shape_h"] = np.uint32(depth.shape[0])
    header["depth_shape_w"] = np.uint32(depth.shape[1])

    return header, rgb, depth


def bytes_to_camera_frame(
    header: np.ndarray, rgb_bytes: np.ndarray, depth_bytes: np.ndarray
) -> dict:
    """Reconstruct a camera frame dict from raw bytes.

    Returns a dict matching CameraProcess.poll_latest_frame() output format.
    """
    h = header[0]
    rgb = rgb_bytes.reshape(
        (int(h["rgb_shape_h"]), int(h["rgb_shape_w"]), int(h["rgb_shape_c"]))
    ).copy()
    depth = depth_bytes.reshape(
        (int(h["depth_shape_h"]), int(h["depth_shape_w"]))
    ).copy()
    return {
        "rgb": rgb,
        "depth": depth,
        "timestamp": float(h["timestamp"]),
        "frame_number": int(h["frame_number"]),
    }
