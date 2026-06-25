"""Shared memory infrastructure for zero-copy cross-process communication.

Provides lock-free FILO ring buffers, numpy dtype layouts, and a centralized
frame manager for VR and camera data streams.

PID process channels (PIDStateChannel, PIDTargetChannel) have been removed —
the inner loop now runs as an in-process thread (see robot/inner_loop.py).
"""

from dexmani_real.shm.frame_manager import SharedMemoryFrameManager
from dexmani_real.shm.layouts import (
    CAMERA_FRAME_HEADER_DTYPE,
    VR_FRAME_DTYPE,
    array_to_vr_frame,
    bytes_to_camera_frame,
    camera_frame_to_bytes,
    vr_frame_to_array,
)
from dexmani_real.shm.ring_buffer import CameraRingBuffer, SharedMemoryRingBuffer

__all__ = [
    "array_to_vr_frame",
    "bytes_to_camera_frame",
    "CAMERA_FRAME_HEADER_DTYPE",
    "CameraRingBuffer",
    "camera_frame_to_bytes",
    "SharedMemoryFrameManager",
    "SharedMemoryRingBuffer",
    "VR_FRAME_DTYPE",
    "vr_frame_to_array",
]
