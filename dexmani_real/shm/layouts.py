"""VR Frame dtype definition — shared by frame_manager.py (legacy VRReceiverProcess).

Camera frame dtypes moved to ring_buffer.py (Phase 2.7).
"""

from __future__ import annotations

import numpy as np

# ── VR Frame Layout (~600 bytes) ──

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


def vr_frame_to_array(frame: dict) -> np.ndarray:
    """Convert a VR frame dict to a structured array (0-d scalar)."""
    arr = np.zeros((), dtype=VR_FRAME_DTYPE)
    arr["wrist_pos"] = np.asarray(frame["wrist_pos"], dtype=np.float64)
    arr["wrist_quat_wxyz"] = np.asarray(frame["wrist_quat_wxyz"], dtype=np.float64)
    arr["landmarks"] = np.asarray(frame["landmarks"], dtype=np.float64).reshape(21, 3)
    arr["head_pos"] = np.asarray(frame.get("head_pos", np.zeros(3)), dtype=np.float64)
    arr["head_quat_wxyz"] = np.asarray(frame.get("head_quat_wxyz", np.zeros(4)), dtype=np.float64)
    arr["recv_ts_ns"] = np.uint64(frame.get("recv_ts_ns") or 0)
    arr["source_ts_ns"] = np.uint64(frame.get("source_ts_ns") or 0)
    arr["sequence_id"] = np.uint64(frame.get("sequence_id") or 0)
    arr["source_frame_seq"] = np.uint64(frame.get("source_frame_seq") or 0)
    arr["local_recv_ns"] = np.uint64(frame.get("local_recv_ns") or 0)
    arr["side"] = np.int32(frame.get("side") if frame.get("side") is not None else -1)  # type: ignore[arg-type]
    return arr


def array_to_vr_frame(arr: np.ndarray) -> dict:
    """Convert a structured array back to a VR frame dict."""
    return {
        "side": int(arr["side"]),
        "wrist_pos": arr["wrist_pos"].copy(),
        "wrist_quat_wxyz": arr["wrist_quat_wxyz"].copy(),
        "landmarks": arr["landmarks"].copy().reshape(21, 3),
        "head_pos": arr["head_pos"].copy(),
        "head_quat_wxyz": arr["head_quat_wxyz"].copy(),
        "recv_ts_ns": int(arr["recv_ts_ns"]),
        "source_ts_ns": int(arr["source_ts_ns"]),
        "sequence_id": int(arr["sequence_id"]),
        "source_frame_seq": int(arr["source_frame_seq"]),
        "coordinate_frame": "flu",
        "local_recv_ns": int(arr["local_recv_ns"]),
    }
