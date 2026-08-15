"""Causal read primitives for the shared-memory sensor and robot rings.

These are the public readers shared by the VR teleop snapshot builder
(``teleop/snapshot.py``) and the learned-policy observation builder
(``deployment``). The core invariant is ``0 < source_ns <= publish_ns <=
anchor`` for every selected frame; the camera ring adds an intermediate
``receive_ns`` layer (``source <= receive <= publish <= anchor``).

An ``anchor_monotonic_ns=None`` argument falls back to ``read_latest()`` as a
convenience for callers that do not need a causal cut; deployment observation
always passes an explicit anchor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.shm.shared_storage import read_arm_state as _read_arm_state_latest
from dexmani_real.shm.shared_storage import read_hand_state as _read_hand_state_latest
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def read_causal_structured_frame(
    ring: Any,
    *,
    source_field: str,
    anchor_monotonic_ns: int,
) -> tuple[np.ndarray, int, int] | None:
    """Return the newest frame whose source and publication precede *anchor*."""
    for data, ring_publish_ns, sequence in reversed(ring.get_last_k(ring.maxlen)):
        source_ns = int(data[source_field][0])
        names = data.dtype.names or ()
        publish_ns = (
            int(data["publish_monotonic_ns"][0])
            if "publish_monotonic_ns" in names and int(data["publish_monotonic_ns"][0]) > 0
            else int(ring_publish_ns)
        )
        if 0 < source_ns <= publish_ns <= anchor_monotonic_ns:
            return data, publish_ns, int(sequence)
    return None


def read_arm_state_causal(
    shared: SharedStorage, *, anchor_monotonic_ns: int | None = None
) -> np.ndarray | None:
    if anchor_monotonic_ns is None:
        return _read_arm_state_latest(shared)
    result = read_causal_structured_frame(
        shared.arm_state_ring,
        source_field="source_monotonic_ns",
        anchor_monotonic_ns=int(anchor_monotonic_ns),
    )
    return None if result is None else result[0]


def read_hand_state_causal(
    shared: SharedStorage, *, anchor_monotonic_ns: int | None = None
) -> np.ndarray | None:
    if anchor_monotonic_ns is None:
        return _read_hand_state_latest(shared)
    result = read_causal_structured_frame(
        shared.hand_state_ring,
        source_field="source_monotonic_ns",
        anchor_monotonic_ns=int(anchor_monotonic_ns),
    )
    return None if result is None else result[0]


def read_vr_frame_causal(
    shared: SharedStorage, *, anchor_monotonic_ns: int | None = None
) -> dict | None:
    """Read the latest or newest causal verified VR frame."""
    result = (
        shared.vr_ring.read_latest()
        if anchor_monotonic_ns is None
        else read_causal_structured_frame(
            shared.vr_ring,
            source_field="local_recv_ns",
            anchor_monotonic_ns=int(anchor_monotonic_ns),
        )
    )
    if result is None:
        return None
    data, publish_ns, sequence = result
    rec = data[0]
    return {
        "wrist_pos": np.asarray(rec["wrist_pos"], dtype=np.float64),
        "wrist_quat_wxyz": np.asarray(rec["wrist_quat_wxyz"], dtype=np.float64),
        "landmarks": np.asarray(rec["landmarks"], dtype=np.float64),
        "head_pos": np.asarray(rec["head_pos"], dtype=np.float64),
        "head_quat_wxyz": np.asarray(rec["head_quat_wxyz"], dtype=np.float64),
        "head_sequence_id": int(rec["head_sequence_id"]),
        "head_recv_ts_ns": int(rec["head_recv_ts_ns"]),
        "recv_ts_ns": int(rec["recv_ts_ns"]),
        "source_ts_ns": int(rec["source_ts_ns"]),
        "sequence_id": int(rec["sequence_id"]),
        "source_frame_seq": int(rec["source_frame_seq"]),
        "local_recv_ns": int(rec["local_recv_ns"]),
        "publish_monotonic_ns": int(publish_ns),
        "ring_sequence": int(sequence),
        "side": int(rec["side"]),
    }


def read_hand_tactile_causal(
    shared: SharedStorage, *, anchor_monotonic_ns: int | None = None
) -> np.ndarray | None:
    """Read the latest or newest causal hand tactile frame."""
    result = (
        shared.hand_tactile_ring.read_latest()
        if anchor_monotonic_ns is None
        else read_causal_structured_frame(
            shared.hand_tactile_ring,
            source_field="source_monotonic_ns",
            anchor_monotonic_ns=int(anchor_monotonic_ns),
        )
    )
    if result is None:
        return None
    data, _ts_ns, _seq = result
    return data


def read_camera_frame_causal(
    shared: SharedStorage, *, anchor_monotonic_ns: int | None = None
) -> dict | None:
    """Read the latest or newest causal camera frame. Returns None on failure."""
    try:
        if anchor_monotonic_ns is None:
            result = shared.camera_ring.read_latest()
        else:
            result = None
            for header, _publish_ns, sequence in reversed(
                shared.camera_ring.get_last_metadata(shared.camera_ring.maxlen)
            ):
                source_ns = int(header["source_monotonic_ns"][0])
                receive_ns = int(header["receive_monotonic_ns"][0])
                publish_ns = int(header["publish_monotonic_ns"][0]) or int(_publish_ns)
                if 0 < source_ns <= receive_ns <= publish_ns <= int(anchor_monotonic_ns):
                    payload = shared.camera_ring.read_sequence(
                        int(sequence),
                        modalities=("rgb", "depth", "pointcloud"),
                    )
                    if payload is not None:
                        result = (
                            header,
                            payload["rgb"],
                            payload["depth"],
                            payload["pointcloud"],
                            int(sequence),
                        )
                        break
        if result is not None:
            header, rgb, depth, pointcloud, ring_sequence = result
            rec = header[0]
            pointcloud_valid = bool(rec["pointcloud_valid"]) and int(rec["pc_num_points"]) > 0
            return {
                "header": header,
                "rgb": rgb,
                "depth": depth,
                "pointcloud": pointcloud,
                "ring_sequence": ring_sequence,
                "frame_number": int(rec["frame_number"]),
                "device_timestamp_s": float(rec["timestamp"]),
                "capture_monotonic_s": float(rec["capture_monotonic_s"]),
                "source_monotonic_ns": int(rec["source_monotonic_ns"]),
                "receive_monotonic_ns": int(rec["receive_monotonic_ns"]),
                "publish_monotonic_ns": int(rec["publish_monotonic_ns"]),
                "camera_generation": int(rec["camera_generation"]),
                "clock_reset": bool(rec["clock_reset"]),
                "duplicate": bool(rec["duplicate"]),
                "frame_gap": int(rec["frame_gap"]),
                "backlog_s": float(rec["backlog_s"]),
                "camera_health": int(rec["camera_health"]),
                "pointcloud_valid": pointcloud_valid,
                "pc_num_points": int(rec["pc_num_points"]),
                "pointcloud_source_point_count": int(rec["pc_source_point_count"]),
                "valid_depth_ratio": float(rec["pc_valid_depth_ratio"]),
                "pointcloud_padding_count": int(rec["pc_padding_count"]),
            }
    except Exception:
        logger.warning("causal_reader: camera ring read failed", exc_info=True)
    return None
