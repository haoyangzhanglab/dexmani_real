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
    """Return the newest verified frame whose source and publication precede *anchor*.

    Search the resident sequence range newest-first instead of snapshotting the
    whole ring.  A causal cut normally needs only the latest frame before its
    anchor; copying every slot makes the oldest slot race the next producer
    write, despite that old frame rarely being relevant to the result.
    """
    latest_sequence = int(ring.latest_sequence)
    oldest_sequence = max(1, latest_sequence - int(ring.maxlen) + 1)
    for target_sequence in range(latest_sequence, oldest_sequence - 1, -1):
        result = ring.read_sequence(target_sequence)
        if result is None:
            continue
        data, ring_publish_ns, sequence = result
        source_ns = int(data[source_field][0])
        names = data.dtype.names or ()
        publish_ns = (
            int(data["publish_monotonic_ns"][0])
            if "publish_monotonic_ns" in names
            and int(data["publish_monotonic_ns"][0]) > 0
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
                if (
                    0
                    < source_ns
                    <= receive_ns
                    <= publish_ns
                    <= int(anchor_monotonic_ns)
                ):
                    payload = shared.camera_ring.read_sequence(
                        int(sequence),
                        modalities=("rgb", "depth"),
                    )
                    if payload is not None:
                        result = (
                            header,
                            payload["rgb"],
                            payload["depth"],
                            int(sequence),
                        )
                        break
        if result is not None:
            header, rgb, depth, ring_sequence = result
            rec = header[0]
            return {
                "header": header,
                "rgb": rgb,
                "depth": depth,
                "ring_sequence": ring_sequence,
                "frame_number": int(rec["frame_number"]),
                "device_timestamp_s": float(rec["timestamp"]),
                "capture_monotonic_s": float(rec["capture_monotonic_s"]),
                "source_monotonic_ns": int(rec["source_monotonic_ns"]),
                "receive_monotonic_ns": int(rec["receive_monotonic_ns"]),
                "wait_return_monotonic_ns": int(rec["receive_monotonic_ns"]),
                "align_done_monotonic_ns": int(rec["align_done_monotonic_ns"]),
                "timestamp_domain": int(rec["timestamp_domain"]),
                "publish_monotonic_ns": int(rec["publish_monotonic_ns"]),
                "camera_generation": int(rec["camera_generation"]),
                "clock_reset": bool(rec["clock_reset"]),
                "duplicate": bool(rec["duplicate"]),
                "frame_gap": int(rec["frame_gap"]),
                "backlog_s": float(rec["backlog_s"]),
                "delivery_delay_above_floor_s": float(rec["backlog_s"]),
                "camera_health": int(rec["camera_health"]),
                "valid_depth_ratio": float(rec["pc_valid_depth_ratio"]),
            }
    except Exception:
        logger.warning("causal_reader: camera ring read failed", exc_info=True)
    return None
