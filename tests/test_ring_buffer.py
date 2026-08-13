"""Focused seqlock-ring tests — lock in zero-behavior-change for the SeqlockSlot refactor.

The harness tests exercise ``SharedMemoryRingBuffer`` indirectly through the arm
state ring, but neither ring's seqlock write/read protocol nor ``CameraRingBuffer``
had direct coverage.  These tests pin the byte-level semantics that Phase 2.1
(seqlock dedup) must preserve.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from dexmani_real.sensor.camera_process import pack_camera_frame
from dexmani_real.shm.ring_buffer import CameraRingBuffer, SharedMemoryRingBuffer


def _unique_name(tag: str) -> str:
    return f"ring_test_{mp.current_process().pid}_{time.monotonic_ns()}_{tag}"


def _close_unlink(ring) -> None:
    ring.close()
    ring.unlink()


# ---------------------------------------------------------------------------
# SharedMemoryRingBuffer
# ---------------------------------------------------------------------------

_FRAME_DTYPE = np.dtype([("value", "<f8"), ("tag", "<u4")])


def _frame(value: float, tag: int) -> np.ndarray:
    f = np.zeros((), dtype=_FRAME_DTYPE)
    f["value"] = value
    f["tag"] = tag
    return f


def test_shared_ring_write_read_latest():
    ring = SharedMemoryRingBuffer(_unique_name("smr"), _FRAME_DTYPE, maxlen=3, create=True)
    try:
        assert ring.read_latest() is None
        assert ring.frame_age_ns() == -1
        assert ring.latest_sequence == 0

        seq1 = ring.write(_frame(1.5, 7))
        seq2 = ring.write(_frame(2.5, 8))

        assert seq1 == 1 and seq2 == 2
        assert ring.latest_sequence == 2

        data, ts, seq = ring.read_latest()
        assert seq == 2
        assert data.shape == (1,)
        assert data[0]["value"] == 2.5
        assert data[0]["tag"] == 8
        assert ts > 0
        assert ring.frame_age_ns() >= 0
    finally:
        _close_unlink(ring)


def test_shared_ring_get_last_k_oldest_first_and_wraparound():
    ring = SharedMemoryRingBuffer(_unique_name("smr"), _FRAME_DTYPE, maxlen=3, create=True)
    try:
        for v in range(5):
            ring.write(_frame(float(v), v))

        frames = ring.get_last_k(3)
        assert [f[2] for f in frames] == [3, 4, 5]  # logical sequences, oldest first
        assert [f[0][0]["value"] for f in frames] == [2.0, 3.0, 4.0]

        # Requesting more than capacity raises.
        try:
            ring.get_last_k(4)
            raise AssertionError("expected ValueError for k > maxlen")
        except ValueError:
            pass
    finally:
        _close_unlink(ring)


def test_shared_ring_get_last_k_empty_and_nonpositive():
    ring = SharedMemoryRingBuffer(_unique_name("smr"), _FRAME_DTYPE, maxlen=3, create=True)
    try:
        assert ring.get_last_k(3) == []
        assert ring.get_last_k(0) == []
    finally:
        _close_unlink(ring)


# ---------------------------------------------------------------------------
# CameraRingBuffer
# ---------------------------------------------------------------------------

_RGB_SHAPE = (4, 6, 3)
_DEPTH_SHAPE = (4, 6)


def _camera_payload(frame_id: int):
    rgb = np.arange(np.prod(_RGB_SHAPE), dtype=np.uint8).reshape(_RGB_SHAPE)
    depth = np.arange(np.prod(_DEPTH_SHAPE), dtype=np.uint16).reshape(_DEPTH_SHAPE)
    header, rgb_bytes, depth_bytes = pack_camera_frame(
        rgb,
        depth,
        timestamp=1.0,
        capture_monotonic_s=0.001 * frame_id,
        frame_id=frame_id,
    )
    return header, rgb_bytes, depth_bytes


def test_camera_ring_write_read_latest():
    ring = CameraRingBuffer(
        _unique_name("cam"),
        rgb_shape=_RGB_SHAPE,
        depth_shape=_DEPTH_SHAPE,
        maxlen=3,
        create=True,
    )
    try:
        assert ring.read_latest() is None
        assert ring.frame_age_ns() == -1

        header, rgb, depth = _camera_payload(0)
        seq = ring.write(header, rgb, depth)
        assert seq == 1

        got = ring.read_latest()
        assert got is not None
        h, r, d, pc, s = got
        assert s == 1
        assert pc is None  # no pointcloud capacity
        assert h["frame_number"] == 0
        assert np.array_equal(r, rgb)
        assert np.array_equal(d, depth)
        assert ring.frame_age_ns() >= 0
    finally:
        _close_unlink(ring)


def test_camera_ring_get_last_metadata_and_read_sequence():
    ring = CameraRingBuffer(
        _unique_name("cam"),
        rgb_shape=_RGB_SHAPE,
        depth_shape=_DEPTH_SHAPE,
        maxlen=3,
        create=True,
    )
    try:
        seqs = []
        for i in range(2):
            header, rgb, depth = _camera_payload(i)
            seqs.append(ring.write(header, rgb, depth))

        meta = ring.get_last_metadata(2)
        assert [m[2] for m in meta] == seqs  # oldest first

        payload = ring.read_sequence(seqs[-1])
        assert payload is not None
        assert payload["header"]["frame_number"] == 1
        assert "rgb" in payload and "depth" in payload
        assert "pointcloud" not in payload  # no pointcloud capacity

        assert ring.read_sequence(0) is None  # sequence <= 0
        assert ring.read_sequence(999) is None  # not resident / non-matching marker
    finally:
        _close_unlink(ring)
