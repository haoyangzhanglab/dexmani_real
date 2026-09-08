"""Offline camera admission and shared-memory ownership after telemetry removal."""

from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import numpy as np
import pytest

from dexmani_real.deployment.inference.observation import _read_rgb_history
from dexmani_real.ipc.camera_ring import CameraRingBuffer
from dexmani_real.ipc.causal import read_camera_frame_causal
from dexmani_real.ipc.ring import SeqlockSlot
from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE
from dexmani_real.sensor.camera.worker import (
    CameraHealth,
    _camera_health,
    pack_camera_frame,
)
from dexmani_real.sensor.pointcloud_worker import _camera_frame_is_usable
from dexmani_real.teleop.control_loop.camera_freshness import CameraFreshnessTracker


@pytest.fixture
def ring():
    camera = CameraRingBuffer(
        "dexmani_camera_test_" + uuid4().hex,
        rgb_shape=(4, 4, 3),
        depth_shape=(4, 4),
        maxlen=4,
    )
    try:
        yield camera
    finally:
        camera.close()
        camera.unlink()


def _write(ring, *, source=100, publish=120, generation=1, health=CameraHealth.OK):
    header, rgb, depth = pack_camera_frame(
        np.ones((4, 4, 3), np.uint8),
        np.ones((4, 4), np.uint16),
        depth_frame_number=source,
        color_frame_number=source,
        camera_health=int(health),
        source_monotonic_ns=source,
        receive_monotonic_ns=source + 1,
        camera_generation=generation,
    )
    with mock.patch(
        "dexmani_real.ipc.camera_ring.time.monotonic_ns", return_value=publish
    ):
        return ring.write(header, rgb, depth)


def test_only_live_header_and_attach_layout_remain(ring):
    assert set(CAMERA_FRAME_HEADER_DTYPE.names) == {
        "source_monotonic_ns",
        "receive_monotonic_ns",
        "publish_monotonic_ns",
        "camera_generation",
        "depth_frame_number",
        "color_frame_number",
        "camera_health",
        "rgb_size",
        "depth_size",
        "rgb_shape_h",
        "rgb_shape_w",
        "rgb_shape_c",
        "depth_shape_h",
        "depth_shape_w",
    }
    sequence = _write(ring)
    attached = CameraRingBuffer(ring.name, maxlen=4, create=False)
    try:
        frame = attached.read_sequence(sequence)
        assert frame["rgb"].shape == (4, 4, 3)
        assert frame["depth"].shape == (4, 4)
    finally:
        attached.close()


@pytest.mark.parametrize(
    "reset,duplicate,delay,expected",
    [
        (False, False, 0.0, CameraHealth.OK),
        (True, False, 0.0, CameraHealth.CLOCK_RESET),
        (True, True, 1.0, CameraHealth.CLOCK_RESET),
        (False, True, 0.0, CameraHealth.DUPLICATE),
        (False, False, 0.3, CameraHealth.DELIVERY_DELAY),
        (False, False, float("nan"), CameraHealth.DELIVERY_DELAY),
    ],
)
def test_health_owns_reset_duplicate_and_delay(ring, reset, duplicate, delay, expected):
    health = _camera_health(
        clock_reset=reset, duplicate=duplicate, backlog_s=delay, max_age_s=0.25
    )
    assert health == expected
    _write(ring, health=health)
    header = ring.read_latest()[0]
    usable = _camera_frame_is_usable(header, now_ns=130, max_input_age_ns=100)
    history = _read_rgb_history(
        SimpleNamespace(camera_ring=ring),
        anchor_ns=130,
        max_age_ns=100,
        history_len=1,
        not_before_ns=0,
    )
    assert usable == (expected == CameraHealth.OK)
    assert bool(history) == usable


def test_skipped_device_numbers_keep_current_healthy_frame(ring):
    _write(ring, source=100, publish=120)
    _write(ring, source=1000, publish=1020)
    frame = ring.read_latest()[0]
    assert _camera_frame_is_usable(frame, now_ns=1030, max_input_age_ns=100)


def test_causal_reader_and_pointcloud_reject_future_and_stale(ring):
    _write(ring)
    shared = SimpleNamespace(camera_ring=ring)
    assert read_camera_frame_causal(shared, anchor_monotonic_ns=130) is not None
    assert read_camera_frame_causal(shared, anchor_monotonic_ns=90) is None
    header = ring.read_latest()[0]
    assert not _camera_frame_is_usable(header, now_ns=90, max_input_age_ns=100)
    assert not _camera_frame_is_usable(header, now_ns=1000, max_input_age_ns=100)


def test_generation_boundary_keeps_only_new_camera_history(ring):
    _write(ring, source=100, publish=120, generation=1)
    _write(ring, source=140, publish=160, generation=2)
    frames = _read_rgb_history(
        SimpleNamespace(camera_ring=ring),
        anchor_ns=170,
        max_age_ns=100,
        history_len=4,
        not_before_ns=0,
    )
    assert len(frames) == 1
    assert frames[0].camera_generation == 2


def test_camera_torn_slot_is_never_returned(ring):
    sequence = _write(ring)
    slot = SeqlockSlot(
        ring._shm.buf, ring._HEADER_SIZE + sequence % ring.maxlen * ring._slot_size
    )
    slot.begin_write(sequence, 0)
    assert ring.read_sequence(sequence) is None
    assert ring.read_latest() is None
    del slot


@pytest.mark.parametrize(
    "source_ns,healthy",
    [(0, False), (2_100_000_000, False), (1_000_000_000, False), (1_900_000_000, True)],
)
def test_freshness_requires_positive_causal_recent_source(source_ns, healthy):
    tracker = CameraFreshnessTracker(max_age_s=0.25, abort_after_s=1.0)
    tracker.reset(0.5)
    frame, stalled = tracker.observe(
        {
            "ring_sequence": 1,
            "depth_frame_number": 1,
            "source_monotonic_ns": source_ns,
            "camera_health": 0,
        },
        now_s=2.0,
    )
    assert frame["camera_fresh"] == healthy
    assert not stalled
