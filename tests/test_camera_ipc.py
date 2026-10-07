"""Shared-memory tests on the local platform, without camera/VR connections."""

import json
import multiprocessing as mp
import uuid

import numpy as np
import pytest

from dexmani_real.ipc.camera_ring import CameraRingBuffer
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.ipc.ring import SeqlockSlot
from dexmani_real.sensor.camera.worker import pack_camera_frame


def _read_spawn(ring, queue):
    try:
        header, rgb, depth, sequence = ring.read_latest()
        queue.put(
            (rgb.shape, depth.shape, sequence, int(header["timestamp_ns"][0]), int(rgb.sum()))
        )
    finally:
        ring.close()


def frame(value=1):
    return pack_camera_frame(
        np.full((8, 12, 3), value, np.uint8),
        np.full((8, 12), 500, np.uint16),
        timestamp_ns=123,
        depth_frame_number=4,
        color_frame_number=5,
    )


def test_fixed_layout_attach_spawn_owned_copy_and_torn_read():
    ring = CameraRingBuffer("test_" + uuid.uuid4().hex, (8, 12, 3), (8, 12), maxlen=2)
    attached = None
    process = None
    try:
        assert ring.read_latest() is None
        assert ring.write(*frame()) == 1
        attached = CameraRingBuffer(ring.name, create=False)
        assert attached.maxlen == 2
        header, rgb, depth, seq = attached.read_latest()
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(target=_read_spawn, args=(attached, queue))
        process.start()
        assert queue.get(timeout=15) == ((8, 12, 3), (8, 12), 1, 123, 288)
        process.join(15)
        assert process.exitcode == 0
        queue.close()
        queue.join_thread()
        ring.write(*frame(2))
        ring.write(*frame(3))
        assert rgb.sum() == 288 and depth.shape == (8, 12)
        assert attached.read_sequence(seq) is None
        assert int(attached.read_sequence(3)["rgb"].sum()) == 864
        assert int(header["timestamp_ns"][0]) == 123
        slot = SeqlockSlot(ring._shm.buf, ring._HEADER_SIZE + (3 % 2) * ring._slot_size)
        slot.begin_write(5, 0)
        assert attached.read_latest() is None
        assert attached.read_sequence(3) is None
        del slot
        with pytest.raises(ValueError):
            ring.write(frame()[0], np.zeros((12, 8, 3), np.uint8), np.zeros((12, 8), np.uint16))
        with pytest.raises(ValueError):
            ring.write(frame()[0], frame()[1][:, ::-1], frame()[2])
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join()
        if attached is not None:
            attached.close()
        ring.close()
        ring.unlink()


@pytest.mark.parametrize(
    "camera,vr,cloud",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_only_selected_rings_exist(camera, vr, cloud):
    cfg = RuntimeChannelsConfig(
        camera=camera,
        vr=vr,
        pointcloud=cloud,
        camera_rgb_shape=(8, 12, 3),
        camera_depth_shape=(8, 12),
        pointcloud_num_points=16,
    )
    shared = RuntimeChannels.create(prefix="test_" + uuid.uuid4().hex, config=cfg)
    try:
        assert (
            shared.camera_ring is not None,
            shared.vr_ring is not None,
            shared.pointcloud_ring is not None,
        ) == (camera, vr, cloud)
    finally:
        assert shared.close()
        assert shared.close()


def test_partial_allocation_unlinks_camera(monkeypatch):
    from multiprocessing import shared_memory

    from dexmani_real.ipc import channels

    name = "test_" + uuid.uuid4().hex

    def fail(*a, **kw):
        raise MemoryError("VR allocation failed")

    monkeypatch.setattr(channels, "SharedMemoryRingBuffer", fail)
    with pytest.raises(MemoryError):
        RuntimeChannels.create(
            prefix=name,
            config=RuntimeChannelsConfig(
                camera=True, vr=True, camera_rgb_shape=(8, 12, 3), camera_depth_shape=(8, 12)
            ),
        )
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name + "_camera")
    with pytest.raises(ValueError, match="requires camera"):
        RuntimeChannelsConfig(pointcloud=True)


def test_odd_sized_layout_and_attach_validation():
    ring = CameraRingBuffer("test_" + uuid.uuid4().hex, (3, 3, 3), (3, 3), maxlen=2)
    try:
        assert ring._slot_size % 8 == 0
        header = frame()[0]
        rgb = np.full((3, 3, 3), 11, np.uint8)
        depth = np.full((3, 3), 500, np.uint16)
        ring.write(header, rgb, depth)
        np.testing.assert_array_equal(ring.read_latest()[2], depth)
        with pytest.raises(ValueError, match="RGB layout differs"):
            CameraRingBuffer(ring.name, rgb_shape=(8, 12, 3), create=False)
        np.ndarray((8,), dtype=np.uint64, buffer=ring._shm.buf)[7] = 0
        with pytest.raises(ValueError, match="initialization layout"):
            CameraRingBuffer(ring.name, create=False)
    finally:
        ring.close()
        ring.unlink()


def test_camera_metadata_shared_by_cloud_and_recording(tmp_path):
    from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
    from dexmani_real.config.experiment import ExperimentConfig
    from dexmani_real.recording.recorder import snapshot_recording_metadata
    from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry
    from dexmani_real.sensor.pointcloud_worker import _load_static_inputs

    path = tmp_path / "camera.json"
    path.write_text(
        json.dumps(
            {
                "camera": dict(
                    serial="synthetic",
                    type="eye_to_hand",
                    pose=dict(position=[0.4, 0, 0], orientation=[1, 0, 0, 0]),
                )
            }
        )
    )
    calibration = CameraExtrinsics(path)
    intrinsics = CameraIntrinsics(12, 8, 100.0, 100.0, 6.0, 4.0, "none", (0.0,) * 5)
    geometry = RGBDGeometry(intrinsics, intrinsics, np.eye(4))
    shared = RuntimeChannels.create(
        prefix="test_" + uuid.uuid4().hex,
        config=RuntimeChannelsConfig(
            camera=True,
            pointcloud=True,
            pointcloud_num_points=16,
            camera_rgb_shape=(8, 12, 3),
            camera_depth_shape=(8, 12),
        ),
    )
    try:
        shared.camera_serial.value = b"synthetic"
        shared.camera_geometry.value = json.dumps(geometry.to_dict()).encode("utf-8")
        shared.camera_depth_scale.value = 0.001
        shared.camera_ready.set()
        cloud_geometry, scale, transform = _load_static_inputs(shared, calibration)
        recorded = snapshot_recording_metadata(
            shared,
            ExperimentConfig(),
            collection_source="teleop",
            camera_calibration=calibration,
        )
        assert scale == recorded["depth_scale"] == 0.001
        assert (
            cloud_geometry.to_dict()
            == recorded["camera_geometry"].aligned_depth_to_color().to_dict()
        )
        np.testing.assert_array_equal(transform, recorded["camera_T_xarm_base_from_color"])
        np.testing.assert_array_equal(transform[:3, 3], [0.4, 0, 0])
    finally:
        assert shared.close()
