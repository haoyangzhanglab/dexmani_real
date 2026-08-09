from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

import dexmani_real.recording.episode_recorder as recorder_module
from dexmani_real.policy.vr_teleop_policy import CameraFreshnessTracker
from dexmani_real.recording.camera_stream_writer import (
    CameraStreamWriter,
    CameraStreamWriterConfig,
    CameraStreamWriterError,
)
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.recording.transaction import atomic_publish
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.sensor.camera_process import pack_camera_frame
from dexmani_real.shm.ring_buffer import CameraRingBuffer
from dexmani_real.tools.episode_quality import EpisodeQuality


def _camera_frame(sequence: int, frame_number: int, capture_s: float) -> dict:
    return {
        "ring_sequence": sequence,
        "frame_number": frame_number,
        "capture_monotonic_s": capture_s,
        "camera_health": 0,
    }


def test_camera_freshness_rejects_duplicate_old_and_cross_episode_frames() -> None:
    tracker = CameraFreshnessTracker(max_age_s=0.25, abort_after_s=2.0)
    tracker.reset(10.0)

    frame, stalled = tracker.observe(_camera_frame(1, 100, 10.01), now_s=10.05)
    assert frame is not None and frame["camera_fresh"] is True
    assert frame["camera_age_s"] == pytest.approx(0.04)
    assert not stalled

    frame, stalled = tracker.observe(_camera_frame(1, 100, 10.01), now_s=10.10)
    assert frame is not None and frame["camera_fresh"] is False
    assert not stalled

    frame, stalled = tracker.observe(_camera_frame(2, 101, 9.0), now_s=12.11)
    assert frame is not None and frame["camera_fresh"] is False
    assert stalled

    # A successful capture after a transient read-failure/stall resets the
    # continuous-stale timer instead of poisoning the rest of the session.
    frame, stalled = tracker.observe(_camera_frame(3, 102, 12.12), now_s=12.15)
    assert frame is not None and frame["camera_fresh"] is True
    assert not stalled
    assert tracker.stale_since_s is None

    tracker.reset(20.0)
    frame, stalled = tracker.observe(_camera_frame(4, 103, 19.99), now_s=20.01)
    assert frame is not None and frame["camera_fresh"] is False
    assert not stalled


def test_camera_ring_strict_validation_and_metadata_round_trip() -> None:
    name = f"test_camera_{uuid.uuid4().hex}"
    ring = CameraRingBuffer(
        name=name,
        rgb_shape=(4, 6, 3),
        depth_shape=(4, 6),
        pc_shape=(2, 6),
        maxlen=2,
        create=True,
    )
    rgb = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    depth = np.arange(24, dtype=np.uint16).reshape(4, 6)
    pointcloud = np.arange(12, dtype=np.float32).reshape(2, 6)
    header, _, _ = pack_camera_frame(
        rgb,
        depth,
        timestamp=123.5,
        capture_monotonic_s=456.25,
        frame_id=77,
        pc_num_points=2,
        pc_source_point_count=1,
        pc_valid_depth_ratio=0.75,
        pc_padding_count=1,
    )
    try:
        assert ring.write(header, rgb, depth, pointcloud) == 1
        result = ring.read_latest()
        assert result is not None
        read_header, read_rgb, read_depth, read_pc, sequence = result
        assert sequence == 1
        assert int(read_header["frame_number"][0]) == 77
        assert float(read_header["capture_monotonic_s"][0]) == pytest.approx(456.25)
        assert int(read_header["receive_monotonic_ns"][0]) == 456_250_000_000
        assert bool(read_header["pointcloud_valid"][0])
        assert int(read_header["pc_source_point_count"][0]) == 1
        assert float(read_header["pc_valid_depth_ratio"][0]) == pytest.approx(0.75)
        assert int(read_header["pc_padding_count"][0]) == 1
        np.testing.assert_array_equal(read_rgb, rgb)
        np.testing.assert_array_equal(read_depth, depth)
        np.testing.assert_array_equal(read_pc, pointcloud)

        with pytest.raises(ValueError, match="depth must"):
            ring.write(header, rgb, depth.astype(np.float32), pointcloud)
        with pytest.raises(ValueError, match="RGB shape"):
            bad_header = header.copy()
            bad_header["rgb_shape_w"] = 5
            ring.write(bad_header, rgb, depth, pointcloud)
        with pytest.raises(ValueError, match="pointcloud must"):
            ring.write(header, rgb, depth, np.zeros((3, 6), dtype=np.float32))
    finally:
        ring.close()
        ring.unlink()


def test_camera_ring_never_returns_a_torn_concurrent_frame() -> None:
    name = f"test_camera_torn_{uuid.uuid4().hex}"
    ring = CameraRingBuffer(
        name=name,
        rgb_shape=(12, 16, 3),
        depth_shape=(12, 16),
        pc_shape=(4, 6),
        maxlen=2,
        create=True,
    )
    producer_done = threading.Event()

    def write_tag(tag: int) -> None:
        rgb = np.full((12, 16, 3), tag, dtype=np.uint8)
        depth = np.full((12, 16), tag, dtype=np.uint16)
        pointcloud = np.full((4, 6), tag, dtype=np.float32)
        header, _, _ = pack_camera_frame(
            rgb,
            depth,
            timestamp=float(tag),
            capture_monotonic_s=float(tag),
            frame_id=tag,
            pc_num_points=4,
        )
        ring.write(header, rgb, depth, pointcloud)

    def producer() -> None:
        for tag in range(1, 100):
            write_tag(tag)
        producer_done.set()

    try:
        write_tag(1)
        thread = threading.Thread(target=producer)
        thread.start()
        while not producer_done.is_set():
            result = ring.read_latest()
            if result is None:
                continue
            header, rgb, depth, pointcloud, _ = result
            tag = int(header["frame_number"][0])
            assert np.all(rgb == tag)
            assert np.all(depth == tag)
            assert pointcloud is not None and np.all(pointcloud == tag)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        ring.close()
        ring.unlink()


class _FakeEncoder:
    instances: list[_FakeEncoder] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.frames: list[np.ndarray] = []
        self.closed = False
        self.instances.append(self)

    def write_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True


def test_camera_stream_writer_writes_equal_length_sidecars(tmp_path: Path) -> None:
    _FakeEncoder.instances.clear()
    cfg = CameraStreamWriterConfig(
        rgb_shape=(4, 6, 3),
        depth_shape=(4, 6),
        pointcloud_shape=(2, 6),
        fps=10.0,
        queue_size=2,
    )
    writer = CameraStreamWriter(tmp_path, cfg, encoder_factory=_FakeEncoder)
    for value in (1, 2):
        assert writer.submit(
            np.full(cfg.rgb_shape, value, dtype=np.uint8),
            np.full(cfg.depth_shape, value, dtype=np.uint16),
            np.full(cfg.pointcloud_shape, value, dtype=np.float32),
        )
    writer.close()

    assert writer.frame_count == 2
    assert len(_FakeEncoder.instances) == 1
    assert len(_FakeEncoder.instances[0].frames) == 2
    assert _FakeEncoder.instances[0].closed
    with h5py.File(tmp_path / "depth.h5", "r") as depth_file:
        assert depth_file["depth"].shape == (2, 4, 6)
    with h5py.File(tmp_path / "pointcloud.h5", "r") as pc_file:
        assert pc_file["pointcloud"].shape == (2, 2, 6)


def test_camera_stream_writer_queue_full_is_fatal(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class SlowEncoder(_FakeEncoder):
        def write_frame(self, frame: np.ndarray) -> None:
            entered.set()
            release.wait(timeout=5.0)
            super().write_frame(frame)

    cfg = CameraStreamWriterConfig(
        rgb_shape=(2, 2, 3),
        depth_shape=(2, 2),
        pointcloud_shape=(1, 6),
        fps=10.0,
        queue_size=1,
    )
    writer = CameraStreamWriter(tmp_path, cfg, encoder_factory=SlowEncoder)
    rgb = np.zeros(cfg.rgb_shape, dtype=np.uint8)
    depth = np.zeros(cfg.depth_shape, dtype=np.uint16)
    pointcloud = np.zeros(cfg.pointcloud_shape, dtype=np.float32)
    assert writer.submit(rgb, depth, pointcloud)
    assert entered.wait(timeout=2.0)
    assert writer.submit(rgb, depth, pointcloud)
    assert not writer.submit(rgb, depth, pointcloud)
    assert writer.error is not None and "queue full" in writer.error
    release.set()
    with pytest.raises(CameraStreamWriterError, match="queue full"):
        writer.close(timeout=5.0)


def test_camera_stream_writer_io_error_is_latched(tmp_path: Path) -> None:
    class BrokenEncoder(_FakeEncoder):
        def write_frame(self, frame: np.ndarray) -> None:
            raise OSError(28, "No space left on device")

    cfg = CameraStreamWriterConfig(
        rgb_shape=(2, 2, 3),
        depth_shape=(2, 2),
        pointcloud_shape=(1, 6),
        fps=10.0,
        queue_size=1,
    )
    writer = CameraStreamWriter(tmp_path, cfg, encoder_factory=BrokenEncoder)
    assert writer.submit(
        np.zeros(cfg.rgb_shape, dtype=np.uint8),
        np.zeros(cfg.depth_shape, dtype=np.uint16),
        np.zeros(cfg.pointcloud_shape, dtype=np.float32),
    )
    deadline = time.monotonic() + 2.0
    while writer.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.error is not None and "No space left" in writer.error
    with pytest.raises(CameraStreamWriterError, match="No space left"):
        writer.close(timeout=2.0)


def _robot_state(timestamp: float) -> RobotState:
    return RobotState(
        arm_qpos=np.zeros(7),
        arm_qvel=np.zeros(7),
        arm_tau=np.zeros(7),
        eef_pos=np.zeros(3),
        eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        eef_rot6d=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        hand_qpos=np.zeros(12),
        hand_tactile_sum=np.zeros((5, 3)),
        hand_tactile_force=np.zeros((5, 120, 3)),
        hand_tactile_contact=np.zeros(5, dtype=bool),
        hand_tipboard_err=np.zeros(12, dtype=np.int32),
        hand_commboard_err=np.zeros(12, dtype=np.int32),
        hand_jointboard_err=np.zeros(12, dtype=np.int32),
        hand_qpos_stale=False,
        fingertip_pos=np.zeros((5, 3)),
        arm_connected=True,
        hand_connected=True,
        timestamp=timestamp,
    )


def test_schema_v15_round_trip_and_invalid_pointcloud_placeholder(tmp_path: Path) -> None:
    cfg = CameraStreamWriterConfig(
        rgb_shape=(8, 8, 3),
        depth_shape=(8, 8),
        pointcloud_shape=(2, 6),
        fps=10.0,
        queue_size=4,
    )
    recorder = EpisodeRecorder(
        str(tmp_path),
        max_frames=10,
        control_hz=10.0,
        min_frames=1,
        camera_writer_config=cfg,
    )
    assert recorder.start_episode(
        camera_metadata={
            "camera_serial": "serial-test",
            "camera_firmware": "test-fw",
            "camera_sdk_version": "2.50.0",
            "camera_actual_profile_json": "{}",
        }
    )
    start = float(recorder._start_time)
    action = RobotAction(np.zeros(7), np.zeros(12))
    vr = {
        "wrist_pos": np.zeros(3),
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.zeros((21, 3)),
    }
    for index, pc_valid in enumerate((True, False)):
        camera_frame = {
            "rgb": np.full(cfg.rgb_shape, index + 1, dtype=np.uint8),
            "depth": np.full(cfg.depth_shape, index + 1, dtype=np.uint16),
            "pointcloud": np.full(cfg.pointcloud_shape, index + 1, dtype=np.float32),
            "pointcloud_valid": pc_valid,
            "camera_fresh": True,
            "camera_health": 0,
            "frame_number": index + 10,
            "ring_sequence": index + 20,
            "device_timestamp_s": 100.0 + index,
            "capture_monotonic_s": start + index * 0.1,
            "camera_age_s": 0.01,
        }
        assert recorder.add_frame(_robot_state(start + index * 0.1), action, vr, camera_frame=camera_frame)

    episode_path = recorder.stop_episode(success=True)
    assert episode_path is not None
    assert recorder.join_stop(timeout=10.0), recorder.stop_error

    with EpisodeReader(episode_path) as reader:
        h5f = reader.h5f
        assert int(h5f["meta"].attrs["schema_version"]) == 15
        assert h5f["meta"].attrs["camera_serial"] == "serial-test"
        assert h5f["meta"].attrs["camera_firmware"] == "test-fw"
        assert h5f["meta"].attrs["camera_sdk_version"] == "2.50.0"
        assert h5f["meta"].attrs["camera_actual_profile_json"] == "{}"
        assert h5f["meta"].attrs["camera_encoding_codec"] == "libx264"
        assert float(h5f["meta"].attrs["camera_encoding_fps"]) == pytest.approx(10.0)
        assert int(h5f["meta"].attrs["camera_writer_queue_high_watermark"]) >= 1
        assert float(h5f["meta"].attrs["camera_encode_p99_s"]) >= 0.0
        assert float(h5f["meta"].attrs["camera_hdf5_p99_s"]) >= 0.0
        assert float(h5f["meta"].attrs["camera_writer_close_s"]) >= 0.0
        assert h5f["flag_camera_fresh"][:].tolist() == [True, True]
        assert h5f["flag_pointcloud_valid"][:].tolist() == [True, False]
        assert h5f["camera_frame_number"][:].tolist() == [10, 11]
        assert h5f["depth"].shape[0] == 2
        pointcloud = h5f["pointcloud"][:]
        assert pointcloud.shape == (2, 2, 6)
        assert np.count_nonzero(pointcloud[1]) == 0
        assert reader.read_camera_all("rgb").shape[0] == 2


def test_grid_backfill_marks_camera_and_pointcloud_invalid(tmp_path: Path) -> None:
    cfg = CameraStreamWriterConfig(
        rgb_shape=(8, 8, 3),
        depth_shape=(8, 8),
        pointcloud_shape=(2, 6),
        fps=10.0,
        queue_size=4,
    )
    recorder = EpisodeRecorder(str(tmp_path), max_frames=10, control_hz=10.0, min_frames=1, camera_writer_config=cfg)
    assert recorder.start_episode()
    start = float(recorder._start_time)
    action = RobotAction(np.zeros(7), np.zeros(12))
    vr = {
        "wrist_pos": np.zeros(3),
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.zeros((21, 3)),
    }
    camera_frame = {
        "rgb": np.ones(cfg.rgb_shape, dtype=np.uint8),
        "depth": np.ones(cfg.depth_shape, dtype=np.uint16),
        "pointcloud": np.ones(cfg.pointcloud_shape, dtype=np.float32),
        "pointcloud_valid": True,
        "camera_fresh": True,
        "camera_health": 0,
        "frame_number": 1,
        "ring_sequence": 1,
        "device_timestamp_s": 1.0,
        "capture_monotonic_s": start,
        "camera_age_s": 0.0,
    }
    assert recorder.add_frame(_robot_state(start), action, vr, camera_frame=camera_frame)
    camera_frame = dict(camera_frame, frame_number=2, ring_sequence=2, capture_monotonic_s=start + 0.2)
    assert recorder.add_frame(_robot_state(start + 0.2), action, vr, camera_frame=camera_frame)

    assert recorder._buffer is not None
    np.testing.assert_array_equal(recorder._buffer.data["flag_sample_valid"], [True, False, True])
    np.testing.assert_array_equal(recorder._buffer.data["flag_camera_fresh"], [True, False, True])
    np.testing.assert_array_equal(recorder._buffer.data["flag_pointcloud_valid"], [True, False, True])
    np.testing.assert_array_equal(
        recorder._buffer.data["observation_anchor_monotonic_ns"],
        np.rint(recorder._buffer.timestamps * 1e9).astype(np.uint64),
    )
    recorder.stop_episode(success=False)
    assert recorder.join_stop(timeout=10.0)
    aborted = list(tmp_path.glob("*.aborted.json"))
    assert len(aborted) == 1
    assert not list(tmp_path.glob(".tmp_episode_*"))


def test_atomic_publish_never_overwrites_existing_output(tmp_path: Path) -> None:
    source = tmp_path / ".tmp_episode_source"
    target = tmp_path / "episode_existing"
    source.mkdir()
    target.mkdir()

    with pytest.raises(FileExistsError):
        atomic_publish(source, target)

    assert source.is_dir()
    assert target.is_dir()


def test_atomic_publish_rename_failure_has_no_copy_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / ".tmp_episode_source"
    target = tmp_path / "episode_final"
    source.mkdir()
    (source / "data.h5").write_bytes(b"payload")

    def fail_rename(_source: Path, _target: Path) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("dexmani_real.recording.transaction.os.rename", fail_rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_publish(source, target)

    assert (source / "data.h5").read_bytes() == b"payload"
    assert not target.exists()


def test_start_episode_never_deletes_an_unowned_orphan_temp_directory(tmp_path: Path) -> None:
    orphan = tmp_path / ".tmp_episode_previous_crash"
    orphan.mkdir()
    (orphan / "diagnostic.txt").write_text("preserve", encoding="utf-8")
    recorder = EpisodeRecorder(str(tmp_path), max_frames=2, control_hz=10.0, min_frames=1)

    assert recorder.start_episode()
    assert (orphan / "diagnostic.txt").read_text(encoding="utf-8") == "preserve"
    recorder.stop_episode(success=False, reason="test_cleanup")
    assert recorder.join_stop(timeout=10.0)
    assert (orphan / "diagnostic.txt").read_text(encoding="utf-8") == "preserve"


def test_reader_remains_compatible_with_v13_and_legacy(tmp_path: Path) -> None:
    v13 = tmp_path / "episode_v13"
    v13.mkdir()
    with h5py.File(v13 / "data.h5", "w") as data_file:
        meta = data_file.create_group("meta")
        meta.attrs["schema_version"] = 13
        meta.attrs["num_frames"] = 2
        data_file.create_dataset("timestamp", data=[1.0, 1.1])
        data_file.create_dataset("pointcloud", data=np.ones((2, 1, 6), dtype=np.float32))
    with h5py.File(v13 / "depth.h5", "w") as depth_file:
        depth_file.create_dataset("depth", data=np.ones((2, 2, 2), dtype=np.uint16))

    with EpisodeReader(v13) as reader:
        assert reader.h5f["pointcloud"].shape == (2, 1, 6)
        assert reader.h5f["depth"].shape == (2, 2, 2)

    legacy = tmp_path / "episode_legacy.h5"
    with h5py.File(legacy, "w") as legacy_file:
        legacy_file.create_group("meta").attrs["num_frames"] = 1
        legacy_file.create_dataset("depth", data=np.ones((1, 2, 2), dtype=np.uint16))
    with EpisodeReader(legacy) as reader:
        assert reader.h5f["depth"].shape == (1, 2, 2)


def test_writer_failure_never_publishes_partial_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenWriter:
        error = "OSError: No space left on device"
        frame_count = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def submit(self, *_args, **_kwargs) -> bool:
            return False

        def close(self, timeout: float = 60.0) -> None:
            raise CameraStreamWriterError(self.error)

    monkeypatch.setattr(recorder_module, "CameraStreamWriter", BrokenWriter)
    recorder = EpisodeRecorder(str(tmp_path), max_frames=2, control_hz=10.0, min_frames=1)
    assert recorder.start_episode()
    final_path = recorder.stop_episode(success=True)
    assert final_path is not None
    assert not recorder.join_stop(timeout=5.0)
    assert recorder.stop_error is not None and "No space left" in recorder.stop_error
    assert not Path(final_path).exists()
    assert not list(tmp_path.glob(".tmp_episode_*"))


def test_quality_and_training_filter_use_v14_camera_flags(tmp_path: Path) -> None:
    path = tmp_path / "episode.h5"
    with h5py.File(path, "w") as h5_file:
        meta = h5_file.create_group("meta")
        meta.attrs["schema_version"] = 14
        meta.attrs["control_hz"] = 10.0
        h5_file.create_dataset("arm_qpos", data=np.zeros((4, 7)))
        h5_file.create_dataset("timestamp", data=[1.0, 1.1, 1.2, 1.3])
        h5_file.create_dataset("flag_camera_fresh", data=[True, False, False, True])
        h5_file.create_dataset("flag_pointcloud_valid", data=[True, True, False, True])
        h5_file.create_dataset("camera_frame_number", data=[10, 10, 10, 11])

    with EpisodeQuality(path) as quality:
        health = quality.health()
        mask, counts = quality.build_filter_mask(drop_held=False)

    assert health.camera_fresh_pct == pytest.approx(50.0)
    assert health.camera_stale_frames == 2
    assert health.camera_longest_stale_run == 2
    assert health.camera_repeated_frame_numbers == 2
    assert health.pointcloud_invalid_frames == 1
    np.testing.assert_array_equal(mask, [True, False, False, True])
    assert counts["camera_stale"] == 2
    assert counts["pointcloud_invalid"] == 1
