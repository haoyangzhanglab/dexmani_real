import shutil
import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.config import RealConfig
from dexmani_real.data.export import ExportConfig, export_zarr, validate_export_episode
from dexmani_real.data.raw import (
    TACTILE_PROCESSING,
    RawReader,
    RawWriter,
    snapshot_recording_metadata,
)
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.perception.config import HandCloudConfig, PointCloudConfig
from dexmani_real.replay.trajectory import load_trajectory
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation


def write_raw(path, *, status="normal", jitter=False, count=6):
    path.mkdir()
    runtime = RealConfig()
    camera = CameraSnapshot("fake-camera", CameraIntrinsics(16, 16, 250., 250., 8., 8.,
                                                            "none", (0.,) * 5),
                            0.001, np.eye(4))
    metadata = snapshot_recording_metadata(camera, tactile_reset_metadata={}, tactile_hold_max_age_s=4/30)
    metadata.update(task_label="test", collection_source="teleop")
    writer = RawWriter(path, control_hz=16)
    writer.open(metadata)
    try:
        for index in range(count):
            control = 1_000_000_000 + index * 62_500_000
            arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
            arm["qpos"] = runtime.arm.home_qpos
            arm["timestamp_ns"] = control - 1_000_000
            hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
            hand["qpos"] = np.deg2rad(runtime.hand.home_qpos_deg)
            hand["tactile_aggregate"] = index
            hand["tactile_dense"] = index
            for field in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
                hand[field] = control - 1_000_000
            row = Observation(arm, hand, dict(rgb=np.full((16, 16, 3), index * 30, np.uint8),
                                              depth=np.full((16, 16), 500 + index * 10, np.uint16),
                                              timestamp_ns=control - 1_000_000, ring_sequence=index + 1),
                              None, None,
                              control + 1_000_000 + (20_000_000 if jitter and index == 2 else 0),
                              control_timestamp_ns=control)
            command = RobotCommand(1, arm["qpos"][0] + index * 0.001,
                                   hand["qpos"][0] + index * 0.001)
            writer.append_frame(row, command)
        writer.finish("operator" if status == "normal" else "hardware_fault", [],
                      expected_frames=count, technical_status=status)
    finally:
        writer.close()
    return path


def export(raw, target):
    return export_zarr(raw, target, processing=ExportConfig(
        pointcloud=PointCloudConfig(remove_table=False, num_points=16),
        hand_pointcloud=HandCloudConfig(num_points=16),
    ))


def test_whole_tactile_frame_hold_preserves_values_and_source_times(tmp_path):
    raw = write_raw(tmp_path / "episode_tactile")
    with h5py.File(raw / "data.h5", "r+") as file:
        rows = file["timing/rows"][:]
        rows["tactile_dense_ns"][1] = rows["tactile_dense_ns"][0]
        file["timing/rows"][:] = rows
        file["hand_tactile_force"][1] = file["hand_tactile_force"][0]
        expected = file["hand_tactile_force"][:]
    with RawReader(raw) as reader:
        assert reader.meta["hand_tactile_processing"] == TACTILE_PROCESSING
        np.testing.assert_array_equal(reader["hand_tactile_force"][:], expected)
        np.testing.assert_array_equal(reader["timing/rows"][:], rows)
        reader.validate_rows()
    result = export(raw, tmp_path / "train.zarr")
    assert result["total_frames"] == 6
    root = zarr.open_group(str(tmp_path / "train.zarr"), mode="r")
    np.testing.assert_array_equal(root["data/taxel_tactile_force"][:], expected)
    with RawReader(raw) as reader:
        assert reader.meta["hand_tactile_processing"] == TACTILE_PROCESSING
        np.testing.assert_array_equal(reader["timing/rows"][:], rows)


@pytest.mark.parametrize("field,value", [
    ("hand_tactile_processing", "unknown"),
    ("hand_tactile_processing", "sdk_native_bounded_previous_valid_vector_hold"),
    ("hand_tactile_units", "newtons"),
    ("hand_tactile_axes", "world"),
    ("hand_tactile_finger_order", "reversed"),
    ("hand_tactile_hold_max_age_s", 0),
    ("hand_tactile_hold_max_age_s", np.nan),
    ("joint_units", "deg"),
])
def test_tactile_metadata_rejects_unsupported_semantics(tmp_path, field, value):
    raw = write_raw(tmp_path / "episode_bad_metadata")
    with h5py.File(raw / "data.h5", "r+") as file:
        file["meta"].attrs[field] = value
    with pytest.raises(ValueError, match=field), RawReader(raw) as reader:
        reader["hand_contact"]


def test_tactile_age_budget_rejects_stale_held_frames(tmp_path):
    raw = write_raw(tmp_path / "episode_stale_tactile")
    with h5py.File(raw / "data.h5", "r+") as file:
        rows = file["timing/rows"][:]
        rows["tactile_dense_ns"] = rows["tactile_dense_ns"][0]
        file["timing/rows"][:] = rows
    with RawReader(raw) as reader:
        with pytest.raises(ValueError, match="dense exceeds hold age budget"):
            reader.validate_rows()
        _, quality = validate_export_episode(reader)
    assert quality["row_stop"] == 3 and quality["trim_reason"] == "stale_source:tactile_dense_ns"


def test_readout_jitter_exports_without_changing_physical_replay(tmp_path):
    raw = write_raw(tmp_path / "episode_jitter", jitter=True)
    with RawReader(raw) as reader:
        assert reader.meta["hand_tactile_processing"] == TACTILE_PROCESSING
    result = export(raw, tmp_path / "train.zarr")
    assert result["total_frames"] == 6
    quality = result["episode_quality"][0]
    assert quality["readout_over_tolerance_rows"] == 1 and quality["max_readout_ns"] == 21_000_000
    root = zarr.open_group(str(tmp_path / "train.zarr"), mode="r")
    assert root.attrs["timing_admission"]["control_clock"] == "control_ns"
    assert root.attrs["timing_admission"]["resampled"] is False
    assert len(root["data"]) == 13
    with pytest.raises(ValueError, match="timing row 2"):
        load_trajectory(raw)
    clean = write_raw(tmp_path / "episode_clean")
    assert len(load_trajectory(clean)) == 6


@pytest.mark.parametrize("delayed_rows", [[0], list(range(6))])
def test_readout_quality_uses_each_control_start(tmp_path, delayed_rows):
    raw = write_raw(tmp_path / "episode_readout")
    with h5py.File(raw / "data.h5", "r+") as file:
        rows = file["timing/rows"][:]
        rows["observation_ns"][delayed_rows] += 20_000_000
        file["timing/rows"][:] = rows
    with RawReader(raw) as reader:
        _, quality = validate_export_episode(reader)
    assert quality["row_stop"] == 6 and quality["trim_reason"] is None
    assert quality["readout_over_tolerance_rows"] == len(delayed_rows)
    assert quality["max_readout_ns"] == 21_000_000


@pytest.mark.parametrize("case, stop", [("gap", 3), ("drift", 4), ("invalid", 3),
                                      ("failed", 5), ("future_source", 2)])
def test_training_prefix_all_fields_share_rows(tmp_path, case, stop):
    raw = write_raw(tmp_path / "episode_test", status="failed" if case == "failed" else "normal")
    with h5py.File(raw / "data.h5", "r+") as file:
        if case in {"gap", "drift"}:
            times = file["timing/rows"][:]
            shift = np.where(np.arange(6) >= 3, 62_500_000, 0) if case == "gap" else np.arange(6) * 5_000_000
            for name in times.dtype.names:
                if name != "camera_sequence":
                    times[name] += shift
            file["timing/rows"][:] = times
        elif case == "invalid":
            file["action_hand_joint_target"][3] = np.nan
        elif case == "future_source":
            times = file["timing/rows"][:]
            times["camera_ns"][2] = times["observation_ns"][2] + 1
            file["timing/rows"][:] = times
    result = export(raw, tmp_path / "train.zarr")
    assert result["total_frames"] == stop
    root = zarr.open_group(str(tmp_path / "train.zarr"), mode="r")
    assert all(len(array) == stop for _, array in root["data"].arrays())
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [stop])
    quality = root.attrs["episode_quality"][0]
    assert quality["row_start"] == 0 and quality["row_stop"] == stop and quality["trim_reason"]
    with RawReader(raw) as reader:
        np.testing.assert_array_equal(root["data/action"][:, :7],
                                      reader["action_arm_joint_target"][:stop].astype(np.float32))
        np.testing.assert_array_equal(root["data/depth"][:], reader["depth"][:stop])
        np.testing.assert_array_equal(root["data/rgb"][:], np.stack(list(reader.iter_rgb_frames())[:stop]))
        assert reader.meta["technical_status"] == ("failed" if case == "failed" else "normal")
    if case == "failed":
        with pytest.raises(ValueError, match="technical failure"):
            load_trajectory(raw)


@pytest.mark.parametrize("case", ["missing", "shape", "units", "empty_prefix", "short_video"])
def test_prefix_does_not_hide_corruption(tmp_path, case):
    raw = write_raw(tmp_path / "episode_bad")
    with h5py.File(raw / "data.h5", "r+") as file:
        file["action_hand_joint_target"][3] = np.nan
        if case == "missing":
            del file["hand_tactile_force"]
        elif case == "shape":
            file["arm_qpos"].resize(5, axis=0)
        elif case == "units":
            file["meta"].attrs["joint_units"] = "deg"
        elif case == "empty_prefix":
            file["action_arm_joint_target"][0] = np.nan
    if case == "short_video":
        short = write_raw(tmp_path / "short", count=4)
        shutil.copyfile(short / "rgb.mp4", raw / "rgb.mp4")
        with pytest.raises(ValueError, match="row 4"):
            export(raw, tmp_path / "train.zarr")
        assert not (tmp_path / "train.zarr").exists()
    else:
        if case == "units":
            with pytest.raises(ValueError, match="joint_units"):
                RawReader(raw)
        else:
            with RawReader(raw) as reader:
                with pytest.raises(ValueError):
                    validate_export_episode(reader)


@pytest.mark.parametrize("stage", ["write", "close", "normal"])
def test_writer_failure_and_single_publication(tmp_path, monkeypatch, stage):
    import dexmani_real.data.recorder as module

    original = OSError(f"{stage} failed")
    closed, published = [], []
    publish = module.publish

    class Writer:
        frame_count = 0

        def __init__(self, *args, **kwargs):
            pass

        def open(self, metadata):
            pass

        def append_frame(self, *args):
            if stage == "write":
                raise original
            self.frame_count += 1

        def flush(self):
            pass

        def finish(self, *args, **kwargs):
            assert kwargs["expected_frames"] == self.frame_count

        def close(self):
            closed.append(True)
            if stage == "close":
                raise original

    def publish_once(*args):
        published.append(args)
        publish(*args)

    monkeypatch.setattr(module, "RawWriter", Writer)
    monkeypatch.setattr(module, "publish", publish_once)
    recorder = EpisodeRecorder(tmp_path, control_hz=16)
    recorder.start(task_label="test", collection_source="teleop", metadata={})
    row = SimpleNamespace(arm=None, hand=None, camera=None, observation_timestamp_ns=1,
                          control_timestamp_ns=1)
    recorder.append(row)
    if stage == "normal":
        first = recorder.finish(save=True, reason="operator", technical_status="normal")
        assert recorder.finish(save=True, reason="close") == first
        recorder.close()
        assert len(published) == 1 and recorder.result()["status"] == "published"
    else:
        with pytest.raises(RecordingError) as failure:
            recorder.finish(save=True, reason="operator")
        assert failure.value.__cause__ is original
        assert recorder.result()["status"] == "failed" and not published
    assert closed == [True] and not recorder.writer_pending


def test_writer_pending_retains_ownership_and_cancels_publication(tmp_path, monkeypatch):
    import dexmani_real.data.recorder as module

    entered, release = threading.Event(), threading.Event()

    class Writer:
        frame_count = 0

        def __init__(self, *args, **kwargs):
            pass

        def open(self, metadata):
            pass

        def append_frame(self, *args):
            entered.set()
            release.wait()
            self.frame_count += 1

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "RawWriter", Writer)
    recorder = EpisodeRecorder(tmp_path, control_hz=16)
    recorder.start(task_label="test", collection_source="teleop", metadata={})
    recorder.queue.put((None, None))
    try:
        assert entered.wait(1)
        recorder.stop_deadline = time.monotonic()
        with pytest.raises(RecordingError, match="pending"):
            recorder.finish(save=True, reason="operator")
        assert recorder.writer_pending and recorder.result()["status"] == "pending"
    finally:
        release.set()
        recorder.thread.join(1)
    assert not recorder.writer_pending and recorder.published_path is None


def test_model_close_does_not_claim_running_thread_cancelled():
    entered, release = threading.Event(), threading.Event()
    closed = []

    def predict(rows):
        entered.set()
        release.wait()
        return np.zeros((1, 19))

    worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
    worker.model = SimpleNamespace(predict=predict, close=lambda: closed.append(True))
    worker.predict(())
    future = worker.future
    try:
        assert entered.wait(1)
        worker.close()
        assert worker.wait_closed(0) == ("pending", [])
        assert not future.cancelled() and not closed
    finally:
        release.set()
    assert worker.wait_closed(1) == ("completed", [])
    assert closed == [True]


def test_model_close_reports_retired_and_cleanup_tracebacks(caplog):
    original, cleanup = ValueError("prediction broke"), OSError("model close broke")

    def predict(rows):
        raise original

    def close():
        raise cleanup

    worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
    worker.model = SimpleNamespace(predict=predict, close=close)
    worker.predict(())
    worker.close()
    status, errors = worker.wait_closed(1)
    assert status == "failed"
    assert [error["stage"] for error in errors] == ["retired_predict", "model_close"]
    assert [record.exc_info[1] for record in caplog.records if record.exc_info] == [original, cleanup]
