"""Offline current raw source-row recording and storage regressions."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import pytest

from dexmani_real.ipc.schema import make_record_sample_dtype
from dexmani_real.recording.frame import (
    EpisodeFrame,
    build_episode_frame,
    decode_record_sample,
)
from dexmani_real.recording.sample import EpisodeAction, build_episode_state
from dexmani_real.recording.recorder import EpisodeRecorder
from dexmani_real.recording.storage.camera_writer import CameraStreamWriterConfig
from dexmani_real.recording.storage.camera_writer import CameraStreamWriter
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import (
    DATASET_SPECS,
    EPISODE_SCHEMA_VERSION,
    SOURCE_FRAME_DATASET_NAMES,
)
from dexmani_real.recording.storage.video import VideoDecoder


def test_schema_keeps_control_step_sensor_facts():
    assert EPISODE_SCHEMA_VERSION == 28
    assert len(DATASET_SPECS) == 41
    assert len(SOURCE_FRAME_DATASET_NAMES) == 37
    assert {
        "arm_qpos",
        "hand_qpos",
        "hand_contact",
        "hand_tactile_force",
        "hand_contact_source_monotonic_ns",
        "action_arm_joint_sent",
        "action_hand_joint",
        "action_arm_ee",
        "observation_anchor_monotonic_ns",
        "arm_source_monotonic_ns",
        "hand_source_monotonic_ns",
        "tactile_source_monotonic_ns",
        "camera_source_monotonic_ns",
        "camera_health",
        "tactile_sum_fresh",
        "tactile_fresh",
        "tactile_calibrated",
        "tactile_unit_code",
    }.issubset(SOURCE_FRAME_DATASET_NAMES)


def test_decoded_frame_owns_sample_ring_arrays():
    sample = np.zeros(1, dtype=make_record_sample_dtype((16, 16, 3), (16, 16)))
    sample["timestamp"] = 1.25
    sample["camera_present"] = True
    sample["hand_contact"] = 3
    sample["action_arm_joint_sent"] = 4
    sample["camera_rgb"] = 5
    sample["camera_depth"] = 6
    frame = decode_record_sample(sample[0])
    sample[...] = np.zeros(1, dtype=sample.dtype)
    assert frame.timestamp_s == 1.25
    np.testing.assert_array_equal(frame.data["hand_contact"], 3)
    np.testing.assert_array_equal(frame.data["action_arm_joint_sent"], 4)
    np.testing.assert_array_equal(frame.camera_rgb, 5)
    np.testing.assert_array_equal(frame.camera_depth, 6)
    assert set(frame.data) == SOURCE_FRAME_DATASET_NAMES


def test_direct_builder_requires_sent_target_and_copies_source():
    state = build_episode_state(None, None, timestamp_s=1)
    assert np.isnan(state.hand_tactile_force).all()
    action = EpisodeAction(np.zeros(7), np.zeros(12))
    vr = {
        "wrist_pos": np.zeros(3),
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.zeros((21, 3)),
        "head_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    with pytest.raises(ValueError, match="explicit submitted"):
        build_episode_frame(state, action, vr)
    sent = np.ones(7)
    frame = build_episode_frame(state, action, vr, arm_qpos_sent=sent)
    sent[:] = 2
    vr["head_quat_wxyz"][:] = 0
    np.testing.assert_array_equal(frame.data["action_arm_joint_sent"], 1)
    np.testing.assert_array_equal(frame.data["head_quat_wxyz"], [1, 0, 0, 0])
    assert set(frame.data) == SOURCE_FRAME_DATASET_NAMES


def _recorder(directory):
    return EpisodeRecorder(
        str(directory),
        min_frames=1,
        camera_writer_config=CameraStreamWriterConfig(
            rgb_shape=(16, 16, 3), depth_shape=(16, 16), fps=16, queue_size=64
        ),
    )


def _frame(timestamp, value=0):
    data = {
        name: np.full(spec.tail_shape, value, dtype=spec.dtype)
        for name, spec in DATASET_SPECS.items()
        if name in SOURCE_FRAME_DATASET_NAMES
    }
    data["observation_anchor_monotonic_ns"] = np.uint64(round(timestamp * 1e9))
    return EpisodeFrame(
        timestamp_s=timestamp,
        data=data,
        camera_rgb=np.full((16, 16, 3), value, np.uint8),
        camera_depth=np.full((16, 16), value, np.uint16),
    )


def _save(recorder):
    return Path(recorder.finish_episode(save=True))


@pytest.mark.parametrize("save", [True, False])
def test_synchronous_finish_returns_reserved_path_and_resets(tmp_path, save):
    recorder = _recorder(tmp_path)
    assert recorder.finish_episode() is None
    assert recorder.start_episode()
    assert recorder.add_episode_frame(_frame(1))
    reserved = recorder.episode_path
    assert recorder.finish_episode(save=save, reason="synchronous") == reserved
    assert Path(reserved).is_dir() is save
    assert not recorder.is_recording
    assert recorder.frame_count == 0
    assert recorder.finish_episode() is None
    assert recorder.start_episode()
    assert recorder.add_episode_frame(_frame(2))
    assert Path(recorder.finish_episode()).is_dir()


def test_synchronous_finish_raises_after_failed_validation_cleanup(tmp_path):
    recorder = _recorder(tmp_path)
    assert recorder.start_episode()
    assert recorder.add_episode_frame(_frame(1))
    reserved = Path(recorder.episode_path)
    with mock.patch.object(
        recorder, "_validate_temp_episode", side_effect=OSError("invalid sidecar")
    ):
        with pytest.raises(RuntimeError, match="invalid sidecar"):
            recorder.finish_episode()
    assert not reserved.exists()
    assert not list(tmp_path.glob(".tmp_episode_*"))
    assert not recorder.is_recording
    assert recorder._camera_writer is None and recorder._data_writer is None
    assert recorder.start_episode()
    assert recorder.add_episode_frame(_frame(2))
    assert Path(recorder.finish_episode()).is_dir()


@pytest.mark.parametrize("failed_resource", ["encoder", "depth"])
def test_camera_close_failure_retains_resource_after_thread_exit(
    tmp_path, failed_resource
):
    encoder = mock.Mock()
    depth_file = mock.Mock()
    resource = encoder if failed_resource == "encoder" else depth_file
    resource.close.side_effect = OSError("resource close failed")
    with mock.patch(
        "dexmani_real.recording.storage.camera_writer.h5py.File",
        return_value=depth_file,
    ):
        writer = CameraStreamWriter(
            tmp_path,
            CameraStreamWriterConfig(
                rgb_shape=(16, 16, 3), depth_shape=(16, 16), fps=16, queue_size=8
            ),
            encoder_factory=mock.Mock(return_value=encoder),
        )
        with pytest.raises(RuntimeError, match="resource close failed"):
            writer.close(timeout=2)
    assert not writer._thread.is_alive()
    assert not writer.resources_released
    assert writer._unreleased_resources == [resource]
    with pytest.raises(RuntimeError):
        writer.close(timeout=0)
    assert writer._unreleased_resources == [resource]
    resource.close.assert_called_once()


@pytest.mark.parametrize("failed_resource", ["camera", "hdf5", "staging"])
def test_finish_retains_unsafe_resource_and_refuses_restart(tmp_path, failed_resource):
    recorder = _recorder(tmp_path)
    staging = tmp_path / ".tmp_episode_test"
    staging.mkdir()
    recorder._recording = True
    recorder._episode_dir = str(tmp_path / "episode_test")
    recorder._temp_dir = str(staging)
    camera_writer = mock.Mock(resources_released=failed_resource != "camera")
    data_writer = mock.Mock()
    recorder._camera_writer = camera_writer
    recorder._data_writer = data_writer
    if failed_resource == "camera":
        camera_writer.close.side_effect = OSError("camera remains live")
    if failed_resource == "hdf5":
        data_writer.close.side_effect = OSError("HDF5 remains open")
    with (
        mock.patch.object(
            recorder,
            "_finalize_episode_files",
            side_effect=OSError("transaction failed"),
        ),
        mock.patch.object(
            recorder,
            "_discard_temp_files",
            side_effect=(
                OSError("staging remains") if failed_resource == "staging" else None
            ),
        ),
    ):
        with pytest.raises(RuntimeError):
            recorder.finish_episode()
    assert not recorder.resources_released
    assert recorder._temp_dir == str(staging) and staging.exists()
    assert not recorder.start_episode()
    if failed_resource == "camera":
        assert recorder._camera_writer is camera_writer
    if failed_resource == "hdf5":
        assert recorder._data_writer is data_writer


def test_camera_calibration_survives_minimal_shared_metadata(tmp_path):
    from dexmani_real.recording.io_worker import _build_start_metadata
    from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry

    intrinsics = CameraIntrinsics(16, 16, 20, 21, 8, 8, "none", (0, 0, 0, 0, 0))
    geometry = RGBDGeometry(intrinsics, intrinsics, np.eye(4))
    shared = SimpleNamespace(
        camera_depth_scale=SimpleNamespace(value=0.001),
        camera_serial=SimpleNamespace(value=b"offline-serial"),
        camera_geometry=SimpleNamespace(value=json.dumps(geometry.to_dict()).encode()),
    )
    calibration = mock.Mock()
    calibration.resolve_name_by_serial.return_value = "offline-camera"
    calibration.to_meta_dict.return_value = {
        "camera_serial": "offline-serial",
        "camera_type": "eye_to_hand",
        "camera_T_world_camera": np.eye(4).reshape(-1).tolist(),
    }
    metadata = _build_start_metadata(
        shared,
        task_label="fixture",
        operator="offline",
        calibration=calibration,
        provenance={"eval_seed": "1"},
    )
    recorder = _recorder(tmp_path)
    assert recorder.start_episode(**metadata)
    assert recorder.add_episode_frame(_frame(1))
    with h5py.File(_save(recorder) / "data.h5", "r") as raw:
        attrs = raw["meta"].attrs
        assert attrs["camera_serial"] == "offline-serial"
        assert attrs["depth_scale"] == 0.001
        assert attrs["provenance_eval_seed"] == "1"
        np.testing.assert_array_equal(
            attrs["camera_depth_intrinsics"], intrinsics.matrix().ravel()
        )
        np.testing.assert_array_equal(
            attrs["camera_T_xarm_base_from_depth"], np.eye(4).ravel()
        )
        assert not any("firmware" in key or "device_identity" in key for key in attrs)


def test_direct_rows_cross_batch_and_preserve_gap(tmp_path):
    recorder = _recorder(tmp_path)
    assert recorder.start_episode(task_label="fixture", operator="offline")
    timestamps = 1 + np.arange(35) / 16
    timestamps[20:] += 0.75
    for index, timestamp in enumerate(timestamps):
        assert recorder.add_episode_frame(_frame(timestamp, index))
    with (
        mock.patch.object(
            VideoDecoder, "count_decoded_frames", side_effect=AssertionError
        ),
        mock.patch.object(VideoDecoder, "iter_frames", side_effect=AssertionError),
    ):
        path = _save(recorder)
        with EpisodeReader(path) as reader:
            f = reader.h5f
            assert int(f["meta"].attrs["schema_version"]) == EPISODE_SCHEMA_VERSION
            np.testing.assert_array_equal(f["timestamp"][:], timestamps)
            np.testing.assert_array_equal(f["source_sample_index"][:], np.arange(35))
            np.testing.assert_array_equal(f["fill_reason"][:], 0)
            np.testing.assert_array_equal(f["flag_sample_valid"][:], True)
            np.testing.assert_array_equal(f["arm_qpos"][:, 0], np.arange(35))
            np.testing.assert_array_equal(f["depth"][:, 0, 0], np.arange(35))
            assert reader.timing.grid_dt_s == 1 / 16
            assert reader.timing.grid_duration_s == timestamps[-1] - timestamps[0]
            for name, spec in DATASET_SPECS.items():
                assert f[name].dtype == spec.dtype


def test_reader_rejects_old_schema_and_inconsistent_rows(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.start_episode()
    recorder.add_episode_frame(_frame(1))
    path = _save(recorder)
    with h5py.File(path / "data.h5", "r+") as f:
        f["meta"].attrs["schema_version"] = 24
    with pytest.raises(ValueError, match="unsupported"):
        EpisodeReader(path)
    with h5py.File(path / "data.h5", "r+") as f:
        f["meta"].attrs["schema_version"] = EPISODE_SCHEMA_VERSION
        f["meta"].attrs["num_frames"] = 2
    with pytest.raises(ValueError, match="validity"):
        EpisodeReader(path)


def test_camera_close_failure_never_publishes(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.start_episode()
    recorder.add_episode_frame(_frame(1))
    writer = recorder._camera_writer
    close = writer.close

    def close_then_fail(*args, **kwargs):
        close(*args, **kwargs)
        raise OSError("injected camera close failure")

    with mock.patch.object(writer, "close", side_effect=close_then_fail):
        destination = Path(recorder.episode_path)
        with pytest.raises(RuntimeError, match="injected camera close failure"):
            recorder.finish_episode(save=True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".tmp_episode_*"))


@pytest.mark.parametrize("damage", ["missing_sent", "wrong_shape", "wrong_dtype"])
def test_reader_owns_required_dataset_contract(tmp_path, damage):
    recorder = _recorder(tmp_path)
    recorder.start_episode()
    recorder.add_episode_frame(_frame(1))
    path = _save(recorder)
    with h5py.File(path / "data.h5", "r+") as f:
        del f["action_arm_joint_sent"]
        if damage == "wrong_shape":
            f.create_dataset("action_arm_joint_sent", data=np.zeros((1, 6)))
        elif damage == "wrong_dtype":
            f.create_dataset("action_arm_joint_sent", data=np.zeros((1, 7), np.float32))
    with pytest.raises(ValueError, match="validity"):
        EpisodeReader(path)


def test_discard_then_next_episode(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.start_episode()
    recorder.add_episode_frame(_frame(1))
    destination = Path(recorder.finish_episode(save=False))
    assert not destination.exists()
    assert recorder.start_episode()
    recorder.add_episode_frame(_frame(2))
    assert _save(recorder).is_dir()
