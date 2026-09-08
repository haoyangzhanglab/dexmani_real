"""Offline raw-v25 source-row recording and storage regressions."""

from pathlib import Path
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
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import (
    DATASET_SPECS,
    SOURCE_FRAME_DATASET_NAMES,
)
from dexmani_real.recording.storage.video import VideoDecoder
from tools.convert_raw_v24_to_v25 import KEEP_DATASETS


def test_schema_matches_frozen_converter_projection():
    assert len(DATASET_SPECS) == 41
    assert set(DATASET_SPECS) == set(KEEP_DATASETS)
    assert len(SOURCE_FRAME_DATASET_NAMES) == 37


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
    destination = Path(recorder.stop_episode(save=True))
    assert recorder.join_stop(timeout=10), recorder.stop_error
    return destination


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
            assert int(f["meta"].attrs["schema_version"]) == 25
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
        f["meta"].attrs["schema_version"] = 25
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
        destination = Path(recorder.stop_episode(save=True))
        assert not recorder.join_stop(timeout=10)
    assert not destination.exists()
    assert not list(tmp_path.glob(".tmp_episode_*"))
    assert "injected" in recorder.stop_error


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
    destination = Path(recorder.stop_episode(save=False))
    assert recorder.join_stop(timeout=10)
    assert not destination.exists()
    assert recorder.start_episode()
    recorder.add_episode_frame(_frame(2))
    assert _save(recorder).is_dir()
