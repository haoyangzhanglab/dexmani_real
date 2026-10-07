"""Current Raw arrays are read directly; missing evidence is never synthesized."""

import h5py
import numpy as np
import pytest

from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError
from dexmani_real.recording.storage.schema import RAW_FORMAT


def synthetic_raw(tmp_path, *, dispatch=None, metadata_dispatch=None):
    path = tmp_path / "episode_synthetic"
    path.mkdir()
    with h5py.File(path / "data.h5", "w") as data:
        data.create_group("meta").attrs.update(
            format=RAW_FORMAT,
            termination_reason="synthetic_fixture",
            num_frames=2,
            control_hz=10,
            task_label="synthetic",
            collection_source="teleop",
        )
        if metadata_dispatch is not None:
            data["meta"].attrs["dispatch_status"] = metadata_dispatch
        if dispatch is not None:
            data.create_dataset("dispatch_status", data=dispatch)
        for name, size in (
            ("arm_qpos", 7),
            ("hand_qpos", 12),
            ("action_arm_joint_target", 7),
            ("action_hand_joint_target", 12),
        ):
            data.create_dataset(name, data=np.zeros((2, size), dtype=np.float64))
    return path


def test_current_dispatch_preserves_values_and_slice(tmp_path):
    expected = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    path = synthetic_raw(tmp_path, dispatch=expected)
    with EpisodeReader(path) as reader:
        assert reader["dispatch_status"].dtype == np.uint8
        np.testing.assert_array_equal(reader["dispatch_status"][:], expected)
        np.testing.assert_array_equal(reader["dispatch_status"][1:2], expected[1:])


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float64, np.bool_])
def test_dispatch_dtype_is_not_coerced(tmp_path, dtype):
    path = synthetic_raw(tmp_path, dispatch=np.ones((2, 2), dtype=dtype))
    before = (path / "data.h5").read_bytes()
    with EpisodeReader(path) as reader:
        with pytest.raises(RawDataError, match="dispatch_status: incompatible shape/dtype"):
            reader["dispatch_status"]
    assert (path / "data.h5").read_bytes() == before


@pytest.mark.parametrize("metadata_dispatch", [None, np.ones((2, 2), dtype=np.uint8)])
def test_replay_requires_dispatch_dataset_before_fk(tmp_path, monkeypatch, metadata_dispatch):
    from dexmani_real.replay import trajectory

    path = synthetic_raw(tmp_path, metadata_dispatch=metadata_dispatch)
    before = (path / "data.h5").read_bytes()
    monkeypatch.setattr(
        trajectory,
        "compute_eef_pose_history_xarm_base",
        lambda *a: pytest.fail("missing dispatch reached replay FK"),
    )
    with pytest.raises(RawDataError, match="dispatch_status"):
        trajectory.load_trajectory(path)
    assert (path / "data.h5").read_bytes() == before


@pytest.mark.parametrize("status", [0, 1, 2, 3, 4, 255])
def test_replay_checks_current_dispatch_without_execution_path(tmp_path, monkeypatch, status):
    from dexmani_real.replay import trajectory

    path = synthetic_raw(tmp_path, dispatch=np.full((2, 2), status, dtype=np.uint8))
    monkeypatch.setattr(
        trajectory, "compute_eef_pose_history_xarm_base", lambda q: np.zeros((len(q), 9))
    )
    if status in (1, 2):
        assert trajectory.load_trajectory(path).num_frames == 2
    else:
        with pytest.raises(ValueError, match="continued dispatch"):
            trajectory.load_trajectory(path)


def test_missing_time_is_not_filled(tmp_path):
    path = synthetic_raw(tmp_path)
    with EpisodeReader(path) as reader:
        with pytest.raises(RawDataError, match="camera_timestamp_ns"):
            reader.require_fields("camera_timestamp_ns")
        with pytest.raises(KeyError):
            reader["camera_timestamp_ns"]


def test_export_rejects_missing_row_evidence(tmp_path):
    from test_policy_recording import start_recording

    from dexmani_real.dataset.processing import validate_export_episode

    recorder = start_recording(tmp_path)
    path = recorder.save_episode(reason="synthetic_fixture")
    with h5py.File(path / "data.h5", "r+") as data:
        del data["camera_timestamp_ns"]
    before = (path / "data.h5").read_bytes()
    with EpisodeReader(path) as reader:
        with pytest.raises(RawDataError, match="camera_timestamp_ns"):
            validate_export_episode(reader)
    assert (path / "data.h5").read_bytes() == before
