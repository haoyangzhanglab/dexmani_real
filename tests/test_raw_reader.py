"""Historical dispatch must survive loading without lossy coercion."""

import h5py
import numpy as np
import pytest

from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError
from dexmani_real.recording.storage.schema import RAW_FORMAT


def synthetic_raw(tmp_path, statuses=None, *, current=None):
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
        if statuses is not None:
            data["meta"].attrs["dispatch_status"] = statuses
        if current is not None:
            data.create_dataset("dispatch_status", data=current)
        for name, size in (
            ("arm_qpos", 7),
            ("hand_qpos", 12),
            ("action_arm_joint_target", 7),
            ("action_hand_joint_target", 12),
        ):
            data.create_dataset(name, data=np.zeros((2, size), dtype=np.float64))
    return path


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64, np.float64])
def test_historical_dispatch_preserves_numeric_codes_and_slice(tmp_path, dtype):
    expected = np.array([[0, 1], [2, 4]], dtype=dtype)
    path = synthetic_raw(tmp_path, expected)
    with EpisodeReader(path) as reader:
        actual = reader.read_row_info("dispatch_status", 0, 2)
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(reader.read_row_info("dispatch_status", 1, 2), expected[1:])


@pytest.mark.parametrize("value", [257, 258, -255, 1.5, np.nan, np.inf, True, b"1"])
def test_malformed_historical_dispatch_is_rejected_without_modifying_raw(tmp_path, value):
    path = synthetic_raw(tmp_path, np.full((2, 2), value))
    before = (path / "data.h5").read_bytes()
    with EpisodeReader(path) as reader:
        with pytest.raises(RawDataError, match="historical dispatch status values"):
            reader.read_row_info("dispatch_status", 0, 2)
    assert (path / "data.h5").read_bytes() == before


def test_malformed_historical_dispatch_cannot_admit_replay(tmp_path, monkeypatch):
    from dexmani_real.replay import trajectory

    path = synthetic_raw(tmp_path, np.full((2, 2), 257))
    monkeypatch.setattr(
        trajectory,
        "compute_eef_pose_history_xarm_base",
        lambda *a: pytest.fail("invalid dispatch reached replay FK"),
    )
    with pytest.raises(RawDataError):
        trajectory.load_trajectory(path)


def test_current_dispatch_takes_precedence_over_historical_metadata(tmp_path):
    current = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    path = synthetic_raw(tmp_path, np.full((2, 2), 257), current=current)
    with EpisodeReader(path) as reader:
        np.testing.assert_array_equal(reader.read_row_info("dispatch_status", 0, 2), current)


def test_missing_historical_evidence_stays_unknown(tmp_path):
    path = synthetic_raw(tmp_path)
    with EpisodeReader(path) as reader:
        np.testing.assert_array_equal(reader.read_row_info("dispatch_status", 0, 2), [[4, 4]] * 2)
        np.testing.assert_array_equal(reader.read_row_info("camera_timestamp_ns", 0, 2), [0, 0])
