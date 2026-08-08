from __future__ import annotations

import h5py
import numpy as np
import pytest

from dexmani_real.config.defaults import hand
from dexmani_real.recording.timestamp_buffer import TimestampAlignedBuffer, _get_accumulate_timestamp_idxs
from dexmani_real.tools.episode_quality import EpisodeQuality, _grid_fill_mask


def test_timestamp_mapping_backfills_every_skipped_slot() -> None:
    local, global_, next_idx = _get_accumulate_timestamp_idxs([10.0, 10.2], start_time=10.0, dt=0.1)

    assert local == [0, 1, 1]
    assert global_ == [0, 1, 2]
    assert next_idx == 3


def test_timestamp_buffer_backfills_data_without_zero_holes() -> None:
    buffer = TimestampAlignedBuffer(start_time=10.0, dt=0.1, max_record_steps=8)
    buffer.add({"value": np.array([1.0, 2.0])}, timestamp=10.0)
    buffer.add({"value": np.array([3.0, 4.0])}, timestamp=10.2)

    np.testing.assert_array_equal(buffer.data["value"], [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    np.testing.assert_array_equal(buffer.data["flag_sample_valid"], [True, False, True])
    np.testing.assert_allclose(buffer.timestamps, [10.0, 10.1, 10.2])


def test_timestamp_buffer_keeps_first_sample_in_grid_bucket() -> None:
    buffer = TimestampAlignedBuffer(start_time=10.0, dt=0.1, max_record_steps=8)
    buffer.add({"value": np.array([1])}, timestamp=10.01)
    buffer.add({"value": np.array([2])}, timestamp=10.02)

    assert buffer.size == 1
    np.testing.assert_array_equal(buffer.data["value"], [[1]])
    np.testing.assert_array_equal(buffer.data["flag_sample_valid"], [True])


def test_grid_fill_mask_prefers_v12_validity_flag(tmp_path) -> None:
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("flag_sample_valid", data=[True, False, True, False])
        h5_file.create_dataset("timestamp", data=[1.0, 1.1, 1.2, 1.3])
        np.testing.assert_array_equal(_grid_fill_mask(h5_file, 4), [False, True, False, True])


def test_grid_fill_mask_detects_legacy_zero_hole() -> None:
    class LegacyFile(dict):
        pass

    legacy = LegacyFile(timestamp=np.array([1.0, 1.1, 0.0, 1.3]))
    np.testing.assert_array_equal(_grid_fill_mask(legacy, 4), [False, False, True, False])


def test_validation_excludes_legacy_hole_from_variance(tmp_path) -> None:
    path = tmp_path / "episode.h5"
    with h5py.File(path, "w") as h5_file:
        meta = h5_file.create_group("meta")
        meta.attrs["control_hz"] = 10.0
        h5_file.create_dataset("timestamp", data=[1.0, 1.1, 0.0, 1.3])
        h5_file.create_dataset("arm_qpos", data=np.arange(28, dtype=np.float64).reshape(4, 7))
        h5_file.create_dataset("action_arm_joint", data=np.arange(28, dtype=np.float64).reshape(4, 7))
        hand_command = np.full((4, 12), 0.2)
        hand_command[2] = 0.0  # zero-initialized v11 hole, not real motion
        h5_file.create_dataset("action_hand_joint", data=hand_command)

    with EpisodeQuality(path) as quality:
        report = quality.validate(min_frames=1)

    hand_check = next(
        check
        for check in report.checks
        if check["name"] == "non_zero_variance" and check["detail"].startswith("action_hand_joint")
    )
    assert not hand_check["passed"]
    assert "12/12 dims zero variance" in hand_check["detail"]


def test_health_reports_hand_feedback_bound_tolerance(tmp_path) -> None:
    path = tmp_path / "episode.h5"
    lower = np.asarray(hand.qpos_min_rad, dtype=np.float64)
    upper = np.asarray(hand.qpos_max_rad, dtype=np.float64)
    qpos = np.repeat(((lower + upper) / 2.0)[np.newaxis], 3, axis=0)
    qpos[1] = lower
    qpos[1, 0] -= 0.005  # tolerated feedback excursion
    qpos[2] = lower
    qpos[2, 0] -= 0.020  # over-tolerance feedback excursion

    with h5py.File(path, "w") as h5_file:
        meta = h5_file.create_group("meta")
        meta.attrs["control_hz"] = 10.0
        meta.attrs["hand_feedback_bound_tolerance_rad"] = 0.01
        h5_file.create_dataset("arm_qpos", data=np.zeros((3, 7)))
        h5_file.create_dataset("hand_qpos", data=qpos)

    with EpisodeQuality(path) as quality:
        report = quality.health()

    assert report.hand_feedback_bound_violation_pct == pytest.approx(200.0 / 3.0)
    assert report.hand_feedback_bound_over_tolerance_pct == pytest.approx(100.0 / 3.0)
    assert report.hand_feedback_bound_max_violation_deg == pytest.approx(np.rad2deg(0.02))
    assert report.hand_feedback_bound_violation_per_joint_pct[0] == pytest.approx(200.0 / 3.0)
    assert report.hand_feedback_bound_over_tolerance_per_joint_pct[0] == pytest.approx(100.0 / 3.0)
    assert any("outside command bounds beyond tolerance" in warning for warning in report.warnings)
