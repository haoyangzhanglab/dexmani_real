from __future__ import annotations

import h5py
import numpy as np
import pytest

from dexmani_real.policy.vr_teleop_policy import (
    PolicyConfig,
    _hand_ramp_frame_count,
    _record_frame,
    _smoothstep_hand_ramp,
    _update_audio_motion_gate,
)
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.recording.episode_recorder import EpisodeRecorder


def test_responsiveness_defaults_are_time_based() -> None:
    config = PolicyConfig()

    assert config.hand_output_smoothing_alpha == pytest.approx(0.5)
    assert config.hand_ramp_duration_s == pytest.approx(0.5)
    assert config.begin_motion_gate_timeout_s == pytest.approx(0.35)
    assert _hand_ramp_frame_count(config.hand_ramp_duration_s, config.control_hz) == 8


def test_hand_ramp_is_monotonic_and_reaches_target_on_last_frame() -> None:
    start = np.zeros(12)
    target = np.ones(12)
    total = _hand_ramp_frame_count(0.5, 16.0)
    samples = np.stack([_smoothstep_hand_ramp(start, target, step, total) for step in range(total)])

    assert np.all(samples[0] > start)
    assert np.all(samples[0] < target)
    assert np.all(np.diff(samples[:, 0]) > 0)
    np.testing.assert_allclose(samples[-1], target)


def test_begin_audio_gate_expires_without_reentering_while_cue_plays() -> None:
    hold, deadline, ignore = _update_audio_motion_gate(
        audio_playing=True,
        begin_deadline_s=10.35,
        ignore_begin_until_silent=False,
        now_s=10.0,
    )
    assert hold is True

    hold, deadline, ignore = _update_audio_motion_gate(
        audio_playing=True,
        begin_deadline_s=deadline,
        ignore_begin_until_silent=ignore,
        now_s=10.36,
    )
    assert hold is False
    assert deadline is None
    assert ignore is True

    hold, deadline, ignore = _update_audio_motion_gate(
        audio_playing=True,
        begin_deadline_s=deadline,
        ignore_begin_until_silent=ignore,
        now_s=10.5,
    )
    assert hold is False

    hold, deadline, ignore = _update_audio_motion_gate(
        audio_playing=False,
        begin_deadline_s=deadline,
        ignore_begin_until_silent=ignore,
        now_s=11.0,
    )
    assert hold is False
    assert ignore is False


def test_legacy_reader_prefers_grid_rate_over_wall_time_fps(tmp_path) -> None:
    episode_path = tmp_path / "legacy.h5"
    with h5py.File(episode_path, "w") as h5_file:
        meta = h5_file.create_group("meta")
        meta.attrs["schema_version"] = 12
        meta.attrs["control_hz"] = 16.0
        meta.attrs["fps"] = 11.99
        meta.attrs["duration"] = 35.6
        meta.attrs["num_frames"] = 427
        h5_file.create_dataset("timestamp", data=np.arange(427, dtype=np.float64) / 16.0)

    with EpisodeReader(episode_path) as reader:
        timing = reader.timing

    assert timing.rate_hz == pytest.approx(16.0)
    assert timing.grid_dt_s == pytest.approx(1.0 / 16.0)
    assert timing.grid_duration_s == pytest.approx(26.625)
    assert timing.wall_duration_s == pytest.approx(35.6)
    assert timing.non_sampled_duration_s == pytest.approx(8.975)


def test_active_frame_persists_raw_targets_and_policy_stage_timing(tmp_path) -> None:
    recorder = EpisodeRecorder(str(tmp_path), max_frames=4, control_hz=16.0, min_frames=1)
    assert recorder.start_episode(task_label="telemetry-test")
    vr_frame = {
        "wrist_pos": np.zeros(3),
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.zeros((21, 3)),
        "head_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    raw_pos = np.array([0.4, 0.1, 0.2])
    raw_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    raw_hand = np.linspace(0.0, 0.11, 12)

    _record_frame(
        recorder,
        arm_state=None,
        hand_state=None,
        arm_cmd=np.zeros(7),
        hand_cmd=np.zeros(12),
        target_pos=raw_pos,
        target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        vr_frame=vr_frame,
        cam=None,
        ik_solve_time_ms=1.2,
        target_pos_before_clamp=raw_pos,
        target_eef_pos_raw=raw_pos,
        target_eef_rot6d_raw=raw_rot6d,
        action_hand_joint_raw=raw_hand,
        policy_map_time_ms=0.2,
        hand_retarget_time_ms=2.3,
        transition_check_time_ms=0.4,
        policy_compute_time_ms=4.5,
    )
    episode_path = recorder.stop_episode(success=True)
    assert episode_path is not None
    assert recorder.join_stop(timeout=5.0)

    with h5py.File(f"{episode_path}/data.h5", "r") as h5_file:
        np.testing.assert_allclose(h5_file["target_eef_pos_raw"][0], raw_pos)
        np.testing.assert_allclose(h5_file["target_eef_rot6d_raw"][0], raw_rot6d)
        np.testing.assert_allclose(h5_file["action_hand_joint_raw"][0], raw_hand)
        assert float(h5_file["policy_map_time_ms"][0]) == pytest.approx(0.2)
        assert float(h5_file["hand_retarget_time_ms"][0]) == pytest.approx(2.3)
        assert float(h5_file["transition_check_time_ms"][0]) == pytest.approx(0.4)
        assert float(h5_file["policy_compute_time_ms"][0]) == pytest.approx(4.5)
