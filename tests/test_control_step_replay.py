"""Offline processed-source loading; no hardware or physical replay."""

import h5py
import numpy as np
import pytest

from dexmani_real.dataset.contracts import OutputProfile, ProcessingConfig
from dexmani_real.dataset.processing import process_episode_root
from dexmani_real.replay.trajectory import load_processed_trajectory
from test_control_step_dataset import write_control_episode


def test_processed_replay_loads_all_raw_float64_sent_targets(tmp_path):
    raw_path = write_control_episode(tmp_path / "raw")
    output = tmp_path / "processed"
    process_episode_root(raw_path, output, ProcessingConfig(profile=OutputProfile.JOINT))
    artifact = output / f"{raw_path.name}.h5"
    with h5py.File(artifact, "r+") as processed:
        processed["action"][:] = 99.0
    trajectory = load_processed_trajectory(str(artifact))
    with h5py.File(raw_path / "data.h5", "r") as raw:
        assert trajectory.num_frames == 40
        assert trajectory.action_arm_joint.dtype == np.float64
        np.testing.assert_array_equal(
            trajectory.action_arm_joint, raw["action_arm_joint_sent"][:]
        )
        np.testing.assert_array_equal(
            trajectory.action_hand_joint, raw["action_hand_joint"][:]
        )


def test_processed_replay_refuses_different_raw_length(tmp_path):
    raw_path = write_control_episode(tmp_path / "raw")
    other_path = write_control_episode(tmp_path / "other", frames=35)
    output = tmp_path / "processed"
    process_episode_root(raw_path, output, ProcessingConfig(profile=OutputProfile.JOINT))
    artifact = output / f"{raw_path.name}.h5"
    with h5py.File(artifact, "r+") as processed:
        processed.attrs["source_path"] = str(other_path.resolve())
    with pytest.raises(ValueError, match="source frame count does not match"):
        load_processed_trajectory(str(artifact))
