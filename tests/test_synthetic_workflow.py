"""The public synthetic entry uses the real writer, reader, cloud processing and windows."""

import numpy as np
import pytest
import zarr

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.dataset.contracts import ProcessingConfig
from dexmani_real.dataset.export import export_raw_to_zarr
from dexmani_real.recording.storage.reader import EpisodeReader
from examples.create_synthetic_episode import create_synthetic_episode
from examples.read_policy_windows import policy_windows


def test_synthetic_raw_export_and_windows(tmp_path):
    saved = create_synthetic_episode(tmp_path / "raw")
    with EpisodeReader(saved) as reader:
        assert reader.num_frames == 8
        assert reader.meta["execution_path"] == "synthetic_offline_v1"
        assert reader.read_camera_frame("rgb", 7).shape == (32, 32, 3)
    processing = ProcessingConfig.from_runtime(
        ExperimentConfig(), table_plane_abcd=(0, 0, 1, -0.022)
    )
    output = tmp_path / "canonical.zarr"
    report = export_raw_to_zarr(saved, output, processing=processing)
    assert report["total_frames"] == 8
    root = zarr.open_group(str(output), mode="r")
    assert len(root["data"]) == 13
    assert np.isfinite(root["data/point_cloud"][:]).all()
    assert np.isnan(root["data/tactile_force"][:]).all()
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [8])
    windows = policy_windows(root, ("joint_state", "point_cloud"), 4)
    assert len(windows) == 5
    np.testing.assert_array_equal(windows[-1], [4, 5, 6, 7])
    assert policy_windows(root, ("tactile_force",), 4) == []
    with pytest.raises(FileExistsError):
        create_synthetic_episode(tmp_path / "raw")
    from dexmani_policy.datasets.base_dataset import BaseDataset

    dataset = BaseDataset(str(output), horizon=4)
    assert len(dataset) > 0
