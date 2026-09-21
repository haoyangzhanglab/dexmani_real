"""Real raw media, FK, point clouds, atomic export and the public policy reader."""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.dataset.export import export_raw_to_zarr, PolicyZarrExportConfig
from tests.raw_episode_fixture import build_policy_raw_episode, policy_processing_config


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def fixture(tmp_path):
    root = tmp_path / "provenance_fixture"
    build_policy_raw_episode(root / "episode_a", num_frames=3)
    build_policy_raw_episode(root / "episode_b", num_frames=4)
    return root


def test_raw_export_and_actual_policy_reader(tmp_path):
    from dexmani_policy.datasets.real_policy_contract import (
        build_real_policy_data_semantics,
    )
    from dexmani_policy.datasets.base_dataset import BaseDataset

    root = fixture(tmp_path)
    before = hashes(root)
    target = tmp_path / "policy.zarr"
    config = PolicyZarrExportConfig(chunk_frames=2)
    dry = export_raw_to_zarr(
        root, target, config, processing=policy_processing_config(), dry_run=True
    )
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))
    report = export_raw_to_zarr(
        root, target, config, processing=policy_processing_config()
    )
    assert report["episode_ends"] == dry["episode_ends"] == [3, 7]
    assert hashes(root) == before
    result = zarr.open_group(str(target), mode="r")
    data = result["data"]
    assert len(data) == 18
    with h5py.File(root / "episode_b/data.h5") as source:
        for output, raw in [
            ("action", "action_arm_joint_sent"),
            ("joint_state", "arm_qpos"),
        ]:
            np.testing.assert_array_equal(
                data[output][3:, :7], source[raw][:].astype(np.float32)
            )
        np.testing.assert_array_equal(
            data["camera_source_monotonic_ns"][3:],
            source["camera_source_monotonic_ns"][:],
        )
    assert np.any(data["point_cloud"][:, :, :3] != 0)
    assert np.any(data["fingertip_points"][:] != 0)
    modalities = [
        "joint_state",
        "point_cloud",
        "rgb",
        "eef_pose",
        "fingertip_points",
        "contact_force",
        "tactile_force",
    ]
    for action_key in ["action", "action_ee"]:
        build_real_policy_data_semantics(
            target,
            task_name="provenance_fixture",
            observation_fields=modalities,
            agent_config={"num_points": 1024},
            action_key=action_key,
        )
        reader = BaseDataset(
            zarr_path=str(target),
            sensor_modalities=modalities,
            action_key=action_key,
            horizon=2,
            val_ratio=0,
        )
        sample = reader[0]
        assert set(sample["obs"]) == set(modalities)
        assert tuple(sample["action"].shape) == (
            2,
            19 if action_key == "action" else 21,
        )
        np.testing.assert_array_equal(sample["action"].numpy(), data[action_key][:2])
        assert reader.replay_buffer.n_episodes == 2


@pytest.mark.parametrize(
    "case",
    [
        "missing_depth",
        "short_rgb",
        "nonfinite",
        "future_source",
        "invalid_tactile",
        "policy_eval",
        "task_conflict",
        "empty_cloud",
    ],
)
@pytest.mark.parametrize("dry", [True, False])
def test_corrupt_episode_rejects_whole_output_and_keeps_raw(tmp_path, case, dry):
    root = fixture(tmp_path)
    episode = root / "episode_b"
    if case == "missing_depth":
        (episode / "depth.h5").unlink()
    elif case == "short_rgb":
        shutil.copyfile(root / "episode_a/rgb.mp4", episode / "rgb.mp4")
    elif case == "empty_cloud":
        with h5py.File(episode / "depth.h5", "r+") as f:
            f["depth"][:] = 0
    else:
        with h5py.File(episode / "data.h5", "r+") as f:
            if case == "nonfinite":
                f["arm_qpos"][1, 0] = np.nan
            if case == "future_source":
                f["hand_source_monotonic_ns"][1] = 100000
            if case == "invalid_tactile":
                f["hand_tactile_force_valid"][1] = False
            if case == "policy_eval":
                f["meta"].attrs["provenance_workflow"] = "policy_eval"
            if case == "task_conflict":
                f["meta"].attrs["task_label"] = "other"
    before = hashes(root)
    target = tmp_path / "policy.zarr"
    with pytest.raises((ValueError, FileNotFoundError)):
        export_raw_to_zarr(
            root, target, processing=policy_processing_config(), dry_run=dry
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))
    assert hashes(root) == before


def test_masks_and_timestamp_gaps_are_preserved_not_trimmed(tmp_path):
    root = fixture(tmp_path)
    with h5py.File(root / "episode_b/data.h5", "r+") as f:
        f["timestamp"][2:] += 1
        f["observation_anchor_monotonic_ns"][2:] += 1000000000
        f["hand_contact_valid"][1] = False
        f["hand_contact"][1] = np.nan
        f["hand_tactile_force_valid"][1] = False
        f["hand_tactile_force"][1] = np.nan
    target = tmp_path / "policy.zarr"
    export_raw_to_zarr(root, target, processing=policy_processing_config())
    result = zarr.open_group(str(target), mode="r")
    assert result["meta/episode_ends"][:].tolist() == [3, 7]
    assert not result["data/contact_force_valid"][4]
    assert np.isnan(result["data/contact_force"][4]).all()
    assert np.isnan(result["data/tactile_force"][4]).all()
    with h5py.File(root / "episode_b/data.h5") as f:
        np.testing.assert_array_equal(
            result["data/observation_anchor_monotonic_ns"][3:],
            f["observation_anchor_monotonic_ns"][:],
        )


def test_annotation_exclusion_task_override_conflict_and_overwrite(tmp_path):
    root = fixture(tmp_path)
    (root / "episode_b/data.h5").unlink()
    annotations = tmp_path / "annotations.yaml"
    annotations.write_text(
        "episodes:\n  episode_b:\n    include: false\n  episode_a:\n    task_name: selected\n"
    )
    target = tmp_path / "policy.zarr"
    with pytest.raises(ValueError):
        export_raw_to_zarr(
            root,
            target,
            annotations_path=annotations,
            task_name="conflict",
            processing=policy_processing_config(),
        )
    report = export_raw_to_zarr(
        root,
        target,
        annotations_path=annotations,
        processing=policy_processing_config(),
    )
    assert report["episode_ends"] == [3] and report["task_name"] == "selected"
    assert report["excluded_episodes"] == ["episode_b"]
    before = hashes(target)
    for dry in [True, False]:
        with pytest.raises(FileExistsError):
            export_raw_to_zarr(
                root, target, dry_run=dry, processing=policy_processing_config()
            )
    assert hashes(target) == before


def test_actual_cli_dry_run_and_export(tmp_path):
    import yaml

    root = fixture(tmp_path)
    config = policy_processing_config()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "pointcloud": config.pointcloud.to_dict(),
                "environment": {"table": {"enabled": False}},
            }
        )
    )
    target = tmp_path / "result.zarr"
    before = hashes(root)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "examples/export_policy_zarr.py"),
        str(root),
        "--output",
        str(target),
        "--config",
        str(yaml_path),
    ]
    for dry in [True, False]:
        result = subprocess.run(
            command + (["--dry-run"] if dry else []),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert target.exists() is not dry
    assert hashes(root) == before
