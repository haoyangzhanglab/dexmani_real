"""Complete episode projection, uniformity and transactional Zarr publication."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

import dexmani_real.dataset.export as exporter
from dexmani_real.dataset.contracts import ProcessingConfig
from dexmani_real.dataset.processed import MULTIMODAL_DATASET_KEYS
from dexmani_real.dataset.processing import process_episode_root
from test_control_step_dataset import permissive_test_config, write_control_episode


def processed_pair(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    write_control_episode(raw, "episode_a")
    write_control_episode(raw, "episode_b")
    with h5py.File(raw / "episode_b" / "data.h5", "r+") as second:
        second["hand_contact"][:] += 100.0
    processed = tmp_path / "processed"
    process_episode_root(raw, processed, permissive_test_config())
    return processed


def test_d8_two_files_are_two_complete_zarr_episodes(tmp_path, monkeypatch):
    processed = processed_pair(tmp_path)
    target = tmp_path / "fixture.zarr"
    validate_calls = []
    real_validate = exporter.validate_processed_hdf5

    def count_validation(path):
        validate_calls.append(path.name)
        return real_validate(path)

    monkeypatch.setattr(exporter, "validate_processed_hdf5", count_validation)
    report = exporter.export_processed_hdf5_to_zarr(
        processed, target, exporter.PolicyZarrExportConfig(chunk_frames=13)
    )
    assert validate_calls == ["episode_a.h5", "episode_b.h5"]
    assert report["episode_count"] == 2
    assert report["episode_ends"] == [40, 80]
    store = zarr.open_group(str(target), mode="r")
    assert store.attrs["schema_version"] == 11
    assert store.attrs["observation_alignment"] == "control_step_latest_causal"
    assert store.attrs["state_alignment"] == "control_step"
    assert "episode_start_policy" not in store.attrs
    assert set(store["data"].array_keys()) == set(MULTIMODAL_DATASET_KEYS)
    np.testing.assert_array_equal(store["meta"]["episode_ends"][:], [40, 80])
    for key in MULTIMODAL_DATASET_KEYS:
        expected = []
        for path in sorted(processed.glob("*.h5")):
            with h5py.File(path, "r") as source:
                expected.append(source[key][:])
        np.testing.assert_array_equal(store["data"][key][:], np.concatenate(expected))


@pytest.mark.parametrize(
    "mismatch", ["task_name", "dt", "source_frames", "dtype", "action_nan"]
)
def test_invalid_or_nonuniform_inputs_do_not_publish(tmp_path, mismatch):
    processed = processed_pair(tmp_path)
    with h5py.File(processed / "episode_b.h5", "r+") as second:
        if mismatch == "task_name":
            second.attrs["task_name"] = "other_task"
        elif mismatch == "dt":
            second.attrs["dt"] = 0.1
        elif mismatch == "source_frames":
            second.attrs["source_frames"] = 41
        elif mismatch == "dtype":
            values = second["action"][:].astype(np.float64)
            del second["action"]
            second.create_dataset("action", data=values)
        else:
            second["action"][20, 0] = np.nan
    target = tmp_path / "fixture.zarr"
    with pytest.raises(ValueError):
        exporter.export_processed_hdf5_to_zarr(processed, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".fixture.zarr.tmp-*"))


def test_dry_run_has_no_output_side_effect(tmp_path):
    processed = processed_pair(tmp_path)
    before = set(tmp_path.iterdir())
    report = exporter.preflight_processed_hdf5_to_zarr(processed)
    assert report["episode_ends"] == [40, 80]
    assert set(tmp_path.iterdir()) == before


def test_failure_after_writing_removes_unpublished_staging(tmp_path, monkeypatch):
    processed = processed_pair(tmp_path)
    target = tmp_path / "fixture.zarr"

    def fail_verification(*args, **kwargs):
        raise OSError("injected verification failure")

    monkeypatch.setattr(exporter, "_validate_zarr", fail_verification)
    with pytest.raises(OSError, match="injected verification failure"):
        exporter.export_processed_hdf5_to_zarr(processed, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".fixture.zarr.tmp-*"))
    assert len(list(processed.glob("*.h5"))) == 2


def test_dangling_target_is_occupied(tmp_path):
    processed = processed_pair(tmp_path)
    target = tmp_path / "fixture.zarr"
    target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        exporter.export_processed_hdf5_to_zarr(processed, target)
    assert target.is_symlink()
