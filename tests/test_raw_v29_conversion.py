"""Preserve scientific data while removing obsolete raw runtime bookkeeping."""

import hashlib
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


def test_v29_conversion_preserves_source_and_scientific_values(tmp_path):
    from tests.raw_episode_fixture import build_raw_episode
    from dexmani_real.recording.storage.reader import EpisodeReader

    path = Path(__file__).parents[1] / "examples" / "convert_raw_v29.py"
    spec = importlib.util.spec_from_file_location("convert_raw_v29", path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    source = build_raw_episode(tmp_path / "old")
    source = source.parent if source.is_file() else source
    with h5py.File(source / "data.h5", "r+") as data:
        data["meta"].attrs["schema_version"] = 29
        count = int(data["meta"].attrs["num_frames"])
        for name, values in {
            "source_sample_index": np.arange(count, dtype=np.int64),
            "arm_last_cmd_seq": np.arange(count, dtype=np.int64),
            "fill_reason": np.zeros(count, dtype=np.uint8),
            "flag_sample_valid": np.ones(count, dtype=np.bool_),
        }.items():
            data[name] = values
        original = {k: data[k][:] for k in data if k != "meta" and k not in converter._REMOVED}
    hashes = {p.relative_to(source): hashlib.sha256(p.read_bytes()).digest()
              for p in source.rglob("*") if p.is_file()}
    destination = tmp_path / "converted"
    converter.convert_episode(source, destination)
    assert hashes == {p.relative_to(source): hashlib.sha256(p.read_bytes()).digest()
                      for p in source.rglob("*") if p.is_file()}
    with h5py.File(destination / "data.h5", "r") as data:
        assert data["meta"].attrs["schema_version"] == 30
        for name, values in original.items():
            np.testing.assert_array_equal(data[name][:], values)
        assert not (set(data) & set(converter._REMOVED))
    with EpisodeReader(destination) as reader:
        assert reader.schema_version == 30
    with pytest.raises(FileExistsError):
        converter.convert_episode(source, destination)
    with pytest.raises(ValueError):
        converter.convert_episode(source, source / "recursive")
