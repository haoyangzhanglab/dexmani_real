"""End-to-end strict whole-episode export regressions (raw -> processed -> Zarr).

No hardware. A complete JOINT-profile raw fixture flows through ``analyze_episode``
and ``_write_processed_episode`` into a processed HDF5 and then into Policy Zarr,
proving one HDF5 yields exactly one episode. A second fixture inserts a
persistent IK-failure hold: the processed HDF5 stays writable for audit, but the
strict exporter contributes zero Zarr episodes from it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.dataset.clean import analyze_episode
from dexmani_real.dataset.contracts import EpisodeAnnotation, ProcessingConfig, OutputProfile
from dexmani_real.dataset.export import export_processed_hdf5_to_zarr
from dexmani_real.dataset.processing import _write_processed_episode

from test_processed_v15 import (
    _fake_reader,
    _joint_config,
    _write_raw_fixture,
)


def _process(workdir: Path, *, mutate=None) -> tuple[Path, object]:
    """Write one raw fixture, clean it, and publish a processed HDF5."""
    raw_path = workdir / "episode_fixture.h5"
    _write_raw_fixture(raw_path, frames=40)
    if mutate is not None:
        with h5py.File(raw_path, "r+") as raw:
            mutate(raw)
    reader = _fake_reader(raw_path)
    config = _joint_config()
    annotation = EpisodeAnnotation(task_name="fixture_task")
    decision = analyze_episode(
        reader, config, annotation, source_already_validated=True
    )
    out_root = workdir / "processed"
    out_root.mkdir(parents=True, exist_ok=True)
    _write_processed_episode(reader, decision, out_root, config, task_name="fixture_task")
    out_path = out_root / f"{raw_path.name}.h5"
    return out_path, reader


class TestStrictExportEndToEnd(unittest.TestCase):
    def test_complete_episode_exports_exactly_one_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "complete"
            workdir.mkdir(parents=True)
            out_path, reader = _process(workdir)
            try:
                zarr_path = workdir / "task.zarr"
                report = export_processed_hdf5_to_zarr(out_path.parent, zarr_path)
                self.assertEqual(report["episode_count"], 1)
                self.assertEqual(report["rejected_episode_count"], 0)
                np.testing.assert_array_equal(
                    np.asarray(report["episode_ends"]),
                    np.asarray([40], dtype=np.int64),
                )
            finally:
                reader.h5f.close()

    def test_interior_ik_gap_contributes_zero_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "ik_gap"
            workdir.mkdir(parents=True)

            def mutate(raw: h5py.File) -> None:
                # Persistent IK hold (5 consecutive frames > transient threshold 4).
                raw["flag_frame_status"][17:22] = 2

            out_path, reader = _process(workdir, mutate=mutate)
            try:
                with h5py.File(out_path, "r") as processed:
                    self.assertEqual(int(processed.attrs["episode_steps"]), 35)
                zarr_path = workdir / "task.zarr"
                report = export_processed_hdf5_to_zarr(out_path.parent, zarr_path)
                self.assertEqual(report["episode_count"], 0)
                self.assertEqual(report["rejected_episode_count"], 1)
                self.assertEqual(
                    report["rejected_episodes"][0]["episode"], "episode_fixture.h5"
                )
            finally:
                reader.h5f.close()


if __name__ == "__main__":
    unittest.main()
