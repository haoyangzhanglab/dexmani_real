"""Legacy normalized-v26 conservative validity and telemetry regressions.

The frozen v25->v26 converter copied the collapsed legacy freshness flag into
``tactile_sum_fresh`` as a conservative proxy, and v26 has no independent
aggregate contact source.  A finite legacy payload alone therefore never proves
a valid measurement (old drivers could copy zero placeholders), while a zero
payload is not intrinsically invalid either: the mask comes from provenance
and representation flags only, never from payload values.  These tests pin
the v26 write-path rules, the exact-copied telemetry arrays, and that
current v28 freshness flags stay pure telemetry.

Run with:

    python -m pytest -q tests/test_legacy_v26_processing.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dexmani_real.dataset.processing import process_episode_root
from test_control_step_dataset import permissive_test_config, write_control_episode


def _write_legacy_v26_episode(root: Path, name: str = "episode_v26") -> Path:
    """Current-v28 fixture narrowed to the historical v26 schema.

    Mirrors the recipe pinned by test_recording_control_contact: v26 kept
    aggregate contact with its hand row and had no independent contact source.
    """
    episode = write_control_episode(root, name)
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["meta"].attrs["schema_version"] = 26
        del raw["hand_contact_source_monotonic_ns"]
    return episode


def _process(tmp_path: Path, episode: Path):
    output = tmp_path / "processed"
    report = process_episode_root(episode, output, permissive_test_config())
    assert report["accepted_source_episode_count"] == 1
    assert report["processed_frame_count"] == report["source_frame_count"]
    return output / f"{episode.name}.h5"


def test_v26_finite_contact_with_stale_sum_fresh_is_invalid(tmp_path):
    """The historical over-claim regression: finite (even non-zero) legacy
    contact without the conservative freshness proxy is not a valid
    measurement, and the row is still retained."""
    episode = _write_legacy_v26_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["tactile_sum_fresh"][7] = False
    artifact = _process(tmp_path, episode)
    with h5py.File(artifact, "r") as processed:
        assert processed["contact_force_valid"].shape == (40,)
        assert bool(processed["contact_force_valid"][7]) is False
        assert bool(processed["contact_force_valid"][6]) is True
        assert bool(processed["contact_force_valid"][8]) is True
        assert np.all(np.isfinite(processed["contact_force"][7]))


def test_v26_fresh_finite_contact_is_valid(tmp_path):
    episode = _write_legacy_v26_episode(tmp_path / "raw")
    artifact = _process(tmp_path, episode)
    with h5py.File(artifact, "r") as processed:
        assert bool(np.all(processed["contact_force_valid"][:]))


def test_v26_zero_contact_is_not_intrinsically_invalid(tmp_path):
    """Zero can be a real no-contact reading; only provenance decides."""
    episode = _write_legacy_v26_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["hand_contact"][:] = 0.0
    artifact = _process(tmp_path, episode)
    with h5py.File(artifact, "r") as processed:
        assert bool(np.all(processed["contact_force_valid"][:]))
        assert np.all(processed["contact_force"][:] == 0.0)


def test_v26_dense_stale_fresh_is_invalid_while_contact_stays_valid(tmp_path):
    episode = _write_legacy_v26_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["tactile_fresh"][9] = False
    artifact = _process(tmp_path, episode)
    with h5py.File(artifact, "r") as processed:
        assert bool(processed["tactile_force_valid"][9]) is False
        assert bool(processed["tactile_force_valid"][8]) is True
        # Aggregate validity is independent of dense freshness.
        assert bool(processed["contact_force_valid"][9]) is True


def test_v26_uncalibrated_contact_is_invalid(tmp_path):
    """Freshness alone never proves the bias-corrected representation: a
    legacy row recorded while software calibration had failed is invalid even
    with its conservative freshness proxy set."""
    episode = _write_legacy_v26_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["tactile_calibrated"][6] = False
    artifact = _process(tmp_path, episode)
    with h5py.File(artifact, "r") as processed:
        assert bool(processed["contact_force_valid"][6]) is False
        assert bool(processed["contact_force_valid"][5]) is True
        assert np.all(np.isfinite(processed["contact_force"][6]))


def test_v26_wrong_unit_contact_is_invalid(tmp_path):
    episode = _write_legacy_v26_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["tactile_unit_code"][8] = 1
    artifact = _process(tmp_path, episode)
    with h5py.File(artifact, "r") as processed:
        assert bool(processed["contact_force_valid"][8]) is False
        assert bool(processed["contact_force_valid"][7]) is True


def test_v26_telemetry_arrays_are_exact_copies(tmp_path):
    episode = _write_legacy_v26_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["tactile_sum_fresh"][3] = False
        raw["tactile_fresh"][5] = False
        raw["tactile_calibrated"][6] = False
        raw["tactile_unit_code"][8] = 1
        expected = {
            "contact_force_fresh": raw["tactile_sum_fresh"][:].copy(),
            "tactile_force_fresh": raw["tactile_fresh"][:].copy(),
            "tactile_calibrated": raw["tactile_calibrated"][:].copy(),
            "tactile_unit_code": raw["tactile_unit_code"][:].copy(),
        }
    artifact = _process(tmp_path, episode)
    with h5py.File(artifact, "r") as processed:
        for key, values in expected.items():
            np.testing.assert_array_equal(processed[key][:], values)
            assert processed[key].dtype == values.dtype
        # The provenance flags behind the copies also drive dense validity.
        assert bool(processed["tactile_force_valid"][6]) is False
        assert bool(processed["tactile_force_valid"][8]) is False
        assert bool(processed["contact_force_valid"][3]) is False


def test_v28_freshness_flags_never_gate_validity(tmp_path):
    """Current raw keeps direct producer evidence (independent contact source,
    calibrated native-unit dense with positive source); *_fresh stays timing
    telemetry and is still copied exactly."""
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["tactile_sum_fresh"][4] = False
        raw["tactile_fresh"][6] = False
    output = tmp_path / "processed"
    process_episode_root(episode, output, permissive_test_config())
    artifact = output / f"{episode.name}.h5"
    with h5py.File(artifact, "r") as processed:
        assert bool(processed["contact_force_valid"][4]) is True
        assert bool(processed["tactile_force_valid"][6]) is True
        assert bool(processed["contact_force_fresh"][4]) is False
        assert bool(processed["tactile_force_fresh"][6]) is False
        assert bool(np.all(processed["tactile_calibrated"][:]))
        assert np.all(processed["tactile_unit_code"][:] == 0)
