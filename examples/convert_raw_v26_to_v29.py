#!/usr/bin/env python3
"""LEGACY PROJECTION: raw v26 -> raw v29 (a *false semantic migration*).

This is NOT one of the repo's sanctioned frozen migrations.  The current raw
contract (schema v29) requires ``hand_contact`` and ``hand_tactile_force`` to
originate from a single causal hand sample (one ``hand_state_ring`` read).  v26
selected the aggregate and dense tactile payloads from separate rings, so that
atomicity invariant cannot be reconstructed from stored data.  Project docs
explicitly call converting old raw into the vNext schema and claiming atomic
hand-sample semantics a false semantic migration.

This tool exists only to produce a *best-effort* v29-shaped projection so that
historical pick_place_toy episodes can pass the current structural gates and be
re-processed / re-exported alongside genuine ``collect_teleop.py`` v29 data.
The projection is lossy in these documented ways:

  * ``hand_contact_valid``   := tactile_sum_fresh & tactile_calibrated
  * ``hand_tactile_force_valid`` := tactile_fresh   & tactile_calibrated
    (v26 freshness carries an extra 250ms age gate v29 does not, so a few
    valid-but-old rows map to invalid and are NaN-masked.)
  * ``camera_health`` := 0 (OK) where flag_camera_fresh else 4 (DELIVERY_DELAY);
    v26 never persisted the health enum, and nothing downstream consumes it.
  * ``hand_contact`` / ``hand_tactile_force`` are downcast float64 -> float32
    (no scale change: v26 and v29 are both SDK-native).

It never rewrites the source and never touches hardware.  Inputs are read-only;
``depth.h5`` / ``rgb.mp4`` are hard-linked (fallback copy on EXDEV).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Load the authoritative v29 schema module directly, bypassing the package
# __init__ chain (dexmani_real.recording -> reader -> PyAV), which this
# offline tool does not need.  schema.py only depends on dataclasses/enum/numpy.
_schema_path = _REPO_ROOT / "dexmani_real" / "recording" / "storage" / "schema.py"
_schema_spec = importlib.util.spec_from_file_location("_raw_episode_schema", _schema_path)
_schema = importlib.util.module_from_spec(_schema_spec)
_schema_spec.loader.exec_module(_schema)
DATASET_SPECS = _schema.DATASET_SPECS
EPISODE_SCHEMA_VERSION = _schema.EPISODE_SCHEMA_VERSION
validate_data_layout = _schema.validate_data_layout

# v26-only source datasets with no v29 home; dropped silently (not in DATASET_SPECS).
_DROPPED_DATASETS = (
    "policy_observation_arm_qpos",
    "policy_observation_hand_qpos",
    "policy_observation_valid",
    "tactile_calibrated",
    "tactile_fresh",
    "tactile_source_monotonic_ns",
    "tactile_sum_fresh",
    "tactile_unit_code",
)
# float64 -> float32 payload downcast (no scale change; v26 is SDK-native).
_TACTILE_DATASETS = ("hand_contact", "hand_tactile_force")
# v29-only synthesized fields.
_SYNTH_FIELDS = ("hand_contact_valid", "hand_tactile_force_valid", "camera_health")
# v29 reader does not write these legacy attrs; drop them from the projection.
_META_DROP = (
    "converted_from_schema",
    "grid_dt_s",
    "grid_duration_s",
    "non_sampled_duration_s",
)
_CAMERA_HEALTH_OK = np.uint8(0)
_CAMERA_HEALTH_DELIVERY_DELAY = np.uint8(4)


def _verify(directory: Path, version: int) -> int:
    """Structural pre/post check; returns the frame count."""
    with h5py.File(directory / "data.h5", "r") as data:
        meta = data["meta"].attrs
        if int(meta.get("schema_version", -1)) != version:
            raise ValueError(f"expected raw schema v{version}")
        count = int(meta["num_frames"])
        if count < 0:
            raise ValueError("num_frames must be non-negative")
        if version == EPISODE_SCHEMA_VERSION:
            shapes = {
                name: tuple(ds.shape) for name, ds in data.items() if isinstance(ds, h5py.Dataset)
            }
            dtypes = {name: ds.dtype for name, ds in data.items() if isinstance(ds, h5py.Dataset)}
            errors = validate_data_layout(shapes, dtypes, frame_count=count)
            if errors:
                raise ValueError("layout: " + "; ".join(errors))
        else:
            for name in _DROPPED_DATASETS + _TACTILE_DATASETS:
                if name not in data or data[name].shape[:1] != (count,):
                    raise ValueError(f"missing/wrong first dimension: {name}")
    with h5py.File(directory / "depth.h5", "r") as depth:
        if "depth" not in depth or depth["depth"].shape[:1] != (count,):
            raise ValueError("depth frame count does not match num_frames")
    if not (directory / "rgb.mp4").is_file() or (directory / "rgb.mp4").stat().st_size == 0:
        raise ValueError("rgb.mp4 must exist and be non-empty")
    return count


def convert_episode(source: Path, destination: Path) -> Path:
    """Publish a v29-shaped projection of one v26 episode, out-of-place."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.resolve() == source or source in destination.resolve().parents:
        raise ValueError("destination must be outside the source episode")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    _verify(source, 26)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive mkdir owns this staging path; never clean another invocation's work.
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    staging.mkdir()
    try:
        with (
            h5py.File(source / "data.h5", "r") as src,
            h5py.File(staging / "data.h5", "w") as dst,
        ):
            meta = dst.create_group("meta")
            for name in src["meta"].attrs:
                if name in _META_DROP or name == "schema_version":
                    continue
                meta.attrs[name] = src["meta"].attrs[name]
            meta.attrs["schema_version"] = EPISODE_SCHEMA_VERSION
            meta.attrs["camera_writer_error"] = ""

            # Synthesized validity masks from v26 freshness/calibration bits.
            tactile_sum_fresh = np.asarray(src["tactile_sum_fresh"][:], dtype=bool)
            tactile_fresh = np.asarray(src["tactile_fresh"][:], dtype=bool)
            tactile_calibrated = np.asarray(src["tactile_calibrated"][:], dtype=bool)
            contact_valid = tactile_sum_fresh & tactile_calibrated
            dense_valid = tactile_fresh & tactile_calibrated
            camera_fresh = np.asarray(src["flag_camera_fresh"][:], dtype=bool)

            for name, spec in DATASET_SPECS.items():
                if name in _TACTILE_DATASETS:
                    payload = np.asarray(src[name][:], dtype=np.float32)
                    valid = contact_valid if name == "hand_contact" else dense_valid
                    payload[~valid] = np.nan
                    dst.create_dataset(name, data=payload, compression="gzip")
                elif name == "hand_contact_valid":
                    dst.create_dataset(name, data=contact_valid, compression="gzip")
                elif name == "hand_tactile_force_valid":
                    dst.create_dataset(name, data=dense_valid, compression="gzip")
                elif name == "camera_health":
                    health = np.where(
                        camera_fresh, _CAMERA_HEALTH_OK, _CAMERA_HEALTH_DELIVERY_DELAY
                    )
                    dst.create_dataset(name, data=health, compression="gzip")
                else:
                    src.copy(name, dst)

        for name in ("depth.h5", "rgb.mp4"):
            try:
                os.link(source / name, staging / name)
            except OSError as exc:
                if exc.errno != 18:  # POSIX EXDEV; no unrelated errors are hidden.
                    raise
                shutil.copy2(source / name, staging / name)
        _verify(staging, EPISODE_SCHEMA_VERSION)
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination


def _discover(source: Path) -> list[Path]:
    if (source / "data.h5").is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith("."))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="one v26 episode or a directory of episodes")
    parser.add_argument("destination", type=Path, help="new episode path or destination root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source episodes without writing anything",
    )
    args = parser.parse_args(argv)
    source = args.source.resolve()
    destination = args.destination.absolute()
    if not source.is_dir():
        parser.error(f"source is not a directory: {source}")
    if destination.resolve() == source or source in destination.resolve().parents:
        parser.error("destination must be outside the source")

    episodes = _discover(source)
    failures = 0
    for episode in episodes:
        target = destination if episode == source else destination / episode.name
        try:
            _verify(episode, 26)
            if args.dry_run:
                print(f"OK (dry-run) {episode}")
                continue
            convert_episode(episode, target)
            print(f"OK {episode} -> {target}")
        except (OSError, ValueError, KeyError) as exc:
            failures += 1
            print(f"FAIL {episode}: {exc}")
    print(
        f"Converted {len(episodes) - failures}; failed {failures}; total {len(episodes)}"
    )
    return int(bool(failures) or not episodes)


if __name__ == "__main__":
    raise SystemExit(main())
