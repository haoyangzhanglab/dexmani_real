"""Offline, one-time copy of a v29 raw episode to v30; never modify the source.

Usage: python examples/convert_raw_v29.py OLD_EPISODE NEW_EPISODE
Only redundant runtime bookkeeping is removed. Images, timestamps, actions,
robot state, tactile validity, calibration and experiment metadata are retained.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

_REMOVED = ("arm_last_cmd_seq", "source_sample_index", "fill_reason", "flag_sample_valid")


def convert_episode(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source):
        raise ValueError("output must be outside the source episode")
    with h5py.File(source / "data.h5", "r") as data:
        if int(data["meta"].attrs["schema_version"]) != 29:
            raise ValueError("this one-time converter accepts only raw v29")
        count = int(data["meta"].attrs["num_frames"])
        if not np.array_equal(data["source_sample_index"][:], np.arange(count)):
            raise ValueError("v29 source rows are not sequential")
        if np.any(data["fill_reason"][:] != 0) or not np.all(data["flag_sample_valid"][:]):
            raise ValueError("v29 episode contains non-source or invalid rows")
        for name in _REMOVED:
            if data[name].shape != (count,):
                raise ValueError(f"invalid v29 column: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".convert-v29-", dir=destination.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        with h5py.File(staging / "data.h5", "r+") as data:
            for name in _REMOVED:
                del data[name]
            data["meta"].attrs["schema_version"] = 30
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert_episode(args.source, args.destination)
    print(args.destination)
