"""EpisodeReader — lazy-load HDF5 episode data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from dexmani_real.recording.quality_flags import ALL_GOOD_MASK


class EpisodeReader:
    """Lazy-loading HDF5 episode reader.

    Does not load the entire file into memory — datasets are read on demand.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Episode file not found: {path}")
        self._file: h5py.File | None = None

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(str(self.path), "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def read(self, key: str) -> np.ndarray:
        """Read any dataset by path, e.g. read("obs/arm_qpos")."""
        f = self._open()
        return np.asarray(f[key])

    def iter_frames(self, skip_rejected: bool = True) -> Iterator[dict[str, Any]]:
        """Iterate frame by frame. skip_rejected=True skips frames where
        not all quality flags are good."""
        f = self._open()
        quality_flags = np.asarray(f["quality_flags"], dtype=np.uint16).ravel()
        n = int(quality_flags.shape[0])

        for i in range(n):
            if skip_rejected:
                if (quality_flags[i] & ALL_GOOD_MASK) != ALL_GOOD_MASK:
                    continue
            yield self._read_frame(i)

    def get_valid_mask(self) -> np.ndarray:
        """Return (T,) bool array marking fully-valid frames."""
        f = self._open()
        qf = np.asarray(f["quality_flags"], dtype=np.uint16).ravel()
        return (qf & np.uint16(ALL_GOOD_MASK)) == np.uint16(ALL_GOOD_MASK)

    @property
    def num_frames(self) -> int:
        f = self._open()
        return int(np.asarray(f["quality_flags"]).shape[0])

    @property
    def num_valid_frames(self) -> int:
        return int(self.get_valid_mask().sum())

    @property
    def metadata(self) -> dict:
        f = self._open()
        meta = f["meta"]
        return {k: meta.attrs[k] for k in meta.attrs}

    def _read_frame(self, idx: int) -> dict[str, Any]:
        f = self._open()
        return {
            "arm_qpos": np.asarray(f["obs/arm_qpos"][idx]),
            "arm_qvel": np.asarray(f["obs/arm_qvel"][idx]),
            "arm_tau": np.asarray(f["obs/arm_tau"][idx]),
            "eef_pos": np.asarray(f["obs/eef_pos"][idx]),
            "eef_quat": np.asarray(f["obs/eef_quat"][idx]),
            "hand_qpos": np.asarray(f["obs/hand_qpos"][idx]),
            "hand_current": np.asarray(f["obs/hand_current"][idx]),
            "hand_tactile_sum": np.asarray(f["obs/hand_tactile_sum"][idx]),
            "hand_temperature": np.asarray(f["obs/hand_temperature"][idx]),
            "action_arm_qpos": np.asarray(f["action/arm_qpos"][idx]),
            "action_hand_qpos": np.asarray(f["action/hand_qpos"][idx]),
            "wrist_pos": np.asarray(f["vr/wrist_pos"][idx]),
            "wrist_quat": np.asarray(f["vr/wrist_quat"][idx]),
            "landmarks": np.asarray(f["vr/landmarks"][idx]),
            "quality_flags": int(np.asarray(f["quality_flags"][idx])),
            "index": idx,
        }

    def __enter__(self) -> "EpisodeReader":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self.num_frames

    def __repr__(self) -> str:
        return (
            f"EpisodeReader({self.path.name}, "
            f"frames={self.num_frames}, valid={self.num_valid_frames})"
        )
