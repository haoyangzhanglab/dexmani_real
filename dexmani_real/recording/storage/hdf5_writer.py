"""Single owner of one episode's HDF5 handle, datasets, and append offset."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from dexmani_real.recording.storage.schema import DATASET_SPECS, validate_data_layout


class EpisodeDataWriter:
    """Own and append controller source rows to one lazy HDF5 transaction."""

    def __init__(
        self,
        path: str | Path,
        *,
        write_initial_meta: Callable[[h5py.Group], None],
    ) -> None:
        self.path = Path(path)
        self._write_initial_meta = write_initial_meta
        self._file: h5py.File | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._flushed_frames = 0

    @property
    def datasets(self) -> Mapping[str, h5py.Dataset]:
        return self._datasets

    @property
    def flushed_frames(self) -> int:
        return self._flushed_frames

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "w")
            self._write_initial_meta(self._file.create_group("meta"))
            for name, spec in DATASET_SPECS.items():
                self._datasets[name] = self._file.create_dataset(
                    name, shape=(0,) + spec.tail_shape,
                    maxshape=(None,) + spec.tail_shape, dtype=spec.dtype,
                    compression="gzip",
                )
        return self._file

    def append(self, data: Mapping[str, np.ndarray]) -> None:
        """Append exactly this batch of newly emitted rows."""
        count = len(data["timestamp"])
        errors = validate_data_layout(
            {name: values.shape for name, values in data.items()},
            {name: values.dtype for name, values in data.items()},
            frame_count=count,
        )
        if errors:
            raise RuntimeError("episode row batch mismatch: " + "; ".join(errors))
        self._ensure_open()
        end = self._flushed_frames + count
        for name, values in data.items():
            dataset = self._datasets[name]
            dataset.resize(end, axis=0)
            dataset[self._flushed_frames:end] = values
        self._flushed_frames = end

    def update_meta(self, write_meta: Callable[[h5py.Group], None]) -> None:
        """Apply final transaction metadata while retaining handle ownership."""
        data_h5 = self._ensure_open()
        write_meta(data_h5["meta"])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
