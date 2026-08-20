"""Single owner of one episode's data.h5 handle, datasets, and append offset."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from dexmani_real.recording.episode_schema import validate_data_layout_v17


class EpisodeDataWriter:
    """Own and append aligned control rows to one lazy HDF5 transaction."""

    def __init__(
        self,
        path: str | Path,
        *,
        arm_sent_stream: bool,
        write_initial_meta: Callable[[h5py.Group], None],
    ) -> None:
        self.path = Path(path)
        self.arm_sent_stream = bool(arm_sent_stream)
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
        return self._file

    def append(self, data: Mapping[str, np.ndarray], timestamps: np.ndarray) -> None:
        """Validate and append the unflushed prefix of one aligned buffer."""
        frame_count = int(timestamps.shape[0])
        if frame_count == self._flushed_frames:
            return
        shapes = {name: tuple(values.shape) for name, values in data.items()}
        dtypes = {name: values.dtype for name, values in data.items()}
        shapes["timestamp"] = tuple(timestamps.shape)
        dtypes["timestamp"] = timestamps.dtype
        errors = validate_data_layout_v17(
            shapes,
            dtypes,
            frame_count=frame_count,
            arm_sent_stream=self.arm_sent_stream,
        )
        if errors:
            raise RuntimeError("episode recorder buffer mismatch: " + "; ".join(errors))

        data_h5 = self._ensure_open()
        new_start = self._flushed_frames
        for name, values in data.items():
            if name not in self._datasets:
                self._datasets[name] = data_h5.create_dataset(
                    name,
                    data=values[:frame_count].copy(),
                    maxshape=(None,) + values.shape[1:],
                    dtype=values.dtype,
                    compression="gzip",
                )
            else:
                dataset = self._datasets[name]
                dataset.resize(frame_count, axis=0)
                dataset[new_start:frame_count] = values[new_start:frame_count]
        if "timestamp" not in self._datasets:
            self._datasets["timestamp"] = data_h5.create_dataset(
                "timestamp",
                data=timestamps[:frame_count].copy(),
                maxshape=(None,),
                dtype=np.float64,
                compression="gzip",
            )
        else:
            timestamp_dataset = self._datasets["timestamp"]
            timestamp_dataset.resize(frame_count, axis=0)
            timestamp_dataset[new_start:frame_count] = timestamps[new_start:frame_count]
        self._flushed_frames = frame_count

    def update_meta(self, write_meta: Callable[[h5py.Group], None]) -> None:
        """Apply final transaction metadata while retaining handle ownership."""
        data_h5 = self._ensure_open()
        write_meta(data_h5["meta"])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
