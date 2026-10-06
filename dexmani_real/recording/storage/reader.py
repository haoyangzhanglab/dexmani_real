"""Read Raw metadata at open and validate numerical layout only when requested."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.recording.storage.schema import DATASET_SPECS, RAW_FORMAT
from dexmani_real.recording.storage.video import VideoDecoder


class RawDataError(ValueError):
    """Missing or malformed Raw data; distinct from I/O and format incompatibility."""


def _scalar_attr(attrs: h5py.AttributeManager, name: str) -> object:
    if name not in attrs:
        raise RawDataError(f"episode metadata is missing {name}")
    value = np.asarray(attrs[name])
    if value.shape != ():
        raise RawDataError(f"episode metadata {name} must be scalar")
    return value.item()


def _text_attr(attrs: h5py.AttributeManager, name: str) -> str:
    value = _scalar_attr(attrs, name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise RawDataError(f"episode metadata {name} must be a non-empty string")
    return value


def _positive_int_attr(attrs: h5py.AttributeManager, name: str, *, allow_zero: bool = False) -> int:
    value = _scalar_attr(attrs, name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RawDataError(f"episode metadata {name} must be an integer")
    result = int(value)
    if result < 0 or (not allow_zero and result == 0):
        raise RawDataError(f"episode metadata {name} must be positive")
    return result


def _positive_float_attr(attrs: h5py.AttributeManager, name: str) -> float:
    value = _scalar_attr(attrs, name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RawDataError(f"episode metadata {name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise RawDataError(f"episode metadata {name} must be finite and positive")
    return result


class EpisodeReader:
    """Read current immutable Raw evidence without deciding training eligibility."""

    def __init__(self, episode_path: str | Path) -> None:
        self._path = Path(episode_path)
        self._data = self._rgb_decoder = None
        self._fields = None
        self._datasets = {}
        if not self._path.is_dir():
            raise ValueError(f"episode must be a directory: {self._path}")
        self._rgb_path = self._path / "rgb.mp4"
        try:
            self._data = h5py.File(self._path / "data.h5", "r")
            if "meta" not in self._data or not isinstance(self._data["meta"], h5py.Group):
                raise RawDataError("episode is missing meta")
            attrs = self.meta
            if _text_attr(attrs, "format") != RAW_FORMAT:
                raise ValueError(f"unsupported Raw format (expected {RAW_FORMAT!r})")
            _text_attr(attrs, "termination_reason")
            self._num_frames = _positive_int_attr(attrs, "num_frames", allow_zero=True)
            self._control_hz = _positive_float_attr(attrs, "control_hz")
            _text_attr(attrs, "task_label")
            _text_attr(attrs, "collection_source")
        except BaseException:
            self.close()
            raise

    @property
    def meta(self):
        return self._data["meta"].attrs

    @property
    def fields(self) -> frozenset[str]:
        if self._data is None:
            raise RuntimeError("Raw reader is closed")
        if self._fields is None:
            names = {key for key, value in self._data.items() if isinstance(value, h5py.Dataset)}
            if self._rgb_path.is_file():
                names.add("rgb")
            self._fields = frozenset(names)
        return self._fields

    @property
    def path(self):
        return self._path

    def require_fields(self, *names: str) -> None:
        missing = set(names) - self.fields
        if missing:
            raise RawDataError(f"Raw {self._path.name} missing required fields: {sorted(missing)}")

    def require_metadata(self, *names: str) -> None:
        missing = set(names) - self.meta.keys()
        if missing:
            raise RawDataError(
                f"Raw {self._path.name} missing required metadata: {sorted(missing)}"
            )

    def __getitem__(self, name):
        if self._data is None:
            raise RuntimeError("Raw reader is closed")
        if name in self._datasets:
            return self._datasets[name]
        array = self._data[name]
        if name in DATASET_SPECS:
            spec = DATASET_SPECS[name]
            if array.shape != (self.num_frames, *spec.tail_shape) or array.dtype != spec.dtype:
                raise RawDataError(f"Raw {name}: incompatible shape/dtype")
        self._datasets[name] = array
        return array

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def dt(self) -> float:
        return 1.0 / self._control_hz

    def read_row_info(self, name, start, end):
        """Old Raw has unknown time/status; never infer measured time or success."""
        from dexmani_real.recording.storage.schema import ROW_INFO_SPECS

        if name in self.fields:
            return np.asarray(self[name][start:end])
        if name == "dispatch_status" and name in self.meta:
            statuses = np.asarray(self.meta[name], dtype=np.uint8)
            if statuses.shape != (self.num_frames, 2):
                raise RawDataError("invalid historical dispatch status shape")
            return statuses[start:end]
        spec = ROW_INFO_SPECS[name]
        return np.full(
            (end - start, *spec.tail_shape), 4 if name == "dispatch_status" else 0, dtype=spec.dtype
        )

    def _decoder(self):
        self.require_fields("rgb")
        if self._rgb_decoder is None:
            self._rgb_decoder = VideoDecoder(self._rgb_path)
        return self._rgb_decoder

    def read_camera_frame(self, key: str, index: int) -> np.ndarray:
        self.require_fields(key)
        return self._decoder().read_frame(index) if key == "rgb" else np.asarray(self[key][index])

    def iter_camera_frames(self, key: str) -> Iterator[np.ndarray]:
        if key != "rgb":
            raise ValueError("iter_camera_frames supports only MP4-backed RGB")
        yield from self._decoder().iter_frames()

    def close(self) -> None:
        self._datasets.clear()
        self._fields = None
        errors = []
        for name in ("_rgb_decoder", "_data"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(exc)
                else:
                    setattr(self, name, None)
        if errors:
            raise RuntimeError(f"Raw reader close failures: {errors}") from errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
