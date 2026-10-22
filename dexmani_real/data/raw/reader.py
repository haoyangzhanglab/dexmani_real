"""Read Raw metadata at open and validate numerical layout only when requested."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import av
import h5py
import numpy as np

from dexmani_real.data.raw.schema import (
    DATASET_SPECS,
    RAW_FORMAT,
    TIMING_ROW_DTYPE,
    validate_arm_execution,
)


class RawDataError(ValueError):
    """Malformed or unsupported Raw input; distinct from program and system errors."""


def scalar_attr(attrs: h5py.AttributeManager, name: str) -> object:
    if name not in attrs:
        raise RawDataError(f"episode metadata is missing {name}")
    value = np.asarray(attrs[name])
    if value.shape != ():
        raise RawDataError(f"episode metadata {name} must be scalar")
    return value.item()


def text_attr(attrs: h5py.AttributeManager, name: str) -> str:
    value = scalar_attr(attrs, name)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RawDataError(f"episode metadata {name}: invalid UTF-8") from exc
    if not isinstance(value, str) or not value.strip():
        raise RawDataError(f"episode metadata {name} must be a non-empty string")
    try:
        # h5py may expose malformed variable-length bytes as surrogate escapes.
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RawDataError(f"episode metadata {name}: invalid UTF-8") from exc
    return value


def nonnegative_int_attr(attrs: h5py.AttributeManager, name: str) -> int:
    value = scalar_attr(attrs, name)
    if type(value) is not int:
        raise RawDataError(f"episode metadata {name} must be an integer")
    if value < 0:
        raise RawDataError(f"episode metadata {name} must be nonnegative")
    return value


def positive_int_attr(attrs: h5py.AttributeManager, name: str) -> int:
    """Parse a Raw dimension without coercing text/bools or truncating fractional values."""
    value = scalar_attr(attrs, name)
    if (
        type(value) not in (int, float)
        or value <= 0
        or (type(value) is float and (not np.isfinite(value) or not value.is_integer()))
    ):
        raise RawDataError(f"episode metadata {name} must be a positive integer")
    return int(value)


def positive_float_attr(attrs: h5py.AttributeManager, name: str) -> float:
    value = scalar_attr(attrs, name)
    if type(value) not in (int, float):
        raise RawDataError(f"episode metadata {name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise RawDataError(f"episode metadata {name} must be finite and positive")
    return result


class EpisodeReader:
    """Read current immutable Raw evidence without deciding training eligibility."""

    def __init__(self, episode_path: str | Path) -> None:
        self.episode_path = Path(episode_path)
        self.data_file = self.video_container = None
        self.fields_cache = None
        self.dataset_cache = {}
        if not self.episode_path.is_dir():
            raise ValueError(f"episode must be a directory: {self.episode_path}")
        self.rgb_path = self.episode_path / "rgb.mp4"
        try:
            self.data_file = h5py.File(self.episode_path / "data.h5", "r")
            if "meta" not in self.data_file or not isinstance(self.data_file["meta"], h5py.Group):
                raise RawDataError("episode is missing meta")
            attrs = self.meta
            if text_attr(attrs, "format") != RAW_FORMAT:
                raise RawDataError(f"unsupported Raw format (expected {RAW_FORMAT!r})")
            if (
                attrs.get("arm_control_mode") != 1
                or text_attr(attrs, "arm_action_semantics") != "q_goal_submitted"
            ):
                raise RawDataError("unknown arm control/action semantics")
            if text_attr(attrs, "timestamp_source") != "host_monotonic_relative":
                raise RawDataError("Raw timestamp must use host monotonic relative seconds")
            text_attr(attrs, "termination_reason")
            self.frame_count = nonnegative_int_attr(attrs, "num_frames")
            self.sampling_hz = positive_float_attr(attrs, "control_hz")
            text_attr(attrs, "task_label")
            text_attr(attrs, "collection_source")
        except BaseException:
            self.close()
            raise

    @property
    def meta(self):
        return self.data_file["meta"].attrs

    @property
    def fields(self) -> frozenset[str]:
        if self.data_file is None:
            raise RuntimeError("Raw reader is closed")
        if self.fields_cache is None:
            names = {key for key, value in self.data_file.items() if isinstance(value, h5py.Dataset)}
            if self.rgb_path.is_file():
                names.add("rgb")
            self.fields_cache = frozenset(names)
        return self.fields_cache

    @property
    def path(self):
        return self.episode_path

    def require_fields(self, *names: str) -> None:
        missing = set(names) - self.fields
        if missing:
            raise RawDataError(f"Raw {self.episode_path.name} missing required fields: {sorted(missing)}")

    def __getitem__(self, name):
        if self.data_file is None:
            raise RuntimeError("Raw reader is closed")
        if name in self.dataset_cache:
            return self.dataset_cache[name]
        array = self.data_file[name]
        if not isinstance(array, h5py.Dataset):
            raise RawDataError(f"Raw {name}: expected a dataset")
        if name in DATASET_SPECS:
            spec = DATASET_SPECS[name]
            if array.shape != (self.num_frames, *spec.tail_shape) or array.dtype != spec.dtype:
                raise RawDataError(f"Raw {name}: incompatible shape/dtype")
        self.dataset_cache[name] = array
        return array

    def read_timing_rows(self, fields=("run_id", "command_seq")):
        """Read the consumer's required columns without imposing diagnostic layout."""
        if "timing/rows" not in self.data_file:
            raise RawDataError("Raw is missing timing rows")
        rows = self.data_file["timing/rows"]
        if (
            not isinstance(rows, h5py.Dataset)
            or rows.shape != (self.num_frames,)
            or rows.dtype.names is None
        ):
            raise RawDataError("timing rows must be a compound dataset aligned with core rows")
        for name in fields:
            if name not in rows.dtype.names or rows.dtype[name] != TIMING_ROW_DTYPE[name]:
                raise RawDataError(f"timing rows require int64 column {name}")
        return rows.fields(fields)[:]

    def validate_arm_execution(self):
        """Check execution evidence independently of inference diagnostics."""
        try:
            group = self.data_file["arm_execution"]
            if not isinstance(group, h5py.Group):
                raise RawDataError("arm_execution must be a group")
            for name in ("start", "end"):
                value = group[name]
                if not isinstance(value, h5py.Dataset) or value.shape != ():
                    raise RawDataError(f"{name} must be a scalar dataset")
            validate_arm_execution(
                json.loads(group["start"].asstr()[()]),
                json.loads(group["end"].asstr()[()]),
                self.read_timing_rows(),
                self["dispatch_status"][:],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise RawDataError(f"Mode-1 execution evidence: {exc}") from exc

    @property
    def num_frames(self) -> int:
        return self.frame_count

    @property
    def control_hz(self) -> float:
        return self.sampling_hz

    @property
    def dt(self) -> float:
        return 1.0 / self.sampling_hz

    def iter_rgb_frames(self) -> Iterator[np.ndarray]:
        """Decode RGB in HDF5 row order; sampling times come from Raw ``timestamp``."""
        self.require_fields("rgb")
        try:
            if self.video_container is None:
                self.video_container = av.open(str(self.rgb_path), "r")
            self.video_container.seek(0)
            # The reader owns the container if the consumer stops iteration early.
            for frame in self.video_container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")
        except (av.error.InvalidDataError, av.error.EOFError) as exc:
            raise RawDataError(f"{self.rgb_path}: invalid RGB video: {exc}") from exc

    def close(self) -> None:
        self.dataset_cache.clear()
        self.fields_cache = None
        errors = []
        for name in ("video_container", "data_file"):
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
