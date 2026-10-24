"""Read Raw metadata at open and validate numerical layout only when requested."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import h5py
import numpy as np

from dexmani_real.data.raw.schema import (
    ARM_ACTION_SEMANTICS,
    DATASET_SPECS,
    RAW_FORMAT,
    TIMING_ROW_DTYPE,
    validate_fixed_rate_rows,
)


class RawDataError(ValueError):
    """Raw is unsuitable for the requested use; distinct from program and system errors."""


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
    """Require positive integer dimensions without numeric or text coercion."""
    value = scalar_attr(attrs, name)
    if type(value) is not int or value <= 0:
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
    """Read immutable Raw; consumers explicitly request their needed validation."""

    def __init__(self, episode_path: str | Path) -> None:
        self.path = Path(episode_path)
        self.data_file = self.video_container = None
        self.fields_cache = None
        self.dataset_cache = {}
        if not self.path.is_dir():
            raise ValueError(f"episode must be a directory: {self.path}")
        self.rgb_path = self.path / "rgb.mp4"
        try:
            try:
                self.data_file = h5py.File(self.path / "data.h5", "r")
            except OSError as exc:
                # h5py reports format failures without a filesystem errno. Keep
                # permission/device/filesystem failures distinct from bad episodes.
                if exc.errno is None and any(
                    message in str(exc).lower()
                    for message in (
                        "file signature not found", "truncated file", "bad object header",
                        "bad superblock", "incorrect metadata checksum",
                    )
                ):
                    raise RawDataError(f"invalid Raw HDF5: {exc}") from exc
                raise
            if "meta" not in self.data_file or not isinstance(self.data_file["meta"], h5py.Group):
                raise RawDataError("episode is missing meta")
            attrs = self.meta
            if text_attr(attrs, "format") != RAW_FORMAT:
                raise RawDataError(f"unsupported Raw format (expected {RAW_FORMAT!r})")
            if text_attr(attrs, "arm_action_semantics") != ARM_ACTION_SEMANTICS:
                raise RawDataError("unknown arm action semantics")
            if text_attr(attrs, "timestamp_source") != "host_monotonic_relative":
                raise RawDataError("Raw timestamp must use host monotonic relative seconds")
            text_attr(attrs, "termination_reason")
            self.num_frames = nonnegative_int_attr(attrs, "num_frames")
            self.control_hz = positive_float_attr(attrs, "control_hz")
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

    def require_fields(self, *names: str) -> None:
        missing = set(names) - self.fields
        if missing:
            raise RawDataError(f"Raw {self.path.name} missing required fields: {sorted(missing)}")

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

    def read_timing_rows(self, fields=TIMING_ROW_DTYPE.names, *, start=0, stop=None):
        """Read required source-clock columns in a bounded row range."""
        if "timing/rows" not in self.data_file:
            raise RawDataError("Raw is missing timing rows")
        rows = self.data_file["timing/rows"]
        if (
            not isinstance(rows, h5py.Dataset)
            or rows.shape != (self.num_frames,)
            or rows.dtype != TIMING_ROW_DTYPE
        ):
            raise RawDataError("timing rows must match the current eight-column layout and row count")
        return rows.fields(fields)[start:stop]

    def validate_completion(self):
        """Strict numerical consumers require a technically normal capture."""
        if text_attr(self.meta, "technical_status") != "normal":
            raise RawDataError("capture ended with a technical failure")

    def validate_rows(
        self, *, required_tactile=("aggregate", "dense"), require_camera=True, chunk_frames=4096
    ):
        """Check sampling and source ages in blocks with one overlap row."""
        if type(chunk_frames) is not int or chunk_frames <= 0:
            raise ValueError("chunk_frames must be a positive integer")
        if not set(required_tactile) <= {"aggregate", "dense"}:
            raise ValueError("unknown tactile source")
        fields = ("observation_ns", "control_ns", "arm_ns", "hand_ns")
        fields += tuple(f"tactile_{name}_ns" for name in required_tactile)
        if require_camera:
            fields += ("camera_ns", "camera_sequence")
        max_age_ns = (
            positive_float_attr(self.meta, "hand_tactile_hold_max_age_s") * 1e9
            if required_tactile else None
        )
        timestamps = self["timestamp"]
        first_observation = None
        for start in range(0, self.num_frames, chunk_frames):
            first, end = max(0, start - 1), min(start + chunk_frames, self.num_frames)
            timestamp = timestamps[first:end]
            errors = validate_fixed_rate_rows(
                timestamp, self.control_hz, row_offset=first
            )
            if errors:
                raise RawDataError("; ".join(errors))
            rows = self.read_timing_rows(fields, start=first, stop=end)
            observation = rows["observation_ns"]
            if np.any(observation <= 0) or np.any(np.diff(observation) <= 0):
                raise RawDataError("observation clocks must be positive and strictly increasing")
            if first_observation is None:
                first_observation = int(observation[0])
            for name in fields[1:]:
                values = rows[name]
                if name == "camera_sequence":
                    if np.any(values < 0) or np.any(np.diff(values) < 0):
                        raise RawDataError("camera sequence must be nonnegative and nondecreasing")
                    continue
                if np.any(values <= 0) or np.any(values > observation):
                    raise RawDataError(f"invalid consumed source clock: {name}")
                if np.any(np.diff(values) < 0):
                    raise RawDataError(f"consumed source clock regressed: {name}")
            if not np.allclose(
                timestamp * 1e9, observation - first_observation, rtol=0, atol=1.0
            ):
                raise RawDataError("timestamp differs from actual observation clock")
            for name in required_tactile:
                if np.any(observation - rows[f"tactile_{name}_ns"] >= max_age_ns):
                    raise RawDataError(f"consumed tactile {name} exceeds hold age budget")

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

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
