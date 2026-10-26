"""打开时校验 Raw 元数据，按消费需求校验数组和触觉语义。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import av
import h5py
import numpy as np

from dexmani_real.data.raw.schema import (
    ARM_ACTION_SEMANTICS,
    DATASET_SPECS,
    TACTILE_AXES,
    TACTILE_FINGER_ORDER,
    TACTILE_PROCESSING,
    TACTILE_UNITS,
    TIMING_ROW_DTYPE,
    validate_fixed_rate_rows,
)
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES


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


def positive_float_attr(attrs: h5py.AttributeManager, name: str) -> float:
    value = scalar_attr(attrs, name)
    if type(value) is not int and type(value) is not float:
        raise RawDataError(f"episode metadata {name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise RawDataError(f"episode metadata {name} must be finite and positive")
    return result


class EpisodeReader:
    """Read immutable Raw; consumers explicitly request their needed validation."""

    def __init__(self, episode_path: str | Path) -> None:
        self.path = Path(episode_path)
        self.data_file: h5py.File | None = None
        self.video_container: av.container.InputContainer | None = None
        self.fields_cache: frozenset[str] | None = None
        self.dataset_cache: dict[str, h5py.Dataset] = {}
        self.tactile_hold_max_age_s: float | None = None
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
            if text_attr(attrs, "arm_action_semantics") != ARM_ACTION_SEMANTICS:
                raise RawDataError("unknown arm action semantics")
            if text_attr(attrs, "timestamp_source") != "host_monotonic_relative":
                raise RawDataError("Raw timestamp must use host monotonic relative seconds")
            if text_attr(attrs, "joint_units") != "rad":
                raise RawDataError("Raw joint units must be rad")
            for name, expected in (
                ("arm_joint_names", XARM7_JOINT_NAMES),
                ("hand_joint_names", XHAND_SDK_JOINT_NAMES),
            ):
                try:
                    names = json.loads(text_attr(attrs, name))
                except json.JSONDecodeError as exc:
                    raise RawDataError(f"Raw {name}: invalid JSON") from exc
                if names != list(expected):
                    raise RawDataError(f"Raw {name}: incompatible joint order")
            text_attr(attrs, "termination_reason")
            self.num_frames = nonnegative_int_attr(attrs, "num_frames")
            self.control_hz = positive_float_attr(attrs, "control_hz")
            text_attr(attrs, "task_label")
            text_attr(attrs, "collection_source")
        except BaseException:
            self.close()
            raise

    @property
    def meta(self) -> h5py.AttributeManager:
        if self.data_file is None:
            raise RuntimeError("Raw reader is closed")
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

    def __getitem__(self, name: str) -> h5py.Dataset:
        if self.data_file is None:
            raise RuntimeError("Raw reader is closed")
        if name in self.dataset_cache:
            return self.dataset_cache[name]
        if name not in self.data_file:
            raise RawDataError(f"Raw is missing dataset: {name}")
        if name in {"hand_contact", "hand_tactile_force"}:
            self.validate_tactile_metadata()
        array = self.data_file[name]
        if not isinstance(array, h5py.Dataset):
            raise RawDataError(f"Raw {name}: expected a dataset")
        spec = ((), TIMING_ROW_DTYPE) if name == "timing/rows" else DATASET_SPECS.get(name)
        if spec is not None:
            tail_shape, dtype = spec
            if array.shape != (self.num_frames, *tail_shape) or array.dtype != dtype:
                raise RawDataError(f"Raw {name}: incompatible shape/dtype")
        self.dataset_cache[name] = array
        return array

    def validate_completion(self) -> None:
        """Strict numerical consumers require a technically normal capture."""
        if text_attr(self.meta, "technical_status") != "normal":
            raise RawDataError("capture ended with a technical failure")

    def validate_tactile_metadata(self) -> float:
        """仅在消费触觉时确认文件语义；Raw 打开期间保持不可变。"""
        if self.tactile_hold_max_age_s is None:
            for name, expected in (
                ("hand_tactile_processing", TACTILE_PROCESSING),
                ("hand_tactile_units", TACTILE_UNITS),
                ("hand_tactile_axes", TACTILE_AXES),
                ("hand_tactile_finger_order", TACTILE_FINGER_ORDER),
            ):
                if text_attr(self.meta, name) != expected:
                    raise RawDataError(f"Raw {name}: incompatible tactile semantics")
            self.tactile_hold_max_age_s = positive_float_attr(self.meta, "hand_tactile_hold_max_age_s")
        return self.tactile_hold_max_age_s

    def read_timing_rows(self, fields: Sequence[str]) -> np.ndarray:
        """读取指定来源时钟字段，保留原值；不隐式执行训练准入校验。"""
        if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
            raise TypeError("timing fields must be a sequence of field names")
        fields = tuple(fields)
        if any(not isinstance(name, str) for name in fields):
            raise TypeError("timing field names must be strings")
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("timing fields must be nonempty and unique")
        unknown = set(fields).difference(cast(tuple[str, ...], TIMING_ROW_DTYPE.names))
        if unknown:
            raise ValueError(f"unknown timing fields: {sorted(unknown)}")
        return self["timing/rows"].fields(fields)[:]

    def validate_rows(
        self, *, required_tactile=("aggregate", "dense"), require_camera=True, chunk_frames=4096
    ):
        """Check sampling and source ages in blocks with one overlap row."""
        if type(chunk_frames) is not int or chunk_frames <= 0:
            raise ValueError("chunk_frames must be a positive integer")
        if not set(required_tactile) <= {"aggregate", "dense"}:
            raise ValueError("unknown tactile source")
        timing = self["timing/rows"]
        fields = ("observation_ns", "control_ns", "arm_ns", "hand_ns")
        fields += tuple(f"tactile_{name}_ns" for name in required_tactile)
        if require_camera:
            fields += ("camera_ns", "camera_sequence")
        max_age_ns = (
            self.validate_tactile_metadata() * 1e9
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
            rows = timing.fields(fields)[first:end]
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
        self.tactile_hold_max_age_s = None
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

    def __enter__(self) -> EpisodeReader:
        return self

    def __exit__(self, *args):
        self.close()
