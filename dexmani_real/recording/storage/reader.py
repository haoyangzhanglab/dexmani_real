"""Capability-based reader for the current Raw format.

Known present fields are validated; additive unknown fields do not invalidate an
otherwise usable episode. Legacy Raw belongs to the explicit offline migration
tool, not the normal runtime/export/replay path.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.recording.storage.schema import DATASET_SPECS, RAW_FORMAT
from dexmani_real.recording.storage.video import VideoDecoder
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, validate_aligned_depth_distortion


def _scalar_attr(attrs: h5py.AttributeManager, name: str) -> object:
    if name not in attrs:
        raise ValueError(f"episode metadata is missing {name}")
    value = np.asarray(attrs[name])
    if value.shape != ():
        raise ValueError(f"episode metadata {name} must be scalar")
    return value.item()


def _text_attr(attrs: h5py.AttributeManager, name: str) -> str:
    value = _scalar_attr(attrs, name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"episode metadata {name} must be a non-empty string")
    return value


def _positive_int_attr(attrs: h5py.AttributeManager, name: str, *, allow_zero: bool = False) -> int:
    value = _scalar_attr(attrs, name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"episode metadata {name} must be an integer")
    result = int(value)
    if result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"episode metadata {name} must be positive")
    return result


def _positive_float_attr(attrs: h5py.AttributeManager, name: str) -> float:
    value = _scalar_attr(attrs, name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"episode metadata {name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"episode metadata {name} must be finite and positive")
    return result


def _flat_float_attr(attrs: h5py.AttributeManager, name: str, size: int) -> np.ndarray:
    if name not in attrs:
        raise ValueError(f"episode metadata is missing {name}")
    try:
        value = np.asarray(attrs[name], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"episode metadata {name} must be numeric") from exc
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"episode metadata {name} must have shape ({size},) and finite values")
    return value


def _validate_rigid_transform(value: np.ndarray, *, label: str) -> None:
    transform = value.reshape(4, 4)
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError(f"episode metadata {label} must be a finite rigid homogeneous transform")


def _validate_color_camera_metadata(attrs: h5py.AttributeManager) -> None:
    if _text_attr(attrs, "camera_payload_mode") != "depth_to_color_aligned_rgbd":
        raise ValueError("episode camera payload must be depth_to_color_aligned_rgbd")
    width = _positive_int_attr(attrs, "camera_color_width")
    height = _positive_int_attr(attrs, "camera_color_height")
    matrix = _flat_float_attr(attrs, "camera_color_intrinsics", 9).reshape(3, 3)
    coefficients = _flat_float_attr(attrs, "camera_color_distortion_coeffs", 5)
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        ppx=float(matrix[0, 2]),
        ppy=float(matrix[1, 2]),
        distortion_model=_text_attr(attrs, "camera_color_distortion_model"),
        distortion_coeffs=tuple(float(value) for value in coefficients),
    )
    validate_aligned_depth_distortion(intrinsics.distortion_model)
    if not np.allclose(matrix, intrinsics.matrix(), atol=1e-9, rtol=0.0):
        raise ValueError(
            "episode metadata camera_color_intrinsics is not a canonical pinhole matrix"
        )
    _validate_rigid_transform(
        _flat_float_attr(attrs, "camera_T_xarm_base_from_color", 16),
        label="camera_T_xarm_base_from_color",
    )
    _positive_float_attr(attrs, "depth_scale")


def _validate_hand_mount_metadata(attrs: h5py.AttributeManager) -> None:
    _flat_float_attr(attrs, "handbase_position_eef_m", 3)
    quaternion = _flat_float_attr(attrs, "handbase_quat_eef_wxyz", 4)
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("episode metadata handbase_quat_eef_wxyz must be unit length within 1e-6")


class EpisodeReader:
    """Read current immutable Raw evidence without deciding training eligibility."""

    def __init__(self, episode_path: str | Path) -> None:
        self._path = Path(episode_path)
        self._data = self._rgb_decoder = None
        self._cache = {}
        if not self._path.is_dir():
            raise ValueError(f"episode must be a directory: {self._path}")
        self._rgb_path = self._path / "rgb.mp4"
        try:
            self._data = h5py.File(self._path / "data.h5", "r")
            if "meta" not in self._data or not isinstance(self._data["meta"], h5py.Group):
                raise ValueError("episode is missing meta")
            attrs = self.meta
            if _text_attr(attrs, "format") != RAW_FORMAT:
                raise ValueError(
                    f"unsupported Raw format; migrate the episode before use (expected {RAW_FORMAT!r})"
                )
            _text_attr(attrs, "termination_reason")
            self._num_frames = _positive_int_attr(attrs, "num_frames", allow_zero=True)
            self._control_hz = _positive_float_attr(attrs, "control_hz")
            _text_attr(attrs, "task_label")
            _text_attr(attrs, "collection_source")
            self._validate_present_fields()
        except BaseException:
            self.close()
            raise

    @property
    def meta(self):
        return self._data["meta"].attrs

    @property
    def fields(self) -> frozenset[str]:
        names = {key for key, value in self._data.items() if isinstance(value, h5py.Dataset)}
        if self._rgb_path.is_file():
            names.add("rgb")
        return frozenset(names)

    def require_fields(self, *names: str) -> None:
        missing = set(names) - self.fields
        if missing:
            raise ValueError(f"Raw {self._path.name} missing required fields: {sorted(missing)}")

    def require_metadata(self, *names: str) -> None:
        missing = set(names) - self.meta.keys()
        if missing:
            raise ValueError(f"Raw {self._path.name} missing required metadata: {sorted(missing)}")

    def __getitem__(self, name):
        return self._data[name]

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def dt(self) -> float:
        return 1.0 / self._control_hz

    def _validate_present_fields(self):
        for name, spec in DATASET_SPECS.items():
            if name in self._data:
                array = self._data[name]
                if (
                    not isinstance(array, h5py.Dataset)
                    or array.shape != (self.num_frames, *spec.tail_shape)
                    or array.dtype != spec.dtype
                ):
                    raise ValueError(f"Raw {name}: incompatible shape/dtype")
        fields = self.fields
        if fields & {"rgb", "depth"} or any(k.startswith("camera_") for k in self.meta):
            _validate_color_camera_metadata(self.meta)
        if "depth" in self._data:
            depth = self._data["depth"]
            expected = (
                self.num_frames,
                int(self.meta["camera_color_height"]),
                int(self.meta["camera_color_width"]),
            )
            if (
                not isinstance(depth, h5py.Dataset)
                or depth.shape != expected
                or depth.dtype != np.uint16
            ):
                raise ValueError(f"Raw depth must be uint16 with shape {expected}")
        if "rgb" in fields and self._rgb_path.stat().st_size == 0:
            raise ValueError("Raw RGB file is empty")
        if any(k in self.meta for k in ("handbase_position_eef_m", "handbase_quat_eef_wxyz")):
            _validate_hand_mount_metadata(self.meta)

    def _decoder(self):
        self.require_fields("rgb")
        if self._rgb_decoder is None:
            self._rgb_decoder = VideoDecoder(self._rgb_path)
        return self._rgb_decoder

    def read_camera_frame(self, key: str, index: int) -> np.ndarray:
        self.require_fields(key)
        return self._decoder().read_frame(index) if key == "rgb" else np.asarray(self[key][index])

    def read_camera_all(self, key: str) -> np.ndarray:
        self.require_fields(key)
        if key not in self._cache:
            data = self._decoder().read_all() if key == "rgb" else np.asarray(self[key][:])
            if len(data) != self.num_frames:
                raise ValueError(f"{key} length differs from num_frames")
            self._cache[key] = data
        return self._cache[key]

    def iter_camera_frames(self, key: str) -> Iterator[np.ndarray]:
        if key != "rgb":
            raise ValueError("iter_camera_frames supports only MP4-backed RGB")
        yield from self._decoder().iter_frames()

    def close(self) -> None:
        self._cache.clear()
        for name in ("_rgb_decoder", "_data"):
            resource = getattr(self, name)
            if resource is not None:
                resource.close()
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
