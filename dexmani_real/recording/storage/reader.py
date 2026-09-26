"""Read published Raw v34 episodes.

The reader validates the on-disk contract without deciding whether an episode
is eligible for training. Failed control rows and nonfinite research payloads
remain readable for audit and whole-episode export rejection.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.recording.storage.schema import (
    DATASET_SPECS,
    EPISODE_SCHEMA_VERSION,
    validate_data_layout,
)
from dexmani_real.recording.storage.video import VideoDecoder
from dexmani_real.sensor.camera.geometry import CameraIntrinsics


class MergedH5File:
    """Transparent merged view of ``data.h5`` and camera HDF5 sidecars."""

    __slots__ = ("_data", "_sidecars")

    def __init__(self, data_h5f: h5py.File, sidecars: dict[str, h5py.File] | None = None) -> None:
        self._data = data_h5f
        self._sidecars = sidecars or {}

    def __getitem__(self, key: str) -> Any:
        sidecar = self._sidecars.get(key)
        if sidecar is not None and key in sidecar:
            return sidecar[key]
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        sidecar = self._sidecars.get(key)
        return key in self._data or (sidecar is not None and key in sidecar)

    def keys(self) -> list[str]:
        keys = list(self._data.keys())
        for sidecar in self._sidecars.values():
            keys.extend(key for key in sidecar.keys() if key not in keys)
        return keys

    def __iter__(self):
        return iter(self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def close(self) -> None:
        self._data.close()
        seen: set[int] = set()
        for sidecar in self._sidecars.values():
            if id(sidecar) not in seen:
                sidecar.close()
                seen.add(id(sidecar))


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
    """Read one published Raw v34 episode without training-admission gates."""

    def __init__(self, h5_path: str | Path) -> None:
        self._path = Path(h5_path)
        self._closed = False
        self._cache: dict[str, np.ndarray] = {}
        if not self._path.is_dir():
            raise ValueError(f"episode must be a published directory: {self._path}")

        paths = {
            "data": self._path / "data.h5",
            "depth": self._path / "depth.h5",
            "rgb": self._path / "rgb.mp4",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"episode is missing required files {missing}: {self._path}")
        if paths["rgb"].stat().st_size == 0:
            raise ValueError(f"episode RGB file is empty: {paths['rgb']}")

        self._rgb_path = paths["rgb"]
        self._rgb_decoder: VideoDecoder | None = None
        self._data_h5f = h5py.File(paths["data"], "r")
        try:
            depth_h5f = h5py.File(paths["depth"], "r")
        except Exception:
            self._data_h5f.close()
            raise
        self._depth_h5f = depth_h5f
        self._h5f = MergedH5File(self._data_h5f, {"depth": self._depth_h5f})
        try:
            if self.schema_version != EPISODE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported episode schema v{self.schema_version}; expected v"
                    f"{EPISODE_SCHEMA_VERSION}"
                )
            self._validate_layout()
        except Exception:
            self.close()
            raise

    @property
    def h5f(self) -> MergedH5File:
        """Merged view of ``data.h5`` and the ``depth.h5`` sidecar."""
        return self._h5f

    @property
    def h5_path(self) -> Path:
        return self._path

    @property
    def schema_version(self) -> int:
        meta = self._h5f.get("meta")
        if not isinstance(meta, h5py.Group):
            raise ValueError("episode is missing meta")
        return _positive_int_attr(meta.attrs, "schema_version")

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def dt(self) -> float:
        return 1.0 / self._control_hz

    @property
    def episode_valid(self) -> bool:
        return self._episode_valid

    def _validate_layout(self) -> None:
        """Validate v34 structure, static calibration, and physical mount."""
        meta = self._h5f.get("meta")
        if not isinstance(meta, h5py.Group):
            raise ValueError("episode is missing meta")
        attrs = meta.attrs
        self._num_frames = _positive_int_attr(attrs, "num_frames", allow_zero=True)
        self._control_hz = _positive_float_attr(attrs, "control_hz")
        _text_attr(attrs, "task_label")
        _text_attr(attrs, "collection_source")
        episode_valid = _scalar_attr(attrs, "episode_valid")
        if not isinstance(episode_valid, (bool, np.bool_)):
            raise ValueError("episode metadata episode_valid must be bool")
        self._episode_valid = bool(episode_valid)
        _validate_color_camera_metadata(attrs)
        _validate_hand_mount_metadata(attrs)

        unexpected_entries = set(self._data_h5f.keys()) - ({"meta"} | set(DATASET_SPECS))
        if unexpected_entries:
            raise ValueError(f"unexpected data.h5 entries: {sorted(unexpected_entries)}")
        datasets = {
            key: value for key, value in self._data_h5f.items() if isinstance(value, h5py.Dataset)
        }
        errors = validate_data_layout(
            {key: value.shape for key, value in datasets.items()},
            {key: value.dtype for key, value in datasets.items()},
            frame_count=self._num_frames,
        )
        if errors:
            raise ValueError("episode layout invalid: " + "; ".join(errors))

        depth_h5f = self._depth_h5f
        if set(depth_h5f.keys()) != {"depth"}:
            raise ValueError("depth.h5 must contain exactly the depth dataset")
        depth = depth_h5f["depth"]
        expected_depth_shape = (
            self._num_frames,
            _positive_int_attr(attrs, "camera_color_height"),
            _positive_int_attr(attrs, "camera_color_width"),
        )
        if not isinstance(depth, h5py.Dataset) or depth.shape != expected_depth_shape:
            raise ValueError(
                f"depth shape must be {expected_depth_shape}, got {getattr(depth, 'shape', None)}"
            )
        if np.dtype(depth.dtype) != np.dtype(np.uint16):
            raise ValueError(f"depth dtype must be uint16, got {depth.dtype}")

    def _decoder(self) -> VideoDecoder:
        if self._rgb_decoder is None:
            self._rgb_decoder = VideoDecoder(self._rgb_path)
        return self._rgb_decoder

    def read_camera_frame(self, key: str, index: int) -> np.ndarray:
        """Read one RGB or depth frame by index."""
        if key == "rgb":
            decoder = self._decoder()
            if decoder.frame_count == 0:
                raise ValueError(f"MP4 file contains no frames: {self._path}")
            return decoder.read_frame(index)
        if key in self._h5f:
            return np.asarray(self._h5f[key][index])
        raise KeyError(f"camera dataset {key!r} not found in {self._path}")

    def read_camera_all(self, key: str) -> np.ndarray:
        """Read all camera frames, caching the requested modality."""
        if key in self._cache:
            return self._cache[key]
        if key == "rgb":
            data = self._decoder().read_all()
            if data.shape[0] != self._num_frames:
                raise ValueError(
                    f"RGB length {data.shape[0]} does not match num_frames {self._num_frames}"
                )
        elif key in self._h5f:
            data = np.asarray(self._h5f[key][:])
        else:
            raise KeyError(f"camera dataset {key!r} not found in {self._path}")
        self._cache[key] = data
        return data

    def iter_camera_frames(self, key: str) -> Iterator[np.ndarray]:
        """Yield RGB frames sequentially without retaining the episode."""
        if key != "rgb":
            raise ValueError("iter_camera_frames supports only MP4-backed RGB")
        yield from self._decoder().iter_frames()

    def close(self) -> None:
        """Close sidecars and any lazily opened RGB decoder."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._cache.clear()
        if self._rgb_decoder is not None:
            self._rgb_decoder.close()
            self._rgb_decoder = None
        if hasattr(self, "_h5f") and self._h5f is not None:
            self._h5f.close()
            self._h5f = None  # type: ignore[assignment]

    def __enter__(self) -> "EpisodeReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
