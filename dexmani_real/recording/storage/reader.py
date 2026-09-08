"""Reader for the current transactional HDF5 episode format.

Reads camera frames from DexMani episodes. Non-camera datasets
(arm_qpos, hand_qpos, flags, etc.) are accessed directly through
:attr:`h5f` — a merged view of the episode HDF5 sidecars.

An episode is one published directory containing ``data.h5``, ``depth.h5``,
and ``rgb.mp4``. Other raw layouts require an external migration tool.

Usage::

    with EpisodeReader("episode_001") as reader:
        arm_qpos = reader.h5f["arm_qpos"][:]

        rgb_frame  = reader.read_camera_frame("rgb", 42)
        all_depth  = reader.read_camera_all("depth")
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.recording.storage.schema import (
    EPISODE_SCHEMA_VERSION,
    validate_data_layout,
)
from dexmani_real.recording.storage.video import VideoDecoder
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class ValidityState(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class EpisodeTiming:
    """Timing metadata from a supported episode schema."""

    rate_hz: float
    grid_dt_s: float
    grid_duration_s: float
    wall_duration_s: float
    non_sampled_duration_s: float


class MergedH5File:
    """Transparent merged view of ``data.h5`` and camera HDF5 sidecars.

    Camera keys are routed to their sidecar files; everything else goes to
    the data file.  ``"rgb"`` is handled by
    :class:`VideoDecoder` and is **not** present in either file.
    """

    __slots__ = ("_data", "_sidecars")

    def __init__(
        self, data_h5f: h5py.File, sidecars: dict[str, h5py.File] | None = None
    ) -> None:
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
        ks = list(self._data.keys())
        for sidecar in self._sidecars.values():
            ks.extend(k for k in sidecar.keys() if k not in ks)
        return ks

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


class EpisodeReader:
    """Read camera frames from DexMani episodes.

    :attr:`h5f` returns a merged dict-like view over ``data.h5`` and camera
    sidecars so downstream code can access datasets by key
    (``f["arm_qpos"]``, ``f["depth"]``).
    """

    def __init__(self, h5_path: str | Path) -> None:
        """Open one published episode.

        Check the supported schema and structural layout without decoding RGB
        or replaying the runtime's admission proofs.
        """
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
            raise FileNotFoundError(
                f"episode is missing required files {missing}: {self._path}"
            )

        self._rgb_decoder: VideoDecoder | None = None
        self._data_h5f = h5py.File(paths["data"], "r")
        try:
            depth_h5f = h5py.File(paths["depth"], "r")
        except Exception:
            self._data_h5f.close()
            raise
        self._h5f = MergedH5File(
            self._data_h5f,
            {"depth": depth_h5f},
        )
        try:
            schema_version = self.schema_version
            if schema_version != EPISODE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported episode schema v{schema_version}; expected v"
                    f"{EPISODE_SCHEMA_VERSION}"
                )
            self._rgb_decoder = VideoDecoder(paths["rgb"])
            self.require_valid(purpose="episode read")
        except Exception:
            self.close()
            raise

    @property
    def h5f(self) -> MergedH5File:
        """Merged view of ``data.h5`` and camera HDF5 sidecars.

        ``f["rgb"]`` raises ``KeyError`` — use :meth:`read_camera_frame`
        or :meth:`read_camera_all` for RGB frames (MP4 decoding).
        """
        return self._h5f

    @property
    def h5_path(self) -> Path:
        return self._path

    @property
    def schema_version(self) -> int:
        meta = self._h5f.get("meta")
        return 0 if meta is None else int(meta.attrs.get("schema_version", 0) or 0)

    @property
    def min_frames_met(self) -> bool:
        """Quality label only; short episodes may still be internally valid."""
        meta = self._h5f.get("meta")
        return bool(meta is not None and meta.attrs.get("min_frames_met", False))

    @property
    def validity(self) -> ValidityState:
        """Return whether the current supported episode is internally consistent."""
        meta = self._h5f.get("meta")
        if meta is None:
            return ValidityState.INVALID
        frame_count = int(meta.attrs.get("num_frames", -1))
        datasets = {
            key: dataset
            for key, dataset in self._data_h5f.items()
            if isinstance(dataset, h5py.Dataset)
        }
        dataset_shapes = {
            key: tuple(dataset.shape) for key, dataset in datasets.items()
        }
        dataset_dtypes = {key: dataset.dtype for key, dataset in datasets.items()}
        layout_errors = validate_data_layout(
            dataset_shapes, dataset_dtypes, frame_count=frame_count
        )
        if layout_errors or "depth" not in self._h5f:
            return ValidityState.INVALID
        depth = self._h5f["depth"]
        if not isinstance(depth, h5py.Dataset) or depth.shape[:1] != (frame_count,):
            return ValidityState.INVALID
        return ValidityState.VALID

    def require_valid(self, purpose: str = "training") -> None:
        state = self.validity
        if state is not ValidityState.VALID:
            raise ValueError(
                f"episode validity is {state.value}; {purpose} requires VALID data "
                f"from raw schema v{self.schema_version}"
            )

    @property
    def timing(self) -> EpisodeTiming:
        """Nominal controller period and actual recorded timestamp span."""
        attrs = self._h5f["meta"].attrs
        control_hz = float(attrs["control_hz"])
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("episode control_hz must be finite and positive")
        timestamps = self._h5f["timestamp"]
        span = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        wall_duration = float(attrs.get("wall_duration_s", span))
        return EpisodeTiming(
            rate_hz=control_hz,
            grid_dt_s=1.0 / control_hz,
            grid_duration_s=span,
            wall_duration_s=wall_duration,
            non_sampled_duration_s=max(
                0.0, span - max(0, len(timestamps) - 1) / control_hz
            ),
        )

    def read_camera_frame(self, key: str, index: int) -> np.ndarray:
        """Read a single camera frame by index.

        The MP4 sidecar must have exactly one frame per episode grid slot.
        """
        if key == "rgb" and self._rgb_decoder is not None:
            n = self._rgb_decoder.frame_count
            if n == 0:
                raise ValueError(f"MP4 file contains no frames: {self._path}")
            return self._rgb_decoder.read_frame(index)
        if key in self._h5f:
            return np.asarray(self._h5f[key][index])
        raise KeyError(f"Camera dataset '{key}' not found in {self._path}")

    def read_camera_all(self, key: str) -> np.ndarray:
        """Read all camera frames. Cached after the first call.

        Returns a ``(T, ...)`` array (``uint8`` for RGB, ``uint16`` for depth).
        """
        if key in self._cache:
            return self._cache[key]

        if key == "rgb" and self._rgb_decoder is not None:
            data = self._rgb_decoder.read_all()
            grid_len = int(self._h5f["meta"].attrs.get("num_frames", 0))
            if grid_len != data.shape[0]:
                raise ValueError(
                    f"RGB length {data.shape[0]} does not match grid length {grid_len}"
                )
        elif key in self._h5f:
            data = np.asarray(self._h5f[key][:])
        else:
            raise KeyError(f"Camera dataset '{key}' not found in {self._path}")

        self._cache[key] = data
        return data

    def iter_camera_frames(self, key: str) -> Iterator[np.ndarray]:
        """Yield one camera modality sequentially without caching the episode.

        Streaming is currently meaningful for the MP4-backed RGB modality.
        HDF5 camera modalities remain directly sliceable through :attr:`h5f`.
        """
        if key != "rgb":
            raise ValueError(
                "iter_camera_frames currently supports only MP4-backed RGB"
            )
        if self._rgb_decoder is None:
            raise RuntimeError("EpisodeReader has no RGB decoder")
        yield from self._rgb_decoder.iter_frames()

    def close(self) -> None:
        """Close all files and decoders. Idempotent."""
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
