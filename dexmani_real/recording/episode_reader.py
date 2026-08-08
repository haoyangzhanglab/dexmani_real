"""Unified HDF5 episode reader.

Reads camera frames from DexMani episodes. Non-camera datasets
(arm_qpos, hand_qpos, flags, etc.) are accessed directly through
:attr:`h5f` — a merged view of ``data.h5`` + ``depth.h5``.

Supports two formats:

- **Legacy** (single ``.h5`` file): everything in one flat HDF5.
- **New** (directory): ``data.h5`` (non-camera + pointcloud),
  ``depth.h5`` (depth frames), ``rgb.mp4`` (RGB video).

Usage::

    with EpisodeReader("episode_001") as reader:
        # Non-camera data — direct merged-h5py access
        arm_qpos = reader.h5f["arm_qpos"][:]

        # Camera data
        rgb_frame  = reader.read_camera_frame("rgb", 42)
        all_depth  = reader.read_camera_all("depth")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.recording.video_codec import VideoDecoder
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Camera dataset keys routed to depth.h5 (new format).
# "rgb" is handled by VideoDecoder (MP4 sidecar) — not in either HDF5 file.
_CAM_KEYS = {"depth"}


@dataclass(frozen=True)
class EpisodeTiming:
    """Normalized timing view for both legacy and schema-v13 episodes."""

    rate_hz: float
    grid_dt_s: float
    grid_duration_s: float
    wall_duration_s: float
    non_sampled_duration_s: float


class MergedH5File:
    """Transparent merged view of ``data.h5`` + ``depth.h5``.

    Camera keys (``"depth"``) are routed to the depth file; everything
    else goes to the data file.  ``"rgb"`` is handled by
    :class:`VideoDecoder` and is **not** present in either file.
    """

    __slots__ = ("_data", "_depth")

    def __init__(self, data_h5f: h5py.File, depth_h5f: h5py.File | None) -> None:
        self._data = data_h5f
        self._depth = depth_h5f

    def __getitem__(self, key: str) -> Any:
        if self._depth is not None and key in _CAM_KEYS and key in self._depth:
            return self._depth[key]
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data or (self._depth is not None and key in self._depth)

    def keys(self) -> list[str]:
        ks = list(self._data.keys())
        if self._depth is not None:
            ks.extend(k for k in self._depth.keys() if k not in ks)
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
        if self._depth is not None:
            self._depth.close()


class EpisodeReader:
    """Read camera frames from DexMani episodes.

    :attr:`h5f` returns a merged dict-like view over ``data.h5`` and
    ``depth.h5`` so downstream code that accesses datasets by key
    (``f["arm_qpos"]``, ``f["depth"]``, ``f["pointcloud"]``) works
    transparently across both old (single ``.h5``) and new (directory)
    formats.
    """

    def __init__(self, h5_path: str | Path) -> None:
        self._path = Path(h5_path)
        self._is_legacy = self._path.is_file()
        self._closed = False
        if self._is_legacy:
            # Old format: single episode_XXX.h5 file.
            self._data_h5f: h5py.File = h5py.File(self._path, "r")
            self._h5f = MergedH5File(self._data_h5f, None)
            self._rgb_decoder: VideoDecoder | None = None
        elif self._path.is_dir():
            # New format: episode_XXX/ directory.
            data_path = self._path / "data.h5"
            if not data_path.is_file():
                raise FileNotFoundError(f"data.h5 not found in {self._path}")
            depth_path = self._path / "depth.h5"
            self._data_h5f = h5py.File(str(data_path), "r")
            depth_h5f = h5py.File(str(depth_path), "r") if depth_path.is_file() else None
            self._h5f = MergedH5File(self._data_h5f, depth_h5f)
            # RGB sidecar (optional).
            rgb_mp4 = self._path / "rgb.mp4"
            self._rgb_decoder = VideoDecoder(rgb_mp4) if rgb_mp4.is_file() else None
        else:
            raise FileNotFoundError(f"Episode not found: {self._path}")

        # Lazy pre-decode cache for read_camera_all().
        self._cache: dict[str, np.ndarray] = {}

    # -- public properties ------------------------------------------------

    @property
    def h5f(self) -> MergedH5File:
        """Merged view of ``data.h5`` + ``depth.h5``.

        ``f["rgb"]`` raises ``KeyError`` — use :meth:`read_camera_frame`
        or :meth:`read_camera_all` for RGB frames (MP4 decoding).
        """
        return self._h5f

    @property
    def h5_path(self) -> Path:
        return self._path

    @property
    def timing(self) -> EpisodeTiming:
        """Return timing with v13 attrs preferred and safe legacy fallbacks.

        Legacy ``fps`` may have been diluted by pauses because it was computed
        from wall time.  A valid ``control_hz`` or timestamp grid therefore
        takes precedence over that attribute.
        """
        meta = self._h5f.get("meta")
        attrs = meta.attrs if meta is not None else {}

        def _positive(value: Any) -> float | None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if np.isfinite(result) and result > 0 else None

        timestamps = np.asarray(self._h5f["timestamp"][:], dtype=np.float64) if "timestamp" in self._h5f else None
        timestamp_dt_s: float | None = None
        timestamp_duration_s: float | None = None
        if timestamps is not None and timestamps.size >= 2:
            finite = timestamps[np.isfinite(timestamps)]
            if finite.size >= 2:
                positive_deltas = np.diff(finite)
                positive_deltas = positive_deltas[positive_deltas > 0]
                if positive_deltas.size:
                    timestamp_dt_s = float(np.median(positive_deltas))
                span = float(finite[-1] - finite[0])
                if span >= 0:
                    timestamp_duration_s = span

        control_hz = _positive(attrs.get("control_hz"))
        explicit_grid_dt_s = _positive(attrs.get("grid_dt_s"))
        legacy_fps = _positive(attrs.get("fps"))
        grid_dt_s = explicit_grid_dt_s or timestamp_dt_s
        if grid_dt_s is None and control_hz is not None:
            grid_dt_s = 1.0 / control_hz
        if grid_dt_s is None and legacy_fps is not None:
            grid_dt_s = 1.0 / legacy_fps
        if grid_dt_s is None:
            grid_dt_s = 1.0 / 16.0

        rate_hz = control_hz or (1.0 / grid_dt_s)
        explicit_grid_duration_s = attrs.get("grid_duration_s")
        try:
            grid_duration_s = float(explicit_grid_duration_s) if explicit_grid_duration_s is not None else float("nan")
        except (TypeError, ValueError):
            grid_duration_s = float("nan")
        if not np.isfinite(grid_duration_s) or grid_duration_s < 0:
            if timestamp_duration_s is not None:
                grid_duration_s = timestamp_duration_s
            else:
                num_frames = int(attrs.get("num_frames", len(timestamps) if timestamps is not None else 0))
                grid_duration_s = max(0, num_frames - 1) * grid_dt_s

        wall_duration_s = float(attrs.get("wall_duration_s", attrs.get("duration", grid_duration_s)))
        if not np.isfinite(wall_duration_s) or wall_duration_s < 0:
            wall_duration_s = grid_duration_s
        non_sampled_duration_s = float(attrs.get("non_sampled_duration_s", max(0.0, wall_duration_s - grid_duration_s)))
        if not np.isfinite(non_sampled_duration_s) or non_sampled_duration_s < 0:
            non_sampled_duration_s = max(0.0, wall_duration_s - grid_duration_s)

        return EpisodeTiming(
            rate_hz=rate_hz,
            grid_dt_s=grid_dt_s,
            grid_duration_s=grid_duration_s,
            wall_duration_s=wall_duration_s,
            non_sampled_duration_s=non_sampled_duration_s,
        )

    # -- camera queries ---------------------------------------------------

    def read_camera_frame(self, key: str, index: int) -> np.ndarray:
        """Read a single camera frame by index.

        For MP4 RGB frames, tail indices beyond the unique frame count
        are clamped to the last available frame (forward-fill).
        """
        if key == "rgb" and self._rgb_decoder is not None:
            n = self._rgb_decoder.frame_count
            if n == 0:
                raise ValueError(f"MP4 file contains no frames: {self._path}")
            return self._rgb_decoder.read_frame(min(index, n - 1))
        if key in self._h5f:
            return np.asarray(self._h5f[key][index])
        raise KeyError(f"Camera dataset '{key}' not found in {self._path}")

    def read_camera_all(self, key: str) -> np.ndarray:
        """Read all camera frames. Cached after the first call.

        Returns a ``(T, ...)`` array (``uint8`` for RGB, ``uint16`` for depth).
        RGB frames from MP4 are forward-filled to match the grid length
        (``num_frames`` in ``/meta``) so all streams have the same ``T``.
        """
        if key in self._cache:
            return self._cache[key]

        if key == "rgb" and self._rgb_decoder is not None:
            data = self._rgb_decoder.read_all()
            # Forward-fill RGB to match grid length (MP4 stores unique frames only).
            grid_len = self._h5f["meta"].attrs.get("num_frames", 0)
            if grid_len > data.shape[0]:
                pad = np.repeat(data[-1:], grid_len - data.shape[0], axis=0)
                data = np.concatenate([data, pad], axis=0)
        elif key in self._h5f:
            data = np.asarray(self._h5f[key][:])
        else:
            raise KeyError(f"Camera dataset '{key}' not found in {self._path}")

        self._cache[key] = data
        return data

    # -- context manager --------------------------------------------------

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
