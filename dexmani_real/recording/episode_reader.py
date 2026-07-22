"""Unified HDF5 episode reader with transparent MP4 sidecar support.

Reads camera frames from DexMani HDF5 episodes, auto-detecting video
sidecar files (``.rgb.mp4``, ``.depth.mp4``) written by the opt-in
H.264 encoding path.  When a sidecar is present camera frames are
decoded from video; otherwise they fall back to the HDF5 dataset
(legacy LZF-compressed path).

Non-camera datasets (arm_qpos, hand_qpos, flags, etc.) are accessed
directly through :attr:`h5f` — the underlying ``h5py.File``.

Usage::

    with EpisodeReader("episode_001.h5") as reader:
        # Non-camera data — direct h5py access
        arm_qpos = reader.h5f["arm_qpos"][:]

        # Camera data — transparent video/HDF5
        rgb_frame  = reader.read_camera_frame("rgb", 42)
        all_frames = reader.read_camera_all("rgb")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.recording.video_codec import VideoDecoder
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Camera dataset keys that may have video sidecars.
_VIDEO_KEYS = ("rgb", "depth")


class EpisodeReader:
    """Read camera frames from HDF5 episodes with optional MP4 sidecar support.

    Auto-detects ``.rgb.mp4`` / ``.depth.mp4`` files alongside the ``.h5``
    episode.  When present, :meth:`read_camera_frame` and :meth:`read_camera_all`
    decode from video; otherwise they read from the HDF5 dataset (legacy LZF).

    :attr:`h5f` gives direct access to the underlying ``h5py.File`` for all
    non-camera datasets (arm state, actions, flags, timestamps, metadata, etc.).
    """

    def __init__(self, h5_path: str | Path) -> None:
        self._h5_path = Path(h5_path)
        if not self._h5_path.is_file():
            raise FileNotFoundError(f"Episode not found: {self._h5_path}")

        self._h5f: h5py.File = h5py.File(self._h5_path, "r")

        # Auto-detect video sidecars — one decoder per camera key.
        self._video: dict[str, VideoDecoder] = {}
        for key in _VIDEO_KEYS:
            mp4 = self._h5_path.with_suffix(f".{key}.mp4")
            if mp4.is_file():
                self._video[key] = VideoDecoder(mp4)
                logger.debug("Video sidecar detected: %s", mp4.name)

        # Lazy pre-decode cache for read_camera_all().
        self._cache: dict[str, np.ndarray] = {}

    # -- public properties ------------------------------------------------

    @property
    def h5f(self) -> h5py.File:
        """Underlying ``h5py.File`` for direct dataset access."""
        return self._h5f

    @property
    def h5_path(self) -> Path:
        return self._h5_path

    # -- camera queries ---------------------------------------------------

    def has_video(self, key: str) -> bool:
        """Return True if *key* has an MP4 sidecar (vs legacy HDF5 dataset)."""
        return key in self._video

    def video_frame_count(self, key: str) -> int | None:
        """Number of frames in the video sidecar, or None if no sidecar."""
        dec = self._video.get(key)
        return dec.frame_count if dec is not None else None

    # -- single-frame read ------------------------------------------------

    def read_camera_frame(self, key: str, index: int) -> np.ndarray:
        """Read a single camera frame by index.

        Legacy HDF5 path: O(1) random access.
        Video path: O(index) — seeks to nearest keyframe, decodes forward.
        For repeated random access prefer :meth:`read_camera_all` + index.
        """
        if key in self._video:
            return self._video[key].read_frame(index)
        if key in self._h5f:
            return np.asarray(self._h5f[key][index])
        raise KeyError(f"Camera dataset '{key}' not found in {self._h5_path}")

    # -- bulk read --------------------------------------------------------

    def read_camera_all(self, key: str) -> np.ndarray:
        """Read all camera frames.  Cached after the first call.

        Returns a ``(T, ...)`` array with the same dtype as the source
        (``uint8`` for RGB, ``uint16`` for depth).
        """
        if key in self._cache:
            return self._cache[key]

        if key in self._video:
            data = self._video[key].read_all()
        elif key in self._h5f:
            data = np.asarray(self._h5f[key][:])
        else:
            raise KeyError(f"Camera dataset '{key}' not found in {self._h5_path}")

        self._cache[key] = data
        return data

    # -- context manager --------------------------------------------------

    def close(self) -> None:
        """Close video decoders and the HDF5 file.  Idempotent."""
        for dec in self._video.values():
            dec.close()
        self._video.clear()
        self._cache.clear()
        if self._h5f is not None:
            self._h5f.close()
            self._h5f = None  # type: ignore[assignment]

    def __enter__(self) -> "EpisodeReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
