"""Video codec utilities for HDF5 episode recording.

Provides streaming H.264 encoding (write) and frame decoding (read) for
camera data stored as MP4 sidecar files alongside HDF5 episodes.

Typical usage (encode)::

    with VideoEncoder(path, fps=16.0, width=640, height=480) as enc:
        for frame in camera_frames:
            enc.write_frame(frame)

Typical usage (decode)::

    with VideoDecoder(path) as dec:
        all_frames = dec.read_all()
        single = dec.read_frame(42)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class VideoEncoderConfig:
    """Configuration for real-time H.264 video encoding.

    Attributes:
        codec: FFmpeg encoder name. ``libx264`` (YUV) — the standard choice
            for player-compatible MP4.  ``libx264rgb`` exists but produces
            RGB-native streams that most players misinterpret as YUV
            (washed-out / color-shifted output).  We use ``libx264`` +
            ``yuv444p`` for full chroma resolution with automatic
            RGB↔YUV conversion in the encode/decode path.
        crf: Constant Rate Factor (0–51).  0 = lossless, 18 = visually
            near-lossless (default), 23 = default, 51 = worst.
        preset: libx264 speed/compression preset.  ``ultrafast`` is the
            right choice for real-time encoding at moderate resolutions.
        pixel_format: Pixel format string. ``yuv444p`` preserves full
            chroma resolution (no subsampling) for CV data fidelity.
            ``rgb24`` is only valid with ``libx264rgb`` (avoid — see above).
    """

    codec: str = "libx264"
    crf: int = 18
    preset: str = "ultrafast"
    pixel_format: str = "yuv444p"


class VideoEncoder:
    """Streaming H.264 encoder that writes a ``.mp4`` sidecar file.

    Frames are written sequentially via :meth:`write_frame`.  The
    underlying FFmpeg muxer is :mod:`threading.Lock`-protected so a
    single encoder can be driven from the recorder's background writer
    thread and/or the stop-time drain path without races.

    The output container is created lazily on the first frame so the
    constructor never blocks on I/O.  Call :meth:`close` (or use the
    context manager) to finalise the MP4 — otherwise the file will be
    unplayable.
    """

    def __init__(
        self,
        path: Path,
        config: VideoEncoderConfig | None = None,
        fps: float = 16.0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._path = Path(path)
        self._cfg = config or VideoEncoderConfig()
        self._fps = fps
        self._width = width
        self._height = height

        self._container: av.container.OutputContainer | None = None
        self._stream: Any = None  # av.video.VideoStream — PyAV stubs incomplete
        self._frame_count: int = 0
        self._lock = threading.Lock()
        self._closed = False

    # -- public ----------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    @property
    def closed(self) -> bool:
        return self._closed

    def write_frame(self, frame: np.ndarray) -> None:
        """Encode and mux one RGB frame.

        Frame shape must be ``(height, width, 3)`` with dtype ``uint8``.
        The caller is responsible for feeding frames in display order;
        duplicate frames (forward-filled grid slots) are fine — H.264
        encodes them as near-zero-cost skip blocks.
        """
        if self._closed:
            raise RuntimeError("VideoEncoder is closed")
        # _write_frame_impl does the lock dance so the public method
        # signature stays clean.
        with self._lock:
            if self._closed:
                raise RuntimeError("VideoEncoder is closed")
            self._write_frame_impl(frame)

    def close(self) -> None:
        """Flush the encoder and finalise the MP4 container.

        Idempotent — safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        with self._lock:
            if self._container is None:
                return
            # Flush any buffered frames still inside the encoder.
            if self._stream is not None:
                for packet in self._stream.encode(None):  # type: ignore[attr-defined]
                    self._container.mux(packet)
            self._container.close()
            self._container = None
            self._stream = None
            logger.debug(
                "VideoEncoder closed: %s (%d frames)",
                self._path.name,
                self._frame_count,
            )

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "VideoEncoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- internals -------------------------------------------------------

    def _write_frame_impl(self, frame: np.ndarray) -> None:
        """Lock must be held by caller."""
        # Validate shape so we fail early with a clear message.
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) uint8 RGB frame, got shape {frame.shape}"
            )

        if self._container is None:
            self._container = av.open(str(self._path), "w", format="mp4")
            # add_stream returns a VideoStream at runtime for video codecs;
            # PyAV stubs are incomplete so use Any for _stream.
            self._stream = self._container.add_stream(
                self._cfg.codec, rate=int(self._fps)
            )
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = self._cfg.pixel_format
            self._stream.options = {
                "crf": str(self._cfg.crf),
                "preset": self._cfg.preset,
            }

        # Convert numpy RGB → PyAV frame → reformat to encoder pixel format.
        # libx264 (yuv444p): reformat() does RGB→YUV automatically;
        # libx264rgb (rgb24): reformat() is a no-op (same format).
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        if self._cfg.pixel_format != "rgb24":
            av_frame = av_frame.reformat(format=self._cfg.pixel_format)
        av_frame.pts = self._frame_count

        # encode() returns packets; mux writes them to the container.
        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

        self._frame_count += 1


class VideoDecoder:
    """Read frames from an MP4 sidecar file back into numpy arrays.

    Supports both sequential bulk reads (``read_all()``) and indexed
    single-frame reads (``read_frame(idx)``).  Indexed reads seek to the
    nearest keyframe and decode forward, so random access is O(N) in the
    worst case — pre-decode to memory for interactive use.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._container: av.container.InputContainer | None = None
        self._stream: Any = None  # av.video.VideoStream — PyAV stubs are incomplete
        self._frame_count: int = 0
        self._fps_val: float = 0.0
        self._opened = False

    # -- public ----------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total frame count in the video (cached on first open)."""
        if not self._opened:
            self._open()
        return self._frame_count

    def read_all(self) -> np.ndarray:
        """Decode all frames and return as a ``(T, H, W, 3)`` uint8 array."""
        if not self._opened:
            self._open()
        frames: list[np.ndarray] = []
        assert self._container is not None
        self._container.seek(0)  # rewind to start
        # Demux + decode in one pass (no seeking — pure sequential).
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                # PyAV stubs include subtitle types in the decode union;
                # filter to VideoFrame (the only type we care about).
                if isinstance(frame, av.VideoFrame):
                    frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError(f"No frames decoded from {self._path}")
        return np.stack(frames, axis=0)

    def read_frame(self, index: int) -> np.ndarray:
        """Decode a single frame by index.

        Note: this seeks to the nearest keyframe and decodes forward to
        ``index``, so repeated random access is inefficient.  For
        interactive scrubbing, call ``read_all()`` once and index into
        the result.
        """
        if not self._opened:
            self._open()
        assert self._container is not None
        if index < 0 or index >= self._frame_count:
            raise IndexError(
                f"Frame index {index} out of range [0, {self._frame_count})"
            )

        # Seek to the target timestamp (best-effort — lands on preceding keyframe).
        time_base = self._stream.time_base  # Fraction (PyAV stubs: Fraction | None)
        if time_base is None:
            time_base = self._stream.average_rate  # fallback
            if time_base is None:
                raise RuntimeError(f"Cannot determine time base for {self._path}")
        pts = int(index * time_base.denominator / (self._fps_val * time_base.numerator))
        self._container.seek(pts, stream=self._stream)

        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame) and frame.pts is not None:
                    t = frame.pts * time_base.numerator / time_base.denominator * self._fps_val
                    if t >= index:
                        return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"Failed to decode frame {index} from {self._path}")

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
            self._stream = None
            self._opened = False

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "VideoDecoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- internals -------------------------------------------------------

    def _open(self) -> None:
        if self._opened:
            return
        self._container = av.open(str(self._path), "r")
        self._stream = self._container.streams.video[0]
        self._frame_count = int(self._stream.frames) if self._stream.frames > 0 else 0
        avg_rate = self._stream.average_rate
        if avg_rate is None:
            raise ValueError(f"No average frame rate in {self._path}")
        self._fps_val = float(avg_rate)
        self._opened = True
