"""Encode and decode H.264 MP4 sidecars for HDF5 episodes.

Encode::

    with VideoEncoder(path, fps=16.0, width=640, height=480) as enc:
        for frame in camera_frames:
            enc.write_frame(frame)

Decode::

    with VideoDecoder(path) as dec:
        all_frames = dec.read_all()
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class VideoEncoderConfig:
    """Real-time H.264 settings: libx264 + yuv444p preserves full chroma.

    RGB/YUV conversion is automatic; libx264rgb may display incorrect colors
    in players. crf ranges from 0 (lossless) to 51 (worst), default 18;
    ultrafast favors encoding speed. rgb24 requires libx264rgb.
    """

    codec: str = "libx264"
    crf: int = 18
    preset: str = "ultrafast"
    pixel_format: str = "yuv444p"


class VideoEncoder:
    """Write MP4 frames sequentially in the recorder process.

    Opens the container on the first frame. close() or a context manager must
    finalize the MP4 for playback.
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
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    def write_frame(self, frame: np.ndarray) -> None:
        """Encode and mux one RGB frame.

        Frame shape must be ``(height, width, 3)`` with dtype ``uint8``.
        The caller is responsible for feeding frames in display order;
        repeated camera samples are encoded efficiently by H.264.
        """
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
        if self._container is None:
            return
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

    def __enter__(self) -> "VideoEncoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _write_frame_impl(self, frame: np.ndarray) -> None:
        # Validate shape so we fail early with a clear message.
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) uint8 RGB frame, got shape {frame.shape}")
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError(
                f"Frame shape {(frame.shape[0], frame.shape[1])} does not match "
                f"encoder size ({self._height}, {self._width})"
            )

        if self._container is None:
            self._container = av.open(str(self._path), "w", format="mp4")
            # add_stream returns a VideoStream at runtime for video codecs;
            # PyAV stubs are incomplete so use Any for _stream.
            self._stream = self._container.add_stream(self._cfg.codec, rate=int(self._fps))
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = self._cfg.pixel_format
            self._stream.options = {
                "crf": str(self._cfg.crf),
                "preset": self._cfg.preset,
            }

        # Convert RGB to the encoder pixel format; PyAV performs RGB→YUV.
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        if self._cfg.pixel_format != "rgb24":
            av_frame = av_frame.reformat(format=self._cfg.pixel_format)
        av_frame.pts = self._frame_count

        # encode() returns packets; mux writes them to the container.
        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

        self._frame_count += 1


class VideoDecoder:
    """Decode MP4 frames to NumPy arrays, in bulk or by index.

    Indexed reads decode forward from a keyframe; prefer read_all() for
    interactive scrubbing to avoid repeated seeks.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._container: av.container.InputContainer | None = None
        self._stream: Any = None  # av.video.VideoStream — PyAV stubs are incomplete
        self._frame_count: int = 0
        self._fps_val: float = 0.0
        self._opened = False

    @property
    def frame_count(self) -> int:
        """Total frame count in the video (cached on first open)."""
        if not self._opened:
            self._open()
        return self._frame_count

    def read_all(self) -> np.ndarray:
        """Decode all frames and return as a ``(T, H, W, 3)`` uint8 array."""
        frames = list(self.iter_frames())
        if not frames:
            raise ValueError(f"No frames decoded from {self._path}")
        return np.stack(frames, axis=0)

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield decoded RGB frames sequentially without retaining the video.

        The iterator rewinds the stream before decoding.  It is intended for
        offline transforms that write selected frames directly to another
        container and must not materialize a full recording in memory.
        """
        if not self._opened:
            self._open()
        if self._container is None:
            raise RuntimeError("VideoDecoder: container is None after _open()")
        self._container.seek(0)
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                # Decode only VideoFrame values; PyAV stubs include subtitle types.
                if isinstance(frame, av.VideoFrame):
                    yield frame.to_ndarray(format="rgb24")

    def read_frame(self, index: int) -> np.ndarray:
        """Decode from the nearest keyframe to index; prefer read_all() for repeated access."""
        if not self._opened:
            self._open()
        if self._container is None:
            raise RuntimeError("VideoDecoder: container is None after _open()")
        if index < 0 or index >= self._frame_count:
            raise IndexError(f"Frame index {index} out of range [0, {self._frame_count})")

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

    def __enter__(self) -> "VideoDecoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

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
