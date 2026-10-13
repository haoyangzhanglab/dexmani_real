"""Encode and decode H.264 MP4 sidecars for HDF5 episodes."""

from __future__ import annotations

import logging

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class VideoEncoderConfig:
    """H.264 settings that preserve full chroma and favor encoding speed."""

    codec: str = "libx264"
    crf: int = 18
    preset: str = "ultrafast"
    pixel_format: str = "yuv444p"


class VideoEncoder:
    """Writer-thread MP4 encoder; open before START readiness or on first write, close to finalize."""

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
        self._width = width
        self._height = height
        self._rate = Fraction(str(fps)).limit_denominator(1_000_000)

        self._container: av.container.OutputContainer | None = None
        self._stream: Any = None
        self._frame_count: int = 0
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def write_frame(self, frame: np.ndarray) -> None:
        """Encode one uint8 RGB (height, width, 3) frame in caller-supplied display order."""
        if self._closed:
            raise RuntimeError("VideoEncoder is closed")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) uint8 RGB frame, got shape {frame.shape}")
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError(
                f"Frame shape {(frame.shape[0], frame.shape[1])} does not match "
                f"encoder size ({self._height}, {self._width})"
            )

        self.open()

        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        if self._cfg.pixel_format != "rgb24":
            av_frame = av_frame.reformat(format=self._cfg.pixel_format)
        av_frame.pts = self._frame_count
        av_frame.time_base = 1 / self._rate

        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

        self._frame_count += 1

    def close(self) -> None:
        """Flush and finalize the MP4 container; repeated calls are safe."""
        if self._closed:
            return
        if self._container is None:
            self._closed = True
            return
        try:
            if self._stream is not None:
                for packet in self._stream.encode(None):
                    self._container.mux(packet)
        finally:
            # Attempt file closure even after codec failure; retain a failed close's handle.
            self._stream = None
            self._container.close()
            self._container = None
            self._closed = True
        logger.debug(
            "VideoEncoder closed: %s (%d frames)",
            self._path.name,
            self._frame_count,
        )

    def open(self) -> None:
        """Open and initialize the encoder on its sole owning thread."""
        if self._closed:
            raise RuntimeError("VideoEncoder is closed")
        if self._container is None:
            self._container = av.open(str(self._path), "w", format="mp4")
            self._stream = self._container.add_stream(
                self._cfg.codec, rate=self._rate
            )
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = self._cfg.pixel_format
            self._stream.options = {
                "crf": str(self._cfg.crf),
                "preset": self._cfg.preset,
            }
            # Reserve CPU capacity for control and inference.
            self._stream.codec_context.thread_count = 1
            self._stream.codec_context.open()
            self._container.start_encoding()


class VideoDecoder:
    """Decode RGB by ordinal for HDF5 row alignment; video PTS is not the Raw sampling clock."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._container: av.container.InputContainer | None = None
        self._stream: Any = None

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Rewind and yield RGB frames sequentially without caching the video."""
        self._open()
        self._container.seek(0)
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    yield frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
            self._stream = None

    def _open(self) -> None:
        if self._stream is not None:
            return
        if self._container is not None:
            self.close()
        try:
            self._container = av.open(str(self._path), "r")
            self._stream = self._container.streams.video[0]
        except BaseException:
            self.close()
            raise
