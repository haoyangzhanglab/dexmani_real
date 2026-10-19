"""Writer-thread H.264 MP4 sidecar for a Raw episode."""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

logger = logging.getLogger(__name__)


class VideoEncoder:
    """Writer-thread MP4 encoder; open before START readiness, close to finalize."""

    def __init__(self, path: Path, *, fps: float, width: int, height: int) -> None:
        self._path = Path(path)
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
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv444p")
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
            self._stream = self._container.add_stream("libx264", rate=self._rate)
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = "yuv444p"
            self._stream.options = {
                "crf": "18",
                "preset": "ultrafast",
            }
            # Reserve CPU capacity for control and inference.
            self._stream.codec_context.thread_count = 1
            self._stream.codec_context.open()
            self._container.start_encoding()
