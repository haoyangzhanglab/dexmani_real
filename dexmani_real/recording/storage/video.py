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
        self.path = Path(path)
        self.width = width
        self.height = height
        self.rate = Fraction(str(fps)).limit_denominator(1_000_000)

        self.container: av.container.OutputContainer | None = None
        self.stream: Any = None
        self.encoded_frames: int = 0
        self.closed = False

    @property
    def frame_count(self) -> int:
        return self.encoded_frames

    def write_frame(self, frame: np.ndarray) -> None:
        """Encode one uint8 RGB (height, width, 3) frame in caller-supplied display order."""
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv444p")
        av_frame.pts = self.encoded_frames
        av_frame.time_base = 1 / self.rate

        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)

        self.encoded_frames += 1

    def close(self) -> None:
        """Flush and finalize the MP4 container; repeated calls are safe."""
        if self.closed:
            return
        if self.container is None:
            self.closed = True
            return
        try:
            if self.stream is not None:
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
        finally:
            # Attempt file closure even after codec failure; retain a failed close's handle.
            self.stream = None
            self.container.close()
            self.container = None
            self.closed = True
        logger.debug(
            "VideoEncoder closed: %s (%d frames)",
            self.path.name,
            self.encoded_frames,
        )

    def open(self) -> None:
        """Open and initialize the encoder on its sole owning thread."""
        if self.closed:
            raise RuntimeError("VideoEncoder is closed")
        if self.container is None:
            self.container = av.open(str(self.path), "w", format="mp4")
            self.stream = self.container.add_stream("libx264", rate=self.rate)
            self.stream.width = self.width
            self.stream.height = self.height
            self.stream.pix_fmt = "yuv444p"
            self.stream.options = {
                "crf": "18",
                "preset": "ultrafast",
            }
            # Reserve CPU capacity for control and inference.
            self.stream.codec_context.thread_count = 1
            self.stream.codec_context.open()
            self.container.start_encoding()
