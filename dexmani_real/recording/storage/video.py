"""Encode and decode H.264 MP4 sidecars for HDF5 episodes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
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
    """Write MP4 frames sequentially on the episode writer thread.

    open() initializes the codec before START readiness; standalone callers may
    still open on first write. close() finalizes the MP4 for playback.
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

        # Convert RGB to the encoder pixel format; PyAV performs RGB→YUV.
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        if self._cfg.pixel_format != "rgb24":
            av_frame = av_frame.reformat(format=self._cfg.pixel_format)
        av_frame.pts = self._frame_count
        av_frame.time_base = 1 / Fraction(str(self._fps)).limit_denominator(1_000_000)

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
            # Even a codec flush error must release its file. If close itself
            # fails, keep the handle visible for writer-thread cleanup retry.
            self._stream = None
            self._container.close()
            self._container = None
            self._closed = True
        logger.debug(
            "VideoEncoder closed: %s (%d frames)",
            self._path.name,
            self._frame_count,
        )

    def __enter__(self) -> "VideoEncoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def open(self) -> None:
        """Open and initialize the encoder on its sole owning thread."""
        if self._closed:
            raise RuntimeError("VideoEncoder is closed")
        if self._container is None:
            self._container = av.open(str(self._path), "w", format="mp4")
            self._stream = self._container.add_stream(
                self._cfg.codec, rate=Fraction(str(self._fps)).limit_denominator(1_000_000)
            )
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = self._cfg.pixel_format
            self._stream.options = {
                "crf": str(self._cfg.crf),
                "preset": self._cfg.preset,
            }

            self._stream.codec_context.open()
            self._container.start_encoding()


class VideoDecoder:
    """Stream RGB frames or decode one frame by index without a full-video cache."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._container: av.container.InputContainer | None = None
        self._stream: Any = None  # av.video.VideoStream — PyAV stubs are incomplete
        self._frame_count: int = 0
        self._rate = None
        self._origin_time = None
        self._origin_time_base = None
        self._opened = False

    @property
    def frame_count(self) -> int:
        """Total frame count in the video (cached on first open)."""
        if not self._opened:
            self._open()
        return self._frame_count

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Rewind and yield RGB frames sequentially without caching the video."""
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
        """Decode one indexed frame; use iter_frames() for sequential access."""
        if not self._opened:
            self._open()
        if self._container is None:
            raise RuntimeError("VideoDecoder: container is None after _open()")
        if index < 0 or index >= self._frame_count:
            raise IndexError(f"Frame index {index} out of range [0, {self._frame_count})")

        time_base = self._stream.time_base
        target_time = self._origin_time + Fraction(index, 1) / self._rate
        pts = target_time // time_base
        self._container.seek(pts, stream=self._stream, backward=True, any_frame=False)
        previous = None
        for frame in self._container.decode(self._stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("CFR frame is missing its presentation timestamp")
            ordinal = (frame.pts * frame.time_base - self._origin_time) * self._rate
            # Each PTS (including origin) may have half a time-base tick of
            # quantization error. Ambiguous adjacent identities are unsupported.
            tolerance = (frame.time_base + self._origin_time_base) * self._rate / 2
            nearest = round(ordinal)
            if tolerance >= Fraction(1, 2) or abs(ordinal - nearest) > tolerance:
                raise ValueError("presentation timestamp cannot identify a unique CFR frame")
            if previous is not None and nearest <= previous:
                raise ValueError("duplicate or unordered CFR presentation timestamp")
            previous = nearest
            if nearest == index:
                return frame.to_ndarray(format="rgb24")
            if nearest > index:
                raise ValueError(f"CFR decode skipped requested frame {index}")
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
        if self._container is not None:
            self.close()
        try:
            self._container = av.open(str(self._path), "r")
            self._stream = self._container.streams.video[0]
            self._frame_count = int(self._stream.frames) if self._stream.frames > 0 else 0
            avg_rate = self._stream.average_rate
            if avg_rate is None:
                raise ValueError(f"No average frame rate in {self._path}")
            self._rate = Fraction(avg_rate)
            if self._rate <= 0 or self._stream.time_base is None:
                raise ValueError("CFR video requires a positive rate and a time base")
            first = next(self._container.decode(self._stream), None)
            if first is None or first.pts is None or first.time_base is None:
                raise ValueError("CFR video requires a first presentation timestamp")
            self._origin_time = first.pts * first.time_base
            self._origin_time_base = first.time_base
            self._opened = True
        except BaseException:
            self.close()
            raise
