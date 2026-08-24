"""Bounded background writer for grid-aligned camera episode streams."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from dexmani_real.recording.video import VideoEncoder, VideoEncoderConfig
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class CameraStreamWriterError(RuntimeError):
    """Fatal writer error; the enclosing episode must be discarded."""


@dataclass(frozen=True)
class CameraStreamWriterConfig:
    """Shapes, rate, and bounded buffering for one episode writer."""

    rgb_shape: tuple[int, int, int]
    depth_shape: tuple[int, int]
    fps: float
    queue_size: int
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)

    def __post_init__(self) -> None:
        if len(self.rgb_shape) != 3 or self.rgb_shape[2] != 3:
            raise ValueError(f"rgb_shape must be (H, W, 3), got {self.rgb_shape}")
        if len(self.depth_shape) != 2 or self.depth_shape != self.rgb_shape[:2]:
            raise ValueError("depth_shape must match the RGB image plane")
        if self.fps <= 0 or self.queue_size <= 0:
            raise ValueError("writer fps and queue_size must be > 0")


@dataclass(frozen=True)
class _CameraWriteItem:
    rgb: np.ndarray
    depth: np.ndarray


class CameraStreamWriter:
    """Write RGB and depth from a bounded worker queue.

    ``submit`` never waits for disk or encoding.  A full queue, worker crash,
    codec failure, or filesystem error is fatal and latched in :attr:`error`;
    callers must abort the episode instead of dropping a camera grid slot.
    """

    _SENTINEL = object()

    def __init__(
        self,
        directory: str | Path,
        config: CameraStreamWriterConfig,
        *,
        encoder_factory: Callable[..., VideoEncoder] = VideoEncoder,
    ) -> None:
        self.directory = Path(directory)
        self.config = config
        self._encoder_factory = encoder_factory
        self._queue: queue.Queue[_CameraWriteItem | object] = queue.Queue(maxsize=config.queue_size)
        self._error: str | None = None
        self._error_lock = threading.Lock()
        self._frame_count = 0
        self._queue_high_watermark = 0
        self._encode_durations_s: deque[float] = deque(maxlen=4096)
        self._hdf5_durations_s: deque[float] = deque(maxlen=4096)
        self._close_duration_s = 0.0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="camera-stream-writer", daemon=False)
        self._thread.start()

    @property
    def error(self) -> str | None:
        with self._error_lock:
            return self._error

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def metrics(self) -> dict[str, float | int]:
        def summarize(name: str, values: deque[float]) -> dict[str, float]:
            if not values:
                return {f"{name}_{suffix}_s": 0.0 for suffix in ("p50", "p95", "p99", "max")}
            array = np.asarray(values, dtype=np.float64)
            return {
                f"{name}_p50_s": float(np.percentile(array, 50)),
                f"{name}_p95_s": float(np.percentile(array, 95)),
                f"{name}_p99_s": float(np.percentile(array, 99)),
                f"{name}_max_s": float(np.max(array)),
            }

        return {
            "camera_writer_queue_high_watermark": self._queue_high_watermark,
            "camera_writer_queue_capacity": self.config.queue_size,
            "camera_writer_close_s": self._close_duration_s,
            **summarize("camera_encode", self._encode_durations_s),
            **summarize("camera_hdf5", self._hdf5_durations_s),
        }

    def submit(self, rgb: np.ndarray, depth: np.ndarray) -> bool:
        """Copy and enqueue one complete grid slot without blocking."""
        if self._closed:
            self._set_error("submit attempted after writer close")
            return False
        if self.error is not None:
            return False

        try:
            rgb_frame = self._validated_copy(rgb, np.uint8, self.config.rgb_shape, "rgb")
            depth_frame = self._validated_copy(depth, np.uint16, self.config.depth_shape, "depth")
            queue_depth_before = self._queue.qsize()
            self._queue.put_nowait(_CameraWriteItem(rgb_frame, depth_frame))
            self._queue_high_watermark = max(
                self._queue_high_watermark,
                min(self.config.queue_size, queue_depth_before + 1),
            )
            return True
        except queue.Full:
            self._set_error(f"queue full (capacity={self.config.queue_size}); no frames were dropped silently")
        except (TypeError, ValueError) as exc:
            self._set_error(str(exc))
        return False

    def close(self, timeout: float = 60.0) -> None:
        """Drain queued frames, finalize sidecars, and raise on any fatal error."""
        close_started_s = time.perf_counter()
        first_close = not self._closed
        if not self._closed:
            self._closed = True
        if first_close and self._thread.is_alive():
            if self.error is not None:
                # Bound shutdown latency after an episode is already doomed.
                while True:
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except queue.Empty:
                        break
            try:
                self._queue.put(self._SENTINEL, timeout=max(0.1, min(timeout, 5.0)))
            except queue.Full:
                self._set_error("queue remained full while closing")
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout))
            if self._thread.is_alive():
                self._set_error(f"writer did not stop within {timeout:.1f}s")

        self._close_duration_s = time.perf_counter() - close_started_s

        error = self.error
        if error is not None:
            raise CameraStreamWriterError(error)

    @staticmethod
    def _validated_copy(array: np.ndarray, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"{name} must be {shape} {np.dtype(dtype)}, got {array.shape} {array.dtype}")
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN/Inf")
        return np.ascontiguousarray(array).copy()

    def _set_error(self, message: str) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = message
                logger.error("CameraStreamWriter: %s", message)

    @staticmethod
    def _append(dataset: h5py.Dataset, frame: np.ndarray) -> None:
        index = dataset.shape[0]
        dataset.resize(index + 1, axis=0)
        dataset[index] = frame

    def _run(self) -> None:
        encoder: VideoEncoder | None = None
        depth_file: h5py.File | None = None
        try:
            height, width, _ = self.config.rgb_shape
            encoder = self._encoder_factory(
                self.directory / "rgb.mp4",
                config=self.config.video,
                fps=self.config.fps,
                width=width,
                height=height,
            )
            depth_file = h5py.File(self.directory / "depth.h5", "w")
            depth_ds = depth_file.create_dataset(
                "depth",
                shape=(0,) + self.config.depth_shape,
                maxshape=(None,) + self.config.depth_shape,
                chunks=(1,) + self.config.depth_shape,
                dtype=np.uint16,
                compression="gzip",
                compression_opts=1,
            )

            while True:
                item = self._queue.get()
                try:
                    if item is self._SENTINEL:
                        break
                    if not isinstance(item, _CameraWriteItem):
                        raise TypeError(f"unexpected queue item: {type(item).__name__}")
                    encode_started_s = time.perf_counter()
                    encoder.write_frame(item.rgb)
                    self._encode_durations_s.append(time.perf_counter() - encode_started_s)
                    hdf5_started_s = time.perf_counter()
                    self._append(depth_ds, item.depth)
                    self._hdf5_durations_s.append(time.perf_counter() - hdf5_started_s)
                    self._frame_count += 1
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._set_error(f"{type(exc).__name__}: {exc}")
            logger.error("CameraStreamWriter worker failed", exc_info=True)
        finally:
            for resource in (encoder, depth_file):
                if resource is None:
                    continue
                try:
                    resource.close()
                except Exception as exc:
                    self._set_error(f"close failed: {type(exc).__name__}: {exc}")
