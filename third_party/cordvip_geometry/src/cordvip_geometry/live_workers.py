"""One source owner, one model owner and bounded mailboxes; shutdown waits for native calls."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

from .rgbd import Frame


@dataclass(frozen=True)
class CloseStatus:
    stopped: bool
    error: str | None

    @property
    def pending(self) -> bool:
        return not self.stopped


@dataclass(frozen=True)
class SourceSnapshot:
    frame: Frame | None
    version: int
    finished: bool
    error: str | None


def format_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def join_thread(thread: threading.Thread, timeout_s: float) -> bool:
    if not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError("Shutdown timeout must be finite and nonnegative")
    if thread.ident is not None:
        thread.join(timeout_s)
    return not thread.is_alive()


class LatestFrameReader:
    """Own the source iterator and retain its latest immutable frame.
    Offline replay may pause on frame one until registration; live sources keep advancing."""

    def __init__(
        self,
        source_factory: Callable[[], Iterator[Frame]],
        *,
        replay_interval_s: float = 0.0,
        wait_on_first: bool = False,
    ) -> None:
        if not math.isfinite(replay_interval_s) or replay_interval_s < 0:
            raise ValueError("Replay interval must be finite and nonnegative")
        self._source_factory = source_factory
        self._replay_interval_s = replay_interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._resume = threading.Event()
        if not wait_on_first:
            self._resume.set()
        self._frame = None
        self._version = 0
        self._finished = False
        self._error = None
        self._thread = threading.Thread(target=self.run, name="cordvip-camera", daemon=True)

    def start(self) -> LatestFrameReader:
        self._thread.start()
        return self

    def snapshot(self) -> SourceSnapshot:
        with self._lock:
            return SourceSnapshot(self._frame, self._version, self._finished, self._error)

    def resume(self) -> None:
        """Release an offline first-frame hold after a valid registration."""
        self._resume.set()

    def close(self, timeout_s: float = 1.5) -> CloseStatus:
        self._stop.set()
        self._resume.set()
        stopped = join_thread(self._thread, timeout_s)
        with self._lock:
            return CloseStatus(stopped, self._error)

    def run(self) -> None:
        source = None
        try:
            if self._stop.is_set():
                return
            source = iter(self._source_factory())
            deadline = None
            while not self._stop.is_set():
                try:
                    frame = next(source)
                except StopIteration:
                    break
                if deadline is not None:
                    deadline += self._replay_interval_s
                    if self._stop.wait(max(0.0, deadline - time.monotonic())):
                        break
                if self._stop.is_set():
                    break
                with self._lock:
                    self._frame = frame
                    self._version += 1
                if deadline is None:
                    self._resume.wait()
                    deadline = time.monotonic()
        except Exception as error:
            with self._lock:
                self._error = format_error(error)
        finally:
            # Generator.close() must run on its owner after next() has returned.
            if source is not None and hasattr(source, "close"):
                try:
                    source.close()
                except Exception as error:
                    with self._lock:
                        message = "source cleanup: " + format_error(error)
                        self._error = f"{self._error}; {message}" if self._error else message
            with self._lock:
                self._finished = True


@dataclass(frozen=True)
class SceneStatus:
    ready: bool
    busy: bool
    error: str | None


@dataclass(frozen=True)
class SceneResult:
    operation: str
    frame: Frame
    value: Any
    error: str | None


class SceneWorker:
    """Own SAM IPC, poses and one result mailbox; close revokes results before owner-thread cleanup."""

    def __init__(self, segmenter, tracker_factory):
        self.segmenter = segmenter
        self._factory = tracker_factory
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ready = self._running = False
        self._pending = self._result = self._error = None
        self.metadata = None
        self._thread = threading.Thread(target=self.run, name="cordvip-scene", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def snapshot(self):
        with self._lock:
            return SceneStatus(self._ready, self._running or self._pending is not None, self._error)

    def submit(self, operation, frame, *args, **kwargs):
        if operation not in {"predict", "step"}:
            raise ValueError("Scene worker accepts only predict or step")
        if kwargs.get("initial_masks") is not None:
            kwargs["initial_masks"] = {
                name: mask.copy() for name, mask in kwargs["initial_masks"].items()
            }
        with self._lock:
            if (
                self._stop.is_set()
                or not self._ready
                or self._running
                or self._pending is not None
                or self._result is not None
            ):
                return False
            self._pending = operation, frame, args, kwargs
            self._wake.set()
            return True

    def poll(self):
        with self._lock:
            result, self._result = self._result, None
            return result

    def close(self, timeout_s=1.5):
        self._stop.set()
        with self._lock:
            self._result = self._pending = None
        self.segmenter.cancel()
        self._wake.set()
        stopped = join_thread(self._thread, timeout_s)
        with self._lock:
            return CloseStatus(stopped, self._error)

    def run(self):
        tracker = None
        started = time.monotonic_ns()
        try:
            self.segmenter.start()
            sam_load_metrics = dict(self.segmenter.last_metrics)
            if self._stop.is_set():
                return
            tracker = self._factory(self.segmenter)
            self.metadata = dict(
                segmentation=self.segmenter.provenance,
                sam_load_metrics=sam_load_metrics,
                pose={name: item.provenance for name, item in tracker.trackers.items()},
                model_load_ms=(time.monotonic_ns() - started) / 1e6,
            )
            with self._lock:
                self._ready = not self._stop.is_set()
            while not self._stop.is_set():
                self._wake.wait()
                if self._stop.is_set():
                    break
                with self._lock:
                    task, self._pending = self._pending, None
                    self._running = task is not None
                    self._wake.clear()
                if task is None:
                    continue
                operation, frame, args, kwargs = task
                error_text = value = None
                try:
                    if operation == "predict":
                        frame_key = dict(
                            source_id=frame.source_id,
                            row_index=frame.row_index,
                            clock=frame.clock,
                            timestamp_ns=frame.timestamp_ns,
                        )
                        value = self.segmenter.predict(
                            frame.rgb, *args, frame_key=frame_key, **kwargs
                        )
                    else:
                        value = tracker.step(frame, *args, **kwargs)
                except Exception as error:
                    error_text = format_error(error)
                with self._lock:
                    self._running = False
                    if not self._stop.is_set():
                        self._result = SceneResult(operation, frame, value, error_text)
        except Exception as error:
            with self._lock:
                if not self._stop.is_set():
                    self._error = format_error(error)
        finally:
            try:
                if tracker is not None:
                    tracker.close()
                else:
                    self.segmenter.close()
            except Exception as error:
                with self._lock:
                    self._error = format_error(error)
            with self._lock:
                self._ready = self._running = False
                self._pending = None
