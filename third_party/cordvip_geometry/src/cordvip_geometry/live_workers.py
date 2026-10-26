"""One source owner, one model owner and bounded mailboxes; shutdown waits for native calls."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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


def wait_worker(worker, heartbeat, *, loading=False):
    while True:
        status = worker.snapshot()
        if status.error:
            raise RuntimeError(status.error)
        if loading and status.ready:
            return
        if not loading:
            result = worker.poll()
            if result is not None:
                if result.error:
                    raise RuntimeError(result.error)
                return result
        heartbeat()
        time.sleep(0.005)


class LatestFrameReader:
    """独占数据源并保留最新帧；离线首帧可等待注册，实时源持续推进。"""

    def __init__(
        self,
        source_factory: Callable[[], Iterator[Frame]],
        *,
        replay_interval_s: float = 0.0,
        wait_on_first: bool = False,
    ) -> None:
        if not math.isfinite(replay_interval_s) or replay_interval_s < 0:
            raise ValueError("Replay interval must be finite and nonnegative")
        self.source_factory = source_factory
        self.replay_interval_s = replay_interval_s
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        if not wait_on_first:
            self.resume_event.set()
        self.frame = None
        self.version = 0
        self.finished = False
        self.error = None
        self.thread = threading.Thread(target=self.run, name="cordvip-camera", daemon=True)

    def start(self) -> LatestFrameReader:
        self.thread.start()
        return self

    def snapshot(self) -> SourceSnapshot:
        with self.lock:
            return SourceSnapshot(self.frame, self.version, self.finished, self.error)

    def resume(self) -> None:
        """Release an offline first-frame hold after a valid registration."""
        self.resume_event.set()

    def close(self, timeout_s: float = 1.5) -> CloseStatus:
        self.stop_event.set()
        self.resume_event.set()
        stopped = join_thread(self.thread, timeout_s)
        with self.lock:
            return CloseStatus(stopped, self.error)

    def run(self) -> None:
        source = None
        try:
            if self.stop_event.is_set():
                return
            source = iter(self.source_factory())
            deadline = None
            while not self.stop_event.is_set():
                try:
                    frame = next(source)
                except StopIteration:
                    break
                if deadline is not None:
                    deadline += self.replay_interval_s
                    if self.stop_event.wait(max(0.0, deadline - time.monotonic())):
                        break
                if self.stop_event.is_set():
                    break
                with self.lock:
                    self.frame = frame
                    self.version += 1
                if deadline is None:
                    self.resume_event.wait()
                    deadline = time.monotonic()
        except Exception as error:
            with self.lock:
                self.error = format_error(error)
        finally:
            # Generator.close() must run on its owner after next() has returned.
            if source is not None and hasattr(source, "close"):
                try:
                    source.close()
                except Exception as error:
                    with self.lock:
                        message = "source cleanup: " + format_error(error)
                        self.error = f"{self.error}; {message}" if self.error else message
            with self.lock:
                self.finished = True


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
    """独占模型与单槽结果；关闭先撤销结果，再由模型线程释放资源。"""

    def __init__(self, segmenter, tracker_factory):
        self.segmenter = segmenter
        self.tracker_factory = tracker_factory
        self.lock = threading.Lock()
        self.wake_event = threading.Event()
        self.stop_event = threading.Event()
        self.ready = self.busy = False
        self.pending_task = self.result = self.error = None
        self.metadata = None
        self.thread = threading.Thread(target=self.run, name="cordvip-scene", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def snapshot(self):
        with self.lock:
            return SceneStatus(self.ready, self.busy or self.pending_task is not None, self.error)

    def submit(self, operation, frame, *args, **kwargs):
        if operation not in {"predict", "step"}:
            raise ValueError("Scene worker accepts only predict or step")
        if kwargs.get("initial_masks") is not None:
            kwargs["initial_masks"] = {
                name: mask.copy() for name, mask in kwargs["initial_masks"].items()
            }
        with self.lock:
            if (
                self.stop_event.is_set()
                or not self.ready
                or self.busy
                or self.pending_task is not None
                or self.result is not None
            ):
                return False
            self.pending_task = operation, frame, args, kwargs
            self.wake_event.set()
            return True

    def poll(self):
        with self.lock:
            result, self.result = self.result, None
            return result

    def close(self, timeout_s=1.5):
        self.stop_event.set()
        with self.lock:
            self.result = self.pending_task = None
        self.segmenter.cancel()
        self.wake_event.set()
        stopped = join_thread(self.thread, timeout_s)
        with self.lock:
            return CloseStatus(stopped, self.error)

    def run(self):
        tracker = None
        started = time.monotonic_ns()
        try:
            self.segmenter.start()
            sam_load_metrics = dict(self.segmenter.last_metrics)
            if self.stop_event.is_set():
                return
            tracker = self.tracker_factory(self.segmenter)
            self.metadata = dict(
                segmentation=self.segmenter.provenance,
                sam_load_metrics=sam_load_metrics,
                pose={name: item.provenance for name, item in tracker.trackers.items()},
                model_load_ms=(time.monotonic_ns() - started) / 1e6,
            )
            with self.lock:
                self.ready = not self.stop_event.is_set()
            while not self.stop_event.is_set():
                self.wake_event.wait()
                if self.stop_event.is_set():
                    break
                with self.lock:
                    task, self.pending_task = self.pending_task, None
                    self.busy = task is not None
                    self.wake_event.clear()
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
                with self.lock:
                    self.busy = False
                    if not self.stop_event.is_set():
                        self.result = SceneResult(operation, frame, value, error_text)
        except Exception as error:
            with self.lock:
                if not self.stop_event.is_set():
                    self.error = format_error(error)
        finally:
            try:
                if tracker is not None:
                    tracker.close()
                else:
                    self.segmenter.close()
            except Exception as error:
                with self.lock:
                    self.error = format_error(error)
            with self.lock:
                self.ready = self.busy = False
                self.pending_task = None
