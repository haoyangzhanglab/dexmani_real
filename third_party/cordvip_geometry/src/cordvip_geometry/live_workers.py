"""Two bounded workers for the standalone viewer; importing never opens hardware.

The camera generator and the CUDA estimator each have exactly one owner thread.
The GUI reads a latest-frame mailbox and submits at most one inference at a time.
Closing requests cooperative shutdown; a CUDA call or SDK read already in progress
cannot safely be interrupted from the GUI thread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np

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


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _join(thread: threading.Thread, timeout_s: float) -> bool:
    if not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError("Shutdown timeout must be finite and nonnegative")
    if thread.ident is not None:
        thread.join(timeout_s)
    return not thread.is_alive()


class LatestFrameReader:
    """Consume one camera/replay iterator continuously, retaining only its latest frame.

    ``source_factory`` is invoked by the owner thread, including camera connection.
    A positive ``replay_interval_s`` paces offline replay against deadlines, including
    decoding time. ``wait_on_first`` holds the first frame until ``resume`` after
    registration; it is never enabled for a live camera. Frame arrays must not
    be mutated by their producer after yield, or by snapshot consumers.
    """

    def __init__(
        self, source_factory: Callable[[], Iterator[Frame]], *, replay_interval_s: float = 0.0,
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
        self._thread = threading.Thread(target=self._run, name="cordvip-camera", daemon=True)

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
        stopped = _join(self._thread, timeout_s)
        with self._lock:
            return CloseStatus(stopped, self._error)

    def _run(self) -> None:
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
                    if self._stop.wait(max(0., deadline - time.monotonic())):
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
                self._error = _error_text(error)
        finally:
            # Generator.close() must run on its owner after next() has returned.
            if source is not None and hasattr(source, "close"):
                try:
                    source.close()
                except Exception as error:
                    with self._lock:
                        message = "source cleanup: " + _error_text(error)
                        self._error = f"{self._error}; {message}" if self._error else message
            with self._lock:
                self._finished = True


@dataclass(frozen=True)
class PoseStatus:
    ready: bool
    busy: bool
    registered: bool
    generation: int
    error: str | None


@dataclass(frozen=True)
class PoseResult:
    generation: int
    frame: Frame
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    operation: str
    mask: np.ndarray | None
    pose: np.ndarray | None
    started_ns: int
    completed_ns: int
    error: str | None


@dataclass(frozen=True)
class _PoseTask:
    generation: int
    frame: Frame
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    mask: np.ndarray | None


class PoseWorker:
    """Load and call a single FoundationPose instance outside the GUI thread.

    A supplied mask means registration; otherwise submission requires a successful
    registration in the current generation. ``reset`` invalidates prior work without
    racing its CUDA calls. Poll each result before submitting another frame; the
    worker never accumulates a frame queue or silently replaces unconsumed results.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ready = False
        self._running = False
        self._generation = 0
        self._registered_generation = None
        self._pending = None
        self._result = None
        self._error = None
        self._thread = threading.Thread(target=self._run, name="cordvip-pose", daemon=True)

    def start(self) -> PoseWorker:
        self._thread.start()
        return self

    def snapshot(self) -> PoseStatus:
        with self._lock:
            return PoseStatus(
                ready=self._ready,
                busy=self._running or self._pending is not None,
                registered=self._registered_generation == self._generation,
                generation=self._generation,
                error=self._error,
            )

    def submit(
        self,
        frame: Frame,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        *,
        mask: np.ndarray | None = None,
    ) -> bool:
        """Return false while loading/busy/closed or before the prior result is polled.

        The GUI retains ownership of frame/image arrays and must keep them immutable
        after submission. The editable selection mask is copied on submission.
        """
        with self._lock:
            if (
                self._stop.is_set() or not self._ready or self._running
                or self._pending is not None or self._result is not None
            ):
                return False
            if mask is None and self._registered_generation != self._generation:
                raise ValueError("Register an object in this session before tracking")
            if mask is not None:
                self._registered_generation = None
            self._pending = _PoseTask(
                self._generation, frame, rgb, depth_m, K,
                None if mask is None else np.asarray(mask).copy(),
            )
            self._wake.set()
            return True

    def poll(self) -> PoseResult | None:
        with self._lock:
            result, self._result = self._result, None
            return result

    def reset(self) -> int:
        """Revoke the old pose now; the next accepted task must be registration."""
        with self._lock:
            self._generation += 1
            self._registered_generation = None
            self._pending = None
            self._result = None
            self._wake.clear()
            return self._generation

    def close(self, timeout_s: float = 1.5) -> CloseStatus:
        self._stop.set()
        self._wake.set()
        stopped = _join(self._thread, timeout_s)
        with self._lock:
            return CloseStatus(stopped, self._error)

    def _run(self) -> None:
        tracker = None
        try:
            if self._stop.is_set():
                return
            tracker = self._factory()
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
                operation = "register" if task.mask is not None else "track"
                started_ns = time.monotonic_ns()
                pose = error_text = None
                try:
                    if task.mask is not None:
                        pose = tracker.register(task.rgb, task.depth_m, task.K, task.mask)
                    else:
                        pose = tracker.track(task.rgb, task.depth_m, task.K)
                    # Upstream may retain/mutate its estimate on the next call.
                    pose = np.asarray(pose).copy()
                except Exception as error:
                    error_text = _error_text(error)
                completed_ns = time.monotonic_ns()
                with self._lock:
                    self._running = False
                    if task.generation == self._generation and not self._stop.is_set():
                        self._registered_generation = task.generation if error_text is None else None
                        self._result = PoseResult(
                            task.generation, task.frame, task.rgb, task.depth_m, task.K,
                            operation, task.mask, pose, started_ns, completed_ns, error_text,
                        )
        except Exception as error:
            with self._lock:
                self._error = _error_text(error)
        finally:
            # A future estimator cleanup, like construction/inference, stays here.
            if tracker is not None and hasattr(tracker, "close"):
                try:
                    tracker.close()
                except Exception as error:
                    with self._lock:
                        message = "pose cleanup: " + _error_text(error)
                        self._error = f"{self._error}; {message}" if self._error else message
            with self._lock:
                self._ready = False
                self._running = False
                self._pending = None
                self._registered_generation = None
