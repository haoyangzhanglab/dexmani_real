"""RecordingSession — driver-agnostic, thread-owned episode recorder.

A single writer thread owns the full episode lifecycle (start → record →
stop/flush), so START/FRAME/STOP are processed in order on one queue and the
HDF5 file is touched by exactly one thread.  This removes the cross-thread
teardown race that dropped an episode's trailing frames when stop() ran on the
main thread while a separate writer thread was still draining queued frames.

Any driver (teleop controller, future policy-inference loop) feeds the same
session, so recorded episodes are byte-compatible regardless of what produced
them.  Mirrors ManiUniCon's falling-edge dump pattern (core/robot.py), where
the thread that fills the buffer is the one that dumps it.
"""

from __future__ import annotations

__all__ = ["RecordingSession"]

import queue
import threading
from typing import Any

from dexmani_real.recording.collection_loop import CollectionLoop
from dexmani_real.recording.data_validator import DataValidator
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Control message kinds
_START = "start"
_FRAME = "frame"
_STOP = "stop"
_SHUTDOWN = "shutdown"


class RecordingSession:
    """Owns a queue + single writer thread wrapping a CollectionLoop."""

    def __init__(
        self,
        collection_loop: CollectionLoop,
        *,
        validate: bool = False,
        max_queue: int = 2000,
        stop_timeout_s: float = 10.0,
    ) -> None:
        self._loop = collection_loop
        self._validate = validate
        self._stop_timeout_s = stop_timeout_s
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, name="recording_session", daemon=True)
        self._thread.start()

    # ── Public API (called by any driver) ──

    def start(self, meta: dict[str, Any]) -> None:
        """Begin a new episode; ``meta`` is forwarded as start_episode(**meta)."""
        self._queue.put((_START, meta))

    def record(self, bundle: dict[str, Any]) -> None:
        """Enqueue one frame bundle (forwarded as record_frame(**bundle))."""
        try:
            self._queue.put_nowait((_FRAME, bundle))
        except queue.Full:
            logger.warning("RecordingSession queue full — dropping frame")

    def stop(self, save: bool) -> str | None:
        """Stop the current episode; block (bounded) until flushed. Returns path.

        Because STOP is queued after every FRAME and drained by the same thread,
        no trailing frame is lost and the file is fully written before return.
        """
        done = threading.Event()
        box: dict[str, Any] = {"path": None}
        self._queue.put((_STOP, (save, done, box)))
        if not done.wait(timeout=self._stop_timeout_s):
            logger.error("RecordingSession.stop timed out after %.1fs", self._stop_timeout_s)
            return None
        return box["path"]

    def shutdown(self) -> None:
        """Stop the writer thread (does not finalize an in-progress episode)."""
        self._queue.put((_SHUTDOWN, None))
        self._thread.join(timeout=self._stop_timeout_s)

    # ── Writer thread ──

    def _run(self) -> None:
        while True:
            kind, payload = self._queue.get()
            if kind == _SHUTDOWN:
                break
            try:
                if kind == _START:
                    self._loop.start_episode(**payload)
                elif kind == _FRAME:
                    self._loop.record_frame(**payload)
                elif kind == _STOP:
                    self._handle_stop(payload)
            except (ValueError, OSError, KeyError, RuntimeError) as e:
                logger.exception("RecordingSession writer error on %s: %s", kind, e)

    def _handle_stop(self, payload: tuple) -> None:
        save, done, box = payload
        try:
            if save:
                path = self._loop.stop_episode(success=True, reason="manual")
                if path and self._validate:
                    report = DataValidator().validate(path)
                    if not report.is_valid:
                        logger.warning("Episode failed validation: %s", path)
                box["path"] = path
            else:
                # stop first (closes the file), then unlink — avoids deleting an open file.
                self._loop.stop_episode(success=False, reason="discard")
                self._loop.discard_episode()
        finally:
            done.set()
