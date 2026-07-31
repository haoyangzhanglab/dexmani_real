"""AsyncEpisodeSaver — background episode flush worker.

Wraps :class:`~dexmani_real.recording.episode_recorder.EpisodeRecorder` so
that ``stop_episode()`` + ``join_stop()`` runs on a daemon thread, freeing
the main control loop to start the next episode or perform other work while
the previous episode's HDF5 data is still being written.

Usage in an entry point::

    recorder = EpisodeRecorder(...)
    saver = AsyncEpisodeSaver(recorder)

    # ... record an episode ...
    recorder.stop_episode(success=True)
    saver.submit()               # queues join_stop() on background thread

    # Immediately start the next episode (start_episode waits for the
    # previous daemon only if it's still flushing — usually it's done).
    recorder.start_episode()

    # Before exit: wait for all pending saves.
    saver.close()

Pattern adapted from ``lerobot_robot_ufactory`` (AsyncEpisodeSaver in
``uf_lerobot_record.py``).
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.recording.episode_recorder import EpisodeRecorder

logger = get_logger(__name__)

_STOP_SENTINEL = object()


class AsyncEpisodeSaver:
    """Background worker that drains ``join_stop()`` calls from a queue.

    The recorder's ``stop_episode()`` already spawns a daemon for the heavy
    HDF5 flush and returns immediately.  This saver moves the follow-up
    ``join_stop()`` (which waits for that daemon to finish) onto a background
    thread so the main loop doesn't block on I/O between episodes.

    Parameters:
        recorder: An :class:`EpisodeRecorder` instance that has already been
                  constructed (but not necessarily started).
    """

    def __init__(self, recorder: EpisodeRecorder) -> None:
        self._recorder = recorder
        self._queue: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="async-episode-saver",
            daemon=True,
        )
        self._thread.start()

    # ── Public API ──

    def submit(self) -> None:
        """Queue a background ``join_stop()`` for the current episode.

        Must be called **after** ``recorder.stop_episode()``.  The saver
        calls ``recorder.join_stop()`` on its background thread and captures
        any error for later inspection via :meth:`raise_if_failed`.

        Raises:
            RuntimeError: if a previous background save failed.
        """
        self.raise_if_failed()
        self._queue.put(True)

    def wait_idle(self) -> None:
        """Block until all previously submitted saves have finished."""
        self._queue.join()
        self.raise_if_failed()

    def close(self) -> None:
        """Wait for pending saves, then shut down the background thread."""
        self._queue.join()
        self._queue.put(_STOP_SENTINEL)
        self._queue.join()
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        """Re-raise any exception caught by the background thread."""
        if self._error is not None:
            raise RuntimeError(
                "AsyncEpisodeSaver: background save failed"
            ) from self._error

    # ── Internal ──

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP_SENTINEL:
                    return
                # item is True → one pending join_stop()
                try:
                    self._recorder.join_stop()
                except Exception as exc:
                    logger.error(
                        "AsyncEpisodeSaver: join_stop failed: %s", exc
                    )
                    self._error = exc
            finally:
                self._queue.task_done()
