"""Play assets/audio/*.wav prompts without blocking the control loop.

A daemon thread uses one player selected at construction; play() preempts queued
prompts. Missing players or playback failures disable audio for the session.
Unknown events and missing files are logged and skipped.
"""

from __future__ import annotations

__all__ = ["AudioFeedback"]

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_AUDIO_DIR = str(ASSET_DIR / "audio")

_EVENT_MAP: dict[str, str] = {
    "begin": "遥操作启动.wav",
    "pause": "操作暂停.wav",
    "resume": "工作继续.wav",
    "save": "成功保存轨迹.wav",
    "discard": "放弃保存轨迹.wav",
    "home": "即将回到初始姿态.wav",
    "home_done": "已经回到初始姿态.wav",
    "quit": "准备退出遥操作.wav",
    "end": "操作结束.wav",
    "emergency": "意外的事情出现了.wav",
    "calibrated": "轴向已标定.wav",
}

_STDERR_LOG_LIMIT = 512


@dataclass(frozen=True)
class _AudioRequest:
    path: str
    generation: int


class AudioFeedback:
    """Non-blocking voice prompt player.

    Usage::

        audio = AudioFeedback()
        audio.play("begin")   # starts playing, returns immediately
        audio.play("save")    # cancels "begin", starts "save"
    """

    def __init__(self, audio_dir: str | None = None) -> None:
        self._audio_dir = audio_dir or _AUDIO_DIR
        self._condition = threading.Condition()
        self._current_proc: subprocess.Popen | None = None
        self._active_request: _AudioRequest | None = None
        self._pending: list[_AudioRequest] = []
        self._generation = 0
        self._closed = False
        self._player = next((name for name in ("aplay", "paplay") if shutil.which(name)), None)
        self._disabled = self._player is None
        if self._disabled:
            logger.warning("No audio player; audio disabled")
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="audio-feedback",
        )
        self._worker.start()

    def _event_path(self, event: str) -> str | None:
        """Resolve an audio event to an existing WAV path, logging failures."""
        filename = _EVENT_MAP.get(event)
        if filename is None:
            logger.warning("Unknown audio event: %s", event)
            return None
        path = os.path.join(self._audio_dir, filename)
        if not os.path.isfile(path):
            logger.warning("Audio file not found: %s", path)
            return None
        return path

    def play(self, event: str) -> None:
        """Play the voice prompt for *event* (non-blocking).

        If a previous prompt is still playing it is cancelled first.
        Any pending queued events are also cleared.
        Unknown events and missing audio files are logged and skipped.
        """
        path = self._event_path(event)
        if path is None:
            return

        # An immediate cue supersedes queued prompts; the worker checks generation.
        with self._condition:
            if self._closed or self._disabled:
                logger.warning("Audio event ignored: player closed or disabled (%s)", event)
                return
            self._generation += 1
            self._pending.clear()
            self._terminate_current_locked()
            self._pending.append(_AudioRequest(path, self._generation))
            logger.debug("Audio queued: event=%s mode=play", event)
            self._condition.notify_all()

    def queue(self, event: str) -> None:
        """Append a prompt after pending playback, or start immediately if idle.

        Usage::

            audio.play("calibrated")
            audio.queue("begin")
        """
        path = self._event_path(event)
        if path is None:
            return

        with self._condition:
            if self._closed or self._disabled:
                logger.warning("Audio event ignored: player closed or disabled (%s)", event)
                return
            self._pending.append(_AudioRequest(path, self._generation))
            logger.debug("Audio queued: event=%s mode=queue", event)
            self._condition.notify_all()

    def wait_until_idle(self, timeout_s: float | None = None) -> bool:
        """Wait until the active and pending prompts finish."""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("audio idle timeout must be non-negative")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while self._active_request is not None or self._pending:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self, timeout_s: float = 1.0) -> None:
        """Cancel playback and stop the daemon worker."""
        if timeout_s < 0:
            raise ValueError("audio close timeout must be non-negative")
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._pending.clear()
            self._terminate_current_locked()
            self._condition.notify_all()
        if self._worker is not threading.current_thread():
            self._worker.join(timeout=timeout_s)
        if self._worker.is_alive():
            logger.warning("Audio worker did not stop within %.2fs", timeout_s)

    def _terminate_current_locked(self) -> None:
        """Request current playback termination while holding ``_condition``."""
        proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.kill()
        except Exception:
            logger.warning("Audio cancel: process kill failed", exc_info=True)

    def _worker_loop(self) -> None:
        """Serialize prompts; ``play`` generation changes preempt safely."""
        while True:
            with self._condition:
                while not self._pending and not self._closed and not self._disabled:
                    self._condition.wait()
                if self._closed or self._disabled:
                    return
                request = self._pending.pop(0)
                self._active_request = request
            try:
                self._play_one(request)
            except Exception:
                logger.warning("Audio failed; disabling playback", exc_info=True)
                with self._condition:
                    self._disabled = True
                    self._pending.clear()
            finally:
                with self._condition:
                    if self._active_request is request:
                        self._active_request = None
                    self._condition.notify_all()

    def _play_one(self, request: _AudioRequest) -> None:
        with self._condition:
            if request.generation != self._generation or self._closed:
                return
        player = self._player
        cmd = [player, "-q", request.path] if player == "aplay" else [player, request.path]
        proc = None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            with self._condition:
                self._current_proc = proc
                cancelled = request.generation != self._generation or self._closed
            if cancelled:
                proc.kill()
            # Never wait for a child while holding the submission lock.
            _, stderr = proc.communicate()
            with self._condition:
                cancelled = request.generation != self._generation or self._closed
            if not cancelled and proc.returncode:
                raise RuntimeError(f"{player} exited {proc.returncode}: {_stderr_text(stderr)}")
        except BaseException:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    logger.warning("Audio cleanup kill failed", exc_info=True)
                try:
                    proc.communicate()
                except Exception:
                    logger.warning("Audio cleanup reap failed", exc_info=True)
            raise
        finally:
            with self._condition:
                if self._current_proc is proc:
                    self._current_proc = None
                self._condition.notify_all()


def _stderr_text(stderr: bytes | str | None) -> str:
    if stderr is None:
        return ""
    text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else str(stderr)
    return " ".join(text.strip().split())[:_STDERR_LOG_LIMIT]
