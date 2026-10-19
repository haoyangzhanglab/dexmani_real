"""Off-thread WAV prompts preempt queued audio; unavailable players disable playback."""

from __future__ import annotations

import logging

__all__ = ["AudioFeedback"]

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from dexmani_real import ASSET_DIR

logger = logging.getLogger(__name__)

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
class AudioRequest:
    path: str
    generation: int


class AudioFeedback:
    """Nonblocking voice prompts; interrupt=False appends after existing playback."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._current_proc: subprocess.Popen | None = None
        self._active_request: AudioRequest | None = None
        self._pending: list[AudioRequest] = []
        self._generation = 0
        self._closed = False
        self._player = next((name for name in ("aplay", "paplay") if shutil.which(name)), None)
        self._disabled = self._player is None
        if self._disabled:
            logger.warning("No audio player; audio disabled")
        self._worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,
            name="audio-feedback",
        )
        self._worker.start()

    def event_path(self, event: str) -> str | None:
        filename = _EVENT_MAP.get(event)
        if filename is None:
            logger.warning("Unknown audio event: %s", event)
            return None
        path = os.path.join(_AUDIO_DIR, filename)
        if not os.path.isfile(path):
            logger.warning("Audio file not found: %s", path)
            return None
        return path

    def play(self, event: str, *, interrupt: bool = True) -> None:
        """Submit a prompt without waiting for playback or process creation."""
        with self._condition:
            if self._closed or self._disabled:
                return
        path = self.event_path(event)
        if path is None:
            return

        # An immediate cue supersedes queued prompts; the worker checks generation.
        with self._condition:
            if self._closed or self._disabled:
                return
            if interrupt:
                self._generation += 1
                self._pending.clear()
                self.terminate_current_locked()
            self._pending.append(AudioRequest(path, self._generation))
            logger.debug("Audio queued: event=%s interrupt=%s", event, interrupt)
            self._condition.notify_all()

    def wait_until_idle(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._active_request is not None or self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._pending.clear()
            self.terminate_current_locked()
            self._condition.notify_all()
        self._worker.join(timeout=1.0)
        if self._worker.is_alive():
            logger.warning("Audio worker did not stop within 1s")

    def terminate_current_locked(self) -> None:
        """Request current playback termination while holding ``_condition``."""
        proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.kill()
        except Exception:
            logger.warning("Audio cancel: process kill failed", exc_info=True)

    def worker_loop(self) -> None:
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
                self.play_one(request)
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

    def play_one(self, request: AudioRequest) -> None:
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
                detail = " ".join(stderr.decode("utf-8", errors="replace").split())[
                    :_STDERR_LOG_LIMIT
                ]
                raise RuntimeError(f"{player} exited {proc.returncode}: {detail}")
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
