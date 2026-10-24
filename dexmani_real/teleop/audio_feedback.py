"""Off-thread WAV prompts preempt queued audio; unavailable players disable playback."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from dexmani_real import ASSET_DIR

__all__ = ["AudioFeedback"]

logger = logging.getLogger(__name__)

AUDIO_DIR = str(ASSET_DIR / "audio")

EVENT_MAP: dict[str, str] = {
    "begin": "遥操作启动.wav",
    "save": "成功保存轨迹.wav",
    "discard": "放弃保存轨迹.wav",
    "home": "即将回到初始姿态.wav",
    "home_done": "已经回到初始姿态.wav",
    "end": "操作结束.wav",
    "emergency": "意外的事情出现了.wav",
    "calibrated": "轴向已标定.wav",
}

STDERR_LOG_LIMIT = 512


@dataclass(frozen=True)
class AudioRequest:
    path: str
    generation: int


class AudioFeedback:
    """Nonblocking voice prompts; interrupt=False appends after existing playback."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.current_proc: subprocess.Popen | None = None
        self.active_request: AudioRequest | None = None
        self.pending: list[AudioRequest] = []
        self.generation = 0
        self.closed = False
        self.player = next((name for name in ("aplay", "paplay") if shutil.which(name)), None)
        self.disabled = self.player is None
        if self.disabled:
            logger.warning("No audio player; audio disabled")
        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,
            name="audio-feedback",
        )
        self.worker.start()

    def event_path(self, event: str) -> str | None:
        filename = EVENT_MAP.get(event)
        if filename is None:
            logger.warning("Unknown audio event: %s", event)
            return None
        path = os.path.join(AUDIO_DIR, filename)
        if not os.path.isfile(path):
            logger.warning("Audio file not found: %s", path)
            return None
        return path

    def play(self, event: str, *, interrupt: bool = True) -> None:
        """Submit a prompt without waiting for playback or process creation."""
        with self.condition:
            if self.closed or self.disabled:
                return
        path = self.event_path(event)
        if path is None:
            return

        # An immediate cue supersedes queued prompts; the worker checks generation.
        with self.condition:
            if self.closed or self.disabled:
                return
            if interrupt:
                self.generation += 1
                self.pending.clear()
                self.terminate_current_locked()
            self.pending.append(AudioRequest(path, self.generation))
            logger.debug("Audio queued: event=%s interrupt=%s", event, interrupt)
            self.condition.notify_all()

    def wait_until_idle(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.active_request is not None or self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(timeout=remaining)
            return True

    def close(self) -> None:
        with self.condition:
            if self.closed:
                return
            self.closed = True
            self.generation += 1
            self.pending.clear()
            self.terminate_current_locked()
            self.condition.notify_all()
        self.worker.join(timeout=1.0)
        if self.worker.is_alive():
            logger.warning("Audio worker did not stop within 1s")

    def terminate_current_locked(self) -> None:
        """Request current playback termination while holding ``condition``."""
        proc = self.current_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.kill()
        except Exception:
            logger.warning("Audio cancel: process kill failed", exc_info=True)

    def worker_loop(self) -> None:
        """Serialize prompts; ``play`` generation changes preempt safely."""
        while True:
            with self.condition:
                while not self.pending and not self.closed and not self.disabled:
                    self.condition.wait()
                if self.closed or self.disabled:
                    return
                request = self.pending.pop(0)
                self.active_request = request
            try:
                self.play_one(request)
            except Exception:
                logger.warning("Audio failed; disabling playback", exc_info=True)
                with self.condition:
                    self.disabled = True
                    self.pending.clear()
            finally:
                with self.condition:
                    if self.active_request is request:
                        self.active_request = None
                    self.condition.notify_all()

    def play_one(self, request: AudioRequest) -> None:
        with self.condition:
            if request.generation != self.generation or self.closed:
                return
        player = self.player
        cmd = [player, "-q", request.path] if player == "aplay" else [player, request.path]
        proc = None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            with self.condition:
                self.current_proc = proc
                cancelled = request.generation != self.generation or self.closed
            if cancelled:
                proc.kill()
            # Never wait for a child while holding the submission lock.
            _, stderr = proc.communicate()
            with self.condition:
                cancelled = request.generation != self.generation or self.closed
            if not cancelled and proc.returncode:
                detail = " ".join(stderr.decode("utf-8", errors="replace").split())[
                    :STDERR_LOG_LIMIT
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
            with self.condition:
                if self.current_proc is proc:
                    self.current_proc = None
                self.condition.notify_all()
