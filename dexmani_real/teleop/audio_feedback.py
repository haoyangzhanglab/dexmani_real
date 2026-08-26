"""Non-blocking voice prompt player for teleop state transitions.

Plays pre-recorded .wav files via system audio (aplay/paplay) in a daemon
thread so the configured control loop is never blocked. Only one prompt plays at
a time — a new play() call cancels any in-progress playback.

Events are short keys (e.g. "begin", "save") that map to filenames under
assets/audio/.  Missing player or file degrades silently to a log warning.
"""

from __future__ import annotations

__all__ = ["AudioFeedback", "update_motion_gate"]

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
    "emergency": "意外的事情出现了.wav",
    "quit": "准备退出遥操作.wav",
    "quit_save_prompt": "已退出，是否需要保存轨迹.wav",
    "calibrated": "轴向已标定.wav",
    "end": "操作结束.wav",
}

_PLAYER_FALLBACK_MAX_RUNTIME_S = 0.25
_STDERR_LOG_LIMIT = 512


@dataclass(frozen=True)
class _AudioRequest:
    event: str
    path: str
    generation: int


def update_motion_gate(
    *,
    audio_playing: bool,
    begin_deadline_s: float | None,
    ignore_begin_until_silent: bool,
    now_s: float,
) -> tuple[bool, float | None, bool]:
    """Bound how long the begin cue may hold robot motion."""
    begin_active = begin_deadline_s is not None and now_s < begin_deadline_s
    if begin_deadline_s is not None and not begin_active:
        ignore_begin_until_silent = audio_playing
        begin_deadline_s = None

    if ignore_begin_until_silent:
        if not audio_playing:
            ignore_begin_until_silent = False
        should_hold = False
    else:
        should_hold = begin_active or audio_playing
    return should_hold, begin_deadline_s, ignore_begin_until_silent


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
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="audio-feedback",
        )
        self._worker.start()

    @property
    def is_playing(self) -> bool:
        """True if a voice prompt is currently playing or queued."""
        with self._condition:
            return self._active_request is not None or bool(self._pending)

    @property
    def worker_alive(self) -> bool:
        """Whether the serialization worker is alive."""
        return self._worker.is_alive()

    def _ensure_worker_locked(self) -> None:
        """Restart an unexpectedly terminated worker while holding the condition."""
        if self._worker.is_alive() or self._closed:
            return
        logger.warning("Audio worker was not alive; restarting")
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
            if self._closed:
                logger.warning("Audio event ignored after close: %s", event)
                return
            self._generation += 1
            self._pending.clear()
            self._terminate_current_locked()
            self._pending.append(_AudioRequest(event, path, self._generation))
            self._ensure_worker_locked()
            logger.debug("Audio queued: event=%s mode=play", event)
            self._condition.notify_all()

    def queue(self, event: str) -> None:
        """Queue a voice prompt to play after the current one finishes.

        Unlike :meth:`play`, which cancels any in-progress playback and
        plays immediately, ``queue`` appends the event to a sequential
        playback queue.  If nothing is currently playing, the queued
        event starts immediately.

        Usage::

            audio.play("calibrated")   # starts playing immediately
            audio.queue("begin")       # plays after "calibrated" finishes
        """
        path = self._event_path(event)
        if path is None:
            return

        with self._condition:
            if self._closed:
                logger.warning("Audio event ignored after close: %s", event)
                return
            self._pending.append(_AudioRequest(event, path, self._generation))
            self._ensure_worker_locked()
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
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                request = self._pending.pop(0)
                self._active_request = request
            try:
                self._play_one(request)
            except Exception:
                # Keep the permanent worker alive after an unexpected player failure.
                logger.warning(
                    "Audio worker recovered from unexpected failure: event=%s",
                    request.event,
                    exc_info=True,
                )
            finally:
                with self._condition:
                    if self._active_request is request:
                        self._active_request = None
                    self._condition.notify_all()

    def _play_one(self, request: _AudioRequest) -> None:
        players = _find_players()
        if not players:
            logger.warning("Audio failed: event=%s reason=no-player", request.event)
            return

        for player_index, player in enumerate(players):
            with self._condition:
                if request.generation != self._generation or self._closed:
                    logger.debug(
                        "Audio cancelled before start: event=%s", request.event
                    )
                    return
            cmd = (
                [player, "-q", request.path]
                if player == "aplay"
                else [player, request.path]
            )
            proc: subprocess.Popen | None = None
            started_s = time.monotonic()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                with self._condition:
                    if request.generation != self._generation or self._closed:
                        proc.kill()
                        proc.communicate()
                        logger.debug(
                            "Audio cancelled during start: event=%s", request.event
                        )
                        return
                    self._current_proc = proc
                logger.info(
                    "Audio started: event=%s player=%s pid=%s",
                    request.event,
                    player,
                    getattr(proc, "pid", "?"),
                )
                _stdout, stderr = proc.communicate()
                runtime_s = time.monotonic() - started_s
                returncode = int(proc.returncode or 0)
            except Exception:
                logger.warning(
                    "Audio launch/playback failed: event=%s player=%s",
                    request.event,
                    player,
                    exc_info=True,
                )
                runtime_s = time.monotonic() - started_s
                returncode = -1
                stderr = b""
            finally:
                with self._condition:
                    if proc is not None and self._current_proc is proc:
                        self._current_proc = None
                    self._condition.notify_all()

            with self._condition:
                cancelled = request.generation != self._generation or self._closed
            if cancelled:
                logger.info("Audio cancelled: event=%s", request.event)
                return
            if returncode == 0:
                logger.info("Audio completed: event=%s", request.event)
                return

            stderr_text = _stderr_text(stderr)
            logger.warning(
                "Audio failed: event=%s player=%s returncode=%d runtime=%.3fs stderr=%s",
                request.event,
                player,
                returncode,
                runtime_s,
                stderr_text or "<empty>",
            )
            has_fallback = player_index + 1 < len(players)
            if not has_fallback or runtime_s > _PLAYER_FALLBACK_MAX_RUNTIME_S:
                return
            logger.info(
                "Audio retrying with fallback: event=%s next_player=%s",
                request.event,
                players[player_index + 1],
            )


def _stderr_text(stderr: bytes | str | None) -> str:
    if stderr is None:
        return ""
    text = (
        stderr.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes)
        else str(stderr)
    )
    return " ".join(text.strip().split())[:_STDERR_LOG_LIMIT]


def _find_players() -> list[str]:
    """Return available system players in fallback order."""
    return [name for name in ("aplay", "paplay") if shutil.which(name)]
