"""Non-blocking voice prompt player for teleop state transitions.

Plays pre-recorded .wav files via system audio (aplay/paplay) in a daemon
thread so the 16 Hz control loop is never blocked.  Only one prompt plays at
a time — a new play() call cancels any in-progress playback.

Events are short keys (e.g. "begin", "save") that map to filenames under
assets/audio/.  Missing player or file degrades silently to a log warning.
"""

from __future__ import annotations

__all__ = ["AudioFeedback"]

import os
import shutil
import subprocess
import threading

from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_AUDIO_DIR = str(ASSET_DIR / "audio")

# Event key → audio filename (under assets/audio/)
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


class AudioFeedback:
    """Non-blocking voice prompt player.

    Usage::

        audio = AudioFeedback()
        audio.play("begin")   # starts playing, returns immediately
        audio.play("save")    # cancels "begin", starts "save"
    """

    def __init__(self, audio_dir: str | None = None) -> None:
        self._audio_dir = audio_dir or _AUDIO_DIR
        self._lock = threading.Lock()
        self._current_proc: subprocess.Popen | None = None
        self._cancel_flag: threading.Event | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        """True if a voice prompt is currently playing."""
        with self._lock:
            return self._current_proc is not None and self._current_proc.poll() is None

    def play(self, event: str) -> None:
        """Play the voice prompt for *event* (non-blocking).

        If a previous prompt is still playing it is cancelled first.
        Unknown events and missing audio files are logged and skipped.
        """
        filename = _EVENT_MAP.get(event)
        if filename is None:
            logger.warning("Unknown audio event: %s", event)
            return
        path = os.path.join(self._audio_dir, filename)
        if not os.path.isfile(path):
            logger.warning("Audio file not found: %s", path)
            return

        self._cancel_current()

        cancel = threading.Event()
        t = threading.Thread(target=self._play_thread, args=(path, cancel), daemon=True, name=f"audio-{event}")
        with self._lock:
            self._cancel_flag = cancel
        t.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cancel_current(self) -> None:
        """Kill the current subprocess and signal its thread to exit."""
        with self._lock:
            proc = self._current_proc
            self._current_proc = None
            flag = self._cancel_flag
            self._cancel_flag = None

        if flag is not None:
            flag.set()
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                logger.warning("Audio cancel: process kill failed", exc_info=True)

    def _play_thread(self, path: str, cancel: threading.Event) -> None:
        player = _find_player()
        if player is None:
            return

        # aplay -q suppresses the "Playing WAVE ..." banner
        cmd = [player, "-q", path] if player == "aplay" else [player, path]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with self._lock:
                if cancel.is_set():
                    proc.kill()
                    proc.wait()
                    return
                self._current_proc = proc

            # Busy-wait until playback finishes or cancel is signalled
            while proc.poll() is None:
                if cancel.wait(timeout=0.1):
                    proc.kill()
                    proc.wait()
                    return
        except Exception:
            logger.warning("Audio playback failed for %s", path, exc_info=True)
        finally:
            with self._lock:
                if self._current_proc is proc:
                    self._current_proc = None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _find_player() -> str | None:
    """Return the first available system audio player binary."""
    for name in ("aplay", "paplay"):
        if shutil.which(name):
            return name
    logger.warning("No system audio player found (tried aplay, paplay)")
    return None
