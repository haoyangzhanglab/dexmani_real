"""pynput keyboard listener → multiprocessing.Queue for control signals."""

from __future__ import annotations

import multiprocessing
import queue
import threading
from enum import Enum


class ControlSignal(Enum):
    TELEOP = "T"
    RECORD = "R"
    STOP = "S"
    HOME = "H"
    EMERGENCY_STOP = "ESC"
    REARM = "X"
    QUIT = "Q"


# Key mapping
_KEY_MAP = {
    "t": ControlSignal.TELEOP,
    "r": ControlSignal.RECORD,
    "s": ControlSignal.STOP,
    "h": ControlSignal.HOME,
    "x": ControlSignal.REARM,
    "q": ControlSignal.QUIT,
}


class KeyboardHandler:
    """Non-blocking keyboard listener.

    Starts a daemon thread that listens for key presses via pynput and
    pushes ControlSignal values into a multiprocessing.Queue.  The main
    thread consumes signals via poll().
    """

    def __init__(self, queue: multiprocessing.Queue) -> None:
        self._queue: multiprocessing.Queue = queue
        self._listener: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    def stop(self) -> None:
        self._running = False

    def poll(self) -> list[ControlSignal]:
        """Drain the queue and return all pending signals (non-blocking)."""
        signals: list[ControlSignal] = []
        while True:
            try:
                signals.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return signals

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _listen(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            print("[KeyboardHandler] pynput not installed; keyboard controls disabled.")
            self._running = False
            return

        def on_press(key):
            if not self._running:
                return False  # stop listener
            try:
                if hasattr(key, "char") and key.char is not None:
                    ch = key.char.lower()
                else:
                    ch = str(key)
            except (OSError, ValueError):
                return True

            if ch == "esc" or (hasattr(key, "name") and key.name == "esc"):
                self._queue.put(ControlSignal.EMERGENCY_STOP)
            else:
                signal = _KEY_MAP.get(ch)
                if signal is not None:
                    self._queue.put(signal)
            return True

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
