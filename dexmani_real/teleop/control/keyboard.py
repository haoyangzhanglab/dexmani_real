"""pynput-based keyboard handler — global key capture, no terminal focus required.

Replaces the termios cbreak + select.select pattern with pynput.keyboard.Listener
running in a daemon thread.  Key events are captured globally (even when the
terminal window is NOT focused), stored in a thread-safe buffer, and drained by
poll() on the main thread.

7-key mapping:
    B  → BEGIN             (IDLE → TELEOP + auto-recording)
    C  → PAUSE             (TELEOP ⇄ PAUSED)
    S  → STOP              (stop recording → auto-save → IDLE)
    D  → DISCARD           (stop recording → discard → IDLE)
    H  → HOME              (stop recording → return to home → IDLE)
    Q  → QUIT              (always quit; auto-save if recording)
    ESC → EMERGENCY_STOP
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from enum import Enum
from typing import Any


class ControlSignal(Enum):
    BEGIN = "BEGIN"
    PAUSE = "PAUSE"
    STOP = "STOP"
    DISCARD = "DISCARD"
    HOME = "HOME"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    QUIT = "QUIT"


# Character key → ControlSignal mapping (lowercase)
_KEY_MAP: dict[str, ControlSignal] = {
    "b": ControlSignal.BEGIN,
    "c": ControlSignal.PAUSE,
    "s": ControlSignal.STOP,
    "d": ControlSignal.DISCARD,
    "h": ControlSignal.HOME,
    "q": ControlSignal.QUIT,
}


class KeyboardHandler:
    """Global keyboard handler using pynput.

    Captures keystrokes globally — works even when the terminal window
    does not have focus (e.g. user interacting with the SAPIEN viewer).

    Compatible with the existing API:
        handler = KeyboardHandler()
        handler.start()                    # start global listener thread
        signals = handler.poll(timeout=0.05)  # drain pending signals
        handler.stop()                     # stop listener, restore state

    Context manager:
        with KeyboardHandler() as kb:
            ...
            for sig in kb.poll(timeout=0.0):
                ...

    The ``queue`` parameter is accepted (and ignored) for backward compatibility
    with callers that still pass a multiprocessing.Queue.
    """

    def __init__(self, queue: object = None, debounce_s: float = 0.5) -> None:
        """Initialize keyboard handler.

        Args:
            queue: Ignored (backward compat with multiprocessing.Queue pattern).
            debounce_s: Per-signal debounce interval in seconds (default 0.5).
                        Suppresses X11/Wayland auto-repeat for the same key.
        """
        self._buffer: deque[ControlSignal] = deque()
        self._lock = threading.Lock()
        self._listener: Any = None  # pynput.keyboard.Listener
        self._running: bool = False
        self._debounce_s = float(debounce_s)
        self._last_signal_time: dict[ControlSignal, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the global pynput keyboard listener in a daemon thread.

        Idempotent: calling on an already-started handler is a no-op.
        """
        if self._running:
            return

        # Headless guard: without a display, pynput's Listener dies with an
        # obscure Xlib/uinput error deep in its backend. Fail fast and clear.
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError(
                "KeyboardHandler requires a graphical session (pynput global "
                "capture uses X11/Wayland). No DISPLAY/WAYLAND_DISPLAY set — "
                "headless/SSH is not supported; run on the workstation desktop."
            )

        try:
            from pynput import keyboard  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("pynput is required for global keyboard capture. " "Install with: pip install pynput")

        def on_press(key: object) -> None:
            try:
                if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                    sig = _KEY_MAP.get(key.char.lower())  # type: ignore[union-attr]
                elif key == keyboard.Key.esc:
                    sig = ControlSignal.EMERGENCY_STOP
                else:
                    return
                if sig is not None:
                    # Per-signal debounce — suppress X11/Wayland auto-repeat
                    # (holding a key fires repeated press events at ~30-60 Hz).
                    # ESC always bypasses debounce (emergency stop).
                    if sig != ControlSignal.EMERGENCY_STOP:
                        now = time.perf_counter()
                        last = self._last_signal_time.get(sig, 0.0)
                        if now - last < self._debounce_s:
                            return
                        self._last_signal_time[sig] = now
                    with self._lock:
                        self._buffer.append(sig)
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        self._running = True

    def stop(self) -> None:
        """Stop the pynput listener.

        Idempotent: calling on an already-stopped handler is a no-op.
        Safe to call from finally blocks.
        """
        if not self._running:
            return
        try:
            if self._listener is not None:
                self._listener.stop()
                self._listener = None
        except Exception:
            pass
        finally:
            self._running = False
            with self._lock:
                self._buffer.clear()
            self._last_signal_time.clear()

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "KeyboardHandler":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll(self, timeout: float = 0.05) -> list[ControlSignal]:
        """Drain all pending ControlSignals from the keyboard buffer.

        Args:
            timeout: Seconds to wait if buffer is empty (default 0.05s).
                     Use 0 for completely non-blocking poll.

        Returns:
            List of ControlSignal values (may be empty).
        """
        # Fast path: drain buffer under lock
        with self._lock:
            if self._buffer:
                signals = list(self._buffer)
                self._buffer.clear()
                return signals

        # Slow path: buffer empty, wait briefly for events
        if timeout > 0:
            import time

            deadline = time.perf_counter() + timeout
            while time.perf_counter() < deadline:
                with self._lock:
                    if self._buffer:
                        signals = list(self._buffer)
                        self._buffer.clear()
                        return signals
                time.sleep(0.005)  # 5 ms polling granularity
        return []
