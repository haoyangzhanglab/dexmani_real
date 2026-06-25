"""cbreak-mode keyboard listener — single-key non-blocking input.

Replaces the pynput + multiprocessing.Queue pattern with termios cbreak
mode + select.select — no extra process/thread, no queue overhead.

Unified 6-key mapping:
    B  → BEGIN          (IDLE → TELEOP + auto-recording)
    C  → PAUSE          (TELEOP ⇄ PAUSED)
    S  → STOP           (stop recording → auto-save → IDLE)
    H  → HOME           (return to home)
    Q  → QUIT           (IDLE→quit; teleop→auto-save→IDLE)
    ESC → EMERGENCY_STOP

Context-overloaded: BEGIN merges teleop start + recording (always together);
QUIT serves dual role (stop→auto-save vs idle→exit).

Ref: T-Rex main_teleop.py EpisodeKeyListener — termios.tcgetattr /
     tty.setcbreak + select.select single-key response pattern.
"""

from __future__ import annotations

import select
import sys
import termios
import tty
from enum import Enum


class ControlSignal(Enum):
    BEGIN = "BEGIN"
    PAUSE = "PAUSE"
    STOP = "STOP"
    HOME = "HOME"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    QUIT = "QUIT"


# Key → ControlSignal mapping (lowercase)
_KEY_MAP: dict[str, ControlSignal] = {
    "b": ControlSignal.BEGIN,
    "c": ControlSignal.PAUSE,
    "s": ControlSignal.STOP,
    "h": ControlSignal.HOME,
    "q": ControlSignal.QUIT,
}


class KeyboardHandler:
    """cbreak-mode non-blocking keyboard handler.

    Uses termios cbreak + select.select for single-key response without
    requiring Enter.  Restores terminal settings on exit via context
    manager protocol (__enter__ / __exit__) or explicit stop().

    Compatible with the existing controller API:
        handler = KeyboardHandler()
        handler.start()     # enter cbreak mode, save terminal settings
        ...
        signals = handler.poll(timeout=0.05)   # non-blocking signal drain
        ...
        handler.stop()      # restore terminal settings

    The ``queue`` parameter is accepted (and ignored) for backward compatibility
    with callers that still pass a multiprocessing.Queue.
    """

    def __init__(self, queue: object = None) -> None:
        """Initialize keyboard handler.

        Args:
            queue: Ignored (backward compat with multiprocessing.Queue pattern).
        """
        self._old_settings: list | None = None
        self._stdin_fd: int = sys.stdin.fileno()
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enter cbreak mode and start capturing keystrokes.

        Idempotent: calling on an already-started handler is a no-op.
        Safe no-op when stdin is not a TTY (e.g. headless / piped input).
        """
        if self._running:
            return
        if not sys.stdin.isatty():
            self._running = False
            return
        self._old_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        self._running = True

    def stop(self) -> None:
        """Restore terminal settings and stop capturing.

        Idempotent: calling on an already-stopped handler is a no-op.
        Safe to call from finally blocks.  Safe when stdin is not a TTY.
        """
        if not self._running:
            return
        try:
            if self._old_settings is not None and sys.stdin.isatty():
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_settings)
        except (termios.error, OSError):
            pass  # stdin may already be closed
        finally:
            self._running = False
            self._old_settings = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "KeyboardHandler":
        if sys.stdin.isatty():
            self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll(self, timeout: float = 0.05) -> list[ControlSignal]:
        """Non-blocking poll for pending keystrokes.

        Drains all available input from stdin and returns a list of
        ControlSignal values (may be empty).  Returns empty list when
        stdin is not a TTY.

        Args:
            timeout: Seconds to wait for input (default 0.05s).
                     Use 0 for completely non-blocking poll.
        """
        if not sys.stdin.isatty():
            return []
        signals: list[ControlSignal] = []
        while True:
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if not r:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            # Interpret the character
            sig = self._interpret(ch)
            if sig is not None:
                signals.append(sig)
            # After first character, switch to immediate poll for remaining
            timeout = 0.0
        return signals

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret(ch: str) -> ControlSignal | None:
        """Interpret a single character as a ControlSignal.

        Handles:
          - Regular keys (t, r, c, s, h, q, backtick)
          - ESC (\x1b)

        Returns None for unrecognized characters.
        """
        if ch == "\x1b":  # ESC
            return ControlSignal.EMERGENCY_STOP
        lower = ch.lower()
        return _KEY_MAP.get(lower)
