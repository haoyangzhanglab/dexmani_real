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
import sys
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
    """

    def __init__(self, debounce_s: float = 0.5) -> None:
        """Initialize keyboard handler.

        Args:
            debounce_s: Per-signal debounce interval in seconds (default 0.5).
                        Suppresses X11/Wayland auto-repeat for the same key.
        """
        self._buffer: deque[ControlSignal] = deque()
        self._lock = threading.Lock()
        self._listener: Any = None  # pynput.keyboard.Listener
        self._running: bool = False
        self._debounce_s = float(debounce_s)
        self._last_signal_time: dict[ControlSignal, float] = {}
        self._saved_termios: list | None = None  # saved terminal attributes for echo restore

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

        # Suppress terminal echo so keys captured by pynput don't also
        # appear as stray characters on stdout (interleaving with print()
        # output).  pynput's ``suppress=True`` requires the uinput kernel
        # module + write access to /dev/uinput on Linux; termios is more
        # portable (no extra permissions) and equally effective.
        self._suppress_terminal_echo()

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

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
        except Exception:
            self._restore_terminal_echo()  # roll back echo suppression on failure
            raise

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
            self._restore_terminal_echo()

    # ------------------------------------------------------------------
    # Terminal echo suppression
    # ------------------------------------------------------------------

    def _suppress_terminal_echo(self) -> None:
        """Disable terminal ECHO so pynput-captured keys don't echo to stdout.

        pynput captures at the X11/Wayland level but does not suppress the
        event from also reaching the terminal driver.  In cooked mode the
        TTY echoes every character, which interleaves with ``print()``
        output — particularly visible during blocking operations like
        ``do_return_home`` where the user holds H and gets dozens of stray
        'h' characters mixed with progress lines.

        Only ECHO and ECHONL are cleared; canonical mode (line editing,
        backspace) is preserved.  Falls back silently when stdin is not a
        TTY (e.g. piped input or headless SSH).
        """
        if not sys.stdin.isatty():
            return
        try:
            import termios

            fd = sys.stdin.fileno()
            self._saved_termios = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            # lflags[3]: clear ECHO (character echo) and ECHONL (echo
            # newline in canonical mode when ICANON is off).
            new[3] = new[3] & ~(termios.ECHO | termios.ECHONL)
            termios.tcsetattr(fd, termios.TCSADRAIN, new)
        except (termios.error, OSError):
            self._saved_termios = None  # permission denied or not a TTY

    def _restore_terminal_echo(self) -> None:
        """Restore terminal attributes saved by :meth:`_suppress_terminal_echo`."""
        if self._saved_termios is None:
            return
        try:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved_termios)
        except (termios.error, OSError):
            pass
        self._saved_termios = None

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

    def drain_signal(self, target: ControlSignal | None) -> int:
        """Remove all occurrences of *target* from the buffer, preserving others.

        Use to suppress auto-repeat of a trigger signal after a blocking
        operation, without discarding unrelated signals the user may have
        pressed during the block.

        Args:
            target: The signal to remove.  ``None`` is a no-op (returns 0).

        Returns:
            Number of signals removed.
        """
        if target is None:
            return 0
        with self._lock:
            old_len = len(self._buffer)
            self._buffer = deque(s for s in self._buffer if s != target)
            return old_len - len(self._buffer)


class GlobalKeyState:
    """Non-blocking key-hold tracker (pynput, thread-safe).

    Tracks which keys are **currently held down** via on_press/on_release.
    Use for continuous key detection (WASD / arrow keys) where polling
    "is this key down right now?" matters, as opposed to the event-queue
    model of :class:`KeyboardHandler`.

    Usage::

        keys = GlobalKeyState()
        keys.start()
        ...
        if keys.is_pressed("w"):
            move_forward()
        ...
        keys.stop()
    """

    def __init__(self) -> None:
        self._keys: set[str] = set()
        self._running = True
        self._thread: threading.Thread | None = None
        self._listener: Any = None  # pynput keyboard.Listener

    def _run(self) -> None:
        from pynput import keyboard

        def on_press(key: object) -> None:
            try:
                if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                    self._keys.add(key.char.lower())  # type: ignore[union-attr]
                elif key == keyboard.Key.esc:
                    self._keys.add("esc")
                elif key == keyboard.Key.up:
                    self._keys.add("up")
                elif key == keyboard.Key.down:
                    self._keys.add("down")
                elif key == keyboard.Key.left:
                    self._keys.add("left")
                elif key == keyboard.Key.right:
                    self._keys.add("right")
            except Exception:
                pass

        def on_release(key: object) -> None:
            try:
                if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                    self._keys.discard(key.char.lower())  # type: ignore[union-attr]
                elif key == keyboard.Key.esc:
                    self._keys.discard("esc")
                elif key == keyboard.Key.up:
                    self._keys.discard("up")
                elif key == keyboard.Key.down:
                    self._keys.discard("down")
                elif key == keyboard.Key.left:
                    self._keys.discard("left")
                elif key == keyboard.Key.right:
                    self._keys.discard("right")
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        while self._running:
            time.sleep(0.1)
        self._listener.stop()
        self._listener = None

    def stop(self) -> None:
        self._running = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_pressed(self, key: str) -> bool:
        return key in self._keys

    @property
    def any_pressed(self) -> bool:
        return len(self._keys) > 0
