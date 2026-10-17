"""Shared pynput-based operator keyboard input for interactive robot workflows."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)

# stop() disables callbacks, but Linux/XRecord may remain blocked; avoid long waits.
_LISTENER_STOP_TIMEOUT_S = 0.25
_LISTENER_START_TIMEOUT_S = 2.0

__all__ = [
    "OperatorCommand",
    "KeyboardInput",
]


class OperatorCommand(Enum):
    BEGIN = "BEGIN"
    PAUSE = "PAUSE"
    STOP = "STOP"
    DISCARD = "DISCARD"
    HOME = "HOME"
    RESET_TACTILE = "RESET_TACTILE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    QUIT = "QUIT"


_KEY_MAP: dict[str, OperatorCommand] = {
    "b": OperatorCommand.BEGIN,
    "c": OperatorCommand.PAUSE,
    "s": OperatorCommand.STOP,
    "d": OperatorCommand.DISCARD,
    "h": OperatorCommand.HOME,
    "t": OperatorCommand.RESET_TACTILE,
    "q": OperatorCommand.QUIT,
}

_EVENT_ONLY_KEYS = frozenset({"x", "space", "enter", "backspace"})


def _suppress_terminal_echo() -> list[Any] | None:
    """Disable terminal ECHO; returns saved termios attrs or None on failure."""
    if not sys.stdin.isatty():
        return None
    try:
        import termios

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] = new[3] & ~(termios.ECHO | termios.ECHONL)
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        return saved
    except (termios.error, OSError):
        return None


def _restore_terminal_echo(saved: list[Any] | None) -> None:
    """Restore terminal attributes and discard canonical input typed meanwhile."""
    if saved is None:
        return
    try:
        import termios

        # TCSAFLUSH discards terminal input accumulated alongside pynput events.
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, saved)
    except (termios.error, OSError):
        logger.warning("terminal echo restoration failed", exc_info=True)


def _stop_listener_bounded(listener: Any) -> bool:
    """Request pynput shutdown without treating a slow daemon join as a fault."""
    if listener is None:
        return True
    try:
        listener.stop()
        listener.join(timeout=_LISTENER_STOP_TIMEOUT_S)
        alive = bool(listener.is_alive())
    except Exception:
        logger.warning("keyboard listener shutdown raised", exc_info=True)
        return False
    if alive:
        # Restore the terminal even if pynput's daemon remains blocked after stop().
        logger.debug(
            "keyboard listener backend did not confirm exit within %.1fs; "
            "terminal input will still be restored and flushed",
            _LISTENER_STOP_TIMEOUT_S,
        )
        return False
    return True


class KeyboardInput:
    """One global listener for command edges, held keys, and optional raw events."""

    def __init__(
        self,
        *,
        estop_callback: Callable[[], None] | None = None,
        stop_callback: Callable[[], None] | None = None,
        quit_callback: Callable[[], None] | None = None,
        capture_commands: bool = True,
        capture_raw_events: bool = False,
    ) -> None:
        self._buffer: deque[OperatorCommand] = deque()
        self._events: deque[str] = deque()
        self._keys: set[str] = set()
        self._condition = threading.Condition(threading.Lock())
        self._listener: Any = None
        self._running: bool = False
        # A daemon can outlive shutdown; disable its callbacks before releasing the session.
        self._callbacks_active: bool = False
        self._estop_callback = estop_callback
        self._stop_callback = stop_callback
        self._quit_callback = quit_callback
        self._capture_commands = bool(capture_commands)
        self._capture_raw_events = bool(capture_raw_events)
        self._estop_latched = threading.Event()
        self._listener_failure_reported = False
        self._saved_termios: list | None = None

    def _latch_emergency_stop(self) -> None:
        callback: Callable[[], None] | None = None
        with self._condition:
            if not self._callbacks_active:
                return
            if self._estop_latched.is_set():
                return
            self._estop_latched.set()
            self._buffer.append(OperatorCommand.EMERGENCY_STOP)
            self._condition.notify()
            callback = self._estop_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.error("keyboard emergency-stop callback failed", exc_info=True)

    def _dispatch_immediate_callback(self, signal: OperatorCommand) -> bool:
        """Complete the immediate effect before the command becomes pollable."""
        with self._condition:
            if not self._callbacks_active:
                return False
            callback = (
                self._stop_callback
                if signal is OperatorCommand.STOP
                else self._quit_callback
                if signal is OperatorCommand.QUIT
                else None
            )
        if callback is None:
            return True
        try:
            callback()
        except Exception:
            logger.error("keyboard %s callback failed", signal.value, exc_info=True)
            # Consumers must not proceed as if a failed stop/quit was applied.
            self._latch_emergency_stop()
            return False
        return True

    def _callbacks(
        self, keyboard: Any
    ) -> tuple[Callable[[object], None], Callable[[object], None]]:
        """Build pynput callbacks that preserve held, raw-event, and command modes."""

        def key_name(key: object) -> str | None:
            if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                return key.char.lower()  # type: ignore[union-attr]
            if key == keyboard.Key.esc:
                return "esc"
            if key == keyboard.Key.up:
                return "up"
            if key == keyboard.Key.down:
                return "down"
            if key == keyboard.Key.left:
                return "left"
            if key == keyboard.Key.right:
                return "right"
            if key == keyboard.Key.space:
                return "space"
            if key == keyboard.Key.enter:
                return "enter"
            if key == keyboard.Key.backspace:
                return "backspace"
            return None

        def on_press(key: object) -> None:
            try:
                name = key_name(key)
                if name is None:
                    return
                with self._condition:
                    if not self._callbacks_active:
                        return
                    if name in _EVENT_ONLY_KEYS:
                        if self._capture_raw_events:
                            self._events.append(name)
                        return
                    # Held keys also deduplicate command edges across OS repeat.
                    first_press = name not in self._keys
                    self._keys.add(name)
                if name == "esc":
                    self._latch_emergency_stop()
                    return
                signal = _KEY_MAP.get(name)
                if (
                    (
                        self._capture_commands
                        or (signal is OperatorCommand.QUIT and self._quit_callback is not None)
                    )
                    and signal is not None
                    and first_press
                ):
                    if self._dispatch_immediate_callback(signal):
                        with self._condition:
                            if self._capture_commands and self._callbacks_active:
                                self._buffer.append(signal)
                                self._condition.notify()
            except Exception:
                logger.warning("keyboard press callback failed", exc_info=True)

        def on_release(key: object) -> None:
            try:
                name = key_name(key)
                with self._condition:
                    if self._callbacks_active:
                        self._keys.discard(name)
            except Exception:
                logger.warning("keyboard release callback failed", exc_info=True)

        return on_press, on_release

    def start(self) -> None:
        """Start the daemon keyboard listener once."""
        if self._running:
            return
        if self._listener is not None:
            raise RuntimeError(
                "keyboard listener is retained from a prior shutdown; "
                "refusing to start a replacement"
            )

        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError(
                "KeyboardInput requires a graphical session (pynput global "
                "capture uses X11/Wayland). No DISPLAY/WAYLAND_DISPLAY set — "
                "headless/SSH is not supported; run on the workstation desktop."
            )

        try:
            from pynput import keyboard  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pynput is required for global keyboard capture. Install with: pip install pynput"
            ) from None

        self._saved_termios = _suppress_terminal_echo()
        self._estop_latched.clear()
        self._listener_failure_reported = False
        with self._condition:
            self._buffer.clear()
            self._events.clear()
            self._keys.clear()
            self._callbacks_active = True
        on_press, on_release = self._callbacks(keyboard)

        try:
            self._listener = keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
                suppress=False,
            )
            self._listener.start()
            listener = self._listener
            ready = threading.Event()
            startup_errors: list[BaseException] = []

            def wait_until_ready() -> None:
                try:
                    listener.wait()
                except BaseException as exc:
                    startup_errors.append(exc)
                finally:
                    ready.set()

            threading.Thread(
                target=wait_until_ready, name="keyboard-events-ready", daemon=True
            ).start()
            if not ready.wait(timeout=_LISTENER_START_TIMEOUT_S):
                raise RuntimeError(
                    f"keyboard listener startup timed out after {_LISTENER_START_TIMEOUT_S:.1f}s"
                )
            if startup_errors:
                raise RuntimeError("keyboard listener failed during startup") from startup_errors[0]
            if not self._listener.is_alive():
                raise RuntimeError("keyboard listener exited during startup")
        except BaseException:
            self.stop()
            raise

        self._running = True

    def stop(self) -> None:
        """Stop the listener idempotently for bounded cleanup."""
        listener = self._listener
        with self._condition:
            self._callbacks_active = False
            self._running = False
            self._buffer.clear()
            self._condition.notify_all()
        try:
            if _stop_listener_bounded(listener):
                self._listener = None
        finally:
            with self._condition:
                self._events.clear()
                self._keys.clear()
            self._estop_latched.clear()
            _restore_terminal_echo(self._saved_termios)
            self._saved_termios = None

    def poll(self, timeout: float = 0.05) -> list[OperatorCommand]:
        """Drain commands, waiting at most timeout seconds; zero is nonblocking and may return an empty list."""
        if self._running and not self.healthy:
            if not self._listener_failure_reported:
                logger.error("keyboard listener exited; latching emergency stop")
                self._listener_failure_reported = True

        with self._condition:
            if timeout > 0:
                self._condition.wait_for(lambda: self._buffer or not self._running, timeout)
            signals = list(self._buffer)
            self._buffer.clear()
            return signals

    @property
    def healthy(self) -> bool:
        """Whether the listener that owns emergency-stop input is alive."""
        listener = self._listener
        try:
            healthy = bool(self._running and listener is not None and listener.is_alive())
        except Exception:
            logger.error("keyboard listener health check failed", exc_info=True)
            healthy = False
        if not healthy:
            with self._condition:
                self._keys.clear()
            if self._running:
                # SDK fences also check keyboard health; losing emergency-stop input must fail safe.
                self._latch_emergency_stop()
        return healthy

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched.is_set()

    def is_pressed(self, key: str) -> bool:
        """Return whether a key is currently held, with sticky ESC semantics."""
        if key == "esc" and self._estop_latched.is_set():
            return True
        with self._condition:
            return key in self._keys

    def pressed_keys(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(sorted(self._keys))

    def pop_event(self) -> str | None:
        with self._condition:
            return self._events.popleft() if self._events else None

    def drain_signal(self, target: OperatorCommand) -> None:
        """Remove stale commands after blocking operations, preserving other requests."""
        with self._condition:
            self._buffer = deque(s for s in self._buffer if s != target)
