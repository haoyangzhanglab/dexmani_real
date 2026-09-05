"""pynput-based global keyboard handler for teleop control (B/C/S/D/H/Q/ESC)."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# stop() disables callbacks, but Linux/XRecord may remain blocked; avoid long waits.
_LISTENER_STOP_TIMEOUT_S = 0.25

__all__ = [
    "OperatorCommand",
    "KeyboardInput",
    "KeyboardState",
]


class OperatorCommand(Enum):
    BEGIN = "BEGIN"
    PAUSE = "PAUSE"
    STOP = "STOP"
    DISCARD = "DISCARD"
    HOME = "HOME"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    QUIT = "QUIT"


_KEY_MAP: dict[str, OperatorCommand] = {
    "b": OperatorCommand.BEGIN,
    "c": OperatorCommand.PAUSE,
    "s": OperatorCommand.STOP,
    "d": OperatorCommand.DISCARD,
    "h": OperatorCommand.HOME,
    "q": OperatorCommand.QUIT,
}


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


def _stop_listener_bounded(listener: Any, *, label: str) -> bool:
    """Request pynput shutdown without treating a slow daemon join as a fault."""
    if listener is None:
        return True
    try:
        listener.stop()
        listener.join(timeout=_LISTENER_STOP_TIMEOUT_S)
        alive = bool(listener.is_alive())
    except Exception:
        logger.warning("%s listener shutdown raised", label, exc_info=True)
        return False
    if alive:
        # pynput may remain blocked after stop(); terminal restoration is the
        # user-visible shutdown boundary.
        logger.debug(
            "%s listener backend did not confirm exit within %.1fs; "
            "terminal input will still be restored and flushed",
            label,
            _LISTENER_STOP_TIMEOUT_S,
        )
        return False
    return True


class KeyboardInput:
    """Thread-safe global keyboard events with a latched emergency stop."""

    def __init__(
        self,
        debounce_s: float = 0.0,
        *,
        estop_callback: Callable[[], None] | None = None,
        stop_callback: Callable[[], None] | None = None,
        quit_callback: Callable[[], None] | None = None,
        startup_timeout_s: float = 2.0,
    ) -> None:
        if not np.isfinite(debounce_s) or debounce_s < 0.0:
            raise ValueError("debounce_s must be finite and non-negative")
        if not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be finite and positive")
        self._buffer: deque[OperatorCommand] = deque()
        self._lock = threading.Lock()
        self._listener: Any = None  # pynput.keyboard.Listener
        self._running: bool = False
        self._debounce_s = float(debounce_s)
        self._startup_timeout_s = float(startup_timeout_s)
        self._estop_callback = estop_callback
        self._stop_callback = stop_callback
        self._quit_callback = quit_callback
        self._estop_latched = threading.Event()
        self._listener_failure_reported = False
        self._last_signal_time: dict[OperatorCommand, float] = {}
        self._pressed_signals: set[OperatorCommand] = set()
        self._saved_termios: list | None = None

    def _accept_control_press(self, signal: OperatorCommand, now_s: float) -> bool:
        """Return whether a physical key-down edge should emit *signal*.

        ``pynput`` delivers operating-system auto-repeat as repeated press
        callbacks.  A control is therefore emitted only once until its matching
        release callback arrives.  Optional time debounce remains available for
        callers that need it, but the default is edge-triggered with no added
        operator latency.
        """
        with self._lock:
            if signal in self._pressed_signals:
                return False
            self._pressed_signals.add(signal)
            last = self._last_signal_time.get(signal, float("-inf"))
            if now_s - last < self._debounce_s:
                return False
            self._last_signal_time[signal] = now_s
            self._buffer.append(signal)
            return True

    def _release_control(self, signal: OperatorCommand) -> None:
        """Mark the matching physical key as released."""
        with self._lock:
            self._pressed_signals.discard(signal)

    def _latch_emergency_stop(self) -> None:
        callback: Callable[[], None] | None = None
        with self._lock:
            if self._estop_latched.is_set():
                return
            self._estop_latched.set()
            self._buffer.append(OperatorCommand.EMERGENCY_STOP)
            callback = self._estop_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.error("keyboard emergency-stop callback failed", exc_info=True)

    def _dispatch_immediate_callback(self, signal: OperatorCommand) -> None:
        """Run the optional non-e-stop callback for a newly pressed key."""
        callback = (
            self._stop_callback
            if signal is OperatorCommand.STOP
            else self._quit_callback if signal is OperatorCommand.QUIT else None
        )
        if callback is None:
            return
        try:
            callback()
        except Exception:
            logger.error("keyboard %s callback failed", signal.value, exc_info=True)

    def start(self) -> None:
        """Start the global pynput keyboard listener in a daemon thread.

        Idempotent: calling on an already-started handler is a no-op.
        """
        if self._running:
            return

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
                "pynput is required for global keyboard capture. "
                "Install with: pip install pynput"
            ) from None

        self._suppress_terminal_echo()
        self._estop_latched.clear()
        self._listener_failure_reported = False
        with self._lock:
            self._buffer.clear()
            self._last_signal_time.clear()
            self._pressed_signals.clear()

        def control_signal(key: object) -> OperatorCommand | None:
            if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                return _KEY_MAP.get(key.char.lower())  # type: ignore[union-attr]
            return None

        def on_press(key: object) -> None:
            try:
                sig = control_signal(key)
                if sig is None and key == keyboard.Key.esc:
                    self._latch_emergency_stop()
                    return
                if sig is not None and self._accept_control_press(
                    sig, time.perf_counter()
                ):
                    self._dispatch_immediate_callback(sig)
            except Exception:
                logger.warning("keyboard press callback failed", exc_info=True)

        def on_release(key: object) -> None:
            try:
                sig = control_signal(key)
                if sig is not None:
                    self._release_control(sig)
            except Exception:
                logger.warning("keyboard release callback failed", exc_info=True)

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
            ready = threading.Event()
            startup_errors: list[BaseException] = []

            def wait_until_ready() -> None:
                try:
                    listener = self._listener
                    if listener is None:
                        raise RuntimeError(
                            "keyboard listener disappeared during startup"
                        )
                    listener.wait()
                except BaseException as exc:
                    startup_errors.append(exc)
                finally:
                    ready.set()

            threading.Thread(
                target=wait_until_ready, name="keyboard-events-ready", daemon=True
            ).start()
            if not ready.wait(timeout=self._startup_timeout_s):
                raise RuntimeError(
                    f"keyboard listener startup timed out after {self._startup_timeout_s:.1f}s"
                )
            if startup_errors:
                raise RuntimeError(
                    "keyboard listener failed during startup"
                ) from startup_errors[0]
            if not self._listener.is_alive():
                raise RuntimeError("keyboard listener exited during startup")
        except Exception:
            listener = self._listener
            try:
                if listener is not None:
                    listener.stop()
                    listener.join(timeout=1.0)
            except Exception:
                logger.warning("keyboard listener rollback failed", exc_info=True)
            try:
                listener_alive = bool(listener is not None and listener.is_alive())
            except Exception:
                logger.warning(
                    "keyboard listener rollback health check failed", exc_info=True
                )
                listener_alive = True
            if not listener_alive:
                self._listener = None
            self._restore_terminal_echo()
            raise

        self._running = True

    def stop(self) -> None:
        """Stop the pynput listener.

        Idempotent: calling on an already-stopped handler is a no-op.
        Safe to call from finally blocks.
        """
        listener = self._listener
        self._running = False
        try:
            if _stop_listener_bounded(listener, label="keyboard"):
                self._listener = None
        finally:
            with self._lock:
                self._buffer.clear()
                self._pressed_signals.clear()
            self._last_signal_time.clear()
            self._estop_latched.clear()
            self._restore_terminal_echo()

    def _suppress_terminal_echo(self) -> None:
        """Disable terminal ECHO (delegates to module-level helper)."""
        self._saved_termios = _suppress_terminal_echo()

    def _restore_terminal_echo(self) -> None:
        """Restore terminal attributes (delegates to module-level helper)."""
        _restore_terminal_echo(self._saved_termios)
        self._saved_termios = None

    def __enter__(self) -> "KeyboardInput":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def poll(self, timeout: float = 0.05) -> list[OperatorCommand]:
        """Drain all pending OperatorCommands from the keyboard buffer.

        Args:
            timeout: Seconds to wait if buffer is empty (default 0.05s).
                     Use 0 for completely non-blocking poll.

        Returns:
            List of OperatorCommand values (may be empty).
        """
        if not np.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout must be finite and non-negative")
        if self._running and not self.healthy:
            if not self._listener_failure_reported:
                logger.error("keyboard listener exited; latching emergency stop")
                self._listener_failure_reported = True
            self._latch_emergency_stop()

        with self._lock:
            if self._buffer:
                signals = list(self._buffer)
                self._buffer.clear()
                return signals

        if timeout > 0:
            deadline = time.perf_counter() + timeout
            while time.perf_counter() < deadline:
                with self._lock:
                    if self._buffer:
                        signals = list(self._buffer)
                        self._buffer.clear()
                        return signals
                time.sleep(0.005)  # 5 ms polling granularity
        return []

    @property
    def healthy(self) -> bool:
        """Whether the listener that owns emergency-stop input is alive."""
        listener = self._listener
        try:
            return bool(self._running and listener is not None and listener.is_alive())
        except Exception:
            logger.error("keyboard listener health check failed", exc_info=True)
            return False

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched.is_set()

    def drain_signal(self, target: OperatorCommand | None) -> int:
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


class KeyboardState:
    """Non-blocking key-hold tracker (pynput, thread-safe).

    Tracks which keys are **currently held down** via on_press/on_release.
    Use for continuous key detection (WASD / arrow keys) where polling
    "is this key down right now?" matters, as opposed to the event-queue
    model of :class:`KeyboardInput`.

    Usage::

        keys = KeyboardState()
        keys.start()
        ...
        if keys.is_pressed("w"):
            move_forward()
        ...
        keys.stop()
    """

    def __init__(
        self,
        suppress_echo: bool = False,
        *,
        estop_callback: Callable[[], None] | None = None,
        startup_timeout_s: float = 2.0,
    ) -> None:
        if not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be finite and positive")
        self._keys: set[str] = set()
        self._events: deque[str] = deque()
        self._lock = threading.Lock()
        self._estop_latched = threading.Event()
        self._running = False
        self._listener: Any = None  # pynput keyboard.Listener
        self._suppress_echo = suppress_echo
        self._saved_termios: list[Any] | None = None
        self._estop_callback = estop_callback
        self._startup_timeout_s = float(startup_timeout_s)

    def _callbacks(self, keyboard: Any) -> tuple[Any, Any]:
        def on_press(key: object) -> None:
            try:
                name: str | None = None
                event: str | None = None
                if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                    ch = key.char.lower()  # type: ignore[union-attr]
                    if ch == "x":
                        event = "x"
                    else:
                        name = ch
                elif key == keyboard.Key.esc:
                    name = "esc"
                    self._estop_latched.set()
                    if self._estop_callback is not None:
                        self._estop_callback()
                elif key == keyboard.Key.up:
                    name = "up"
                elif key == keyboard.Key.down:
                    name = "down"
                elif key == keyboard.Key.left:
                    name = "left"
                elif key == keyboard.Key.right:
                    name = "right"
                elif key == keyboard.Key.space:
                    event = "space"
                elif key == keyboard.Key.enter:
                    event = "enter"
                elif key == keyboard.Key.backspace:
                    event = "backspace"
                with self._lock:
                    if name is not None:
                        self._keys.add(name)
                    if event is not None:
                        self._events.append(event)
            except Exception:
                logger.warning("global key press callback failed", exc_info=True)

        def on_release(key: object) -> None:
            try:
                name: str | None = None
                if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                    name = key.char.lower()  # type: ignore[union-attr]
                elif key == keyboard.Key.esc:
                    name = "esc"
                elif key == keyboard.Key.up:
                    name = "up"
                elif key == keyboard.Key.down:
                    name = "down"
                elif key == keyboard.Key.left:
                    name = "left"
                elif key == keyboard.Key.right:
                    name = "right"
                if name is not None:
                    with self._lock:
                        self._keys.discard(name)
            except Exception:
                logger.warning("global key release callback failed", exc_info=True)

        return on_press, on_release

    def stop(self) -> None:
        """Stop the listener and restore terminal state. Idempotent."""
        listener = self._listener
        self._running = False
        try:
            if _stop_listener_bounded(listener, label="global keyboard"):
                self._listener = None
        finally:
            with self._lock:
                self._keys.clear()
                self._events.clear()
            self._estop_latched.clear()
            if self._suppress_echo:
                _restore_terminal_echo(self._saved_termios)
                self._saved_termios = None

    def quiesce(self) -> None:
        """Disable external callbacks while retaining capture through shutdown."""
        self._estop_callback = None
        with self._lock:
            self._events.clear()

    def wait_for_release(self, timeout_s: float = 2.0) -> bool:
        """Wait boundedly for tracked keys to be released before stopping capture."""
        deadline_s = time.monotonic() + timeout_s
        while time.monotonic() < deadline_s:
            with self._lock:
                if not self._keys:
                    return True
            time.sleep(0.01)
        with self._lock:
            return not self._keys

    def start(self) -> None:
        """Synchronously create and start the global listener."""
        if self._running:
            return
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError(
                "KeyboardState requires an X11/Wayland graphical session"
            )
        try:
            from pynput import keyboard  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pynput is required for global keyboard capture"
            ) from None

        if self._suppress_echo:
            self._saved_termios = _suppress_terminal_echo()
        try:
            on_press, on_release = self._callbacks(keyboard)
            # Avoid pynput's X11-wide grab, which can outlive device shutdown.
            listener = keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
                suppress=False,
            )
            self._listener = listener
            listener.start()
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
                target=wait_until_ready, name="keyboard-ready", daemon=True
            ).start()
            if not ready.wait(timeout=self._startup_timeout_s):
                raise RuntimeError(
                    f"global keyboard listener startup timed out after {self._startup_timeout_s:.1f}s"
                )
            if startup_errors:
                raise RuntimeError(
                    "global keyboard listener failed during startup"
                ) from startup_errors[0]
            if not listener.is_alive():
                raise RuntimeError("global keyboard listener exited during startup")
            self._running = True
        except Exception:
            rollback_listener = self._listener
            try:
                if rollback_listener is not None:
                    rollback_listener.stop()
                    rollback_listener.join(timeout=1.0)
            except Exception:
                logger.warning(
                    "global keyboard listener rollback failed", exc_info=True
                )
            if rollback_listener is None or not rollback_listener.is_alive():
                self._listener = None
            else:
                logger.error("global keyboard listener survived failed startup")
            if self._suppress_echo:
                _restore_terminal_echo(self._saved_termios)
                self._saved_termios = None
            raise

    def is_pressed(self, key: str) -> bool:
        if key == "esc" and self._estop_latched.is_set():
            return True
        with self._lock:
            return key in self._keys

    def pressed_keys(self) -> tuple[str, ...]:
        """Return a stable snapshot of currently held keys for diagnostics."""
        with self._lock:
            return tuple(sorted(self._keys))

    @property
    def healthy(self) -> bool:
        """Whether the listener that owns the emergency-stop input is alive."""
        listener = self._listener
        healthy = bool(self._running and listener is not None and listener.is_alive())
        if not healthy:
            with self._lock:
                self._keys.clear()
        return healthy

    def pop_event(self) -> str | None:
        """Pop the next one-shot event (space/enter/backspace), or None."""
        with self._lock:
            return self._events.popleft() if self._events else None
