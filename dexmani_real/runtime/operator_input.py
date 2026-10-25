"""Shared pynput-based operator keyboard input for interactive robot workflows."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import threading
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# stop() disables callbacks, but Linux/XRecord may remain blocked; avoid long waits.
LISTENER_STOP_TIMEOUT_S = 0.25
LISTENER_START_TIMEOUT_S = 2.0

__all__ = [
    "CARTESIAN_JOG_KEYS",
    "OperatorCommand",
    "KeyboardInput",
]


class OperatorCommand(Enum):
    BEGIN = "BEGIN"
    STOP = "STOP"
    DISCARD = "DISCARD"
    HOME = "HOME"
    RESET_TACTILE = "RESET_TACTILE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    QUIT = "QUIT"


KEY_MAP: dict[str, OperatorCommand] = {
    "b": OperatorCommand.BEGIN,
    "s": OperatorCommand.STOP,
    "d": OperatorCommand.DISCARD,
    "h": OperatorCommand.HOME,
    "t": OperatorCommand.RESET_TACTILE,
    "q": OperatorCommand.QUIT,
}

CARTESIAN_JOG_KEYS = frozenset(
    {"w", "s", "a", "d", "up", "down", "left", "right", "i", "k", "j", "l"}
)

EVENT_ONLY_KEYS = frozenset({"x", "space", "enter", "backspace"})


def suppress_terminal_echo() -> list[Any] | None:
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


def restore_terminal_echo(saved: list[Any] | None) -> None:
    """Restore terminal attributes and discard canonical input typed meanwhile."""
    if saved is None:
        return
    try:
        import termios

        # TCSAFLUSH discards terminal input accumulated alongside pynput events.
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, saved)
    except (termios.error, OSError):
        logger.warning("terminal echo restoration failed", exc_info=True)


def stop_listener_bounded(listener: Any) -> bool:
    """Request pynput shutdown without treating a slow daemon join as a fault."""
    if listener is None:
        return True
    try:
        listener.stop()
        listener.join(timeout=LISTENER_STOP_TIMEOUT_S)
        alive = bool(listener.is_alive())
    except Exception:
        logger.warning("keyboard listener shutdown raised", exc_info=True)
        return False
    if alive:
        # Restore the terminal even if pynput's daemon remains blocked after stop().
        logger.debug(
            "keyboard listener backend did not confirm exit within %.1fs; "
            "terminal input will still be restored and flushed",
            LISTENER_STOP_TIMEOUT_S,
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
        raw_event_keys: frozenset[str] = EVENT_ONLY_KEYS,
    ) -> None:
        self.buffer: deque[OperatorCommand] = deque()
        self.events: deque[str] = deque()
        self.held_keys: set[str] = set()
        self.generation_signal = mp.get_context("spawn").Value("q", 0)
        self.condition = threading.Condition(threading.Lock())
        self.listener: Any = None
        self.running: bool = False
        # A daemon can outlive shutdown; disable its callbacks before releasing the session.
        self.callbacks_active: bool = False
        self.estop_callback = estop_callback
        self.stop_callback = stop_callback
        self.quit_callback = quit_callback
        self.capture_commands = bool(capture_commands)
        self.capture_raw_events = bool(capture_raw_events)
        self.raw_event_keys = raw_event_keys
        self.estop_event = threading.Event()
        self.listener_failure_reported = False
        self.saved_termios: list | None = None

    def latch_emergency_stop(self) -> None:
        callback: Callable[[], None] | None = None
        with self.condition:
            if not self.callbacks_active:
                return
            if self.estop_event.is_set():
                return
            self.estop_event.set()
            callback = self.estop_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.error("keyboard emergency-stop callback failed", exc_info=True)
        with self.condition:
            if self.callbacks_active:
                self.generation += 1
                self.buffer.append(OperatorCommand.EMERGENCY_STOP)
                self.condition.notify()

    def dispatch_immediate_callback(self, signal: OperatorCommand) -> bool:
        """Complete the immediate effect before the command becomes pollable."""
        with self.condition:
            if not self.callbacks_active:
                return False
            callback = (
                self.stop_callback
                if signal is OperatorCommand.STOP
                else self.quit_callback if signal is OperatorCommand.QUIT else None
            )
        if callback is None:
            return True
        try:
            callback()
        except Exception:
            # Consumers must not proceed as if a failed stop/quit was applied.
            self.latch_emergency_stop()
            logger.error("keyboard %s callback failed", signal.value, exc_info=True)
            return False
        return True

    def callbacks(self, keyboard: Any) -> tuple[Callable[[object], None], Callable[[object], None]]:
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
                if name == "esc":
                    self.latch_emergency_stop()
                    return
                signal = KEY_MAP.get(name)
                with self.condition:
                    if not self.callbacks_active:
                        return
                    if name in EVENT_ONLY_KEYS:
                        if self.capture_raw_events and name in self.raw_event_keys:
                            self.events.append(name)
                        return
                    # Held keys also deduplicate command edges across OS repeat.
                    if name in self.held_keys:
                        return
                # Fence STOP/QUIT before exposing the edge. These callbacks only
                # revoke authority; target IPC and SDK I/O belong to their owners.
                if (
                    signal is not None
                    and (
                        self.capture_commands
                        or (signal is OperatorCommand.QUIT and self.quit_callback is not None)
                    )
                    and not self.dispatch_immediate_callback(signal)
                ):
                    return
                with self.condition:
                    if not self.callbacks_active:
                        return
                    self.held_keys.add(name)
                    if self.capture_raw_events and name in self.raw_event_keys:
                        self.events.append(name)
                    self.jog_changed(name, True)
                    if self.capture_commands and signal is not None:
                        self.buffer.append(signal)
                        self.condition.notify()
            except Exception:
                self.latch_emergency_stop()
                logger.warning("keyboard press callback failed", exc_info=True)

        def on_release(key: object) -> None:
            try:
                name = key_name(key)
                with self.condition:
                    changed = self.callbacks_active and name in self.held_keys
                    if self.callbacks_active:
                        self.held_keys.discard(name)
                    self.jog_changed(name, changed)
            except Exception:
                self.latch_emergency_stop()
                logger.warning("keyboard release callback failed", exc_info=True)

        return on_press, on_release

    @property
    def generation(self):
        return self.generation_signal.value

    @generation.setter
    def generation(self, value):
        self.generation_signal.value = value

    def jog_changed(self, name: str | None, changed: bool) -> None:
        if not changed:
            return
        if name in CARTESIAN_JOG_KEYS or name == "q" or (name == "r" and name in self.held_keys):
            self.generation += 1

    def start(self) -> None:
        """Start the daemon keyboard listener once."""
        if self.running:
            return
        if self.listener is not None:
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

        self.saved_termios = suppress_terminal_echo()
        self.estop_event.clear()
        self.listener_failure_reported = False
        with self.condition:
            self.buffer.clear()
            self.events.clear()
            self.held_keys.clear()
            self.generation += 1  # A restarted listener cannot revive old proposals.
            self.callbacks_active = True
        on_press, on_release = self.callbacks(keyboard)

        try:
            self.listener = keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
                suppress=False,
            )
            self.listener.start()
            listener = self.listener
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
            if not ready.wait(timeout=LISTENER_START_TIMEOUT_S):
                raise RuntimeError(
                    f"keyboard listener startup timed out after {LISTENER_START_TIMEOUT_S:.1f}s"
                )
            if startup_errors:
                raise RuntimeError("keyboard listener failed during startup") from startup_errors[0]
            if not self.listener.is_alive():
                raise RuntimeError("keyboard listener exited during startup")
        except BaseException:
            self.stop()
            raise

        self.running = True

    def stop(self) -> None:
        """Stop the listener idempotently for bounded cleanup."""
        listener = self.listener
        with self.condition:
            self.callbacks_active = False
            self.running = False
            self.buffer.clear()
            self.condition.notify_all()
        try:
            if stop_listener_bounded(listener):
                self.listener = None
        finally:
            with self.condition:
                self.events.clear()
                self.held_keys.clear()
            self.estop_event.clear()
            restore_terminal_echo(self.saved_termios)
            self.saved_termios = None

    def poll(self, timeout: float = 0.05) -> list[OperatorCommand]:
        """Drain commands, waiting at most timeout seconds; zero is nonblocking and may return an empty list."""
        if self.running and not self.healthy:
            if not self.listener_failure_reported:
                logger.error("keyboard listener exited; latching emergency stop")
                self.listener_failure_reported = True

        with self.condition:
            if timeout > 0:
                self.condition.wait_for(lambda: self.buffer or not self.running, timeout)
            signals = list(self.buffer)
            self.buffer.clear()
            return signals

    @property
    def healthy(self) -> bool:
        """Whether the listener that owns emergency-stop input is alive."""
        listener = self.listener
        try:
            healthy = bool(self.running and listener is not None and listener.is_alive())
        except Exception:
            logger.error("keyboard listener health check failed", exc_info=True)
            healthy = False
        if not healthy:
            if self.running:
                # Fence the SDK owner before publishing the cleared input generation.
                self.latch_emergency_stop()
            with self.condition:
                if self.held_keys:
                    self.held_keys.clear()
                    self.generation += 1
        return healthy

    @property
    def estop_latched(self) -> bool:
        return self.estop_event.is_set()

    def is_pressed(self, key: str) -> bool:
        """Return whether a key is currently held, with sticky ESC semantics."""
        if key == "esc" and self.estop_event.is_set():
            return True
        with self.condition:
            return key in self.held_keys

    def pressed_keys(self) -> tuple[str, ...]:
        with self.condition:
            return tuple(sorted(self.held_keys))

    def snapshot(self) -> tuple[frozenset[str], int]:
        """Atomic held keys and jog generation; motion release/repress invalidates proposals."""
        with self.condition:
            return frozenset(self.held_keys), self.generation

    def commit_if_current(self, generation: int, commit: Callable[[], Any]) -> Any:
        """Guard local authority/reference updates only; never wait on target IPC or SDK I/O."""
        with self.condition:
            if (
                not self.callbacks_active
                or self.estop_event.is_set()
                or generation != self.generation
            ):
                return None
            # Lock order: input -> short RuntimeState updates. IPC waits run outside both.
            return commit()

    def pop_event(self) -> str | None:
        with self.condition:
            return self.events.popleft() if self.events else None

    def drain_signal(self, target: OperatorCommand) -> None:
        """Remove stale commands after blocking operations, preserving other requests."""
        with self.condition:
            self.buffer = deque(s for s in self.buffer if s != target)
