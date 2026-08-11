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

from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "ControlSignal",
    "KeyboardHandler",
    "GlobalKeyState",
    "MotionActivityLatch",
    "validate_arm_feedback",
    "validate_hand_feedback",
    "eef_delta_from_keys",
]


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
    """Restore terminal attributes saved by :func:`_suppress_terminal_echo`."""
    if saved is None:
        return
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
    except (termios.error, OSError):
        logger.warning("terminal echo restoration failed", exc_info=True)


class KeyboardHandler:
    """Thread-safe global keyboard events with a latched emergency stop."""

    def __init__(
        self,
        debounce_s: float = 0.5,
        *,
        estop_callback: Callable[[], None] | None = None,
        startup_timeout_s: float = 2.0,
    ) -> None:
        if not np.isfinite(debounce_s) or debounce_s < 0.0:
            raise ValueError("debounce_s must be finite and non-negative")
        if not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be finite and positive")
        self._buffer: deque[ControlSignal] = deque()
        self._lock = threading.Lock()
        self._listener: Any = None  # pynput.keyboard.Listener
        self._running: bool = False
        self._debounce_s = float(debounce_s)
        self._startup_timeout_s = float(startup_timeout_s)
        self._estop_callback = estop_callback
        self._estop_latched = threading.Event()
        self._listener_failure_reported = False
        self._last_signal_time: dict[ControlSignal, float] = {}
        self._saved_termios: list | None = None

    def _latch_emergency_stop(self) -> None:
        callback: Callable[[], None] | None = None
        with self._lock:
            if self._estop_latched.is_set():
                return
            self._estop_latched.set()
            self._buffer.append(ControlSignal.EMERGENCY_STOP)
            callback = self._estop_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.error("keyboard emergency-stop callback failed", exc_info=True)

    def start(self) -> None:
        """Start the global pynput keyboard listener in a daemon thread.

        Idempotent: calling on an already-started handler is a no-op.
        """
        if self._running:
            return

        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError(
                "KeyboardHandler requires a graphical session (pynput global "
                "capture uses X11/Wayland). No DISPLAY/WAYLAND_DISPLAY set — "
                "headless/SSH is not supported; run on the workstation desktop."
            )

        try:
            from pynput import keyboard  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pynput is required for global keyboard capture. " "Install with: pip install pynput"
            ) from None

        self._suppress_terminal_echo()
        self._estop_latched.clear()
        self._listener_failure_reported = False

        def on_press(key: object) -> None:
            try:
                if hasattr(key, "char") and key.char is not None:  # type: ignore[union-attr]
                    sig = _KEY_MAP.get(key.char.lower())  # type: ignore[union-attr]
                elif key == keyboard.Key.esc:
                    self._latch_emergency_stop()
                    return
                else:
                    return
                if sig is not None:
                    now = time.perf_counter()
                    with self._lock:
                        last = self._last_signal_time.get(sig, 0.0)
                        if now - last < self._debounce_s:
                            return
                        self._last_signal_time[sig] = now
                        self._buffer.append(sig)
            except Exception:
                logger.warning("keyboard press callback failed", exc_info=True)

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            ready = threading.Event()
            startup_errors: list[BaseException] = []

            def wait_until_ready() -> None:
                try:
                    listener = self._listener
                    if listener is None:
                        raise RuntimeError("keyboard listener disappeared during startup")
                    listener.wait()
                except BaseException as exc:
                    startup_errors.append(exc)
                finally:
                    ready.set()

            threading.Thread(target=wait_until_ready, name="keyboard-events-ready", daemon=True).start()
            if not ready.wait(timeout=self._startup_timeout_s):
                raise RuntimeError(f"keyboard listener startup timed out after {self._startup_timeout_s:.1f}s")
            if startup_errors:
                raise RuntimeError("keyboard listener failed during startup") from startup_errors[0]
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
                logger.warning("keyboard listener rollback health check failed", exc_info=True)
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
            if listener is not None:
                listener.stop()
                listener.join(timeout=1.0)
                if listener.is_alive():
                    logger.error("keyboard listener did not stop within 1s")
                else:
                    self._listener = None
        except Exception:
            logger.warning("keyboard listener stop failed", exc_info=True)
        finally:
            with self._lock:
                self._buffer.clear()
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

    def __enter__(self) -> "KeyboardHandler":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def poll(self, timeout: float = 0.05) -> list[ControlSignal]:
        """Drain all pending ControlSignals from the keyboard buffer.

        Args:
            timeout: Seconds to wait if buffer is empty (default 0.05s).
                     Use 0 for completely non-blocking poll.

        Returns:
            List of ControlSignal values (may be empty).
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

        # Slow path: buffer empty, wait briefly for events
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
                    name = key.char.lower()  # type: ignore[union-attr]
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
            if listener is not None:
                listener.stop()
                listener.join(timeout=1.0)
                if listener.is_alive():
                    logger.error("global keyboard listener did not stop within 1s")
                else:
                    self._listener = None
        except Exception:
            logger.warning("global keyboard listener stop failed", exc_info=True)
        finally:
            with self._lock:
                self._keys.clear()
                self._events.clear()
            self._estop_latched.clear()
            if self._suppress_echo:
                _restore_terminal_echo(self._saved_termios)
                self._saved_termios = None

    def start(self) -> None:
        """Synchronously create and start the global listener."""
        if self._running:
            return
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError("GlobalKeyState requires an X11/Wayland graphical session")
        try:
            from pynput import keyboard  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("pynput is required for global keyboard capture") from None

        if self._suppress_echo:
            self._saved_termios = _suppress_terminal_echo()
        try:
            on_press, on_release = self._callbacks(keyboard)
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
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

            threading.Thread(target=wait_until_ready, name="keyboard-ready", daemon=True).start()
            if not ready.wait(timeout=self._startup_timeout_s):
                raise RuntimeError(f"global keyboard listener startup timed out after {self._startup_timeout_s:.1f}s")
            if startup_errors:
                raise RuntimeError("global keyboard listener failed during startup") from startup_errors[0]
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
                logger.warning("global keyboard listener rollback failed", exc_info=True)
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

    @property
    def any_pressed(self) -> bool:
        with self._lock:
            return bool(self._keys)


class MotionActivityLatch:
    """Detect the active-to-idle edge of continuous keyboard motion."""

    def __init__(self) -> None:
        self._active = False

    def update(self, active: bool) -> bool:
        """Store *active* and return True exactly once when motion is released."""
        released = self._active and not active
        self._active = bool(active)
        return released

    def reset(self) -> None:
        self._active = False


def validate_arm_feedback(
    *,
    connected: bool,
    state_valid: bool,
    source_monotonic_ns: int,
    now_monotonic_ns: int,
    max_age_s: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    eef_pos: np.ndarray,
    eef_rot6d: np.ndarray,
) -> str | None:
    """Return why required arm feedback is unusable, or ``None``."""
    if not connected:
        return "arm disconnected"
    if not state_valid:
        return "arm state marked invalid"
    if source_monotonic_ns <= 0:
        return "arm state has no source timestamp"
    age_s = (now_monotonic_ns - source_monotonic_ns) * 1e-9
    if age_s < 0.0:
        return f"arm state timestamp is {abs(age_s):.3f}s in the future"
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    if age_s > max_age_s:
        return f"arm state stale ({age_s:.2f}s)"
    fields = {
        "qpos": (np.asarray(qpos), ARM_JOINT_SHAPE),
        "qvel": (np.asarray(qvel), ARM_JOINT_SHAPE),
        "eef_pos": (np.asarray(eef_pos), (3,)),
        "eef_rot6d": (np.asarray(eef_rot6d), (6,)),
    }
    for name, (value, expected_shape) in fields.items():
        if value.shape != expected_shape:
            return f"arm {name} has shape {value.shape}, expected {expected_shape}"
        if not np.all(np.isfinite(value)):
            return f"arm {name} is non-finite"
    return None


def validate_hand_feedback(
    *,
    connected: bool,
    error_state: bool,
    qpos_stale: bool,
    state_valid: bool,
    send_healthy: bool,
    read_healthy: bool,
    source_monotonic_ns: int,
    now_monotonic_ns: int,
    max_age_s: float,
    qpos: np.ndarray,
) -> str | None:
    """Return why measured XHand feedback is unusable, or ``None``."""
    if not connected:
        return "hand disconnected"
    if error_state:
        return "hand reported a hardware error"
    if qpos_stale:
        return "hand joint feedback is stale"
    if not state_valid:
        return "hand state marked invalid"
    if not send_healthy or not read_healthy:
        return "hand command/state I/O is unhealthy"
    if source_monotonic_ns <= 0:
        return "hand state has no source timestamp"
    age_s = (now_monotonic_ns - source_monotonic_ns) * 1e-9
    if age_s < 0.0:
        return f"hand state timestamp is {abs(age_s):.3f}s in the future"
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    if age_s > max_age_s:
        return f"hand state stale ({age_s:.2f}s)"
    value = np.asarray(qpos)
    if value.shape != HAND_JOINT_SHAPE:
        return f"hand qpos has shape {value.shape}, expected {HAND_JOINT_SHAPE}"
    if not np.all(np.isfinite(value)):
        return "hand qpos is non-finite"
    return None


def eef_delta_from_keys(
    keys: GlobalKeyState,
    delta_pos: float,
    delta_rpy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map held WASD/arrow/IJKL keys to EEF delta position (dx) and rotation (drpy).

    Args:
        keys: Active :class:`GlobalKeyState` instance.
        delta_pos: Per-frame translation step (m).
        delta_rpy: Per-frame rotation step (rad).

    Returns:
        ``(dx, drpy)`` — each is a (3,) ``np.float64`` array.  Zero when no
        relevant keys are held.
    """
    dx = np.zeros(3, dtype=np.float64)
    if keys.is_pressed("w"):
        dx[0] += delta_pos
    if keys.is_pressed("s"):
        dx[0] -= delta_pos
    if keys.is_pressed("a"):
        dx[1] -= delta_pos
    if keys.is_pressed("d"):
        dx[1] += delta_pos
    if keys.is_pressed("up"):
        dx[2] += delta_pos
    if keys.is_pressed("down"):
        dx[2] -= delta_pos

    drpy = np.zeros(3, dtype=np.float64)
    if keys.is_pressed("left"):
        drpy[0] += delta_rpy
    if keys.is_pressed("right"):
        drpy[0] -= delta_rpy
    if keys.is_pressed("i"):
        drpy[1] += delta_rpy
    if keys.is_pressed("k"):
        drpy[1] -= delta_rpy
    if keys.is_pressed("j"):
        drpy[2] -= delta_rpy
    if keys.is_pressed("l"):
        drpy[2] += delta_rpy

    return dx, drpy
