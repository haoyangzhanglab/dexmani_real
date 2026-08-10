"""pynput-based global keyboard handler for teleop control (B/C/S/D/H/Q/ESC)."""

from __future__ import annotations

__all__ = [
    "ControlSignal",
    "KeyboardHandler",
    "GlobalKeyState",
    "MotionActivityLatch",
    "MotionTraceSample",
    "ReleaseMotionTracer",
    "eef_delta_from_keys",
]

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


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
        pass


class KeyboardHandler:
    """Global keyboard handler using pynput.

    Captures keystrokes globally - works even when the terminal window
    does not have focus.

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
            raise ImportError(
                "pynput is required for global keyboard capture. " "Install with: pip install pynput"
            ) from None

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
        """Disable terminal ECHO (delegates to module-level helper)."""
        self._saved_termios = _suppress_terminal_echo()

    def _restore_terminal_echo(self) -> None:
        """Restore terminal attributes (delegates to module-level helper)."""
        _restore_terminal_echo(self._saved_termios)
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

    def __init__(self, suppress_echo: bool = False) -> None:
        self._keys: set[str] = set()
        self._events: list[str] = []  # one-shot event queue
        self._running = False
        self._thread: threading.Thread | None = None
        self._listener: Any = None  # pynput keyboard.Listener
        self._suppress_echo = suppress_echo
        self._saved_termios: list[Any] | None = None

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
                elif key == keyboard.Key.space:
                    self._events.append("space")
                elif key == keyboard.Key.enter:
                    self._events.append("enter")
                elif key == keyboard.Key.backspace:
                    self._events.append("backspace")
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
        if self._suppress_echo:
            _restore_terminal_echo(self._saved_termios)
            self._saved_termios = None

    def start(self) -> None:
        if self._suppress_echo:
            self._saved_termios = _suppress_terminal_echo()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_pressed(self, key: str) -> bool:
        return key in self._keys

    def pop_event(self) -> str | None:
        """Pop the next one-shot event (space/enter/backspace), or None."""
        if self._events:
            return self._events.pop(0)
        return None

    @property
    def any_pressed(self) -> bool:
        return len(self._keys) > 0


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


@dataclass
class MotionTraceSample:
    """One state-aligned sample for keyboard release diagnostics."""

    frame: int
    timestamp_s: float
    input_active: bool
    eef_pos_m: np.ndarray
    command_pos_m: np.ndarray
    qpos_error_rad: float
    qvel_peak_rad_s: float
    state_age_s: float
    queue_latency_s: float
    apply_latency_s: float

    def __post_init__(self) -> None:
        self.eef_pos_m = self._vector3(self.eef_pos_m, "eef_pos_m")
        self.command_pos_m = self._vector3(self.command_pos_m, "command_pos_m")
        if self.frame < 0:
            raise ValueError("motion trace frame must be >= 0")
        for name in (
            "timestamp_s",
            "qpos_error_rad",
            "qvel_peak_rad_s",
            "state_age_s",
            "queue_latency_s",
            "apply_latency_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            setattr(self, name, value)

    @staticmethod
    def _vector3(value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite (3,) vector")
        return vector.copy()


class ReleaseMotionTracer:
    """Capture and summarize a short high-rate window around key release.

    The tracer is deliberately control-agnostic: callers provide state-aligned
    samples and the last translational input direction. It returns printable
    lines, so the helper remains deterministic and hardware-free in tests.
    """

    def __init__(
        self,
        *,
        pre_frames: int = 6,
        post_frames: int = 20,
        cooldown_s: float = 5.0,
        velocity_deadband_m_s: float = 0.01,
    ) -> None:
        if pre_frames <= 0 or post_frames <= 0:
            raise ValueError("release trace pre/post frames must be > 0")
        if not np.isfinite(cooldown_s) or cooldown_s < 0.0:
            raise ValueError("release trace cooldown_s must be finite and >= 0")
        if not np.isfinite(velocity_deadband_m_s) or velocity_deadband_m_s < 0.0:
            raise ValueError("release trace velocity deadband must be finite and >= 0")
        self.pre_frames = int(pre_frames)
        self.post_frames = int(post_frames)
        self.cooldown_s = float(cooldown_s)
        self.velocity_deadband_m_s = float(velocity_deadband_m_s)
        self._history: deque[MotionTraceSample] = deque(maxlen=self.pre_frames)
        self._pre_samples: list[MotionTraceSample] = []
        self._window_samples: list[MotionTraceSample] = []
        self._release_prev_sample: MotionTraceSample | None = None
        self._direction: np.ndarray | None = None
        self._release_pos_m: np.ndarray | None = None
        self._post_seen = 0
        self._window_id = 0
        self._last_start_s = float("-inf")
        self.last_summary: dict[str, float | int | str | bool] | None = None

    @property
    def active(self) -> bool:
        return self._direction is not None

    def reset(self) -> None:
        """Discard history/window state after a blocking mode change such as HOME."""
        self._history.clear()
        self._pre_samples.clear()
        self._window_samples.clear()
        self._release_prev_sample = None
        self._direction = None
        self._release_pos_m = None
        self._post_seen = 0
        self._last_start_s = float("-inf")
        self.last_summary = None

    def observe(
        self,
        sample: MotionTraceSample,
        *,
        release_edge: bool = False,
        translation_direction: np.ndarray | None = None,
    ) -> list[str]:
        """Consume a sample and return zero or more diagnostic log lines."""
        lines: list[str] = []
        if release_edge and not self.active:
            direction = self._normalize_direction(translation_direction)
            cooldown_elapsed = sample.timestamp_s - self._last_start_s >= self.cooldown_s
            if direction is not None and cooldown_elapsed:
                pre_samples = list(self._history)
                self._window_id += 1
                self._last_start_s = sample.timestamp_s
                self._direction = direction
                self._release_pos_m = sample.eef_pos_m.copy()
                self._release_prev_sample = pre_samples[-1] if pre_samples else None
                self._pre_samples = pre_samples
                self._window_samples = [sample]
                self._post_seen = 0
                self.last_summary = None
                self._history.clear()
                return lines
            self._history.clear()

        if self.active:
            self._window_samples.append(sample)
            self._post_seen += 1
            if sample.input_active:
                # The sample is contaminated and detailed lines would also add
                # terminal I/O to the newly active control interval. Emit only
                # one summary line in this case.
                lines.extend(self._finish("reengaged", include_samples=False))
            elif self._post_seen >= self.post_frames:
                # Buffer the whole window until capture is complete. Printing
                # 30 Hz samples during capture would alter the timing that this
                # diagnostic is intended to measure.
                lines.extend(self._finish("completed", include_samples=True))

        if sample.input_active:
            if self._normalize_direction(translation_direction) is None:
                self._history.clear()
            else:
                self._history.append(sample)
        return lines

    @staticmethod
    def _normalize_direction(direction: np.ndarray | None) -> np.ndarray | None:
        if direction is None:
            return None
        vector = np.asarray(direction, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("translation_direction must be a finite (3,) vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return None
        return vector / norm

    def _format_sample(
        self,
        sample: MotionTraceSample,
        previous: MotionTraceSample | None,
        phase: str,
    ) -> str:
        assert self._direction is not None
        assert self._release_pos_m is not None
        velocity = np.zeros(3, dtype=np.float64)
        dt_s = 0.0
        if previous is not None:
            dt_s = sample.timestamp_s - previous.timestamp_s
            if dt_s > 1e-6:
                velocity = (sample.eef_pos_m - previous.eef_pos_m) / dt_s
        displacement = sample.eef_pos_m - self._release_pos_m
        along_velocity = float(np.dot(velocity, self._direction))
        lateral_velocity = velocity - along_velocity * self._direction
        command_error = sample.command_pos_m - sample.eef_pos_m
        return (
            f"[RELTRACE#{self._window_id} {phase} f={sample.frame}] "
            f"dt={dt_s * 1000:.1f}ms in={int(sample.input_active)} "
            f"eef={self._fmt_vec(sample.eef_pos_m, scale=1000.0, precision=1)}mm "
            f"drel={self._fmt_vec(displacement, scale=1000.0, precision=1)}mm "
            f"v={self._fmt_vec(velocity, scale=1000.0, precision=1)}mm/s "
            f"va={along_velocity * 1000:+.1f} vl={np.linalg.norm(lateral_velocity) * 1000:.1f}mm/s "
            f"cmd_err={self._fmt_vec(command_error, scale=1000.0, precision=1)}mm "
            f"qerr={np.rad2deg(sample.qpos_error_rad):.1f}deg qv={np.rad2deg(sample.qvel_peak_rad_s):.1f}deg/s "
            f"age/q/apply={sample.state_age_s * 1000:.0f}/"
            f"{sample.queue_latency_s * 1000:.1f}/{sample.apply_latency_s * 1000:.1f}ms"
        )

    def _finish(self, reason: str, *, include_samples: bool) -> list[str]:
        assert self._direction is not None
        assert self._release_pos_m is not None
        samples = self._window_samples
        displacement = np.stack([sample.eef_pos_m - self._release_pos_m for sample in samples])
        along = displacement @ self._direction
        lateral = displacement - along[:, None] * self._direction[None, :]

        velocity_samples: list[np.ndarray] = []
        velocity_timestamps_s: list[float] = []
        previous = self._release_prev_sample
        for sample in samples:
            if previous is not None:
                dt_s = sample.timestamp_s - previous.timestamp_s
                if dt_s > 1e-6:
                    velocity_samples.append((sample.eef_pos_m - previous.eef_pos_m) / dt_s)
                    velocity_timestamps_s.append(sample.timestamp_s)
            previous = sample
        velocities = np.stack(velocity_samples) if velocity_samples else np.zeros((1, 3), dtype=np.float64)
        along_velocity = velocities @ self._direction
        lateral_velocity = velocities - along_velocity[:, None] * self._direction[None, :]

        acceleration_peak_m_s2 = 0.0
        if len(velocities) >= 2:
            velocity_intervals_s = np.diff(velocity_timestamps_s)
            valid = velocity_intervals_s > 1e-6
            if np.any(valid):
                accelerations = np.diff(velocities, axis=0)[valid] / velocity_intervals_s[valid, None]
                acceleration_peak_m_s2 = float(np.max(np.linalg.norm(accelerations, axis=1)))

        signs = np.sign(along_velocity[np.abs(along_velocity) >= self.velocity_deadband_m_s])
        direction_reversals = int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) >= 2 else 0
        peak_forward_m = max(0.0, float(np.max(along)))
        final_along_m = float(along[-1])
        rollback_m = max(0.0, peak_forward_m - final_along_m)
        duration_s = max(0.0, samples[-1].timestamp_s - samples[0].timestamp_s)
        summary: dict[str, float | int | str | bool] = {
            "reason": reason,
            "clean": reason == "completed",
            "duration_s": duration_s,
            "final_along_m": final_along_m,
            "peak_forward_m": peak_forward_m,
            "rollback_m": rollback_m,
            "peak_lateral_m": float(np.max(np.linalg.norm(lateral, axis=1))),
            "peak_reverse_velocity_m_s": max(0.0, float(-np.min(along_velocity))),
            "peak_lateral_velocity_m_s": float(np.max(np.linalg.norm(lateral_velocity, axis=1))),
            "direction_reversals": direction_reversals,
            "peak_acceleration_m_s2": acceleration_peak_m_s2,
            "peak_qpos_error_rad": max(sample.qpos_error_rad for sample in samples),
            "peak_qvel_rad_s": max(sample.qvel_peak_rad_s for sample in samples),
            "max_state_age_s": max(sample.state_age_s for sample in samples),
            "max_queue_latency_s": max(sample.queue_latency_s for sample in samples),
            "max_apply_latency_s": max(sample.apply_latency_s for sample in samples),
        }
        self.last_summary = summary
        summary_line = (
            f"[RELTRACE#{self._window_id} SUMMARY] reason={reason} clean={int(bool(summary['clean']))} "
            f"duration={duration_s * 1000:.0f}ms final={final_along_m * 1000:+.1f}mm "
            f"peak={peak_forward_m * 1000:+.1f}mm rollback={rollback_m * 1000:.1f}mm "
            f"lateral={float(summary['peak_lateral_m']) * 1000:.1f}mm "
            f"reverse_v={float(summary['peak_reverse_velocity_m_s']) * 1000:.1f}mm/s "
            f"lateral_v={float(summary['peak_lateral_velocity_m_s']) * 1000:.1f}mm/s "
            f"reversals={direction_reversals} a_peak={acceleration_peak_m_s2:.2f}m/s2 "
            f"qerr={np.rad2deg(float(summary['peak_qpos_error_rad'])):.1f}deg "
            f"qv={np.rad2deg(float(summary['peak_qvel_rad_s'])):.1f}deg/s "
            f"age/q/apply_max={float(summary['max_state_age_s']) * 1000:.0f}/"
            f"{float(summary['max_queue_latency_s']) * 1000:.1f}/"
            f"{float(summary['max_apply_latency_s']) * 1000:.1f}ms"
        )

        lines: list[str] = []
        if include_samples:
            direction_text = self._fmt_vec(self._direction, scale=1.0, precision=2)
            release_frame = samples[0].frame
            lines.append(
                f"[RELTRACE#{self._window_id} START f={release_frame}] "
                f"dir={direction_text} pre={len(self._pre_samples)} post={self._post_seen} buffered=1"
            )
            format_previous: MotionTraceSample | None = None
            for offset, pre_sample in enumerate(self._pre_samples, start=-len(self._pre_samples)):
                lines.append(self._format_sample(pre_sample, format_previous, f"PRE{offset:+d}"))
                format_previous = pre_sample
            lines.append(self._format_sample(samples[0], format_previous, "REL"))
            format_previous = samples[0]
            for index, post_sample in enumerate(samples[1:], start=1):
                lines.append(self._format_sample(post_sample, format_previous, f"POST{index:02d}"))
                format_previous = post_sample
        lines.append(summary_line)

        self._pre_samples = []
        self._window_samples = []
        self._release_prev_sample = None
        self._direction = None
        self._release_pos_m = None
        self._post_seen = 0
        return lines

    @staticmethod
    def _fmt_vec(vector: np.ndarray, *, scale: float, precision: int) -> str:
        values = np.asarray(vector, dtype=np.float64) * scale
        return f"({values[0]:+.{precision}f},{values[1]:+.{precision}f},{values[2]:+.{precision}f})"


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
