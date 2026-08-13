"""Headless KeyboardHandler double for teleop_loop tests.

The production handler starts a pynput global listener (X11/Wayland) and fails
closed headless.  This double keeps the same public surface the loop uses
(``start`` / ``stop`` / ``poll`` / ``drain_signal`` / ``healthy`` /
``estop_latched``) and lets the harness inject control signals deterministically
through :meth:`press` and :meth:`latch_estop`.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from dexmani_real.teleop.keyboard import ControlSignal


class FakeKeyboardHandler:
    """Stand-in for ``teleop.keyboard.KeyboardHandler`` (no pynput/display)."""

    last_instance: "FakeKeyboardHandler | None" = None

    def __init__(
        self,
        debounce_s: float = 0.5,
        *,
        estop_callback: Callable[[], None] | None = None,
        startup_timeout_s: float = 2.0,
    ) -> None:
        self._buffer: deque[ControlSignal] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._estop_latched = threading.Event()
        self._estop_callback = estop_callback
        FakeKeyboardHandler.last_instance = self

    def start(self) -> None:
        self._running = True
        self._estop_latched.clear()

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._buffer.clear()

    def press(self, sig: ControlSignal) -> None:
        with self._lock:
            self._buffer.append(sig)

    def latch_estop(self) -> None:
        self._estop_latched.set()
        with self._lock:
            self._buffer.append(ControlSignal.EMERGENCY_STOP)
        if self._estop_callback is not None:
            self._estop_callback()

    def poll(self, timeout: float = 0.05) -> list[ControlSignal]:
        del timeout  # non-blocking by construction
        with self._lock:
            if self._buffer:
                signals = list(self._buffer)
                self._buffer.clear()
                return signals
        return []

    @property
    def healthy(self) -> bool:
        return self._running

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched.is_set()

    def drain_signal(self, target: ControlSignal | None) -> int:
        if target is None:
            return 0
        with self._lock:
            old_len = len(self._buffer)
            self._buffer = deque(s for s in self._buffer if s != target)
            return old_len - len(self._buffer)
