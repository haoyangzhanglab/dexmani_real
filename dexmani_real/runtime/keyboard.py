"""全局按键事件、持续按键与 jog 输入代际。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import termios
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

LISTENER_START_TIMEOUT_S = 2.0
LISTENER_STOP_TIMEOUT_S = 0.25
CARTESIAN_JOG_KEYS = frozenset(
    {"w", "s", "a", "d", "up", "down", "left", "right", "i", "k", "j", "l"}
)


def suppress_terminal_echo():
    if not sys.stdin.isatty():
        return None
    saved = termios.tcgetattr(sys.stdin.fileno())
    changed = saved.copy()
    changed[3] &= ~(termios.ECHO | termios.ECHONL)
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, changed)
    return saved


def restore_terminal_echo(saved):
    if saved is not None:
        # 丢弃全局监听期间留在终端行缓冲中的输入。
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, saved)


class Keyboard:
    def __init__(
        self, *, estop_callback=None, stop_callback=None, discard_callback=None, quit_callback=None,
    ):
        self.events: deque[str] = deque()
        self.held_keys: set[str] = set()
        self.generation_signal = mp.get_context("spawn").Value("q", 0)
        self.condition = threading.Condition(threading.Lock())
        self.listener: Any = None
        self.active = False
        self.estop_event = threading.Event()
        self.estop_callback = estop_callback
        self.stop_callback = stop_callback
        self.discard_callback = discard_callback
        self.quit_callback = quit_callback
        self.saved_termios = None

    def latch_emergency_stop(self):
        with self.condition:
            if not self.active or self.estop_event.is_set():
                return
            self.estop_event.set()
        try:
            if self.estop_callback is not None:
                self.estop_callback()
        except Exception:
            logger.exception("keyboard emergency-stop callback failed")
        finally:
            with self.condition:
                if self.active:
                    self.generation_signal.value += 1
                    self.events.append("esc")
                    self.condition.notify()

    def handle_key(self, name, pressed):
        if name == "esc" and pressed:
            self.latch_emergency_stop()
            return
        with self.condition:
            if not self.active or pressed == (name in self.held_keys):
                return
            if not pressed:
                self.held_keys.remove(name)
                if name in CARTESIAN_JOG_KEYS or name == "q":
                    self.generation_signal.value += 1
                return

        # 停止回调先撤销权限，再发布按下事件；回调不执行 SDK I/O。
        callback = {
            "s": self.stop_callback, "d": self.discard_callback, "q": self.quit_callback,
        }.get(name)
        if callback is not None:
            callback()
        with self.condition:
            if not self.active:
                return
            self.held_keys.add(name)
            if name in CARTESIAN_JOG_KEYS or name in {"r", "q"}:
                self.generation_signal.value += 1
            self.events.append(name)
            self.condition.notify()

    def callbacks(self, keyboard):
        special = {
            getattr(keyboard.Key, name): name
            for name in ("esc", "up", "down", "left", "right", "space", "enter", "backspace")
        }

        def on_key(key, pressed):
            try:
                if isinstance(key, keyboard.KeyCode):
                    name = key.char.lower() if key.char is not None else None
                else:
                    name = special.get(key)
                if name is not None:
                    self.handle_key(name, pressed)
            except Exception:
                self.latch_emergency_stop()
                logger.exception("keyboard callback failed")

        return lambda key: on_key(key, True), lambda key: on_key(key, False)

    def start(self):
        if self.listener is not None:
            raise RuntimeError("keyboard instances are single-use")
        from pynput import keyboard

        self.saved_termios = suppress_terminal_echo()
        self.active = True
        try:
            on_press, on_release = self.callbacks(keyboard)
            self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.listener.start()
            ready = threading.Event()
            errors = []

            def wait_until_ready():
                try:
                    self.listener.wait()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    ready.set()

            threading.Thread(target=wait_until_ready, name="keyboard-ready", daemon=True).start()
            if not ready.wait(LISTENER_START_TIMEOUT_S):
                raise RuntimeError("keyboard listener startup timed out")
            if errors:
                raise errors[0]
            if not self.listener.is_alive():
                self.listener.join(timeout=0)
                raise RuntimeError("keyboard listener exited during startup")
        except BaseException:
            try:
                self.stop()
            except Exception:
                logger.exception("keyboard startup cleanup failed")
            raise

    def stop(self):
        with self.condition:
            if not self.active:
                return
            self.active = False
            self.events.clear()
            self.held_keys.clear()
            self.generation_signal.value += 1
            self.condition.notify_all()
        try:
            if self.listener is not None:
                self.listener.stop()
                self.listener.join(timeout=LISTENER_STOP_TIMEOUT_S)
                # XRecord 线程可能仍阻塞；active 已禁止它再次调用会话。
                if self.listener.is_alive():
                    logger.debug("keyboard backend still exiting after stop")
        finally:
            restore_terminal_echo(self.saved_termios)
            self.saved_termios = None

    def poll(self, timeout=0.05) -> list[str]:
        if self.active and not self.healthy:
            timeout = 0
        with self.condition:
            if timeout > 0:
                self.condition.wait_for(lambda: self.events or not self.active, timeout)
            events = list(self.events)
            self.events.clear()
            return events

    @property
    def healthy(self):
        healthy = bool(self.active and self.listener is not None and self.listener.is_alive())
        if not healthy and self.active:
            if not self.estop_latched:
                logger.error("keyboard listener exited; latching emergency stop")
            self.latch_emergency_stop()
            with self.condition:
                if self.held_keys:
                    self.held_keys.clear()
                    self.generation_signal.value += 1
        return healthy

    @property
    def estop_latched(self):
        return self.estop_event.is_set()

    def is_pressed(self, key):
        if key == "esc":
            return self.estop_latched
        with self.condition:
            return key in self.held_keys

    def snapshot(self):
        with self.condition:
            return frozenset(self.held_keys), self.generation_signal.value

    def commit_if_current(self, generation: int, commit: Callable[[], Any]) -> Any:
        with self.condition:
            if not self.active or self.estop_latched or generation != self.generation_signal.value:
                return None
            # 锁顺序为键盘 → 会话状态；IPC 等待和 SDK I/O 在锁外执行。
            return commit()

    def discard_events(self, *keys):
        with self.condition:
            self.events = deque(key for key in self.events if key not in keys)
