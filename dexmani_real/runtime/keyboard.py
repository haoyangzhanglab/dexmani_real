"""全局按键事件与持续按键快照。"""

from __future__ import annotations

import logging
import sys
import termios
import threading
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

LISTENER_START_TIMEOUT_S = 2.0
LISTENER_STOP_TIMEOUT_S = 0.25
JOG_KEY_PAIRS = (
    ("w", "s"), ("d", "a"), ("up", "down"),
    ("left", "right"), ("i", "k"), ("j", "l"),
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
    def __init__(self, *, bindings=None, estop_callback=None):
        self.events: deque[str] = deque()
        self.accepting_starts = False
        self.held_keys: set[str] = set()
        self.condition = threading.Condition(threading.Lock())
        self.listener: Any = None
        self.active = False
        self.estop_latched = False
        self.estop_callback = estop_callback
        self.bindings = dict(bindings or {})
        self.saved_termios = None

    def latch_emergency_stop(self):
        with self.condition:
            if not self.active or self.estop_latched:
                return
            self.estop_latched = True
            try:
                if self.estop_callback is not None:
                    self.estop_callback()
            finally:
                self.events.append("esc")
                self.condition.notify()

    def handle_key(self, name: str, pressed: bool):
        if name == "esc" and pressed:
            self.latch_emergency_stop()
            return
        with self.condition:
            if not self.active or pressed == (name in self.held_keys):
                return
            if not pressed:
                self.held_keys.remove(name)
                return
            self.held_keys.add(name)
            if name in {"b", "h", "r", "t"} and not self.accepting_starts:
                return
            callback = self.bindings.get(name)
            if callback is not None:
                callback()
            self.events.append(name)
            self.condition.notify()

    def allow_start(self):
        """主线程仅在上个操作及资源收尾结束后调用。"""
        with self.condition:
            self.accepting_starts = True

    def accept_operation(self, state):
        """与快停回调排序；返回主线程当前操作的取消上下文。"""
        with self.condition:
            if not self.accepting_starts or any(
                event in self.bindings or event == "esc" for event in self.events
            ):
                return None
            self.accepting_starts = False
            self.events = deque(event for event in self.events if event not in {"b", "h", "t", "r"})
            with state.motion_lock:
                if not state.requests_allow_motion():
                    return None
                return state.cancel_epoch

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
            except BaseException:
                logger.exception("keyboard startup cleanup failed")
            raise

    def stop(self):
        with self.condition:
            self.active = False
            self.events.clear()
            self.held_keys.clear()
            self.condition.notify_all()
        failure = None

        def remember_failure(stage, exc):
            nonlocal failure
            first_failure = failure is None
            if first_failure:
                failure = exc
            if isinstance(exc, KeyboardInterrupt) and self.estop_callback is not None:
                self.estop_callback()
            if not first_failure:
                logger.exception("keyboard %s also failed", stage)

        try:
            if self.listener is not None:
                try:
                    self.listener.stop()
                except BaseException as exc:
                    remember_failure("listener stop", exc)
                finally:
                    try:
                        if self.listener.ident is not None:
                            self.listener.join(timeout=LISTENER_STOP_TIMEOUT_S)
                    except BaseException as exc:
                        remember_failure("listener join", exc)
        finally:
            try:
                restore_terminal_echo(self.saved_termios)
            except BaseException as exc:
                remember_failure("terminal restore", exc)
            else:
                self.saved_termios = None
        if self.listener is not None and self.listener.is_alive():
            pending = TimeoutError("keyboard listener remains active after join")
            if failure is None:
                failure = pending
            else:
                logger.error("%s", pending)
        if failure is not None:
            raise failure

    def poll(self, state, timeout=0.05) -> tuple[list[str], bool]:
        """事件与 stop 在同一输入临界区取快照；调用方在锁外处理。"""
        if self.active and not self.check_health():
            timeout = 0
        with self.condition:
            if timeout > 0:
                self.condition.wait_for(lambda: self.events or not self.active, timeout)
            events = list(self.events)
            self.events.clear()
            return events, state.stop_signal.is_set()

    def check_health(self):
        healthy = bool(self.active and self.listener is not None and self.listener.is_alive())
        if not healthy and self.active:
            if not self.estop_latched:
                logger.error("keyboard listener exited; latching emergency stop")
            self.latch_emergency_stop()
            with self.condition:
                self.held_keys.clear()
        return healthy

    def snapshot(self) -> frozenset[str]:
        with self.condition:
            return frozenset(self.held_keys)
