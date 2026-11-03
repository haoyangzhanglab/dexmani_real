"""会话持有运行状态，仅停止信号跨进程共享。"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing.synchronize import Event

from dexmani_real.robot.commands import OperationCancelled


class SafetyState(IntEnum):
    DISARMED = 0
    ARMED = 1
    RUNNING = 2


class RunEndReason(IntEnum):
    """操作结束意图；异常详情由操作结果保存。"""
    CANCELLED = 1
    OPERATOR = 2
    QUIT = 3
    ESTOP = 4
    FAILED = 5
    DURATION_COMPLETE = 6


@dataclass
class SessionState:
    quit_requested: bool = False
    safety_state: SafetyState = SafetyState.DISARMED
    # 取消计数和流程锁仅供父进程线程使用。
    motion_lock: AbstractContextManager = field(default_factory=threading.RLock, repr=False)
    cancel_epoch: int = 1
    stop_signal: Event = field(default_factory=lambda: mp.get_context("spawn").Event(), repr=False)
    emergency_signal: Event = field(default_factory=lambda: mp.get_context("spawn").Event(), repr=False)

    @property
    def estop_request(self) -> bool:
        return self.emergency_signal.is_set()

    def requests_allow_motion(self):
        return (
            not self.estop_request
            and not self.quit_requested
        )

    def arm(self, *, cancel_epoch) -> None:
        with self.motion_lock:
            if self.cancel_epoch != cancel_epoch or not self.requests_allow_motion():
                raise OperationCancelled("startup cancelled before arming")
            self.revoke_motion()

    def begin_motion(self, *, cancel_epoch: int) -> int | None:
        with self.motion_lock:
            if (
                self.cancel_epoch != cancel_epoch
                or self.safety_state != SafetyState.ARMED
                or not self.requests_allow_motion() or self.stop_signal.is_set()
            ):
                return None
            started = time.monotonic_ns()
            self.safety_state = SafetyState.RUNNING
            return started

    def can_move(self, *, cancel_epoch, required_safety_state=SafetyState.RUNNING):
        with self.motion_lock:
            return (
                self.requests_allow_motion()
                and self.cancel_epoch == cancel_epoch
                and self.safety_state == required_safety_state
                and (required_safety_state != SafetyState.RUNNING or not self.stop_signal.is_set())
            )

    def revoke_motion(self, new_state=SafetyState.ARMED):
        with self.motion_lock:
            self.stop_signal.set()
            self.safety_state = new_state

    def request_emergency_stop(self):
        self.emergency_signal.set()
        self.stop_signal.set()
        with self.motion_lock:
            self.revoke_motion(SafetyState.DISARMED if self.safety_state == SafetyState.DISARMED
                               else SafetyState.ARMED)

    def request_stop(self):
        with self.motion_lock:
            self.cancel_epoch += 1
            self.revoke_motion(SafetyState.DISARMED if self.safety_state == SafetyState.DISARMED
                               else SafetyState.ARMED)

    def request_quit(self):
        with self.motion_lock:
            self.quit_requested = True
            self.request_stop()
