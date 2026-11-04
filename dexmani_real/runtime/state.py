"""会话持有运行状态，仅停止信号跨进程共享。"""

from __future__ import annotations

import multiprocessing as mp
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing.synchronize import Event

# 首次关节反馈或控制观测就绪的等待预算。
INITIAL_STATE_TIMEOUT_S = 1.0


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

    def request_emergency_stop(self):
        self.emergency_signal.set()
        self.stop_signal.set()
        # 准备边界可能正在锁内清 stop；急停先锁存，再确保停止信号保留。
        with self.motion_lock:
            self.stop_signal.set()

    def request_stop(self):
        with self.motion_lock:
            self.cancel_epoch += 1
            self.stop_signal.set()

    def request_quit(self):
        with self.motion_lock:
            self.quit_requested = True
            self.request_stop()
