"""会话持有运行状态，仅运行代际与停止信号跨进程共享。"""

from __future__ import annotations

import multiprocessing as mp
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event

from dexmani_real.robot.commands import RobotError


class SafetyState(IntEnum):
    DISARMED = 0
    ARMED = 1
    RUNNING = 2


class RunEndReason(IntEnum):
    """保留本轮首次终止原因，不表示物理收敛。"""

    NONE = 0
    EXECUTOR_BOUNDARY = 1
    OPERATOR = 2
    QUIT = 3
    TIMEOUT = 4
    POLICY_FAILURE = 5
    ESTOP = 6
    HARDWARE_FAULT = 7
    RUNTIME_SHUTDOWN = 8
    RECORDING_FAILURE = 9
    RUNTIME_FAILURE = 10
    DURATION_COMPLETE = 11
    PREDICTION_TIMEOUT = 12


def device_failure_reason(error):
    if isinstance(error, KeyboardInterrupt):
        return RunEndReason.ESTOP
    if isinstance(error, RobotError):
        return RunEndReason.HARDWARE_FAULT
    return None


@dataclass
class SessionState:
    running: bool = True
    run_ended_id: int = 0
    run_ended_reason: RunEndReason = RunEndReason.NONE
    error_state: bool = False
    estop_request: bool = False
    quit_requested: bool = False
    stop_request: bool = False
    safety_state: SafetyState = SafetyState.DISARMED
    # 共享锁仅保护代际与停止状态，SDK I/O 在锁外执行。
    motion_lock: AbstractContextManager = field(
        default_factory=lambda: mp.get_context("spawn").RLock(), repr=False
    )
    run_signal: Synchronized = field(
        default_factory=lambda: mp.get_context("spawn").Value("q", 1), repr=False
    )
    stop_signal: Event = field(default_factory=lambda: mp.get_context("spawn").Event(), repr=False)
    emergency_signal: Event = field(default_factory=lambda: mp.get_context("spawn").Event(), repr=False)

    @property
    def run_id(self):
        return self.run_signal.value

    def requests_allow_motion(self):
        return (
            self.running
            and not self.error_state
            and not self.estop_request
            and not self.quit_requested
            and not self.stop_request
        )

    def arm(self) -> None:
        with self.motion_lock:
            if not self.requests_allow_motion():
                raise RuntimeError("arming rejected")
            self.revoke_motion()

    def begin_motion(self) -> tuple[int, int] | None:
        with self.motion_lock:
            if (
                self.safety_state != SafetyState.ARMED
                or not self.requests_allow_motion() or self.stop_signal.is_set()
            ):
                return None
            started = time.monotonic_ns()
            self.run_ended_reason = RunEndReason.NONE
            self.safety_state = SafetyState.RUNNING
            return self.run_id, started

    def can_move(self, *, run_id, required_safety_state=SafetyState.RUNNING):
        with self.motion_lock:
            return (
                self.requests_allow_motion()
                and self.run_id == run_id
                and self.safety_state == required_safety_state
                and (required_safety_state != SafetyState.RUNNING or not self.stop_signal.is_set())
            )

    def revoke_motion(
        self,
        new_state=SafetyState.ARMED,
        *,
        reason=RunEndReason.EXECUTOR_BOUNDARY,
        expected_run_id=None,
    ) -> None:
        with self.motion_lock:
            if expected_run_id is not None and self.run_id != expected_run_id:
                return
            if self.safety_state == SafetyState.RUNNING:
                if self.estop_request:
                    reason = RunEndReason.ESTOP
                self.run_ended_id = self.run_id
                self.run_ended_reason = reason
            self.stop_signal.set()
            self.run_signal.value += 1
            self.safety_state = new_state

    def request_emergency_stop(self):
        # 先通知 SDK 所属进程，再等待本地状态锁。
        self.emergency_signal.set()
        self.stop_signal.set()
        with self.motion_lock:
            already_requested = self.estop_request
            self.estop_request = True
            if not already_requested:
                self.revoke_motion(
                    (
                        SafetyState.DISARMED
                        if self.safety_state == SafetyState.DISARMED else SafetyState.ARMED
                    ),
                    reason=RunEndReason.ESTOP,
                )

    def mark_error(self):
        """保留会话错误并通知执行器停止。"""
        with self.motion_lock:
            self.error_state = True
            self.stop_signal.set()

    def request_stop(self, *, reason=RunEndReason.OPERATOR):
        self.stop_signal.set()
        with self.motion_lock:
            self.stop_request = True
            if self.safety_state in (SafetyState.ARMED, SafetyState.RUNNING):
                self.revoke_motion(reason=reason)

    def acknowledge_stop(self, expected_run_id):
        """完成停止的所有者只能确认对应代际的请求。"""
        with self.motion_lock:
            if self.run_id == expected_run_id and self.safety_state != SafetyState.RUNNING:
                self.stop_request = False

    def request_quit(self):
        with self.motion_lock:
            self.quit_requested = True
            self.request_stop(reason=RunEndReason.QUIT)

    def end_session(self, *, reason=RunEndReason.RUNTIME_SHUTDOWN) -> None:
        with self.motion_lock:
            self.running = False
            if self.safety_state == SafetyState.RUNNING:
                self.revoke_motion(reason=reason)
            else:
                self.stop_signal.set()
