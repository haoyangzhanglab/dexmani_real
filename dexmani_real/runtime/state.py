"""Coordinator-owned lifecycle; only run identity and stop signals cross spawn."""

from __future__ import annotations

import multiprocessing as mp
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event


class SafetyState(IntEnum):
    DISARMED = 0
    ARMED = 1
    RUNNING = 2
    FAULT = 3


class RunEndReason(IntEnum):
    """First software RUNNING termination cause; never physical convergence."""

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


@dataclass
class RuntimeState:
    running: bool = True
    run_ended_id: int = 0
    run_ended_reason: RunEndReason = RunEndReason.NONE
    error_state: bool = False
    estop_request: bool = False
    quit_requested: bool = False
    stop_request: bool = False
    safety_state: SafetyState = SafetyState.DISARMED
    # The shared lock protects only identity/stop transitions, never SDK I/O.
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
            if not self.requests_allow_motion() or not self.revoke_motion():
                raise RuntimeError("arming rejected")

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

    def command_may_cross_sdk(self, *, run_id, required_safety_state=SafetyState.RUNNING):
        """Last software fence; the subsequent SDK call runs without this lock."""
        with self.motion_lock:
            return (
                self.requests_allow_motion()
                and self.run_id == run_id
                and self.safety_state == required_safety_state
                and required_safety_state in (SafetyState.ARMED, SafetyState.RUNNING)
                and (required_safety_state != SafetyState.RUNNING or not self.stop_signal.is_set())
            )

    def revoke_motion(
        self,
        new_state=SafetyState.ARMED,
        *,
        reason=RunEndReason.EXECUTOR_BOUNDARY,
        expected_run_id=None,
    ) -> bool:
        with self.motion_lock:
            if new_state not in (SafetyState.ARMED, SafetyState.DISARMED, SafetyState.FAULT):
                raise ValueError("revocation must leave streaming mode")
            if expected_run_id is not None and self.run_id != expected_run_id:
                return False
            if self.safety_state == SafetyState.FAULT and new_state == SafetyState.ARMED:
                return False
            if self.safety_state == SafetyState.RUNNING:
                if self.estop_request:
                    reason = RunEndReason.ESTOP
                elif new_state == SafetyState.FAULT and reason == RunEndReason.EXECUTOR_BOUNDARY:
                    reason = RunEndReason.HARDWARE_FAULT
                self.run_ended_id = self.run_id
                self.run_ended_reason = reason
            self.stop_signal.set()
            self.run_signal.value += 1
            self.safety_state = new_state
            return True

    def request_emergency_stop(self):
        # Reach the SDK owner before any local authority bookkeeping can wait.
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
        """Latch a session error without conflating it with a hardware FAULT."""
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
        """Only an owner which finished stopping may clear this generation's fence."""
        with self.motion_lock:
            if self.run_id == expected_run_id and self.safety_state != SafetyState.RUNNING:
                self.stop_request = False

    def request_quit(self):
        with self.motion_lock:
            self.quit_requested = True
            self.request_stop(reason=RunEndReason.QUIT)

    def end_session(self, *, reason=RunEndReason.RUNTIME_SHUTDOWN, fault=False) -> None:
        """Fence motion before the resource owner notifies workers or closes devices."""
        with self.motion_lock:
            self.running = False
            if fault:
                self.mark_error()
                self.revoke_motion(SafetyState.FAULT, reason=reason)
            elif self.safety_state == SafetyState.RUNNING:
                self.revoke_motion(reason=reason)
            else:
                self.stop_signal.set()
