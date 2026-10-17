"""Local motion authority; only sensor channels and the xArm authority signal cross spawn."""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from threading import RLock

from dexmani_real.ipc.channels import SensorChannels


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
    sensors: SensorChannels | None = None
    running: bool = True
    run_id: int = 1
    run_ended_id: int = 0
    run_ended_reason: RunEndReason = RunEndReason.NONE
    error_state: bool = False
    estop_request: bool = False
    quit_requested: bool = False
    start_request: bool = False
    stop_request: bool = False
    safety_state: SafetyState = SafetyState.DISARMED
    # Never held across an SDK call; keyboard callbacks revoke from another thread.
    motion_lock: object = field(default_factory=RLock, repr=False)
    arm_authority: object = field(default=None, repr=False)
    arm_emergency: object = field(default=None, repr=False)

    def _requests_allow_motion(self):
        return (self.running and not self.error_state and not self.estop_request
                and not self.quit_requested and not self.stop_request)

    def arm(self) -> None:
        with self.motion_lock:
            if not self._requests_allow_motion() or not self.revoke_motion():
                raise RuntimeError("arming rejected")

    def begin_motion(self) -> tuple[int, int] | None:
        with self.motion_lock:
            if self.safety_state != SafetyState.ARMED or not self._requests_allow_motion():
                return None
            self.run_id += 1
            started = time.monotonic_ns()
            self.run_ended_reason = RunEndReason.NONE
            self.safety_state = SafetyState.RUNNING
            self.publish_arm_authority()
            return self.run_id, started

    def command_may_cross_sdk(self, *, run_id, required_safety_state=SafetyState.RUNNING):
        """Last software fence; the subsequent SDK call runs without this lock."""
        with self.motion_lock:
            return (
                self._requests_allow_motion()
                and self.run_id == run_id
                and self.safety_state == required_safety_state
                and required_safety_state in (SafetyState.ARMED, SafetyState.RUNNING)
            )

    def revoke_motion(
        self, new_state=SafetyState.ARMED, *,
        reason=RunEndReason.EXECUTOR_BOUNDARY, expected_run_id=None,
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
            self.run_id += 1
            self.safety_state = new_state
            self.publish_arm_authority()
            return True

    def publish_arm_authority(self):
        """Signed epoch: ARMED preparation < 0, RUNNING > 0, revoked = 0."""
        with self.motion_lock:
            if self.arm_authority is not None:
                if not self._requests_allow_motion():
                    self.arm_authority.value = 0
                elif self.safety_state == SafetyState.RUNNING:
                    self.arm_authority.value = self.run_id
                elif self.safety_state == SafetyState.ARMED:
                    self.arm_authority.value = -self.run_id
                else:
                    self.arm_authority.value = 0
            if self.estop_request and self.arm_emergency is not None:
                self.arm_emergency.set()

    def request_emergency_stop(self):
        with self.motion_lock:
            already_requested = self.estop_request
            self.estop_request = True
            self.publish_arm_authority()
            if not already_requested:
                self.revoke_motion(reason=RunEndReason.ESTOP)

    def end_session(self, *, reason=RunEndReason.RUNTIME_SHUTDOWN, fault=False) -> None:
        """Fence motion and notify sensor workers before any blocking cleanup."""
        with self.motion_lock:
            self.running = False
            if self.sensors is not None:
                self.sensors.is_running.value = False
            if fault:
                self.error_state = True
                self.revoke_motion(SafetyState.FAULT, reason=reason)
            elif self.safety_state == SafetyState.RUNNING:
                self.revoke_motion(reason=reason)
            else:
                self.publish_arm_authority()
