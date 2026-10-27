"""父进程独占手部 I/O，机械臂 SDK 由子进程持有。"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.robot.commands import (
    OperationCancelled,
    RobotError,
    validate_arm_target,
    validate_hand_target,
)
from dexmani_real.robot.model import (
    XHAND_RATED_QPOS_MAX_RAD,
    XHAND_RATED_QPOS_MIN_RAD,
)
from dexmani_real.robot.xhand.feedback import XHandFeedback

if TYPE_CHECKING:
    from dexmani_real.robot.xarm7.process import ArmProcess
    from dexmani_real.robot.xhand.driver import XHand

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobotState:
    arm: np.ndarray
    hand: np.ndarray | None


class Robot:
    def __init__(
        self,
        arm_config,
        hand_config,
        *,
        run_signal,
        stop_signal,
        emergency_signal,
        signal_lock,
        check_services=None,
    ):
        self.arm_config, self.hand_config = arm_config, hand_config
        self.arm_lower = np.asarray(arm_config.joint_limit_lower, dtype=np.float64)
        self.arm_upper = np.asarray(arm_config.joint_limit_upper, dtype=np.float64)
        self.hand_lower = np.asarray(XHAND_RATED_QPOS_MIN_RAD, dtype=np.float64)
        self.hand_upper = np.asarray(XHAND_RATED_QPOS_MAX_RAD, dtype=np.float64)
        self.run_signal, self.stop_signal = run_signal, stop_signal
        self.emergency_signal, self.signal_lock = emergency_signal, signal_lock
        self.check_services = check_services
        self.arm: ArmProcess | None = None
        self.hand: XHand | None = None
        self.owner = None
        self.connected = False
        self.motion_active = False
        self.hand_stop_pending = False
        self.stop_failure = None
        self.hand_feedback = XHandFeedback(hand_config)
        self.idle_next_ns = 0
        self.input_signal = None

    @property
    def stop_required(self):
        return self.motion_active or self.hand_stop_pending

    def check_owner(self):
        if self.owner != threading.get_ident():
            raise RuntimeError("robot I/O belongs to its connecting thread")

    def check(self):
        self.check_owner()
        if self.check_services is not None and not self.check_services():
            raise RobotError("required runtime service unavailable")
        if self.arm is not None:
            self.arm.check_error()

    def check_run(self, run_id):
        if (
            self.run_signal.value != run_id
            or self.stop_signal.is_set()
            or self.emergency_signal.is_set()
        ):
            raise OperationCancelled("device operation cancelled")
        if self.arm is not None:
            self.arm.check_error()

    def connect(self):
        if self.owner is not None:
            raise RuntimeError("robot connection is single-use")
        self.owner = threading.get_ident()
        run_id = self.run_signal.value

        def cancelled():
            self.check()
            return (
                self.run_signal.value != run_id
                or self.stop_signal.is_set()
                or self.emergency_signal.is_set()
            )

        try:
            if cancelled():
                raise OperationCancelled("robot startup cancelled")
            from dexmani_real.robot.xarm7.process import ArmProcess

            self.arm = ArmProcess(
                self.arm_config,
                run_signal=self.run_signal,
                stop_signal=self.stop_signal,
                emergency_signal=self.emergency_signal,
                signal_lock=self.signal_lock,
                input_signal=self.input_signal,
            )
            self.arm.connect(check_services=self.check_services)
            if cancelled():
                raise OperationCancelled("robot startup cancelled before hand connection")
            deadline_ns = None
            if self.hand_config.enabled:
                from dexmani_real.robot.xhand.driver import XHand

                self.hand = XHand(self.hand_config)
                deadline_ns = self.hand.connect(cancel_requested=cancelled)
            self.connected = True
            self.reset_hand_feedback(deadline_ns=deadline_ns)
            self.read_initial_state(cancel_requested=cancelled)
        except BaseException:
            self.stop_signal.set()
            try:
                self.close()
            except Exception:
                logger.exception("partial robot connection cleanup failed")
            raise

    def reset_tactile_sensors(self, *, cancel_requested):
        self.check()
        if self.hand is None:
            return False
        self.stop()
        run_id = self.run_signal.value

        def cancelled():
            self.check()
            return bool(
                cancel_requested()
                or self.run_signal.value != run_id
                or self.emergency_signal.is_set()
            )

        try:
            reset = self.hand.reset_tactile_sensors(cancel_requested=cancelled)
        finally:
            self.reset_hand_feedback()
        if reset:
            self.read_initial_state(cancel_requested=cancelled)
        return reset

    def reset_hand_feedback(self, *, deadline_ns=None):
        self.check_owner()
        if self.motion_active:
            raise RuntimeError("cannot clear hand feedback during motion")
        self.hand_feedback.reset(time.monotonic_ns(), deadline_ns=deadline_ns)

    def read_hand(self):
        if self.hand is None:
            return None
        if not self.hand.is_connected:
            raise RobotError("XHand disconnected")
        query_started_ns = time.monotonic_ns()
        state = self.hand.get_state()
        query_completed_ns = time.monotonic_ns()
        return self.hand_feedback.consume(
            state,
            source_ns=query_started_ns,
            now_ns=query_completed_ns,
        )

    def read_state(self):
        self.check()
        if not self.connected or self.arm is None:
            raise RobotError("robot is not connected")
        try:
            return RobotState(self.arm.snapshot(), self.read_hand())
        except Exception as exc:
            self.stop_signal.set()
            raise RobotError(f"robot feedback failed: {exc}") from exc

    def read_initial_state(self, *, cancel_requested=None):
        self.check_owner()
        run_id = self.run_signal.value
        deadline_ns = self.hand_feedback.initializing_until_ns if self.hand is not None else None

        def check_wait():
            if (
                (cancel_requested is not None and cancel_requested())
                or self.run_signal.value != run_id
                or self.emergency_signal.is_set()
            ):
                raise OperationCancelled("XHand initial feedback acquisition cancelled")
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise RobotError("XHand initial feedback acquisition timed out")

        while True:
            check_wait()
            state = self.read_state()
            check_wait()
            if deadline_ns is None or state.hand is not None:
                return state
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(min(0.01, remaining_ns / 1e9))

    def prepare_streaming(self, run_id, *, service):
        self.check_owner()
        if self.hand_stop_pending or self.stop_failure is not None:
            raise RobotError("previous robot stop remains unresolved")
        self.end_hand_send_interval()
        self.motion_active = True
        cast("ArmProcess", self.arm).prepare(run_id, service=service)
        self.check_run(run_id)

    def send_action(self, command, *, valid_until_ns, execute_until_ns=None):
        self.check_owner()
        if self.hand_stop_pending or self.stop_failure is not None:
            raise RobotError("previous robot stop remains unresolved")
        if command.arm_qpos is not None:
            if not self.connected or self.arm is None or not self.arm.is_connected:
                raise RobotError("arm device unavailable")
            validate_arm_target(command.arm_qpos, self.arm_lower, self.arm_upper)
            if execute_until_ns is None or execute_until_ns <= 0:
                raise ValueError("arm requires an explicit decision execution deadline")
        if command.hand_qpos is not None:
            if not self.connected or self.hand is None or not self.hand.is_connected:
                raise RobotError("hand device unavailable")
            validate_hand_target(command.hand_qpos, self.hand_lower, self.hand_upper)

        def check_send():
            self.check_run(command.run_id)
            now = time.monotonic_ns()
            if now >= valid_until_ns or (execute_until_ns is not None and now >= execute_until_ns):
                raise TimeoutError("target submission deadline expired")
            self.motion_active = True

        try:
            if command.arm_qpos is not None:
                check_send()
                cast("ArmProcess", self.arm).submit(
                    command, submit_until_ns=valid_until_ns, execute_until_ns=execute_until_ns
                )
            if command.hand_qpos is not None:
                check_send()
                try:
                    cast("XHand", self.hand).send_action(command.hand_qpos)
                except Exception as exc:
                    raise RobotError(f"XHand send failed: {exc}") from exc
            self.check_run(command.run_id)
        except BaseException:
            self.stop_signal.set()
            raise

    def end_hand_send_interval(self):
        self.check_owner()
        if self.hand is not None:
            self.hand.end_send_interval()

    def send_hand_home(self, command, *, valid_until_ns):
        self.check_owner()
        if command.arm_qpos is not None:
            raise ValueError("hand HOME cannot carry an arm streaming target")
        if self.hand_stop_pending or self.stop_failure is not None:
            raise RobotError("previous robot stop remains unresolved")
        cast("ArmProcess", self.arm).begin_operation(command.run_id)
        self.end_hand_send_interval()
        self.send_action(command, valid_until_ns=valid_until_ns)

    def home_arm(self, waypoints, target, run_id, abort_check, *, service, deadline_ns):
        self.check_owner()
        if self.hand_stop_pending or self.stop_failure is not None:
            raise RobotError("previous robot stop remains unresolved")
        if abort_check():
            raise OperationCancelled("arm HOME cancelled before submission")
        self.motion_active = True
        cast("ArmProcess", self.arm).home(
            waypoints,
            target,
            run_id,
            abort=abort_check,
            service=service,
            deadline_ns=deadline_ns,
        )
        if abort_check():
            raise OperationCancelled("arm HOME cancelled during execution")

    def stop_hand(self, *, passive=False):
        if self.hand is None:
            if self.hand_stop_pending:
                raise RobotError("XHand stop unresolved after disconnect")
            return
        self.hand_stop_pending = True
        frame = self.hand_feedback.latest
        if not passive and frame is not None:
            self.hand.stop(frame["qpos"][0], timestamp_ns=int(frame["timestamp_ns"][0]))
        else:
            self.hand.stop()
        self.hand_stop_pending = False

    def stop(self):
        self.check_owner()
        self.stop_signal.set()
        self.end_hand_send_interval()
        errors = []
        unresolved_motion = self.motion_active
        if self.arm is not None or self.hand is not None:
            self.motion_active = True
        if self.arm is not None:
            self.arm.request_stop()
        try:
            self.stop_hand()
        except Exception as exc:
            errors.append(exc)
        if self.arm is not None:
            try:
                self.arm.wait_stopped()
            except Exception as exc:
                errors.append(exc)
        if self.arm is None and self.hand is None and unresolved_motion:
            errors.append(RobotError("motion remains unresolved after disconnect"))
        if errors:
            self.stop_failure = self.stop_failure or errors[0]
            raise RobotError(f"robot stop failures: {errors}") from errors[0]
        self.motion_active = False

    def service_idle(self):
        self.end_hand_send_interval()
        if self.motion_active and (self.stop_signal.is_set() or self.emergency_signal.is_set()):
            self.stop()
        now = time.monotonic_ns()
        if now >= self.idle_next_ns and self.connected:
            self.idle_next_ns = now + int(1e9 / 30)
            self.read_state()

    def close(self):
        if self.owner is None:
            return
        self.check_owner()
        errors = []
        try:
            self.stop()
        except Exception as exc:
            errors.append(exc)
        if self.arm is not None:
            try:
                self.arm.close()
            except Exception as exc:
                errors.append(exc)
            else:
                self.arm = None

        if self.hand is not None:
            try:
                if self.hand.is_connected:
                    self.stop_hand(passive=True)
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    self.hand.disconnect()
                except Exception as exc:
                    errors.append(exc)
                else:
                    self.hand = None

        self.connected = False
        if errors:
            self.motion_active = True
            raise RobotError(f"robot close failures: {errors}") from errors[0]
