"""Low-frequency coordination: submit xArm goals and send XHand targets once."""

import logging
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import (
    DispatchStatus,
    DispatchResult,
    DispatchError,
    DispatchInterrupted,
    check_arm_target,
    check_hand_target,
)
from dexmani_real.robot.xhand.feedback import XHandFeedback, hand_feedback_time_ns
from dexmani_real.runtime.state import RunEndReason, SafetyState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobotState:
    arm: np.ndarray
    hand: np.ndarray | None


class DexManiRobot:
    @property
    def stop_required(self):
        """Software cleanup requirement, not evidence of physical standstill."""
        return self.motion_active or self.hand_stop_pending

    def __init__(
        self, runtime_state, runtime, *, arm_factory=None, hand_factory=None, check_services=None
    ):
        self.runtime_state, self.runtime = runtime_state, runtime
        self.arm_factory, self.hand_factory = arm_factory, hand_factory
        self.check_services = check_services
        self.arm = self.hand = None
        self.owner = None
        self.connected = False
        self.motion_active = False
        self.hand_stop_pending = False
        self.hand_feedback = XHandFeedback(runtime.hand)
        self.hand_send_failure_started_ns = None
        self.idle_next_ns = 0

    def check_owner(self):
        if self.owner != threading.get_ident():
            raise RuntimeError("robot I/O belongs to its connecting thread")

    def check(self):
        self.check_owner()
        self.runtime_state.publish_arm_authority()
        if self.check_services is not None and not self.check_services():
            # The service owner applies its own fail-safe and records the cause.
            raise RuntimeError("required runtime service unavailable")
        if self.arm is not None:
            self.arm.update_health(
                service_until_ns=time.monotonic_ns()
                + int(self.runtime.arm.feedback_max_age_s * 1e9)
            )
            self.arm.check_error()

    def connect(self):
        if self.owner is not None:
            raise RuntimeError("robot connection is single-use")
        self.owner = threading.get_ident()
        epoch = self.runtime_state.run_id

        def cancelled():
            self.check()
            return self.initial_state_cancelled(epoch)

        try:
            from dexmani_real.robot.xarm7.executor import ArmExecutor

            self.arm = ArmExecutor(
                self.runtime.arm, self.runtime_state, driver_factory=self.arm_factory
            )
            self.arm.connect()
            self.check()
            deadline_ns = None
            if self.runtime.control.hand_enabled:
                if self.hand_factory is None:
                    from dexmani_real.robot.xhand.driver import XHand

                    self.hand_factory = XHand
                self.hand = self.hand_factory(self.runtime.hand)
                deadline_ns = self.hand.connect(cancel_requested=cancelled)
            self.connected = True
            # Joint command seeding and complete tactile acquisition share one budget.
            self.reset_hand_feedback(deadline_ns=deadline_ns)
            self.read_initial_state(cancel_requested=cancelled)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            elif isinstance(exc, Exception):
                self.runtime_state.mark_error()
            try:
                self.close()
            except Exception:
                logger.exception("partial robot connection cleanup failed")
            raise

    def reset_tactile_sensors(self, *, cancel_requested):
        """Explicit idle-owner preparation; never HOME or move fingers for reset."""
        self.check()
        if self.runtime_state.safety_state != SafetyState.ARMED:
            raise RuntimeError("tactile reset requires idle ARMED state")
        if self.hand is None:
            return False
        self.runtime_state.revoke_motion()
        epoch = self.runtime_state.run_id
        self.stop()

        def cancelled():
            self.check()
            return bool(cancel_requested() or self.initial_state_cancelled(epoch))

        try:
            reset = self.hand.reset_tactile_sensors(cancel_requested=cancelled)
        finally:
            # Even a partial/cancelled reset invalidates the previous tactile baseline.
            self.reset_hand_feedback()
        if reset:
            self.read_initial_state(cancel_requested=cancelled)
        return reset

    def reset_hand_feedback(self, *, deadline_ns=None):
        """Reacquire feedback without resetting or moving hardware."""
        self.check_owner()
        if self.runtime_state.safety_state == SafetyState.RUNNING:
            raise RuntimeError("cannot clear hand feedback during motion")
        self.hand_feedback.reset(time.monotonic_ns(), deadline_ns=deadline_ns)

    def read_hand(self):
        if self.hand is None:
            return None
        if not self.hand.is_connected:
            raise RuntimeError("XHand disconnected")
        state = self.hand.get_state()
        frame = self.hand_feedback.consume(state, time.monotonic_ns())
        if frame is not None and self.arm is not None:
            self.arm.update_health(
                hand_until_ns=hand_feedback_time_ns(frame)
                + int(self.runtime.hand.feedback_max_age_s * 1e9)
            )
        return frame

    def read_state(self):
        # One service scan per feedback sample; SDK dispatch uses local flags.
        self.check()
        if not self.connected:
            raise RuntimeError("robot is not connected")
        try:
            arm = self.arm.snapshot()
            hand = self.read_hand()
        except Exception:
            self.runtime_state.mark_error()
            self.runtime_state.revoke_motion(SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise
        return RobotState(arm, hand)

    def initial_state_cancelled(self, epoch):
        return bool(
            epoch != self.runtime_state.run_id
            or self.runtime_state.stop_request
            or self.runtime_state.estop_request
            or self.runtime_state.error_state
            or self.runtime_state.quit_requested
            or not self.runtime_state.running
        )

    def read_initial_state(self, *, cancel_requested=None):
        """Wait only for XHand's bounded initial cache, without resetting its deadline."""
        self.check_owner()
        epoch = self.runtime_state.run_id
        # A valid read clears the shared initialization field, not this attempt's deadline.
        deadline_ns = self.hand_feedback.initializing_until_ns if self.hand is not None else None

        def check_cancelled():
            if (
                cancel_requested is not None and cancel_requested()
            ) or self.initial_state_cancelled(epoch):
                raise CancelledError("XHand initial feedback acquisition cancelled")

        def check_deadline():
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                # This timeout is outside read_state's exception fence; apply the same fault boundary.
                self.runtime_state.mark_error()
                self.runtime_state.revoke_motion(
                    SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT
                )
                raise RuntimeError(
                    f"XHand initial feedback timed out after {self.runtime.hand.initial_state_timeout_s:.3f}s"
                )

        while True:
            check_cancelled()
            check_deadline()
            state = self.read_state()
            check_cancelled()
            check_deadline()
            if self.hand is None or state.hand is not None:
                return state
            # Only the feedback cache can withhold a connected hand within this budget.
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(min(0.01, remaining_ns / 1e9))

    def prepare_streaming(self, run_id):
        """Prepare Mode 1 from measured standstill under the owner's epoch; the caller owns stop on failure."""
        self.check_owner()
        if self.hand_stop_pending:
            raise DispatchError(
                "previous XHand stop remains unconfirmed",
                DispatchResult(),
                revoked=True,
                cause="stop_unconfirmed",
            )
        if not self.connected or self.arm is None or not self.arm.is_connected:
            raise DispatchError("arm device unavailable", DispatchResult())

        def check_abort():
            self.check()
            if not self.runtime_state.command_may_cross_sdk(
                run_id=run_id, required_safety_state=SafetyState.ARMED
            ):
                raise DispatchError(
                    "motion authority revoked during streaming preparation",
                    DispatchResult(),
                    revoked=True,
                    cause="authority_revoked",
                )

        check_abort()
        self.end_hand_send_interval()
        self.motion_active = True
        # Waiting services the hand and health on this thread, never on xarm-owner.
        from dexmani_real.robot.xarm7.protocol import OperationCancelled

        def service():
            self.read_hand()
            self.check()

        try:
            self.arm.prepare(
                run_id,
                service=service,
                abort=lambda: not self.runtime_state.command_may_cross_sdk(
                    run_id=run_id,
                    required_safety_state=SafetyState.ARMED,
                ),
            )
        except OperationCancelled as exc:
            raise DispatchError(
                str(exc), DispatchResult(), revoked=True, cause="authority_revoked"
            ) from exc
        check_abort()

    def send(self, command, state, *, valid_until_ns, execute_until_ns=None):
        self.check_owner()
        arm_status, hand_status = DispatchStatus.NOT_CALLED, DispatchStatus.NOT_CALLED
        command_id = (-1, -1)

        def receipt():
            return DispatchResult(arm_status, hand_status, time.monotonic_ns(), command_id)

        def check_dispatch():
            if not self.runtime_state.command_may_cross_sdk(
                run_id=command.run_id, required_safety_state=state
            ):
                raise DispatchError(
                    "motion authority revoked", receipt(), revoked=True, cause="authority_revoked"
                )
            if self.arm is not None:
                self.arm.check_error()
            now = time.monotonic_ns()
            if now >= valid_until_ns or (execute_until_ns is not None and now >= execute_until_ns):
                raise DispatchError(
                    "dispatch deadline expired", receipt(), revoked=True, cause="deadline_expired"
                )
            self.motion_active = True

        try:
            if self.hand_stop_pending:
                raise DispatchError(
                    "previous XHand stop remains unconfirmed",
                    receipt(),
                    revoked=True,
                    cause="stop_unconfirmed",
                )
            # Validate both targets before submitting either device's command.
            if command.arm_qpos is not None:
                if not self.connected or self.arm is None or not self.arm.is_connected:
                    raise RuntimeError("arm device unavailable")
                issue = check_arm_target(
                    command.arm_qpos,
                    joint_limit_lower_rad=np.asarray(self.runtime.arm.joint_limit_lower),
                    joint_limit_upper_rad=np.asarray(self.runtime.arm.joint_limit_upper),
                )
                if issue:
                    raise ValueError(f"unsafe arm target: {issue}")
                if execute_until_ns is None or execute_until_ns <= 0:
                    raise ValueError("arm requires an explicit input/decision execution deadline")
            if command.hand_qpos is not None:
                if not self.connected or self.hand is None or not self.hand.is_connected:
                    raise RuntimeError("hand device unavailable")
                issue = check_hand_target(
                    command.hand_qpos,
                    mechanical_lower_rad=np.asarray(self.runtime.hand.mechanical_qpos_min_rad),
                    mechanical_upper_rad=np.asarray(self.runtime.hand.mechanical_qpos_max_rad),
                )
                if issue:
                    raise ValueError(f"unsafe hand target: {issue}")
            if command.arm_qpos is not None:
                check_dispatch()
                command_id = self.arm.submit(
                    command, submit_until_ns=valid_until_ns, execute_until_ns=execute_until_ns
                )
                arm_status = DispatchStatus.SUBMITTED
            if command.hand_qpos is not None:
                check_dispatch()
                hand_status = DispatchStatus.UNKNOWN
                hand_status = self.hand.send_action(command.hand_qpos)
                self.check_hand_send(hand_status)
                if hand_status in (DispatchStatus.REJECTED, DispatchStatus.UNKNOWN):
                    raise RuntimeError("hand SDK rejected target")
            if not self.runtime_state.command_may_cross_sdk(
                run_id=command.run_id, required_safety_state=state
            ):
                raise DispatchError(
                    "motion authority revoked during dispatch",
                    receipt(),
                    revoked=True,
                    cause="authority_revoked",
                )
            return receipt()
        except KeyboardInterrupt as exc:
            if self.arm is not None:
                self.arm.request_stop()
            raise DispatchInterrupted(receipt()) from exc
        except Exception as exc:
            if self.arm is not None and command.arm_qpos is not None:
                self.arm.request_stop()
            result = receipt()
            if isinstance(exc, DispatchError):
                exc.result = result
                raise
            raise DispatchError(str(exc), result) from exc

    def check_hand_send(self, status):
        """Bound consecutive ordinary-send CRCs without retries or readback."""
        now_ns = time.monotonic_ns()
        budget_ns = int(self.runtime.hand.feedback_max_age_s * 1e9)
        if status == DispatchStatus.CRC_UNCONFIRMED:
            if self.hand_send_failure_started_ns is None:
                self.hand_send_failure_started_ns = now_ns
                logger.warning("XHand send CRC; delivery unconfirmed, continuing with next target")
            if now_ns - self.hand_send_failure_started_ns >= budget_ns:
                self.runtime_state.mark_error()
                self.runtime_state.revoke_motion(
                    SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT
                )
                raise RuntimeError(
                    "XHand continuous send CRC timed out; delivery remains unconfirmed"
                )
        elif status == DispatchStatus.ACCEPTED:
            if self.hand_send_failure_started_ns is not None:
                logger.info("XHand send responses recovered")
            self.hand_send_failure_started_ns = None

    def end_hand_send_interval(self):
        """An explicit idle/WAIT is not part of a continuous ordinary-send failure."""
        self.check_owner()
        self.hand_send_failure_started_ns = None

    def send_action(self, command, *, valid_until_ns, execute_until_ns):
        return self.send(
            command,
            SafetyState.RUNNING,
            valid_until_ns=valid_until_ns,
            execute_until_ns=execute_until_ns,
        )

    def send_hand_home(self, command, *, valid_until_ns):
        if command.arm_qpos is not None:
            raise ValueError("hand HOME cannot carry an arm streaming target")
        # HOME submits once, then its caller checks measured convergence separately.
        self.end_hand_send_interval()
        return self.send(command, SafetyState.ARMED, valid_until_ns=valid_until_ns)

    def home_arm(self, waypoints, target, run_id, abort_check):
        self.check_owner()
        if self.hand_stop_pending:
            raise DispatchError(
                "previous XHand stop remains unconfirmed",
                DispatchResult(),
                revoked=True,
                cause="stop_unconfirmed",
            )

        def check_abort():
            self.check()
            if abort_check() or not self.runtime_state.command_may_cross_sdk(
                run_id=run_id, required_safety_state=SafetyState.ARMED
            ):
                return "home authority revoked"
            return None

        def service():
            self.read_hand()
            self.check()

        if check_abort():
            return False
        self.motion_active = True
        self.arm.home(waypoints, target, run_id, abort=check_abort, service=service)
        return check_abort() is None

    def stop_hand(self, *, passive=False):
        if self.hand is None:
            if self.hand_stop_pending:
                raise RuntimeError("XHand stop unresolved after disconnect")
            return
        self.hand_stop_pending = True
        frame = self.hand_feedback.latest
        if not passive and frame is not None:
            self.hand.stop(frame["qpos"][0], timestamp_ns=int(frame["timestamp_ns"][0]))
        else:
            self.hand.stop()
        # Exceptions and interrupts retain the fence against subsequent motion.
        self.hand_stop_pending = False

    def stop(self):
        # XHand stop retains its bounded protocol retries; ordinary sends never retry.
        self.end_hand_send_interval()
        if self.runtime_state.safety_state == SafetyState.RUNNING:
            self.runtime_state.revoke_motion()
        errors = []
        # A failed/interrupted stop must not advertise a clean software stop.
        unresolved_motion = self.motion_active
        if self.arm is not None or self.hand is not None:
            self.motion_active = True
        if self.arm is not None:
            try:
                self.arm.request_stop()
            except Exception as exc:
                errors.append(exc)
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
            errors.append(RuntimeError("motion remains unresolved after disconnect"))
        if errors:
            self.motion_active = True
            self.runtime_state.mark_error()
            self.runtime_state.revoke_motion(SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise RuntimeError(f"robot stop failures: {errors}") from errors[0]
        self.motion_active = False

    def service_idle(self):
        self.end_hand_send_interval()
        if self.motion_active and (
            self.runtime_state.estop_request
            or self.runtime_state.error_state
            or self.runtime_state.quit_requested
            or self.runtime_state.safety_state != SafetyState.RUNNING
        ):
            self.stop()
        now = time.monotonic_ns()
        if now >= self.idle_next_ns and self.connected:
            self.idle_next_ns = now + int(1e9 / 30)
            self.read_state()
            if self.hand_stop_pending:
                self.stop_hand()

    def close(self):
        if self.owner is None:
            return
        self.check_owner()
        errors = []
        try:
            self.stop()
        except Exception as exc:
            errors.append(exc)
        for name, method in (("arm", "close"), ("hand", "disconnect")):
            device = getattr(self, name)
            if device is not None:
                try:
                    if name == "hand" and device.is_connected:
                        self.stop_hand(passive=True)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    try:
                        getattr(device, method)()
                    except Exception as exc:
                        errors.append(exc)
                    else:
                        setattr(self, name, None)
        self.connected = False
        if errors:
            self.motion_active = True
            self.runtime_state.mark_error()
            raise RuntimeError(f"robot close failures: {errors}") from errors[0]
