"""Single application-thread owner of the existing arm and optional hand drivers."""

import threading
import time
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.robot.command_validation import check_arm_target, check_hand_target
from dexmani_real.robot.model import XARM7_HARD_LOWER, XARM7_HARD_UPPER
from dexmani_real.runtime.observation import sample_is_fresh
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    command_may_cross_sdk,
    revoke_motion,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Stop retries belong to the synchronous I/O owner, outside normal motion pacing.
_HAND_STOP_RETRY_INTERVAL_S = 0.01
_HAND_STOP_RETRY_TIMEOUT_S = 0.25
_HAND_STOP_MAX_ATTEMPTS = 26


class DispatchStatus(IntEnum):
    NOT_CALLED = 0
    ACCEPTED = 1
    CRC_UNCONFIRMED = 2
    REJECTED = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class DispatchResult:
    arm: DispatchStatus = DispatchStatus.NOT_CALLED
    hand: DispatchStatus = DispatchStatus.NOT_CALLED
    arm_code: int | None = None
    hand_status: object = None
    timestamp_ns: int = 0  # host dispatch completion, never arrival

    @property
    def continued(self):
        return (
            self.arm not in (DispatchStatus.REJECTED, DispatchStatus.UNKNOWN)
            and self.hand not in (DispatchStatus.REJECTED, DispatchStatus.UNKNOWN)
            and (self.arm != DispatchStatus.NOT_CALLED or self.hand != DispatchStatus.NOT_CALLED)
        )


class DispatchError(RuntimeError):
    def __init__(self, message, result, *, revoked=False, cause=None):
        super().__init__(message)
        self.result = result
        self.revoked = revoked
        self.cause = cause


class DispatchInterrupted(KeyboardInterrupt):
    """Cancellation evidence local to one attempted dispatch."""

    def __init__(self, result):
        super().__init__("dispatch interrupted")
        self.result = result


@dataclass(frozen=True)
class RobotState:
    arm: np.ndarray
    hand: np.ndarray | None


class DexManiRobot:
    @property
    def stop_required(self):
        """Software cleanup requirement, not evidence of physical standstill."""
        return self._motion_active or self._hand_stop_pending

    def __init__(
        self, shared, runtime, *, arm_factory=None, hand_factory=None, check_services=None
    ):
        from dexmani_real.config.experiment import validate_robot_config

        validate_robot_config(runtime)
        self.shared, self.runtime = shared, runtime
        self._arm_factory, self._hand_factory = arm_factory, hand_factory
        self.check_services = check_services
        self.arm = self.hand = None
        self._owner = None
        self._connected = False
        self._arm_stopped = False
        self._motion_active = False
        self._hand_stop_pending = False
        self._last_hand = None
        self._hand_failure_started = None
        self._previous_errors = None
        self._tactile_availability = {}
        self._idle_next_ns = 0

    def _check_owner(self):
        if self._owner != threading.get_ident():
            raise RuntimeError("robot I/O belongs to its connecting thread")

    def check(self):
        self._check_owner()
        if self.check_services is not None and not self.check_services():
            # The service owner applies its own fail-safe and records the cause.
            raise RuntimeError("required runtime service unavailable")

    def connect(self):
        if self._owner is not None:
            raise RuntimeError("robot connection is single-use")
        self._owner = threading.get_ident()
        try:
            if self._arm_factory is None:
                from dexmani_real.robot.drivers.xarm7 import XArm7

                self._arm_factory = XArm7
            self.arm = self._arm_factory(self.runtime.arm)
            self.arm.connect()
            self.check()
            if self.runtime.policy.hand_enabled:
                if self._hand_factory is None:
                    from dexmani_real.robot.drivers.xhand import XHand

                    self._hand_factory = XHand
                self.hand = self._hand_factory(self.runtime.hand)
                self.hand.connect()
            self._connected = True
            deadline = time.monotonic() + self.runtime.hand.state_read_failure_timeout_s
            while True:
                self.check()
                state = self.read_state()
                if self.hand is None or state.hand is not None:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("initial hand feedback unavailable")
                time.sleep(0.01)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.shared.estop_request.value = True
            elif isinstance(exc, Exception):
                self.shared.error_state.value = True
            try:
                self.close()
            except Exception:
                logger.exception("partial robot connection cleanup failed")
            raise

    def tare_tactile(self, *, cancel_requested):
        """Explicit idle-owner preparation; never HOME or move fingers for tare."""
        self.check()
        if int(self.shared.safety_state.value) != int(SafetyState.ARMED):
            raise RuntimeError("tactile tare requires idle ARMED state")
        if self.hand is None:
            return False, False
        revoke_motion(self.shared)
        self.stop()

        def cancelled():
            self.check()
            return bool(
                cancel_requested()
                or not self.shared.is_running.value
                or self.shared.stop_request.value
                or self.shared.estop_request.value
                or self.shared.error_state.value
                or self.shared.quit_requested.value
            )

        self._last_hand = None
        return self.hand.tare_tactile(cancel_requested=cancelled)

    def arm_feedback(self, qpos, qvel, effort):
        if any(np.shape(x) != (7,) or not np.isfinite(x).all() for x in (qpos, qvel)):
            raise RuntimeError("unusable arm feedback")
        if np.shape(effort) != (7,):
            raise RuntimeError("invalid arm effort shape")
        frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
        frame["qpos"], frame["qvel"], frame["effort"] = qpos, qvel, effort
        frame["timestamp_ns"] = time.monotonic_ns()
        frame.flags.writeable = False
        return frame

    def _read_hand(self):
        if self.hand is None:
            return None
        state = self.hand.get_state()
        if state is None:
            now = time.monotonic()
            if self._hand_failure_started is None:
                self._hand_failure_started = now
            if now - self._hand_failure_started >= self.runtime.hand.state_read_failure_timeout_s:
                raise RuntimeError("hand joint feedback timed out")
            # Cached feedback is only for stop/hold, never a current control observation.
            return None
        self._hand_failure_started = None
        if not self.hand.is_connected:
            raise RuntimeError("XHand disconnected")
        if np.shape(state.qpos) != (12,) or not np.isfinite(state.qpos).all():
            raise RuntimeError("unusable hand joint feedback")
        if np.shape(state.current_ma) != (12,):
            raise RuntimeError("invalid hand current shape")
        errors = tuple(
            tuple(getattr(state, name))
            for name in ("commboard_err", "jointboard_err", "tipboard_err")
        )
        if errors != self._previous_errors and any(any(row) for row in errors):
            logger.warning("XHand board errors (comm, joint, tip): %s", errors)
        self._previous_errors = errors
        frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
        frame["qpos"], frame["current"] = state.qpos, state.current_ma
        for name, shape in (("aggregate", (5, 3)), ("dense", (5, 120, 3))):
            value = getattr(state, f"tactile_{name}")
            valid = bool(getattr(state, f"tactile_{name}_valid"))
            if valid and (np.shape(value) != shape or not np.isfinite(value).all()):
                raise RuntimeError(f"unusable valid {name} tactile feedback")
            if self._tactile_availability.get(name) != valid:
                logger.log(20 if valid else 30, "XHand %s tactile available=%s", name, valid)
                self._tactile_availability[name] = valid
            frame[f"tactile_{name}_valid"] = valid
            frame[f"tactile_{name}"] = value if valid else np.nan
        frame["timestamp_ns"] = time.monotonic_ns()
        frame.flags.writeable = False
        self._last_hand = frame
        return frame

    def read_state(self):
        self.check()
        if not self._connected:
            raise RuntimeError("robot is not connected")
        try:
            q, v, effort = self.arm.read()
            if self.arm.error_code:
                raise RuntimeError(f"arm controller error: {self.arm.error_code}")
            arm = self.arm_feedback(q, v, effort)
            hand = self._read_hand()
        except Exception:
            self.shared.error_state.value = True
            revoke_motion(self.shared, SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise
        self.check()
        return RobotState(arm, hand)

    def _authorized(self, command, state):
        self.check()
        return (
            not self.shared.quit_requested.value
            and not self.shared.stop_request.value
            and command_may_cross_sdk(
                self.shared, run_id=command.run_id, required_safety_state=state
            )
        )

    def _send(self, command, state, *, valid_until_ns):
        self._check_owner()
        result = DispatchResult()
        try:
            if self._hand_stop_pending:
                raise DispatchError(
                    "previous XHand stop remains unconfirmed",
                    result,
                    revoked=True,
                    cause="stop_unconfirmed",
                )
            # All present targets are checked before either SDK sees a target.
            for name in ("arm", "hand"):
                target = getattr(command, f"{name}_qpos")
                if target is None:
                    continue
                device = getattr(self, name)
                if not self._connected or device is None or not device.is_connected:
                    raise DispatchError(f"{name} device unavailable", result)
                issue = (
                    check_arm_target(
                        target,
                        joint_limit_lower_rad=np.asarray(XARM7_HARD_LOWER),
                        joint_limit_upper_rad=np.asarray(XARM7_HARD_UPPER),
                    )
                    if name == "arm"
                    else check_hand_target(
                        target,
                        mechanical_lower_rad=np.asarray(self.runtime.hand.mechanical_qpos_min_rad),
                        mechanical_upper_rad=np.asarray(self.runtime.hand.mechanical_qpos_max_rad),
                    )
                )
                if issue:
                    raise DispatchError(f"unsafe {name} target: {issue}", result)
            for name in ("arm", "hand"):
                target = getattr(command, f"{name}_qpos")
                if target is None:
                    continue
                if not self._authorized(command, state):
                    raise DispatchError(
                        "motion authority revoked", result, revoked=True, cause="authority_revoked"
                    )
                if time.monotonic_ns() >= valid_until_ns:
                    raise DispatchError(
                        "dispatch deadline expired", result, revoked=True, cause="deadline_expired"
                    )
                if name == "arm" and self._arm_stopped:
                    self.arm.enter_mode6()
                    self._arm_stopped = False
                    if not self._authorized(command, state):
                        raise DispatchError(
                            "motion authority revoked after mode restoration",
                            result,
                            revoked=True,
                            cause="authority_revoked",
                        )
                    if time.monotonic_ns() >= valid_until_ns:
                        raise DispatchError(
                            "dispatch deadline expired after mode restoration",
                            result,
                            revoked=True,
                            cause="deadline_expired",
                        )
                # From here, an exception cannot prove that nothing reached the device.
                result = replace(result, **{name: DispatchStatus.UNKNOWN})
                self._motion_active = True
                if name == "arm":
                    code = self.arm.servo(target)
                    status = DispatchStatus.ACCEPTED if code == 0 else DispatchStatus.REJECTED
                    result = replace(result, arm=status, arm_code=code)
                else:
                    raw_status = self.hand.send_action(target)
                    status = DispatchStatus[raw_status.name]
                    result = replace(result, hand=status, hand_status=raw_status)
                if status in (DispatchStatus.REJECTED, DispatchStatus.UNKNOWN):
                    raise DispatchError(f"{name} SDK rejected target", result)
            if not self._authorized(command, state):
                raise DispatchError(
                    "motion authority revoked during dispatch",
                    result,
                    revoked=True,
                    cause="authority_revoked",
                )
            return replace(result, timestamp_ns=time.monotonic_ns())
        except KeyboardInterrupt as exc:
            raise DispatchInterrupted(replace(result, timestamp_ns=time.monotonic_ns())) from exc
        except Exception as exc:
            result = replace(result, timestamp_ns=time.monotonic_ns())
            raise DispatchError(
                str(exc),
                result,
                revoked=getattr(exc, "revoked", False),
                cause=getattr(exc, "cause", None),
            ) from exc

    def send_action(self, command, *, valid_until_ns):
        return self._send(command, SafetyState.RUNNING, valid_until_ns=valid_until_ns)

    def send_hand_home(self, command, *, valid_until_ns):
        if command.arm_qpos is not None:
            raise ValueError("hand HOME cannot carry an arm streaming target")
        return self._send(command, SafetyState.ARMED, valid_until_ns=valid_until_ns)

    def home_arm(self, waypoints, target, run_id, abort_check):
        self._check_owner()
        if self._hand_stop_pending:
            raise DispatchError(
                "previous XHand stop remains unconfirmed",
                DispatchResult(),
                revoked=True,
                cause="stop_unconfirmed",
            )

        def check_abort():
            self.check()
            if abort_check() or not command_may_cross_sdk(
                self.shared, run_id=run_id, required_safety_state=SafetyState.ARMED
            ):
                return "home authority revoked"
            return None

        def feedback(q, v, effort, _target):
            self.arm_feedback(q, v, effort)
            self._read_hand()
            self.check()

        if check_abort():
            return False
        self._motion_active = True
        self.arm.home(waypoints, target, abort_check=check_abort, feedback_callback=feedback)
        self._arm_stopped = False
        return check_abort() is None

    def _stop_hand(self, *, passive=False):
        if self.hand is None:
            if self._hand_stop_pending:
                raise RuntimeError("XHand stop unresolved after disconnect")
            return
        self._hand_stop_pending = True
        if not self.hand.is_connected:
            raise RuntimeError("XHand disconnected before stop")
        deadline = time.monotonic() + _HAND_STOP_RETRY_TIMEOUT_S
        for attempt in range(_HAND_STOP_MAX_ATTEMPTS):
            frame = self._last_hand
            if (
                not passive
                and frame is not None
                and sample_is_fresh(frame["timestamp_ns"][0], self.runtime.hand.feedback_max_age_s)
            ):
                status = self.hand.hold_current(frame["qpos"][0])
            else:
                status = self.hand.set_passive()
            if status.name == "ACCEPTED":
                self._hand_stop_pending = False
                return
            if status.name == "REJECTED":
                raise RuntimeError("XHand stop rejected")
            if status.name != "CRC_UNCONFIRMED":
                raise RuntimeError(f"unexpected XHand stop status: {status.name}")
            # CRC is delivery uncertainty, not rejection. Keep the motion fence
            # until confirmation; only budget exhaustion fails closed. Interrupts
            # propagate through SDK calls/sleep without clearing pending.
            remaining = deadline - time.monotonic()
            if remaining <= 0 or attempt + 1 == _HAND_STOP_MAX_ATTEMPTS:
                break
            time.sleep(min(_HAND_STOP_RETRY_INTERVAL_S, remaining))
        raise RuntimeError("XHand stop confirmation timed out: CRC_UNCONFIRMED")

    def stop(self):
        self._check_owner()
        errors = []
        # A failed/interrupted stop must not advertise a clean software stop.
        unresolved_motion = self._motion_active
        if self.arm is not None or self.hand is not None:
            self._motion_active = True
        if self.arm is not None:
            self._arm_stopped = False
            try:
                if self.shared.estop_request.value:
                    self.arm.emergency_stop()
                else:
                    self.arm.stop()
                self._arm_stopped = True
            except Exception as exc:
                errors.append(exc)
        try:
            self._stop_hand()
        except Exception as exc:
            errors.append(exc)
        if self.arm is None and self.hand is None and unresolved_motion:
            errors.append(RuntimeError("motion remains unresolved after disconnect"))
        if errors:
            self._motion_active = True
            self.shared.error_state.value = True
            revoke_motion(self.shared, SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise RuntimeError(f"robot stop failures: {errors}") from errors[0]
        self._motion_active = False

    def service_idle(self):
        self._check_owner()
        self.check()
        if self._motion_active and (
            self.shared.estop_request.value
            or self.shared.error_state.value
            or self.shared.quit_requested.value
            or int(self.shared.safety_state.value) != int(SafetyState.RUNNING)
        ):
            self.stop()
        now = time.monotonic_ns()
        if now >= self._idle_next_ns and self._connected:
            self._idle_next_ns = now + int(1e9 / 30)
            self.read_state()
            if self._hand_stop_pending:
                self._stop_hand()

    def close(self):
        self._check_owner()
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
                        self._stop_hand(passive=True)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    try:
                        getattr(device, method)()
                    except Exception as exc:
                        errors.append(exc)
                    setattr(self, name, None)
        self._connected = False
        if errors:
            self._motion_active = True
            self.shared.error_state.value = True
            raise RuntimeError(f"robot close failures: {errors}") from errors[0]
