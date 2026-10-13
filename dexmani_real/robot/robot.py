"""Single application-thread owner of the existing arm and optional hand drivers."""

import logging
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace

import numpy as np

from dexmani_real.robot.commands import DispatchStatus

from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.robot.command_validation import check_arm_target, check_hand_target
from dexmani_real.robot.model import XARM7_HARD_LOWER, XARM7_HARD_UPPER
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    command_may_cross_sdk,
    revoke_motion,
)


logger = logging.getLogger(__name__)


def _update_tactile_cache(raw, values, valid_ns, now_ns):
    """Hold whole three-axis vectors; only finite new vectors advance their clocks."""
    # Validate in the observation dtype too: finite float64 may overflow float32.
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.asarray(raw, dtype=values.dtype)
    if raw.shape != values.shape:
        raise RuntimeError(f"invalid tactile shape: {raw.shape}, expected {values.shape}")
    valid = np.isfinite(raw).all(axis=-1)
    np.copyto(values, raw, where=valid[..., None])
    np.copyto(valid_ns, now_ns, where=valid)
    return not valid.all()


@dataclass(frozen=True)
class DispatchResult:
    arm: DispatchStatus = DispatchStatus.NOT_CALLED
    hand: DispatchStatus = DispatchStatus.NOT_CALLED
    arm_code: int | None = None
    timestamp_ns: int = 0  # host dispatch completion, never arrival


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
        self._tactile_cache = {}
        self._tactile_held = set()
        self._hand_read_crc = False
        self._hand_initializing_until_ns = None
        self._hand_send_failure_started_ns = None
        self._previous_errors = None
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
        epoch = int(self.shared.run_id)

        def cancelled():
            self.check()
            return self._initial_state_cancelled(epoch)

        try:
            if self._arm_factory is None:
                from dexmani_real.robot.drivers.xarm7 import XArm7

                self._arm_factory = XArm7
            self.arm = self._arm_factory(self.runtime.arm)
            self.arm.connect()
            self.check()
            deadline_ns = None
            if self.runtime.policy.hand_enabled:
                if self._hand_factory is None:
                    from dexmani_real.robot.drivers.xhand import XHand

                    self._hand_factory = XHand
                self.hand = self._hand_factory(self.runtime.hand)
                deadline_ns = self.hand.connect(cancel_requested=cancelled)
            self._connected = True
            # Joint command seeding and complete tactile acquisition share one budget.
            self.reset_hand_feedback(deadline_ns=deadline_ns)
            self.read_initial_state(cancel_requested=cancelled)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.shared.estop_request = True
            elif isinstance(exc, Exception):
                self.shared.error_state = True
            try:
                self.close()
            except Exception:
                logger.exception("partial robot connection cleanup failed")
            raise

    def reset_tactile_sensors(self, *, cancel_requested):
        """Explicit idle-owner preparation; never HOME or move fingers for reset."""
        self.check()
        if int(self.shared.safety_state) != int(SafetyState.ARMED):
            raise RuntimeError("tactile reset requires idle ARMED state")
        if self.hand is None:
            return False
        revoke_motion(self.shared)
        epoch = int(self.shared.run_id)
        self.stop()

        def cancelled():
            self.check()
            return bool(
                cancel_requested()
                or self._initial_state_cancelled(epoch)
            )

        try:
            reset = self.hand.reset_tactile_sensors(cancel_requested=cancelled)
        finally:
            # Even a partial/cancelled reset invalidates the previous tactile baseline.
            self.reset_hand_feedback()
        if reset:
            self.read_initial_state(cancel_requested=cancelled)
        return reset

    def reset_hand_feedback(self, *, deadline_ns=None):
        """Reacquire a session's initial values without connecting, resetting or moving hardware."""
        self._check_owner()
        if int(self.shared.safety_state) == int(SafetyState.RUNNING):
            raise RuntimeError("cannot clear hand feedback during motion")
        self._last_hand = None
        self._tactile_cache = {
            name: (
                np.full(HAND_STATE_DTYPE[f"tactile_{name}"].shape, np.nan, dtype=np.float32),
                np.zeros(HAND_STATE_DTYPE[f"tactile_{name}"].shape[:-1], dtype=np.int64),
            )
            for name in ("aggregate", "dense")
        }
        self._tactile_held.clear()
        self._hand_read_crc = False
        self._hand_initializing_until_ns = (
            time.monotonic_ns() + int(self.runtime.hand.initial_state_timeout_s * 1e9)
            if deadline_ns is None else deadline_ns
        )

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
        if not self.hand.is_connected:
            raise RuntimeError("XHand disconnected")
        state = self.hand.get_state()
        now_ns = time.monotonic_ns()
        if state is None:
            if not self._hand_read_crc:
                logger.warning("XHand read CRC; holding previous feedback within age budget")
            self._hand_read_crc = True
            # Do not feed cached values back into the update function: no clock advances.
            return self._checked_hand_feedback(now_ns)
        if self._hand_read_crc:
            logger.info("XHand trustworthy reads recovered")
        self._hand_read_crc = False
        if np.shape(state.qpos) != (12,) or not np.isfinite(state.qpos).all():
            raise RuntimeError("unusable hand joint feedback")
        if np.shape(state.current_ma) != (12,):
            raise RuntimeError("invalid hand current shape")
        errors = tuple(
            tuple(getattr(state, name))
            for name in ("commboard_err", "jointboard_err", "tipboard_err")
        )
        if errors != self._previous_errors and any(any(row) for row in errors):
            logger.warning(
                "XHand board errors: %s",
                ", ".join(
                    f"{board}[{index}]={code}"
                    for board, row in zip(("comm", "joint", "tip"), errors)
                    for index, code in enumerate(row) if code
                ),
            )
        self._previous_errors = errors
        frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
        frame["qpos"], frame["current"] = state.qpos, state.current_ma
        for name, (values, valid_ns) in self._tactile_cache.items():
            held = _update_tactile_cache(getattr(state, f"tactile_{name}"), values, valid_ns, now_ns)
            if held and name not in self._tactile_held:
                logger.warning("XHand tactile %s incomplete; holding previous finite vectors", name)
                self._tactile_held.add(name)
            elif not held and name in self._tactile_held:
                logger.info("XHand tactile %s recovered", name)
                self._tactile_held.remove(name)
            # Assignment copies into an independent snapshot for recording/model workers.
            frame[f"tactile_{name}"] = values
            frame[f"tactile_{name}_timestamp_ns"] = valid_ns.min()
        frame["timestamp_ns"] = now_ns
        frame.flags.writeable = False
        self._last_hand = frame
        return self._checked_hand_feedback(now_ns)

    def _checked_hand_feedback(self, now_ns):
        frame = self._last_hand
        max_age_ns = int(self.runtime.hand.feedback_max_age_s * 1e9)
        issue = "hand joint feedback has no valid initial value"
        if frame is not None:
            issue = None
            for name in (
                "timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns",
            ):
                stamp = int(frame[name][0])
                if stamp <= 0 or now_ns - stamp >= max_age_ns:
                    issue = (
                        f"hand {name}: no valid initial value or feedback age reached "
                        f"{self.runtime.hand.feedback_max_age_s:.3f}s"
                    )
                    break
        if self._hand_initializing_until_ns is not None:
            if now_ns >= self._hand_initializing_until_ns:
                raise RuntimeError(
                    f"XHand initial feedback timed out after {self.runtime.hand.initial_state_timeout_s:.3f}s: "
                    f"{issue or 'sample completed after deadline'}"
                )
            if issue is not None:
                return None
        if issue is not None:
            raise RuntimeError(issue)
        self._hand_initializing_until_ns = None
        return frame

    def read_state(self):
        # One service scan per feedback sample; SDK dispatch uses local flags.
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
            self.shared.error_state = True
            revoke_motion(self.shared, SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise
        return RobotState(arm, hand)

    def _initial_state_cancelled(self, epoch):
        return bool(
            epoch != int(self.shared.run_id)
            or self.shared.stop_request
            or self.shared.estop_request
            or self.shared.error_state
            or self.shared.quit_requested
            or not self.shared.sensors.is_running.value
        )

    def read_initial_state(self, *, cancel_requested=None):
        """Wait only for XHand's bounded initial cache, without resetting its deadline."""
        epoch = int(self.shared.run_id)

        def check_cancelled():
            if (
                (cancel_requested is not None and cancel_requested())
                or self._initial_state_cancelled(epoch)
            ):
                raise CancelledError("XHand initial feedback acquisition cancelled")

        while True:
            check_cancelled()
            state = self.read_state()
            check_cancelled()
            if self.hand is None or state.hand is not None:
                return state
            # Only _checked_hand_feedback can withhold a connected hand within this budget.
            remaining_ns = self._hand_initializing_until_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(min(0.01, remaining_ns / 1e9))

    def _authorized(self, run_id, state):
        return (
            not self.shared.quit_requested
            and not self.shared.stop_request
            and command_may_cross_sdk(
                self.shared, run_id=run_id, required_safety_state=state
            )
        )

    def prepare_streaming(self, run_id):
        """Restore Mode 6 under the owner's epoch; the caller owns stop on failure."""
        self._check_owner()
        if self._hand_stop_pending:
            raise DispatchError(
                "previous XHand stop remains unconfirmed", DispatchResult(),
                revoked=True, cause="stop_unconfirmed",
            )
        if not self._connected or self.arm is None or not self.arm.is_connected:
            raise DispatchError("arm device unavailable", DispatchResult())

        def check_abort():
            self.check()
            if not self._authorized(run_id, SafetyState.ARMED):
                raise DispatchError(
                    "motion authority revoked during streaming preparation", DispatchResult(),
                    revoked=True, cause="authority_revoked",
                )

        check_abort()
        self.end_hand_send_interval()
        if self._arm_stopped:
            # Mode restoration changes controller state even without a target.
            self._motion_active = True
            self.arm.enter_mode6(check_abort=check_abort)
            check_abort()
            self._arm_stopped = False

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
            if command.arm_qpos is not None and self._arm_stopped:
                raise DispatchError("arm streaming is not prepared", result)
            for name in ("arm", "hand"):
                target = getattr(command, f"{name}_qpos")
                if target is None:
                    continue
                if not self._authorized(command.run_id, state):
                    raise DispatchError(
                        "motion authority revoked", result, revoked=True, cause="authority_revoked"
                    )
                if time.monotonic_ns() >= valid_until_ns:
                    raise DispatchError(
                        "dispatch deadline expired", result, revoked=True, cause="deadline_expired"
                    )
                # From here, an exception cannot prove that nothing reached the device.
                result = replace(result, **{name: DispatchStatus.UNKNOWN})
                self._motion_active = True
                if name == "arm":
                    code = self.arm.servo(target)
                    status = DispatchStatus.ACCEPTED if code == 0 else DispatchStatus.REJECTED
                    result = replace(result, arm=status, arm_code=code)
                else:
                    status = self.hand.send_action(target)
                    result = replace(result, hand=status)
                    self._check_hand_send(status)
                if status in (DispatchStatus.REJECTED, DispatchStatus.UNKNOWN):
                    raise DispatchError(f"{name} SDK rejected target", result)
            if not self._authorized(command.run_id, state):
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
            if isinstance(exc, DispatchError):
                exc.result = result
                raise
            raise DispatchError(str(exc), result) from exc

    def _check_hand_send(self, status):
        """Bound consecutive ordinary-send CRCs without retries or readback."""
        now_ns = time.monotonic_ns()
        budget_ns = int(self.runtime.hand.feedback_max_age_s * 1e9)
        if status == DispatchStatus.CRC_UNCONFIRMED:
            if self._hand_send_failure_started_ns is None:
                self._hand_send_failure_started_ns = now_ns
                logger.warning("XHand send CRC; delivery unconfirmed, continuing with next target")
            if now_ns - self._hand_send_failure_started_ns >= budget_ns:
                self.shared.error_state = True
                revoke_motion(self.shared, SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
                raise RuntimeError("XHand continuous send CRC timed out; delivery remains unconfirmed")
        elif status == DispatchStatus.ACCEPTED:
            if self._hand_send_failure_started_ns is not None:
                logger.info("XHand send responses recovered")
            self._hand_send_failure_started_ns = None

    def end_hand_send_interval(self):
        """An explicit idle/WAIT is not part of a continuous ordinary-send failure."""
        self._check_owner()
        self._hand_send_failure_started_ns = None

    def send_action(self, command, *, valid_until_ns):
        return self._send(command, SafetyState.RUNNING, valid_until_ns=valid_until_ns)

    def send_hand_home(self, command, *, valid_until_ns):
        if command.arm_qpos is not None:
            raise ValueError("hand HOME cannot carry an arm streaming target")
        # HOME submits once, then its caller checks measured convergence separately.
        self.end_hand_send_interval()
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
        frame = self._last_hand
        if not passive and frame is not None:
            self.hand.stop(frame["qpos"][0], timestamp_ns=int(frame["timestamp_ns"][0]))
        else:
            self.hand.stop()
        # Exceptions and interrupts retain the fence against subsequent motion.
        self._hand_stop_pending = False

    def stop(self):
        # Stop has its own bounded retries; a later experiment starts a new send interval.
        self.end_hand_send_interval()
        errors = []
        # A failed/interrupted stop must not advertise a clean software stop.
        unresolved_motion = self._motion_active
        if self.arm is not None or self.hand is not None:
            self._motion_active = True
        if self.arm is not None:
            self._arm_stopped = False
            try:
                if self.shared.estop_request:
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
            self.shared.error_state = True
            revoke_motion(self.shared, SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise RuntimeError(f"robot stop failures: {errors}") from errors[0]
        self._motion_active = False

    def service_idle(self):
        self.end_hand_send_interval()
        if self._motion_active and (
            self.shared.estop_request
            or self.shared.error_state
            or self.shared.quit_requested
            or int(self.shared.safety_state) != int(SafetyState.RUNNING)
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
            self.shared.error_state = True
            raise RuntimeError(f"robot close failures: {errors}") from errors[0]
