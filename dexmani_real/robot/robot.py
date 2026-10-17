"""Low-frequency coordination: submit xArm goals and send XHand targets once."""

import logging
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import (
    DispatchStatus, DispatchResult, DispatchError, DispatchInterrupted,
    check_arm_target, check_hand_target,
)

from dexmani_real.ipc.schema import HAND_STATE_DTYPE
from dexmani_real.runtime.state import RunEndReason, SafetyState


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
        self.shared, self.runtime = shared, runtime
        self._arm_factory, self._hand_factory = arm_factory, hand_factory
        self.check_services = check_services
        self.arm = self.hand = None
        self._owner = None
        self._connected = False
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
        self.shared.publish_arm_authority()
        if self.check_services is not None and not self.check_services():
            # The service owner applies its own fail-safe and records the cause.
            raise RuntimeError("required runtime service unavailable")
        if self.arm is not None:
            self.arm.update_health(service_until_ns=time.monotonic_ns() +
                                   int(self.runtime.arm.feedback_max_age_s * 1e9))
            self.arm.check_error()

    def connect(self):
        if self._owner is not None:
            raise RuntimeError("robot connection is single-use")
        self._owner = threading.get_ident()
        epoch = self.shared.run_id

        def cancelled():
            self.check()
            return self._initial_state_cancelled(epoch)

        try:
            from dexmani_real.robot.arm_executor import ArmExecutor

            self.arm = ArmExecutor(self.runtime.arm, self.shared, driver_factory=self._arm_factory)
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
                self.shared.request_emergency_stop()
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
        if self.shared.safety_state != SafetyState.ARMED:
            raise RuntimeError("tactile reset requires idle ARMED state")
        if self.hand is None:
            return False
        self.shared.revoke_motion()
        epoch = self.shared.run_id
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
        if self.shared.safety_state == SafetyState.RUNNING:
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
        # Idle feedback may recover after a cancelled wait; the deadline only bounds missing values.
        if issue is not None and self._hand_initializing_until_ns is not None:
            if now_ns >= self._hand_initializing_until_ns:
                raise RuntimeError(
                    f"XHand initial feedback timed out after {self.runtime.hand.initial_state_timeout_s:.3f}s: "
                    f"{issue}"
                )
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
            arm = self.arm.snapshot()
            hand = self._read_hand()
            if hand is not None:
                from dexmani_real.runtime.observation import hand_feedback_time_ns

                self.arm.update_health(hand_until_ns=hand_feedback_time_ns(hand) +
                                       int(self.runtime.hand.feedback_max_age_s * 1e9))
        except Exception:
            self.shared.error_state = True
            self.shared.revoke_motion(SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise
        return RobotState(arm, hand)

    def _initial_state_cancelled(self, epoch):
        return bool(
            epoch != self.shared.run_id
            or self.shared.stop_request
            or self.shared.estop_request
            or self.shared.error_state
            or self.shared.quit_requested
            or not self.shared.running
        )

    def read_initial_state(self, *, cancel_requested=None):
        """Wait only for XHand's bounded initial cache, without resetting its deadline."""
        self._check_owner()
        epoch = self.shared.run_id
        # A valid read clears the shared initialization field, not this attempt's deadline.
        deadline_ns = self._hand_initializing_until_ns if self.hand is not None else None

        def check_cancelled():
            if (
                (cancel_requested is not None and cancel_requested())
                or self._initial_state_cancelled(epoch)
            ):
                raise CancelledError("XHand initial feedback acquisition cancelled")

        def check_deadline():
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                # This timeout is outside read_state's exception fence; apply the same fault boundary.
                self.shared.error_state = True
                self.shared.revoke_motion(SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
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
            # Only _checked_hand_feedback can withhold a connected hand within this budget.
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(min(0.01, remaining_ns / 1e9))

    def prepare_streaming(self, run_id):
        """Prepare Mode 1 from measured standstill under the owner's epoch; the caller owns stop on failure."""
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
            if not self.shared.command_may_cross_sdk(run_id=run_id, required_safety_state=SafetyState.ARMED):
                raise DispatchError(
                    "motion authority revoked during streaming preparation", DispatchResult(),
                    revoked=True, cause="authority_revoked",
                )

        check_abort()
        self.end_hand_send_interval()
        self._motion_active = True
        # Waiting services the hand and health on this thread, never on xarm-owner.
        from dexmani_real.robot.drivers.xarm7 import HomeAborted
        try:
            self.arm.prepare(
                run_id, service=lambda: (self._read_hand(), self.check()),
                abort=lambda: not self.shared.command_may_cross_sdk(
                    run_id=run_id, required_safety_state=SafetyState.ARMED,
                ),
            )
        except HomeAborted as exc:
            raise DispatchError(str(exc), DispatchResult(), revoked=True,
                                cause="authority_revoked") from exc
        check_abort()

    def _send(self, command, state, *, valid_until_ns, execute_until_ns=None):
        self._check_owner()
        arm_status, hand_status = DispatchStatus.NOT_CALLED, DispatchStatus.NOT_CALLED
        command_id = (-1, -1)

        def receipt():
            return DispatchResult(arm_status, hand_status, time.monotonic_ns(), command_id)

        def check_dispatch():
            if not self.shared.command_may_cross_sdk(run_id=command.run_id, required_safety_state=state):
                raise DispatchError("motion authority revoked", receipt(), revoked=True,
                                    cause="authority_revoked")
            if self.arm is not None:
                self.arm.check_error()
            now = time.monotonic_ns()
            if now >= valid_until_ns or (execute_until_ns is not None and now >= execute_until_ns):
                raise DispatchError("dispatch deadline expired", receipt(), revoked=True,
                                    cause="deadline_expired")
            self._motion_active = True

        try:
            if self._hand_stop_pending:
                raise DispatchError(
                    "previous XHand stop remains unconfirmed",
                    receipt(),
                    revoked=True,
                    cause="stop_unconfirmed",
                )
            # Validate both targets before submitting either device's command.
            if command.arm_qpos is not None:
                if not self._connected or self.arm is None or not self.arm.is_connected:
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
                if not self._connected or self.hand is None or not self.hand.is_connected:
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
                command_id = self.arm.submit(command, submit_until_ns=valid_until_ns,
                                             execute_until_ns=execute_until_ns)
                arm_status = DispatchStatus.SUBMITTED
            if command.hand_qpos is not None:
                check_dispatch()
                hand_status = DispatchStatus.UNKNOWN
                hand_status = self.hand.send_action(command.hand_qpos)
                self._check_hand_send(hand_status)
                if hand_status in (DispatchStatus.REJECTED, DispatchStatus.UNKNOWN):
                    raise RuntimeError("hand SDK rejected target")
            if not self.shared.command_may_cross_sdk(run_id=command.run_id, required_safety_state=state):
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
                self.shared.revoke_motion(SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
                raise RuntimeError("XHand continuous send CRC timed out; delivery remains unconfirmed")
        elif status == DispatchStatus.ACCEPTED:
            if self._hand_send_failure_started_ns is not None:
                logger.info("XHand send responses recovered")
            self._hand_send_failure_started_ns = None

    def end_hand_send_interval(self):
        """An explicit idle/WAIT is not part of a continuous ordinary-send failure."""
        self._check_owner()
        self._hand_send_failure_started_ns = None

    def send_action(self, command, *, valid_until_ns, execute_until_ns):
        return self._send(command, SafetyState.RUNNING, valid_until_ns=valid_until_ns,
                          execute_until_ns=execute_until_ns)

    def update_observation_health(self, row):
        deadlines = []
        if row.camera is not None:
            deadlines.append(int(row.camera["timestamp_ns"]) + int(self.runtime.camera.max_frame_age_s * 1e9))
        if row.pointcloud_timestamp_ns > 0:
            deadlines.append(row.pointcloud_timestamp_ns + int(self.runtime.camera.max_frame_age_s * 1e9))
        if row.vr is not None:
            deadlines.append(int(row.vr["recv_ts_ns"]) + int(self.runtime.policy.vr_mapping.stale_threshold_s * 1e9))
        # Arm freshness is independently checked against the executor's live snapshot.
        self.arm.update_health(upstream_until_ns=min(deadlines) if deadlines else 2**63 - 1)

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
            if abort_check() or not self.shared.command_may_cross_sdk(run_id=run_id, required_safety_state=SafetyState.ARMED):
                return "home authority revoked"
            return None

        def service():
            self._read_hand()
            self.check()

        if check_abort():
            return False
        self._motion_active = True
        self.arm.home(waypoints, target, run_id, abort=check_abort, service=service)
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
        # XHand stop retains its bounded protocol retries; ordinary sends never retry.
        self.end_hand_send_interval()
        if self.shared.safety_state == SafetyState.RUNNING:
            self.shared.revoke_motion()
        errors = []
        # A failed/interrupted stop must not advertise a clean software stop.
        unresolved_motion = self._motion_active
        if self.arm is not None or self.hand is not None:
            self._motion_active = True
        if self.arm is not None:
            try:
                self.arm.request_stop()
            except Exception as exc:
                errors.append(exc)
        try:
            self._stop_hand()
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
            self._motion_active = True
            self.shared.error_state = True
            self.shared.revoke_motion(SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise RuntimeError(f"robot stop failures: {errors}") from errors[0]
        self._motion_active = False

    def service_idle(self):
        self.end_hand_send_interval()
        if self._motion_active and (
            self.shared.estop_request
            or self.shared.error_state
            or self.shared.quit_requested
            or self.shared.safety_state != SafetyState.RUNNING
        ):
            self.stop()
        now = time.monotonic_ns()
        if now >= self._idle_next_ns and self._connected:
            self._idle_next_ns = now + int(1e9 / 30)
            self.read_state()
            if self._hand_stop_pending:
                self._stop_hand()

    def close(self):
        if self._owner is None:
            return
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
                    else:
                        setattr(self, name, None)
        self._connected = False
        if errors:
            self._motion_active = True
            self.shared.error_state = True
            raise RuntimeError(f"robot close failures: {errors}") from errors[0]
