"""XHand driver with one SDK call per runtime read or send.

Reads accept known sensor/CRC statuses only with complete, finite 12-DoF joint
positions; missing auxiliary current remains NaN. Aggregate (``calc_force``)
and dense (``raw_force``) tactile validity are independent: an RS485
distributed-force drop leaves aggregate contact force usable. Send CRC responses leave delivery unconfirmed without stopping
the caller; other non-accepted SDK codes are rejected. SDK exceptions propagate
because they cannot establish whether the command reached the device.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from dexmani_real.config.hardware import HandParams
from dexmani_real.robot.model import (
    HAND_DOF,
    HAND_FINGER_COUNT,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    TACTILE_POINTS_PER_FINGER,
)
from dexmani_real.utils.log import (
    capture_native_stdout,
    extract_native_diagnostics,
)

logger = logging.getLogger(__name__)

_SDK_PROTOCOL = {"ethercat": "EtherCAT", "serial": "RS485"}
_OPEN_RETRIES = {"ethercat": 2, "serial": 3}
_OPEN_RETRY_DELAY_S = 2.0
_INITIAL_STATE_READ_ATTEMPTS = 3
_INITIAL_STATE_READ_INTERVAL_S = 0.02
_COMMUNICATION_CRC_ERROR_CODE = 1_501_070
_COMBINED_FORCE_UNAVAILABLE_CODE = 1_501_018
_DISTRIBUTED_FORCE_UNAVAILABLE_CODE = 1_501_019
_TEMPERATURE_UNAVAILABLE_CODE = 1_501_020
READ_USABLE_CODES = frozenset(
    {
        0,
        _COMBINED_FORCE_UNAVAILABLE_CODE,
        _DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
        _TEMPERATURE_UNAVAILABLE_CODE,
        _COMMUNICATION_CRC_ERROR_CODE,  # complete joint payload; tactile invalid
    }
)
SEND_ACCEPTED_CODES = frozenset(
    {
        0,
        _COMBINED_FORCE_UNAVAILABLE_CODE,
        _DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
        _TEMPERATURE_UNAVAILABLE_CODE,
        1_501_035,  # configured-current overrun / expected grasp contact
    }
)

_EC_STATE_INIT = 1
_STALE_EC_RECOVERY_S = 3.0
_POST_EC_DISCONNECT_S = 2.0
_TACTILE_BIAS_SAMPLE_COUNT = 5
_TACTILE_VERIFY_SAMPLE_COUNT = 3
_TACTILE_BIAS_SAMPLE_INTERVAL_S = 0.02
_PASSIVE_MODE = 0
_POSITION_MODE = 3
# Baseline repeatability bound in SDK-native units; it cannot certify no contact.
_TACTILE_BASELINE_RESIDUAL_THRESHOLD = 2.0
_CONNECTION_HINT = {
    "ethercat": "Check XHand power, EtherCAT cable/link, SDK permissions, and stale slave state",
    "serial": "Check XHand power, USB cable, and serial-device permissions",
}


def _error_code(error: Any) -> int | None:
    if error is None:
        return None
    code = getattr(error, "error_code", -1)
    return -1 if code is None else int(code)


def _error_ok(error: Any) -> bool:
    return _error_code(error) == 0


def _force_xyz(force: Any, label: str) -> np.ndarray:
    if force is None:
        raise ValueError(f"{label} is missing")
    value = np.asarray([force.fx, force.fy, force.fz], dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain three finite values")
    return value


def _tactile_validity(code: int | None, *, comm_type: str) -> tuple[bool, bool]:
    """Return ``(aggregate_allowed, dense_allowed)`` for one read status.

    Serial/RS485 exposes partial tactile statuses: a distributed-force drop
    (``1501019``) leaves aggregate ``calc_force`` valid while dense
    ``raw_force`` is unavailable, and a temperature drop (``1501020``) keeps
    both force fields.  EtherCAT is not known to share those partial
    semantics, so any nonzero status fails tactile closed there while joints
    stay usable per the caller.
    """
    if comm_type == "ethercat":
        return (code == 0, code == 0)
    if code == 0:
        return (True, True)
    if code == _COMBINED_FORCE_UNAVAILABLE_CODE:
        return (False, False)
    if code == _DISTRIBUTED_FORCE_UNAVAILABLE_CODE:
        return (True, False)
    if code == _TEMPERATURE_UNAVAILABLE_CODE:
        return (True, True)
    return (False, False)  # CRC and any other nonzero status


class XHandError(RuntimeError):
    """A fail-fast startup or tactile-baseline preparation error."""

    def __init__(self, operation: str, code: int, message: str) -> None:
        self.operation = str(operation)
        self.code = int(code)
        self.message = str(message)
        super().__init__(f"XHand {self.operation} failed: code={self.code} msg={self.message}")


class XHandSendStatus(Enum):
    """SDK result without conflating CRC uncertainty with command acceptance."""

    ACCEPTED = "accepted"
    CRC_UNCONFIRMED = "crc_unconfirmed"
    REJECTED = "rejected"


@dataclass
class XHandState:
    """Joint feedback and independently valid tactile payloads from one SDK read."""

    qpos: np.ndarray
    current_ma: np.ndarray
    tactile_aggregate: np.ndarray
    tactile_dense: np.ndarray
    tactile_aggregate_valid: bool
    tactile_dense_valid: bool
    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray


class XHand:
    """XHand SDK controller owned by the session I/O thread."""

    def __init__(self, config: HandParams):
        config.validate()
        self.cfg = config
        self.connected_flag = False
        self.device_identity = {
            "backend": "hardware",
            "hand_type": "unavailable",
            "sdk_version": "unavailable",
            "serial_number": "unavailable",
        }

        self._control: Any = None
        self._command: Any = None
        self._tactile_bias_aggregate: np.ndarray | None = None
        self._tactile_bias_dense: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self.connected_flag

    def connect(self) -> None:
        """Open the configured device and seed its command buffer from live feedback."""
        from xhand_controller import xhand_control

        self._sdk = xhand_control
        if self.connected_flag:
            return

        try:
            self._open_device()
            if self.cfg.comm_type == "serial" and self.cfg.rs485_post_open_settle_s:
                time.sleep(self.cfg.rs485_post_open_settle_s)

            self.connected_flag = True
            hand_ids = list(self._control.list_hands_id())
            if self.cfg.device_id not in hand_ids:
                raise XHandError(
                    "connect",
                    -1,
                    f"configured device_id={self.cfg.device_id} not found in {hand_ids}",
                )
            self._read_identity()
            self._seed_command_history()
        except XHandError:
            logger.error("XHand initialization failed", exc_info=True)
            self.disconnect()
            raise
        except Exception as exc:
            logger.error("XHand initialization failed", exc_info=True)
            self.disconnect()
            raise XHandError("connect", -1, str(exc)) from exc

    def _open_device(self) -> None:
        retries = _OPEN_RETRIES[self.cfg.comm_type]
        device_name = self.cfg.device_name
        last_error: XHandError | None = None

        for attempt in range(1, retries + 1):
            self._control = self._sdk.XHandControl()
            if device_name is None:
                devices, _ = self._captured_sdk_call(
                    "discovery",
                    lambda: self._control.enumerate_devices(_SDK_PROTOCOL[self.cfg.comm_type]),
                )
                if not devices:
                    self._close_control()
                    raise XHandError(
                        "connect",
                        -2,
                        f"no XHand device found for {self.cfg.comm_type}",
                    )
                device_name = devices[0]

            if self.cfg.comm_type == "serial":

                def open_call():
                    return self._control.open_serial(device_name, self.cfg.baudrate)
            else:

                def open_call():
                    return self._control.open_ethercat(device_name)

            error, output = self._captured_sdk_call(
                f"open attempt {attempt}/{retries}",
                open_call,
                ignore=("Operation not permitted",),
            )

            if _error_ok(error):
                if "Operation not permitted" in output:
                    logger.warning(
                        "XHand real-time scheduling unavailable; using normal scheduling"
                    )
                if attempt > 1 and self.cfg.comm_type == "ethercat":
                    time.sleep(1.0)
                return

            code = _error_code(error)
            last_error = XHandError(
                "connect",
                -1 if code is None else code,
                str(getattr(error, "error_message", "empty error object")),
            )
            self._close_control()
            if attempt < retries:
                delay = _OPEN_RETRY_DELAY_S
                if self.cfg.comm_type == "ethercat" and attempt == 1:
                    delay = max(delay, _STALE_EC_RECOVERY_S)
                logger.warning(
                    "XHand open attempt %d/%d failed: %s; retrying in %.1fs",
                    attempt,
                    retries,
                    last_error.message,
                    delay,
                )
                time.sleep(delay)

        logger.error(
            "XHand connect failed after %d attempts: %s",
            retries,
            last_error.message if last_error is not None else "unknown error",
        )
        logger.warning(_CONNECTION_HINT[self.cfg.comm_type])
        if last_error is None:
            raise XHandError("connect", -1, "device open failed without an SDK error")
        raise last_error

    def _captured_sdk_call(
        self,
        label: str,
        call: Callable[[], Any],
        *,
        ignore: tuple[str, ...] = (),
    ) -> tuple[Any, str]:
        with capture_native_stdout() as capture:
            result = call()
        output = capture.text
        diagnostics = extract_native_diagnostics(output, ignore=ignore)
        if diagnostics:
            logger.warning("XHand SDK %s diagnostics:\n%s", label, "\n".join(diagnostics))
        return result, output

    def _read_identity(self) -> None:
        try:
            self.device_identity["sdk_version"] = str(self._control.get_sdk_version())
            for key, getter in (
                ("hand_type", self._control.get_hand_type),
                ("serial_number", self._control.get_serial_number),
            ):
                error, value = getter(self.cfg.device_id)
                if _error_ok(error):
                    self.device_identity[key] = str(value)
                else:
                    logger.warning("XHand %s unavailable: code=%s", key, _error_code(error))
        except Exception:
            logger.warning("XHand identity incomplete", exc_info=True)
        logger.info(
            "XHand ready: SDK=%s type=%s serial=%s device_id=%d",
            self.device_identity["sdk_version"],
            self.device_identity["hand_type"],
            self.device_identity["serial_number"],
            self.cfg.device_id,
        )

    def _seed_command_history(self) -> None:
        for attempt in range(_INITIAL_STATE_READ_ATTEMPTS):
            sample = self.get_state()
            if sample is not None:
                self._command = self._make_command(sample.qpos)
                return
            if attempt + 1 < _INITIAL_STATE_READ_ATTEMPTS:
                time.sleep(_INITIAL_STATE_READ_INTERVAL_S)
        raise XHandError("connect", -1, "initial XHand state is unavailable or invalid")

    def disconnect(self) -> None:
        """Release the SDK handle; repeated calls are no-ops."""
        connected_ethercat = self.connected_flag and self.cfg.comm_type == "ethercat"
        errors = []
        try:
            if self._control is not None:
                if connected_ethercat:
                    try:
                        self._request_ethercat_init()
                    except Exception as exc:
                        errors.append(exc)
                try:
                    self._close_control()
                except Exception as exc:
                    errors.append(exc)
                if connected_ethercat:
                    time.sleep(_POST_EC_DISCONNECT_S)
        finally:
            self.connected_flag = False
        if errors:
            raise RuntimeError(f"XHand disconnect failed: {errors}") from errors[0]

    def _request_ethercat_init(self) -> None:
        if self.cfg.ethercat_slave_position < 0:
            logger.warning("XHand EtherCAT slave position unknown; skipping explicit INIT request")
            return
        error, _ = self._control.set_firmware_state(
            self.cfg.device_id,
            self.cfg.ethercat_slave_position,
            _EC_STATE_INIT,
            500_000,
        )
        if not _error_ok(error):
            raise XHandError("ethercat INIT", _error_code(error), "INIT request failed")
        time.sleep(0.2)

    def _close_control(self) -> None:
        control, self._control = self._control, None
        if control is None:
            return
        control.close_device()

    def tare_tactile(self, *, cancel_requested=lambda: False) -> tuple[bool, bool]:
        """Publish verified baselines after the operator confirms no contact.

        Stable contact also yields zero residual; this does not certify no contact.
        SDK reads may block; cancellation is checked on both sides of each read.
        """
        self._tactile_bias_aggregate = self._tactile_bias_dense = None
        candidates = self._capture_tactile_bias(cancel_requested=cancel_requested)
        verified = self._verify_tactile_bias(*candidates, cancel_requested=cancel_requested)
        self._check_tare_cancel(cancel_requested)
        for name, candidate, ok in zip(("aggregate", "dense"), candidates, verified):
            if ok:
                setattr(self, f"_tactile_bias_{name}", candidate)
            logger.log(20 if ok else 30, "XHand %s tactile baseline verified=%s", name, ok)
        return (self._tactile_bias_aggregate is not None, self._tactile_bias_dense is not None)

    @staticmethod
    def _check_tare_cancel(cancel_requested):
        if cancel_requested():
            from concurrent.futures import CancelledError

            raise CancelledError("tactile tare cancelled")

    def _capture_tactile_bias(self, *, cancel_requested):
        samples = ([], [])
        for _ in range(_TACTILE_BIAS_SAMPLE_COUNT):
            self._check_tare_cancel(cancel_requested)
            time.sleep(_TACTILE_BIAS_SAMPLE_INTERVAL_S)
            self._check_tare_cancel(cancel_requested)
            state = self._read_state(apply_bias=False)
            self._check_tare_cancel(cancel_requested)
            if state is None:
                raise XHandError("tare_tactile", -1, "joint state unavailable during bias capture")
            for name, channel in zip(("aggregate", "dense"), samples):
                if getattr(state, f"tactile_{name}_valid"):
                    channel.append(getattr(state, f"tactile_{name}"))
        return tuple(
            np.mean(np.stack(channel), axis=0)
            if len(channel) == _TACTILE_BIAS_SAMPLE_COUNT
            else None
            for channel in samples
        )

    def _verify_tactile_bias(self, aggregate, dense, *, cancel_requested):
        ok = [aggregate is not None, dense is not None]
        for _ in range(_TACTILE_VERIFY_SAMPLE_COUNT):
            self._check_tare_cancel(cancel_requested)
            time.sleep(_TACTILE_BIAS_SAMPLE_INTERVAL_S)
            self._check_tare_cancel(cancel_requested)
            state = self._read_state(apply_bias=False)
            self._check_tare_cancel(cancel_requested)
            if state is None:
                raise XHandError(
                    "tare_tactile", -1, "joint state unavailable during bias verification"
                )
            for index, (name, candidate) in enumerate(
                zip(("aggregate", "dense"), (aggregate, dense))
            ):
                if not ok[index]:
                    continue
                if not getattr(state, f"tactile_{name}_valid"):
                    ok[index] = False
                    continue
                residual = getattr(state, f"tactile_{name}") - candidate
                ok[index] = bool(np.isfinite(residual).all())
                if name == "aggregate":
                    ok[index] = (
                        ok[index]
                        and float(np.max(np.linalg.norm(residual, axis=1)))
                        <= _TACTILE_BASELINE_RESIDUAL_THRESHOLD
                    )
        return tuple(ok)

    def get_state(self) -> XHandState | None:
        """Read one fresh state, returning ``None`` for runtime SDK failures."""
        return self._read_state(apply_bias=True)

    def _read_state(self, *, apply_bias):
        if self._control is None or not self.connected_flag:
            raise RuntimeError("XHand is not connected")
        try:
            error, raw_state = self._control.read_state(self.cfg.device_id, True)
        except Exception:
            logger.warning("XHand read_state raised", exc_info=True)
            return None

        code = _error_code(error)
        if raw_state is None:
            logger.warning(
                "XHand read returned no state: code=%s msg=%s",
                code,
                getattr(error, "error_message", ""),
            )
            return None
        if code not in READ_USABLE_CODES:
            logger.warning(
                "XHand read failed: code=%s msg=%s",
                code,
                getattr(error, "error_message", ""),
            )
            return None

        try:
            qpos, current, board_errors = self._parse_joints(raw_state)
        except (AttributeError, TypeError, ValueError, OverflowError):
            logger.warning("XHand joint payload invalid", exc_info=True)
            return None

        tactile_dense = np.full(HAND_TACTILE_FORCE_SHAPE, np.nan, dtype=np.float64)
        tactile_aggregate = np.full(HAND_TACTILE_SUM_SHAPE, np.nan, dtype=np.float64)
        aggregate_valid, dense_valid = _tactile_validity(code, comm_type=self.cfg.comm_type)
        if aggregate_valid:
            try:
                tactile_aggregate = self._parse_tactile_aggregate(raw_state)
            except (AttributeError, TypeError, ValueError, OverflowError):
                logger.warning("XHand tactile aggregate payload invalid", exc_info=True)
                aggregate_valid = False
                tactile_aggregate.fill(np.nan)
            else:
                # Bias subtraction runs only after a successful parse so an
                # internal bias fault is not mislabeled as a malformed read.
                if apply_bias and self._tactile_bias_aggregate is not None:
                    tactile_aggregate = tactile_aggregate - self._tactile_bias_aggregate
        if dense_valid:
            try:
                tactile_dense = self._parse_tactile_dense(raw_state)
            except (AttributeError, TypeError, ValueError, OverflowError):
                logger.warning("XHand tactile dense payload invalid", exc_info=True)
                dense_valid = False
                tactile_dense.fill(np.nan)
            else:
                if apply_bias and self._tactile_bias_dense is not None:
                    tactile_dense = tactile_dense - self._tactile_bias_dense
        if apply_bias:
            aggregate_valid = aggregate_valid and self._tactile_bias_aggregate is not None
            dense_valid = dense_valid and self._tactile_bias_dense is not None
        if not aggregate_valid:
            tactile_aggregate.fill(np.nan)
        if not dense_valid:
            tactile_dense.fill(np.nan)
        return XHandState(
            qpos=qpos,
            current_ma=current,
            tactile_aggregate=tactile_aggregate,
            tactile_dense=tactile_dense,
            tactile_aggregate_valid=aggregate_valid,
            tactile_dense_valid=dense_valid,
            **board_errors,
        )

    def send_action(self, action: np.ndarray) -> XHandSendStatus:
        """Send one absolute endpoint and preserve CRC delivery uncertainty."""
        target = np.asarray(action, dtype=np.float64)
        try:
            self._validate_action(target)
        except ValueError as exc:
            logger.warning("XHand send rejected: %s", exc)
            return XHandSendStatus.REJECTED
        if self._control is None or self._command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")

        try:
            for index, value in enumerate(target):
                joint = self._command.finger_command[index]
                joint.mode = _POSITION_MODE
                joint.position = float(value)
        except Exception:
            logger.warning("XHand command preparation failed", exc_info=True)
            return XHandSendStatus.REJECTED
        return self._send_command()

    def hold_current(self, qpos: np.ndarray) -> XHandSendStatus:
        """Actively hold measured joint positions clipped to mechanical limits."""
        measured = np.asarray(qpos, dtype=np.float64)
        if measured.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(measured)):
            raise ValueError("XHand.hold_current requires twelve finite joint positions")
        return self.send_action(
            np.clip(measured, self.cfg.mechanical_qpos_min_rad, self.cfg.mechanical_qpos_max_rad)
        )

    def set_passive(self) -> XHandSendStatus:
        """Put every XHand joint into vendor passive mode."""
        if self._control is None or self._command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")
        for index in range(HAND_DOF):
            self._command.finger_command[index].mode = _PASSIVE_MODE
        return self._send_command()

    def _send_command(self) -> XHandSendStatus:
        error = self._control.send_command(self.cfg.device_id, self._command)

        code = _error_code(error)
        if code == _COMMUNICATION_CRC_ERROR_CODE:
            logger.warning(
                "XHand send delivery unconfirmed by CRC response: code=%s msg=%s; "
                "continuing without action acknowledgement",
                code,
                getattr(error, "error_message", ""),
            )
            return XHandSendStatus.CRC_UNCONFIRMED
        if code not in SEND_ACCEPTED_CODES:
            logger.warning(
                "XHand send failed: code=%s msg=%s",
                code,
                getattr(error, "error_message", ""),
            )
            return XHandSendStatus.REJECTED
        return XHandSendStatus.ACCEPTED

    def _make_command(self, qpos: np.ndarray) -> Any:
        command = self._sdk.HandCommand_t()
        for index in range(HAND_DOF):
            joint = command.finger_command[index]
            joint.id = index
            joint.position = float(qpos[index])
            joint.kp = int(self.cfg.kp[index])
            joint.ki = int(self.cfg.ki)
            joint.kd = int(self.cfg.kd)
            joint.tor_max = int(self.cfg.tor_max_ma[index])
            joint.mode = _POSITION_MODE
            joint.res0 = 0
            joint.res1 = 0
            joint.res2 = 0
            joint.res3 = 0
        return command

    def _validate_action(self, qpos: np.ndarray) -> None:
        from dexmani_real.robot.command_validation import check_hand_target

        reason = check_hand_target(
            qpos,
            mechanical_lower_rad=np.asarray(self.cfg.mechanical_qpos_min_rad),
            mechanical_upper_rad=np.asarray(self.cfg.mechanical_qpos_max_rad),
        )
        if reason is not None:
            raise ValueError(f"XHand.send_action: {reason}")

    @staticmethod
    def _parse_joints(
        state: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        qpos = np.full(HAND_JOINT_SHAPE, np.nan, dtype=np.float64)
        current = np.full(HAND_JOINT_SHAPE, np.nan, dtype=np.float64)
        errors = {
            name: np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
            for name in ("commboard_err", "jointboard_err", "tipboard_err")
        }
        seen: set[int] = set()
        for joint in state.finger_state:
            index = int(joint.id)
            if index < 0 or index >= HAND_DOF or index in seen:
                raise ValueError(f"invalid or duplicate joint id {index}")
            seen.add(index)
            qpos[index] = float(joint.position)
            current[index] = float(getattr(joint, "torque", np.nan))
            errors["commboard_err"][index] = int(joint.commboard_err)
            # The vendor SDK spells this field jonitboard_err.
            errors["jointboard_err"][index] = int(joint.jonitboard_err)
            errors["tipboard_err"][index] = int(joint.tipboard_err)
        if len(seen) != HAND_DOF:
            raise ValueError(f"{len(seen)}/{HAND_DOF} joints reported")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("non-finite joint position feedback")
        current[~np.isfinite(current)] = np.nan
        return qpos, current, errors

    def _sensor_data(self, state: Any) -> list[Any]:
        sensors = list(state.sensor_data)
        if len(sensors) != HAND_FINGER_COUNT:
            raise ValueError(
                f"sensor_data must contain {HAND_FINGER_COUNT} sensors, got {len(sensors)}"
            )
        return sensors

    def _parse_tactile_aggregate(self, state: Any) -> np.ndarray:
        """Return the SDK-native ``[5,3]`` aggregate ``calc_force`` payload."""
        sensors = self._sensor_data(state)
        force_aggregate = np.empty(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        for sensor_index, sensor in enumerate(sensors):
            force_aggregate[sensor_index] = _force_xyz(
                sensor.calc_force,
                f"sensor_data[{sensor_index}].calc_force",
            )
        return force_aggregate

    def _parse_tactile_dense(self, state: Any) -> np.ndarray:
        """Return the SDK-native ``[5,120,3]`` dense ``raw_force`` payload."""
        sensors = self._sensor_data(state)
        tactile_dense = np.empty(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        for sensor_index, sensor in enumerate(sensors):
            points = list(sensor.raw_force)
            if len(points) != TACTILE_POINTS_PER_FINGER:
                raise ValueError(
                    f"sensor_data[{sensor_index}].raw_force must contain "
                    f"{TACTILE_POINTS_PER_FINGER} points, got {len(points)}"
                )
            for point_index, force in enumerate(points):
                tactile_dense[sensor_index, point_index] = _force_xyz(
                    force, f"sensor_data[{sensor_index}].raw_force[{point_index}]"
                )
        return tactile_dense
