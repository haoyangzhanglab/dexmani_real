"""XHand 12-DOF hardware driver.

The worker-facing API is intentionally small: connect, calibrate tactile
sensors, read state, send an endpoint, and disconnect. SDK objects remain
local to the hand worker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np
from xhand_controller import xhand_control as xhc  # type: ignore[import-untyped]

from dexmani_real.config.defaults import HandParams
from dexmani_real.utils.log import (
    capture_native_stdout,
    extract_native_diagnostics,
    get_logger,
)
from dexmani_real.utils.schema import (
    HAND_CONTACT_SHAPE,
    HAND_DOF,
    HAND_FINGER_COUNT,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    TACTILE_POINTS_PER_FINGER,
)

logger = get_logger(__name__)

_SDK_PROTOCOL = {"ethercat": "EtherCAT", "serial": "RS485"}
# Fixed, bounded driver policies; runtime config only carries deployment tuning.
_OPEN_RETRIES = {"ethercat": 2, "serial": 3}
_OPEN_RETRY_DELAY_S = 2.0
_INITIAL_STATE_READ_ATTEMPTS = 3
_INITIAL_STATE_READ_INTERVAL_S = 0.02
_RS485_CRC_ERROR = 1_501_070
_RS485_SENSOR_STATUS: dict[int, tuple[bool, bool, str]] = {
    # code: (distributed force valid, combined force valid, diagnostic)
    1_501_018: (False, False, "combined force unavailable"),
    1_501_019: (False, True, "distributed force unavailable"),
    1_501_020: (True, True, "temperature unavailable"),
}

_EC_STATE_INIT = 1
_STALE_EC_RECOVERY_S = 3.0
_POST_EC_DISCONNECT_S = 2.0
_TACTILE_SCALE = 0.1
_TACTILE_SENSOR_IDS = tuple(range(0x11, 0x16))
_POSITION_MODE = 3
_TACTILE_CONTACT_THRESHOLD = 1.0
_RAW_FORCE_CONTACT_THRESHOLD = 1.0
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


class XHandError(RuntimeError):
    """One failed XHand SDK operation."""

    def __init__(self, operation: str, code: int, message: str) -> None:
        self.operation = str(operation)
        self.code = int(code)
        self.message = str(message)
        super().__init__(
            f"XHand {self.operation} failed: code={self.code} msg={self.message}"
        )


@dataclass
class XHandState:
    """Validated feedback from one successful fresh SDK read."""

    qpos: np.ndarray
    current_ma: np.ndarray
    tactile_force: np.ndarray
    tactile_sum: np.ndarray
    tactile_contact: np.ndarray
    tactile_sum_valid: bool
    tactile_valid: bool
    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray

    @property
    def has_hardware_fault(self) -> bool:
        return bool(
            np.any(self.commboard_err)
            or np.any(self.jointboard_err)
            or np.any(self.tipboard_err)
        )


class XHand:
    """Thin stateful adapter around one worker-local SDK controller."""

    def __init__(self, config: HandParams):
        self.cfg = config
        self.connected_flag = False
        self.last_qpos_cmd: np.ndarray | None = None
        self.device_identity = {
            "backend": "hardware",
            "hand_type": "unavailable",
            "sdk_version": "unavailable",
            "serial_number": "unavailable",
        }

        self._control: Any = None
        self._command: Any = None
        self._tactile_bias_sum: np.ndarray | None = None
        self._tactile_bias_raw: np.ndarray | None = None
        self._last_tactile_valid: bool | None = None
        self._sensor_status: dict[str, int | None] = {"read": None, "send": None}

    # Lifecycle

    @property
    def is_connected(self) -> bool:
        return self.connected_flag

    @property
    def tactile_calibrated(self) -> bool:
        return self._tactile_bias_sum is not None and self._tactile_bias_raw is not None

    def connect(self) -> None:
        """Open the configured device and seed command history from live feedback."""
        if self.connected_flag:
            return

        try:
            self._open_device()
            if (
                self.cfg.comm_type == "serial"
                and self.cfg.rs485_post_open_settle_s
            ):
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
            self._control = xhc.XHandControl()
            if device_name is None:
                devices, _ = self._captured_sdk_call(
                    "discovery",
                    lambda: self._control.enumerate_devices(
                        _SDK_PROTOCOL[self.cfg.comm_type]
                    ),
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
                open_call = lambda: self._control.open_serial(
                    device_name, self.cfg.baudrate
                )
            else:
                open_call = lambda: self._control.open_ethercat(device_name)
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
            logger.warning(
                "XHand SDK %s diagnostics:\n%s", label, "\n".join(diagnostics)
            )
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
                    logger.warning(
                        "XHand %s unavailable: code=%s", key, _error_code(error)
                    )
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
            try:
                sample = self.get_state()
                self._command = self._make_command(sample.qpos)
                self.last_qpos_cmd = np.clip(
                    sample.qpos,
                    self.cfg.qpos_min_rad,
                    self.cfg.qpos_max_rad,
                )
                return
            except (XHandError, ValueError):
                if attempt + 1 < _INITIAL_STATE_READ_ATTEMPTS:
                    time.sleep(_INITIAL_STATE_READ_INTERVAL_S)
        raise XHandError("connect", -1, "initial XHand state is unavailable or invalid")

    def disconnect(self) -> None:
        """Release the SDK handle; repeated calls are no-ops."""
        connected_ethercat = self.connected_flag and self.cfg.comm_type == "ethercat"
        if self._control is not None:
            if connected_ethercat:
                self._request_ethercat_init()
            self._close_control()
            if connected_ethercat:
                time.sleep(_POST_EC_DISCONNECT_S)
        self.connected_flag = False

    def _request_ethercat_init(self) -> None:
        if self.cfg.ethercat_slave_position < 0:
            logger.warning(
                "XHand EtherCAT slave position unknown; skipping explicit INIT request"
            )
            return
        try:
            error, _ = self._control.set_firmware_state(
                self.cfg.device_id,
                self.cfg.ethercat_slave_position,
                _EC_STATE_INIT,
                500_000,
            )
            if _error_ok(error):
                time.sleep(0.2)
            else:
                logger.debug(
                    "XHand EtherCAT INIT request failed: code=%s", _error_code(error)
                )
        except Exception:
            logger.debug("XHand EtherCAT INIT request unavailable", exc_info=True)

    def _close_control(self) -> None:
        control, self._control = self._control, None
        if control is None:
            return
        try:
            control.close_device()
        except Exception:
            logger.warning("XHand control did not close cleanly", exc_info=True)

    # Tactile initialization

    def calibrate_tactile(self) -> bool:
        """Reset sensors and estimate a no-contact bias without gating joint control."""
        self._tactile_bias_sum = None
        self._tactile_bias_raw = None
        startup = self.get_state()
        if not startup.tactile_sum_valid or self._tactile_load_present(startup):
            logger.error(
                "Tactile calibration refused: contact/load or invalid data at startup"
            )
            return False
        self._reset_tactile_sensors()
        self._capture_tactile_bias()
        return self.tactile_calibrated

    def _tactile_load_present(self, state: XHandState) -> bool:
        if state.tactile_valid:
            magnitude = np.linalg.norm(state.tactile_force, axis=2)
            return bool(np.any(magnitude > _RAW_FORCE_CONTACT_THRESHOLD))
        magnitude = np.linalg.norm(state.tactile_sum, axis=1)
        return bool(np.any(magnitude > _TACTILE_CONTACT_THRESHOLD))

    def _reset_tactile_sensors(self) -> None:
        unsupported = 0
        for sensor_id in _TACTILE_SENSOR_IDS:
            reset_sensor = partial(
                self._control.reset_sensor, self.cfg.device_id, sensor_id
            )
            for attempt in range(3):
                error, output = self._captured_sdk_call(
                    f"reset sensor {sensor_id}",
                    reset_sensor,
                    ignore=("Unknow Cmd!",),
                )
                unsupported += int("Unknow Cmd!" in output)
                if _error_ok(error):
                    break
                if attempt < 2:
                    time.sleep(0.2)
            else:
                logger.warning("XHand tactile sensor %d reset failed", sensor_id)
        if unsupported:
            logger.info(
                "XHand firmware did not implement %d tactile reset request(s)",
                unsupported,
            )

    def _capture_tactile_bias(self, sample_count: int = 5) -> None:
        samples = [self.get_state() for _ in range(sample_count)]
        if not all(sample.tactile_valid for sample in samples):
            raise XHandError(
                "calibrate_tactile", -1, "incomplete tactile data during bias capture"
            )
        if any(self._tactile_load_present(sample) for sample in samples):
            raise XHandError(
                "calibrate_tactile",
                -1,
                "contact/load detected during tactile bias capture",
            )
        self._tactile_bias_sum = np.mean(
            np.stack([sample.tactile_sum for sample in samples]), axis=0
        )
        self._tactile_bias_raw = np.mean(
            np.stack([sample.tactile_force for sample in samples]), axis=0
        )
    # State and commands

    def get_state(self) -> XHandState:
        """Return fresh joint feedback; sensor-only errors degrade tactile fields."""
        try:
            error, state = self._read_raw_state()
            sensor_status = self._is_sensor_status(error)
            if (not _error_ok(error) and not sensor_status) or state is None:
                code = _error_code(error)
                raise XHandError(
                    "read",
                    -1 if code is None else code,
                    str(getattr(error, "error_message", "empty error object")),
                )

            self._update_sensor_status("read", error)
            if sensor_status:
                code = _error_code(error)
                assert code is not None
                raw_valid, sum_valid, _ = _RS485_SENSOR_STATUS[code]
            else:
                raw_valid, sum_valid = True, True
            return self._parse_state(
                state, raw_expected=raw_valid, sum_expected=sum_valid
            )
        except XHandError:
            raise
        except Exception as exc:
            raise XHandError("read", -1, str(exc)) from exc

    def _read_raw_state(self) -> tuple[Any, Any]:
        if self._control is None or not self.connected_flag:
            return None, None
        # Control and recording require a live transaction, never the SDK cache.
        call = lambda: self._control.read_state(self.cfg.device_id, True)
        error, state = call()
        if self.cfg.comm_type != "serial":
            return error, state
        for attempt in range(1, self.cfg.rs485_read_crc_retry_count + 1):
            if _error_code(error) != _RS485_CRC_ERROR:
                break
            self._crc_backoff(
                "state read", attempt, self.cfg.rs485_read_crc_retry_count
            )
            error, state = call()
        return error, state

    def send_action(self, action: np.ndarray) -> None:
        """Validate and send one absolute joint endpoint."""
        target = np.asarray(action, dtype=np.float64)
        self._validate_action(target)
        if self._control is None or self._command is None or not self.connected_flag:
            raise XHandError("send", -1, "XHand command path is not initialized")

        try:
            for index, value in enumerate(target):
                self._command.finger_command[index].position = float(value)
            error = self._send_with_crc_retry()
            if not _error_ok(error) and not self._is_sensor_status(error):
                code = _error_code(error)
                raise XHandError(
                    "send",
                    -1 if code is None else code,
                    str(getattr(error, "error_message", "empty error object")),
                )

            self._update_sensor_status("send", error)
            self.last_qpos_cmd = target.copy()
        except XHandError:
            raise
        except Exception as exc:
            raise XHandError("send", -1, str(exc)) from exc

    def _send_with_crc_retry(self) -> Any:
        call = lambda: self._control.send_command(self.cfg.device_id, self._command)
        error = call()
        if self.cfg.comm_type != "serial":
            return error
        for attempt in range(1, self.cfg.rs485_crc_retry_count + 1):
            if _error_code(error) != _RS485_CRC_ERROR:
                break
            self._crc_backoff("command", attempt, self.cfg.rs485_crc_retry_count)
            error = call()
        return error

    def _crc_backoff(self, operation: str, attempt: int, attempts: int) -> None:
        delay = float(self.cfg.rs485_crc_retry_backoff_s)
        logger.warning(
            "XHand RS485 %s CRC error; retrying %d/%d after %.3fs",
            operation,
            attempt,
            attempts,
            delay,
        )
        if delay:
            time.sleep(delay)

    def _make_command(self, qpos: np.ndarray) -> Any:
        command = xhc.HandCommand_t()
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
        if qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
            raise ValueError("XHand.send_action requires twelve finite joint targets")
        mechanical_bad = np.flatnonzero(
            (qpos < np.asarray(self.cfg.mechanical_qpos_min_rad) - 1e-12)
            | (qpos > np.asarray(self.cfg.mechanical_qpos_max_rad) + 1e-12)
        )
        if mechanical_bad.size:
            index = int(mechanical_bad[0])
            lower = self.cfg.mechanical_qpos_min_rad[index]
            upper = self.cfg.mechanical_qpos_max_rad[index]
            raise ValueError(
                "XHand mechanical joint limit violation: "
                f"joint={index} target={qpos[index]:.6f}rad "
                f"range=[{lower:.6f},{upper:.6f}]rad"
            )

    # Parsing and status handling

    def _parse_state(
        self, state: Any, *, raw_expected: bool, sum_expected: bool
    ) -> XHandState:
        qpos, current, board_errors = self._parse_joints(state)
        tactile_force = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        tactile_sum = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        tactile_sum_valid = False
        tactile_valid = False

        if sum_expected:
            try:
                if raw_expected:
                    tactile_force, tactile_sum = self._parse_tactile(state)
                    tactile_valid = True
                else:
                    tactile_sum = self._parse_tactile_sum(state)
                tactile_sum_valid = True
            except (AttributeError, TypeError, ValueError, OverflowError):
                if self._last_tactile_valid is not False:
                    logger.warning(
                        "XHand tactile payload malformed; publishing invalid zeros",
                        exc_info=True,
                    )
                tactile_force.fill(0.0)
                tactile_sum.fill(0.0)
                tactile_sum_valid = False
                tactile_valid = False

        if tactile_valid and self._last_tactile_valid is False:
            logger.info("XHand tactile payload recovered")
        self._last_tactile_valid = tactile_valid
        contact = (
            np.linalg.norm(tactile_sum, axis=1) > _TACTILE_CONTACT_THRESHOLD
            if tactile_sum_valid
            else np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
        )
        return XHandState(
            qpos=qpos,
            current_ma=current,
            tactile_force=tactile_force,
            tactile_sum=tactile_sum,
            tactile_contact=contact,
            tactile_sum_valid=tactile_sum_valid,
            tactile_valid=tactile_valid,
            **board_errors,
        )

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
        for joint in getattr(state, "finger_state", []):
            index = int(getattr(joint, "id", -1))
            if index < 0 or index >= HAND_DOF or index in seen:
                raise XHandError(
                    "read", -1, f"XHand parse: invalid or duplicate joint id {index}"
                )
            seen.add(index)
            qpos[index] = float(getattr(joint, "position", np.nan))
            current[index] = float(getattr(joint, "torque", np.nan))
            errors["commboard_err"][index] = int(getattr(joint, "commboard_err", 0))
            errors["jointboard_err"][index] = int(
                getattr(joint, "jonitboard_err", getattr(joint, "jointboard_err", 0))
            )
            errors["tipboard_err"][index] = int(getattr(joint, "tipboard_err", 0))
        if len(seen) != HAND_DOF:
            raise XHandError(
                "read", -1, f"XHand parse: {len(seen)}/{HAND_DOF} joints reported"
            )
        if not np.all(np.isfinite(qpos)):
            raise XHandError("read", -1, "non-finite joint position feedback")
        if not np.all(np.isfinite(current)):
            raise XHandError("read", -1, "non-finite joint current feedback")
        return qpos, current, errors

    def _sensor_data(self, state: Any) -> list[Any]:
        sensors = list(state.sensor_data)
        if len(sensors) != HAND_FINGER_COUNT:
            raise ValueError(
                f"sensor_data must contain {HAND_FINGER_COUNT} sensors, got {len(sensors)}"
            )
        return sensors

    def _parse_tactile_sum(self, state: Any) -> np.ndarray:
        force_sum = np.asarray(
            [
                _force_xyz(
                    getattr(sensor, "calc_force", None),
                    f"sensor_data[{index}].calc_force",
                )
                for index, sensor in enumerate(self._sensor_data(state))
            ],
            dtype=np.float64,
        )
        force_sum *= _TACTILE_SCALE
        if self._tactile_bias_sum is not None:
            force_sum -= self._tactile_bias_sum
        return force_sum

    def _parse_tactile(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        sensors = self._sensor_data(state)
        force_sum = np.empty(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        tactile_force = np.empty(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        for sensor_index, sensor in enumerate(sensors):
            force_sum[sensor_index] = _force_xyz(
                getattr(sensor, "calc_force", None),
                f"sensor_data[{sensor_index}].calc_force",
            )
            points = list(sensor.raw_force)
            if len(points) != TACTILE_POINTS_PER_FINGER:
                raise ValueError(
                    f"sensor_data[{sensor_index}].raw_force must contain "
                    f"{TACTILE_POINTS_PER_FINGER} points, got {len(points)}"
                )
            for point_index, force in enumerate(points):
                tactile_force[sensor_index, point_index] = _force_xyz(
                    force, f"sensor_data[{sensor_index}].raw_force[{point_index}]"
                )

        tactile_force *= _TACTILE_SCALE
        force_sum *= _TACTILE_SCALE
        if self._tactile_bias_raw is not None:
            tactile_force -= self._tactile_bias_raw
        if self._tactile_bias_sum is not None:
            force_sum -= self._tactile_bias_sum
        return tactile_force, force_sum

    def _is_sensor_status(self, error: Any) -> bool:
        return (
            self.cfg.comm_type == "serial"
            and _error_code(error) in _RS485_SENSOR_STATUS
        )

    def _update_sensor_status(self, operation: str, error: Any) -> None:
        code = _error_code(error)
        previous = self._sensor_status[operation]
        if self._is_sensor_status(error):
            if code != previous:
                detail = _RS485_SENSOR_STATUS[code][2]  # type: ignore[index]
                logger.warning(
                    "XHand RS485 %s succeeded with degraded sensor data: code=%s (%s)",
                    operation,
                    code,
                    detail,
                )
            self._sensor_status[operation] = code
            return
        if previous is not None:
            logger.info("XHand RS485 %s sensor response recovered", operation)
        self._sensor_status[operation] = None
