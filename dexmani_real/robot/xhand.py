"""XHand 12-DOF hardware driver.

The worker-facing API is intentionally small: connect, initialize tactile
sensors, read state, send an endpoint, and disconnect.  SDK objects remain
local to the hand worker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, ClassVar

import numpy as np
from xhand_controller import xhand_control as xhc  # type: ignore[import-untyped]

from dexmani_real.config.defaults import hand
from dexmani_real.utils.limits import validate_hand_limit_nesting
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
_TACTILE_SENSOR_IDS = range(17, 17 + HAND_FINGER_COUNT)
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


class XHandReadError(RuntimeError):
    """One failed XHand state transaction."""

    def __init__(self, code: int, message: str, *, connected: bool) -> None:
        self.code = int(code)
        self.message = str(message)
        self.connected = bool(connected)
        super().__init__(
            f"XHand read failed: code={self.code} msg={self.message!r} "
            f"connected={int(self.connected)}"
        )


@dataclass
class XHandConfig:
    """Device-bound settings resolved by the hand worker."""

    comm_type: str = field(default_factory=lambda: hand.comm_type)
    device_name: str | None = field(default_factory=lambda: hand.device_name)
    baudrate: int = field(default_factory=lambda: hand.baudrate)
    device_id: int = field(default_factory=lambda: hand.device_id)
    ethercat_slave_position: int = field(
        default_factory=lambda: hand.ethercat_slave_position
    )

    rs485_post_open_settle_s: float = field(
        default_factory=lambda: hand.rs485_post_open_settle_s
    )
    rs485_crc_retry_count: int = field(
        default_factory=lambda: hand.rs485_crc_retry_count
    )
    rs485_read_crc_retry_count: int = field(
        default_factory=lambda: hand.rs485_read_crc_retry_count
    )
    rs485_crc_retry_backoff_s: float = field(
        default_factory=lambda: hand.rs485_crc_retry_backoff_s
    )
    home_qpos: np.ndarray = field(
        default_factory=lambda: np.deg2rad(
            np.asarray(hand.home_qpos_deg, dtype=np.float64)
        )
    )
    qpos_min: np.ndarray = field(
        default_factory=lambda: np.asarray(hand.qpos_min_rad, dtype=np.float64)
    )
    qpos_max: np.ndarray = field(
        default_factory=lambda: np.asarray(hand.qpos_max_rad, dtype=np.float64)
    )
    mechanical_qpos_min: np.ndarray = field(
        default_factory=lambda: np.asarray(
            hand.mechanical_qpos_min_rad, dtype=np.float64
        )
    )
    mechanical_qpos_max: np.ndarray = field(
        default_factory=lambda: np.asarray(
            hand.mechanical_qpos_max_rad, dtype=np.float64
        )
    )

    kp: int = 100
    ki: int = 0
    kd: int = 0
    tor_max: int = 300
    kp_per_joint: np.ndarray | None = None
    ki_per_joint: np.ndarray | None = None
    kd_per_joint: np.ndarray | None = None
    tor_max_per_joint: np.ndarray | None = None
    mode: int = 3

    tactile_contact_threshold: float = 1.0
    raw_force_contact_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.comm_type not in _SDK_PROTOCOL:
            raise ValueError("XHand comm_type must be 'ethercat' or 'serial'")
        if self.device_name is not None and not isinstance(self.device_name, str):
            raise ValueError("XHand device_name must be a string or null")
        if not isinstance(self.baudrate, int) or self.baudrate <= 0:
            raise ValueError("XHand baudrate must be a positive integer")
        if not isinstance(self.device_id, int) or self.device_id < 0:
            raise ValueError("XHand device_id must be a non-negative integer")
        if self.ethercat_slave_position < -1:
            raise ValueError("ethercat_slave_position must be -1 or non-negative")

        nonnegative = {
            "rs485_post_open_settle_s": self.rs485_post_open_settle_s,
            "rs485_crc_retry_backoff_s": self.rs485_crc_retry_backoff_s,
            "tactile_contact_threshold": self.tactile_contact_threshold,
            "raw_force_contact_threshold": self.raw_force_contact_threshold,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("rs485_crc_retry_count", "rs485_read_crc_retry_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        lower = np.asarray(self.qpos_min, dtype=np.float64)
        upper = np.asarray(self.qpos_max, dtype=np.float64)
        mechanical_lower = np.asarray(self.mechanical_qpos_min, dtype=np.float64)
        mechanical_upper = np.asarray(self.mechanical_qpos_max, dtype=np.float64)
        home_qpos = np.asarray(self.home_qpos, dtype=np.float64)
        vectors = (lower, upper, mechanical_lower, mechanical_upper, home_qpos)
        if any(value.shape != HAND_JOINT_SHAPE for value in vectors):
            raise ValueError(
                f"XHand home and limit arrays must have shape {HAND_JOINT_SHAPE}"
            )
        if not np.all(np.isfinite(np.concatenate(vectors))):
            raise ValueError("XHand home and limit arrays must be finite")
        validate_hand_limit_nesting(
            lower,
            upper,
            mechanical_lower,
            mechanical_upper,
            np.asarray(hand.mechanical_qpos_min_rad, dtype=np.float64),
            np.asarray(hand.mechanical_qpos_max_rad, dtype=np.float64),
            label="XHand",
        )
        if np.any(home_qpos < lower - 1e-12) or np.any(home_qpos > upper + 1e-12):
            raise ValueError("XHand home_qpos must be inside command limits")


@dataclass(frozen=True)
class XHandSample:
    """Validated, immutable feedback from one successful read."""

    qpos: np.ndarray
    current: np.ndarray
    tactile_force: np.ndarray
    tactile_sum: np.ndarray
    tactile_contact: np.ndarray
    tactile_sum_valid: bool
    tactile_valid: bool
    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray

    _ARRAYS: ClassVar[tuple[tuple[str, tuple[int, ...], Any], ...]] = (
        ("qpos", HAND_JOINT_SHAPE, np.float64),
        ("current", HAND_JOINT_SHAPE, np.float64),
        ("tactile_force", HAND_TACTILE_FORCE_SHAPE, np.float64),
        ("tactile_sum", HAND_TACTILE_SUM_SHAPE, np.float64),
        ("tactile_contact", HAND_CONTACT_SHAPE, bool),
        ("commboard_err", HAND_JOINT_SHAPE, np.int32),
        ("jointboard_err", HAND_JOINT_SHAPE, np.int32),
        ("tipboard_err", HAND_JOINT_SHAPE, np.int32),
    )

    def __post_init__(self) -> None:
        for name in ("tactile_sum_valid", "tactile_valid"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"XHandSample.{name} must be boolean")
            object.__setattr__(self, name, bool(value))

        for name, shape, dtype in self._ARRAYS:
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != shape:
                raise ValueError(
                    f"XHandSample.{name} must have shape {shape}, got {value.shape}"
                )
            if name != "tactile_contact" and not np.all(np.isfinite(value)):
                raise ValueError(f"XHandSample.{name} must be finite")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class XHand:
    """Thin stateful adapter around one worker-local SDK controller."""

    def __init__(self, config: XHandConfig):
        self.config = config
        self.connected_flag = False
        self.error_state = False
        self.last_error_code: int | None = None
        self.last_error_message = ""
        self.last_qpos_cmd: np.ndarray | None = None
        self.tactile_calibrated = False
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

    def connect(self) -> bool:
        """Open the configured device and seed command history from live feedback."""
        if self.connected_flag:
            return True

        try:
            if not self._open_device():
                return False
            if (
                self.config.comm_type == "serial"
                and self.config.rs485_post_open_settle_s
            ):
                time.sleep(self.config.rs485_post_open_settle_s)

            self.connected_flag = True
            hand_ids = list(self._control.list_hands_id())
            if self.config.device_id not in hand_ids:
                raise RuntimeError(
                    f"configured device_id={self.config.device_id} not found in {hand_ids}"
                )
            self.error_state = False
            self._read_identity()
            self._command = self._make_command(
                np.asarray(self.config.home_qpos, dtype=np.float64)
            )
            self._seed_command_history()
            return True
        except Exception:
            self.error_state = True
            logger.error("XHand initialization failed", exc_info=True)
            self.disconnect()
            return False

    def _open_device(self) -> bool:
        retries = _OPEN_RETRIES[self.config.comm_type]
        device_name = self.config.device_name

        for attempt in range(1, retries + 1):
            self._control = xhc.XHandControl()
            if device_name is None:
                devices, _ = self._captured_sdk_call(
                    "discovery",
                    lambda: self._control.enumerate_devices(
                        _SDK_PROTOCOL[self.config.comm_type]
                    ),
                )
                if not devices:
                    self.last_error_code = -2
                    self.last_error_message = (
                        f"no XHand device found for {self.config.comm_type}"
                    )
                    self.error_state = True
                    self._close_control()
                    logger.warning(_CONNECTION_HINT[self.config.comm_type])
                    return False
                device_name = devices[0]

            if self.config.comm_type == "serial":
                open_call = lambda: self._control.open_serial(
                    device_name, self.config.baudrate
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
                if attempt > 1 and self.config.comm_type == "ethercat":
                    time.sleep(1.0)
                return True

            self._set_error(error)
            self._close_control()
            if attempt < retries:
                delay = _OPEN_RETRY_DELAY_S
                if self.config.comm_type == "ethercat" and attempt == 1:
                    delay = max(delay, _STALE_EC_RECOVERY_S)
                logger.warning(
                    "XHand open attempt %d/%d failed: %s; retrying in %.1fs",
                    attempt,
                    retries,
                    self.last_error_message,
                    delay,
                )
                time.sleep(delay)

        self.error_state = True
        logger.error(
            "XHand connect failed after %d attempts: %s",
            retries,
            self.last_error_message,
        )
        logger.warning(_CONNECTION_HINT[self.config.comm_type])
        return False

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
                error, value = getter(self.config.device_id)
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
            self.config.device_id,
        )

    def _seed_command_history(self) -> None:
        for attempt in range(_INITIAL_STATE_READ_ATTEMPTS):
            try:
                sample = self.get_state()
                self.last_qpos_cmd = np.clip(
                    sample.qpos, self.config.qpos_min, self.config.qpos_max
                )
                return
            except (RuntimeError, ValueError):
                if attempt + 1 < _INITIAL_STATE_READ_ATTEMPTS:
                    time.sleep(_INITIAL_STATE_READ_INTERVAL_S)
        raise RuntimeError("initial XHand state is unavailable or invalid")

    def disconnect(self) -> None:
        """Release the SDK handle; repeated calls are no-ops."""
        connected_ethercat = self.connected_flag and self.config.comm_type == "ethercat"
        if self._control is not None:
            if connected_ethercat:
                self._request_ethercat_init()
            self._close_control()
            if connected_ethercat:
                time.sleep(_POST_EC_DISCONNECT_S)
        self.connected_flag = False

    def _request_ethercat_init(self) -> None:
        if self.config.ethercat_slave_position < 0:
            logger.warning(
                "XHand EtherCAT slave position unknown; skipping explicit INIT request"
            )
            return
        try:
            error, _ = self._control.set_firmware_state(
                self.config.device_id,
                self.config.ethercat_slave_position,
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

    def initialize_tactile(self) -> bool:
        """Reset sensors and estimate a no-contact bias without gating joint control."""
        self._tactile_bias_sum = None
        self._tactile_bias_raw = None
        self.tactile_calibrated = False
        startup = self.get_state()
        if not startup.tactile_sum_valid or self._tactile_load_present(startup):
            logger.error(
                "Tactile calibration refused: contact/load or invalid data at startup"
            )
            return False
        self._reset_tactile_sensors()
        self._capture_tactile_bias()
        return self.tactile_calibrated

    def _tactile_load_present(self, sample: XHandSample) -> bool:
        if sample.tactile_valid:
            magnitude = np.linalg.norm(sample.tactile_force, axis=2)
            return bool(np.any(magnitude > self.config.raw_force_contact_threshold))
        magnitude = np.linalg.norm(sample.tactile_sum, axis=1)
        return bool(np.any(magnitude > self.config.tactile_contact_threshold))

    def _reset_tactile_sensors(self) -> None:
        unsupported = 0
        for sensor_id in _TACTILE_SENSOR_IDS:
            reset_sensor = partial(
                self._control.reset_sensor, self.config.device_id, sensor_id
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
            raise RuntimeError("incomplete tactile data during bias capture")
        if any(self._tactile_load_present(sample) for sample in samples):
            raise RuntimeError("contact/load detected during tactile bias capture")
        self._tactile_bias_sum = np.mean(
            np.stack([sample.tactile_sum for sample in samples]), axis=0
        )
        self._tactile_bias_raw = np.mean(
            np.stack([sample.tactile_force for sample in samples]), axis=0
        )
        self.tactile_calibrated = True

    # State and commands

    def get_state(self) -> XHandSample:
        """Return fresh joint feedback; sensor-only errors degrade tactile fields."""
        error, state = self._read_raw_state()
        sensor_status = self._is_sensor_status(error)
        if (not _error_ok(error) and not sensor_status) or state is None:
            self._set_error(error)
            raise XHandReadError(
                -1 if self.last_error_code is None else self.last_error_code,
                self.last_error_message,
                connected=self.connected_flag,
            )

        self._update_sensor_status("read", error)
        if sensor_status:
            code = _error_code(error)
            assert code is not None
            raw_valid, sum_valid, _ = _RS485_SENSOR_STATUS[code]
        else:
            raw_valid, sum_valid = True, True
        sample = self._parse_sample(
            state, raw_expected=raw_valid, sum_expected=sum_valid
        )
        self.error_state = any(
            np.any(getattr(sample, name))
            for name in ("commboard_err", "jointboard_err", "tipboard_err")
        )
        return sample

    def _read_raw_state(self) -> tuple[Any, Any]:
        if self._control is None or not self.connected_flag:
            return None, None
        # Control and recording require a live transaction, never the SDK cache.
        call = lambda: self._control.read_state(self.config.device_id, True)
        error, state = call()
        if self.config.comm_type != "serial":
            return error, state
        for attempt in range(1, self.config.rs485_read_crc_retry_count + 1):
            if _error_code(error) != _RS485_CRC_ERROR:
                break
            self._crc_backoff(
                "state read", attempt, self.config.rs485_read_crc_retry_count
            )
            error, state = call()
        return error, state

    def send_action(self, action: np.ndarray) -> bool:
        """Validate and send one absolute joint endpoint."""
        target = np.asarray(action, dtype=np.float64)
        problem = self._command_problem(target)
        if problem is not None:
            self.last_error_message = problem
            logger.warning(problem)
            return False
        if self._control is None or self._command is None or not self.connected_flag:
            self.error_state = True
            self.last_error_message = "XHand command path is not initialized"
            return False

        for index, value in enumerate(target):
            self._command.finger_command[index].position = float(value)
        error = self._send_with_crc_retry()
        if not _error_ok(error) and not self._is_sensor_status(error):
            self._set_error(error)
            return False

        self._update_sensor_status("send", error)
        self.last_qpos_cmd = target.copy()
        return True

    def _send_with_crc_retry(self) -> Any:
        call = lambda: self._control.send_command(self.config.device_id, self._command)
        error = call()
        if self.config.comm_type != "serial":
            return error
        for attempt in range(1, self.config.rs485_crc_retry_count + 1):
            if _error_code(error) != _RS485_CRC_ERROR:
                break
            self._crc_backoff("command", attempt, self.config.rs485_crc_retry_count)
            error = call()
        return error

    def _crc_backoff(self, operation: str, attempt: int, attempts: int) -> None:
        delay = float(self.config.rs485_crc_retry_backoff_s)
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
        overrides = (
            ("kp", self.config.kp_per_joint, self.config.kp),
            ("ki", self.config.ki_per_joint, self.config.ki),
            ("kd", self.config.kd_per_joint, self.config.kd),
            ("tor_max", self.config.tor_max_per_joint, self.config.tor_max),
        )
        for index in range(HAND_DOF):
            joint = command.finger_command[index]
            joint.id = index
            joint.position = float(qpos[index])
            joint.mode = int(self.config.mode)
            for name, values, fallback in overrides:
                setattr(joint, name, int(fallback if values is None else values[index]))
            for name in ("res0", "res1", "res2", "res3"):
                setattr(joint, name, 0)
        return command

    def _command_problem(self, qpos: np.ndarray) -> str | None:
        if qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
            return "XHand.send_action rejected invalid shape or NaN/Inf"
        command_bad = np.flatnonzero(
            (qpos < np.asarray(self.config.qpos_min) - 1e-12)
            | (qpos > np.asarray(self.config.qpos_max) + 1e-12)
        )
        mechanical_bad = np.flatnonzero(
            (qpos < np.asarray(self.config.mechanical_qpos_min) - 1e-12)
            | (qpos > np.asarray(self.config.mechanical_qpos_max) + 1e-12)
        )
        bad = mechanical_bad if mechanical_bad.size else command_bad
        if not bad.size:
            return None
        index = int(bad[0])
        if mechanical_bad.size:
            label = "mechanical"
            lower, upper = (
                self.config.mechanical_qpos_min,
                self.config.mechanical_qpos_max,
            )
        else:
            label = "command"
            lower, upper = self.config.qpos_min, self.config.qpos_max
        return (
            f"XHand.send_action rejected {label} joint limit violation: joint={index} "
            f"target={qpos[index]:.6f}rad range=[{lower[index]:.6f},{upper[index]:.6f}]rad"
        )

    # Parsing and status handling

    def _parse_sample(
        self, state: Any, *, raw_expected: bool, sum_expected: bool
    ) -> XHandSample:
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
            np.linalg.norm(tactile_sum, axis=1) > self.config.tactile_contact_threshold
            if tactile_sum_valid
            else np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
        )
        return XHandSample(
            qpos=qpos,
            current=current,
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
                raise RuntimeError(
                    f"XHand parse: invalid or duplicate joint id {index}"
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
            raise RuntimeError(f"XHand parse: {len(seen)}/{HAND_DOF} joints reported")
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
            self.config.comm_type == "serial"
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
            self.last_error_code = code
            self.last_error_message = str(getattr(error, "error_message", ""))
            return
        if previous is not None:
            logger.info("XHand RS485 %s sensor response recovered", operation)
        self._sensor_status[operation] = None
        self.last_error_code = 0
        self.last_error_message = ""

    def _set_error(self, error: Any) -> None:
        code = _error_code(error)
        self.last_error_code = -1 if code is None else code
        self.last_error_message = str(
            getattr(error, "error_message", "empty error object")
        )
        self.error_state = True
