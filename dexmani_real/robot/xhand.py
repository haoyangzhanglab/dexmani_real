"""Worker-local XHand driver with intentionally single-shot runtime I/O.

Runtime reads accept known sensor/CRC statuses only when their returned
12-DoF joint payload is complete and finite; tactile data is then invalid.
Runtime sends make one SDK call.  SDK errors are logged and affect only that
transaction: there is no retry, backoff, watchdog, or recovery state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
READ_USABLE_CODES = frozenset(
    {
        0,
        1_501_018,  # combined force unavailable
        1_501_019,  # distributed force unavailable
        1_501_020,  # temperature unavailable
        1_501_070,  # communication CRC with a complete joint payload
    }
)
SEND_ACCEPTED_CODES = frozenset(
    {
        0,
        1_501_018,
        1_501_019,
        1_501_020,
        1_501_035,  # configured-current overrun / expected grasp contact
    }
)

_EC_STATE_INIT = 1
_STALE_EC_RECOVERY_S = 3.0
_POST_EC_DISCONNECT_S = 2.0
_TACTILE_SCALE = 0.1
_TACTILE_BIAS_SAMPLE_COUNT = 5
_TACTILE_BIAS_SAMPLE_INTERVAL_S = 0.02
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
    """A fail-fast startup or tactile-calibration XHand operation error."""

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
            sample = self.get_state()
            if sample is not None:
                self._command = self._make_command(sample.qpos)
                self.last_qpos_cmd = np.clip(
                    sample.qpos,
                    self.cfg.mechanical_qpos_min_rad,
                    self.cfg.mechanical_qpos_max_rad,
                )
                return
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

    def calibrate_tactile(self) -> bool:
        """Estimate a software-only no-contact bias without gating joint control."""
        self._tactile_bias_sum = None
        self._tactile_bias_raw = None
        startup = self.get_state()
        if (
            startup is None
            or not startup.tactile_valid
            or self._tactile_load_present(startup)
        ):
            logger.error(
                "Tactile calibration refused: contact/load or incomplete data at startup"
            )
            return False
        self._capture_tactile_bias()
        logger.info(
            "XHand tactile software bias calibrated from %d no-contact samples",
            _TACTILE_BIAS_SAMPLE_COUNT,
        )
        return self.tactile_calibrated

    def _tactile_load_present(self, state: XHandState) -> bool:
        if state.tactile_valid:
            magnitude = np.linalg.norm(state.tactile_force, axis=2)
            return bool(np.any(magnitude > _RAW_FORCE_CONTACT_THRESHOLD))
        magnitude = np.linalg.norm(state.tactile_sum, axis=1)
        return bool(np.any(magnitude > _TACTILE_CONTACT_THRESHOLD))

    def _capture_tactile_bias(self) -> None:
        samples: list[XHandState] = []
        for _ in range(_TACTILE_BIAS_SAMPLE_COUNT):
            # Space live RS485 reads so startup calibration does not burst the bus.
            time.sleep(_TACTILE_BIAS_SAMPLE_INTERVAL_S)
            state = self.get_state()
            if state is None:
                raise XHandError(
                    "calibrate_tactile",
                    -1,
                    "joint state unavailable during bias capture",
                )
            samples.append(state)
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
        tactile_bias_sum = np.mean(
            np.stack([sample.tactile_sum for sample in samples]), axis=0
        )
        tactile_bias_raw = np.mean(
            np.stack([sample.tactile_force for sample in samples]), axis=0
        )
        # Publish both biases together so calibration is never half-initialized.
        self._tactile_bias_sum = tactile_bias_sum
        self._tactile_bias_raw = tactile_bias_raw

    def get_state(self) -> XHandState | None:
        """Read one fresh state, returning ``None`` for runtime SDK failures."""
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

        tactile_force = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        tactile_sum = np.zeros(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        tactile_valid = code == 0
        if tactile_valid:
            try:
                tactile_force, tactile_sum = self._parse_tactile(raw_state)
            except (AttributeError, TypeError, ValueError, OverflowError):
                logger.warning("XHand tactile payload invalid", exc_info=True)
                tactile_valid = False
                tactile_force.fill(0.0)
                tactile_sum.fill(0.0)
        tactile_contact = (
            np.linalg.norm(tactile_sum, axis=1) > _TACTILE_CONTACT_THRESHOLD
            if tactile_valid
            else np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
        )
        return XHandState(
            qpos=qpos,
            current_ma=current,
            tactile_force=tactile_force,
            tactile_sum=tactile_sum,
            tactile_contact=tactile_contact,
            tactile_sum_valid=tactile_valid,
            tactile_valid=tactile_valid,
            **board_errors,
        )

    def send_action(self, action: np.ndarray) -> bool:
        """Send one absolute endpoint and report whether the SDK accepted it."""
        target = np.asarray(action, dtype=np.float64)
        self._validate_action(target)
        if self._control is None or self._command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")
        target = np.clip(
            target,
            self.cfg.mechanical_qpos_min_rad,
            self.cfg.mechanical_qpos_max_rad,
        )

        try:
            for index, value in enumerate(target):
                self._command.finger_command[index].position = float(value)
            error = self._control.send_command(self.cfg.device_id, self._command)
        except Exception:
            logger.warning("XHand send_command raised", exc_info=True)
            return False

        code = _error_code(error)
        if code not in SEND_ACCEPTED_CODES:
            logger.warning(
                "XHand send failed: code=%s msg=%s",
                code,
                getattr(error, "error_message", ""),
            )
            return False
        self.last_qpos_cmd = target.copy()
        return True

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
                raise ValueError(f"invalid or duplicate joint id {index}")
            seen.add(index)
            qpos[index] = float(getattr(joint, "position", np.nan))
            current[index] = float(getattr(joint, "torque", np.nan))
            errors["commboard_err"][index] = int(getattr(joint, "commboard_err", 0))
            errors["jointboard_err"][index] = int(
                getattr(joint, "jonitboard_err", getattr(joint, "jointboard_err", 0))
            )
            errors["tipboard_err"][index] = int(getattr(joint, "tipboard_err", 0))
        if len(seen) != HAND_DOF:
            raise ValueError(f"{len(seen)}/{HAND_DOF} joints reported")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("non-finite joint position feedback")
        if not np.all(np.isfinite(current)):
            raise ValueError("non-finite joint current feedback")
        return qpos, current, errors

    def _sensor_data(self, state: Any) -> list[Any]:
        sensors = list(state.sensor_data)
        if len(sensors) != HAND_FINGER_COUNT:
            raise ValueError(
                f"sensor_data must contain {HAND_FINGER_COUNT} sensors, got {len(sensors)}"
            )
        return sensors

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
