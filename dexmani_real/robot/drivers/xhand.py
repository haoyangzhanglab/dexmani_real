"""Worker-local XHand driver with intentionally single-shot runtime I/O.

Runtime reads accept known sensor/CRC statuses only when their returned
12-DoF joint payload is complete and finite.  Aggregate (``calc_force``) and
dense (``raw_force``) tactile payloads carry independent validity so an RS485
distributed-force drop keeps aggregate contact force usable.  Runtime sends
make one SDK call.  A CRC response leaves delivery unconfirmed but does not
stop the worker; other SDK errors remain rejected.  There is no retry,
backoff, watchdog, or recovery state in this driver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
from xhand_controller import xhand_control as xhc  # type: ignore[import-untyped]

from dexmani_real.config.defaults import HandParams
from dexmani_real.robot.model import (
    HAND_CONTACT_SHAPE,
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
    get_logger,
)

logger = get_logger(__name__)

_SDK_PROTOCOL = {"ethercat": "EtherCAT", "serial": "RS485"}
# Fixed, bounded driver policies; runtime config only carries deployment tuning.
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
        _COMBINED_FORCE_UNAVAILABLE_CODE,  # combined force unavailable
        _DISTRIBUTED_FORCE_UNAVAILABLE_CODE,  # distributed force unavailable
        _TEMPERATURE_UNAVAILABLE_CODE,  # temperature unavailable
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
_POSITION_MODE = 3
# Derived convenience contact bit: aggregate ||calc_force|| above this in
# XHand SDK-native unknown units (not Newtons). Dense taxels stay continuous.
_TACTILE_CONTACT_THRESHOLD = 2.0
# Post-bias no-contact residual bound for calibration verification, also in
# XHand SDK-native unknown units. Semantically distinct from the contact
# threshold even though both currently equal 2.0, so they stay tunable apart.
_TACTILE_CALIBRATION_RESIDUAL_THRESHOLD = 2.0
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
    """Return ``(calc_force_valid, raw_force_valid)`` for one read status.

    Serial/RS485 exposes partial tactile statuses: a distributed-force drop
    (``1501019``) leaves combined ``calc_force`` valid while ``raw_force`` is
    unavailable, and a temperature drop (``1501020``) keeps both force fields.
    EtherCAT is not known to share those partial semantics, so any nonzero
    status fails tactile closed there while joints stay usable per the caller.
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
    """A fail-fast startup or tactile-calibration XHand operation error."""

    def __init__(self, operation: str, code: int, message: str) -> None:
        self.operation = str(operation)
        self.code = int(code)
        self.message = str(message)
        super().__init__(
            f"XHand {self.operation} failed: code={self.code} msg={self.message}"
        )


class XHandSendStatus(Enum):
    """SDK result without conflating CRC uncertainty with command acceptance."""

    ACCEPTED = "accepted"
    CRC_UNCONFIRMED = "crc_unconfirmed"
    REJECTED = "rejected"


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
        """Open the configured device and seed its command buffer from live feedback."""
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
        """Estimate a software no-contact bias without gating joint control.

        This assumes the operator started the hand with fingertips free and
        unloaded; uncalibrated absolute force cannot prove no-contact.  A
        candidate bias is captured, published, then independently verified.
        A failed verification clears both biases so a half-calibrated state is
        never left behind.
        """
        self._tactile_bias_sum = None
        self._tactile_bias_raw = None
        bias_sum, bias_raw = self._capture_tactile_bias()
        # Publish both candidate biases together, then verify from fresh reads.
        self._tactile_bias_sum = bias_sum
        self._tactile_bias_raw = bias_raw
        if not self._verify_tactile_bias():
            logger.error(
                "Tactile calibration failed post-bias verification; biases cleared"
            )
            self._tactile_bias_sum = None
            self._tactile_bias_raw = None
            return False
        logger.info(
            "XHand tactile software bias calibrated from %d no-contact samples",
            _TACTILE_BIAS_SAMPLE_COUNT,
        )
        return self.tactile_calibrated

    def _capture_tactile_bias(self) -> tuple[np.ndarray, np.ndarray]:
        """Collect candidate ``(bias_sum, bias_raw)`` without declaring success.

        Every capture read must report both aggregate and dense payloads as
        valid and finite; a failure raises rather than returning a partial
        candidate.
        """
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
            if not state.tactile_sum_valid or not state.tactile_valid:
                raise XHandError(
                    "calibrate_tactile",
                    -1,
                    "incomplete tactile data during bias capture",
                )
            samples.append(state)
        bias_sum = np.mean(np.stack([sample.tactile_sum for sample in samples]), axis=0)
        bias_raw = np.mean(
            np.stack([sample.tactile_force for sample in samples]), axis=0
        )
        return bias_sum, bias_raw

    def _verify_tactile_bias(self) -> bool:
        """Independently verify the published bias from three fresh reads.

        Aggregate no-contact residual must stay within the small SDK-native
        residual threshold.  Dense payloads are checked structurally (valid and
        finite) but never against a hard per-taxel magnitude threshold.
        """
        aggregate_peak = 0.0
        for _ in range(_TACTILE_VERIFY_SAMPLE_COUNT):
            time.sleep(_TACTILE_BIAS_SAMPLE_INTERVAL_S)
            state = self.get_state()
            if state is None or not state.tactile_sum_valid or not state.tactile_valid:
                logger.warning("tactile post-bias verification frame invalid")
                return False
            aggregate_peak = max(
                aggregate_peak,
                float(np.max(np.linalg.norm(state.tactile_sum, axis=1))),
            )
            dense_magnitudes = np.abs(state.tactile_force)
            logger.info(
                "tactile verify: dense abs_max=%.3g p99=%.3g",
                float(np.max(dense_magnitudes)),
                float(np.percentile(dense_magnitudes, 99)),
            )
        return aggregate_peak <= _TACTILE_CALIBRATION_RESIDUAL_THRESHOLD

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
        sum_valid, dense_valid = _tactile_validity(code, comm_type=self.cfg.comm_type)
        if sum_valid:
            try:
                tactile_sum = self._parse_tactile_sum(raw_state)
            except (AttributeError, TypeError, ValueError, OverflowError):
                logger.warning("XHand tactile sum payload invalid", exc_info=True)
                sum_valid = False
                tactile_sum.fill(0.0)
            else:
                # Bias subtraction runs only after a successful parse so an
                # internal bias fault is not mislabeled as a malformed read.
                if self._tactile_bias_sum is not None:
                    tactile_sum = tactile_sum - self._tactile_bias_sum
        if dense_valid:
            try:
                tactile_force = self._parse_tactile_force(raw_state)
            except (AttributeError, TypeError, ValueError, OverflowError):
                logger.warning("XHand tactile force payload invalid", exc_info=True)
                dense_valid = False
                tactile_force.fill(0.0)
            else:
                if self._tactile_bias_raw is not None:
                    tactile_force = tactile_force - self._tactile_bias_raw
        tactile_contact = (
            np.linalg.norm(tactile_sum, axis=1) > _TACTILE_CONTACT_THRESHOLD
            if sum_valid
            else np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
        )
        return XHandState(
            qpos=qpos,
            current_ma=current,
            tactile_force=tactile_force,
            tactile_sum=tactile_sum,
            tactile_contact=tactile_contact,
            tactile_sum_valid=sum_valid,
            tactile_valid=dense_valid,
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
                self._command.finger_command[index].position = float(value)
            error = self._control.send_command(self.cfg.device_id, self._command)
        except Exception:
            logger.warning("XHand send_command raised", exc_info=True)
            return XHandSendStatus.REJECTED

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
        lower = np.asarray(self.cfg.mechanical_qpos_min_rad, dtype=np.float64)
        upper = np.asarray(self.cfg.mechanical_qpos_max_rad, dtype=np.float64)
        if np.any(qpos < lower - 1e-12) or np.any(qpos > upper + 1e-12):
            raise ValueError(
                "XHand.send_action target violates mechanical joint limits"
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

    def _parse_tactile_sum(self, state: Any) -> np.ndarray:
        """Return the SDK-native ``[5,3]`` aggregate ``calc_force`` payload."""
        sensors = self._sensor_data(state)
        force_sum = np.empty(HAND_TACTILE_SUM_SHAPE, dtype=np.float64)
        for sensor_index, sensor in enumerate(sensors):
            force_sum[sensor_index] = _force_xyz(
                getattr(sensor, "calc_force", None),
                f"sensor_data[{sensor_index}].calc_force",
            )
        return force_sum

    def _parse_tactile_force(self, state: Any) -> np.ndarray:
        """Return the SDK-native ``[5,120,3]`` dense ``raw_force`` payload."""
        sensors = self._sensor_data(state)
        tactile_force = np.empty(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)
        for sensor_index, sensor in enumerate(sensors):
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
        return tactile_force
