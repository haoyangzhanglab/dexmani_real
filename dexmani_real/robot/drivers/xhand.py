"""Single-call XHand reads/sends with strict joint validation and SDK status gating.
Tactile payloads retain local NaN/Inf for bounded repair by the robot I/O owner.
CRC reads return None; CRC sends retain unconfirmed delivery. Other rejected codes raise.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dexmani_real.config.hardware import HandParams
from dexmani_real.robot.commands import DispatchStatus
from dexmani_real.robot.model import (
    HAND_DOF,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
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
# These codes permit joint reads and target sends; tactile trust is checked separately.
_NONFATAL_SDK_CODES = frozenset(
    {
        0,
        _COMBINED_FORCE_UNAVAILABLE_CODE,
        _DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
        _TEMPERATURE_UNAVAILABLE_CODE,
    }
)

_EC_STATE_INIT = 1
_STALE_EC_RECOVERY_S = 3.0
_POST_EC_DISCONNECT_S = 2.0
_TACTILE_SENSOR_IDS = (0x11, 0x12, 0x13, 0x14, 0x15)
_PASSIVE_MODE = 0
_POSITION_MODE = 3
_STOP_RETRY_INTERVAL_S = 0.01
_STOP_RETRY_TIMEOUT_S = 0.25
_STOP_MAX_ATTEMPTS = 26
_CONNECTION_HINT = {
    "ethercat": "Check XHand power, EtherCAT cable/link, SDK permissions, and stale slave state",
    "serial": "Check XHand power, USB cable, and serial-device permissions",
}


def _error_code(error: Any) -> int:
    code = getattr(error, "error_code", -1)
    return -1 if code is None else int(code)


def _error_ok(error: Any) -> bool:
    return _error_code(error) == 0


def _tactile_validity(code: int, *, comm_type: str) -> tuple[bool, bool]:
    """Return (aggregate_allowed,dense_allowed) for SDK/tip-board status.
    RS485 1501019/201 drops dense only; 1501020/202 keeps both. EtherCAT rejects all nonzero tactile status.
    """
    if comm_type == "ethercat":
        return (code == 0, code == 0)
    if code == 0:
        return (True, True)
    if code in (_COMBINED_FORCE_UNAVAILABLE_CODE, 200):
        return (False, False)
    if code in (_DISTRIBUTED_FORCE_UNAVAILABLE_CODE, 201):
        return (True, False)
    if code in (_TEMPERATURE_UNAVAILABLE_CODE, 202):
        return (True, True)
    return (False, False)


class XHandError(RuntimeError):
    """An XHand SDK error retaining the device's original code."""

    def __init__(self, operation: str, code: int, message: str) -> None:
        self.operation = str(operation)
        self.code = int(code)
        self.message = str(message)
        super().__init__(f"XHand {self.operation} failed: code={self.code} msg={self.message}")


@dataclass
class XHandState:
    """One trustworthy joint read; tactile may contain NaN/Inf or status-masked NaN."""

    qpos: np.ndarray
    current_ma: np.ndarray
    tactile_aggregate: np.ndarray
    tactile_dense: np.ndarray
    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray


class XHand:
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
        self._tactile_reset_metadata: dict[str, object] = {}

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
            self._tactile_reset_metadata = {}
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
                code,
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
        logger.debug(
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

    def reset_tactile_sensors(self, *, cancel_requested=lambda: False) -> bool:
        """Request one reset per unloaded fingertip while idle; cancellation fences later calls, firmware effect is unproven."""
        if self._control is None or not self.connected_flag:
            raise RuntimeError("XHand is not connected")
        result = {
            "hand_tactile_reset_time_ns": time.time_ns(),
            "hand_tactile_reset_result": "incomplete (0/5)",
        }
        self._tactile_reset_metadata = result
        for completed, sensor_id in enumerate(_TACTILE_SENSOR_IDS):
            if cancel_requested():
                result["hand_tactile_reset_result"] = f"cancelled ({completed}/5)"
                raise CancelledError(
                    f"触觉复位请求已中断，已完成 {completed}/5 次 SDK 调用；已发送的请求不会撤销。"
                )
            result["hand_tactile_reset_result"] = (
                f"incomplete ({completed}/5), sensor=0x{sensor_id:02x}"
            )
            try:
                reply = self._control.reset_sensor(self.cfg.device_id, sensor_id)
            except Exception as exc:
                result["hand_tactile_reset_result"] = (
                    f"exception ({completed}/5), sensor=0x{sensor_id:02x}: {exc}"
                )
                logger.warning(
                    "触觉复位调用异常：sensor=0x%02x，已完成 %d/5 次：%s",
                    sensor_id, completed, exc,
                )
                return False
            code = _error_code(reply)
            if code != 0:
                result["hand_tactile_reset_result"] = (
                    f"failed ({completed}/5), sensor=0x{sensor_id:02x}, code={code}"
                )
                logger.warning(
                    "触觉复位调用失败：sensor=0x%02x，code=%s，已完成 %d/5 次：%s",
                    sensor_id, code, completed, getattr(reply, "error_message", ""),
                )
                return False
            result["hand_tactile_reset_result"] = f"sdk_completed ({completed + 1}/5)"
        logger.info("五指触觉复位请求的 SDK 调用完成（5/5），未确认固件复位或零点效果。")
        return True

    def get_tactile_reset_metadata(self) -> dict[str, object]:
        """Snapshot this connection's last explicit reset request without reads."""
        return {
            "hand_tactile_reset_result": "not_requested",
            **self._tactile_reset_metadata,
        }

    def get_state(self) -> XHandState | None:
        """One SDK read: only CRC returns None; other untrusted joint feedback raises.

        Tactile status masks untrusted channels before the robot owner's bounded hold.
        """
        if self._control is None or not self.connected_flag:
            raise RuntimeError("XHand is not connected")
        error, raw_state = self._control.read_state(self.cfg.device_id, True)
        code = _error_code(error)
        if code == _COMMUNICATION_CRC_ERROR_CODE:
            # Never inspect a CRC payload, even when it looks finite and well formed.
            return None
        if code not in _NONFATAL_SDK_CODES:
            raise XHandError(
                "read", code, getattr(error, "error_message", "")
            )
        if raw_state is None:
            raise XHandError("read", code, "missing state payload")

        try:
            qpos, current, board_errors = self._parse_joints(raw_state)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise XHandError("read", code, f"invalid joint payload: {exc}") from exc

        # The SDK's aggregate return code can hide another board's error.
        if any(np.any(board_errors[name]) for name in ("commboard_err", "jointboard_err")):
            raise XHandError("read", code, f"joint/communication board fault: {board_errors}")

        tactile_dense = np.full(HAND_TACTILE_FORCE_SHAPE, np.nan, dtype=np.float64)
        tactile_aggregate = np.full(HAND_TACTILE_SUM_SHAPE, np.nan, dtype=np.float64)
        aggregate_allowed, dense_allowed = _tactile_validity(code, comm_type=self.cfg.comm_type)
        if not aggregate_allowed or not dense_allowed:
            logger.debug("XHand tactile read incomplete: code=%s", code)
        if aggregate_allowed:
            try:
                tactile_aggregate = self._parse_tactile_aggregate(raw_state)
            except (AttributeError, TypeError, ValueError, OverflowError):
                logger.debug("XHand tactile aggregate payload invalid", exc_info=True)
        if dense_allowed:
            try:
                tactile_dense = self._parse_tactile_dense(raw_state)
            except (AttributeError, TypeError, ValueError, OverflowError):
                logger.debug("XHand tactile dense payload invalid", exc_info=True)
        for sensor_index, finger_id in enumerate(XHAND_TACTILE_SENSOR_FINGER_IDS):
            aggregate_allowed, dense_allowed = _tactile_validity(
                int(board_errors["tipboard_err"][finger_id]), comm_type=self.cfg.comm_type
            )
            if not aggregate_allowed:
                tactile_aggregate[sensor_index] = np.nan
            if not dense_allowed:
                tactile_dense[sensor_index] = np.nan
        return XHandState(
            qpos=qpos,
            current_ma=current,
            tactile_aggregate=tactile_aggregate,
            tactile_dense=tactile_dense,
            **board_errors,
        )

    def send_action(self, action: np.ndarray) -> DispatchStatus:
        """Send one absolute endpoint and preserve CRC delivery uncertainty."""
        target = np.asarray(action, dtype=np.float64)
        try:
            self._validate_action(target)
        except ValueError as exc:
            logger.warning("XHand send rejected: %s", exc)
            return DispatchStatus.REJECTED
        if self._control is None or self._command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")

        try:
            for index, value in enumerate(target):
                joint = self._command.finger_command[index]
                joint.mode = _POSITION_MODE
                joint.position = float(value)
        except Exception:
            logger.warning("XHand command preparation failed", exc_info=True)
            return DispatchStatus.REJECTED
        return self._send_command()

    def hold_current(self, qpos: np.ndarray) -> DispatchStatus:
        """Actively hold measured joint positions clipped to mechanical limits."""
        measured = np.asarray(qpos, dtype=np.float64)
        if measured.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(measured)):
            raise ValueError("XHand.hold_current requires twelve finite joint positions")
        return self.send_action(
            np.clip(measured, self.cfg.mechanical_qpos_min_rad, self.cfg.mechanical_qpos_max_rad)
        )

    def set_passive(self) -> DispatchStatus:
        if self._control is None or self._command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")
        for index in range(HAND_DOF):
            self._command.finger_command[index].mode = _PASSIVE_MODE
        return self._send_command()

    def stop(self, qpos: np.ndarray | None = None, *, timestamp_ns: int = 0) -> None:
        """Hold fresh measured joints or request passive mode; only stopping retries CRC.
        SDK acceptance does not prove physical standstill.
        """
        if not self.is_connected:
            raise RuntimeError("XHand disconnected before stop")
        deadline = time.monotonic() + _STOP_RETRY_TIMEOUT_S
        for attempt in range(_STOP_MAX_ATTEMPTS):
            now_ns = time.monotonic_ns()
            fresh = (
                0 < int(timestamp_ns) <= now_ns
                and now_ns - int(timestamp_ns) < int(self.cfg.feedback_max_age_s * 1e9)
            )
            status = self.hold_current(qpos) if qpos is not None and fresh else self.set_passive()
            if status == DispatchStatus.ACCEPTED:
                if attempt:
                    logger.info("XHand stop SDK response recovered after %d attempts", attempt + 1)
                return
            if status != DispatchStatus.CRC_UNCONFIRMED:
                raise RuntimeError(f"XHand stop rejected: {status.name}")
            if attempt == 0:
                logger.warning("XHand stop CRC; retrying within the stop deadline")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or attempt + 1 == _STOP_MAX_ATTEMPTS:
                break
            time.sleep(min(_STOP_RETRY_INTERVAL_S, remaining))
        raise RuntimeError("XHand stop confirmation timed out: CRC_UNCONFIRMED")

    def _send_command(self) -> DispatchStatus:
        error = self._control.send_command(self.cfg.device_id, self._command)

        code = _error_code(error)
        if code == _COMMUNICATION_CRC_ERROR_CODE:
            return DispatchStatus.CRC_UNCONFIRMED
        if code not in _NONFATAL_SDK_CODES:
            # 1501035 is overcurrent, not contact; termination retains UNKNOWN delivery and the SDK error.
            raise XHandError("send", code, getattr(error, "error_message", ""))
        return DispatchStatus.ACCEPTED

    def _make_command(self, qpos: np.ndarray) -> Any:
        command = self._sdk.HandCommand_t()
        for index in range(HAND_DOF):
            joint = command.finger_command[index]
            joint.id = index
            joint.position = float(qpos[index])
            joint.kp = int(self.cfg.kp[index])
            joint.ki = int(self.cfg.ki)
            joint.kd = int(self.cfg.kd)
            joint.tor_max = int(self.cfg.tor_max[index])
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

    @staticmethod
    def _parse_tactile_aggregate(state: Any) -> np.ndarray:
        """Return the SDK-native ``[5,3]`` aggregate ``calc_force`` payload."""
        force_aggregate = np.asarray(
            [[sensor.calc_force.fx, sensor.calc_force.fy, sensor.calc_force.fz]
             for sensor in state.sensor_data],
            dtype=np.float64,
        )
        if force_aggregate.shape != HAND_TACTILE_SUM_SHAPE:
            raise ValueError("aggregate tactile must have shape [5,3]")
        return force_aggregate

    @staticmethod
    def _parse_tactile_dense(state: Any) -> np.ndarray:
        """Return the SDK-native ``[5,120,3]`` dense ``raw_force`` payload."""
        tactile_dense = np.asarray(
            [[[force.fx, force.fy, force.fz] for force in sensor.raw_force]
             for sensor in state.sensor_data],
            dtype=np.float64,
        )
        if tactile_dense.shape != HAND_TACTILE_FORCE_SHAPE:
            raise ValueError("dense tactile must have shape [5,120,3]")
        return tactile_dense
