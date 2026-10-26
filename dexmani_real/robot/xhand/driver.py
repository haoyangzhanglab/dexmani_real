"""XHand SDK 边界；CRC 不代表发送成功，触觉有效性单独判断。"""

from __future__ import annotations

import logging
import time
from concurrent.futures import CancelledError
from typing import Any, Callable

import numpy as np

from dexmani_real.robot.model import (
    HAND_DOF,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    XHAND_RATED_QPOS_MAX_RAD,
    XHAND_RATED_QPOS_MIN_RAD,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
)
from dexmani_real.robot.xhand.config import (
    INITIAL_STATE_TIMEOUT_S,
    SERIAL_SETTLE_S,
    HandConfig,
)
from dexmani_real.robot.xhand.feedback import (
    COMBINED_FORCE_UNAVAILABLE_CODE,
    DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
    TEMPERATURE_UNAVAILABLE_CODE,
    XHandState,
    parse_joints,
    parse_tactile_aggregate,
    parse_tactile_dense,
    tactile_validity,
)
from dexmani_real.utils.log import (
    capture_native_stdout,
    extract_native_diagnostics,
)

logger = logging.getLogger(__name__)

SDK_PROTOCOL = {"ethercat": "EtherCAT", "serial": "RS485"}
INITIAL_STATE_READ_INTERVAL_S = 0.02
COMMUNICATION_CRC_ERROR_CODE = 1_501_070
# 这些状态仍允许关节读写，触觉有效性单独判断。
NONFATAL_SDK_CODES = frozenset(
    {
        0,
        COMBINED_FORCE_UNAVAILABLE_CODE,
        DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
        TEMPERATURE_UNAVAILABLE_CODE,
    }
)

EC_STATE_INIT = 1
POST_EC_DISCONNECT_S = 2.0
TACTILE_SENSOR_IDS = (0x11, 0x12, 0x13, 0x14, 0x15)
PASSIVE_MODE = 0
POSITION_MODE = 3
STOP_RETRY_INTERVAL_S = 0.01
STOP_RETRY_TIMEOUT_S = 0.25


def error_code(error: Any) -> int:
    return int(error.error_code)


class XHandError(RuntimeError):
    def __init__(self, operation: str, code: int, message: str) -> None:
        self.operation = str(operation)
        self.code = int(code)
        self.message = str(message)
        super().__init__(f"XHand {self.operation} failed: code={self.code} msg={self.message}")


class XHand:
    def __init__(self, config: HandConfig):
        self.cfg = config
        self.mechanical_lower = np.asarray(XHAND_RATED_QPOS_MIN_RAD, dtype=np.float64)
        self.mechanical_upper = np.asarray(XHAND_RATED_QPOS_MAX_RAD, dtype=np.float64)
        self.connected_flag = False
        self.send_crc_started_ns: int | None = None
        self.device_identity = {
            "backend": "hardware",
            "hand_type": "unavailable",
            "sdk_version": "unavailable",
            "serial_number": "unavailable",
        }

        self.control: Any = None
        self.command: Any = None
        self.tactile_reset_metadata: dict[str, object] = {}

    @property
    def is_connected(self) -> bool:
        return self.connected_flag

    def connect(self, *, cancel_requested=lambda: False) -> int | None:
        """返回与首次触觉采样共享的单调时钟截止时间，单位 ns。"""
        from xhand_controller import xhand_control

        self.sdk = xhand_control
        if self.connected_flag:
            return None

        try:
            self.open_device()
            self.tactile_reset_metadata = {}
            if self.cfg.comm_type == "serial":
                time.sleep(SERIAL_SETTLE_S)

            self.connected_flag = True
            hand_ids = list(self.control.list_hands_id())
            if self.cfg.device_id not in hand_ids:
                raise XHandError(
                    "connect",
                    -1,
                    f"configured device_id={self.cfg.device_id} not found in {hand_ids}",
                )
            self.read_identity()
            return self.seed_command_history(cancel_requested=cancel_requested)
        except BaseException:
            try:
                self.disconnect()
            except Exception:
                logger.exception("XHand connection cleanup failed")
            raise

    def open_device(self) -> None:
        self.control = self.sdk.XHandControl()
        device_name = self.cfg.device_name
        if device_name is None:
            devices, _ = self.captured_sdk_call(
                "discovery",
                lambda: self.control.enumerate_devices(SDK_PROTOCOL[self.cfg.comm_type]),
            )
            if not devices:
                raise XHandError("connect", -2, f"no device found for {self.cfg.comm_type}")
            device_name = devices[0]

        if self.cfg.comm_type == "serial":
            open_call = lambda: self.control.open_serial(device_name, self.cfg.baudrate)
        else:
            open_call = lambda: self.control.open_ethercat(device_name)
        error, output = self.captured_sdk_call(
            "open", open_call, ignore=("Operation not permitted",)
        )
        if error_code(error) != 0:
            raise XHandError("connect", error_code(error), error.error_message)
        if "Operation not permitted" in output:
            logger.warning("XHand real-time scheduling unavailable; using normal scheduling")

    def captured_sdk_call(
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

    def read_identity(self) -> None:
        self.device_identity["sdk_version"] = str(self.control.get_sdk_version())
        for key, getter in (
            ("hand_type", self.control.get_hand_type),
            ("serial_number", self.control.get_serial_number),
        ):
            error, value = getter(self.cfg.device_id)
            if error_code(error) == 0:
                self.device_identity[key] = str(value)
            else:
                logger.warning("XHand %s unavailable: code=%s", key, error_code(error))
        logger.debug(
            "XHand ready: SDK=%s type=%s serial=%s device_id=%d",
            self.device_identity["sdk_version"],
            self.device_identity["hand_type"],
            self.device_identity["serial_number"],
            self.cfg.device_id,
        )

    def seed_command_history(self, *, cancel_requested=lambda: False) -> int:
        deadline_ns = time.monotonic_ns() + int(INITIAL_STATE_TIMEOUT_S * 1e9)
        while time.monotonic_ns() < deadline_ns:
            if cancel_requested():
                raise CancelledError("XHand initial joint feedback acquisition cancelled")
            sample = self.get_state()
            if cancel_requested():
                raise CancelledError("XHand initial joint feedback acquisition cancelled")
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                break
            if sample is not None:
                self.command = self.make_command(sample.qpos)
                return deadline_ns
            time.sleep(min(INITIAL_STATE_READ_INTERVAL_S, remaining_ns / 1e9))
        raise XHandError("connect", -1, "initial XHand joint feedback timed out")

    def disconnect(self) -> None:
        connected_ethercat = self.connected_flag and self.cfg.comm_type == "ethercat"
        errors = []
        try:
            if self.control is not None:
                if connected_ethercat:
                    try:
                        self.request_ethercat_init()
                    except Exception as exc:
                        errors.append(exc)
                try:
                    self.close_control()
                except Exception as exc:
                    errors.append(exc)
                if connected_ethercat:
                    time.sleep(POST_EC_DISCONNECT_S)
        finally:
            self.connected_flag = False
        if errors:
            raise RuntimeError(f"XHand disconnect failed: {errors}") from errors[0]

    def request_ethercat_init(self) -> None:
        if self.cfg.ethercat_slave_position < 0:
            logger.warning("XHand EtherCAT slave position unknown; skipping explicit INIT request")
            return
        error, _ = self.control.set_firmware_state(
            self.cfg.device_id,
            self.cfg.ethercat_slave_position,
            EC_STATE_INIT,
            500_000,
        )
        if error_code(error) != 0:
            raise XHandError("ethercat INIT", error_code(error), "INIT request failed")
        time.sleep(0.2)

    def close_control(self) -> None:
        if self.control is not None:
            self.control.close_device()
            self.control = None

    def reset_tactile_sensors(self, *, cancel_requested=lambda: False) -> bool:
        """仅在静止、卸载时复位；SDK 成功不等于固件零点已生效。"""
        if self.control is None or not self.connected_flag:
            raise RuntimeError("XHand is not connected")
        result = {
            "hand_tactile_reset_time_ns": time.time_ns(),
            "hand_tactile_reset_result": "incomplete (0/5)",
        }
        self.tactile_reset_metadata = result
        for completed, sensor_id in enumerate(TACTILE_SENSOR_IDS):
            if cancel_requested():
                result["hand_tactile_reset_result"] = f"cancelled ({completed}/5)"
                raise CancelledError(
                    f"触觉复位请求已中断，已完成 {completed}/5 次 SDK 调用；已发送的请求不会撤销。"
                )
            result["hand_tactile_reset_result"] = (
                f"incomplete ({completed}/5), sensor=0x{sensor_id:02x}"
            )
            try:
                reply = self.control.reset_sensor(self.cfg.device_id, sensor_id)
            except Exception as exc:
                result["hand_tactile_reset_result"] = (
                    f"exception ({completed}/5), sensor=0x{sensor_id:02x}: {exc}"
                )
                raise
            code = error_code(reply)
            if code != 0:
                result["hand_tactile_reset_result"] = (
                    f"failed ({completed}/5), sensor=0x{sensor_id:02x}, code={code}"
                )
                logger.warning(
                    "触觉复位调用失败：sensor=0x%02x，code=%s，已完成 %d/5 次：%s",
                    sensor_id,
                    code,
                    completed,
                    reply.error_message,
                )
                return False
            result["hand_tactile_reset_result"] = f"sdk_completed ({completed + 1}/5)"
        logger.info("五指触觉复位请求的 SDK 调用完成（5/5），未确认固件复位或零点效果。")
        return True

    def get_tactile_reset_metadata(self) -> dict[str, object]:
        return {
            "hand_tactile_reset_result": "not_requested",
            **self.tactile_reset_metadata,
        }

    def get_state(self) -> XHandState | None:
        if self.control is None or not self.connected_flag:
            raise RuntimeError("XHand is not connected")
        error, raw_state = self.control.read_state(self.cfg.device_id, True)
        code = error_code(error)
        if code == COMMUNICATION_CRC_ERROR_CODE:
            # CRC 失败的载荷不可使用，即使数值看似正常。
            return None
        if code not in NONFATAL_SDK_CODES:
            raise XHandError("read", code, error.error_message)
        if raw_state is None:
            raise XHandError("read", code, "missing state payload")

        try:
            qpos, current, board_errors = parse_joints(raw_state)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise XHandError("read", code, f"invalid joint payload: {exc}") from exc

        # SDK 汇总状态可能掩盖其他板卡错误，须检查逐板状态。
        if any(np.any(board_errors[name]) for name in ("commboard_err", "jointboard_err")):
            raise XHandError("read", code, f"joint/communication board fault: {board_errors}")

        tactile_dense = np.full(HAND_TACTILE_FORCE_SHAPE, np.nan, dtype=np.float64)
        tactile_aggregate = np.full(HAND_TACTILE_SUM_SHAPE, np.nan, dtype=np.float64)
        aggregate_allowed, dense_allowed = tactile_validity(code, comm_type=self.cfg.comm_type)
        if not aggregate_allowed or not dense_allowed:
            logger.debug("XHand tactile read incomplete: code=%s", code)
        if aggregate_allowed:
            tactile_aggregate = parse_tactile_aggregate(raw_state)
        if dense_allowed:
            tactile_dense = parse_tactile_dense(raw_state)
        for sensor_index, finger_id in enumerate(XHAND_TACTILE_SENSOR_FINGER_IDS):
            aggregate_allowed, dense_allowed = tactile_validity(
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

    def send_action(self, action: np.ndarray) -> None:
        code = self.send_position(action)
        now_ns = time.monotonic_ns()
        if code == COMMUNICATION_CRC_ERROR_CODE:
            if self.send_crc_started_ns is None:
                self.send_crc_started_ns = now_ns
                logger.warning("XHand send CRC; delivery unconfirmed, continuing with next target")
            if now_ns - self.send_crc_started_ns >= int(self.cfg.feedback_max_age_s * 1e9):
                raise XHandError("send", code, "continuous send CRC exceeded age budget")
        else:
            if self.send_crc_started_ns is not None:
                logger.info("XHand send responses recovered")
            self.send_crc_started_ns = None

    def end_send_interval(self):
        self.send_crc_started_ns = None

    def send_position(self, action: np.ndarray) -> int:
        from dexmani_real.robot.commands import check_hand_target

        target = np.asarray(action, dtype=np.float64)
        reason = check_hand_target(
            target,
            mechanical_lower_rad=self.mechanical_lower,
            mechanical_upper_rad=self.mechanical_upper,
        )
        if reason is not None:
            raise ValueError(f"XHand.send_action: {reason}")
        if self.control is None or self.command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")
        for index, value in enumerate(target):
            joint = self.command.finger_command[index]
            joint.mode = POSITION_MODE
            joint.position = float(value)
        return self.send_command()

    def hold_current(self, qpos: np.ndarray) -> int:
        measured = np.asarray(qpos, dtype=np.float64)
        if measured.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(measured)):
            raise ValueError("XHand.hold_current requires twelve finite joint positions")
        return self.send_position(
            np.clip(measured, XHAND_RATED_QPOS_MIN_RAD, XHAND_RATED_QPOS_MAX_RAD)
        )

    def set_passive(self) -> int:
        if self.control is None or self.command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")
        for index in range(HAND_DOF):
            self.command.finger_command[index].mode = PASSIVE_MODE
        return self.send_command()

    def stop(self, qpos: np.ndarray | None = None, *, timestamp_ns: int = 0) -> None:
        """SDK 接受停止请求不代表物理停稳。"""
        if not self.is_connected:
            raise RuntimeError("XHand disconnected before stop")
        deadline = time.monotonic() + STOP_RETRY_TIMEOUT_S
        attempt = 0
        while time.monotonic() < deadline:
            now_ns = time.monotonic_ns()
            fresh = 0 < int(timestamp_ns) <= now_ns and now_ns - int(timestamp_ns) < int(
                self.cfg.feedback_max_age_s * 1e9
            )
            status = self.hold_current(qpos) if qpos is not None and fresh else self.set_passive()
            if status in NONFATAL_SDK_CODES:
                if attempt:
                    logger.info("XHand stop SDK response recovered after %d attempts", attempt + 1)
                return
            if attempt == 0:
                logger.warning("XHand stop CRC; retrying within the stop deadline")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(STOP_RETRY_INTERVAL_S, remaining))
            attempt += 1
        raise RuntimeError("XHand stop timed out: CRC replies remain unconfirmed")

    def send_command(self) -> int:
        error = self.control.send_command(self.cfg.device_id, self.command)

        code = error_code(error)
        if code == COMMUNICATION_CRC_ERROR_CODE:
            return code
        if code not in NONFATAL_SDK_CODES:
            # 1501035 表示过流，必须终止本次运行。
            raise XHandError("send", code, error.error_message)
        return code

    def make_command(self, qpos: np.ndarray) -> Any:
        command = self.sdk.HandCommand_t()
        for index in range(HAND_DOF):
            joint = command.finger_command[index]
            joint.id = index
            joint.position = float(qpos[index])
            joint.kp = int(self.cfg.kp[index])
            joint.ki = int(self.cfg.ki)
            joint.kd = int(self.cfg.kd)
            joint.tor_max = int(self.cfg.tor_max[index])
            joint.mode = POSITION_MODE
            joint.res0 = 0
            joint.res1 = 0
            joint.res2 = 0
            joint.res3 = 0
        return command
