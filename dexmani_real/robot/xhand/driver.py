"""XHand SDK 边界；过流仅告警，CRC 不代表发送成功，触觉有效性单独判断。"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.commands import (
    OperationCancelled,
    RobotError,
    hand_reference_from_feedback,
)
from dexmani_real.robot.model import (
    HAND_DOF,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    XHAND_RATED_QPOS_MAX_RAD,
    XHAND_RATED_QPOS_MIN_RAD,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
)
from dexmani_real.robot.xhand.feedback import XHandState
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)

SDK_PROTOCOL = {"ethercat": "EtherCAT", "serial": "RS485"}
CONNECT_TIMEOUT_S = 2.0
CONNECT_POLL_INTERVAL_S = 0.02
COMMUNICATION_CRC_ERROR_CODE = 1_501_070
COMBINED_FORCE_UNAVAILABLE_CODE = 1_501_018
DISTRIBUTED_FORCE_UNAVAILABLE_CODE = 1_501_019
TEMPERATURE_UNAVAILABLE_CODE = 1_501_020
OVERCURRENT_WARNING_CODE = 1_501_035
# X-Hand 协议中的关节板原始错误码；SDK 将其映射为 1501035。
OVERCURRENT_BOARD_CODE = 110

# 这些状态仍允许关节读写，触觉有效性单独判断。
NONFATAL_SDK_CODES = frozenset(
    {
        0,
        COMBINED_FORCE_UNAVAILABLE_CODE,
        DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
        TEMPERATURE_UNAVAILABLE_CODE,
        OVERCURRENT_WARNING_CODE,
    }
)

EC_STATE_INIT = 1
POST_EC_DISCONNECT_S = 2.0
TACTILE_SENSOR_IDS = (0x11, 0x12, 0x13, 0x14, 0x15)
PASSIVE_MODE = 0
POSITION_MODE = 3
STOP_RETRY_INTERVAL_S = 0.01
STOP_RETRY_TIMEOUT_S = 0.25
STOP_FEEDBACK_MAX_AGE_S = 4 / 30
MAX_CONSECUTIVE_WRITE_CRC = 5


def _tactile_validity(code: int, *, comm_type: str) -> tuple[bool, bool]:
    """过流不禁用触觉；其余状态按通信协议分别判断。"""
    # 电机过流本身不使触觉失效；仍检查各指尖状态和数据完整性。
    if code == OVERCURRENT_WARNING_CODE:
        return (True, True)
    if comm_type == "ethercat":
        return (code == 0, code == 0)
    if code == 0:
        return (True, True)
    if code in (COMBINED_FORCE_UNAVAILABLE_CODE, 200):
        return (False, False)
    if code in (DISTRIBUTED_FORCE_UNAVAILABLE_CODE, 201):
        return (True, False)
    if code in (TEMPERATURE_UNAVAILABLE_CODE, 202):
        return (True, True)
    return (False, False)


def _parse_joints(
    state: Any,
) -> tuple[np.ndarray, dict[int, Any]]:
    qpos = np.full(HAND_JOINT_SHAPE, np.nan, dtype=np.float64)
    joints = {}
    for joint in state.finger_state:
        index = int(joint.id)
        if index < 0 or index >= HAND_DOF or index in joints:
            raise ValueError(f"invalid or duplicate joint id {index}")
        joints[index] = joint
        qpos[index] = float(joint.position)
        comm_error = int(joint.commboard_err)
        # SDK 字段名实际拼写为 jonitboard_err。
        joint_error = int(joint.jonitboard_err)
        if comm_error or joint_error not in (0, OVERCURRENT_BOARD_CODE):
            raise ValueError(
                f"joint/communication board fault at joint[{index}]: "
                f"commboard_err={comm_error}, jointboard_err={joint_error}"
            )
    if len(joints) != HAND_DOF:
        raise ValueError(f"{len(joints)}/{HAND_DOF} joints reported")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("non-finite joint position feedback")
    return qpos, joints


def _parse_tactile_aggregate(state: Any) -> np.ndarray:
    force_aggregate = np.asarray(
        [
            [sensor.calc_force.fx, sensor.calc_force.fy, sensor.calc_force.fz]
            for sensor in state.sensor_data
        ],
        dtype=np.float32,
    )
    if force_aggregate.shape != HAND_TACTILE_SUM_SHAPE:
        raise ValueError("aggregate tactile must have shape [5,3]")
    if not np.isfinite(force_aggregate).all():
        raise ValueError("aggregate tactile must be finite float32")
    return force_aggregate


def _parse_tactile_dense(state: Any) -> np.ndarray:
    tactile_dense = np.asarray(
        [
            [[force.fx, force.fy, force.fz] for force in sensor.raw_force]
            for sensor in state.sensor_data
        ],
        dtype=np.float32,
    )
    if tactile_dense.shape != HAND_TACTILE_FORCE_SHAPE:
        raise ValueError("dense tactile must have shape [5,120,3]")
    if not np.isfinite(tactile_dense).all():
        raise ValueError("dense tactile must be finite float32")
    return tactile_dense


class XHandError(RobotError):
    def __init__(self, operation: str, code: int, message: str) -> None:
        self.operation = str(operation)
        self.code = int(code)
        self.message = str(message)
        super().__init__(f"XHand {self.operation} failed: code={self.code} msg={self.message}")


class XHand:
    def __init__(self, config: HandConfig):
        self.cfg = config
        self.connected_flag = False
        self.consecutive_write_crc = 0
        self.next_warning_ns: dict[str, int] = {}
        self.control: Any = None
        self.command: Any = None
        self.tactile_reset_metadata: dict[str, object] = {}

    @property
    def is_connected(self) -> bool:
        return self.connected_flag

    def connect(self, *, cancel_requested=lambda: False) -> XHandState:
        """设备发现与首帧读取共用期限；命令与上层缓存使用同一份实测状态。"""
        if self.control is not None:
            raise RuntimeError("XHand still owns a device handle; disconnect the old connection before reconnecting")
        from xhand_controller import xhand_control

        self.sdk = xhand_control
        try:
            if cancel_requested():
                raise OperationCancelled("XHand connection cancelled")
            self.command = None
            self.consecutive_write_crc = 0
            self.next_warning_ns.clear()
            self.tactile_reset_metadata = {}
            self.control = self.sdk.XHandControl()
            self.open_device()
            self.connected_flag = True
            deadline_ns = time.monotonic_ns() + int(CONNECT_TIMEOUT_S * 1e9)
            hand_ids = []
            while time.monotonic_ns() < deadline_ns:
                if cancel_requested():
                    raise OperationCancelled("XHand connection cancelled")
                # SDK 的串口读取线程异步发现设备；这里只查询已发现的 ID。
                hand_ids = list(self.control.list_hands_id())
                sample = self.get_state() if self.cfg.device_id in hand_ids else None
                if cancel_requested():
                    raise OperationCancelled("XHand connection cancelled")
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                if sample is not None:
                    self.command = self.make_command(sample.qpos)
                    return sample
                time.sleep(min(CONNECT_POLL_INTERVAL_S, remaining_ns / 1e9))
            raise XHandError(
                "connect", -1,
                f"initial XHand feedback timed out; device_id={self.cfg.device_id}, "
                f"discovered={hand_ids}",
            )
        except BaseException as failure:
            try:
                self.disconnect()
            except BaseException as cleanup:
                if isinstance(cleanup, KeyboardInterrupt) or isinstance(failure, OperationCancelled):
                    raise
                logger.exception("XHand connection cleanup failed")
            raise

    def open_device(self) -> None:
        device_name = self.cfg.device_name
        if device_name is None:
            devices = self.control.enumerate_devices(SDK_PROTOCOL[self.cfg.comm_type])
            if not devices:
                raise XHandError("connect", -2, f"no device found for {self.cfg.comm_type}")
            device_name = devices[0]

        if self.cfg.comm_type == "serial":
            error = self.control.open_serial(device_name, self.cfg.baudrate)
        else:
            error = self.control.open_ethercat(device_name)
        if error.error_code == OVERCURRENT_WARNING_CODE:
            self.warn_overcurrent("connect", error.error_message)
        elif error.error_code != 0:
            raise XHandError("connect", error.error_code, error.error_message)

    def disconnect(self) -> None:
        connected_ethercat = self.connected_flag and self.cfg.comm_type == "ethercat"
        errors = []
        try:
            if self.control is not None:
                if connected_ethercat:
                    try:
                        self.request_ethercat_init()
                    except BaseException as exc:
                        errors.append(exc)
                try:
                    self.control.close_device()
                    self.control = None
                except BaseException as exc:
                    errors.append(exc)
                if connected_ethercat:
                    try:
                        time.sleep(POST_EC_DISCONNECT_S)
                    except BaseException as exc:
                        errors.append(exc)
        finally:
            self.connected_flag = False
        if errors:
            failure = next((error for error in errors if isinstance(error, KeyboardInterrupt)), errors[0])
            for error in errors:
                if error is not failure:
                    logger.error("additional XHand disconnect failure", exc_info=(type(error), error, error.__traceback__))
            raise failure

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
        if int(error.error_code) == OVERCURRENT_WARNING_CODE:
            self.warn_overcurrent("ethercat INIT", error.error_message)
        elif int(error.error_code) != 0:
            raise XHandError("ethercat INIT", int(error.error_code), "INIT request failed")
        time.sleep(0.2)

    def reset_tactile_sensors(self, *, cancel_requested=lambda: False) -> dict[int, int]:
        """一次五指请求；返回码不证明物理零点效果，操作者可显式重试。"""
        if self.control is None or not self.connected_flag:
            raise RobotError("XHand is not connected")
        results: dict[int, int] = {}
        for sensor_id in TACTILE_SENSOR_IDS:
            if cancel_requested():
                raise OperationCancelled(
                    f"触觉复位请求已中断，已完成 {len(results)}/5 次 SDK 调用；已发送的请求不会撤销。"
                )
            if not results:
                self.tactile_reset_metadata = {
                    "hand_tactile_reset_time_ns": time.time_ns(),
                    "hand_tactile_reset_result": results,
                }
            reply = self.control.reset_sensor(self.cfg.device_id, sensor_id)
            code = int(reply.error_code)
            results[sensor_id] = code
            if code == OVERCURRENT_WARNING_CODE:
                self.warn_overcurrent(f"tactile reset sensor=0x{sensor_id:02x}", reply.error_message)
            elif code == COMMUNICATION_CRC_ERROR_CODE or code in NONFATAL_SDK_CODES - {0}:
                logger.warning("XHand tactile reset sensor=0x%02x unconfirmed/unavailable: code=%d msg=%s",
                               sensor_id, code, reply.error_message)
            elif code != 0:
                raise XHandError(f"tactile reset sensor=0x{sensor_id:02x}", code, reply.error_message)
        logger.info("五指触觉复位请求结果：%s；未确认物理零点效果。", results)
        return results

    def get_state(self) -> XHandState | None:
        """读 CRC 无可信新关节；触觉通道无完整有效新帧时对应字段为 None。"""
        if self.control is None or not self.connected_flag:
            raise RobotError("XHand is not connected")
        query_started_ns = time.monotonic_ns()
        # 本机 SDK 1.5.2 没有缓存帧序号/来源时间，保留主动读取以免旧缓存续期。
        error, raw_state = self.control.read_state(self.cfg.device_id, True)
        code = int(error.error_code)
        if code == COMMUNICATION_CRC_ERROR_CODE:
            self.warn_limited("read_crc", "XHand read CRC; payload discarded")
            return None
        if code not in NONFATAL_SDK_CODES:
            raise XHandError("read", code, error.error_message)
        if raw_state is None:
            raise XHandError("read", code, "missing state payload")

        try:
            qpos, joints = _parse_joints(raw_state)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise XHandError("read", code, f"invalid joint payload: {exc}") from exc

        overcurrent_joints = [
            index for index, joint in joints.items()
            if int(joint.jonitboard_err) == OVERCURRENT_BOARD_CODE
        ]
        if code == OVERCURRENT_WARNING_CODE or overcurrent_joints:
            self.warn_overcurrent("read", f"{error.error_message}; joints={overcurrent_joints}")

        channels = ("aggregate", "dense")
        unavailable = {
            name: f"SDK code={code}"
            for name, allowed in zip(channels, _tactile_validity(code, comm_type=self.cfg.comm_type))
            if not allowed
        }
        for finger_id in XHAND_TACTILE_SENSOR_FINGER_IDS:
            if len(unavailable) == len(channels):
                break
            try:
                tip_code = int(joints[finger_id].tipboard_err)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                for name in channels:
                    unavailable.setdefault(name, f"invalid tip[{finger_id}] status ({type(exc).__name__})")
                break
            for name, allowed in zip(channels, _tactile_validity(tip_code, comm_type=self.cfg.comm_type)):
                if not allowed:
                    unavailable.setdefault(name, f"tip[{finger_id}] code={tip_code}")

        tactile = {}
        for name, parser in (("aggregate", _parse_tactile_aggregate), ("dense", _parse_tactile_dense)):
            values = None
            reason = unavailable.get(name)
            if reason is None:
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        values = parser(raw_state)
                except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                    reason = f"invalid payload ({type(exc).__name__})"
            if values is None:
                self.warn_limited(name, "XHand tactile %s unavailable: %s", name, reason)
            tactile[name] = values

        return XHandState(
            qpos=qpos,
            tactile_aggregate=tactile["aggregate"],
            tactile_dense=tactile["dense"],
            timestamp_ns=query_started_ns,
        )

    def _send_action(self, target: np.ndarray) -> None:
        """目标已由 Robot 的组合预检验证；连续写 CRC 仍会失败。"""
        confirmed = self._send_position(target) != COMMUNICATION_CRC_ERROR_CODE
        if not confirmed and self.consecutive_write_crc >= MAX_CONSECUTIVE_WRITE_CRC:
            raise XHandError(
                "send", COMMUNICATION_CRC_ERROR_CODE,
                f"{self.consecutive_write_crc} consecutive position write CRC replies; delivery unconfirmed",
            )

    def _send_position(self, target: np.ndarray) -> int:
        if self.control is None or self.command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")
        for index, value in enumerate(target):
            joint = self.command.finger_command[index]
            joint.mode = POSITION_MODE
            joint.position = float(value)
        return self.send_command()

    def set_passive(self) -> int:
        if self.control is None or self.command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")
        for index in range(HAND_DOF):
            self.command.finger_command[index].mode = PASSIVE_MODE
        return self.send_command()

    def stop(self, qpos: np.ndarray | None = None, *, timestamp_ns: int = 0) -> None:
        """SDK 接受停止请求不代表物理停稳。"""
        if self.command is None:
            return  # 命令初始化前没有可发送的运动；部分启动只需关闭连接。
        if not self.is_connected:
            raise RuntimeError("XHand disconnected before stop")
        reference = None
        if qpos is not None:
            try:
                reference = hand_reference_from_feedback(
                    qpos, XHAND_RATED_QPOS_MIN_RAD, XHAND_RATED_QPOS_MAX_RAD,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                logger.warning("XHand stop uses passive: invalid position feedback (%s)", exc)
        deadline = time.monotonic() + STOP_RETRY_TIMEOUT_S
        while time.monotonic() < deadline:
            now_ns = time.monotonic_ns()
            fresh = sample_is_fresh(timestamp_ns, STOP_FEEDBACK_MAX_AGE_S, now_ns)
            if reference is not None and not fresh:
                logger.warning("XHand stop uses passive: position feedback stale")
                reference = None
            status = self._send_position(reference) if reference is not None else self.set_passive()
            if status != COMMUNICATION_CRC_ERROR_CODE:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(STOP_RETRY_INTERVAL_S, remaining))
        raise XHandError("stop", COMMUNICATION_CRC_ERROR_CODE,
                         "stop timed out: CRC replies remain unconfirmed")

    def send_command(self) -> int:
        error = self.control.send_command(self.cfg.device_id, self.command)
        code = int(error.error_code)
        if code not in NONFATAL_SDK_CODES and code != COMMUNICATION_CRC_ERROR_CODE:
            raise XHandError("send", code, error.error_message)
        if code == COMMUNICATION_CRC_ERROR_CODE:
            self.consecutive_write_crc += 1
            self.warn_limited("write_crc", "XHand write CRC; target delivery unconfirmed")
        else:
            self.consecutive_write_crc = 0
        if code == OVERCURRENT_WARNING_CODE:
            self.warn_overcurrent("send", error.error_message)
        return code

    def warn_limited(self, key: str, message: str, *args) -> None:
        now_ns = time.monotonic_ns()
        if now_ns >= self.next_warning_ns.get(key, 0):
            logger.warning(message, *args)
            self.next_warning_ns[key] = now_ns + 1_000_000_000

    def warn_overcurrent(self, operation: str, message: str) -> None:
        self.warn_limited("overcurrent", "XHand 过流告警（继续运行）：%s code=%d msg=%s",
                          operation, OVERCURRENT_WARNING_CODE, message)

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
