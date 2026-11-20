"""XHand SDK 读写、命令回执及触觉处理。"""

from __future__ import annotations

import logging
import os
import stat
import time
from operator import index as integer_index
from typing import Any

import numpy as np

from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.commands import (
    CommandExpired,
    HandCommandUnconfirmed,
    OperationCancelled,
    RobotError,
    validate_hand_qpos,
    validate_hand_target,
)
from dexmani_real.robot.model import (
    HAND_DOF,
    HAND_FINGER_COUNT,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    TACTILE_AXIS_COUNT,
    XHAND_RATED_QPOS_MAX_RAD,
    XHAND_RATED_QPOS_MIN_RAD,
    XHAND_SDK_JOINT_NAMES,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
)
from dexmani_real.robot.xhand.feedback import TACTILE_CHANNELS, XHandState
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger(__name__)

SDK_PROTOCOL = {"ethercat": "EtherCAT", "serial": "RS485"}
CONNECT_TIMEOUT_S = 2.0
CONNECT_POLL_INTERVAL_S = 0.02
COMMUNICATION_CRC_ERROR_CODE = 1_501_070
COMBINED_FORCE_UNAVAILABLE_CODE = 1_501_018
DISTRIBUTED_FORCE_UNAVAILABLE_CODE = 1_501_019
TEMPERATURE_UNAVAILABLE_CODE = 1_501_020
# SDK error_manager.hpp 与 XHand 手册中的指尖板错误码。
TACTILE_BOARD_CODES = {
    COMBINED_FORCE_UNAVAILABLE_CODE: 200,
    DISTRIBUTED_FORCE_UNAVAILABLE_CODE: 201,
    TEMPERATURE_UNAVAILABLE_CODE: 202,
    1_501_021: 203,
    1_501_022: 205,
    1_501_023: 206,
    1_501_024: 207,
}
OVERCURRENT_WARNING_CODE = 1_501_035
# X-Hand 协议中的关节板原始错误码；SDK 将其映射为 1501035。
OVERCURRENT_BOARD_CODE = 110

EC_STATE_INIT = 1
POST_EC_DISCONNECT_S = 2.0
TACTILE_SENSOR_IDS = (0x11, 0x12, 0x13, 0x14, 0x15)
TACTILE_CALIBRATION_INTERVAL_NS = 20_000_000
PASSIVE_MODE = 0
POSITION_MODE = 3
RELEASE_TIMEOUT_S = 0.25
HOLD_FEEDBACK_MAX_AGE_S = 4 / 30


def _valid_tactile_modalities(code: int) -> tuple[bool, bool]:
    """根据指尖板状态判断合力和分布力是否可用。"""
    if code in (0, OVERCURRENT_WARNING_CODE, 202):
        return True, True
    if code == 200:
        return False, True
    if code == 201:
        return True, False
    return False, False


def _parse_joints(
    state: Any,
) -> tuple[np.ndarray, dict[int, Any], np.ndarray]:
    qpos = np.full(HAND_JOINT_SHAPE, np.nan, dtype=np.float64)
    joints = {}
    overcurrent_joint_mask = np.zeros(HAND_JOINT_SHAPE, dtype=bool)
    for joint in state.finger_state:
        index = integer_index(joint.id)
        if not 0 <= index < HAND_DOF:
            raise ValueError(f"invalid XHand SDK joint id {index}")
        if index in joints:
            raise ValueError(f"duplicate XHand SDK joint id {index} ({XHAND_SDK_JOINT_NAMES[index]})")
        joints[index] = joint
        qpos[index] = float(joint.position)
        comm_error = int(joint.commboard_err)
        # SDK 字段名实际拼写为 jonitboard_err。
        joint_error = int(joint.jonitboard_err)
        if comm_error or joint_error not in (0, OVERCURRENT_BOARD_CODE):
            raise ValueError(
                f"joint/communication board fault at {XHAND_SDK_JOINT_NAMES[index]} (SDK id {index}): "
                f"commboard_err={comm_error}, jointboard_err={joint_error}"
            )
        if joint_error == OVERCURRENT_BOARD_CODE:
            overcurrent_joint_mask[index] = True
    if len(joints) != HAND_DOF:
        missing = [name for i, name in enumerate(XHAND_SDK_JOINT_NAMES) if i not in joints]
        raise ValueError(f"missing XHand SDK joints: {', '.join(missing)}")
    return validate_hand_qpos(qpos), joints, overcurrent_joint_mask


def _parse_tactile(sensors, allowed: np.ndarray, *, dense: bool) -> tuple[np.ndarray | None, np.ndarray]:
    shape = HAND_TACTILE_FORCE_SHAPE if dense else HAND_TACTILE_SUM_SHAPE
    expected = shape[1:] if dense else (1, TACTILE_AXIS_COUNT)
    values = np.full(shape, np.nan, dtype=np.float32)
    valid = np.zeros(HAND_FINGER_COUNT, dtype=bool)
    for index in np.flatnonzero(allowed):
        try:
            sensor = sensors[index]
            points = sensor.raw_force if dense else (sensor.calc_force,)
            with np.errstate(over="ignore", invalid="ignore"):
                xyz = np.asarray([(p.fx, p.fy, p.fz) for p in points], dtype=np.float32)
            if xyz.shape == expected and np.isfinite(xyz).all():
                values[index] = xyz if dense else xyz[0]
                valid[index] = True
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            # 只丢弃该指的该种触觉载荷。
            continue
    return (values if valid.any() else None), valid


def _within_tactile_limits(channels, limits):
    return all(np.max(np.linalg.norm(values, axis=-1)) <= limit
               for values, limit in zip(channels, limits))


class XHandError(RobotError):
    def __init__(self, operation: str, code: int, message: str) -> None:
        self.operation = str(operation)
        self.code = int(code)
        self.message = str(message)
        super().__init__(f"XHand {self.operation} failed: code={self.code} msg={self.message}")


class XHand:
    def __init__(self, config: HandConfig, *, tactile_channels: tuple[str, ...] = TACTILE_CHANNELS):
        self.cfg = config
        self.tactile_channels = tactile_channels
        self.connected_flag = False
        self.consecutive_read_crc = 0
        self.consecutive_write_crc = 0
        self.crc_counts = {
            "read_errors": 0,
            "write_errors": 0,
            "read_recoveries": 0,
            "write_recoveries": 0,
            "unconfirmed_writes": 0,
        }
        self.next_warning_ns: dict[str, int] = {}
        self._overcurrent_warned = False
        self.control: Any = None
        self.command: Any = None
        self.tactile_reset_metadata: dict[str, object] = {}
        self._tactile_bias: tuple[np.ndarray, np.ndarray] | None = None
        self._tactile_bias_time_ns: int | None = None
        self._tactile_bias_applied = False
        self._tactile_calibration_issue: str | None = "not calibrated"

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
            self.consecutive_read_crc = 0
            self.consecutive_write_crc = 0
            self.next_warning_ns.clear()
            self._overcurrent_warned = False
            self.tactile_reset_metadata = {}
            self._tactile_bias = self._tactile_bias_time_ns = None
            self._tactile_bias_applied = False
            self._tactile_calibration_issue = "not calibrated"
            self.control = self.sdk.XHandControl()
            device_name = self.open_device()
            self.connected_flag = True
            deadline_ns = time.monotonic_ns() + int(CONNECT_TIMEOUT_S * 1e9)
            hand_ids = []
            while time.monotonic_ns() < deadline_ns:
                if cancel_requested():
                    raise OperationCancelled("XHand connection cancelled")
                # SDK 的串口读取线程异步发现设备；这里只查询已发现的 ID。
                hand_ids = list(self.control.list_hands_id())
                sample = self.get_state(deadline_ns=deadline_ns) if self.cfg.device_id in hand_ids else None
                if cancel_requested():
                    raise OperationCancelled("XHand connection cancelled")
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                if sample is not None:
                    self.command = self._make_command(sample.qpos)
                    logger.info("XHand connected via %s on %s; device_id=%d, discovered=%s",
                                self.cfg.comm_type, device_name, self.cfg.device_id, hand_ids)
                    return sample
                time.sleep(min(CONNECT_POLL_INTERVAL_S, remaining_ns / 1e9))
            raise XHandError(
                "connect", -1,
                f"initial XHand feedback timed out on {device_name}; device_id={self.cfg.device_id}, "
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

    def open_device(self) -> str:
        device_name = self.cfg.device_name
        if device_name is None:
            devices = self.control.enumerate_devices(SDK_PROTOCOL[self.cfg.comm_type])
            if not devices:
                raise XHandError("connect", -2, f"no device found for {self.cfg.comm_type}")
            if self.cfg.comm_type == "serial" and len(devices) > 1:
                raise XHandError("connect", -2, f"multiple serial ports found: {devices}; set hand.device_name")
            device_name = devices[0]

        if self.cfg.comm_type == "serial":
            if not stat.S_ISCHR(os.stat(device_name).st_mode):
                raise ValueError(f"serial port {device_name} is not a character device")
            if not os.access(device_name, os.R_OK | os.W_OK):
                raise PermissionError(f"serial port {device_name} is not readable and writable by the current user")
            error = self.control.open_serial(device_name, self.cfg.baudrate)
        else:
            error = self.control.open_ethercat(device_name)
        if error.error_code == OVERCURRENT_WARNING_CODE:
            self.warn_overcurrent("connect", error.error_message)
        elif error.error_code != 0:
            raise XHandError("connect", error.error_code, f"{device_name}: {error.error_message}")
        return device_name

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
                # 复位可能改变硬件零点，即使回执未确认也保留旧 Bias 仅供追溯。
                self._tactile_bias_applied = False
                self._tactile_calibration_issue = "SDK reset without verified calibration"
                self.tactile_reset_metadata = {
                    "hand_tactile_reset_time_ns": time.time_ns(),
                    "hand_tactile_reset_result": results,
                }
            reply = self.control.reset_sensor(self.cfg.device_id, sensor_id)
            code = int(reply.error_code)
            results[sensor_id] = code
            if code == OVERCURRENT_WARNING_CODE:
                self.warn_overcurrent(f"tactile reset sensor=0x{sensor_id:02x}", reply.error_message)
            elif (code == COMMUNICATION_CRC_ERROR_CODE
                  or TACTILE_BOARD_CODES.get(code, code) in TACTILE_BOARD_CODES.values()):
                logger.warning("XHand tactile reset sensor=0x%02x unconfirmed/unavailable: code=%d msg=%s",
                               sensor_id, code, reply.error_message)
            elif code != 0:
                raise XHandError(f"tactile reset sensor=0x{sensor_id:02x}", code, reply.error_message)
        logger.info("五指触觉复位请求结果：%s；未确认物理零点效果。", results)
        return results

    @property
    def tactile_calibration_metadata(self) -> dict[str, object]:
        bias = self._tactile_bias
        return {
            "calibrated": self._tactile_calibration_issue is None,
            "issue": self._tactile_calibration_issue,
            "bias_applied": self._tactile_bias_applied,
            "bias_time_ns": self._tactile_bias_time_ns,
            "bias_sdk": None if bias is None else {
                name: values.tolist() for name, values in zip(TACTILE_CHANNELS, bias)
            },
        }

    def _calibration_sample(self, deadline_ns, cancel_requested):
        while time.monotonic_ns() < deadline_ns:
            if cancel_requested():
                raise OperationCancelled("XHand tactile calibration cancelled")
            state = self._read_state(deadline_ns=deadline_ns, tactile_channels=TACTILE_CHANNELS)
            if cancel_requested():
                raise OperationCancelled("XHand tactile calibration cancelled")
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                break
            if state is not None and sample_is_fresh(state.timestamp_ns, self.cfg.tactile_max_age_s, now_ns):
                channels = (state.tactile_aggregate, state.tactile_dense)
                if all(values is not None and np.isfinite(values).all() for values in channels):
                    return channels
            time.sleep(min(TACTILE_CALIBRATION_INTERVAL_NS, deadline_ns - now_ns) / 1e9)
        return None

    def calibrate_tactile(self, *, no_contact: bool, cancel_requested=lambda: False) -> bool:
        """在操作者确认无接触后复位五指，原始 SDK 样本以均值校正。"""
        def fail(issue):
            self._tactile_calibration_issue = issue
            logger.warning("XHand tactile calibration unsuccessful: %s", issue)
            return False

        self._tactile_calibration_issue = "calibration incomplete"
        if no_contact is not True:
            return fail("operator must confirm no contact")
        timeout_ns = int(self.cfg.tactile_calibration_timeout_s * 1e9)
        sample_count = self.cfg.tactile_calibration_samples + 1  # 含独立验证样本。
        min_timeout_ns = sample_count * TACTILE_CALIBRATION_INTERVAL_NS
        if timeout_ns <= min_timeout_ns:
            return fail(f"post-reset sampling requires a timeout greater than {min_timeout_ns / 1e6:g} ms")
        baseline = self._calibration_sample(time.monotonic_ns() + timeout_ns, cancel_requested)
        if baseline is None:
            return fail("missing fresh pre-reset tactile feedback")
        bias = self._tactile_bias if self._tactile_bias_applied else None
        if bias is not None:
            baseline = tuple(values - offset for values, offset in zip(baseline, bias))
        if not _within_tactile_limits(baseline, self.cfg.tactile_contact_limits):
            return fail("pre-reset contact limit exceeded")

        results = self.reset_tactile_sensors(cancel_requested=cancel_requested)
        if any(results.values()):
            return fail("one or more sensor resets unconfirmed")

        deadline_ns = time.monotonic_ns() + timeout_ns
        samples = []
        # 拟合与残差验证分别用不同的主动查询。
        for _ in range(sample_count):
            time.sleep(TACTILE_CALIBRATION_INTERVAL_NS / 1e9)
            sample = self._calibration_sample(deadline_ns, cancel_requested)
            if sample is None:
                return fail("insufficient fresh calibration or verification samples")
            if not _within_tactile_limits(sample, self.cfg.tactile_contact_limits):
                return fail("post-reset contact limit exceeded")
            samples.append(sample)

        bias = tuple(np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
                     for values in zip(*samples[:-1]))
        for sample in samples:
            if not _within_tactile_limits(
                (values - offset for values, offset in zip(sample, bias)), self.cfg.tactile_residual_limits,
            ):
                return fail("calibration residual limit exceeded")
        self._tactile_bias = bias
        self._tactile_bias_time_ns = time.time_ns()
        self._tactile_bias_applied = True
        self._tactile_calibration_issue = None
        logger.info("XHand tactile calibration accepted (%d fit samples)", len(samples) - 1)
        return True

    def get_state(self, *, deadline_ns: int | None = None, retry_crc: bool = True) -> XHandState | None:
        """读关节及所需触觉，有已核验的 Bias 时应用校正。"""
        state = self._read_state(
            deadline_ns=deadline_ns, tactile_channels=self.tactile_channels, retry_crc=retry_crc,
        )
        bias = self._tactile_bias if self._tactile_bias_applied else None
        if state is not None and bias is not None:
            for name, offset in zip(TACTILE_CHANNELS, bias):
                field = f"tactile_{name}"
                values = getattr(state, field)
                if values is not None:
                    setattr(state, field, values - offset)
        return state

    def _read_state(
        self, *, deadline_ns: int | None, tactile_channels: tuple[str, ...], retry_crc: bool = True,
    ) -> XHandState | None:
        if self.control is None or not self.connected_flag:
            raise RobotError("XHand is not connected")
        retry_deadline_ns = time.monotonic_ns() + int(self.cfg.feedback_max_age_s * 1e9)
        if deadline_ns is not None:
            retry_deadline_ns = min(retry_deadline_ns, deadline_ns)
        interval_s = self.cfg.crc_retry_interval_s
        retries = self.cfg.crc_retries if retry_crc else 0
        for attempt in range(retries + 1):
            if attempt:
                if time.monotonic_ns() + round(interval_s * 1e9) >= retry_deadline_ns:
                    return None
                time.sleep(interval_s)
            # SDK 1.5.2 无硬件采样时间，记录实际发起每次主动查询的时刻。
            query_started_ns = time.monotonic_ns()
            if query_started_ns >= retry_deadline_ns:
                return None
            error, raw_state = self.control.read_state(self.cfg.device_id, True)
            code = int(error.error_code)
            if code != COMMUNICATION_CRC_ERROR_CODE:
                break
            self.consecutive_read_crc += 1
            self.crc_counts["read_errors"] += 1
            self.warn_limited("read_crc", "XHand read CRC; payload discarded")
        else:
            return None
        summary_code = TACTILE_BOARD_CODES.get(code, code)
        if summary_code not in (0, OVERCURRENT_WARNING_CODE) and summary_code not in TACTILE_BOARD_CODES.values():
            raise XHandError("read", code, error.error_message)
        if raw_state is None:
            raise XHandError("read", code, "missing state payload")

        try:
            qpos, joints, overcurrent_joint_mask = _parse_joints(raw_state)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise XHandError("read", code, f"invalid joint payload: {exc}") from exc
        if time.monotonic_ns() >= retry_deadline_ns:
            return None
        if self.consecutive_read_crc:
            self.crc_counts["read_recoveries"] += 1
            self.consecutive_read_crc = 0

        warned = code == OVERCURRENT_WARNING_CODE or bool(overcurrent_joint_mask.any())
        if warned:
            overcurrent_joints = np.flatnonzero(overcurrent_joint_mask).tolist()
            self.warn_overcurrent("read", f"{error.error_message}; joints={overcurrent_joints}")

        state = XHandState(qpos, None, None, query_started_ns, warned, overcurrent_joint_mask)
        if not tactile_channels:
            return state

        tip_codes = []
        for finger_id in XHAND_TACTILE_SENSOR_FINGER_IDS:
            try:
                tip_code = int(joints[finger_id].tipboard_err)
            except (AttributeError, TypeError, ValueError, OverflowError):
                tip_code = -1
            tip_codes.append(TACTILE_BOARD_CODES.get(tip_code, tip_code))
        allowed = np.array([_valid_tactile_modalities(tip_code) for tip_code in tip_codes])
        # 汇总告警已有明确的坏指时，保留其余手指。无逐指定位时废弃受影响的模态。
        if summary_code not in tip_codes:
            allowed &= _valid_tactile_modalities(summary_code)
        sensors = getattr(raw_state, "sensor_data", ())
        for index, name in enumerate(TACTILE_CHANNELS):
            if name not in tactile_channels:
                continue
            values, valid = _parse_tactile(sensors, allowed[:, index], dense=name == "dense")
            if not valid.all():
                self.warn_limited(name, "XHand tactile %s unavailable for finger indices %s (SDK code=%s)",
                                  name, np.flatnonzero(~valid).tolist(), code)
            setattr(state, f"tactile_{name}", values)
        return state

    def send_action(self, target: np.ndarray, *, deadline_ns: int, check_abort) -> None:
        """提交位置目标，按期返回的过流告警只反映设备状态。"""
        self.send_position(
            target, tor_max=self.cfg.tor_max, deadline_ns=deadline_ns, check_abort=check_abort,
        )

    def _send_with_crc_retry(self, prepare, *, deadline_ns: int, check_abort=None) -> int:
        """仅重试 CRC 回执；成功和过流回执都须在期内收到。"""
        interval_s = self.cfg.crc_retry_interval_s
        code = None
        try:
            for attempt in range(self.cfg.crc_retries + 1):
                if attempt:
                    if time.monotonic_ns() + round(interval_s * 1e9) >= deadline_ns:
                        break
                    if check_abort is not None:
                        check_abort()
                    time.sleep(interval_s)
                if check_abort is not None:
                    check_abort()
                if time.monotonic_ns() >= deadline_ns:
                    break
                if not attempt:
                    prepare()
                    if time.monotonic_ns() >= deadline_ns:
                        break
                code = self._send_command()
                if time.monotonic_ns() >= deadline_ns:
                    raise HandCommandUnconfirmed(f"XHand SDK reply arrived after the submission deadline (code={code})")
                if code != COMMUNICATION_CRC_ERROR_CODE:
                    return code
            if code is None:
                raise CommandExpired("XHand command expired before SDK submission")
            raise HandCommandUnconfirmed("XHand command delivery unconfirmed after write CRC replies")
        finally:
            if code == COMMUNICATION_CRC_ERROR_CODE:
                self.crc_counts["unconfirmed_writes"] += 1

    def send_position(
        self, target: np.ndarray, *, tor_max: tuple[int, ...], deadline_ns: int, check_abort=None,
    ) -> int:
        """受限重试同一位置命令，保留 SDK 成功码或过流告警码。"""
        if self.control is None or self.command is None or not self.connected_flag:
            raise RuntimeError("XHand command path is not initialized")
        # 普通运动和 HOME 由 Robot 校验操作限位，实测位置保持使用额定限位。
        target = validate_hand_target(target, XHAND_RATED_QPOS_MIN_RAD, XHAND_RATED_QPOS_MAX_RAD)

        def prepare():
            for index, value in enumerate(target):
                joint = self.command.finger_command[index]
                joint.tor_max = int(tor_max[index])
                joint.mode = POSITION_MODE
                joint.position = float(value)

        return self._send_with_crc_retry(prepare, deadline_ns=deadline_ns, check_abort=check_abort)

    @property
    def hold_feedback_max_age_s(self) -> float:
        return min(self.cfg.feedback_max_age_s, HOLD_FEEDBACK_MAX_AGE_S)

    def hold(
        self, qpos: np.ndarray, *, timestamp_ns: int, tor_max: tuple[int, ...], check_abort=None,
    ) -> bool:
        """用新鲜反馈提交额定范围内的保持目标。"""
        max_age_s = self.hold_feedback_max_age_s
        if not sample_is_fresh(timestamp_ns, max_age_s, time.monotonic_ns()):
            logger.warning("XHand hold skipped: position feedback unavailable/stale")
            return False
        try:
            feedback = validate_hand_qpos(qpos)
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning("XHand hold skipped: invalid position feedback (%s)", exc)
            return False
        target = np.clip(feedback, XHAND_RATED_QPOS_MIN_RAD, XHAND_RATED_QPOS_MAX_RAD)
        self.send_position(
            target, tor_max=tor_max,
            deadline_ns=timestamp_ns + int(max_age_s * 1e9), check_abort=check_abort,
        )
        return True

    def release(self) -> bool:
        """显式提交 Passive，返回 SDK 是否确认提交。"""
        confirmed = False
        try:
            if self.control is None or self.command is None or not self.connected_flag:
                raise RuntimeError("XHand command path is not initialized")

            def prepare():
                for joint in self.command.finger_command:
                    joint.mode = PASSIVE_MODE

            confirmed = self._send_with_crc_retry(
                prepare, deadline_ns=time.monotonic_ns() + int(RELEASE_TIMEOUT_S * 1e9),
            ) == 0
            return confirmed
        finally:
            if confirmed:
                logger.info("XHand release: Passive command acknowledged")
            else:
                logger.warning("XHand release: Passive command unconfirmed")

    def _send_command(self) -> int:
        error = self.control.send_command(self.cfg.device_id, self.command)
        code = int(error.error_code)
        if code not in (0, OVERCURRENT_WARNING_CODE, COMMUNICATION_CRC_ERROR_CODE):
            raise XHandError("send", code, error.error_message)
        if code == COMMUNICATION_CRC_ERROR_CODE:
            self.consecutive_write_crc += 1
            self.crc_counts["write_errors"] += 1
            self.warn_limited("write_crc", "XHand write CRC; target delivery unconfirmed")
        else:
            if self.consecutive_write_crc:
                self.crc_counts["write_recoveries"] += 1
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
        if not self._overcurrent_warned:
            logger.warning("XHand 过流告警：%s code=%d msg=%s", operation, OVERCURRENT_WARNING_CODE, message)
            self._overcurrent_warned = True

    def _make_command(self, qpos: np.ndarray) -> Any:
        """用首帧已校验的关节位置初始化命令。"""
        command = self.sdk.HandCommand_t()
        joints = command.finger_command
        if len(joints) != HAND_DOF:
            raise ValueError(f"XHand SDK command must provide {HAND_DOF} joint slots, got {len(joints)}")
        for index, (joint, position) in enumerate(zip(joints, qpos, strict=True)):
            joint.id = index
            joint.position = float(position)
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
