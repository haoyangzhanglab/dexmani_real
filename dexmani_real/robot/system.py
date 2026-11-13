"""父进程独占手部 I/O，机械臂 SDK 由子进程持有。"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.robot.commands import (
    CommandExpired,
    FeedbackUnavailable,
    HomeTimeout,
    OperationCancelled,
    RobotError,
    validate_arm_target,
    validate_hand_target,
)
from dexmani_real.robot.model import (
    XHAND_RATED_QPOS_MAX_RAD,
    XHAND_RATED_QPOS_MIN_RAD,
)
from dexmani_real.robot.xhand.feedback import XHandFeedback
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S
from dexmani_real.utils.timing import interpolate_joints, next_tick_ns, sample_is_fresh

logger = logging.getLogger(__name__)

FEEDBACK_PERIOD_NS = round(1e9 / 30)

if TYPE_CHECKING:
    from dexmani_real.robot.xarm7.process import ArmProcess
    from dexmani_real.robot.xhand.driver import XHand
    from dexmani_real.runtime.state import SessionState


@dataclass(frozen=True)
class RobotState:
    arm: np.ndarray
    hand: np.ndarray | None


class Robot:
    def __init__(
        self,
        arm_config,
        hand_config,
        *,
        runtime_state: "SessionState",
        check_services=None,
    ):
        self.arm_config, self.hand_config = arm_config, hand_config
        self.arm_lower = np.asarray(arm_config.joint_limit_lower, dtype=np.float64)
        self.arm_upper = np.asarray(arm_config.joint_limit_upper, dtype=np.float64)
        self.hand_lower = np.asarray(XHAND_RATED_QPOS_MIN_RAD, dtype=np.float64)
        self.hand_upper = np.asarray(XHAND_RATED_QPOS_MAX_RAD, dtype=np.float64)
        self.runtime_state = runtime_state
        self.check_services = check_services
        self.arm: ArmProcess | None = None
        self.hand: XHand | None = None
        self.connected = False
        self.hand_stop_required = False
        self.hand_feedback = XHandFeedback()
        self.feedback_next_ns = 0
        self.joint_history: dict[str, deque[tuple[int, np.ndarray]]] = {}
        self.joint_history_ns = 0
        self.home_collision_model = None  # 配置在会话内固定；仅 HOME 按需构造。

    def enable_joint_history(self, duration_s, *, devices):
        """每段重新建立历史，保留点云来源时间附近的反馈。"""
        self.joint_history = {name: deque() for name in devices}
        self.joint_history_ns = int(duration_s * 1e9)

    def aligned_joints(self, timestamp_ns, max_gap_s):
        """仅提供模型输入，不改变真实反馈及其来源时间。"""
        if not self.joint_history:
            return None
        result = {}
        for name, history in self.joint_history.items():
            if not history:
                return None
            stamps, positions = zip(*history)
            q, valid = interpolate_joints(stamps, positions, [timestamp_ns], max_gap_s)
            if not valid[0]:
                return None
            result[name + "_qpos"] = q[0]
        return result

    def check(self) -> None:
        if self.arm is not None:
            self.arm.check_error()
        if self.check_services is not None and not self.check_services():
            raise RobotError("required runtime service unavailable")

    def check_run(self):
        self.check()
        if self.runtime_state.stop_signal.is_set() or self.runtime_state.emergency_signal.is_set():
            raise OperationCancelled("device operation cancelled")

    def connect(self):
        if self.arm is not None or self.hand is not None:
            raise RobotError("robot still owns devices; close the old connection before reconnecting")
        cancel_epoch = self.runtime_state.cancel_epoch

        def cancelled():
            self.check()
            return (
                self.runtime_state.cancel_epoch != cancel_epoch
                or self.runtime_state.stop_signal.is_set()
                or self.runtime_state.emergency_signal.is_set()
            )

        # 各运行入口的 finally 统一关闭设备，也覆盖部分连接失败。
        if cancelled():
            raise OperationCancelled("robot startup cancelled")
        from dexmani_real.robot.xarm7.process import ArmProcess

        self.arm = ArmProcess(
            self.arm_config,
            runtime_state=self.runtime_state,
        )
        self.arm.connect(check_services=self.check_services)
        if cancelled():
            raise OperationCancelled("robot startup cancelled before hand connection")
        if self.hand_config.enabled:
            from dexmani_real.robot.xhand.driver import XHand

            self.hand = XHand(self.hand_config)
            self.hand_stop_required = True
            initial = self.hand.connect(cancel_requested=cancelled)
            self.hand_feedback.consume(initial)
            self.feedback_next_ns = initial.timestamp_ns + FEEDBACK_PERIOD_NS
        self.connected = True
        self._read_initial_state(cancel_requested=cancelled)

    def reset_tactile_sensors(self, *, cancel_epoch, cancel_requested) -> dict[int, int]:
        if self.hand is None:
            raise RobotError("XHand unavailable for tactile reset")

        def cancelled():
            self.check()
            return bool(
                cancel_requested()
                or self.runtime_state.cancel_epoch != cancel_epoch
                or self.runtime_state.emergency_signal.is_set()
            )

        if cancelled():
            raise OperationCancelled("tactile reset cancelled before stopping")
        self.stop()
        if cancelled():
            raise OperationCancelled("tactile reset cancelled after stopping")
        self.hand_feedback.clear_tactile()
        return self.hand.reset_tactile_sensors(cancel_requested=cancelled)

    def read_hand(self):
        if self.hand is None:
            return None
        return self.hand_feedback.consume(self.hand.get_state())

    def read_state(self):
        """消费来源时间不变的反馈缓存，不执行硬件查询。"""
        if not self.connected or self.arm is None:
            raise RobotError("robot is not connected")
        return RobotState(
            self.arm.snapshot(),
            self.hand_feedback.snapshot() if self.hand is not None else None,
        )

    def service_feedback(self) -> bool:
        """服务反馈与故障；返回本次是否到采样机会，不代表收到新帧。"""
        try:
            self.check()
            now = time.monotonic_ns()
            sampled = now >= self.feedback_next_ns
            if sampled:
                self.read_hand()
                if self.joint_history:
                    state = self.read_state()
                    for name, history in self.joint_history.items():
                        frame = getattr(state, name)
                        if frame is None:
                            continue
                        stamp = int(frame["timestamp_ns"][0])
                        if not history or stamp != history[-1][0]:
                            history.append((stamp, frame["qpos"][0].copy()))
                        # 留下窗口左侧的一个样本，供边界时刻插值。
                        while len(history) > 2 and history[1][0] < stamp - self.joint_history_ns:
                            history.popleft()
                self.feedback_next_ns = next_tick_ns(
                    self.feedback_next_ns or now, FEEDBACK_PERIOD_NS, time.monotonic_ns(),
                )
            if not self.runtime_state.stop_signal.is_set():
                self._check_hand_feedback()
            return sampled
        except BaseException:
            self.runtime_state.stop_signal.set()
            raise

    def _check_hand_feedback(self) -> None:
        if self.hand is None:
            return
        frame = self.hand_feedback.snapshot()
        stamp = int(frame["timestamp_ns"][0]) if frame is not None else 0
        if not sample_is_fresh(stamp, self.hand_config.feedback_max_age_s, time.monotonic_ns()):
            raise FeedbackUnavailable("current hand feedback unavailable/stale")

    def _read_initial_state(self, *, cancel_requested):
        """回调负责完整故障检查及当前操作的取消语义，异常原样传播。"""
        deadline_ns = time.monotonic_ns() + int(INITIAL_STATE_TIMEOUT_S * 1e9)

        def check_wait():
            if cancel_requested():
                raise OperationCancelled("robot initial feedback acquisition cancelled")
            if time.monotonic_ns() >= deadline_ns:
                raise RobotError("robot initial feedback acquisition timed out")

        while True:
            check_wait()
            state = self.read_state()
            check_wait()
            now_ns = time.monotonic_ns()
            arm_stamp = int(state.arm["timestamp_ns"][0])
            arm_ready = sample_is_fresh(arm_stamp, self.arm_config.feedback_max_age_s, now_ns)
            hand_ready = self.hand is None
            if self.hand is not None and state.hand is not None:
                hand_ready = sample_is_fresh(
                    int(state.hand["timestamp_ns"][0]), self.hand_config.feedback_max_age_s, now_ns,
                )
            if arm_ready and hand_ready:
                return state
            wait_ns = min(self.feedback_next_ns, deadline_ns) - time.monotonic_ns()
            if wait_ns > 0:
                time.sleep(min(0.01, wait_ns / 1e9))
            check_wait()
            self.service_feedback()

    def prepare_streaming(self, cancel_epoch, *, poll_operator=None):
        def service():
            if poll_operator is not None:
                poll_operator()
            self.service_feedback()
        service()
        cast("ArmProcess", self.arm).prepare(cancel_epoch, service=service)
        self.check_run()

    def send_action(self, command, *, expires_ns):
        """顺序发布 arm、发送 hand；返回不表示原子执行、SDK 回执或到位。"""
        self.validate_command(command)
        try:
            self.check_run()
            if command.arm_qpos is not None or command.arm_hold:
                if command.hand_qpos is not None:
                    self._check_hand_feedback()
                cast("ArmProcess", self.arm).submit(command, expires_ns=expires_ns)
            if command.hand_qpos is not None:
                self.check_run()
                # arm 发布可能等待邮箱锁；真正发送 hand 前仍须复核当前反馈。
                self._check_hand_feedback()
                if time.monotonic_ns() >= expires_ns:
                    raise CommandExpired("command expired before hand submission")
                self.hand_stop_required = True
                cast("XHand", self.hand)._send_action(command.hand_qpos)
            self.check_run()
        except BaseException:
            self.runtime_state.stop_signal.set()
            raise

    def validate_command(self, command):
        """组合命令格式与绝对限位必须先于任一设备副作用。"""
        if command.arm_qpos is not None or command.arm_hold:
            if not self.connected or self.arm is None or not self.arm.is_connected:
                raise RobotError("arm device unavailable")
            if command.arm_qpos is not None:
                validate_arm_target(command.arm_qpos, self.arm_lower, self.arm_upper)
        if command.hand_qpos is not None:
            if not self.connected or self.hand is None or not self.hand.is_connected:
                raise RobotError("hand device unavailable")
            validate_hand_target(command.hand_qpos, self.hand_lower, self.hand_upper)

    def send_hand_home(self, qpos, *, deadline_ns, check_abort) -> bool:
        """提交 HOME 小步目标并返回过流告警；CRC 立即失败。"""
        if not self.connected or self.hand is None or not self.hand.is_connected:
            raise RobotError("hand device unavailable")
        # 操作范围已由连接前配置校验限制在额定范围内。
        target = validate_hand_target(
            qpos, self.hand_config.qpos_min_rad, self.hand_config.qpos_max_rad,
        )
        check_abort()
        self.check()
        check_abort()
        self._check_hand_feedback()
        if time.monotonic_ns() >= deadline_ns:
            raise HomeTimeout("hand HOME budget exhausted before submission")
        self.hand_stop_required = True
        from dexmani_real.robot.xhand.driver import (
            COMMUNICATION_CRC_ERROR_CODE,
            OVERCURRENT_WARNING_CODE,
            XHandError,
        )

        code = self.hand._send_position(target, tor_max=np.minimum(
            self.hand_config.tor_max, self.hand_config.home_tor_max))
        if code == COMMUNICATION_CRC_ERROR_CODE:
            raise XHandError("HOME send", code, "delivery unconfirmed")
        return code == OVERCURRENT_WARNING_CODE

    def stop_hand(self, *, passive=False) -> bool:
        if self.hand is None:
            return False
        self.hand_stop_required = True
        frame = self.hand_feedback.snapshot()
        if not passive and frame is not None:
            return self.hand.stop(frame["qpos"][0], timestamp_ns=int(frame["timestamp_ns"][0]))
        else:
            return self.hand.stop()

    def stop(self):
        """arm 停止结果由 Owner 持有；仅未完成的手部停止需要再次发送。"""
        self.runtime_state.stop_signal.set()
        errors = []
        if self.hand_stop_required:
            try:
                if self.hand is None:
                    raise RobotError("hand stop unresolved after disconnect")
                self.stop_hand()
                self.hand_stop_required = False
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(("hand_stop", exc))
        if self.arm is not None:
            try:
                self.arm.wait_stopped()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(("arm_stop_confirm", exc))
        if errors:
            for stage, error in errors[1:]:
                logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise errors[0][1]

    def service_idle(self):
        if (
            self.runtime_state.stop_signal.is_set() or self.runtime_state.emergency_signal.is_set()
        ):
            self.stop()
        if self.connected:
            self.service_feedback()

    def close(self):
        if self.arm is None and self.hand is None:
            return
        # 共用 stop 先通知 arm；owner 退出后的最终故障也由 close 带回。
        self.runtime_state.stop_signal.set()
        # 首次停止失败由操作结果保留；这里只收集本次关闭失败。
        errors = []
        # disconnect 失败仍持有句柄，但驱动已撤销读写；重试关闭只释放资源。
        if self.hand is not None and self.hand.is_connected:
            try:
                self.stop_hand(passive=True)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(("hand_passive_stop", exc))
        if self.arm is not None:
            try:
                self.arm.close()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(("arm_close", exc))
            finally:
                if self.arm.is_closed:
                    self.arm = None

        if self.hand is not None:
            try:
                self.hand.disconnect()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(("hand_disconnect", exc))
            finally:
                if self.hand.control is None:
                    self.hand = None

        self.connected = False
        self.hand_stop_required = self.hand is not None
        if errors:
            for stage, error in errors[1:]:
                logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise errors[0][1]
