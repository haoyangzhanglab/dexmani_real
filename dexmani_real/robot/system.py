"""父进程独占手部 I/O，机械臂 SDK 由子进程持有。"""

import logging
import time
from bisect import bisect_left
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.robot.commands import (
    CommandExpired,
    FeedbackUnavailable,
    HandCommandUnconfirmed,
    HomeTimeout,
    OperationCancelled,
    RobotError,
    validate_arm_target,
    validate_hand_target,
)
from dexmani_real.robot.xhand.feedback import XHandFeedback
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S
from dexmani_real.utils.timing import next_tick_ns, sample_is_fresh

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
        self.hand_lower = np.array(hand_config.qpos_min_rad, dtype=np.float64, copy=True)
        self.hand_upper = np.array(hand_config.qpos_max_rad, dtype=np.float64, copy=True)
        self.runtime_state = runtime_state
        self.check_services = check_services
        self.arm: ArmProcess | None = None
        self.hand: XHand | None = None
        self.connected = False
        self.hand_stop_required = False
        self.hand_feedback = XHandFeedback()
        self._hand_write_unconfirmed = False
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
            right = bisect_left(history, timestamp_ns, key=lambda sample: sample[0])
            if right == len(history):
                return None
            hi, q_hi = history[right]
            if hi == timestamp_ns:
                q = q_hi.copy()
            else:
                if right == 0:
                    return None
                lo, q_lo = history[right - 1]
                if hi - lo > max_gap_s * 1e9:
                    return None
                q = q_lo + ((timestamp_ns - lo) / (hi - lo)) * (q_hi - q_lo)
            result[name + "_qpos"] = q
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
            self._hand_write_unconfirmed = False
            self.feedback_next_ns = initial.timestamp_ns + FEEDBACK_PERIOD_NS
        self.connected = True
        self._read_initial_state(cancel_requested=cancelled)

    def calibrate_tactile(self, *, no_contact: bool, cancel_epoch, cancel_requested) -> bool:
        """cancel_requested 负责设备故障轮询；此处补充停止与取消上下文。"""
        if self.hand is None:
            raise RobotError("XHand unavailable for tactile calibration")

        def cancelled():
            return bool(
                cancel_requested()
                or self.runtime_state.cancel_epoch != cancel_epoch
                or self.runtime_state.emergency_signal.is_set()
            )

        if cancelled():
            raise OperationCancelled("tactile calibration cancelled before stopping")
        self.stop()
        if cancelled():
            raise OperationCancelled("tactile calibration cancelled after stopping")
        try:
            return self.hand.calibrate_tactile(no_contact=no_contact, cancel_requested=cancelled)
        finally:
            # 下一份触觉必须来自本次校准之后的读取。
            self.hand_feedback.clear_tactile()
            self.feedback_next_ns = 0

    def read_hand(self) -> np.ndarray | None:
        if self.hand is None:
            return None
        state = self.hand.get_state()
        frame = self.hand_feedback.consume(state)
        if state is not None:
            self._hand_write_unconfirmed = False
        return frame

    @property
    def hand_feedback_pending(self) -> bool:
        """读 CRC 尚未恢复，或未确认写入之后尚无有效关节反馈。"""
        return self.hand is not None and (
            self._hand_write_unconfirmed or self.hand.consecutive_read_crc > 0
        )

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
                        # 驱动保证关节有限；arm 的未就绪反馈可能仍带零时间。
                        if stamp <= 0 or (history and stamp < history[-1][0]):
                            raise ValueError(f"{name}: invalid joint feedback timestamp {stamp}")
                        qpos = frame["qpos"][0]
                        if history and stamp == history[-1][0]:
                            if not np.array_equal(qpos, history[-1][1]):
                                raise ValueError(f"{name}: Conflicting joint states at the same timestamp")
                            continue
                        history.append((stamp, qpos.copy()))
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

    def _check_hand_feedback(self) -> int:
        if self.hand is None:
            return 0
        frame = self.hand_feedback.snapshot()
        stamp = int(frame["timestamp_ns"][0]) if frame is not None else 0
        if not sample_is_fresh(stamp, self.hand_config.feedback_max_age_s, time.monotonic_ns()):
            raise FeedbackUnavailable("current hand feedback unavailable/stale")
        return stamp

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
        """顺序发布 arm 邮箱并确认 hand SDK 发送结果。"""
        self.validate_command(command)
        if command.hand_qpos is not None and self.hand_feedback_pending:
            raise FeedbackUnavailable("fresh XHand feedback pending after CRC")
        try:
            self.check_run()
            if command.arm_qpos is not None or command.arm_hold:
                if command.hand_qpos is not None:
                    self._check_hand_feedback()
                cast("ArmProcess", self.arm).submit(command, expires_ns=expires_ns)
            if command.hand_qpos is not None:
                self.check_run()
                # arm 发布可能等待邮箱锁；真正发送 hand 前仍须复核当前反馈。
                stamp = self._check_hand_feedback()
                hand_deadline_ns = min(expires_ns, stamp + int(self.hand_config.feedback_max_age_s * 1e9))
                if time.monotonic_ns() >= hand_deadline_ns:
                    raise CommandExpired("command expired before hand submission")
                self.hand_stop_required = True
                cast("XHand", self.hand).send_action(
                    command.hand_qpos, deadline_ns=hand_deadline_ns, check_abort=self.check_run,
                )
            self.check_run()
        except HandCommandUnconfirmed:
            # 实测反馈可能早于已经生效的写入；立即安排新查询，成功前禁止新手部目标。
            self._hand_write_unconfirmed = True
            self.feedback_next_ns = 0
            raise
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
        """提交 HOME 小步目标并返回过流告警。"""
        if not self.connected or self.hand is None or not self.hand.is_connected:
            raise RobotError("hand device unavailable")
        # 操作范围已由连接前配置校验限制在额定范围内。
        target = validate_hand_target(qpos, self.hand_lower, self.hand_upper)
        # 回调负责完整故障与取消检查；反馈和期限仍在实际发送前独立检查。
        check_abort()
        if self.hand_feedback_pending:
            raise FeedbackUnavailable("HOME requires fresh XHand feedback after CRC")
        stamp = self._check_hand_feedback()
        if time.monotonic_ns() >= deadline_ns:
            raise HomeTimeout("hand HOME budget exhausted before submission")
        self.hand_stop_required = True
        from dexmani_real.robot.xhand.driver import OVERCURRENT_WARNING_CODE

        tor_max = np.minimum(self.hand_config.tor_max, self.hand_config.home_tor_max)
        code = self.hand.send_position(
            target, tor_max=tor_max,
            deadline_ns=min(deadline_ns, stamp + int(self.hand_config.feedback_max_age_s * 1e9)),
            check_abort=check_abort,
        )
        if code is None:
            raise FeedbackUnavailable("HOME XHand position submission unconfirmed within its deadline")
        return code == OVERCURRENT_WARNING_CODE

    def hold_hand(self, *, check_abort=None) -> bool:
        """使用最近可信的反馈保持位置，必要时刷新一次。"""
        if self.hand is None:
            return False
        self.hand_stop_required = True
        frame = self.hand_feedback.snapshot()
        stamp = int(frame["timestamp_ns"][0]) if frame is not None else 0
        if self.hand_feedback_pending or not sample_is_fresh(
            stamp, self.hand.hold_feedback_max_age_s, time.monotonic_ns(),
        ):
            frame = self.read_hand()
        if frame is None or self.hand_feedback_pending:
            return False
        confirmed = self.hand.hold(
            frame["qpos"][0], timestamp_ns=int(frame["timestamp_ns"][0]), check_abort=check_abort,
        )
        self.hand_stop_required = not confirmed
        return confirmed

    @contextmanager
    def _stop_stage(self, errors: list[tuple[str, BaseException]], stage: str) -> Iterator[None]:
        try:
            yield
        except BaseException as exc:  # noqa: BLE001 — 锁存中断并继续清理，最后传播错误。
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            errors.append((stage, exc))

    def stop(self) -> None:
        """请求机械臂停止，同时保持 XHand 位置，并确认机械臂停稳。"""
        self.runtime_state.stop_signal.set()
        errors: list[tuple[str, BaseException]] = []
        if self.hand_stop_required:
            with self._stop_stage(errors, "hand_hold"):
                if not self.hold_hand():
                    raise FeedbackUnavailable("XHand position hold not confirmed")
        if self.arm is not None:
            with self._stop_stage(errors, "arm_stop_confirm"):
                self.arm.wait_physical_stop()
        if errors:
            for stage, error in errors[1:]:
                logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise errors[0][1]

    def service_idle(self):
        stop_requested = self.runtime_state.stop_signal.is_set() or self.runtime_state.emergency_signal.is_set()
        arm_stop_pending = self.arm is not None and not self.arm.physical_stop_confirmed
        if stop_requested and (self.hand_stop_required or arm_stop_pending):
            self.stop()
        if self.connected:
            self.service_feedback()

    def close(self, *, release_hand: bool = False) -> None:
        """请求 Arm 停止并确认停稳，再处理 Hand 和设备关闭。"""
        if self.arm is None and self.hand is None:
            return
        # 先通知 Arm Owner，使停稳确认能在关闭反馈通道前完成。
        self.runtime_state.stop_signal.set()
        errors: list[tuple[str, BaseException]] = []
        if self.arm is not None and self.arm.needs_close_stop_confirmation:
            with self._stop_stage(errors, "arm_stop_confirm"):
                self.arm.wait_physical_stop()
        release_hand = release_hand and not errors
        # disconnect 失败仍持有句柄，但驱动已撤销读写；重试关闭只释放资源。
        if self.hand is not None and self.hand.is_connected:
            with self._stop_stage(errors, "hand_release" if release_hand else "hand_hold"):
                if release_hand:
                    if not self.hand.release():
                        raise RobotError("XHand release not confirmed")
                elif self.hand_stop_required and not self.hold_hand():
                    raise FeedbackUnavailable("XHand position hold not confirmed during close")
        if self.arm is not None:
            try:
                with self._stop_stage(errors, "arm_close"):
                    self.arm.close()
            finally:
                if self.arm.is_closed:
                    self.arm = None

        if self.hand is not None:
            try:
                with self._stop_stage(errors, "hand_disconnect"):
                    self.hand.disconnect()
            finally:
                if self.hand.control is None:
                    self.hand = None

        self.connected = False
        self.hand_stop_required = self.hand is not None
        if errors:
            for stage, error in errors[1:]:
                logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise errors[0][1]
