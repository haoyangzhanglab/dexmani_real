"""父进程独占手部 I/O，机械臂 SDK 由子进程持有。"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
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
from dexmani_real.robot.xhand.feedback import XHandFeedback
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S
from dexmani_real.utils.timing import next_tick_ns, sample_is_fresh

logger = logging.getLogger(__name__)

FEEDBACK_PERIOD_NS = round(1e9 / 30)
HAND_HOME_OUTPUT_LIMIT = 150

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
        timing=None,
        hand_tactile_channels: tuple[str, ...] = (),
        defer_hand_crc: bool = True,
    ):
        self.arm_config, self.hand_config = arm_config, hand_config
        self.arm_lower = np.asarray(arm_config.joint_limit_lower, dtype=np.float64)
        self.arm_upper = np.asarray(arm_config.joint_limit_upper, dtype=np.float64)
        self.hand_lower = np.array(hand_config.qpos_min_rad, dtype=np.float64, copy=True)
        self.hand_upper = np.array(hand_config.qpos_max_rad, dtype=np.float64, copy=True)
        self.runtime_state = runtime_state
        self.check_services = check_services
        self.timing = timing
        self.hand_tactile_channels = hand_tactile_channels
        self.defer_hand_crc = defer_hand_crc
        self.arm: ArmProcess | None = None
        self.hand: XHand | None = None
        self.connected = False
        self._hand_hold_tor_max: tuple[int, ...] | None = None  # 待确认的保持输出上限。
        self.hand_feedback = XHandFeedback()
        self._hand_refresh_required = False
        self.feedback_next_ns = 0
        self.home_collision_model = None  # 配置在会话内固定；仅 HOME 按需构造。

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
            timing=self.timing.arm if self.timing is not None else None,
        )
        self.arm.connect(check_services=self.check_services)
        if cancelled():
            raise OperationCancelled("robot startup cancelled before hand connection")
        if self.hand_config.enabled:
            from dexmani_real.robot.xhand.driver import XHand

            self.hand = XHand(self.hand_config, tactile_channels=self.hand_tactile_channels)
            self._hand_hold_tor_max = self.hand_config.tor_max
            initial = self.hand.connect(cancel_requested=cancelled)
            self.hand_feedback.consume(initial)
            self._hand_refresh_required = False
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

    def read_hand(self, *, retry_crc: bool = True) -> np.ndarray | None:
        if self.hand is None:
            return None
        timing = self.timing
        started_ns = time.monotonic_ns() if timing is not None else 0
        state = None
        try:
            state = self.hand.get_state(retry_crc=retry_crc)
        finally:
            if timing is not None:
                timing.add(
                    "hand_feedback", timestamp_ns=started_ns, finished_ns=time.monotonic_ns(),
                    feedback_timestamp_ns=state.timestamp_ns if state is not None else None,
                    succeeded=state is not None,
                )
        frame = self.hand_feedback.consume(state)
        if state is not None:
            self._hand_refresh_required = False
        return frame

    @property
    def hand_stop_required(self) -> bool:
        return self._hand_hold_tor_max is not None

    @property
    def hand_feedback_pending(self) -> bool:
        """读 CRC 尚未恢复，或写入结果未明确后尚无有效关节反馈。"""
        return self.hand is not None and (
            self._hand_refresh_required or self.hand.consecutive_read_crc > 0
        )

    def read_state(self):
        """消费来源时间不变的反馈缓存，不执行硬件查询。"""
        if not self.connected or self.arm is None:
            raise RobotError("robot is not connected")
        return RobotState(
            self.arm.snapshot(),
            self.hand_feedback.snapshot() if self.hand is not None else None,
        )

    def service_feedback(self) -> None:
        """刷新到期反馈并检查设备故障。"""
        try:
            self.check()
            now = time.monotonic_ns()
            if now >= self.feedback_next_ns:
                self.read_hand(retry_crc=not self.defer_hand_crc)
                finished_ns = time.monotonic_ns()
                if self.defer_hand_crc and self.hand is not None and (
                    0 < self.hand.consecutive_read_crc <= self.hand_config.crc_retries
                ):
                    # 在线退避沿用反馈调度，保留旧样本时间与 pending 状态。
                    self.feedback_next_ns = finished_ns + round(self.hand_config.crc_retry_interval_s * 1e9)
                else:
                    self.feedback_next_ns = next_tick_ns(
                        self.feedback_next_ns or now, FEEDBACK_PERIOD_NS, finished_ns,
                    )
            if not self.runtime_state.stop_signal.is_set():
                self._check_hand_feedback()
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

    def _read_initial_state(self, *, cancel_requested) -> None:
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
                return
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

    def send_action(self, command, *, expires_ns) -> int | None:
        """组合提交成功后返回臂邮箱发布时间，纯手部命令返回 None。"""
        self.validate_command(command)
        hand_target = command.hand_qpos
        if hand_target is not None and self.hand_feedback_pending:
            raise FeedbackUnavailable("fresh XHand feedback pending after CRC or unconfirmed submission")
        timing = self.timing
        arm_published_ns = None
        accepted = False
        try:
            self.check_run()
            if hand_target is not None:
                stamp = self._check_hand_feedback()
                hand_deadline_ns = min(expires_ns, stamp + int(self.hand_config.feedback_max_age_s * 1e9))
            if command.arm_qpos is not None or command.arm_hold:
                arm_published_ns = cast("ArmProcess", self.arm).submit(command, expires_ns=expires_ns)
            if hand_target is not None:
                # hand 独占于主线程，arm 发布后仍按该反馈的期限检查真正发送。
                hand_started_ns = time.monotonic_ns() if timing is not None else 0
                hand_sent = False
                try:
                    cast("XHand", self.hand).send_action(
                        hand_target, deadline_ns=hand_deadline_ns, check_abort=self.check_run,
                    )
                    hand_sent = True
                    self._hand_hold_tor_max = self.hand_config.tor_max
                except CommandExpired:
                    raise  # SDK 未收到本次写入。
                except BaseException:
                    # 其余异常无法排除写入已生效，用新反馈决定保持位置。
                    self._hand_hold_tor_max = self.hand_config.tor_max
                    self._hand_refresh_required = True
                    self.feedback_next_ns = 0
                    raise
                finally:
                    if timing is not None:
                        timing.add(
                            "hand_send", timestamp_ns=hand_started_ns, finished_ns=time.monotonic_ns(),
                            arm_published_ns=arm_published_ns, succeeded=hand_sent,
                        )
            self.check_run()
            accepted = True
        except BaseException:
            self.runtime_state.stop_signal.set()
            raise
        finally:
            if timing is not None and arm_published_ns is not None:
                timing.add(
                    "arm_publish", timestamp_ns=arm_published_ns, expires_ns=expires_ns,
                    qpos=command.arm_qpos, hold=command.arm_hold, accepted=accepted,
                )
        return arm_published_ns

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
        """提交最终 HOME 目标，返回是否收到过流告警。"""
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
        from dexmani_real.robot.xhand.driver import OVERCURRENT_WARNING_CODE

        tor_max = tuple(min(value, HAND_HOME_OUTPUT_LIMIT) for value in self.hand_config.tor_max)
        previous_tor_max = self._hand_hold_tor_max
        self._hand_hold_tor_max = tor_max
        try:
            code = self.hand.send_position(
                target, tor_max=tor_max,
                deadline_ns=min(deadline_ns, stamp + int(self.hand_config.feedback_max_age_s * 1e9)),
                check_abort=check_abort,
            )
        except CommandExpired:
            self._hand_hold_tor_max = previous_tor_max
            raise
        except BaseException:
            self._hand_refresh_required = True
            self.feedback_next_ns = 0
            raise
        return code == OVERCURRENT_WARNING_CODE

    def hold_hand(self, *, check_abort=None) -> bool:
        """使用最近可信的反馈保持位置，必要时刷新一次。"""
        if self.hand is None:
            return False
        tor_max = self._hand_hold_tor_max or self.hand_config.tor_max
        self._hand_hold_tor_max = tor_max
        frame = self.hand_feedback.snapshot()
        stamp = int(frame["timestamp_ns"][0]) if frame is not None else 0
        if self.hand_feedback_pending or not sample_is_fresh(
            stamp, self.hand.hold_feedback_max_age_s, time.monotonic_ns(),
        ):
            frame = self.read_hand()
        if frame is None or self.hand_feedback_pending:
            return False
        try:
            confirmed = self.hand.hold(
                frame["qpos"][0], timestamp_ns=int(frame["timestamp_ns"][0]),
                tor_max=tor_max, check_abort=check_abort,
            )
        except CommandExpired:
            raise
        except BaseException:
            self._hand_refresh_required = True
            self.feedback_next_ns = 0
            raise
        if confirmed:
            self._hand_hold_tor_max = None
        return confirmed

    @contextmanager
    def _stop_stage(self, errors: list[tuple[str, BaseException]], stage: str) -> Iterator[None]:
        try:
            yield
        except BaseException as exc:  # noqa: BLE001 — 锁存中断并继续清理，最后传播错误。
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            errors.append((stage, exc))

    def stop(self, *, service=None) -> None:
        """请求机械臂停止，同时保持 XHand 位置，并确认机械臂停稳。"""
        self.runtime_state.stop_signal.set()
        errors: list[tuple[str, BaseException]] = []
        if self.hand_stop_required:
            with self._stop_stage(errors, "hand_hold"):
                if not self.hold_hand():
                    raise FeedbackUnavailable("XHand position hold not confirmed")
        if self.arm is not None:
            def service_stop():
                if not errors:
                    with self._stop_stage(errors, "stop_service"):
                        service()

            with self._stop_stage(errors, "arm_stop_confirm"):
                self.arm.wait_physical_stop(service=service_stop if service is not None else None)
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

        hand = self.hand
        if hand is not None:
            try:
                with self._stop_stage(errors, "hand_disconnect"):
                    hand.disconnect()
            finally:
                if hand.control is None:
                    self.hand = None
                if self.timing is not None:
                    self.timing.add("xhand_crc", counts=hand.crc_counts.copy())

        self.connected = False
        if self.hand is None:
            self._hand_hold_tor_max = None
        if errors:
            for stage, error in errors[1:]:
                logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise errors[0][1]
