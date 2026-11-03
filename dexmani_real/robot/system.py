"""父进程独占手部 I/O，机械臂 SDK 由子进程持有。"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.config.xarm7 import EXECUTOR_TIMEOUT_S
from dexmani_real.config.xhand import INITIAL_STATE_TIMEOUT_S
from dexmani_real.robot.commands import (
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
        self.hand_lower = np.asarray(XHAND_RATED_QPOS_MIN_RAD, dtype=np.float64)
        self.hand_upper = np.asarray(XHAND_RATED_QPOS_MAX_RAD, dtype=np.float64)
        self.runtime_state = runtime_state
        self.check_services = check_services
        self.arm: ArmProcess | None = None
        self.hand: XHand | None = None
        self.connected = False
        self.stop_required = False
        self.stop_failure: BaseException | None = None
        self.hand_feedback = XHandFeedback()
        self.feedback_next_ns = 0

    def check(self) -> None:
        if self.arm is not None:
            self.arm.check_error()
        if self.stop_failure is not None:
            raise self.stop_failure
        if self.check_services is not None and not self.check_services():
            raise RobotError("required runtime service unavailable")

    def check_run(self, run_id, *, expires_ns=None):
        # stop 与 fault 可以同时到达；不能先把真实失败分类为取消。
        self.check()
        if self.runtime_state.stop_signal.is_set() or self.runtime_state.emergency_signal.is_set():
            raise OperationCancelled("device operation cancelled")
        lock = self.runtime_state.run_signal.get_lock()
        timeout_s = (EXECUTOR_TIMEOUT_S if expires_ns is None
                     else max(0, expires_ns - time.monotonic_ns()) / 1e9)
        if not lock.acquire(timeout=timeout_s):
            raise TimeoutError("device generation lock unavailable")
        try:
            if self.runtime_state.run_signal.get_obj().value != run_id:
                raise OperationCancelled("device operation cancelled")
        finally:
            lock.release()
        if self.arm is not None:
            self.arm.check_error()
        # 等待共享锁期间可能已有取消；不能在手 SDK 之后才观察到。
        if self.runtime_state.stop_signal.is_set() or self.runtime_state.emergency_signal.is_set():
            raise OperationCancelled("device operation cancelled")

    def connect(self):
        run_id = self.runtime_state.run_signal.value

        def cancelled():
            self.check()
            return (
                self.runtime_state.run_signal.value != run_id
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
        self.stop_required = True  # 连接后仍需完成首次整机停止确认。
        self.arm.connect(check_services=self.check_services)
        if cancelled():
            raise OperationCancelled("robot startup cancelled before hand connection")
        if self.hand_config.enabled:
            from dexmani_real.robot.xhand.driver import XHand

            self.hand = XHand(self.hand_config)
            initial = self.hand.connect(cancel_requested=cancelled, runtime_state=self.runtime_state)
            self.hand_feedback.consume(initial)
            self.feedback_next_ns = initial.timestamp_ns + FEEDBACK_PERIOD_NS
        self.connected = True
        self._read_initial_state(cancel_requested=cancelled)

    def reset_tactile_sensors(self, *, run_id, cancel_requested, required_tactile=()) -> None:
        self.check()
        if self.hand is None:
            raise RobotError("XHand unavailable for tactile reset")

        def cancelled():
            self.check()
            return bool(
                cancel_requested()
                or self.runtime_state.run_signal.value != run_id
                or self.runtime_state.emergency_signal.is_set()
            )

        if cancelled():
            raise OperationCancelled("tactile reset cancelled before stopping")
        self.stop()
        try:
            self.hand.reset_tactile_sensors(cancel_requested=cancelled)
        finally:
            self.hand_feedback.reset()
            self.feedback_next_ns = 0
        self._read_initial_state(
            required_tactile=required_tactile, cancel_requested=cancelled,
        )

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

    def service_feedback(self) -> None:
        """服务手部采样与已有故障；arm 观测由消费方显式读取。"""
        self.check()
        try:
            now = time.monotonic_ns()
            if now >= self.feedback_next_ns:
                self.read_hand()
                self.feedback_next_ns = next_tick_ns(
                    self.feedback_next_ns or now, FEEDBACK_PERIOD_NS, time.monotonic_ns(),
                )
        except BaseException:
            self.runtime_state.stop_signal.set()
            raise

    def _read_initial_state(self, *, cancel_requested, required_tactile=()):
        """回调负责完整故障检查及当前操作的取消语义，异常原样传播。"""
        deadline_ns = time.monotonic_ns() + int(INITIAL_STATE_TIMEOUT_S * 1e9)

        def check_wait():
            if cancel_requested():
                raise OperationCancelled("robot initial feedback acquisition cancelled")
            if time.monotonic_ns() >= deadline_ns:
                raise RobotError(
                    "robot initial feedback acquisition timed out; "
                    f"required tactile={required_tactile}"
                )

        while True:
            check_wait()
            state = self.read_state()
            check_wait()
            now_ns = time.monotonic_ns()
            arm_stamp = int(state.arm["timestamp_ns"][0])
            arm_ready = sample_is_fresh(arm_stamp, self.arm_config.feedback_max_age_s, now_ns)
            hand_ready = self.hand is None
            if self.hand is not None and state.hand is not None:
                fields = ("timestamp_ns", *(f"tactile_{c}_timestamp_ns" for c in required_tactile))
                stamps = (int(state.hand[name][0]) for name in fields)
                hand_ready = all(
                    sample_is_fresh(stamp, self.hand_config.feedback_max_age_s, now_ns)
                    for stamp in stamps
                )
            if arm_ready and hand_ready:
                return state
            wait_ns = min(self.feedback_next_ns, deadline_ns) - time.monotonic_ns()
            if wait_ns > 0:
                time.sleep(min(0.01, wait_ns / 1e9))
            check_wait()
            self.service_feedback()

    def prepare_streaming(self, run_id):
        if self.stop_failure is not None:
            raise self.stop_failure
        self.stop_required = True
        cast("ArmProcess", self.arm).prepare(run_id, service=self.service_feedback)
        self.check_run(run_id)

    def send_action(self, command, *, expires_ns):
        """顺序发布 arm、发送 hand；返回不表示原子执行、SDK 回执或到位。"""
        self.validate_command(command)
        self._send_action(command, expires_ns=expires_ns)

    def validate_command(self, command):
        """组合命令格式与绝对限位必须先于任一设备副作用。"""
        if self.stop_failure is not None:
            raise self.stop_failure
        if command.arm_qpos is not None or command.arm_hold:
            if not self.connected or self.arm is None or not self.arm.is_connected:
                raise RobotError("arm device unavailable")
            if command.arm_qpos is not None:
                validate_arm_target(command.arm_qpos, self.arm_lower, self.arm_upper)
        if command.hand_qpos is not None:
            if not self.connected or self.hand is None or not self.hand.is_connected:
                raise RobotError("hand device unavailable")
            validate_hand_target(command.hand_qpos, self.hand_lower, self.hand_upper)

    def _send_action(self, command, *, expires_ns):
        try:
            if command.arm_qpos is not None or command.arm_hold:
                self.stop_required = True
                cast("ArmProcess", self.arm).submit(command, expires_ns=expires_ns)
            if command.hand_qpos is not None:
                self.check_run(command.run_id, expires_ns=expires_ns)
                frame = self.hand_feedback.snapshot()
                now = time.monotonic_ns()
                if now >= expires_ns:
                    raise OperationCancelled("target expired before device submission")
                self.stop_required = True
                stamp = int(frame["timestamp_ns"][0]) if frame is not None else 0
                if not sample_is_fresh(stamp, self.hand_config.feedback_max_age_s, now):
                    raise OperationCancelled("current hand feedback unavailable/stale at send")
                cast("XHand", self.hand)._send_action(command.hand_qpos)
            self.check_run(command.run_id, expires_ns=expires_ns)
        except BaseException:
            self.runtime_state.stop_signal.set()
            raise

    def send_hand_home(self, command, *, expires_ns):
        if command.arm_qpos is not None or command.arm_hold:
            raise ValueError("hand HOME cannot carry an arm streaming target")
        self.validate_command(command)
        self.stop()
        self.service_feedback()
        lock = self.runtime_state.run_signal.get_lock()
        if not lock.acquire(timeout=max(0, expires_ns - time.monotonic_ns()) / 1e9):
            raise TimeoutError("hand HOME generation lock unavailable")
        try:
            if (
                self.runtime_state.run_signal.get_obj().value != command.run_id
                or self.runtime_state.emergency_signal.is_set()
            ):
                raise OperationCancelled("hand HOME cancelled before submission")
            self.runtime_state.stop_signal.clear()
        finally:
            lock.release()
        self._send_action(command, expires_ns=expires_ns)

    def home_arm(self, waypoints, target, run_id, abort_check, *, deadline_ns):
        if self.stop_failure is not None:
            raise self.stop_failure
        if abort_check():
            raise OperationCancelled("arm HOME cancelled before submission")
        self.stop_required = True
        cast("ArmProcess", self.arm).home(
            waypoints,
            target,
            run_id,
            abort=abort_check,
            service=self.service_feedback,
            deadline_ns=deadline_ns,
        )
        if abort_check():
            raise OperationCancelled("arm HOME cancelled during execution")

    def stop_hand(self, *, passive=False):
        if self.hand is None:
            return
        frame = self.hand_feedback.snapshot()
        if not passive and frame is not None:
            self.hand.stop(frame["qpos"][0], timestamp_ns=int(frame["timestamp_ns"][0]))
        else:
            self.hand.stop()

    def stop(self):
        """停止确认成功后可重复调用；准备和目标提交会重新置 stop_required。"""
        self.runtime_state.stop_signal.set()
        if not self.stop_required:
            return
        errors = []
        if self.arm is not None:
            try:
                self.arm.request_stop()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(("arm_stop_request", exc))
        try:
            self.stop_hand()
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
        if self.arm is None and self.hand is None:
            errors.append(("stop", RobotError("motion remains unresolved after disconnect")))
        if errors:
            # 本次会话保留停止故障；后续收尾成功也不能把运行伪装成正常结束。
            self.stop_failure = self.stop_failure or errors[0][1]
            for stage, error in errors:
                if error is not self.stop_failure:
                    logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise self.stop_failure
        self.stop_required = False

    def service_idle(self):
        if self.stop_required and (
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
        # stop_failure 已由 stop/check 上报；这里仅收集本次关闭产生的失败。
        errors = []
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
                self.hand.disconnect(runtime_state=self.runtime_state)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(("hand_disconnect", exc))
            finally:
                if self.hand.control is None:
                    self.hand = None

        self.connected = False
        self.stop_required = self.arm is not None or self.hand is not None
        if errors:
            for stage, error in errors[1:]:
                logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise errors[0][1]
