"""父进程提交目标和生命周期请求，SDK 调用由子进程完成。"""

import logging
import multiprocessing as mp
import time
from typing import TYPE_CHECKING

from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.xarm7.protocol import (
    EXECUTOR_TIMEOUT_S,
    READY_TIMEOUT_S,
    ArmChannels,
    HomeRequest,
    PhysicalStopRequest,
    PrepareRequest,
)
from dexmani_real.utils.timing import sample_is_fresh

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from dexmani_real.runtime.state import SessionState

logger = logging.getLogger(__name__)

STOP_REPLY_GRACE_S = 0.05


class ArmProcess:
    def __init__(
        self,
        cfg,
        *,
        runtime_state: "SessionState",
        timing=None,
    ):
        self.runtime_state = runtime_state
        self.cfg = cfg
        self.timing = timing
        self.ctx = mp.get_context("spawn")
        self.io = ArmChannels(
            self.ctx,
            stop_signal=runtime_state.stop_signal,
            emergency_signal=runtime_state.emergency_signal,
        )
        self.pipe, self.child_pipe = self.ctx.Pipe()
        self.process: BaseProcess | None = None
        self._reported_owner_error: str | None = None
        self._reported_stop_error: str | None = None

    @property
    def is_connected(self) -> bool:
        # ready 只表示连接初始化成功；不代表已准备运动。
        return (
            self.process is not None and self.process.is_alive()
            and self.io.ready.is_set() and not self.io.closing.is_set()
        )

    @property
    def is_closed(self):
        return (
            (self.process is None or not self.process.is_alive())
            and self.pipe.closed and self.child_pipe.closed
        )

    @property
    def physical_stop_confirmed(self) -> bool:
        return (
            self.io.stop.is_set() and self.io.stop_observed.is_set()
            and not self.io.stop_failed.is_set()
        )

    @property
    def needs_close_stop_confirmation(self) -> bool:
        # 同一停止周期内，关闭可以复用刚刚完成的停稳确认。
        confirmed = self.physical_stop_confirmed and sample_is_fresh(
            self.io.stop_observed_ns.value, self.cfg.feedback_max_age_s, time.monotonic_ns(),
        )
        return self.io.device_claimed.is_set() and not confirmed

    @property
    def stop_reason(self):
        self.check_error()
        if self.io.stop.is_set():
            return self.io.stop_reason.value.decode() or None
        return None

    def connect(self, *, check_services=None):
        from dexmani_real.robot.xarm7.worker import run_arm

        if self.process is not None:
            raise RuntimeError("arm process is single-use")

        cancel_epoch = self.runtime_state.cancel_epoch
        startup_deadline_ns = time.monotonic_ns() + int(
            (READY_TIMEOUT_S + EXECUTOR_TIMEOUT_S) * 1e9
        )

        def check_startup():
            self.check_error()
            if check_services is not None and not check_services():
                raise RobotError("runtime service unavailable during arm startup")
            if (self.io.stop.is_set() or self.io.emergency.is_set()
                    or self.io.closing.is_set() or self.runtime_state.cancel_epoch != cancel_epoch):
                raise OperationCancelled("arm startup cancelled")
            if time.monotonic_ns() >= startup_deadline_ns:
                raise TimeoutError("arm initialization unresolved; owner may still be in SDK")

        check_startup()
        self.process = self.ctx.Process(
            target=run_arm,
            args=(self.cfg, self.io, self.child_pipe, startup_deadline_ns, self.timing),
            name="xarm-owner",
            daemon=False,
        )
        try:
            self.process.start()
            self.child_pipe.close()
            while not self.io.ready.wait(timeout=min(
                0.05, max(0, startup_deadline_ns - time.monotonic_ns()) / 1e9,
            )):
                check_startup()
            check_startup()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            self.io.stop.set()
            self.io.closing.set()
            raise

    def _read_owner_fault(self) -> RobotError | None:
        issue = self.io.fault.value.decode(errors="replace")
        return RobotError(f"xArm controller fault: {issue}") if issue else None

    def _read_owner_exit_error(self) -> RobotError | None:
        exitcode = self.process.exitcode if self.process is not None else None
        if exitcode is None:
            return None
        if exitcode != 0 or not self.io.closing.is_set():
            return RobotError(f"arm owner exited: {exitcode}")
        if self.io.device_claimed.is_set() and not self.io.sdk_stop_handled.is_set():
            return RobotError("arm owner exited without handling SDK stop")
        return None

    def _read_stop_error(self) -> RobotError | None:
        if self.io.stop_failed.is_set():
            return RobotError(self.io.stop_error.value.decode(errors="replace"))
        return None

    def _check_stop_result(self) -> None:
        error = self._read_stop_error()
        if error is not None:
            self._reported_stop_error = str(error)
            raise error

    def check_error(self) -> None:
        error = self._read_owner_fault() or self._read_owner_exit_error()
        if error is not None:
            self._reported_owner_error = str(error)
            raise error
        self._check_stop_result()

    def snapshot(self):
        self.check_error()
        if not self.is_connected:
            raise RuntimeError("arm unavailable for feedback snapshot")
        status = self.io.read_status(timeout_s=self.cfg.feedback_max_age_s)
        frame = status.reshape(1)
        frame.flags.writeable = False
        return frame

    def submit(self, command, *, expires_ns) -> int:
        self.check_error()
        if not self.is_connected:
            raise RuntimeError("arm unavailable for target submission")
        return self.io.publish_goal(command, expires_ns=expires_ns)

    def resume(self, cancel_epoch, *, deadline_ns):
        """旧发布调用已返回；确认停止、清空邮箱后才恢复设备接纳。"""
        self.check_error()
        if not self.is_connected:
            raise RuntimeError("arm unavailable for lifecycle operation")
        self.io.stop.set()
        self._wait_for_sdk_stop(deadline_ns=min(
            time.monotonic_ns() + int(EXECUTOR_TIMEOUT_S * 1e9), deadline_ns,
        ))
        self.io.clear_goal(deadline_ns=min(deadline_ns, time.monotonic_ns() + int(EXECUTOR_TIMEOUT_S * 1e9)))
        with self.runtime_state.motion_lock:
            self.check_error()
            if (self.runtime_state.cancel_epoch != cancel_epoch
                    or not self.runtime_state.requests_allow_motion() or self.io.closing.is_set()):
                raise OperationCancelled("operation cancelled before preparation")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("arm operation deadline expired before preparation")
            # Owner 接纳请求时清除上一停止周期的确认。
            self.io.stop.clear()

    def request_operation(self, request: PrepareRequest | HomeRequest, *, service, abort, cancel_epoch):
        """普通取消确认停止并消费回复后抛出；其他失败关闭 owner。"""
        deadline_ns = request.deadline_ns
        # 生命周期仅由一个父控制流程顺序调用；局部事实只决定异常收尾。
        submitted = False
        reply_received = False

        def receive_reply():
            nonlocal reply_received
            error = self.pipe.recv()
            # recv 成功即已消费；取消回复本身也会抛出异常。
            reply_received = True
            if error is not None:
                if isinstance(error, Exception):
                    fault = self._read_owner_fault()
                    if fault is not None:
                        self._reported_owner_error = str(fault)
                        raise fault from error
                raise error

        try:
            self.resume(cancel_epoch, deadline_ns=deadline_ns)
            service()
            if abort():
                raise OperationCancelled("arm operation cancelled before submission")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("arm operation deadline expired before submission")
            self.pipe.send(request)
            submitted = True
            while not self.pipe.poll(0.005):
                service()
                if abort():
                    raise OperationCancelled("arm operation cancelled")
                if time.monotonic_ns() >= deadline_ns or not self.is_connected:
                    raise TimeoutError("arm operation unresolved before deadline")
            receive_reply()
            self.check_error()
            if abort():
                raise OperationCancelled("arm operation cancelled")
        except BaseException as failure:
            if isinstance(failure, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            self.io.stop.set()
            if (isinstance(failure, OperationCancelled)
                    and not self.io.emergency.is_set() and not self.io.closing.is_set()):
                cleanup_deadline = time.monotonic() + EXECUTOR_TIMEOUT_S
                try:
                    # 停止和消费本次回复共用预算，避免旧回复进入下一次操作。
                    while True:
                        service()
                        self.check_error()
                        if not self.is_connected:
                            raise RobotError("arm owner unavailable after cancellation")
                        if self.io.emergency.is_set():
                            raise OperationCancelled("arm emergency stop during cancellation")
                        remaining = cleanup_deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("arm cancellation unresolved: stop or lifecycle reply missing")
                        if self.io.sdk_stop_handled.is_set() and (not submitted or reply_received):
                            break
                        if submitted and not reply_received:
                            if self.pipe.poll(min(0.005, remaining)):
                                try:
                                    receive_reply()
                                except OperationCancelled:
                                    pass  # 本次普通取消；真实错误仍向外传播。
                        else:
                            self.io.sdk_stop_handled.wait(min(0.005, remaining))
                except BaseException as cleanup:
                    if isinstance(cleanup, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    self.io.closing.set()
                    raise
                raise
            self.io.closing.set()
            # 保留当前失败，最终 close 读取 owner 的晚到故障。
            raise

    def prepare(self, cancel_epoch, *, service):
        self.request_operation(
            PrepareRequest(
                time.monotonic_ns() + int((READY_TIMEOUT_S + EXECUTOR_TIMEOUT_S) * 1e9),
            ),
            service=service,
            abort=lambda: self.runtime_state.cancel_epoch != cancel_epoch or self.io.stop.is_set(),
            cancel_epoch=cancel_epoch,
        )

    def publish_home_hand_stamp(self, stamp):
        value = self.io.home_hand_stamp
        if not value.get_lock().acquire(timeout=self.cfg.feedback_max_age_s):
            raise TimeoutError("HOME hand timestamp lock unavailable")
        try:
            value.get_obj().value = stamp
        finally:
            value.get_lock().release()

    def home(self, cancel_epoch, *, samples, hand_max_age_s, service, abort, deadline_ns):
        self.request_operation(
            HomeRequest(deadline_ns, samples, hand_max_age_s),
            service=service,
            abort=abort,
            cancel_epoch=cancel_epoch,
        )

    def _wait_for_stop_event(self, event, *, deadline_ns, stage, service=None) -> None:
        while not event.is_set():
            if self.process is None or not self.process.is_alive():
                if event.is_set():
                    break  # Owner 可能在检查期间发布结果并退出。
                raise RobotError(f"{stage} unresolved: arm owner exited before reporting")
            if service is not None:
                service()
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                raise TimeoutError(f"{stage} unresolved: owner reply timed out")
            event.wait(min(0.005 if service is not None else 0.05, remaining_ns / 1e9))

    def _wait_for_sdk_stop(self, *, deadline_ns) -> None:
        self._wait_for_stop_event(
            self.io.sdk_stop_handled, deadline_ns=deadline_ns, stage="arm SDK stop",
        )
        self._check_stop_result()

    def wait_physical_stop(self, *, service=None) -> None:
        """等待 SDK 停止和实测停稳，独立于此前的运动故障。"""
        if not self.io.stop.is_set():
            raise OperationCancelled("arm stop was not requested")
        self._wait_for_sdk_stop(
            deadline_ns=time.monotonic_ns() + int(EXECUTOR_TIMEOUT_S * 1e9),
        )

        deadline_ns = time.monotonic_ns() + int(EXECUTOR_TIMEOUT_S * 1e9)
        # Owner 到达采样期限后仍需发送失败回复。
        reply_deadline_ns = deadline_ns + round(STOP_REPLY_GRACE_S * 1e9)
        if self.is_connected and not self.io.fault.value:
            try:
                self.pipe.send(PhysicalStopRequest(deadline_ns))
                while not self.pipe.poll(0.005):
                    if not self.is_connected or self.io.fault.value:
                        break
                    if service is not None:
                        service()
                    if time.monotonic_ns() >= reply_deadline_ns:
                        raise TimeoutError("arm physical stop unresolved: owner reply timed out")
                else:
                    self.pipe.recv()
            except (BrokenPipeError, EOFError):
                self.io.closing.set()
            except BaseException:
                # 未消费的晚到回复不能进入下一次操作。
                self.io.closing.set()
                raise
        self._wait_for_stop_event(
            self.io.stop_observed, deadline_ns=reply_deadline_ns,
            stage="arm physical stop", service=service,
        )
        self._check_stop_result()

    def close(self) -> None:
        """回收 Owner 和管道，物理停稳确认须在关闭反馈通道前完成。"""
        self.io.stop.set()
        self.io.closing.set()
        errors: list[BaseException] = []
        if self.process is not None and self.process.pid is not None:
            try:
                self.process.join(EXECUTOR_TIMEOUT_S)
            except BaseException as exc:  # noqa: BLE001 — 中断 join 后仍须检查 Owner 状态。
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(exc)
        # join 后报告晚到的故障和断开错误，已经传播的故障由其调用者保留。
        fault = self._read_owner_fault()
        if fault is not None and str(fault) != self._reported_owner_error:
            errors.insert(0, fault)
        stop_error = self._read_stop_error()
        if stop_error is not None and str(stop_error) != self._reported_stop_error:
            errors.append(stop_error)
        disconnect_error = self.io.disconnect_error.value.decode(errors="replace")
        if disconnect_error:
            errors.append(RobotError(f"xArm disconnect failed: {disconnect_error}"))
        alive = self.process is not None and self.process.is_alive()
        if alive:
            errors.append(TimeoutError("arm owner process still active; stop/disconnect unresolved"))
        else:
            exit_error = self._read_owner_exit_error()
            if exit_error is not None and str(exit_error) != self._reported_owner_error:
                if fault is None:
                    errors.insert(0, exit_error)
                else:
                    errors.append(exit_error)
            # 活着的 owner 保留管道；确认退出后尝试关闭全部端点。
            for pipe in (self.child_pipe, self.pipe):
                try:
                    pipe.close()
                except BaseException as exc:  # noqa: BLE001 — 其余管道仍须回收。
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    errors.append(exc)
        if errors:
            for error in errors[1:]:
                logger.error("additional arm close failure", exc_info=(type(error), error, error.__traceback__))
            raise errors[0]
