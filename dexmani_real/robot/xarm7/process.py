"""父进程提交目标和生命周期请求，SDK 调用由子进程完成。"""

import logging
import multiprocessing as mp
import time
from typing import TYPE_CHECKING

from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.xarm7.protocol import (
    READY_TIMEOUT_S,
    ArmChannels,
    HomeRequest,
    PrepareRequest,
)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from dexmani_real.runtime.state import SessionState

logger = logging.getLogger(__name__)

EXECUTOR_TIMEOUT_S = 3.0


class ArmProcess:
    def __init__(
        self,
        cfg,
        *,
        runtime_state: "SessionState",
    ):
        self.runtime_state = runtime_state
        self.cfg = cfg
        self.ctx = mp.get_context("spawn")
        self.io = ArmChannels(
            self.ctx,
            stop_signal=runtime_state.stop_signal,
            emergency_signal=runtime_state.emergency_signal,
        )
        self.pipe, self.child_pipe = self.ctx.Pipe()
        self.process: BaseProcess | None = None

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
    def fault(self):
        return self.io.fault.value.decode(errors="replace") or None

    @property
    def soft_stop_reason(self):
        self.check_error()
        if self.io.stop.is_set():
            reason = self.io.stop_reason.value.decode()
            if reason in {"command_expired", "arm_feedback_expired"}:
                return reason
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
            args=(self.cfg, self.io, self.child_pipe, startup_deadline_ns),
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
            self.request_stop()
            self.io.closing.set()
            raise

    def _read_fault(self):
        # owner 的首个故障不会被清除；每次检查直接报告当前事实。
        issue = self.fault
        if (not issue and self.process is not None and self.process.pid is not None
                and not self.process.is_alive()):
            if self.process.exitcode != 0 or not self.io.closing.is_set():
                issue = f"arm owner exited: {self.process.exitcode}"
            elif not self.io.stopped.is_set():
                issue = "arm owner exited without confirming stop"
        return RobotError(f"xArm controller fault: {issue}") if issue else None

    def check_error(self):
        error = self._read_fault()
        if error is not None:
            raise error

    def snapshot(self):
        self.check_error()
        if not self.is_connected:
            raise RuntimeError("arm unavailable for feedback snapshot")
        status = self.io.read_status(timeout_s=self.cfg.feedback_max_age_s)
        frame = status.reshape(1)
        frame.flags.writeable = False
        return frame

    def submit(self, command, *, expires_ns):
        self.check_error()
        if not self.is_connected:
            raise RuntimeError("arm unavailable for target submission")
        self.io.publish_goal(command, expires_ns=expires_ns)

    def resume(self, cancel_epoch, *, deadline_ns):
        """旧发布调用已返回；确认停止、清空邮箱后才恢复设备接纳。"""
        self.check_error()
        if not self.is_connected:
            raise RuntimeError("arm unavailable for lifecycle operation")
        self.request_stop()
        self.wait_stopped(deadline=min(
            time.monotonic() + EXECUTOR_TIMEOUT_S, deadline_ns / 1e9,
        ))
        self.io.clear_goal(deadline_ns=min(deadline_ns, time.monotonic_ns() + int(EXECUTOR_TIMEOUT_S * 1e9)))
        with self.runtime_state.motion_lock:
            self.check_error()
            if (self.runtime_state.cancel_epoch != cancel_epoch
                    or not self.runtime_state.requests_allow_motion() or self.io.closing.is_set()):
                raise OperationCancelled("operation cancelled before preparation")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("arm operation deadline expired before preparation")
            # stopped 仅由 owner 写入；它在接纳请求时失效旧确认。
            # 父进程此时清确认会使 owner 把尚未清除的 stop 当作新停止。
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
                if isinstance(error, Exception) and self.fault:
                    raise self._read_fault() from error
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
            self.request_stop()
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
                        if self.io.stopped.is_set() and (not submitted or reply_received):
                            break
                        if submitted and not reply_received:
                            if self.pipe.poll(min(0.005, remaining)):
                                try:
                                    receive_reply()
                                except OperationCancelled:
                                    pass  # 本次普通取消；真实错误仍向外传播。
                        else:
                            self.io.stopped.wait(min(0.005, remaining))
                except BaseException as cleanup:
                    if isinstance(cleanup, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    self.io.closing.set()
                    raise
                raise
            self.io.closing.set()
            # 异常请求不再恢复；最终 close 读取 owner 的晚到故障。
            self.check_error()
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

    def home(self, cancel_epoch, *, service, abort, deadline_ns):
        self.request_operation(
            HomeRequest(deadline_ns),
            service=service,
            abort=abort,
            cancel_epoch=cancel_epoch,
        )

    def request_stop(self):
        self.io.stop.set()

    def wait_stopped(self, *, deadline=None):
        deadline = time.monotonic() + EXECUTOR_TIMEOUT_S if deadline is None else deadline
        if (self.process is not None and self.process.is_alive()
                and not self.io.stopped.wait(max(0, deadline - time.monotonic()))):
            raise TimeoutError("arm stop unresolved; SDK acceptance not confirmed")
        self.check_error()

    def close(self) -> None:
        self.request_stop()
        self.io.closing.set()
        errors: list[BaseException] = []
        if self.process is not None and self.process.pid is not None:
            try:
                self.process.join(EXECUTOR_TIMEOUT_S)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                errors.append(exc)
        # join 后重新读取故障，包含关闭期间晚到的错误。
        fault = self._read_fault()
        if fault is not None:
            errors.insert(0, fault)
        alive = self.process is not None and self.process.is_alive()
        if alive:
            errors.append(TimeoutError("arm owner process still active; stop/disconnect unresolved"))
        else:
            # 无 SDK fault 的异常退出已由 _read_fault 报告；有主错时保留独立退出失败。
            if self.fault and self.process is not None and self.process.pid is not None:
                if self.process.exitcode != 0:
                    errors.append(RobotError(f"arm owner exited: {self.process.exitcode}"))
                elif not self.io.stopped.is_set():
                    errors.append(RobotError("arm owner exited without confirming stop"))
            # 活着的 owner 保留管道；确认退出后尝试关闭全部端点。
            for pipe in (self.child_pipe, self.pipe):
                try:
                    pipe.close()
                except BaseException as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    errors.append(exc)
        if errors:
            for error in errors[1:]:
                logger.error("additional arm close failure", exc_info=(type(error), error, error.__traceback__))
            raise errors[0]
