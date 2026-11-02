"""父进程提交目标和生命周期请求，SDK 调用由子进程完成。"""

import logging
import multiprocessing as mp
import time

from dexmani_real.config.xarm7 import (
    EXECUTOR_TIMEOUT_S,
    HOME_CONVERGENCE_TIMEOUT_S,
)
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.xarm7.protocol import (
    ArmChannels,
    HomeRequest,
    PrepareRequest,
)

logger = logging.getLogger(__name__)


class ArmProcess:
    def __init__(
        self,
        cfg,
        *,
        run_signal,
        stop_signal,
        emergency_signal,
    ):
        self.cfg = cfg
        self.ctx = mp.get_context("spawn")
        self.io = ArmChannels(
            self.ctx,
            run_signal=run_signal,
            stop_signal=stop_signal,
            emergency_signal=emergency_signal,
        )
        self.pipe, self.child_pipe = self.ctx.Pipe()
        self.process = None
        self._fault_error = None
        self._close_error = None

    @property
    def is_connected(self):
        return (
            self.process is not None and self.process.is_alive()
            and self.io.connected.is_set() and not self.io.closing.is_set()
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

    def connect(self, *, check_services=None):
        from dexmani_real.robot.xarm7.worker import run_arm

        if self.process is not None:
            raise RuntimeError("arm process is single-use")

        run_id = self.io.run_id.value

        def check_startup():
            # 故障也会设置 stop；仅无故障的退出可以按操作者取消处理。
            self.check_error()
            if (
                self.io.run_id.value != run_id
                or self.io.stop.is_set()
                or self.io.emergency.is_set()
            ):
                raise OperationCancelled("arm startup cancelled")
            if check_services is not None and not check_services():
                raise RobotError("runtime service unavailable during arm startup")

        check_startup()
        self.process = self.ctx.Process(
            target=run_arm,
            args=(self.cfg, self.io, self.child_pipe),
            name="xarm-owner",
            daemon=False,
        )
        self.process.start()
        self.child_pipe.close()
        deadline = time.monotonic() + (HOME_CONVERGENCE_TIMEOUT_S + EXECUTOR_TIMEOUT_S)
        try:
            while not self.io.ready.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic()))):
                check_startup()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "arm initialization unresolved; owner process may still be in SDK"
                    )
            check_startup()
        except BaseException:
            self.request_stop()
            self.io.closing.set()
            raise

    def _read_fault(self):
        # owner 的首个故障不会被清除；父进程复用同一异常，保留原始 cause。
        issue = self.fault
        if (not issue and self.process is not None and self.process.pid is not None
                and not self.process.is_alive()):
            if self.process.exitcode != 0 or not self.io.closing.is_set():
                issue = f"arm owner exited: {self.process.exitcode}"
            elif not self.io.stopped.is_set():
                issue = "arm owner exited without confirming stop"
        if issue and self._fault_error is None:
            self._fault_error = RobotError(f"xArm controller fault: {issue}")
        return self._fault_error

    def _read_close_error(self):
        message = self.io.close_error.value.decode(errors="replace")
        # disconnect 会追加详情；更新已有异常，避免再附加一份已报告的停止错误。
        if message:
            if self._close_error is None:
                self._close_error = RobotError(message)
            else:
                self._close_error.args = (message,)
        return self._close_error

    def check_error(self):
        error = self._read_fault() or self._read_close_error()
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

    def request_operation(self, request: PrepareRequest | HomeRequest, *, service, abort):
        """准备和 HOME 共用停止、授权、提交与回复收尾边界。"""
        deadline_ns = request.deadline_ns
        self.check_error()
        if not self.is_connected:
            raise RuntimeError("arm unavailable for lifecycle operation")
        self.request_stop()
        self.wait_stopped(deadline=min(
            time.monotonic() + EXECUTOR_TIMEOUT_S, deadline_ns / 1e9,
        ))
        lock = self.io.run_id.get_lock()
        if not lock.acquire(timeout=max(0, min(
            EXECUTOR_TIMEOUT_S, (deadline_ns - time.monotonic_ns()) / 1e9,
        ))):
            raise TimeoutError("arm operation generation lock unavailable")
        try:
            self.check_error()
            if (
                self.io.run_id.get_obj().value != request.run_id
                or self.io.emergency.is_set()
            ):
                raise OperationCancelled("operation cancelled before preparation")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("arm operation deadline expired before preparation")
            self.io.stopped.clear()
            self.io.stop.clear()
        finally:
            lock.release()
        reply_pending = False
        try:
            service()
            if abort():
                raise OperationCancelled("arm operation cancelled before submission")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("arm operation deadline expired before submission")
            self.pipe.send(request)
            reply_pending = True
            while not self.pipe.poll(0.005):
                service()
                if abort():
                    raise OperationCancelled("arm operation cancelled")
                if time.monotonic_ns() >= deadline_ns or not self.is_connected:
                    raise TimeoutError("arm operation unresolved before deadline")
            error = self.pipe.recv()
            reply_pending = False
            if error is not None:
                if isinstance(error, Exception) and self.fault:
                    raise self._read_fault() from error
                raise error
            if abort():
                raise OperationCancelled("arm operation cancelled")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("arm operation completed after deadline")
        except BaseException as failure:
            errors = []
            try:
                self.request_stop()
            except BaseException as exc:
                errors.append(("arm_operation_stop_request", exc))
            # 本请求仍独占自己的回复；停止等待绝不接管 Pipe。
            cleanup_until = time.monotonic() + EXECUTOR_TIMEOUT_S
            try:
                if reply_pending:
                    if not self.pipe.poll(max(0, cleanup_until - time.monotonic())):
                        raise TimeoutError("cancelled arm operation reply remains pending")
                    reply = self.pipe.recv()
                    if reply is not None:
                        from dexmani_real.robot.xarm7.worker import _OperationTimeout

                        # owner 自身的预算回复已消费；SDK TimeoutError 仍是故障。
                        if self.fault or not isinstance(reply, (OperationCancelled, _OperationTimeout)):
                            if self.fault:
                                raise self._read_fault() from reply
                            raise RobotError("arm operation also failed") from reply
            except BaseException as exc:
                errors.append(("arm_operation_reply", exc))
            try:
                self.wait_stopped(deadline=cleanup_until)
            except BaseException as exc:
                errors.append(("arm_operation_stop", exc))
            if errors:
                # 回复和停止分别收尾；任一未确认都禁止复用这个 owner。
                self.io.closing.set()
                for stage, error in errors[1:]:
                    logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
                if isinstance(failure, OperationCancelled):
                    raise errors[0][1]
                stage, error = errors[0]
                logger.error("%s also failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise

    def prepare(self, run_id, *, service):
        self.request_operation(
            PrepareRequest(
                run_id,
                time.monotonic_ns() + int((HOME_CONVERGENCE_TIMEOUT_S + EXECUTOR_TIMEOUT_S) * 1e9),
            ),
            service=service,
            abort=lambda: self.io.run_id.value != run_id or self.io.stop.is_set(),
        )

    def home(self, waypoints, target, run_id, *, service, abort, deadline_ns):
        self.request_operation(
            HomeRequest(run_id, waypoints, target, deadline_ns),
            service=service,
            abort=abort,
        )

    def request_stop(self):
        self.io.stop.set()

    def wait_stopped(self, *, deadline=None):
        deadline = time.monotonic() + EXECUTOR_TIMEOUT_S if deadline is None else deadline
        if self.process is not None and self.process.is_alive():
            if not self.io.stopped.wait(max(0, deadline - time.monotonic())):
                raise TimeoutError("arm stop unresolved; SDK acceptance not confirmed")
        error = self._read_close_error()
        if error is not None:
            raise error
        self.check_error()

    def close(self):
        self.request_stop()
        self.io.closing.set()
        errors: list[BaseException] = []
        reported_fault = self._fault_error
        reported_close_error = str(self._close_error) if self._close_error is not None else None
        if self.process is not None and self.process.pid is not None:
            try:
                self.process.join(EXECUTOR_TIMEOUT_S)
            except BaseException as exc:
                errors.append(exc)
        # join 后保留晚到故障；已上报的运行故障不再冒充一次关闭失败。
        fault, close_error = self._read_fault(), self._read_close_error()
        if fault is not None and fault is not reported_fault:
            errors.insert(0, fault)
        if close_error is not None and str(close_error) != reported_close_error:
            errors.append(close_error)
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
                    errors.append(exc)
        if errors:
            for error in errors[1:]:
                logger.error("additional arm close failure", exc_info=(type(error), error, error.__traceback__))
            raise errors[0]
