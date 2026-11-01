"""父进程提交目标和生命周期请求，SDK 调用由子进程完成。"""

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

    @property
    def is_connected(self):
        return self.process is not None and self.process.is_alive() and self.io.connected.is_set()

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
            if not self.io.fault.value and (
                self.io.run_id.value != run_id
                or self.io.stop.is_set()
                or self.io.emergency.is_set()
            ):
                raise OperationCancelled("arm startup cancelled")
            self.check_error()
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

    def check_error(self):
        issue = self.fault
        if not issue and self.process is not None and not self.process.is_alive():
            issue = f"arm owner exited: {self.process.exitcode}"
        if not issue and self.io.closing.is_set():
            issue = "arm owner closing; lifecycle operation cannot be reused"
        if issue:
            raise RobotError(f"xArm controller fault: {issue}")

    def snapshot(self):
        self.check_error()
        status = self.io.read_status(timeout_s=self.cfg.feedback_max_age_s)
        frame = status.reshape(1)
        frame.flags.writeable = False
        return frame

    def submit(self, command, *, expires_ns):
        self.io.publish_goal(command, expires_ns=expires_ns)

    def request_operation(self, request, *, service, abort, deadline_ns):
        if not self.is_connected or self.io.closing.is_set():
            raise RobotError("arm unavailable for lifecycle operation")
        service()
        if abort():
            raise OperationCancelled("arm operation cancelled before submission")
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("arm operation deadline expired before submission")
        self.pipe.send(request)
        received = False
        try:
            while not self.pipe.poll(0.005):
                service()
                if abort():
                    raise OperationCancelled("arm operation cancelled")
                if time.monotonic_ns() >= deadline_ns or not self.is_connected:
                    raise TimeoutError("arm operation unresolved before deadline")
            error = self.pipe.recv()
            received = True
            if error is not None:
                if (isinstance(error, OperationCancelled)
                        or isinstance(error, TimeoutError) and not self.fault
                        or not isinstance(error, Exception)):
                    raise error
                raise RobotError(f"arm operation failed: {error}") from error
            if abort():
                raise OperationCancelled("arm operation cancelled")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("arm operation completed after deadline")
        except BaseException as failure:
            self.request_stop()
            # 本请求仍独占自己的回复；停止等待绝不接管 Pipe。
            cleanup_until = time.monotonic() + EXECUTOR_TIMEOUT_S
            try:
                if not received:
                    if not self.pipe.poll(max(0, cleanup_until - time.monotonic())):
                        raise TimeoutError("cancelled arm operation reply remains pending")
                    reply = self.pipe.recv()
                    if reply is not None and not isinstance(reply, OperationCancelled):
                        raise RobotError(f"arm operation also failed: {reply}") from reply
                self.wait_stopped(deadline=cleanup_until)
            except BaseException as cleanup_error:
                # 不再复用这个 owner；保留句柄直到明确关闭。
                self.io.closing.set()
                setattr(failure, "cleanup_errors", [
                    *getattr(failure, "cleanup_errors", ()),
                    ("arm_operation_cleanup", cleanup_error),
                ])
                raise failure from cleanup_error
            raise

    def begin_operation(self, run_id):
        self.check_error()
        self.request_stop()
        self.wait_stopped()
        lock = self.io.run_id.get_lock()
        if not lock.acquire(timeout=EXECUTOR_TIMEOUT_S):
            raise TimeoutError("arm preparation generation lock unavailable")
        try:
            self.check_error()
            if (
                self.io.run_id.get_obj().value != run_id
                or self.io.emergency.is_set()
            ):
                raise OperationCancelled("operation cancelled before preparation")
            self.io.stopped.clear()
            self.io.stop.clear()
        finally:
            lock.release()

    def prepare(self, run_id, *, service):
        self.begin_operation(run_id)
        self.request_operation(
            PrepareRequest(run_id),
            service=service,
            abort=lambda: self.io.run_id.value != run_id or self.io.stop.is_set(),
            deadline_ns=time.monotonic_ns() + int((HOME_CONVERGENCE_TIMEOUT_S + 2) * 1e9),
        )

    def home(self, waypoints, target, run_id, *, service, abort, deadline_ns):
        self.begin_operation(run_id)
        self.request_operation(
            HomeRequest(run_id, waypoints, target, deadline_ns),
            service=service,
            abort=abort,
            deadline_ns=deadline_ns,
        )

    def request_stop(self):
        self.io.stop.set()

    def wait_stopped(self, *, deadline=None):
        deadline = time.monotonic() + EXECUTOR_TIMEOUT_S if deadline is None else deadline
        if self.process is not None and self.process.is_alive():
            if not self.io.stopped.wait(max(0, deadline - time.monotonic())):
                raise TimeoutError("arm stop unresolved; SDK acceptance not confirmed")
        elif self.process is not None and (
            self.process.exitcode not in (None, 0)
            or self.process.exitcode == 0 and not self.io.stopped.is_set()
        ):
            raise RobotError(f"arm owner exited without confirming stop: {self.process.exitcode}")
        if self.io.close_error.value:
            raise RobotError(self.io.close_error.value.decode(errors="replace"))
        issue = self.fault
        if issue:
            raise RobotError(f"xArm controller fault: {issue}")

    def close(self):
        self.request_stop()
        self.io.closing.set()
        if self.process is not None and self.process.pid is not None:
            self.process.join(EXECUTOR_TIMEOUT_S)
            if self.process.is_alive():
                raise TimeoutError("arm owner process still active; stop/disconnect unresolved")
        # 活着的 owner 保留管道；确认退出后尝试关闭全部端点，保留首次失败。
        errors = []
        if self.io.close_error.value:
            errors.append(RuntimeError(self.io.close_error.value.decode(errors="replace")))
        if self.process is not None and (
            self.process.exitcode not in (None, 0)
            or self.process.exitcode == 0 and not self.io.stopped.is_set()
        ):
            errors.append(RobotError(f"arm owner exited without a clean stop: {self.process.exitcode}"))
        for pipe in (self.child_pipe, self.pipe):
            try:
                pipe.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            setattr(errors[0], "cleanup_errors", [
                *getattr(errors[0], "cleanup_errors", ()),
                *(("arm_close", error) for error in errors[1:]),
            ])
            raise errors[0]
