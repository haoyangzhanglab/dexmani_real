"""父进程提交目标和生命周期请求，SDK 调用由子进程完成。"""

import multiprocessing as mp
import time

import numpy as np

from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.xarm7.config import (
    EXECUTOR_TIMEOUT_S,
    HOME_CONVERGENCE_TIMEOUT_S,
)
from dexmani_real.robot.xarm7.protocol import (
    GOAL_DTYPE,
    ArmChannels,
    HomeRequest,
    LifecycleReply,
    PrepareRequest,
)


class ArmClient:
    def __init__(
        self,
        cfg,
        *,
        run_signal,
        stop_signal,
        emergency_signal,
        signal_lock,
        input_signal=None,
    ):
        self.cfg = cfg
        self.ctx = mp.get_context("spawn")
        self.io = ArmChannels(
            self.ctx,
            run_signal=run_signal,
            stop_signal=stop_signal,
            emergency_signal=emergency_signal,
            signal_lock=signal_lock,
            input_signal=input_signal,
        )
        self.pipe, self.child_pipe = self.ctx.Pipe()
        self.process = None
        self.sequence = self.operation_id = 0
        self.operation_unresolved = False

    @property
    def is_connected(self):
        return self.process is not None and self.process.is_alive() and self.io.connected.is_set()

    @property
    def fault(self):
        return self.io.fault.value.decode(errors="replace") or None

    def connect(self, *, check_services=None):
        from ruckig import Ruckig

        from dexmani_real.robot.xarm7.worker import run_arm

        Ruckig(7)
        if self.process is not None:
            raise RuntimeError("arm client is single-use")

        run_id = self.io.run_id.value

        def check_startup():
            if check_services is not None and not check_services():
                raise RobotError("runtime service unavailable during arm startup")
            if (
                self.io.run_id.value != run_id
                or self.io.stop.is_set()
                or self.io.emergency.is_set()
            ):
                raise OperationCancelled("arm startup cancelled")

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
                self.check_error()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "arm initialization unresolved; owner process may still be in SDK"
                    )
            check_startup()
            self.check_error()
        except BaseException:
            self.request_stop()
            self.io.closing.set()
            raise

    def check_error(self):
        issue = self.fault
        if not issue and self.process is not None and not self.process.is_alive():
            issue = f"arm owner exited: {self.process.exitcode}"
        if issue:
            raise RobotError(f"xArm controller fault: {issue}")

    def snapshot(self):
        self.check_error()
        status = self.io.read_status()
        stamp = int(status["frame"]["timestamp_ns"])
        if not status["valid"] or not 0 < stamp <= time.monotonic_ns() < stamp + int(
            self.cfg.feedback_max_age_s * 1e9
        ):
            raise RuntimeError("arm feedback unavailable/stale")
        frame = status["frame"].copy().reshape(1)
        frame.flags.writeable = False
        return frame

    def update_health(self, *, health_until_ns):
        self.io.health_until.value = int(health_until_ns)

    def limit_execution_until(self, deadline_ns):
        limit = self.io.execution_limit
        with limit.get_lock():
            limit[0] = (
                min(limit[0], int(deadline_ns)) if limit[1] == self.sequence else int(deadline_ns)
            )
            limit[1] = self.sequence

    def submit(self, command, *, submit_until_ns, execute_until_ns):
        self.check_error()
        if not self.is_connected:
            raise RobotError("arm device unavailable")
        prepared_run_id = self.io.read_status()["prepared_run_id"]
        goal = np.zeros((), dtype=GOAL_DTYPE)
        goal[()] = (
            self.sequence + 1,
            command.run_id,
            command.input_generation,
            submit_until_ns,
            execute_until_ns,
            command.arm_qpos,
        )
        with self.io.goal_lock:
            # IPC 等待之后重新检查运行身份、停止信号和 jog 输入代际。
            if (
                self.io.closing.is_set()
                or self.io.stop.is_set()
                or self.io.emergency.is_set()
                or self.io.run_id.value != command.run_id
            ):
                raise OperationCancelled("arm target run cancelled")
            if (
                command.input_generation >= 0
                and self.io.input_generation.value != command.input_generation
            ):
                raise OperationCancelled("jog input changed before publication")
            if time.monotonic_ns() >= min(submit_until_ns, execute_until_ns):
                raise TimeoutError("arm target deadline expired")
            if prepared_run_id != command.run_id:
                raise RuntimeError("arm target run was not prepared")
            slot = np.frombuffer(self.io.goal, dtype=GOAL_DTYPE, count=1)
            if slot["sequence"][0]:
                raise RuntimeError("arm target slot is full")
            slot[0] = goal
            self.sequence += 1

    def receive_reply(self, identity) -> LifecycleReply:
        reply = self.pipe.recv()
        if reply.operation_id != identity:
            raise RuntimeError("arm lifecycle reply mismatch")
        return reply

    def request_operation(self, request, *, service, abort, deadline_ns):
        if not self.is_connected or self.operation_unresolved:
            raise RobotError("arm unavailable or previous operation unresolved")
        service()
        if abort():
            raise OperationCancelled("arm operation cancelled before submission")
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("arm operation deadline expired before submission")
        self.operation_id = request.operation_id
        self.pipe.send(request)
        service_due, received = 0, False
        try:
            while not self.pipe.poll(0.005):
                now = time.monotonic_ns()
                if now >= service_due:
                    service()
                    service_due = now + round(1e9 / self.cfg.feedback_hz)
                if abort():
                    raise OperationCancelled("arm operation cancelled")
                if time.monotonic_ns() >= deadline_ns or not self.is_connected:
                    raise TimeoutError("arm operation unresolved before deadline")
            reply = self.receive_reply(request.operation_id)
            received = True
            if reply.error is not None:
                raise reply.error
            if abort():
                raise OperationCancelled("arm operation cancelled")
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("arm operation completed after deadline")
        except BaseException:
            self.request_stop()
            self.operation_unresolved = not received
            raise

    def begin_operation(self, run_id):
        self.check_error()
        self.request_stop()
        self.wait_stopped()
        with self.io.signal_lock:
            self.check_error()
            if (
                self.io.run_id.value != run_id
                or self.io.emergency.is_set()
                or self.io.closing.is_set()
            ):
                raise OperationCancelled("operation cancelled before preparation")
            if not self.io.stopped.is_set() or self.operation_unresolved:
                raise RobotError("previous arm operation is still unresolved")
            self.io.stop.clear()
            self.io.stopped.clear()

    def prepare(self, run_id, *, service):
        self.begin_operation(run_id)
        self.request_operation(
            PrepareRequest(self.operation_id + 1, run_id),
            service=service,
            abort=lambda: self.io.run_id.value != run_id or self.io.stop.is_set(),
            deadline_ns=time.monotonic_ns() + int((HOME_CONVERGENCE_TIMEOUT_S + 2) * 1e9),
        )

    def home(self, waypoints, target, run_id, *, service, abort, deadline_ns):
        self.begin_operation(run_id)
        self.request_operation(
            HomeRequest(self.operation_id + 1, run_id, waypoints, target, deadline_ns),
            service=service,
            abort=abort,
            deadline_ns=deadline_ns,
        )

    def request_stop(self):
        self.io.stop.set()

    def wait_stopped(self):
        deadline = time.monotonic() + EXECUTOR_TIMEOUT_S
        if self.process is not None and self.process.is_alive():
            if not self.io.stopped.wait(EXECUTOR_TIMEOUT_S):
                raise TimeoutError(
                    "arm stop unresolved; SDK acceptance/physical stop not confirmed"
                )
            if self.operation_unresolved:
                if not self.pipe.poll(max(0, deadline - time.monotonic())):
                    raise TimeoutError(
                        "arm stopped but cancelled lifecycle operation remains unresolved"
                    )
                self.receive_reply(self.operation_id)
                self.operation_unresolved = False
        elif self.process is not None and self.process.exitcode not in (None, 0):
            raise RuntimeError(f"arm owner exited without confirming stop: {self.process.exitcode}")
        if self.io.close_error.value:
            raise RuntimeError(self.io.close_error.value.decode(errors="replace"))

    def close(self):
        self.request_stop()
        self.io.closing.set()
        if self.process is not None and self.process.pid is not None:
            self.process.join(EXECUTOR_TIMEOUT_S)
            if self.process.is_alive():
                raise TimeoutError("arm owner process still active; stop/disconnect unresolved")
            if self.process.exitcode != 0:
                raise RuntimeError(f"arm owner abnormal exit: {self.process.exitcode}")
        self.pipe.close()
        self.child_pipe.close()
        if self.io.close_error.value:
            raise RuntimeError(self.io.close_error.value.decode(errors="replace"))
