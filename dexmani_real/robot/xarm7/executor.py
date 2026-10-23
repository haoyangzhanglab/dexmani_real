"""Parent target submission and authority; the arm worker owns all SDK calls."""

import multiprocessing as mp
import time

import numpy as np

from dexmani_real.robot.commands import DispatchError, DispatchResult
from dexmani_real.runtime.state import SafetyState, RunEndReason
from dexmani_real.robot.xarm7.protocol import (
    MAX_TIME,
    GOAL_DTYPE,
    STATUS_DTYPE,
    PrepareRequest,
    HomeRequest,
    LifecycleReply,
    OperationStatus,
    OperationCancelled,
)


class ArmChannels:
    """Fixed xArm channels created before spawn; no SDK or RuntimeState inside."""

    def __init__(self, ctx):
        self.authority = ctx.Value("q", 0)
        self.emergency = ctx.Event()
        self.closing, self.stop, self.stopped, self.ready = (ctx.Event() for _ in range(4))
        self.connected, self.cancel_operation = ctx.Event(), ctx.Event()
        self.fault = ctx.Array("c", 1024, lock=False)
        self.close_error = ctx.Array("c", 1024, lock=False)
        self.service_until = ctx.Value("q", 0)
        self.upstream_until = ctx.Value("q", 0)
        self.hand_until = ctx.Value("q", MAX_TIME)
        self.execution_limit = ctx.Array("q", [MAX_TIME, 0])  # deadline, last affected command
        self.goal_lock, self.status_lock = ctx.Lock(), ctx.Lock()
        self.goal = ctx.RawArray("b", GOAL_DTYPE.itemsize)
        self.status = ctx.RawArray("b", STATUS_DTYPE.itemsize)

    def take_goal(self, *, block=False):
        if not self.goal_lock.acquire(block):
            return None  # Never stall the servo on the parent holding this short lock.
        try:
            slot = np.frombuffer(self.goal, dtype=GOAL_DTYPE, count=1)
            if not slot["sequence"][0]:
                return None
            goal = slot[0].copy()
            slot["sequence"][0] = 0
            return goal
        finally:
            self.goal_lock.release()

    def read_status(self):
        with self.status_lock:
            return np.frombuffer(self.status, dtype=STATUS_DTYPE, count=1)[0].copy()

    def publish(self, status):
        # Both sides hold this lock only for a small snapshot copy, never SDK I/O.
        with self.status_lock:
            np.frombuffer(self.status, dtype=STATUS_DTYPE, count=1)[0] = status


class ArmExecutor:
    def __init__(self, cfg, runtime_state, *, driver_factory=None):
        cfg.validate()
        self.cfg, self.runtime_state, self.factory = cfg, runtime_state, driver_factory
        self.ctx = mp.get_context("spawn")
        self.io = ArmChannels(self.ctx)
        self.pipe, self.child_pipe = self.ctx.Pipe()
        self.process = None
        self.sequence = self.operation_id = 0
        self.operation_unresolved = False

    @property
    def is_connected(self):
        return (
            self.process is not None and self.process.is_alive() and self.io.connected.is_set()
        )

    @property
    def fault(self):
        return self.io.fault.value.decode(errors="replace") or None

    def connect(self, *, check_services=None):
        from ruckig import Ruckig
        from dexmani_real.robot.xarm7.worker import run_arm

        Ruckig(7)
        if self.process is not None:
            raise RuntimeError("arm executor is single-use")

        def check_startup():
            if check_services is not None and not check_services():
                self.runtime_state.mark_error()
            if not self.runtime_state.requests_allow_motion():
                raise OperationCancelled("arm startup unauthorized/cancelled")

        check_startup()
        with self.runtime_state.motion_lock:
            if not self.runtime_state.requests_allow_motion():
                raise OperationCancelled("arm startup unauthorized/cancelled")
            self.runtime_state.arm_authority, self.runtime_state.arm_emergency = (
                self.io.authority,
                self.io.emergency,
            )
            # Publish initialization authority atomically with the stop/quit fence.
            self.io.authority.value = -self.runtime_state.run_id
        self.process = self.ctx.Process(
            target=run_arm,
            args=(self.cfg, self.io, self.child_pipe, self.factory),
            name="xarm-owner",
            daemon=False,
        )
        self.process.start()
        self.child_pipe.close()
        deadline = time.monotonic() + (
            self.cfg.homing.convergence_timeout_s + self.cfg.executor_timeout_s
        )
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
        self.runtime_state.publish_arm_authority()

    def check_error(self):
        issue = self.fault
        if not issue and self.process is not None and not self.process.is_alive():
            issue = f"arm owner exited: {self.process.exitcode}"
        if issue:
            self.runtime_state.mark_error()
            self.runtime_state.revoke_motion(SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise RuntimeError(f"xArm executor fault: {issue}")

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

    def update_health(self, *, service_until_ns=None, upstream_until_ns=None, hand_until_ns=None):
        for value, destination in (
            (service_until_ns, self.io.service_until),
            (upstream_until_ns, self.io.upstream_until),
            (hand_until_ns, self.io.hand_until),
        ):
            if value is not None:
                destination.value = int(value)

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
            raise RuntimeError("arm device unavailable")
        prepared_run_id = self.io.read_status()["prepared_run_id"]
        goal = np.zeros((), dtype=GOAL_DTYPE)
        goal[()] = (
            self.sequence + 1,
            command.run_id,
            submit_until_ns,
            execute_until_ns,
            command.arm_qpos,
        )
        with self.io.goal_lock:
            # Both IPC waits precede this fence. Revoked jog epochs are never reused.
            if (
                self.io.closing.is_set()
                or self.io.stop.is_set()
                or self.io.emergency.is_set()
                or self.io.authority.value != command.run_id
            ):
                raise DispatchError(
                    "arm target authority revoked", DispatchResult(),
                    revoked=True, cause="authority_revoked",
                )
            if time.monotonic_ns() >= min(submit_until_ns, execute_until_ns):
                raise DispatchError(
                    "arm target deadline expired", DispatchResult(),
                    revoked=True, cause="deadline_expired",
                )
            if prepared_run_id != command.run_id:
                raise RuntimeError("arm target epoch was not prepared")
            slot = np.frombuffer(self.io.goal, dtype=GOAL_DTYPE, count=1)
            if slot["sequence"][0]:
                raise RuntimeError("arm target slot is full")
            slot[0] = goal
            self.sequence += 1
        return (int(command.run_id), self.sequence)

    def receive_reply(self, identity):
        reply = self.pipe.recv()
        if not isinstance(reply, LifecycleReply) or reply.operation_id != identity:
            raise RuntimeError("arm lifecycle reply mismatch")
        if not isinstance(reply.status, OperationStatus):
            raise RuntimeError(f"unknown arm lifecycle status: {reply.status!r}")
        return reply

    def operation(self, kind, epoch, *, service, abort, timeout_s, waypoints=None, target=None):
        if self.operation_unresolved and self.pipe.poll():
            self.receive_reply(self.operation_id)
            self.operation_unresolved = False
        if not self.is_connected or self.operation_unresolved:
            raise RuntimeError(f"arm {kind}: owner unavailable or previous operation unresolved")
        self.operation_id += 1
        identity = self.operation_id
        service()
        if abort():
            raise OperationCancelled(f"{kind} authority revoked")
        self.io.cancel_operation.clear()
        request = (
            PrepareRequest(identity, epoch)
            if kind == "prepare"
            else HomeRequest(identity, epoch, waypoints, target)
        )
        self.pipe.send(request)
        deadline = time.monotonic() + timeout_s
        service_due, received = 0, False
        try:
            while not self.pipe.poll(0.005):
                now = time.monotonic()
                if now >= service_due:
                    service()  # XHand/input service remains exclusively in the parent.
                    service_due = now + self.cfg.homing.step_interval_s
                if abort():
                    raise OperationCancelled(f"{kind} authority revoked")
                if now >= deadline or not self.is_connected:
                    raise TimeoutError(f"arm {kind} unresolved")
            reply = self.receive_reply(identity)
            received = True
            if reply.status is OperationStatus.CANCELLED:
                raise OperationCancelled(reply.detail)
            if reply.status is OperationStatus.FAILED:
                raise RuntimeError(reply.detail)
        except BaseException:
            self.io.cancel_operation.set()
            self.request_stop()
            self.operation_unresolved = not received
            raise

    def prepare(self, epoch, *, service, abort):
        self.check_error()
        self.runtime_state.publish_arm_authority()
        self.operation(
            "prepare",
            epoch,
            service=service,
            abort=abort,
            timeout_s=self.cfg.homing.convergence_timeout_s + 2,
        )

    def home(self, waypoints, target, epoch, *, service, abort):
        self.check_error()
        self.runtime_state.publish_arm_authority()
        self.operation(
            "home",
            epoch,
            waypoints=waypoints,
            target=target,
            service=service,
            abort=abort,
            timeout_s=3600,
        )

    def request_stop(self):
        self.io.authority.value = 0
        if self.runtime_state.estop_request:
            self.io.emergency.set()
        self.io.stopped.clear()
        self.io.stop.set()

    def wait_stopped(self):
        deadline = time.monotonic() + self.cfg.executor_timeout_s
        if self.process is not None and self.process.is_alive():
            if not self.io.stopped.wait(self.cfg.executor_timeout_s):
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
            raise RuntimeError(
                f"arm owner exited without confirming stop: {self.process.exitcode}"
            )
        if self.io.close_error.value:
            raise RuntimeError(self.io.close_error.value.decode(errors="replace"))

    def close(self):
        self.request_stop()
        self.io.closing.set()
        if self.process is not None and self.process.pid is not None:
            self.process.join(self.cfg.executor_timeout_s)
            if self.process.is_alive():
                raise TimeoutError("arm owner process still active; stop/disconnect unresolved")
            if self.process.exitcode != 0:
                raise RuntimeError(f"arm owner abnormal exit: {self.process.exitcode}")
        self.pipe.close()
        self.child_pipe.close()
        if self.io.close_error.value:
            raise RuntimeError(self.io.close_error.value.decode(errors="replace"))
