"""机械臂共享通道：新目标覆盖未消费目标；反馈快照与低频操作独立传递。"""

import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.model import ARM_JOINT_SHAPE

ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", ARM_JOINT_SHAPE),
        ("qvel", "<f8", ARM_JOINT_SHAPE),
        ("timestamp_ns", "<u8"),
    ]
)

GOAL_DTYPE = np.dtype(
    [
        ("pending", "?"),
        ("run_id", "<i8"),
        ("hold", "?"),
        ("expires_ns", "<i8"),
        ("q_goal", "<f8", ARM_JOINT_SHAPE),
    ]
)


@dataclass(frozen=True)
class PrepareRequest:
    run_id: int
    deadline_ns: int


@dataclass(frozen=True)
class HomeRequest:
    run_id: int
    waypoints: np.ndarray
    target: np.ndarray
    deadline_ns: int


class ArmChannels:
    def __init__(
        self, ctx, *, run_signal, stop_signal, emergency_signal
    ):
        self.run_id, self.stop, self.emergency = run_signal, stop_signal, emergency_signal
        self.closing, self.stopped, self.ready = (ctx.Event() for _ in range(3))
        self.connected = ctx.Event()
        self.fault = ctx.Array("c", 1024, lock=False)
        self.close_error = ctx.Array("c", 1024, lock=False)
        self.goal_lock, self.status_lock = ctx.Lock(), ctx.Lock()
        self.goal = ctx.RawArray("b", GOAL_DTYPE.itemsize)
        self.status = ctx.RawArray("b", ARM_STATE_DTYPE.itemsize)

    def publish_goal(self, command, *, expires_ns):
        if not self.goal_lock.acquire(timeout=max(0, expires_ns - time.monotonic_ns()) / 1e9):
            raise TimeoutError("arm target publication expired waiting for mailbox")
        try:
            # IPC 等待之后重新检查运行身份和停止信号。
            generation_lock = self.run_id.get_lock()
            if not generation_lock.acquire(timeout=max(0, expires_ns - time.monotonic_ns()) / 1e9):
                raise TimeoutError("arm target generation lock unavailable")
            try:
                run_id = self.run_id.get_obj().value
            finally:
                generation_lock.release()
            if (
                self.closing.is_set()
                or self.stop.is_set()
                or self.emergency.is_set()
                or run_id != command.run_id
            ):
                raise OperationCancelled("arm target run cancelled")
            if time.monotonic_ns() >= expires_ns:
                raise TimeoutError("arm target deadline expired")
            np.frombuffer(self.goal, dtype=GOAL_DTYPE, count=1)[0] = (
                True,
                command.run_id,
                command.arm_hold,
                expires_ns,
                command.arm_qpos if command.arm_qpos is not None else 0.0,
            )
        finally:
            self.goal_lock.release()

    def take_goal(self):
        if not self.goal_lock.acquire(False):
            return None  # 父进程持锁时跳过本次读取，不阻塞伺服。
        try:
            slot = np.frombuffer(self.goal, dtype=GOAL_DTYPE, count=1)
            if not slot["pending"][0]:
                return None
            goal = slot[0].copy()
            slot["pending"][0] = False
            return goal
        finally:
            self.goal_lock.release()

    def read_status(self, *, timeout_s):
        if not self.status_lock.acquire(timeout=timeout_s):
            raise TimeoutError("arm feedback snapshot lock unavailable")
        try:
            return np.frombuffer(self.status, dtype=ARM_STATE_DTYPE, count=1)[0].copy()
        finally:
            self.status_lock.release()

    def publish_status(self, status):
        # 发布竞争不能阻塞 owner 的停止或伺服。
        if not self.status_lock.acquire(False):
            return
        try:
            np.frombuffer(self.status, dtype=ARM_STATE_DTYPE, count=1)[0] = status
        finally:
            self.status_lock.release()
