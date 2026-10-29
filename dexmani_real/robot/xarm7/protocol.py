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
        ("submit_until_ns", "<i8"),
        ("execute_until_ns", "<i8"),
        ("q_goal", "<f8", (7,)),
    ]
)
STATUS_DTYPE = np.dtype(
    [
        ("frame", ARM_STATE_DTYPE),
        ("valid", "?"),
        ("state", "<i4"),
        ("prepared_run_id", "<i8"),
    ]
)


@dataclass(frozen=True)
class PrepareRequest:
    run_id: int


@dataclass(frozen=True)
class HomeRequest:
    run_id: int
    waypoints: np.ndarray
    target: np.ndarray
    deadline_ns: int


class ArmChannels:
    def __init__(
        self, ctx, *, run_signal, stop_signal, emergency_signal, signal_lock
    ):
        self.run_id, self.stop, self.emergency = run_signal, stop_signal, emergency_signal
        self.signal_lock = signal_lock
        self.closing, self.stopped, self.ready = (ctx.Event() for _ in range(3))
        self.connected = ctx.Event()
        self.fault = ctx.Array("c", 1024, lock=False)
        self.close_error = ctx.Array("c", 1024, lock=False)
        self.goal_lock, self.status_lock = ctx.Lock(), ctx.Lock()
        self.goal = ctx.RawArray("b", GOAL_DTYPE.itemsize)
        self.status = ctx.RawArray("b", STATUS_DTYPE.itemsize)

    def publish_goal(self, command, *, submit_until_ns, execute_until_ns):
        prepared_run_id = self.read_status()["prepared_run_id"]
        with self.goal_lock:
            # IPC 等待之后重新检查运行身份和停止信号。
            if (
                self.closing.is_set()
                or self.stop.is_set()
                or self.emergency.is_set()
                or self.run_id.value != command.run_id
            ):
                raise OperationCancelled("arm target run cancelled")
            if time.monotonic_ns() >= min(submit_until_ns, execute_until_ns):
                raise TimeoutError("arm target deadline expired")
            if prepared_run_id != command.run_id:
                raise RuntimeError("arm target run was not prepared")
            np.frombuffer(self.goal, dtype=GOAL_DTYPE, count=1)[0] = (
                True,
                command.run_id,
                command.arm_hold,
                submit_until_ns,
                execute_until_ns,
                command.arm_qpos if command.arm_qpos is not None else 0.0,
            )

    def take_goal(self, *, block=False):
        if not self.goal_lock.acquire(block):
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

    def read_status(self):
        with self.status_lock:
            return np.frombuffer(self.status, dtype=STATUS_DTYPE, count=1)[0].copy()

    def publish_status(self, status):
        # 锁内只复制快照，不执行 SDK I/O。
        with self.status_lock:
            np.frombuffer(self.status, dtype=STATUS_DTYPE, count=1)[0] = status
