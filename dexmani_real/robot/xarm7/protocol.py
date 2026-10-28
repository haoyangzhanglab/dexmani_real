"""机械臂共享通道、反馈布局与低频操作消息。"""

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

MAX_TIME = 2**63 - 1
GOAL_DTYPE = np.dtype(
    [
        ("sequence", "<i8"),
        ("run_id", "<i8"),
        ("input_generation", "<i8"),
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
    operation_id: int
    run_id: int


@dataclass(frozen=True)
class HomeRequest:
    operation_id: int
    run_id: int
    waypoints: np.ndarray
    target: np.ndarray
    deadline_ns: int


@dataclass(frozen=True)
class LifecycleReply:
    operation_id: int
    error: BaseException | None = None


class ArmChannels:
    def __init__(
        self, ctx, *, run_signal, stop_signal, emergency_signal, signal_lock, input_signal=None
    ):
        self.run_id, self.stop, self.emergency = run_signal, stop_signal, emergency_signal
        self.signal_lock = signal_lock
        self.input_generation = input_signal if input_signal is not None else ctx.Value("q", -1)
        self.closing, self.stopped, self.ready = (ctx.Event() for _ in range(3))
        self.connected = ctx.Event()
        self.fault = ctx.Array("c", 1024, lock=False)
        self.close_error = ctx.Array("c", 1024, lock=False)
        self.health_until = ctx.Value("q", 0)
        self.execution_limit = ctx.Array("q", [MAX_TIME, 0])  # 截止时间、最后受影响的命令序号
        self.goal_lock, self.status_lock = ctx.Lock(), ctx.Lock()
        self.goal = ctx.RawArray("b", GOAL_DTYPE.itemsize)
        self.status = ctx.RawArray("b", STATUS_DTYPE.itemsize)

    def publish_goal(self, command, *, sequence, submit_until_ns, execute_until_ns):
        prepared_run_id = self.read_status()["prepared_run_id"]
        goal = np.zeros((), dtype=GOAL_DTYPE)
        goal[()] = (
            sequence,
            command.run_id,
            command.input_generation,
            submit_until_ns,
            execute_until_ns,
            command.arm_qpos,
        )
        with self.goal_lock:
            # IPC 等待之后重新检查运行身份、停止信号和 jog 输入代际。
            if (
                self.closing.is_set()
                or self.stop.is_set()
                or self.emergency.is_set()
                or self.run_id.value != command.run_id
            ):
                raise OperationCancelled("arm target run cancelled")
            if (
                command.input_generation >= 0
                and self.input_generation.value != command.input_generation
            ):
                raise OperationCancelled("jog input changed before publication")
            if time.monotonic_ns() >= min(submit_until_ns, execute_until_ns):
                raise TimeoutError("arm target deadline expired")
            if prepared_run_id != command.run_id:
                raise RuntimeError("arm target run was not prepared")
            slot = np.frombuffer(self.goal, dtype=GOAL_DTYPE, count=1)
            if slot["sequence"][0]:
                raise RuntimeError("arm target slot is full")
            slot[0] = goal

    def take_goal(self, *, block=False):
        if not self.goal_lock.acquire(block):
            return None  # 父进程持锁时跳过本次读取，不阻塞伺服。
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

    def publish_status(self, status):
        # 锁内只复制快照，不执行 SDK I/O。
        with self.status_lock:
            np.frombuffer(self.status, dtype=STATUS_DTYPE, count=1)[0] = status
