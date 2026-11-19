"""传感进程启停与会话资源回收。"""

import logging
import signal
import time

from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.runtime.state import SessionState

logger = logging.getLogger(__name__)

CAMERA_READY_TIMEOUT_S = 15.0
POINTCLOUD_READY_TIMEOUT_S = 30.0
VR_READY_TIMEOUT_S = 120.0
# L515 RGB-only 在首帧后立即关闭时，SDK stop 实测约 3.2 s；此预算独立于来源 TTL 和录制收尾。
SENSOR_SHUTDOWN_TIMEOUT_S = 5.0
WORKER_STOP_TIMEOUT_S = 1.0


class SensorWorkers:
    def __init__(self, channels=None):
        self.channels = channels
        self.processes = []
        self._reported_exits = set()

    def start(self, process, *, ring, timeout_s=None, may_start) -> None:
        """传入超时时等待就绪，否则仅启动；失败后仍由本对象回收进程。"""
        allowed = may_start()
        self.check()
        if not allowed:
            raise OperationCancelled("sensor startup cancelled")
        self.processes.append(process)
        process.start()
        if timeout_s is not None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                allowed = may_start()
                sequence = ring.latest_sequence
                self.check()
                if not allowed:
                    raise OperationCancelled(f"{process.name} startup cancelled")
                if sequence > 0:
                    allowed = may_start()
                    self.check()
                    if not allowed:
                        raise OperationCancelled(f"{process.name} startup cancelled")
                    if time.monotonic() >= deadline:
                        break
                    return
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            raise RuntimeError(f"{process.name} failed to become ready")

    def check(self) -> None:
        for process in self.processes:
            exitcode = process.exitcode
            if exitcode is not None:
                self._reported_exits.add(process)
                raise RobotError(f"sensor worker {process.name} exited: exitcode={exitcode}")

    def shutdown(self, runtime_state: SessionState) -> None:
        """短时退出，必要时终止 sensor；仅保留实际未回收的句柄。"""
        if self.channels is not None:
            self.channels.is_running.value = False
        deadline = time.monotonic() + SENSOR_SHUTDOWN_TIMEOUT_S
        errors: list[tuple[str, BaseException]] = []
        unresolved = []

        def join_worker(process, timeout):
            try:
                process.join(timeout=timeout)
            except KeyboardInterrupt as exc:
                # 等待被打断仍要查看进程状态，并在必要时升级停止信号。
                runtime_state.request_emergency_stop()
                errors.append((process.name, exc))

        for process in self.processes:
            try:
                if process.pid is None:
                    continue
                join_worker(process, max(0.0, deadline - time.monotonic()))
                terminated = killed = False
                if process.is_alive():
                    process.terminate()
                    terminated = True
                    join_worker(process, WORKER_STOP_TIMEOUT_S)
                if process.is_alive():
                    process.kill()
                    killed = True
                    join_worker(process, WORKER_STOP_TIMEOUT_S)
                exitcode = process.exitcode
                if exitcode is None:
                    raise RuntimeError(f"process {process.name} could not be confirmed stopped")
                if exitcode == 0 or process in self._reported_exits:
                    continue
                # 升级 kill 后，退出仍可能由先前尝试的 SIGTERM 引起。
                if not (
                    terminated and exitcode == -signal.SIGTERM
                    or killed and exitcode == -signal.SIGKILL
                ):
                    raise RuntimeError(f"worker {process.name}: exitcode={exitcode}")
                logger.warning("worker %s terminated during shutdown: exitcode=%s", process.name, exitcode)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    runtime_state.request_emergency_stop()
                errors.append((process.name, exc))
                if process.pid is not None and process.exitcode is None:
                    unresolved.append(process)
        self.processes = unresolved
        self._reported_exits.clear()
        if errors:
            for stage, error in errors[1:]:
                logger.error("worker %s shutdown failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise errors[0][1]


def close_runtime(
    runtime_state: SessionState, robot=None, *, sensors=None, keyboard=None, release_hand: bool = False,
) -> None:
    """停止后尝试全部关闭；即时锁存中断，首错传播，其余错误记录 traceback。"""
    runtime_state.stop_signal.set()
    errors = []
    for stage, close in (
        ("robot_close", (lambda: robot.close(release_hand=release_hand)) if robot is not None else None),
        ("keyboard_stop", keyboard.stop if keyboard is not None else None),
        ("sensor_shutdown", (lambda: sensors.shutdown(runtime_state)) if sensors is not None else None),
    ):
        if close is None:
            continue
        try:
            close()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                runtime_state.request_emergency_stop()
            errors.append((stage, exc))
    if errors:
        for stage, error in errors[1:]:
            logger.error("%s failed", stage, exc_info=(type(error), error, error.__traceback__))
        raise errors[0][1]
