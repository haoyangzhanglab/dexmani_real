"""传感进程启停与会话资源回收。"""

import logging
import time

from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState

logger = logging.getLogger(__name__)

CAMERA_READY_TIMEOUT_S = 15.0
POINTCLOUD_READY_TIMEOUT_S = 30.0
VR_READY_TIMEOUT_S = 120.0
# 相机 read <= 0.3 s、来源 stall 2 s；不与 writer 的 flush 预算共享。
SENSOR_SHUTDOWN_TIMEOUT_S = 3.0
WORKER_STOP_TIMEOUT_S = 1.0


class SensorWorkers:
    def __init__(self, channels=None, *, required_names):
        self.channels = channels
        self.required_names = required_names
        self.processes = []

    def start(self, process, *, ring, timeout_s=None, may_start) -> None:
        """传入超时时等待就绪，否则仅启动；失败后仍由本对象回收进程。"""
        allowed = may_start()
        if not self.check():
            raise RuntimeError("required sensor worker exited before startup")
        if not allowed:
            raise OperationCancelled("sensor startup cancelled")
        self.processes.append(process)
        process.start()
        if timeout_s is not None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                allowed = may_start()
                sequence = ring.latest_sequence
                if not self.check():
                    raise RuntimeError(f"{process.name} exited during startup")
                if not allowed:
                    raise OperationCancelled(f"{process.name} startup cancelled")
                if sequence > 0:
                    return
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            raise RuntimeError(f"{process.name} failed to become ready")

    def check(self, *, names=None) -> bool:
        healthy = True
        for process in self.processes:
            if (names is None or process.name in names) and not process.is_alive():
                logger.error("required worker %s exited: %s", process.name, process.exitcode)
                healthy = False
        return healthy

    def shutdown(self, runtime_state: SessionState) -> None:
        """短时退出，必要时终止 sensor；仅保留实际未回收的句柄。"""
        if self.channels is not None:
            self.channels.is_running.value = False
        deadline = time.monotonic() + SENSOR_SHUTDOWN_TIMEOUT_S
        errors: list[tuple[str, BaseException]] = []
        unresolved = []
        for process in self.processes:
            try:
                if process.pid is None:
                    continue
                process.join(timeout=max(0.0, deadline - time.monotonic()))
                escalation = "graceful"
                if process.is_alive():
                    escalation = "terminate"
                    process.terminate()
                    process.join(timeout=WORKER_STOP_TIMEOUT_S)
                if process.is_alive():
                    escalation = "kill"
                    process.kill()
                    process.join(timeout=WORKER_STOP_TIMEOUT_S)
                if process.exitcode is None:
                    raise RuntimeError(f"process {process.name} could not be confirmed stopped")
                if escalation != "graceful" or (
                    process.exitcode != 0 and process.name in self.required_names
                ):
                    errors.append((process.name, RuntimeError(f"worker {process.name}: exitcode={process.exitcode}, {escalation}")))
                    logger.warning(
                        "worker %s stopped: exitcode=%s escalation=%s",
                        process.name,
                        process.exitcode,
                        escalation,
                    )
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    runtime_state.request_emergency_stop()
                errors.append((process.name, exc))
                try:
                    pending = process.pid is not None and process.exitcode is None
                except BaseException as status_error:
                    if isinstance(status_error, KeyboardInterrupt):
                        runtime_state.request_emergency_stop()
                    errors.append((process.name, status_error))
                    pending = True
                if pending:
                    unresolved.append(process)
        self.processes = unresolved
        if errors:
            for stage, error in errors[1:]:
                logger.error("worker %s shutdown failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise errors[0][1]


def close_runtime(runtime_state: SessionState, robot=None, *, sensors=None, keyboard=None) -> None:
    """撤权后尝试全部关闭；即时锁存中断，首错传播，其余错误记录 traceback。"""
    runtime_state.stop_signal.set()
    errors = []
    for stage, close in (
        ("revoke_motion", lambda: runtime_state.revoke_motion(
            SafetyState.DISARMED, reason=RunEndReason.CANCELLED,
        )),
        ("robot_close", robot.close if robot is not None else None),
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
