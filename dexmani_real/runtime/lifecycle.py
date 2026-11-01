"""传感进程启停与会话资源回收。"""

import logging
import time

from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.runtime.state import RunEndReason, SafetyState
from dexmani_real.utils.log import attach_cleanup_errors

logger = logging.getLogger(__name__)

CAMERA_READY_TIMEOUT_S = 15.0
POINTCLOUD_READY_TIMEOUT_S = 30.0
VR_READY_TIMEOUT_S = 120.0
# 模型任务与持久化收尾的会话上限，传感器不使用此预算。
SHUTDOWN_TIMEOUT_S = 65.0
# 相机 read <= 0.3 s、来源 stall 2 s；不与 writer 的 flush 预算共享。
SENSOR_SHUTDOWN_TIMEOUT_S = 3.0
WORKER_STOP_TIMEOUT_S = 1.0


class SensorWorkers:
    def __init__(self, channels=None):
        self.channels = channels
        self.started = []

    def start(self, process, *, ring, timeout_s=None, may_start) -> None:
        """传入超时时等待就绪，否则仅启动；失败后仍由本对象回收进程。"""
        allowed = may_start()
        if not self.check():
            raise RuntimeError("required sensor worker exited before startup")
        if not allowed:
            raise OperationCancelled("sensor startup cancelled")
        process.start()
        self.started.append(process)
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
        for process in self.started:
            if (names is None or process.name in names) and not process.is_alive():
                logger.error("required worker %s exited: %s", process.name, process.exitcode)
                healthy = False
        return healthy

    def shutdown(self):
        """短时退出，必要时终止 sensor；仅保留实际未回收的句柄。"""
        if self.channels is not None:
            self.channels.is_running.value = False
        deadline = time.monotonic() + SENSOR_SHUTDOWN_TIMEOUT_S
        errors: list[BaseException] = []
        unresolved = []
        for process in self.started:
            try:
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
                if process.is_alive() or process.exitcode is None:
                    raise RuntimeError(f"process {process.name} could not be confirmed stopped")
                if process.exitcode != 0 or escalation != "graceful":
                    errors.append(RuntimeError(f"worker {process.name}: exitcode={process.exitcode}, {escalation}"))
                    logger.warning(
                        "worker %s stopped: exitcode=%s escalation=%s",
                        process.name,
                        process.exitcode,
                        escalation,
                    )
            except BaseException as exc:
                unresolved.append(process)
                errors.append(exc)
                logger.exception("worker %s shutdown failed", process.name)
        self.started = unresolved
        if errors:
            attach_cleanup_errors(errors[0], (("sensor_shutdown", error) for error in errors[1:]))
            raise errors[0]


def close_runtime(runtime_state, robot=None, *, sensors=None, keyboard=None):
    """撤权后尝试全部关闭；正常返回或保留首个异常及其补充关闭错误。"""
    runtime_state.stop_signal.set()
    errors = []
    for stage, close in (
        ("revoke_motion", lambda: runtime_state.revoke_motion(
            SafetyState.DISARMED, reason=RunEndReason.RUNTIME_SHUTDOWN,
        )),
        ("robot_close", robot.close if robot is not None else None),
        ("keyboard_stop", keyboard.stop if keyboard is not None else None),
        ("sensor_shutdown", sensors.shutdown if sensors is not None else None),
    ):
        if close is None:
            continue
        try:
            close()
        except BaseException as exc:
            errors.append((stage, exc))
            logger.exception("%s failed", stage)
    if errors:
        first = errors[0][1]
        attach_cleanup_errors(first, errors[1:])
        raise first
