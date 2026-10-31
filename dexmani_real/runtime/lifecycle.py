"""传感进程启停与会话资源回收。"""

import logging
import time

from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.runtime.state import RunEndReason, SafetyState
from dexmani_real.utils.log import error_detail

logger = logging.getLogger(__name__)

CAMERA_READY_TIMEOUT_S = 15.0
POINTCLOUD_READY_TIMEOUT_S = 30.0
VR_READY_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 65.0
WORKER_STOP_TIMEOUT_S = 1.0


class SensorWorkers:
    def __init__(self, channels=None):
        self.channels = channels
        self.started = []

    def request_shutdown(self):
        if self.channels is not None:
            self.channels.is_running.value = False

    def start(self, process, *, ready, timeout_s=None, may_start) -> None:
        """传入超时时等待就绪，否则仅启动；失败后仍由本对象回收进程。"""
        if not may_start():
            raise OperationCancelled("sensor startup cancelled")
        process.start()
        self.started.append((process, ready))
        if timeout_s is not None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if not may_start():
                    raise OperationCancelled(f"{process.name} startup cancelled")
                if not self.check():
                    raise RuntimeError(f"{process.name} exited during startup")
                if ready.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic()))):
                    if not may_start():
                        raise OperationCancelled(f"{process.name} startup cancelled")
                    if not self.check():
                        raise RuntimeError(f"{process.name} exited during startup")
                    return
            raise RuntimeError(f"{process.name} failed to become ready")

    def check(self, *, names=None) -> bool:
        healthy = True
        for process, ready in self.started:
            if (names is None or process.name in names) and not process.is_alive():
                ready.clear()
                logger.error("required worker %s exited: %s", process.name, process.exitcode)
                healthy = False
        return healthy

    def shutdown(self, *, deadline=None):
        """共享正常退出期限，超时后逐个终止；仅保留回收失败的句柄。"""
        self.request_shutdown()
        if deadline is None:
            deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        clean = True
        errors = []
        unresolved = []
        for process, ready in self.started:
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
                ready.clear()
                if process.exitcode != 0 or escalation != "graceful":
                    clean = False
                    logger.warning(
                        "worker %s stopped: exitcode=%s escalation=%s",
                        process.name,
                        process.exitcode,
                        escalation,
                    )
            except BaseException as exc:
                unresolved.append((process, ready))
                errors.append(exc)
                logger.exception("worker %s shutdown failed", process.name)
        self.started = unresolved
        if errors:
            raise errors[0]
        return clean


def close_runtime(
    runtime_state, robot=None, *, sensors=None, keyboard=None, deadline=None, errors=None,
):
    """先撤销运动权限，再关闭设备、输入和传感进程。"""
    if deadline is None:
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    runtime_state.revoke_motion(
        SafetyState.DISARMED,
        reason=RunEndReason.RUNTIME_SHUTDOWN,
    )
    for stage, close in (
        ("robot_close", robot.close if robot is not None else None),
        ("keyboard_stop", keyboard.stop if keyboard is not None else None),
    ):
        if close is None:
            continue
        try:
            close()
        except BaseException as exc:
            if errors is not None:
                errors.append(error_detail(stage, exc))
            if isinstance(exc, KeyboardInterrupt):
                runtime_state.request_emergency_stop()
            runtime_state.mark_error()
            logger.exception("local runtime cleanup failed")
    runtime_state.end_session()
    sensors_clean = True
    if sensors is not None:
        try:
            sensors_clean = sensors.shutdown(deadline=deadline)
        except BaseException as exc:
            if errors is not None:
                errors.append(error_detail("sensor_shutdown", exc))
            if isinstance(exc, KeyboardInterrupt):
                runtime_state.request_emergency_stop()
            runtime_state.mark_error()
            logger.exception("sensor shutdown failed; unresolved process handles retained")
            sensors_clean = False
    clean = (
        sensors_clean
        and not runtime_state.error_state
        and not runtime_state.estop_request
    )
    if not clean and not runtime_state.estop_request:
        runtime_state.mark_error()
    # DISARMED 只表示软件权限已撤销，不证明机器人已物理停止。
    runtime_state.revoke_motion(SafetyState.DISARMED)
    return clean
