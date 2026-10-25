"""传感进程的启动所有权、就绪检查和有界退出。"""

import logging
import math
import time

logger = logging.getLogger(__name__)


class SensorProcesses:
    """Own started sensor processes; workflows choose devices and start order."""

    def __init__(self, channels=None):
        self.channels = channels
        self.started = []

    def request_shutdown(self):
        """Notify sensor producers after the coordinator revokes motion; never join here."""
        if self.channels is not None:
            self.channels.is_running.value = False

    def start(self, process, *, ready, timeout_s=None, may_start) -> None:
        """Start ownership immediately; a timeout also requests a blocking readiness wait."""
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("readiness timeout must be finite and positive")

        if not may_start():
            raise RuntimeError("sensor startup cancelled")
        process.start()
        self.started.append((process, ready))
        if timeout_s is not None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if not may_start() or not self.check():
                    break
                if ready.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic()))):
                    if may_start() and self.check():
                        return
                    break
            raise RuntimeError(f"{process.name} failed to become ready")

    def check(self, *, names=None) -> bool:
        dead = [(process, ready) for process, ready in self.started
                if (names is None or process.name in names) and not process.is_alive()]
        if not dead:
            return True
        for process, ready in dead:
            ready.clear()
            logger.error("required worker %s exited: %s", process.name, process.exitcode)
        return False

    def shutdown(self, *, graceful_timeout_s=5.0, terminate_timeout_s=1.0, kill_timeout_s=1.0):
        """Verify every child exited; retain handles if any exit remains unresolved."""
        self.request_shutdown()
        deadline = time.monotonic() + graceful_timeout_s
        clean = True
        errors = []
        for process, ready in self.started:
            try:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
                escalation = "graceful"
                if process.is_alive():
                    escalation = "terminate"
                    process.terminate()
                    process.join(timeout=terminate_timeout_s)
                if process.is_alive():
                    escalation = "kill"
                    process.kill()
                    process.join(timeout=kill_timeout_s)
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
            except Exception as exc:
                errors.append(exc)
                logger.exception("worker %s shutdown failed", process.name)
        if errors:
            raise RuntimeError(f"sensor shutdown unresolved: {errors}") from errors[0]
        self.started.clear()
        return clean
