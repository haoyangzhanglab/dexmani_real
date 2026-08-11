"""Narrow owner for spawn, verified shutdown, and shared-resource cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.runtime.processes import ShutdownReport, shutdown_processes_verified
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class ManagedProcessGroup:
    """Own one already-constructed process group and its SharedStorage.

    Domain modules still choose workers, readiness rules, safety transitions,
    and supervision. This class only removes repeated start/teardown shells.
    """

    shared: Any
    processes: list[Any]
    graceful_timeout_s: float
    _started: bool = field(init=False, default=False)
    _started_processes: list[Any] = field(init=False, default_factory=list)
    _shutdown_report: ShutdownReport | None = field(init=False, default=None)

    def __enter__(self) -> "ManagedProcessGroup":
        return self

    def start(self) -> None:
        if self._started:
            raise RuntimeError("process group has already been started")
        self._started = True
        for process in self.processes:
            process.start()
            self._started_processes.append(process)

    def shutdown(self, *, disarm_if_clean: bool = False) -> ShutdownReport:
        if self._shutdown_report is not None:
            return self._shutdown_report
        if not self._started_processes:
            self.shared.is_running.value = False
            closed = bool(self.shared.close())
            self._shutdown_report = ShutdownReport((), shared_closed=closed)
            return self._shutdown_report
        self._shutdown_report = shutdown_processes_verified(
            self.shared,
            self._started_processes,
            graceful_timeout_s=self.graceful_timeout_s,
            disarm_if_clean=disarm_if_clean,
        )
        return self._shutdown_report

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> Literal[False]:
        del exc_type, traceback
        if exc is not None:
            self.shared.error_state.value = True
            transition(self.shared, SafetyState.FAULT)
        if self._shutdown_report is None:
            try:
                self.shutdown()
            except Exception:
                logger.critical("managed process group cleanup failed", exc_info=True)
                raise
        return False
