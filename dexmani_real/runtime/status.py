"""Structured health/status vocabulary shared by supervisors and workers."""

from __future__ import annotations

from enum import IntEnum


class ExitReason(IntEnum):
    NONE = 0
    EXPLICIT_QUIT = 1
    ESTOP = 2
    STICKY_FAULT = 3
    WORKER_DEATH = 4
    HEARTBEAT_TIMEOUT = 5
    # A non-FAULT session/service failure: an owning workflow has latched
    # session_failed, or a configured service process died/timed out.
    # Supervision uses the verified non-FAULT shutdown path.
    SERVICE_FAILURE = 6
