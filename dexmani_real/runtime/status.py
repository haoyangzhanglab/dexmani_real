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
    # Terminal session failure; EVIDENCE_FAILURE allows teleop H/Q recovery.
    SERVICE_FAILURE = 6
    EVIDENCE_FAILURE = 7
