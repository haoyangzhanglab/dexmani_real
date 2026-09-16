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
    # A service (non-critical, e.g. recorder/recording-only camera) process
    # died or timed out its heartbeat. Exits via the normal verified-shutdown
    # path, not SafetyState.FAULT; the session is marked session_failed.
    SERVICE_FAILURE = 6
