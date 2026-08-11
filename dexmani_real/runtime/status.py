"""Structured health/status vocabulary shared by supervisors and workers."""

from __future__ import annotations

from enum import IntEnum


class ComponentPhase(IntEnum):
    INIT = 0
    LOADING = 1
    WARMING_UP = 2
    READY = 3
    RUNNING = 4
    STOPPING = 5
    STOPPED = 6
    FAULT = 7


class FaultCode(IntEnum):
    NONE = 0
    CONFIG_INVALID = 1
    STARTUP_FAILED = 2
    HEARTBEAT_TIMEOUT = 3
    WORKER_DIED = 4
    DEVICE_IO = 5
    SDK_REJECTED = 6
    COMMAND_INVALID = 7
    COMMAND_EXPIRED = 8
    PREPARE_TIMEOUT = 9
    ACTUATOR_MISMATCH = 10
    INFERENCE_FAILED = 11
    INFERENCE_TIMEOUT = 12
    RECORDING_ABORTED = 13
    CAMERA_INVALID = 14
    ESTOP = 15


class ExitReason(IntEnum):
    NONE = 0
    EXPLICIT_QUIT = 1
    ESTOP = 2
    STICKY_FAULT = 3
    WORKER_DEATH = 4
    HEARTBEAT_TIMEOUT = 5
    STARTUP_FAILURE = 6
    SUPERVISOR_SHUTDOWN = 7
    KILLED = 8
