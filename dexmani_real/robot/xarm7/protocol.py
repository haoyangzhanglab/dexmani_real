"""Fixed arm channels and low-rate lifecycle messages shared by parent and child."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.ipc.schema import ARM_STATE_DTYPE

SDK_CODE_UNAVAILABLE = -999
MAX_TIME = 2**63 - 1
GOAL_DTYPE = np.dtype(
    [
        ("sequence", "<i8"),
        ("run_id", "<i8"),
        ("submit_until_ns", "<i8"),
        ("execute_until_ns", "<i8"),
        ("q_goal", "<f8", (7,)),
    ]
)
CALL_DTYPE = np.dtype(
    [
        ("run_id", "<i8"),
        ("command_seq", "<i8"),
        ("planned_ns", "<i8"),
        ("sampled_ns", "<i8"),
        ("started_ns", "<i8"),
        ("completed_ns", "<i8"),
        ("status", "<i4"),
        ("code", "<i4"),
        ("q_ref", "<f8", (7,)),
    ]
)
COUNTERS = (
    "first_accepted",
    "sdk_attempted",
    "sdk_completed",
    "sdk_accepted",
    "cancelled_before_first",
)
STAT_NAMES = ("max_lateness_ns", "max_call_ns", "max_interval_ns")
STATUS_DTYPE = np.dtype(
    [
        ("frame", ARM_STATE_DTYPE),
        ("valid", "?"),
        ("state", "<i4"),
        ("prepared_run_id", "<i8"),
        ("active_sequence", "<i8"),
        ("stop_started_ns", "<i8"),
        ("stop_completed_ns", "<i8"),
        ("stop_accepted", "?"),
        *[(name, "<i8") for name in COUNTERS],
        ("point", CALL_DTYPE),
        ("stats", [(name, "<i8") for name in STAT_NAMES]),
    ]
)


class OperationStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OperationCancelled(RuntimeError):
    """Lifecycle authority was revoked; independent of any vendor driver."""


@dataclass(frozen=True)
class PrepareRequest:
    operation_id: int
    epoch: int


@dataclass(frozen=True)
class HomeRequest:
    operation_id: int
    epoch: int
    waypoints: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class LifecycleReply:
    operation_id: int
    status: OperationStatus
    detail: str = ""
