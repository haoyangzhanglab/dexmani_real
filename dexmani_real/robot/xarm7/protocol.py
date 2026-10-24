"""Fixed arm channels and low-rate lifecycle messages shared by parent and child."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.robot.feedback import ARM_STATE_DTYPE

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
STATUS_DTYPE = np.dtype(
    [
        ("frame", ARM_STATE_DTYPE),
        ("valid", "?"),
        ("state", "<i4"),
        ("prepared_run_id", "<i8"),
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
