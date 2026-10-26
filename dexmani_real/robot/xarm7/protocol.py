"""Fixed arm channels and low-rate lifecycle messages shared by parent and child."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE

ARM_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", ARM_JOINT_SHAPE),
        ("qvel", "<f8", ARM_JOINT_SHAPE),
        ("timestamp_ns", "<u8"),
    ]
)

MAX_TIME = 2**63 - 1
GOAL_DTYPE = np.dtype(
    [
        ("sequence", "<i8"),
        ("run_id", "<i8"),
        ("input_generation", "<i8"),
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


@dataclass(frozen=True)
class PrepareRequest:
    operation_id: int
    run_id: int


@dataclass(frozen=True)
class HomeRequest:
    operation_id: int
    run_id: int
    waypoints: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class LifecycleReply:
    operation_id: int
    status: OperationStatus
    detail: str = ""
