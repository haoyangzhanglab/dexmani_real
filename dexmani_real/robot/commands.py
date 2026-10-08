"""Owned absolute targets for direct SDK dispatch."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class DispatchStatus(IntEnum):
    NOT_CALLED = 0
    ACCEPTED = 1
    CRC_UNCONFIRMED = 2
    REJECTED = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class RobotCommand:
    run_id: int
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None

    def __post_init__(self):
        if self.arm_qpos is None and self.hand_qpos is None:
            raise ValueError("command requires an actuator target")
        for name in ("arm_qpos", "hand_qpos"):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=np.float64, copy=True)
                value.flags.writeable = False
                object.__setattr__(self, name, value)
