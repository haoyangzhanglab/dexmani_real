"""Owned absolute targets for arm submission and hand SDK dispatch."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class DispatchStatus(IntEnum):
    """Dispatch facts. SUBMITTED is a host goal, ACCEPTED is an actual SDK result."""

    NOT_CALLED = 0
    ACCEPTED = 1
    CRC_UNCONFIRMED = 2
    REJECTED = 3
    UNKNOWN = 4
    SUBMITTED = 5


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


@dataclass(frozen=True)
class DispatchResult:
    arm: DispatchStatus = DispatchStatus.NOT_CALLED
    hand: DispatchStatus = DispatchStatus.NOT_CALLED
    timestamp_ns: int = 0  # host dispatch completion, never arrival
    command_id: tuple[int, int] = (-1, -1)


class DispatchError(RuntimeError):
    def __init__(self, message, result, *, revoked=False, cause=None):
        super().__init__(message)
        self.result = result
        self.revoked = revoked
        self.cause = cause


class DispatchInterrupted(KeyboardInterrupt):
    """Cancellation evidence local to one attempted dispatch."""

    def __init__(self, result):
        super().__init__("dispatch interrupted")
        self.result = result


def check_arm_target(
    target_qpos_rad: np.ndarray,
    *,
    joint_limit_lower_rad: np.ndarray,
    joint_limit_upper_rad: np.ndarray,
) -> str | None:
    """Return why an arm target cannot cross the SDK boundary (hard limits)."""
    target = np.asarray(target_qpos_rad, dtype=np.float64)
    if target.shape != (7,):
        return "invalid target shape"
    if not np.all(np.isfinite(target)):
        return "non-finite target"
    if np.any(target < joint_limit_lower_rad) or np.any(target > joint_limit_upper_rad):
        return "joint limit violation"
    return None


def check_hand_target(
    target_qpos_rad: np.ndarray,
    *,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
) -> str | None:
    """Return why a hand target cannot cross the SDK boundary."""
    target = np.asarray(target_qpos_rad, dtype=np.float64)
    if target.shape != (12,):
        return "invalid target shape"
    if not np.all(np.isfinite(target)):
        return "non-finite target"
    if np.any(target < mechanical_lower_rad - 1e-12) or np.any(
        target > mechanical_upper_rad + 1e-12
    ):
        return "mechanical joint limit violation"
    return None
