from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotCommand:
    run_id: int
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    input_generation: int = -1

    def __post_init__(self):
        if self.arm_qpos is None and self.hand_qpos is None:
            raise ValueError("command requires an actuator target")
        for name in ("arm_qpos", "hand_qpos"):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=np.float64, copy=True)
                value.flags.writeable = False
                object.__setattr__(self, name, value)


class OperationCancelled(RuntimeError):
    """本次运行或输入已撤销。"""


class RobotError(RuntimeError):
    """设备或执行器失败，调用方须终止本次运行。"""


def check_arm_target(
    target_qpos_rad: np.ndarray,
    *,
    joint_limit_lower_rad: np.ndarray,
    joint_limit_upper_rad: np.ndarray,
) -> str | None:
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
