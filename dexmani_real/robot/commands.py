from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE


@dataclass(frozen=True)
class RobotCommand:
    run_id: int
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    arm_hold: bool = False  # 从当前伺服参考减速到零并保持，不退出 Mode 1。

    def __post_init__(self):
        if self.arm_qpos is None and self.hand_qpos is None and not self.arm_hold:
            raise ValueError("command requires an actuator target")
        if self.arm_hold and self.arm_qpos is not None:
            raise ValueError("arm hold cannot carry a position target")
        for name in ("arm_qpos", "hand_qpos"):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=np.float64, copy=True)
                value.flags.writeable = False
                object.__setattr__(self, name, value)


class OperationCancelled(RuntimeError):
    """本次运行已撤销。"""


class RobotError(RuntimeError):
    """设备或执行器失败，调用方须终止本次运行。"""


def validate_arm_target(qpos, lower, upper) -> np.ndarray:
    target = np.asarray(qpos, dtype=np.float64)
    if target.shape != ARM_JOINT_SHAPE:
        raise ValueError("arm target must have shape (7,)")
    if not np.all(np.isfinite(target)):
        raise ValueError("arm target must be finite")
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError("arm target violates joint limits")
    return target


def validate_hand_target(qpos, lower, upper) -> np.ndarray:
    target = np.asarray(qpos, dtype=np.float64)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if target.shape != HAND_JOINT_SHAPE:
        raise ValueError("hand target must have shape (12,)")
    if not np.all(np.isfinite(target)):
        raise ValueError("hand target must be finite")
    if np.any(target < lower - 1e-12) or np.any(target > upper + 1e-12):
        raise ValueError("hand target violates joint limits")
    return target
