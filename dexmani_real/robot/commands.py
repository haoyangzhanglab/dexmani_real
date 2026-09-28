"""Owned absolute targets for direct SDK dispatch."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotCommand:
    run_id: int
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None

    def __post_init__(self):
        if self.arm_qpos is None and self.hand_qpos is None:
            raise ValueError("command requires an actuator target")
        for name, size in (("arm_qpos", 7), ("hand_qpos", 12)):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=np.float64, copy=True)
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError(f"{name} must be finite shape ({size},)")
                value.flags.writeable = False
                object.__setattr__(self, name, value)
