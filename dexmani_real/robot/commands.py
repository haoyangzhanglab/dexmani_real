"""Latest absolute targets; lifecycle authority is checked at publication and SDK IO."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.ipc.schema import ROBOT_COMMAND_DTYPE
from dexmani_real.runtime.safety import SafetyState, command_may_cross_sdk


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


def read_robot_command(shared: Any) -> tuple[RobotCommand, int] | None:
    """Return the latest target and a worker-private transport sequence."""
    result = shared.robot_command_ring.read_latest()
    if result is None:
        return None
    record = result[0][0]
    return RobotCommand(
        run_id=int(record["run_id"]),
        arm_qpos=record["arm_qpos"] if record["arm_present"] else None,
        hand_qpos=record["hand_qpos"] if record["hand_present"] else None,
    ), result[2]


def publish_command(shared: Any, target: RobotCommand) -> int:
    """Return the host publication time, or zero when the motion epoch was revoked."""
    with shared.motion_lock:
        if not command_may_cross_sdk(
            shared, run_id=target.run_id, required_safety_state=SafetyState.RUNNING
        ):
            return 0
        frame = np.zeros(1, dtype=ROBOT_COMMAND_DTYPE)
        frame["run_id"] = target.run_id
        for name in ("arm", "hand"):
            value = getattr(target, f"{name}_qpos")
            frame[f"{name}_present"] = value is not None
            if value is not None:
                frame[f"{name}_qpos"] = value
        shared.robot_command_ring.write(frame)
        return time.monotonic_ns()
