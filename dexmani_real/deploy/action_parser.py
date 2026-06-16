"""ActionParser — convert policy output to RobotAction."""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.robot_interface import RobotAction, RobotState


class ActionParser:
    """Parse policy output into RobotAction, supporting arm/hand/full modes.

    - "full": arm + hand both driven by policy output
    - "arm_only": hand_cmd filled from current hand state (hold in place)
    - "hand_only": arm_cmd filled from current arm state (hold in place)
    """

    def __init__(self, action_mode: str = "full") -> None:
        if action_mode not in ("full", "arm_only", "hand_only"):
            raise ValueError(
                f"action_mode must be 'full', 'arm_only', or 'hand_only', got '{action_mode}'"
            )
        self.action_mode = action_mode

    def parse(
        self,
        policy_output: np.ndarray,
        state: RobotState,
        action_mode: str | None = None,
    ) -> RobotAction:
        mode = action_mode or self.action_mode
        out = np.asarray(policy_output, dtype=np.float64).ravel()

        if mode == "full":
            if len(out) < 19:
                raise ValueError(
                    f"Policy output too short for 'full' mode: "
                    f"got {len(out)}, expected >= 19 (7 arm + 12 hand)"
                )
            arm_cmd = out[:7].copy()
            hand_cmd = out[7:19].copy()

        elif mode == "arm_only":
            if len(out) < 7:
                raise ValueError(
                    f"Policy output too short for 'arm_only' mode: "
                    f"got {len(out)}, expected >= 7"
                )
            arm_cmd = out[:7].copy()
            hand_cmd = state.hand_qpos.copy()

        elif mode == "hand_only":
            if len(out) < 12:
                raise ValueError(
                    f"Policy output too short for 'hand_only' mode: "
                    f"got {len(out)}, expected >= 12"
                )
            arm_cmd = state.arm_qpos.copy()
            hand_cmd = out[:12].copy()

        else:
            raise ValueError(f"Unknown action_mode: {mode}")

        return RobotAction(
            arm_qpos_cmd=arm_cmd,
            hand_qpos_cmd=hand_cmd,
        )
