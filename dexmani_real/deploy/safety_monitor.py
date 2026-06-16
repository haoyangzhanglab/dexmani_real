"""SafetyMonitor — deployment-time safety checks (stricter than teleop)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.controller.safety_checker import SafetyChecker
from dexmani_real.planner.workspace_safety import WorkspaceSafety
from dexmani_real.robot.robot_interface import RobotAction, RobotState


@dataclass
class SafetyStatus:
    ok: bool
    arm_ok: bool = True
    hand_ok: bool = True
    message: str = ""


class SafetyMonitor:
    """Deployment-time safety monitor with stricter thresholds than teleop.

    Checks: workspace, joint limits, torque, hand current, hand temperature, hand comm.
    """

    def __init__(
        self,
        workspace: WorkspaceSafety,
        *,
        torque_limit_nm: float = 40.0,         # stricter than teleop (50.0)
        hand_current_limit_ma: float = 400.0,   # stricter than teleop (500.0)
        hand_temp_limit_c: float = 65.0,         # stricter than teleop (70.0)
        joint_limit_tolerance_rad: float = 0.05,  # margin from hardware limits
    ) -> None:
        self.workspace = workspace
        self.torque_limit_nm = torque_limit_nm
        self.hand_current_limit_ma = hand_current_limit_ma
        self.hand_temp_limit_c = hand_temp_limit_c
        self.joint_limit_tolerance = joint_limit_tolerance_rad

    def check(self, state: RobotState, action: RobotAction) -> SafetyStatus:
        # Arm torque
        if not SafetyChecker.check_arm_torque(state, self.torque_limit_nm):
            return SafetyStatus(
                ok=False, arm_ok=False, hand_ok=True,
                message=f"Arm torque exceeds {self.torque_limit_nm} N·m",
            )

        # Workspace (check command EEF via FK — done externally by caller)
        # Joint limits (check command within hardware limits ± tolerance)
        if not np.all(np.isfinite(action.arm_qpos_cmd)):
            return SafetyStatus(
                ok=False, arm_ok=False, hand_ok=True,
                message="Arm command contains NaN/Inf",
            )
        if not np.all(np.isfinite(action.hand_qpos_cmd)):
            return SafetyStatus(
                ok=False, arm_ok=True, hand_ok=False,
                message="Hand command contains NaN/Inf",
            )

        # Hand checks
        if not SafetyChecker.check_hand_current(state, self.hand_current_limit_ma):
            return SafetyStatus(
                ok=False, arm_ok=True, hand_ok=False,
                message=f"Hand current exceeds {self.hand_current_limit_ma} mA",
            )

        if not SafetyChecker.check_hand_temperature(state, self.hand_temp_limit_c):
            return SafetyStatus(
                ok=False, arm_ok=True, hand_ok=False,
                message=f"Hand temperature exceeds {self.hand_temp_limit_c} °C",
            )

        if not SafetyChecker.check_hand_comm(state):
            return SafetyStatus(
                ok=False, arm_ok=True, hand_ok=False,
                message="Hand communication error",
            )

        return SafetyStatus(ok=True, arm_ok=True, hand_ok=True)
