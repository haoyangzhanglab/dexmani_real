"""ReturnToHomeOrchestrator — extracted two-phase return-to-home execution.

Refactored from RobotInterface.return_to_home() (~100 lines) into a standalone
orchestrator class.  The orchestrator owns the phase sequencing, cancellation,
and fallback logic; the RobotInterface sub-methods (_execute_phase1_eef_cartesian,
_execute_phase2_joint_space, etc.) are injected as callables.

Design:
  - Orchestrator is stateless except for cancel_event wiring.
  - All hardware calls go through injected callables → testable with mocks.
  - RobotInterface.return_to_home() is a thin wrapper that instantiates the
    orchestrator and delegates.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.planning.types import Pose

logger = get_logger(__name__)


class ReturnToHomeOrchestrator:
    """Two-phase return-to-home execution orchestrator.

    Phase 1: plan_path(home_eef) — Cartesian path to move EEF to home.
    Phase 2: Joint-space interpolation — current qpos → home_qpos,
             per-waypoint collision check.

    Falls back to direct reset (arm.reset() + hand.reset()) if planning fails
    or planner is unavailable.
    """

    def __init__(
        self,
        *,
        arm_connected: Callable[[], bool],
        arm_ready: Callable[[], np.ndarray | None],
        arm_is_error: Callable[[], bool],
        home_qpos: np.ndarray,
        home_eef_pose: Pose,
        dt: float,
        # Phase execution callables (injected from RobotInterface)
        execute_phase1: Callable[[Pose, np.ndarray, float, object], bool],
        execute_phase2: Callable[[np.ndarray, float, object], None],
        return_to_home_direct: Callable[[], bool],
        snap_to_nearest_equivalent: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reset_hand_before_planning: Callable[[np.ndarray], np.ndarray | None],
        at_home: Callable[[np.ndarray, np.ndarray], bool],
        joint_error_to_home: Callable[[np.ndarray, np.ndarray], float],
        hand_reset: Callable[[], bool],
        hand_is_connected: Callable[[], bool],
        # Config
        home_joint_threshold_rad: float = np.deg2rad(3.0),
        max_residual_error_rad: float = np.deg2rad(10.0),
    ) -> None:
        self._arm_connected = arm_connected
        self._arm_ready = arm_ready
        self._arm_is_error = arm_is_error
        self._home_qpos = home_qpos
        self._home_eef_pose = home_eef_pose
        self._dt = dt
        self._home_joint_threshold_rad = home_joint_threshold_rad
        self._max_residual_error_rad = max_residual_error_rad

        # Phase executors
        self._execute_phase1 = execute_phase1
        self._execute_phase2 = execute_phase2
        self._return_to_home_direct = return_to_home_direct
        self._snap = snap_to_nearest_equivalent
        self._reset_hand = reset_hand_before_planning
        self._at_home = at_home
        self._joint_err = joint_error_to_home
        self._hand_reset = hand_reset
        self._hand_connected = hand_is_connected

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, cancel_event: object = None) -> bool:
        """Execute the two-phase return-to-home sequence.

        Returns True on success, False on failure or partial completion.
        """
        # 1. Arm not connected → bail out
        if not self._arm_connected():
            return False

        # 2. Read current qpos; NaN → fallback
        current_qpos = self._arm_ready()
        if current_qpos is None:
            return self._return_to_home_direct()

        # 3. Snap continuous joints to nearest 2π-equivalent of home
        current_qpos = self._snap(current_qpos, self._home_qpos)

        # 4. Already at home?
        if self._at_home(current_qpos, self._home_qpos):
            hand_ok = self._hand_reset() if self._hand_connected() else True
            return hand_ok

        # 5. Phase 0: Hand reset (must happen before plan_path)
        current_qpos = self._reset_hand(current_qpos)
        if current_qpos is None:
            return self._return_to_home_direct()

        # 6. Phase 1: EEF path → home EEF
        phase1_ok = self._execute_phase1(self._home_eef_pose, current_qpos, self._dt, cancel_event)

        # 7. Phase 2: Joint-space homing
        if phase1_ok:
            self._execute_phase2(self._home_qpos, self._dt, cancel_event)
        else:
            logger.warning("Phase 1 plan_path failed, falling back to direct reset " "(SDK joint-space home)")
            return self._return_to_home_direct()

        # 8. Verify final distance to home
        final_qpos = self._arm_ready()
        if final_qpos is not None:
            err_rad = self._joint_err(final_qpos, self._home_qpos)
            if err_rad > self._max_residual_error_rad:
                warnings.warn(f"return_to_home incomplete: joint error " f"{np.rad2deg(err_rad):.1f}° from home")
                return False
            if err_rad < self._home_joint_threshold_rad:
                arm_ok = not self._arm_is_error()
                return arm_ok

        # 9. Final hand reset
        hand_ok = self._hand_reset() if self._hand_connected() else True
        arm_ok = not self._arm_is_error()
        return arm_ok and hand_ok


def create_return_to_home_orchestrator(robot_interface) -> ReturnToHomeOrchestrator:
    """Factory: create a ReturnToHomeOrchestrator from a RobotInterface instance.

    Wires all the sub-methods from RobotInterface as callables so the
    orchestrator doesn't need to reference RobotInterface internals directly.
    """
    ri = robot_interface
    planner = ri.planner

    # Home pose
    home_qpos = ri.arm.config.init_qpos.copy()
    home_eef_pose = ri.kinematics.compute_eef_pose_world(home_qpos)
    dt = float(ri.arm.config.dt)

    return ReturnToHomeOrchestrator(
        arm_connected=ri.arm.is_connected,
        arm_ready=lambda: _arm_ready(ri),
        arm_is_error=ri.arm.is_error,
        home_qpos=home_qpos,
        home_eef_pose=home_eef_pose,
        dt=dt,
        execute_phase1=ri._execute_phase1_eef_cartesian,
        execute_phase2=ri._execute_phase2_joint_space,
        return_to_home_direct=ri._return_to_home_direct,
        snap_to_nearest_equivalent=ri._snap_to_nearest_equivalent,
        reset_hand_before_planning=ri._reset_hand_before_planning,
        at_home=ri._at_home,
        joint_error_to_home=ri._joint_error_to_home,
        hand_reset=lambda: ri.hand.reset() if ri.hand.is_connected() else False,
        hand_is_connected=ri.hand.is_connected,
    )


def _arm_ready(robot_interface) -> np.ndarray | None:
    """Read arm qpos; return None on NaN (caller should fall back)."""
    arm_state = robot_interface.arm.get_state()
    current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
    if not np.all(np.isfinite(current_qpos)):
        return None
    return current_qpos
