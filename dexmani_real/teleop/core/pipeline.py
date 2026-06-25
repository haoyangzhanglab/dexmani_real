"""TeleopPipeline — stateless action computation (simplified).

Flow: arm IK → robust EMA → hand retarget → assemble action.

Workspace clamping is handled by PIDTargetChannel + workspace safety in validate_action().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.planning.types import Pose
from dexmani_real.robot.types import RobotAction
from dexmani_real.utils.hand_utils import OPERATOR2MANO_RIGHT, estimate_frame_from_hand_points
from dexmani_real.utils.signal_utils import robust_ema

if TYPE_CHECKING:
    from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
    from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
    from dexmani_real.planning.planner import XArm7MotionPlanner

logger = get_logger(__name__)


class TeleopPipeline:
    """Stateless action computation pipeline.

    Holds references to arm_mapper, retargeter, planner — but no control state.
    Uses robust_ema() for arm smoothing with anomaly detection.
    """

    def __init__(
        self,
        arm_mapper: ArmWristMapper,
        retargeter: XHandRetargeter,
        planner: XArm7MotionPlanner,
        *,
        ema_alpha_arm: float = 0.95,
    ) -> None:
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self.ema_alpha_arm = float(ema_alpha_arm)
        self._prev_raw_arm: np.ndarray | None = None

    def compute_action(
        self,
        vr_frame: dict,
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray,
        prev_arm_cmd: np.ndarray,
        prev_hand_cmd: np.ndarray,
        *,
        check_workspace: Callable[[np.ndarray], bool] | None = None,
        clamp_workspace_pos: Callable[[np.ndarray], np.ndarray] | None = None,
        last_arm_cmd: np.ndarray | None = None,
    ) -> tuple[RobotAction, dict[str, bool]]:
        """Full action computation: IK → robust EMA → retarget → assemble.

        Returns (action, status) where status has keys: ik_ok, retarget_ok.
        """
        arm_cmd, ik_ok, target_eef_pos = self.compute_arm_command(
            vr_frame, current_arm_qpos, prev_arm_cmd, last_arm_cmd=last_arm_cmd,
        )

        hand_cmd, retarget_ok = self.compute_hand_command(vr_frame, prev_hand_cmd)

        action = RobotAction(
            arm_qpos_cmd=arm_cmd,
            hand_qpos_cmd=hand_cmd,
            target_eef_pos=target_eef_pos,
        )
        return action, {"ik_ok": ik_ok, "retarget_ok": retarget_ok}

    def compute_arm_command(
        self,
        vr_frame: dict,
        arm_qpos: np.ndarray,
        prev_arm_cmd: np.ndarray,
        *,
        last_arm_cmd: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, np.ndarray | None]:
        """Compute arm IK command from VR wrist pose with robust EMA."""
        wrist_pos = np.asarray(vr_frame["wrist_pos"], dtype=np.float64)
        wrist_quat_wxyz = np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)

        arm_cmd = prev_arm_cmd.copy()
        ik_ok = False
        target_eef_pos: np.ndarray | None = None

        if not np.all(np.isfinite(wrist_pos)) or not np.all(np.isfinite(wrist_quat_wxyz)):
            if not getattr(self, "_nan_warned_wrist", False):
                logger.warning("VR wrist pose contains NaN/Inf — holding position")
                self._nan_warned_wrist = True
            return arm_cmd, ik_ok, target_eef_pos
        self._nan_warned_wrist = False

        if not self.arm_mapper.is_ready():
            return arm_cmd, ik_ok, target_eef_pos

        mapped = self.arm_mapper.map(wrist_pos, wrist_quat_wxyz)
        if mapped is None:
            return arm_cmd, ik_ok, target_eef_pos

        target_eef_pos = np.asarray(mapped["pos"], dtype=np.float64)
        target_eef_quat = np.asarray(mapped["quat_wxyz"], dtype=np.float64)

        target_pose = Pose(p=target_eef_pos, q=target_eef_quat)
        ik_result = self.planner.solve_teleop_ik(target_pose, arm_qpos, prev_arm_cmd)

        if ik_result.success and ik_result.qpos is not None:
            ik_ok = True
            raw_arm = np.asarray(ik_result.qpos, dtype=np.float64)
            arm_cmd, self._prev_raw_arm = robust_ema(
                raw_arm, last_arm_cmd, self._prev_raw_arm,
                alpha_normal=self.ema_alpha_arm,
            )

        return arm_cmd, ik_ok, target_eef_pos

    def compute_hand_command(
        self,
        vr_frame: dict,
        prev_hand_cmd: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Compute hand retargeting from VR landmarks."""
        landmarks = vr_frame.get("landmarks")
        hand_cmd = prev_hand_cmd.copy()
        retarget_ok = False

        if landmarks is None or landmarks.shape != (21, 3):
            return hand_cmd, retarget_ok

        if not np.all(np.isfinite(landmarks)):
            if not getattr(self, "_nan_warned_landmarks", False):
                logger.warning("VR landmarks contain NaN/Inf — holding hand position")
                self._nan_warned_landmarks = True
            return hand_cmd, retarget_ok
        self._nan_warned_landmarks = False

        try:
            wrist_rot = estimate_frame_from_hand_points(landmarks)
            mano_landmarks = landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT
            target_hand = self.retargeter.retarget(mano_landmarks)
            if target_hand is not None and len(target_hand) == 12:
                retarget_ok = True
                hand_cmd = np.asarray(target_hand, dtype=np.float64)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            pass

        return hand_cmd, retarget_ok

    @staticmethod
    def soft_deceleration(
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return current_arm_qpos.copy(), current_hand_qpos.copy()
