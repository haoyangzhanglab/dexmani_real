"""TeleopPipeline — stateless action computation.

Flow: arm mapper → workspace clamp → Cartesian EMA → IK → assemble.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from dexmani_real.utils.log import get_logger
from dexmani_real.planning.types import Pose
from dexmani_real.robot.types import RobotAction
from dexmani_real.utils.hand_utils import OPERATOR2MANO_RIGHT, estimate_frame_from_hand_points
from dexmani_real.utils.signal_utils import ema_smooth_pose

if TYPE_CHECKING:
    from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
    from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
    from dexmani_real.planning.planner import XArm7MotionPlanner

logger = get_logger(__name__)


class TeleopPipeline:
    """Stateless action computation pipeline.

    Smoothing: Cartesian-space EMA (position R³ + rotation vector so(3))
    applied BEFORE IK.  Position and rotation use independent α factors
    because rotation has higher orientation noise and benefits from
    stronger filtering, while position tolerates lower latency.

    Defaults: alpha_pos=0.8 (light smoothing, low latency), alpha_rot=0.4
    (heavy smoothing, suppresses orientation jitter).  Set both to 1.0
    for pass-through.
    """

    def __init__(
        self,
        arm_mapper: ArmWristMapper,
        retargeter: XHandRetargeter,
        planner: XArm7MotionPlanner,
        *,
        ema_alpha_pos: float = 0.8,
        ema_alpha_rot: float = 0.4,
    ) -> None:
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self._ema_alpha_pos = float(np.clip(ema_alpha_pos, 0.0, 1.0))
        self._ema_alpha_rot = float(np.clip(ema_alpha_rot, 0.0, 1.0))

        # Cartesian EMA state: previous smoothed target pose
        self._prev_target_pos: np.ndarray | None = None
        self._prev_target_quat: np.ndarray | None = None

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
    ) -> tuple[RobotAction, dict[str, bool]]:
        """Full action computation: IK → retarget → assemble.

        Returns (action, status) where status has keys: ik_ok, retarget_ok.
        """
        self.planner.set_hand_qpos(prev_hand_cmd)

        arm_cmd, ik_ok = self.compute_arm_command(
            vr_frame, current_arm_qpos, prev_arm_cmd,
            check_workspace=check_workspace,
            clamp_workspace_pos=clamp_workspace_pos,
        )

        hand_cmd, retarget_ok = self.compute_hand_command(vr_frame, prev_hand_cmd)

        action = RobotAction(arm_qpos_cmd=arm_cmd, hand_qpos_cmd=hand_cmd)
        return action, {"ik_ok": ik_ok, "retarget_ok": retarget_ok}

    def compute_arm_command(
        self,
        vr_frame: dict,
        arm_qpos: np.ndarray,
        prev_arm_cmd: np.ndarray,
        *,
        check_workspace: Callable[[np.ndarray], bool] | None = None,
        clamp_workspace_pos: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Compute arm IK command from VR wrist pose.

        Pipeline: VR wrist → mapper → workspace clamp → Cartesian EMA → IK.
        """
        wrist_pos = np.asarray(vr_frame["wrist_pos"], dtype=np.float64)
        wrist_quat_wxyz = np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)

        if not np.all(np.isfinite(wrist_pos)) or not np.all(np.isfinite(wrist_quat_wxyz)):
            return prev_arm_cmd.copy(), False

        if not self.arm_mapper.is_ready():
            return prev_arm_cmd.copy(), False

        mapped = self.arm_mapper.map(wrist_pos, wrist_quat_wxyz)
        if mapped is None:
            return prev_arm_cmd.copy(), False

        target_pos = np.asarray(mapped["pos"], dtype=np.float64)
        target_quat = np.asarray(mapped["quat_wxyz"], dtype=np.float64)

        # ── Workspace clamp ──
        if check_workspace is not None and not check_workspace(target_pos):
            if clamp_workspace_pos is not None:
                target_pos = clamp_workspace_pos(target_pos)
            else:
                return prev_arm_cmd.copy(), False

        # ── Cartesian EMA (sole smoothing stage, before IK) ──
        # First frame seeds the filter; subsequent frames apply EMA.
        if self._prev_target_pos is not None:
            target_pos, target_quat = ema_smooth_pose(
                target_pos, target_quat,
                self._prev_target_pos, self._prev_target_quat,
                self._ema_alpha_pos,
                self._ema_alpha_rot,
            )
        self._prev_target_pos = target_pos.copy()
        self._prev_target_quat = target_quat.copy()

        # ── IK ──
        target_pose = Pose(p=target_pos, q=target_quat)
        ik_result = self.planner.solve_teleop_ik(target_pose, arm_qpos, prev_arm_cmd)

        if ik_result.success and ik_result.qpos is not None:
            return np.asarray(ik_result.qpos, dtype=np.float64), True

        return prev_arm_cmd.copy(), False

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

