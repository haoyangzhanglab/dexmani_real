"""TeleopPipeline — stateless action computation.

Flow: arm mapper → workspace clamp → Cartesian EMA → IK → assemble.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d
from dexmani_real.planning.types import Pose
from dexmani_real.robot.types import RobotAction
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.signal_utils import ema_smooth_pose

if TYPE_CHECKING:
    from dexmani_real.planning.planner import XArm7MotionPlanner
    from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
    from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter

logger = get_logger(__name__)


class TeleopPipeline:
    """Stateless action computation pipeline.

    Smoothing: Cartesian-space EMA (position R³ + rotation vector so(3))
    applied BEFORE IK.  Position and rotation use independent α factors
    because rotation has higher orientation noise and benefits from
    stronger filtering, while position tolerates lower latency.

    Defaults: alpha_pos=0.5 (heavier smoothing, τ≈90ms@16Hz), alpha_rot=0.25
    (heaviest smoothing, τ≈223ms@16Hz, suppresses wrist orientation jitter).
    Set both to 1.0 for pass-through.
    """

    def __init__(
        self,
        arm_mapper: ArmWristMapper,
        retargeter: XHandRetargeter,
        planner: XArm7MotionPlanner,
        *,
        ema_alpha_pos: float = 0.5,
        ema_alpha_rot: float = 0.25,
    ) -> None:
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self._ema_alpha_pos = float(np.clip(ema_alpha_pos, 0.0, 1.0))
        self._ema_alpha_rot = float(np.clip(ema_alpha_rot, 0.0, 1.0))

        # Cartesian EMA state: previous smoothed target pose
        self._prev_target_pos: np.ndarray | None = None
        self._prev_target_quat: np.ndarray | None = None

        # One-shot warn flag (reset per episode in compute_hand_command)
        self._nan_warned_landmarks: bool = False

    def compute_action(
        self,
        vr_frame: dict,
        current_arm_qpos: np.ndarray,
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

        arm_cmd, ik_ok, target_pos, target_quat = self.compute_arm_command(
            vr_frame,
            current_arm_qpos,
            prev_arm_cmd,
            check_workspace=check_workspace,
            clamp_workspace_pos=clamp_workspace_pos,
        )

        hand_cmd, retarget_ok = self.compute_hand_command(vr_frame, prev_hand_cmd)

        action = RobotAction(
            arm_qpos_cmd=arm_cmd,
            hand_qpos_cmd=hand_cmd,
            target_eef_pos=target_pos,
            target_eef_rot6d=quat_wxyz_to_rot6d(target_quat) if target_quat is not None else None,
        )
        return action, {"ik_ok": ik_ok, "retarget_ok": retarget_ok}

    def compute_arm_command(
        self,
        vr_frame: dict,
        arm_qpos: np.ndarray,
        prev_arm_cmd: np.ndarray,
        *,
        check_workspace: Callable[[np.ndarray], bool] | None = None,
        clamp_workspace_pos: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, bool, np.ndarray | None, np.ndarray | None]:
        """Compute arm IK command from VR wrist pose.

        Pipeline: VR wrist → mapper → workspace clamp → Cartesian EMA → IK.

        Returns (arm_cmd, ik_ok, target_pos, target_quat_wxyz) where the target
        is the smoothed Cartesian EEF pose that IK tracks (None if unavailable).
        """
        wrist_pose = self._extract_vr_pose(vr_frame)
        if wrist_pose is None:
            return prev_arm_cmd.copy(), False, None, None

        target_pose = self._map_vr_to_world(*wrist_pose)
        if target_pose is None:
            return prev_arm_cmd.copy(), False, None, None

        target_pos = self._clamp_workspace(target_pose[0], check_workspace, clamp_workspace_pos)
        if target_pos is None:
            return prev_arm_cmd.copy(), False, None, None

        # Cartesian EMA — freeze on IK failure (prev state only updated below)
        target_pos, target_quat = self._apply_cartesian_ema(
            target_pos, target_pose[1],
            self._prev_target_pos, self._prev_target_quat,
        )

        # IK solve
        ik_result = self.planner.solve_teleop_ik(
            Pose(p=target_pos, q=target_quat), arm_qpos, prev_arm_cmd,
        )

        if ik_result.success and ik_result.qpos is not None:
            self._prev_target_pos = target_pos.copy()
            self._prev_target_quat = target_quat.copy()
            return np.asarray(ik_result.qpos, dtype=np.float64), True, target_pos, target_quat

        return prev_arm_cmd.copy(), False, target_pos, target_quat

    # ── Pipeline stages (extracted for testability) ──

    @staticmethod
    def _extract_vr_pose(vr_frame: dict) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract and validate wrist pose from VR frame. Returns None on NaN."""
        wrist_pos = np.asarray(vr_frame["wrist_pos"], dtype=np.float64)
        wrist_quat_wxyz = np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)
        if not np.all(np.isfinite(wrist_pos)) or not np.all(np.isfinite(wrist_quat_wxyz)):
            return None
        return wrist_pos, wrist_quat_wxyz

    def _map_vr_to_world(
        self, wrist_pos: np.ndarray, wrist_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Map VR wrist pose to world frame. Returns None if mapper not ready or
        mapping fails."""
        if not self.arm_mapper.is_ready():
            return None
        mapped = self.arm_mapper.map(wrist_pos, wrist_quat_wxyz)
        if mapped is None:
            return None
        return (
            np.asarray(mapped["pos"], dtype=np.float64),
            np.asarray(mapped["quat_wxyz"], dtype=np.float64),
        )

    @staticmethod
    def _clamp_workspace(
        pos: np.ndarray,
        check_fn: Callable[[np.ndarray], bool] | None,
        clamp_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> np.ndarray | None:
        """Clamp target position to workspace. Returns None if out-of-bounds
        and no clamp function is available (hard failure)."""
        if check_fn is None or check_fn(pos):
            return pos
        if clamp_fn is not None:
            return clamp_fn(pos)
        return None

    def _apply_cartesian_ema(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        prev_pos: np.ndarray | None,
        prev_quat: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Cartesian EMA smoothing. First frame seeds the filter;
        subsequent frames smooth toward target using per-axis alpha factors.

        EMA state update is the **caller's** responsibility — this method
        is a pure function over its inputs.
        """
        if prev_pos is None:
            return target_pos.copy(), target_quat.copy()
        return ema_smooth_pose(
            target_pos, target_quat,
            prev_pos, prev_quat,  # type: ignore[arg-type]
            self._ema_alpha_pos, self._ema_alpha_rot,
        )

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
            target_hand = self.retargeter.retarget(landmarks)
            if target_hand is not None and len(target_hand) == 12:
                retarget_ok = True
                hand_cmd = np.asarray(target_hand, dtype=np.float64)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            pass  # hand holds last-good position; not spam-worthy (retry next frame)

        return hand_cmd, retarget_ok
