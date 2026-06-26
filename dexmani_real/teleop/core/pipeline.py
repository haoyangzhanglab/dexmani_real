"""TeleopPipeline — stateless action computation.

Flow: arm mapper → workspace clamp → complementary filter → deadzone → IK → assemble.

Workspace clamping is handled by ArmInnerLoop timeout hold + validate_action().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from dexmani_real.utils.log import get_logger
from dexmani_real.planning.types import Pose
from dexmani_real.robot.types import RobotAction
from dexmani_real.utils.hand_utils import OPERATOR2MANO_RIGHT, estimate_frame_from_hand_points
from dexmani_real.utils.signal_utils import ComplementaryFilter, ema_smooth

if TYPE_CHECKING:
    from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
    from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
    from dexmani_real.planning.planner import XArm7MotionPlanner

logger = get_logger(__name__)


class TeleopPipeline:
    """Stateless action computation pipeline.

    Holds references to arm_mapper, retargeter, planner — but no control state.

    Smoothing architecture (configurable, opt-in):
      - **Complementary Filter** (pose-space, BEFORE IK): decoupled position
        EMA + orientation Slerp.  Tune ``comp_ratio_pos`` / ``comp_ratio_rot``
        independently.  Ref: OpenTeach franka.py:21-33.
      - **Joint-space EMA** (AFTER IK): single-alpha exponential moving average
        on the 7-DOF arm command.  Ref: LeFranX xhand_vr_teleoperator.py:306-308.

    When the complementary filter is active, set ``ema_alpha_arm=1.0``
    (pass-through) to avoid double smoothing.

    Deadzone filtering (ref: LeFranX config_franka_fer_vr.py:32-33):
    VR wrist poses that haven't moved beyond position_deadzone (m) AND
    orientation_deadzone (rad) are silently skipped — no IK solve, prev
    command is held.  This eliminates wasted IK cycles and micro-jitter
    when the operator's hand is stationary.
    """

    def __init__(
        self,
        arm_mapper: ArmWristMapper,
        retargeter: XHandRetargeter,
        planner: XArm7MotionPlanner,
        *,
        ema_alpha_arm: float = 1.0,
        position_deadzone: float = 0.001,
        orientation_deadzone: float = 0.03,
        comp_ratio_pos: float = 0.0,
        comp_ratio_rot: float = 0.0,
    ) -> None:
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self.ema_alpha_arm = float(ema_alpha_arm)
        self.position_deadzone = float(position_deadzone)
        self.orientation_deadzone = float(orientation_deadzone)

        # Deadzone state: last transmitted VR pose (not last received)
        self._last_deadzone_pos: np.ndarray | None = None
        self._last_deadzone_quat: np.ndarray | None = None

        # Complementary filter (opt-in, default 0.0 = disabled)
        if comp_ratio_pos > 0 or comp_ratio_rot > 0:
            self._comp_filter: ComplementaryFilter | None = ComplementaryFilter(
                pos_comp_ratio=comp_ratio_pos,
                rot_comp_ratio=comp_ratio_rot,
            )
        else:
            self._comp_filter = None

    def reset_filter(self) -> None:
        """Reset complementary filter state (e.g. after calibration/recalibration).

        The next ``compute_arm_command()`` call will seed the filter with the
        current VR pose and pass it through unfiltered.
        """
        if self._comp_filter is not None:
            self._comp_filter.reset()

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
        """Full action computation: IK → EMA → retarget → assemble.

        Returns (action, status) where status has keys: ik_ok, retarget_ok.
        """
        arm_cmd, ik_ok, target_eef_pos = self.compute_arm_command(
            vr_frame, current_arm_qpos, prev_arm_cmd,
            check_workspace=check_workspace,
            clamp_workspace_pos=clamp_workspace_pos,
            last_arm_cmd=last_arm_cmd,
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
        check_workspace: Callable[[np.ndarray], bool] | None = None,
        clamp_workspace_pos: Callable[[np.ndarray], np.ndarray] | None = None,
        last_arm_cmd: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, np.ndarray | None]:
        """Compute arm IK command from VR wrist pose with EMA smoothing.

        LeFranX-style simple exponential moving average in joint space.
        Formula: smoothed = alpha * raw + (1-alpha) * prev_smoothed.
        Default alpha=1.0 (no smoothing, pass-through).
        """
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

        # ── Workspace clamping: keep IK target within reachable bounds ──
        # On real hardware, RobotInterface.validate_action() gates the final
        # command. In sim / headless workflows, clamping the target BEFORE IK
        # avoids flooding the solver with unreachable poses and keeps the arm
        # moving toward the closest feasible position instead of freezing.
        if check_workspace is not None and not check_workspace(target_eef_pos):
            if clamp_workspace_pos is not None:
                clamped = clamp_workspace_pos(target_eef_pos)
                if not getattr(self, "_workspace_clamped_warned", False):
                    logger.warning(
                        "VR target outside workspace: raw=%s clamped=%s",
                        np.array2string(target_eef_pos, precision=3, suppress_small=True),
                        np.array2string(clamped, precision=3, suppress_small=True),
                    )
                    self._workspace_clamped_warned = True
                target_eef_pos = clamped
            else:
                if not getattr(self, "_workspace_oob_warned", False):
                    logger.warning(
                        "VR target outside workspace (no clamp): pos=%s — holding",
                        np.array2string(target_eef_pos, precision=3, suppress_small=True),
                    )
                    self._workspace_oob_warned = True
                return arm_cmd, ik_ok, target_eef_pos
        else:
            self._workspace_clamped_warned = False
            self._workspace_oob_warned = False

        # ── Complementary Filter: pose-space smoothing BEFORE IK ──
        # Position EMA + orientation Slerp, independently tuned.
        # Ref: OpenTeach franka.py:21-33.
        # NOTE: operate on the clamped target so filter state stays
        # consistent even when the workspace clamp is active.
        if self._comp_filter is not None:
            target_eef_pos, target_eef_quat = self._comp_filter(target_eef_pos, target_eef_quat)

        # ── Deadzone check: skip IK if VR hand hasn't moved enough ──
        # Ref: LeFranX config_franka_fer_vr.py — position_deadzone=0.001, orientation_deadzone=0.03
        # Returning ik_ok=True: prev_arm_cmd is a valid, previously-computed
        # command — the arm is still tracking correctly, we just didn't need
        # to recompute this frame.
        if self._within_deadzone(target_eef_pos, target_eef_quat):
            return arm_cmd, True, target_eef_pos

        # Mark pose as "transmitted" for next deadzone comparison
        self._last_deadzone_pos = target_eef_pos.copy()
        self._last_deadzone_quat = target_eef_quat.copy()

        target_pose = Pose(p=target_eef_pos, q=target_eef_quat)
        ik_result = self.planner.solve_teleop_ik(target_pose, arm_qpos, prev_arm_cmd)

        if ik_result.success and ik_result.qpos is not None:
            ik_ok = True
            raw_arm = np.asarray(ik_result.qpos, dtype=np.float64)
            arm_cmd = ema_smooth(raw_arm, last_arm_cmd, alpha=self.ema_alpha_arm)

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

    # ── Deadzone helpers ──

    def _within_deadzone(self, target_pos: np.ndarray, target_quat: np.ndarray) -> bool:
        """Return True if the VR pose has not moved enough to warrant a new IK solve.

        Ref: LeFranX config_franka_fer_vr.py:32-33
             position_deadzone=0.001 m (1 mm), orientation_deadzone=0.03 rad (~1.7°)
        """
        if self._last_deadzone_pos is None or self._last_deadzone_quat is None:
            return False
        pos_delta = float(np.linalg.norm(target_pos - self._last_deadzone_pos))
        if pos_delta >= self.position_deadzone:
            return False
        rot_delta = self._angular_distance(target_quat, self._last_deadzone_quat)
        return bool(rot_delta < self.orientation_deadzone)

    @staticmethod
    def _angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        """Angular distance (rad) between two quaternions in wxyz order."""
        dot = float(np.abs(np.dot(q1, q2)))
        dot = min(dot, 1.0)
        return 2.0 * np.arccos(dot)

    @staticmethod
    def soft_deceleration(
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return current_arm_qpos.copy(), current_hand_qpos.copy()
