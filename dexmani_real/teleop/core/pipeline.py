"""TeleopPipeline — stateless action computation shared by real & sim controllers.

Extracted from TeleopController._compute_action (controller.py) and
vr_teleop_sim._compute_teleop_action (vr_teleop_sim.py) to eliminate
~350 lines of duplicated control logic.

Usage:
    pipeline = TeleopPipeline(arm_mapper, retargeter, planner)
    action, quality = pipeline.compute_action(
        vr_frame, current_arm_qpos, current_hand_qpos,
        prev_arm_cmd, prev_hand_cmd,
        check_workspace=robot.check_workspace,
        ...
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.planning.types import Pose
from dexmani_real.robot.types import RobotAction
from dexmani_real.utils.hand_utils import (
    OPERATOR2MANO_RIGHT,
    estimate_frame_from_hand_points,
)
from dexmani_real.utils.signal_utils import ema_smooth

if TYPE_CHECKING:
    from dexmani_real.teleop.vr.arm_mapper import ArmWristMapper
    from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
    from dexmani_real.planning.planner import XArm7MotionPlanner

logger = get_logger(__name__)


class TeleopPipeline:
    """Stateless action computation pipeline.

    Holds references to arm_mapper, retargeter, planner — but no control state.
    Control state (prev_arm_cmd, prev_hand_cmd, last_arm_cmd for EMA) is passed
    in as parameters by the caller (TeleopController or sim loop).

    Workspace checking is injected via callables so the same pipeline works for
    both real hardware (RobotInterface.check_workspace) and simulation
    (is_in_workspace / clamp_to_workspace).
    """

    def __init__(
        self,
        arm_mapper: ArmWristMapper,
        retargeter: XHandRetargeter,
        planner: XArm7MotionPlanner,
        *,
        ema_alpha_arm: float = 1.0,
    ) -> None:
        self.arm_mapper = arm_mapper
        self.retargeter = retargeter
        self.planner = planner
        self.ema_alpha_arm = float(ema_alpha_arm)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

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
        """Full action computation pipeline.

        Flow:
          1. Arm IK from VR wrist pose
          2. Workspace check on computed EEF → clamp + re-IK if out of bounds
          3. Hand retarget from VR landmarks (MANO frame transform)

        Args:
            vr_frame: VR tracker frame dict (wrist_pos, wrist_quat_wxyz, landmarks).
            current_arm_qpos: Current arm joint positions (7,).
            current_hand_qpos: Current hand joint positions (12,).
            prev_arm_cmd: Previous arm command (7,) for fallback reference.
            prev_hand_cmd: Previous hand command (12,) for fallback reference.
            check_workspace: pos → bool, checks if position is within workspace.
            clamp_workspace_pos: pos → pos, clamps position to workspace boundaries.
            last_arm_cmd: Last successfully sent arm command (for EMA smoothing).

        Returns:
            (action, status) where action is RobotAction and status is a dict
            with keys: ik_ok, retarget_ok.
        """
        # ── 1. Arm IK ──
        arm_cmd, ik_ok, target_eef_pos = self.compute_arm_command(
            vr_frame,
            current_arm_qpos,
            prev_arm_cmd,
            last_arm_cmd=last_arm_cmd,
        )

        # ── 2. Workspace check + clamp + re-IK (optional) ──
        #     If workspace checker is provided and the computed EEF pose is
        #     out of bounds, clamp the target position and re-solve IK rather than
        #     immediately holding — allows the arm to track the VR wrist up to
        #     the boundary instead of stopping abruptly.
        #     (ref: ManiUniCon _clip_action_to_bounds → re-IK)
        in_workspace = True
        retarget_ok = False  # init; only set True by compute_hand_command below

        if check_workspace is not None and ik_ok:
            arm_eef_pose = self.planner.compute_eef_pose_world(arm_cmd)
            in_workspace = check_workspace(arm_eef_pose.p)

            if not in_workspace:
                # Clamp target position to workspace boundaries and re-solve IK
                if clamp_workspace_pos is not None:
                    clamped_pos = clamp_workspace_pos(arm_eef_pose.p)
                    clamped_pose = Pose(p=clamped_pos, q=arm_eef_pose.q)
                    re_ik = self.planner.solve_teleop_ik(
                        clamped_pose,
                        current_arm_qpos,
                        prev_arm_cmd,
                    )
                    if re_ik.success and re_ik.qpos is not None:
                        arm_cmd = np.asarray(re_ik.qpos, dtype=np.float64)
                        ik_ok = True
                        # Re-evaluate workspace on the clamped result
                        clamped_eef = self.planner.compute_eef_pose_world(arm_cmd)
                        in_workspace = check_workspace(clamped_eef.p)
                        # Hand retarget — proceed with clamped arm position
                        hand_cmd, retarget_ok = self.compute_hand_command(
                            vr_frame,
                            prev_hand_cmd,
                        )
                    else:
                        # Re-IK on clamped pose also failed — hold in place
                        arm_cmd = prev_arm_cmd.copy()
                        hand_cmd = prev_hand_cmd.copy()
                else:
                    # No clamper provided — hold
                    arm_cmd = prev_arm_cmd.copy()
                    hand_cmd = prev_hand_cmd.copy()
            else:
                # Workspace OK — proceed to hand retarget
                hand_cmd, retarget_ok = self.compute_hand_command(
                    vr_frame,
                    prev_hand_cmd,
                )
        else:
            # No workspace check or IK failed — proceed to hand retarget
            hand_cmd, retarget_ok = self.compute_hand_command(
                vr_frame,
                prev_hand_cmd,
            )

        # ── 3. Assemble action ──
        action = RobotAction(
            arm_qpos_cmd=arm_cmd,
            hand_qpos_cmd=hand_cmd,
            target_eef_pos=target_eef_pos,
        )
        status = {"ik_ok": ik_ok, "retarget_ok": retarget_ok}
        return action, status

    # ------------------------------------------------------------------
    # Arm IK
    # ------------------------------------------------------------------

    def compute_arm_command(
        self,
        vr_frame: dict,
        arm_qpos: np.ndarray,
        prev_arm_cmd: np.ndarray,
        *,
        last_arm_cmd: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, np.ndarray | None]:
        """Compute arm IK command from VR wrist pose.

        Args:
            vr_frame: VR frame with wrist_pos, wrist_quat_wxyz.
            arm_qpos: Current arm joint positions (7,) for IK seed.
            prev_arm_cmd: Previous arm command (7,) for fallback.
            last_arm_cmd: Last successfully sent arm command for EMA reference.

        Returns:
            (arm_cmd, ik_ok, target_eef_pos).
        """
        wrist_pos = np.asarray(vr_frame["wrist_pos"], dtype=np.float64)
        wrist_quat_wxyz = np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)

        arm_cmd = prev_arm_cmd.copy()
        ik_ok = False
        target_eef_pos: np.ndarray | None = None

        # NaN/Inf guard: a VR tracking glitch producing degenerate wrist pose
        # would propagate through IK → NaN joint command → crash or dangerous
        # motion.  Hold position and log once per burst.
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

        # EMA smoothing (arm only) — fixed alpha for frame-to-frame filtering.
        # 1.0 = no smoothing; 0.95 = ~1-frame time constant at 50Hz (~2ms lag).
        # Hand smoothing is handled by dex-retargeting's built-in low_pass_alpha.
        alpha = self.ema_alpha_arm

        target_pose = Pose(p=target_eef_pos, q=target_eef_quat)
        ik_result = self.planner.solve_teleop_ik(
            target_pose,
            arm_qpos,
            prev_arm_cmd,
        )

        if ik_result.success and ik_result.qpos is not None:
            ik_ok = True
            raw_arm = np.asarray(ik_result.qpos, dtype=np.float64)
            arm_cmd = ema_smooth(raw_arm, last_arm_cmd, alpha)
        # else: hold previous command

        return arm_cmd, ik_ok, target_eef_pos

    # ------------------------------------------------------------------
    # Hand retarget
    # ------------------------------------------------------------------

    def compute_hand_command(
        self,
        vr_frame: dict,
        prev_hand_cmd: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Compute hand retargeting command from VR landmarks.

        Transform flow (matching TeleopController._compute_hand_command):
          1. estimate_frame_from_hand_points(landmarks) → wrist_rot
          2. landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT → MANO convention
          3. retargeter.retarget(mano_landmarks) → 12-DOF hand joints

        Args:
            vr_frame: VR frame with landmarks (21, 3).
            prev_hand_cmd: Previous hand command (12,) for fallback.

        Returns:
            (hand_cmd, retarget_ok).
        """
        landmarks = vr_frame.get("landmarks")
        hand_cmd = prev_hand_cmd.copy()
        retarget_ok = False

        if landmarks is None or landmarks.shape != (21, 3):
            return hand_cmd, retarget_ok

        # NaN/Inf guard: degenerate landmarks (e.g. from VR tracking glitch)
        # produce NaN wrist_rot from SVD, which crashes the retargeter optimizer
        # or produces NaN joint commands.  Hold position and log once per burst.
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
            # else: hold previous command
        except (ValueError, TypeError, np.linalg.LinAlgError):
            # Retarget exception (incl. SVD non-convergence): hold previous
            # command, quality marked as fail
            pass

        return hand_cmd, retarget_ok

    # ------------------------------------------------------------------
    # Soft deceleration (static — no instance state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def soft_deceleration(
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hold current physical position during VR loss.

        Returns:
            (arm_cmd, hand_cmd): Current position copies.
        """
        return current_arm_qpos.copy(), current_hand_qpos.copy()
