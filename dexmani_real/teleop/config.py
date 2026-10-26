"""遥操作、手部重定向及本地采集参数；不导入优化器或设备后端。"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from math import radians

import numpy as np


@dataclass(frozen=True)
class TeleopParams:
    control_hz: float = 16.0
    ema_alpha_pos: float = 0.7
    ema_alpha_rot: float = 0.5
    vr_pos_scale: float = 0.4
    vr_stale_threshold_s: float = 0.5
    hand_retargeting_type: str = "tag"

    def validate(self) -> None:
        if not (0.0 <= self.ema_alpha_pos <= 1.0 and 0.0 <= self.ema_alpha_rot <= 1.0):
            raise ValueError("EMA alphas must be in [0, 1]")
        if not all(
            np.isfinite(v) and v > 0
            for v in (self.vr_pos_scale, self.vr_stale_threshold_s)
        ):
            raise ValueError("VR position scale and stale threshold must be finite and positive")
        if self.hand_retargeting_type not in {"tag", "dexpilot"}:
            raise ValueError("hand_retargeting_type must be 'tag' or 'dexpilot'")


@dataclass(frozen=True)
class CollectionOptions:
    """Local capture options; unrecorded teleop has no capture-duration limit."""

    enabled: bool = True
    episodes_dir: str = "episodes"
    max_duration_s: float = 60.0

    def validate(self) -> None:
        if not isinstance(self.episodes_dir, str) or not self.episodes_dir.strip():
            raise ValueError("collection.episodes_dir must be non-empty")
        if self.enabled and (not np.isfinite(self.max_duration_s) or self.max_duration_s <= 0):
            raise ValueError("collection.max_duration_s must be finite and positive")


@dataclass(frozen=True)
class KeyboardTeleopParams:
    control_hz: float = 30.0
    target_max_age_s: float = 0.5
    delta_pos_m: float = 0.008
    delta_rpy_rad: float = 0.03
    workspace_command_margin_m: float = 0.005
    ik_max_pose_error_pos_m: float = 0.002
    ik_max_pose_error_rot_rad: float = radians(2.0)


@dataclass(frozen=True)
class TAGRetargetingParams:
    """TAG two-stage NLopt hand retargeting parameters."""

    # Thumb→pinky lengths in meters.
    robot_finger_lengths: tuple[float, ...] = (0.161, 0.208, 0.206, 0.204, 0.145)
    human_finger_lengths: tuple[float, ...] = (0.13, 0.18, 0.19, 0.18, 0.145)
    # Multiplies the robot/human length ratio.
    finger_scale_boost: float = 1.0

    # Pinky MCP→TIP scale before TAG optimization.
    pinky_scale: float = 1.3
    # Pinky wrist→MCP scale before TAG optimization.
    pinky_palm_scale: float = 1.25

    # MANO→URDF extrinsic XYZ Euler angles, radians.
    mano_to_urdf_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)

    smooth_weight: float = 0.003
    ftol_abs_s1: float = 1e-4
    maxeval_s1: int = 80

    ftol_abs_s2: float = 1e-6
    maxeval_s2: int = 100
    pinch_base_weight: float = 2000.0
    pinch_start_dist_m: float = 0.030
    pinch_full_dist_m: float = 0.008
    pinch_ema_alpha: float = 0.75
    pinch_skip_threshold: float = 0.01
    reg_stage1_weight: float = 1.0
    reg_last_weight: float = 0.8

    def validate(self) -> None:
        if not all(np.all(np.isfinite(value)) for value in astuple(self)):
            raise ValueError("TAG parameters must be finite")
        # Zero weights and backend stopping tolerances are experiment choices.
        robot = np.asarray(self.robot_finger_lengths, dtype=np.float64)
        human = np.asarray(self.human_finger_lengths, dtype=np.float64)
        if robot.shape != (5,) or human.shape != (5,):
            raise ValueError("TAG requires five robot and human finger lengths")
        if np.any(robot <= 0) or np.any(human <= 0):
            raise ValueError("TAG finger lengths must be positive")
        if self.maxeval_s1 <= 0 or self.maxeval_s2 <= 0:
            raise ValueError("TAG optimizer maxeval values must be positive")
        if not 0 <= self.pinch_full_dist_m < self.pinch_start_dist_m:
            raise ValueError("TAG pinch distances must satisfy 0 <= full < start")
        if not 0.0 <= self.pinch_ema_alpha <= 1.0:
            raise ValueError("TAG pinch EMA alpha must be in [0, 1]")


@dataclass(frozen=True)
class DexPilotRetargetingParams:
    # Human-to-robot length scale.
    scaling_factor: float = 1.15
    # Pinky-chain scale before scaling_factor.
    pinky_scale: float = 1.15
    # Pinky wrist→MCP scale before scaling_factor.
    pinky_palm_scale: float = 1.0
    low_pass_alpha: float = 0.6
    # Enter/exit thresholds for the projected grasp regime.
    project_dist_m: float = 0.03
    escape_dist_m: float = 0.05

    def validate(self) -> None:
        if not all(np.all(np.isfinite(value)) for value in astuple(self)):
            raise ValueError("DexPilot parameters must be finite")
        if not 0.0 <= self.low_pass_alpha <= 1.0:
            raise ValueError("DexPilot low_pass_alpha must be in [0, 1]")
        if not 0 < self.project_dist_m <= self.escape_dist_m:
            raise ValueError("DexPilot distances must satisfy 0 < project_dist_m <= escape_dist_m")


DEFAULT_TASK_NAME = "test"
