"""Control, retargeting and runtime timing parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.config.environment import WorkspaceBounds

_READINESS_SUBSYSTEMS = frozenset(
    {"arm", "hand", "camera", "pointcloud", "recorder", "policy", "vr"}
)


@dataclass(frozen=True)
class EMAParams:
    """Cartesian-space EMA smoothing parameters."""

    alpha_pos: float = 0.5
    alpha_rot: float = 0.5

    def validate(self) -> None:
        if not (np.isfinite(self.alpha_pos) and np.isfinite(self.alpha_rot)):
            raise ValueError("EMA alphas must be finite")
        if not (0.0 <= self.alpha_pos <= 1.0 and 0.0 <= self.alpha_rot <= 1.0):
            raise ValueError("EMA alphas must be in [0, 1]")


@dataclass(frozen=True)
class VRMappingParams:
    """VR wrist → EEF mapping parameters."""

    pos_scale: float = 1.0
    rot_scale: float = 1.0
    max_delta_rot_rad: float = 3.0  # total-from-reset rotation cap
    stale_threshold_s: float = 0.5

    def validate(self) -> None:
        values = (
            self.pos_scale,
            self.rot_scale,
            self.max_delta_rot_rad,
            self.stale_threshold_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError(
                "VR mapping scales, delta, and stale threshold must be finite and positive"
            )


@dataclass(frozen=True)
class TeleopTimingParams:
    """VR control and recording cadence."""

    control_hz: float = 16.0

    def validate(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("teleop.control_hz must be finite and positive")


@dataclass(frozen=True)
class PolicyParams:
    """Shared experiment/control settings; learned policy timing is PolicySpec-owned."""

    quit_save_timeout_s: float = 30.0
    post_teleop_timeout_s: float = 60.0

    ema: EMAParams = field(default_factory=EMAParams)

    vr_mapping: VRMappingParams = field(default_factory=VRMappingParams)

    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)

    recording_enabled: bool = True
    max_record_duration_s: float = 60.0
    min_record_duration_s: float = 1.0
    episodes_dir: str = "episodes"

    ik_max_pose_error_pos_m: float = 0.02
    ik_max_pose_error_rot_rad: float = np.deg2rad(5.0)

    hand_enabled: bool = True
    hand_retargeting_type: str = "tag"

    def validate(self) -> None:
        timing = (
            self.quit_save_timeout_s,
            self.post_teleop_timeout_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in timing):
            raise ValueError("policy operator timeouts must be finite and positive")
        if (
            not np.isfinite(self.max_record_duration_s)
            or not np.isfinite(self.min_record_duration_s)
            or self.max_record_duration_s <= 0
            or self.min_record_duration_s < 0
            or self.min_record_duration_s > self.max_record_duration_s
        ):
            raise ValueError("recording durations must be finite, ordered, and non-negative")
        if not self.episodes_dir:
            raise ValueError("policy episodes_dir must be non-empty")
        if (
            not np.isfinite(self.ik_max_pose_error_pos_m)
            or not np.isfinite(self.ik_max_pose_error_rot_rad)
            or self.ik_max_pose_error_pos_m <= 0
            or self.ik_max_pose_error_rot_rad <= 0
        ):
            raise ValueError("policy online IK limits must be finite and positive")
        if self.hand_retargeting_type not in {"tag", "dexpilot"}:
            raise ValueError("hand_retargeting_type must be 'tag' or 'dexpilot'")


@dataclass(frozen=True)
class KeyboardTeleopParams:
    """Keyboard teleoperation parameters — single source of truth."""

    control_hz: float = 30.0
    delta_pos_m: float = 0.008
    delta_rpy_rad: float = 0.03
    workspace_command_margin_m: float = 0.005
    ik_max_pose_error_pos_m: float = 0.002
    ik_max_pose_error_rot_rad: float = np.deg2rad(2.0)
    idle_interval_frames: int = 150

    def validate(self) -> None:
        numeric = (
            self.control_hz,
            self.delta_pos_m,
            self.delta_rpy_rad,
            self.workspace_command_margin_m,
            self.ik_max_pose_error_pos_m,
            self.ik_max_pose_error_rot_rad,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("keyboard teleop numeric parameters must be finite")
        if self.control_hz <= 0:
            raise ValueError(f"control_hz={self.control_hz} must be > 0")
        if self.delta_pos_m <= 0:
            raise ValueError(f"delta_pos_m={self.delta_pos_m} must be > 0")
        if self.delta_rpy_rad <= 0:
            raise ValueError(f"delta_rpy_rad={self.delta_rpy_rad} must be > 0")
        if self.workspace_command_margin_m < 0:
            raise ValueError("workspace_command_margin_m must be non-negative")
        if self.ik_max_pose_error_pos_m <= 0 or self.ik_max_pose_error_rot_rad <= 0:
            raise ValueError("keyboard IK pose-error limits must be > 0")
        if self.idle_interval_frames <= 0:
            raise ValueError("keyboard idle interval must be > 0")


@dataclass(frozen=True)
class TAGRetargetingParams:
    """TAG two-stage NLopt hand retargeting parameters (``retargeting_type="tag"``)."""

    robot_finger_lengths: tuple[float, ...] = (0.161, 0.208, 0.206, 0.204, 0.145)
    # XHand finger lengths (thumb through pinky, meters).

    human_finger_lengths: tuple[float, ...] = (0.13, 0.18, 0.19, 0.18, 0.145)
    # Human finger lengths (thumb through pinky, meters).

    finger_scale_boost: float = 1.0
    # Multiplier on the robot/human length ratio.

    pinky_scale: float = 1.3
    # Scale the pinky MCP→TIP chain before TAG optimization.

    pinky_palm_scale: float = 1.25
    # Scale the pinky wrist→MCP baseline before TAG optimization.

    mano_to_urdf_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Frame alignment rotation.

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
    prior_weight: float = 0.01
    # Weight for the optional human-flexion prior in both NLopt stages.

    def validate(self) -> None:
        robot = np.asarray(self.robot_finger_lengths, dtype=np.float64)
        human = np.asarray(self.human_finger_lengths, dtype=np.float64)
        euler = np.asarray(self.mano_to_urdf_euler, dtype=np.float64)
        if robot.shape != (5,) or human.shape != (5,) or euler.shape != (3,):
            raise ValueError("TAG finger lengths/Euler alignment have invalid shape")
        if (
            not np.all(np.isfinite(np.concatenate((robot, human, euler))))
            or np.any(robot <= 0)
            or np.any(human <= 0)
        ):
            raise ValueError(
                "TAG finger lengths/Euler alignment must be finite and lengths positive"
            )
        positive = (
            self.finger_scale_boost,
            self.pinky_scale,
            self.pinky_palm_scale,
            self.ftol_abs_s1,
            self.ftol_abs_s2,
            self.pinch_base_weight,
            self.pinch_start_dist_m,
            self.pinch_full_dist_m,
            self.reg_stage1_weight,
            self.reg_last_weight,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError(
                "TAG scales, tolerances, distances, and regularization weights must be positive"
            )
        if not np.isfinite(self.smooth_weight) or self.smooth_weight < 0:
            raise ValueError("TAG smooth_weight must be finite and non-negative")
        if not np.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("TAG prior_weight must be finite and non-negative")
        if self.maxeval_s1 <= 0 or self.maxeval_s2 <= 0:
            raise ValueError("TAG optimizer maxeval values must be positive")
        if self.pinch_full_dist_m > self.pinch_start_dist_m:
            raise ValueError("TAG pinch_full_dist_m must not exceed pinch_start_dist_m")
        if not (0.0 <= self.pinch_ema_alpha <= 1.0) or not (
            0.0 <= self.pinch_skip_threshold <= 1.0
        ):
            raise ValueError("TAG pinch EMA/skip thresholds must be in [0, 1]")


@dataclass(frozen=True)
class DexPilotRetargetingParams:
    """Runtime parameters for the DexPilot backend."""

    # Human-to-robot size compensation.
    scaling_factor: float = 1.15
    # Pinky-chain scale applied before scaling_factor.
    pinky_scale: float = 1.15
    # Pinky wrist→MCP baseline scale, applied before scaling_factor.
    pinky_palm_scale: float = 1.0
    low_pass_alpha: float = 0.6
    # Enter/exit thresholds for the projected grasp regime.
    project_dist_m: float = 0.03
    escape_dist_m: float = 0.05
    prior_weight: float = 0.05
    # Weight for the optional human-flexion prior.

    def validate(self) -> None:
        numeric = (
            self.scaling_factor,
            self.pinky_scale,
            self.pinky_palm_scale,
            self.low_pass_alpha,
            self.project_dist_m,
            self.escape_dist_m,
            self.prior_weight,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("DexPilot retargeting parameters must be finite")
        if self.scaling_factor <= 0 or self.pinky_scale <= 0 or self.pinky_palm_scale <= 0:
            raise ValueError(
                "DexPilot scaling_factor, pinky_scale, and pinky_palm_scale must be positive"
            )
        if not 0.0 <= self.low_pass_alpha <= 1.0:
            raise ValueError("DexPilot low_pass_alpha must be in [0, 1]")
        if self.project_dist_m <= 0 or self.escape_dist_m < self.project_dist_m:
            raise ValueError("DexPilot distances must satisfy 0 < project_dist_m <= escape_dist_m")
        if self.prior_weight < 0:
            raise ValueError("DexPilot prior_weight must be non-negative")


@dataclass(frozen=True)
class SafetyParams:
    """Process startup/shutdown parameters — single source of truth."""

    readiness_timeouts_s: Mapping[str, float] = field(
        default_factory=lambda: {
            "arm": 15.0,
            "hand": 15.0,
            "camera": 15.0,
            "pointcloud": 30.0,
            "recorder": 15.0,
            "policy": 120.0,
            "vr": 120.0,
        }
    )
    shutdown_timeout_s: float = 65.0

    def validate(self) -> None:
        if not self.readiness_timeouts_s or any(
            not name or not np.isfinite(value) or value <= 0
            for name, value in self.readiness_timeouts_s.items()
        ):
            raise ValueError("readiness timeout names/values must be non-empty, finite, and > 0")
        if _READINESS_SUBSYSTEMS - self.readiness_timeouts_s.keys():
            raise ValueError("readiness_timeouts_s is missing a runtime subsystem")
        if not np.isfinite(self.shutdown_timeout_s) or self.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be finite and positive")
