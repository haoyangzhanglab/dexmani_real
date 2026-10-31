"""遥操作、手部重定向及本地采集参数；不导入优化器或设备后端。"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

import numpy as np


@dataclass(frozen=True)
class TeleopConfig:
    control_hz: float = 16.0
    # 每控制步中新目标的权重：0 保留参考，1 直接跟随新目标。
    ema_alpha_pos: float = 0.95
    ema_alpha_rot: float = 0.85
    vr_pos_scale: float = 0.7
    vr_stale_threshold_s: float = 0.5

    def validate(self) -> None:
        if not (0.0 <= self.ema_alpha_pos <= 1.0 and 0.0 <= self.ema_alpha_rot <= 1.0):
            raise ValueError("EMA alphas must be in [0, 1]")
        if not all(
            np.isfinite(v) and v > 0
            for v in (self.vr_pos_scale, self.vr_stale_threshold_s)
        ):
            raise ValueError("VR position scale and stale threshold must be finite and positive")


@dataclass(frozen=True)
class CollectionConfig:
    """录制路径与时长；入口仅在启用录制时校验。"""

    enabled: bool = True
    episodes_dir: str = "episodes"
    max_duration_s: float = 60.0

    def validate(self) -> None:
        if not isinstance(self.episodes_dir, str) or not self.episodes_dir.strip():
            raise ValueError("collection.episodes_dir must be non-empty")
        if not np.isfinite(self.max_duration_s) or self.max_duration_s <= 0:
            raise ValueError("collection.max_duration_s must be finite and positive")


@dataclass(frozen=True)
class KeyboardTeleopConfig:
    control_hz: float = 30.0
    target_max_age_s: float = 0.5
    delta_pos_m: float = 0.008
    delta_rpy_rad: float = 0.03
    workspace_command_margin_m: float = 0.005
    ik_max_pose_error_pos_m: float = 0.002
    ik_max_pose_error_rot_rad: float = radians(2.0)

    def validate(self) -> None:
        for name in ("target_max_age_s", "delta_pos_m", "delta_rpy_rad"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"keyboard {name} must be finite and positive")
        margin = self.workspace_command_margin_m
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("workspace_command_margin_m must be finite and non-negative")


@dataclass(frozen=True)
class HandRetargetingConfig:
    """米/弧度/秒；权重作用于无量纲残差，默认值是待真机验证的起点。"""

    calibration_samples: int = 25
    calibration_timeout_s: float = 5.0
    calibration_max_relative_mad: float = 0.10
    bone_length_tolerance: float = 0.35
    huber_delta_m: float = 0.02
    shape_weight: float = 0.25
    tip_weight: float = 2.0
    proximal_direction_weight: float = 2.0
    direction_weight: float = 1.0
    # 单位方向弦长；0.3 约为 17.3°。Huber / delta 限制大残差的牵引。
    direction_huber_delta: float = 0.3
    pinch_weight: float = 8.0
    pinch_start_m: float = 0.03
    pinch_full_m: float = 0.015
    # 近端正则系数按 1/dt 缩放，相当于离散阻尼；不是力或实测速度约束。
    damping_s: float = 0.002
    smoothing_tau_s: float = 0.02
    max_joint_speed_rad_s: float = 4.0
    max_evaluations: int = 50
    max_solve_time_s: float = 0.025
    ftol_abs: float = 1e-7

    def validate(self) -> None:
        for name in ("calibration_samples", "max_evaluations"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"hand_retargeting.{name} must be an integer >= 2")
        for name in (
            "calibration_timeout_s", "calibration_max_relative_mad", "bone_length_tolerance",
            "huber_delta_m", "direction_huber_delta", "pinch_start_m", "pinch_full_m",
            "max_joint_speed_rad_s", "max_solve_time_s", "ftol_abs",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"hand_retargeting.{name} must be finite and positive")
        for name in (
            "shape_weight", "tip_weight", "direction_weight", "pinch_weight",
            "proximal_direction_weight", "damping_s", "smoothing_tau_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"hand_retargeting.{name} must be finite and non-negative")
        if not 0 < self.pinch_full_m < self.pinch_start_m:
            raise ValueError("pinch distances must satisfy 0 < full < start")
        if not 0 < self.calibration_max_relative_mad < self.bone_length_tolerance < 1:
            raise ValueError("calibration MAD and bone tolerance must satisfy 0 < MAD < tolerance < 1")


DEFAULT_TASK_NAME = "test"
