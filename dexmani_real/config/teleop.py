"""遥操作、手部重定向及本地采集参数；不导入优化器或设备后端。"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from math import radians

import numpy as np


@dataclass(frozen=True)
class TeleopConfig:
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
class CollectionConfig:
    """时长限制仅用于录制。"""

    enabled: bool = True
    episodes_dir: str = "episodes"
    max_duration_s: float = 60.0

    def validate(self) -> None:
        if not isinstance(self.episodes_dir, str) or not self.episodes_dir.strip():
            raise ValueError("collection.episodes_dir must be non-empty")
        if self.enabled and (not np.isfinite(self.max_duration_s) or self.max_duration_s <= 0):
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
class TAGConfig:
    # 长度单位米，顺序为拇指到小指。
    human_finger_lengths: tuple[float, ...] = (0.13, 0.18, 0.19, 0.18, 0.145)
    finger_scale_boost: float = 1.0

    pinky_scale: float = 1.3
    pinky_palm_scale: float = 1.25

    smooth_weight: float = 0.003

    pinch_base_weight: float = 2000.0
    pinch_start_dist_m: float = 0.030
    pinch_full_dist_m: float = 0.008
    pinch_ema_alpha: float = 0.75
    reg_stage1_weight: float = 1.0
    reg_last_weight: float = 0.8

    def validate(self) -> None:
        if not all(np.all(np.isfinite(value)) for value in astuple(self)):
            raise ValueError("TAG parameters must be finite")
        human = np.asarray(self.human_finger_lengths, dtype=np.float64)
        if human.shape != (5,):
            raise ValueError("TAG requires five human finger lengths")
        if np.any(human <= 0):
            raise ValueError("TAG finger lengths must be positive")
        if not 0 <= self.pinch_full_dist_m < self.pinch_start_dist_m:
            raise ValueError("TAG pinch distances must satisfy 0 <= full < start")
        if not 0.0 <= self.pinch_ema_alpha <= 1.0:
            raise ValueError("TAG pinch EMA alpha must be in [0, 1]")


@dataclass(frozen=True)
class DexPilotConfig:
    scaling_factor: float = 1.15
    pinky_scale: float = 1.15
    pinky_palm_scale: float = 1.0
    low_pass_alpha: float = 0.6
    # 抓取投影的进入与退出距离。
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
