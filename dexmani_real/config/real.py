"""实验配置的直接组装与可信本地 Python 文件入口。"""

from __future__ import annotations

import runpy
from dataclasses import dataclass, field
from pathlib import Path

from dexmani_real.config.control import (
    EnvironmentConfig,
    MotionConfig,
)
from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.config.sensor import CameraConfig, VRConfig
from dexmani_real.config.teleop import (
    HandRetargetingConfig,
    KeyboardTeleopConfig,
    TeleopConfig,
)
from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.config.xhand import HandConfig


@dataclass(frozen=True)
class RealConfig:
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    keyboard_teleop: KeyboardTeleopConfig = field(default_factory=KeyboardTeleopConfig)
    vr: VRConfig = field(default_factory=VRConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    hand_pointcloud: HandCloudConfig = field(default_factory=HandCloudConfig)
    hand_retargeting: HandRetargetingConfig = field(default_factory=HandRetargetingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def load_config(path: str | Path | None = None) -> RealConfig:
    """省略路径使用默认值；实验文件直接导出 CONFIG，不连接设备或读取标定。"""
    config = RealConfig()
    if path is not None:
        path = Path(path).expanduser()
        if path.suffix != ".py":
            raise ValueError("experiment config must be a Python .py file")
        loaded = runpy.run_path(str(path)).get("CONFIG")
        if not isinstance(loaded, RealConfig):
            raise TypeError(f"{path} must define CONFIG: RealConfig")
        config = loaded
    # 业务校验和序列化由实际使用或归档该配置的边界负责。
    return config
