"""合并默认配置、实验 YAML 与显式覆盖。"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from dexmani_real.config.control import (
    EnvironmentConfig,
    MotionConfig,
    StaticCollisionBox,
)
from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.config.sensor import CameraConfig, VRConfig
from dexmani_real.config.teleop import (
    CollectionConfig,
    DexPilotConfig,
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
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    keyboard_teleop: KeyboardTeleopConfig = field(default_factory=KeyboardTeleopConfig)
    vr: VRConfig = field(default_factory=VRConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    hand_pointcloud: HandCloudConfig = field(default_factory=HandCloudConfig)
    dexpilot_retargeting: DexPilotConfig = field(default_factory=DexPilotConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def _replace_config(current: Any, changes: Any) -> Any:
    if dataclasses.is_dataclass(current):
        if not isinstance(changes, Mapping):
            raise TypeError(f"{type(current).__name__} overrides must be a mapping")
        updated = dict(changes)
        for name, value in changes.items():
            if isinstance(current, EnvironmentConfig) and name == "static_boxes":
                updated[name] = tuple(
                    _replace_config(StaticCollisionBox(), box) for box in value
                )
            elif hasattr(current, name):
                updated[name] = _replace_config(getattr(current, name), value)
        return dataclasses.replace(cast(Any, current), **updated)
    if isinstance(current, tuple):
        if not isinstance(changes, (list, tuple)):
            raise TypeError("config arrays must use YAML sequences")
        # 空默认序列只有 static_boxes，已在上面按 StaticCollisionBox 构建。
        return tuple(_replace_config(current[0], value) for value in changes)
    if isinstance(current, bool):
        # 引号中的 "false" 不能启用设备。
        if type(changes) is not bool:
            raise TypeError("config switches must use YAML true/false")
    elif isinstance(current, (int, float)):
        # SDK 可能静默截断整数增益。
        if type(changes) not in (int, float) or (
            isinstance(current, int) and type(changes) is not int
        ):
            raise TypeError(f"config value must be {type(current).__name__}")
    elif isinstance(current, str) and changes is not None and not isinstance(changes, str):
        raise TypeError("config text values must be strings or null")
    return changes


def _merge_overrides(values, overrides):
    for name, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            section = values.setdefault(name, {})
            if not isinstance(section, dict):
                raise TypeError(f"config section {name!r} must be a mapping")
            _merge_overrides(section, value)
        else:
            values[name] = value


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> RealConfig:
    """覆盖优先级为 CLI > YAML > 默认值；此处不连接设备或读取标定。"""
    values: dict[str, Any] = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise TypeError("experiment config root must be a mapping")
            values = loaded
    if overrides is not None:
        _merge_overrides(values, overrides)
    return _replace_config(RealConfig(), values)
