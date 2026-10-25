"""Load package defaults with explicit experiment and CLI overrides."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dexmani_real.control.config import (
    EnvironmentConfig,
    MotionParams,
    StaticCollisionBox,
)
from dexmani_real.deployment.config import ExecutionConfig
from dexmani_real.perception.hand_config import HandPointCloudConfig
from dexmani_real.perception.pointcloud.config import PointCloudConfig
from dexmani_real.robot.xarm7.config import ArmParams
from dexmani_real.robot.xhand.config import HandParams
from dexmani_real.runtime.config import RuntimeTimeouts
from dexmani_real.sensor.camera.config import CameraParams
from dexmani_real.sensor.vr.config import VRParams
from dexmani_real.teleop.config import (
    CollectionOptions,
    DexPilotRetargetingParams,
    KeyboardTeleopParams,
    TAGRetargetingParams,
    TeleopParams,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment values loaded together and validated at their usage boundaries."""

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    arm: ArmParams = field(default_factory=ArmParams)
    hand: HandParams = field(default_factory=HandParams)
    motion: MotionParams = field(default_factory=MotionParams)
    teleop: TeleopParams = field(default_factory=TeleopParams)
    collection: CollectionOptions = field(default_factory=CollectionOptions)
    keyboard_teleop: KeyboardTeleopParams = field(default_factory=KeyboardTeleopParams)
    vr: VRParams = field(default_factory=VRParams)
    safety: RuntimeTimeouts = field(default_factory=RuntimeTimeouts)
    camera: CameraParams = field(default_factory=CameraParams)
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    hand_pointcloud: HandPointCloudConfig = field(default_factory=HandPointCloudConfig)
    tag_retargeting: TAGRetargetingParams = field(default_factory=TAGRetargetingParams)
    dexpilot_retargeting: DexPilotRetargetingParams = field(
        default_factory=DexPilotRetargetingParams
    )
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def apply_config_overrides(current: Any, changes: Any) -> Any:
    """Update dataclass sections; numerical constraints belong to consumers."""
    if dataclasses.is_dataclass(current) and not isinstance(current, type):
        if not isinstance(changes, Mapping):
            raise TypeError(f"{type(current).__name__} overrides must be a mapping")
        updated = dict(changes)
        for name, value in changes.items():
            if isinstance(current, EnvironmentConfig) and name == "static_boxes":
                updated[name] = tuple(
                    apply_config_overrides(StaticCollisionBox(), box) for box in value
                )
            elif hasattr(current, name):
                updated[name] = apply_config_overrides(getattr(current, name), value)
        # The dataclass constructor rejects unknown fields.
        return dataclasses.replace(current, **updated)
    if isinstance(current, tuple):
        if not isinstance(changes, (list, tuple)):
            raise TypeError("config arrays must use YAML sequences")
        return (
            tuple(apply_config_overrides(current[0], value) for value in changes)
            if current
            else tuple(changes)
        )
    if isinstance(current, bool):
        # A quoted "false" must never enable a hardware or processing switch.
        if type(changes) is not bool:
            raise TypeError("config switches must use YAML true/false")
    elif isinstance(current, (int, float)):
        # SDKs may silently truncate integer gains; reject malformed numbers before connecting.
        if type(changes) not in (int, float) or (
            isinstance(current, int) and type(changes) is not int
        ):
            raise TypeError(f"config value must be {type(current).__name__}")
    elif isinstance(current, str) and changes is not None and not isinstance(changes, str):
        raise TypeError("config text values must be strings or null")
    return changes


def load_experiment_config(
    *,
    yaml_path: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """合并 CLI > YAML > 默认值；运行约束由使用方校验，不连接设备或读取标定。"""
    loaded = {}
    if yaml_path is not None:
        with Path(yaml_path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if values is not None:
            if not isinstance(values, Mapping):
                raise TypeError("experiment config root must be a mapping")
            loaded = dict(values)
    for key, value in (cli_overrides or {}).items():
        if value is None:
            continue
        *sections, name = key.split(".")
        cursor = loaded
        for section in sections:
            if section not in cursor:
                cursor[section] = {}
            elif not isinstance(cursor[section], dict):
                raise TypeError(f"config section {section!r} must be a mapping")
            cursor = cursor[section]
        cursor[name] = value
    return apply_config_overrides(ExperimentConfig(), loaded)
