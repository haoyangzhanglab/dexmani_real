"""Load package defaults with explicit experiment and CLI overrides."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dexmani_real.config.control import (
    CollectionOptions,
    DexPilotRetargetingParams,
    KeyboardTeleopParams,
    MotionParams,
    SafetyParams,
    TAGRetargetingParams,
    TeleopParams,
)
from dexmani_real.config.environment import (
    EnvironmentConfig,
    StaticCollisionBox,
    TableCollisionConfig,
    validate_table_plane,
)
from dexmani_real.config.execution import ExecutionConfig
from dexmani_real.robot.xarm7.config import ArmParams
from dexmani_real.robot.xhand.config import HandParams
from dexmani_real.sensor.camera.config import CameraParams
from dexmani_real.sensor.pointcloud.config import PointCloudConfig
from dexmani_real.sensor.pointcloud.hand_config import HandPointCloudConfig
from dexmani_real.sensor.vr.config import VRParams


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
    safety: SafetyParams = field(default_factory=SafetyParams)
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
    if dataclasses.is_dataclass(current):
        if not isinstance(changes, Mapping):
            raise TypeError(f"{type(current).__name__} overrides must be a mapping")
        updated = dict(changes)
        if isinstance(current, EnvironmentConfig) and "static_boxes" in updated:
            updated["static_boxes"] = tuple(
                apply_config_overrides(StaticCollisionBox(), box) for box in updated["static_boxes"]
            )
        for name, value in updated.items():
            if hasattr(current, name):
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


def validate_robot_config(cfg: ExperimentConfig) -> None:
    """Values used by every robot owner, including HOME and hand-disabled geometry."""
    cfg.arm.validate()
    cfg.arm.homing.validate()
    if cfg.hand.enabled:
        cfg.hand.validate()
    else:
        cfg.hand.validate_home_pose()
    cfg.safety.validate()


def validate_control_rate(cfg: ExperimentConfig, control_hz: float) -> None:
    """Validate cadence before connecting; never silently decimate hand targets."""
    if not np.isfinite(control_hz) or not 0 < control_hz <= cfg.arm.servo_hz:
        raise ValueError("target frequency must be positive and no greater than arm.servo_hz")
    if 1 / control_hz >= cfg.arm.feedback_max_age_s:
        raise ValueError(
            "target period must fit arm.feedback_max_age_s (coordinator health budget)"
        )
    if cfg.hand.enabled:
        if control_hz > 83:
            raise ValueError("XHand targets exceed the documented 83 Hz upper bound")
        if 1 / control_hz >= cfg.hand.feedback_max_age_s:
            raise ValueError(
                "target period must fit hand.feedback_max_age_s; configure both explicitly"
            )


def resolve_table_plane_path(table: TableCollisionConfig) -> Path:
    """Resolve the configured plane path relative to the repository root."""
    if table.plane_path is None:
        raise RuntimeError("runtime table calibration has no plane_path")
    path = Path(table.plane_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def resolve_table_plane(
    table: TableCollisionConfig, *, required=True,
) -> tuple[float, float, float, float] | None:
    """仅在消费者需要时读取一次当前桌面，数值由装配层共享。"""
    if not required:
        return None
    plane_path = resolve_table_plane_path(table)
    with plane_path.open(encoding="utf-8") as stream:
        plane = json.load(stream)
    return validate_table_plane(tuple(float(plane[k]) for k in ("a", "b", "c", "d")))


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


def validate_keyboard_workspace(runtime):
    """Validate the selected jog configuration before any device connects."""
    cfg = runtime.keyboard_teleop
    cfg.validate()
    validate_control_rate(runtime, cfg.control_hz)
    workspace = runtime.environment.workspace.as_array()
    if np.any(
        workspace[:, 0] + cfg.workspace_command_margin_m
        >= workspace[:, 1] - cfg.workspace_command_margin_m
    ):
        raise ValueError("keyboard workspace command margin leaves no interior workspace")
    return workspace


DEFAULT_TASK_NAME = "test"


def validate_task_dir_name(value: str) -> str:
    """Validate one task name as a safe directory component."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("task_name must be a non-empty string without surrounding whitespace")
    if value.startswith("."):
        raise ValueError("task_name must not be a hidden or relative directory name")
    if "/" in value or "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("task_name must be one safe directory component")
    return value
