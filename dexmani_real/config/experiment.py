"""Load package defaults with explicit experiment and CLI overrides."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dexmani_real.config.control import (
    ControlParams,
    DexPilotRetargetingParams,
    KeyboardTeleopParams,
    SafetyParams,
    TAGRetargetingParams,
    TeleopTimingParams,
)
from dexmani_real.config.environment import (
    EnvironmentConfig,
    StaticCollisionBox,
    TableCollisionConfig,
)
from dexmani_real.config.hardware import ArmParams, CameraParams, HandParams, VRParams
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.deployment.config import ExecutionConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment values loaded together and validated at their usage boundaries."""

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    arm: ArmParams = field(default_factory=ArmParams)
    hand: HandParams = field(default_factory=HandParams)
    policy: ControlParams = field(default_factory=ControlParams)
    teleop: TeleopTimingParams = field(default_factory=TeleopTimingParams)
    keyboard_teleop: KeyboardTeleopParams = field(default_factory=KeyboardTeleopParams)
    vr: VRParams = field(default_factory=VRParams)
    safety: SafetyParams = field(default_factory=SafetyParams)
    camera: CameraParams = field(default_factory=CameraParams)
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    tag_retargeting: TAGRetargetingParams = field(default_factory=TAGRetargetingParams)
    dexpilot_retargeting: DexPilotRetargetingParams = field(
        default_factory=DexPilotRetargetingParams
    )
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def _patch(current: Any, changes: Any) -> Any:
    """Update dataclass sections; numerical constraints belong to consumers."""
    if dataclasses.is_dataclass(current):
        if not isinstance(changes, Mapping):
            raise TypeError(f"{type(current).__name__} overrides must be a mapping")
        updated = dict(changes)
        if isinstance(current, EnvironmentConfig) and "static_boxes" in updated:
            updated["static_boxes"] = tuple(
                _patch(StaticCollisionBox(), box) for box in updated["static_boxes"]
            )
        for name, value in updated.items():
            if hasattr(current, name):
                updated[name] = _patch(getattr(current, name), value)
        # The dataclass constructor rejects unknown fields.
        return dataclasses.replace(current, **updated)
    if isinstance(current, tuple):
        if not isinstance(changes, (list, tuple)):
            raise TypeError("config arrays must use YAML sequences")
        return tuple(_patch(current[0], value) for value in changes) if current else tuple(changes)
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
    for section in (cfg.arm, cfg.arm.homing, cfg.hand, cfg.safety):
        section.validate()


def validate_control_rate(cfg: ExperimentConfig, control_hz: float) -> None:
    """Validate cadence before connecting; never silently decimate hand targets."""
    if not np.isfinite(control_hz) or not 0 < control_hz <= cfg.arm.servo_hz:
        raise ValueError('target frequency must be positive and no greater than arm.servo_hz')
    if 1 / control_hz >= cfg.arm.feedback_max_age_s:
        raise ValueError('target period must fit arm.feedback_max_age_s (coordinator health budget)')
    if cfg.policy.hand_enabled:
        if control_hz > 83:
            raise ValueError('XHand targets exceed the documented 83 Hz upper bound')
        if 1 / control_hz >= cfg.hand.feedback_max_age_s:
            raise ValueError('target period must fit hand.feedback_max_age_s; configure both explicitly')


def resolve_runtime_table(cfg: ExperimentConfig, *, pointcloud=None) -> ExperimentConfig:
    """One session snapshot for HOME and the effective (possibly saved) cloud recipe."""
    table = cfg.environment.table
    if table.enabled or (pointcloud is not None and pointcloud.remove_table):
        table = dataclasses.replace(table, plane_abcd=resolve_table_plane(table))
        cfg = dataclasses.replace(
            cfg, environment=dataclasses.replace(cfg.environment, table=table)
        )
    return cfg


def resolve_table_plane_path(table: TableCollisionConfig) -> Path:
    """Resolve the configured plane path relative to the repository root."""
    if table.plane_path is None:
        raise RuntimeError("runtime table calibration has no plane_path")
    path = Path(table.plane_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def resolve_table_plane(table: TableCollisionConfig) -> tuple[float, float, float, float]:
    """Read the current plane file, or use inline geometry when plane_path is null."""
    if table.plane_path is None:
        table.validate()
        return tuple(float(value) for value in table.plane_abcd)
    plane_path = resolve_table_plane_path(table)
    try:
        with plane_path.open(encoding="utf-8") as stream:
            plane = json.load(stream)
        resolved = dataclasses.replace(
            table, plane_abcd=tuple(float(plane[k]) for k in ("a", "b", "c", "d"))
        )
        resolved.validate()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"failed to load calibrated table plane from {plane_path}: {exc}") from exc
    return resolved.plane_abcd


def load_experiment_config(
    *,
    yaml_path: str | Path | None = None,
    data: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Merge CLI > YAML/data > defaults without hardware/calibration I/O; consumers validate runtime values."""
    if yaml_path is not None and data is not None:
        raise ValueError("provide at most one of yaml_path or data")
    loaded = data if data is not None else {}
    if yaml_path is not None:
        path = Path(yaml_path)
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if loaded is None:
            loaded = {}
    if not isinstance(loaded, Mapping):
        raise TypeError("experiment config root must be a mapping")
    loaded = deepcopy(dict(loaded))
    for key, value in (cli_overrides or {}).items():
        if value is None:
            continue
        *sections, name = key.split(".")
        cursor = loaded
        for section in sections:
            if not isinstance(cursor.get(section), dict):
                cursor[section] = {}
            cursor = cursor[section]
        cursor[name] = value
    cfg = _patch(ExperimentConfig(), loaded)
    if cfg.execution.action_steps is not None:
        raise ValueError("execution.action_steps is derived; use --n-action-steps or the saved Policy default")
    return cfg
