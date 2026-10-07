"""Load independent runtime dataclasses with CLI > YAML > defaults precedence."""

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


def config_as_dict(value: Any) -> Any:
    """Convert configuration values for printing and run-configuration serialization."""
    if dataclasses.is_dataclass(value):
        return {f.name: config_as_dict(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {key: config_as_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [config_as_dict(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _patch(current: Any, changes: Any, path: str = "") -> Any:
    """Apply external fields to concrete defaults; no annotation reconstruction."""
    if dataclasses.is_dataclass(current) or isinstance(current, Mapping):
        if not isinstance(changes, Mapping):
            raise TypeError(f"config {path!r} must be an object")
        fields = (
            {f.name: getattr(current, f.name) for f in dataclasses.fields(current)}
            if dataclasses.is_dataclass(current)
            else dict(current)
        )
        unknown = set(changes) - fields.keys()
        if unknown:
            raise TypeError(f"unknown config fields in {path!r}: {sorted(unknown)}")
        updated = {
            name: _patch(fields[name], value, f"{path}.{name}".lstrip("."))
            for name, value in changes.items()
        }
        return (
            dataclasses.replace(current, **updated)
            if dataclasses.is_dataclass(current)
            else fields | updated
        )
    if path == "environment.static_boxes":
        if not isinstance(changes, (tuple, list)):
            raise TypeError("environment.static_boxes must be an array")
        return tuple(
            _patch(StaticCollisionBox(), box, f"{path}[{i}]") for i, box in enumerate(changes)
        )
    if isinstance(current, bool) and not isinstance(changes, bool):
        raise TypeError(f"config {path!r} must be a boolean")
    if isinstance(current, tuple):
        if not isinstance(changes, (tuple, list)):
            raise TypeError(f"config {path!r} must be an array")
        if not current:
            return tuple(changes)
        return tuple(
            _patch(current[min(i, len(current) - 1)], value, f"{path}[{i}]")
            for i, value in enumerate(changes)
        )
    if current is None:
        if path in {"execution.prefetch_steps", "camera.l515_confidence_threshold"}:
            if changes is not None and (type(changes) is not int):
                raise TypeError(f"config {path!r} must be an integer or null")
        elif path.startswith("execution."):
            if changes is not None and (
                isinstance(changes, bool) or not isinstance(changes, (int, float))
            ):
                raise TypeError(f"config {path!r} must be numeric or null")
        elif changes is not None and not isinstance(changes, str):
            raise TypeError(f"config {path!r} must be a string or null")
        return changes
    if isinstance(current, bool):
        return changes
    if isinstance(current, int):
        if type(changes) is not int:
            raise TypeError(f"config {path!r} must be an integer")
    elif isinstance(current, (float, np.floating)):
        if isinstance(changes, bool) or not isinstance(changes, (int, float)):
            raise TypeError(f"config {path!r} must be numeric")
        return float(changes)
    elif isinstance(current, str):
        if changes is None and path == "environment.table.plane_path":
            return None
        if not isinstance(changes, str):
            raise TypeError(f"config {path!r} must be a string")
    return changes


def _expand_dotted(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for raw_key, value in (overrides or {}).items():
        if value is None:
            continue
        cursor = expanded
        parts = str(raw_key).split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise TypeError(f"conflicting CLI config paths at {raw_key!r}")
        cursor[parts[-1]] = value
    return expanded


def _overlay(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for name, value in overrides.items():
        previous = result.get(name)
        result[name] = (
            _overlay(previous, value)
            if isinstance(previous, Mapping) and isinstance(value, Mapping)
            else value
        )
    return result


def validate_robot_config(cfg: ExperimentConfig) -> None:
    """Values used by every robot owner, including HOME and hand-disabled geometry."""
    for section in (cfg.arm, cfg.arm.homing, cfg.hand, cfg.safety):
        section.validate()


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
    """Merge YAML/data and CLI over defaults without runtime checks or table-file I/O."""
    if yaml_path is not None and data is not None:
        raise ValueError("provide at most one of yaml_path or data")
    loaded = data if data is not None else {}
    if yaml_path is not None:
        path = Path(yaml_path)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("experiment config path must use a .yaml or .yml suffix")
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if loaded is None:
            loaded = {}
    if not isinstance(loaded, Mapping):
        raise TypeError("experiment config root must be a mapping")
    cfg = ExperimentConfig()
    return _patch(cfg, _overlay(loaded, _expand_dotted(cli_overrides)))
