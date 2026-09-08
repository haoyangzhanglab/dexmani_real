"""Load independent runtime dataclasses with CLI > YAML > defaults precedence."""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dexmani_real.config import defaults
from dexmani_real.config.defaults import (
    ArmParams,
    CameraParams,
    DexPilotRetargetingParams,
    EnvironmentConfig,
    HandParams,
    KeyboardTeleopParams,
    PolicyParams,
    SafetyParams,
    TAGRetargetingParams,
    VRParams,
)
from dexmani_real.config.pointcloud import PointCloudConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime values validated together at the configuration loading boundary."""

    arm: ArmParams
    hand: HandParams
    policy: PolicyParams
    keyboard_teleop: KeyboardTeleopParams
    vr: VRParams
    safety: SafetyParams
    camera: CameraParams
    pointcloud: PointCloudConfig
    tag_retargeting: TAGRetargetingParams
    dexpilot_retargeting: DexPilotRetargetingParams
    environment: EnvironmentConfig


def config_as_dict(value: Any) -> Any:
    """Convert actual configuration values for on-demand YAML printing."""
    if dataclasses.is_dataclass(value):
        return {
            f.name: config_as_dict(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
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
            _patch(defaults.StaticCollisionBox(), box, f"{path}[{i}]")
            for i, box in enumerate(changes)
        )
    if isinstance(current, bool) and not isinstance(changes, bool):
        raise TypeError(f"config {path!r} must be a boolean")
    if isinstance(current, tuple):
        if not isinstance(changes, (tuple, list)):
            raise TypeError(f"config {path!r} must be an array")
        return tuple(changes)
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


def validate_config(cfg: ExperimentConfig) -> None:
    """Validate numerical and cross-section invariants once before startup."""
    for section in (
        cfg.arm,
        cfg.arm.homing,
        cfg.hand,
        cfg.policy,
        cfg.policy.ema,
        cfg.policy.vr_mapping,
        cfg.policy.workspace,
        cfg.keyboard_teleop,
        cfg.vr,
        cfg.safety,
        cfg.camera,
        cfg.tag_retargeting,
        cfg.dexpilot_retargeting,
        cfg.environment,
        cfg.environment.table,
        *cfg.environment.static_boxes,
    ):
        section.validate()
    # PointCloudConfig owns its external persisted-policy validation in its
    # constructor, shared with processed/Zarr input admission.
    widths = np.diff(cfg.policy.workspace.as_array(), axis=1)
    if 2 * cfg.keyboard_teleop.workspace_command_margin_m >= float(widths.min()):
        raise ValueError(
            "keyboard workspace command margin leaves no interior workspace"
        )


def resolve_experiment_config(
    *,
    yaml_path: str | Path | None = None,
    data: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load YAML/data plus CLI overrides without changing global defaults."""
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
    cfg = ExperimentConfig(
        **{
            f.name: copy.deepcopy(getattr(defaults, f.name))
            for f in dataclasses.fields(ExperimentConfig)
        }
    )
    cfg = _patch(cfg, _overlay(loaded, _expand_dotted(cli_overrides)))
    table = cfg.environment.table
    if table.enabled and table.plane_path is not None:
        plane_path = Path(table.plane_path)
        if not plane_path.is_absolute():
            plane_path = Path(__file__).resolve().parents[2] / plane_path
        try:
            with plane_path.open(encoding="utf-8") as stream:
                plane = json.load(stream)
            table = dataclasses.replace(
                table, plane_abcd=tuple(float(plane[k]) for k in ("a", "b", "c", "d"))
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"failed to load calibrated table plane from {plane_path}: {exc}"
            ) from exc
        cfg = dataclasses.replace(
            cfg, environment=dataclasses.replace(cfg.environment, table=table)
        )
    validate_config(cfg)
    return cfg
