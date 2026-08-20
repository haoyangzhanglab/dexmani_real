"""Resolve an immutable experiment configuration from YAML.

The module-level objects in :mod:`dexmani_real.config.defaults` are convenient
templates, but they are not a safe runtime configuration transport: mutating a
template after another module imported it makes process startup order affect
the effective configuration. This module resolves a fresh snapshot using the
single precedence rule ``CLI > file > defaults`` and gives that snapshot a
stable canonical-JSON SHA-256 identity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, get_args, get_origin, get_type_hints

import numpy as np
import yaml

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
from dexmani_real.config.defaults import arm as arm_defaults

_SECTION_NAMES = (
    "arm",
    "hand",
    "policy",
    "keyboard_teleop",
    "vr",
    "safety",
    "camera",
    "tag_retargeting",
    "dexpilot_retargeting",
    "environment",
)


def _plain_value(value: Any) -> Any:
    """Return a deterministic YAML-safe copy of *value*."""
    if dataclasses.is_dataclass(value):
        return {field.name: _plain_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_plain_value(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported runtime config value {type(value).__name__}")


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    """Deeply immutable, validated and statically named runtime snapshot."""

    arm: ArmParams
    hand: HandParams
    policy: PolicyParams
    keyboard_teleop: KeyboardTeleopParams
    vr: VRParams
    safety: SafetyParams
    camera: CameraParams
    tag_retargeting: TAGRetargetingParams
    dexpilot_retargeting: DexPilotRetargetingParams
    environment: EnvironmentConfig
    canonical_json: str
    canonical_yaml: str
    sha256: str


@dataclass
class ArmLoopConfig:
    """Arm-worker projection of :class:`ArmParams` with unit conversion.

    Projects the frozen defaults into the units the arm worker and driver
    consume (radians, Hz).  Transitional duplicate: ``runtime.arm`` remains
    the source of truth.
    """

    joint_max_speed_rad_per_s: float = field(
        default_factory=lambda: arm_defaults.max_joint_velocity_rad_per_s
    )
    joint_max_acc_rad_per_s2: float = field(
        default_factory=lambda: arm_defaults.max_joint_acceleration_rad_per_s2
    )
    arm_loop_hz: float = field(default_factory=lambda: arm_defaults.loop_hz)

    joint_limit_lower: tuple[float, ...] = field(
        default_factory=lambda: arm_defaults.joint_limit_lower
    )
    joint_limit_upper: tuple[float, ...] = field(
        default_factory=lambda: arm_defaults.joint_limit_upper
    )

    arm_ip: str = field(default_factory=lambda: arm_defaults.ip)

    home_qpos: tuple[float, ...] = field(default_factory=lambda: arm_defaults.home_qpos)

    collision_sensitivity: int = field(
        default_factory=lambda: arm_defaults.collision_sensitivity
    )

    expected_axis: int = field(default_factory=lambda: arm_defaults.expected_axis)
    device_profile: str | None = field(default_factory=lambda: arm_defaults.device_profile)

    homing_convergence_rad: float = field(
        default_factory=lambda: arm_defaults.homing.convergence_rad
    )
    homing_step_interval_s: float = field(
        default_factory=lambda: arm_defaults.homing.step_interval_s
    )
    homing_max_speed_rad_per_s: float = field(
        default_factory=lambda: np.deg2rad(arm_defaults.homing.max_speed_deg_s)
    )
    homing_target_timeout_s: float = field(
        default_factory=lambda: arm_defaults.homing.target_timeout_s
    )
    homing_velocity_convergence_rad_s: float = field(
        default_factory=lambda: arm_defaults.homing.velocity_convergence_rad_s
    )
    homing_dwell_s: float = field(default_factory=lambda: arm_defaults.homing.dwell_s)

    tcp_load_mass_kg: float = field(default_factory=lambda: arm_defaults.tcp_load_mass_kg)
    tcp_load_cog_mm: tuple[float, float, float] = field(
        default_factory=lambda: arm_defaults.tcp_load_cog_mm
    )

    @classmethod
    def from_runtime(cls, runtime: Any) -> "ArmLoopConfig":
        cfg = runtime.arm
        return cls(
            joint_max_speed_rad_per_s=float(
                np.deg2rad(cfg.max_joint_velocity_deg_per_s)
            ),
            joint_max_acc_rad_per_s2=float(
                np.deg2rad(cfg.max_joint_acceleration_deg_per_s2)
            ),
            arm_loop_hz=float(cfg.loop_hz),
            joint_limit_lower=tuple(cfg.joint_limit_lower),
            joint_limit_upper=tuple(cfg.joint_limit_upper),
            arm_ip=str(cfg.ip),
            home_qpos=tuple(cfg.home_qpos),
            collision_sensitivity=int(cfg.collision_sensitivity),
            expected_axis=int(cfg.expected_axis),
            device_profile=cfg.device_profile,
            homing_convergence_rad=float(cfg.homing.convergence_rad),
            homing_step_interval_s=float(cfg.homing.step_interval_s),
            homing_max_speed_rad_per_s=float(np.deg2rad(cfg.homing.max_speed_deg_s)),
            homing_target_timeout_s=float(cfg.homing.target_timeout_s),
            homing_velocity_convergence_rad_s=float(
                cfg.homing.velocity_convergence_rad_s
            ),
            homing_dwell_s=float(cfg.homing.dwell_s),
            tcp_load_mass_kg=float(cfg.tcp_load_mass_kg),
            tcp_load_cog_mm=tuple(cfg.tcp_load_cog_mm),
        )


def _merge(base: dict[str, Any], overrides: Mapping[str, Any], *, path: str = "") -> dict[str, Any]:
    result = {key: _plain_value(value) for key, value in base.items()}
    for key, value in overrides.items():
        key = str(key)
        location = f"{path}.{key}" if path else key
        if key not in result:
            raise TypeError(f"unknown runtime config field {location!r}")
        if isinstance(result[key], dict):
            if not isinstance(value, Mapping):
                raise TypeError(f"runtime config field {location!r} must be an object")
            result[key] = _merge(result[key], value, path=location)
        else:
            result[key] = _plain_value(value)
    return result


def _expand_dotted(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for raw_key, value in (overrides or {}).items():
        if value is None:
            continue
        cursor = expanded
        parts = str(raw_key).split(".")
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise TypeError(f"conflicting CLI config paths at {raw_key!r}")
            cursor = existing
        cursor[parts[-1]] = value
    return expanded


def _validated_defaults_snapshot(data: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild every defaults dataclass so all field validators run."""
    from dexmani_real.config import defaults

    def rebuild(template: Any, raw: Any, path: str, annotation: Any | None = None) -> Any:
        if raw is None and path in {"environment.table.plane_path"}:
            return None
        if dataclasses.is_dataclass(template):
            if not isinstance(raw, Mapping):
                raise TypeError(f"runtime config section {path!r} must be an object")
            field_names = {field.name for field in dataclasses.fields(template)}
            unknown = set(raw) - field_names
            if unknown:
                raise TypeError(f"unknown runtime config field(s) in {path}: {sorted(unknown)}")
            kwargs: dict[str, Any] = {}
            type_hints = get_type_hints(type(template))
            for field in dataclasses.fields(template):
                current = getattr(template, field.name)
                value = raw.get(field.name, _plain_value(current))
                kwargs[field.name] = rebuild(
                    current,
                    value,
                    f"{path}.{field.name}",
                    type_hints.get(field.name),
                )
            return type(template)(**kwargs)  # type: ignore[misc]
        if isinstance(template, tuple):
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"runtime config field {path!r} must be an array")
            origin = get_origin(annotation)
            args = get_args(annotation)
            if not template and origin is tuple and len(args) == 2 and args[1] is Ellipsis:
                item_type = args[0]
                if dataclasses.is_dataclass(item_type):
                    item_factory = cast(Any, item_type)
                    return tuple(rebuild(item_factory(), item, f"{path}[{index}]") for index, item in enumerate(raw))
                return tuple(raw)
            if len(raw) != len(template):
                raise ValueError(f"runtime config field {path!r} must contain {len(template)} values")
            return tuple(
                rebuild(item, value, f"{path}[{index}]") for index, (item, value) in enumerate(zip(template, raw))
            )
        if isinstance(template, frozenset):
            if not isinstance(raw, (list, tuple, set, frozenset)):
                raise TypeError(f"runtime config field {path!r} must be an array")
            if not template:
                return frozenset(raw)
            prototype = next(iter(template))
            return frozenset(rebuild(prototype, value, f"{path}[]") for value in raw)
        if isinstance(template, Mapping):
            if not isinstance(raw, Mapping):
                raise TypeError(f"runtime config field {path!r} must be an object")
            unknown = set(raw) - set(template)
            if unknown:
                raise TypeError(f"unknown runtime config field(s) in {path}: {sorted(unknown)}")
            return {key: rebuild(template[key], raw.get(key, template[key]), f"{path}.{key}") for key in template}
        if template is None:
            if raw is not None and not isinstance(raw, str):
                raise TypeError(f"runtime config field {path!r} must be a string or null")
            return raw
        if isinstance(template, bool):
            if not isinstance(raw, bool):
                raise TypeError(f"runtime config field {path!r} must be a boolean")
            return raw
        if isinstance(template, int):
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TypeError(f"runtime config field {path!r} must be an integer")
            return int(raw)
        if isinstance(template, float):
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not np.isfinite(raw):
                raise TypeError(f"runtime config field {path!r} must be a finite number")
            return float(raw)
        if isinstance(template, str):
            if not isinstance(raw, str):
                raise TypeError(f"runtime config field {path!r} must be a string")
            return raw
        return raw

    rebuilt: dict[str, Any] = {}
    for name in _SECTION_NAMES:
        rebuilt[name] = rebuild(getattr(defaults, name), data[name], name)

    # Cross-section validation belongs here rather than in a worker.
    if rebuilt["camera"].recording_stall_abort_s <= rebuilt["camera"].max_frame_age_s:
        raise ValueError("camera.recording_stall_abort_s must exceed camera.max_frame_age_s")
    if rebuilt["policy"].control_hz <= 0 or rebuilt["arm"].loop_hz <= 0 or rebuilt["hand"].loop_hz <= 0:
        raise ValueError("all configured control rates must be positive")
    workspace = rebuilt["policy"].workspace
    if workspace.x_min > workspace.x_max or workspace.y_min > workspace.y_max or workspace.z_min > workspace.z_max:
        raise ValueError("policy.workspace lower bounds must not exceed upper bounds")
    workspace_widths = np.array(
        [workspace.x_max - workspace.x_min, workspace.y_max - workspace.y_min, workspace.z_max - workspace.z_min],
        dtype=np.float64,
    )
    if 2.0 * rebuilt["keyboard_teleop"].workspace_command_margin_m >= float(np.min(workspace_widths)):
        raise ValueError("keyboard workspace command margin leaves no interior workspace")
    return rebuilt


def resolve_runtime_config(
    *,
    yaml_path: str | Path | None = None,
    data: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ResolvedRuntimeConfig:
    """Resolve ``CLI > file/data > defaults`` without mutating global defaults.

    CLI keys may be nested mappings or dotted paths such as
    ``{"arm.ip": "192.0.2.1"}``.  ``None`` CLI values mean "not supplied" and
    therefore never mask file/default values.
    """
    from dexmani_real.config import defaults

    sources = [yaml_path is not None, data is not None]
    if sum(sources) > 1:
        raise ValueError("provide at most one of yaml_path or data")
    base = {name: _plain_value(getattr(defaults, name)) for name in _SECTION_NAMES}

    file_overrides: Mapping[str, Any] = {}
    if yaml_path is not None:
        config_path = Path(yaml_path)
        if config_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("experiment config path must use a .yaml or .yml suffix")
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise TypeError("experiment config root must be a mapping")
        file_overrides = loaded
    elif data is not None:
        file_overrides = data

    merged = _merge(base, file_overrides)
    merged = _merge(merged, _expand_dotted(cli_overrides))
    table_config = merged["environment"]["table"]
    if bool(table_config["enabled"]) and table_config["plane_path"] is not None:
        plane_path = Path(str(table_config["plane_path"]))
        if not plane_path.is_absolute():
            plane_path = Path(__file__).resolve().parents[2] / plane_path
        try:
            with plane_path.open("r", encoding="utf-8") as stream:
                plane_data = json.load(stream)
            table_config["plane_abcd"] = [float(plane_data[name]) for name in ("a", "b", "c", "d")]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load calibrated table plane from {plane_path}: {exc}") from exc
    sections = _validated_defaults_snapshot(merged)
    validated = {name: _plain_value(value) for name, value in sections.items()}
    canonical_json = json.dumps(
        validated,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    canonical_yaml = yaml.safe_dump(validated, allow_unicode=True, sort_keys=True)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return ResolvedRuntimeConfig(
        **sections,
        canonical_json=canonical_json,
        canonical_yaml=canonical_yaml,
        sha256=digest,
    )
