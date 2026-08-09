"""Immutable, validated runtime configuration resolution.

The module-level objects in :mod:`dexmani_real.config.defaults` are convenient
templates, but they are not a safe runtime configuration transport: mutating a
template after another module imported it makes process startup order affect
the effective configuration.  This module resolves a fresh snapshot using the
single precedence rule ``CLI > JSON > defaults`` and gives that snapshot a
canonical JSON representation and SHA-256 identity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_SECTION_NAMES = (
    "arm",
    "hand",
    "policy",
    "keyboard_teleop",
    "vr",
    "safety",
    "camera",
    "tag_retargeting",
)


def _json_value(value: Any) -> Any:
    """Return a deterministic, JSON-compatible copy of *value*."""
    if dataclasses.is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_json_value(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported runtime config value {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenConfigNode(tuple((str(key), _freeze(item)) for key, item in sorted(value.items())))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenConfigNode):
        return {key: _thaw(item) for key, item in value._items}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class FrozenConfigNode(Mapping[str, Any]):
    """Pickle-safe immutable mapping with attribute access."""

    _items: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return {key: _thaw(value) for key, value in self._items}


@dataclass(frozen=True)
class ResolvedRuntimeConfig:
    """Deeply immutable, validated runtime configuration snapshot."""

    arm: FrozenConfigNode
    hand: FrozenConfigNode
    policy: FrozenConfigNode
    keyboard_teleop: FrozenConfigNode
    vr: FrozenConfigNode
    safety: FrozenConfigNode
    camera: FrozenConfigNode
    tag_retargeting: FrozenConfigNode
    canonical_json: str
    sha256: str

    @property
    def config_hash(self) -> str:
        """Alias used by recording and preflight metadata."""
        return self.sha256

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in _SECTION_NAMES}


def _merge(base: dict[str, Any], overrides: Mapping[str, Any], *, path: str = "") -> dict[str, Any]:
    result = {key: _json_value(value) for key, value in base.items()}
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
            result[key] = _json_value(value)
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

    def rebuild(template: Any, raw: Any, path: str) -> Any:
        if raw is None and path == "hand.max_delta_rad":
            return None
        if dataclasses.is_dataclass(template):
            if not isinstance(raw, Mapping):
                raise TypeError(f"runtime config section {path!r} must be an object")
            field_names = {field.name for field in dataclasses.fields(template)}
            unknown = set(raw) - field_names
            if unknown:
                raise TypeError(f"unknown runtime config field(s) in {path}: {sorted(unknown)}")
            kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(template):
                current = getattr(template, field.name)
                value = raw.get(field.name, _json_value(current))
                kwargs[field.name] = rebuild(current, value, f"{path}.{field.name}")
            return type(template)(**kwargs)  # type: ignore[misc]
        if isinstance(template, tuple):
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"runtime config field {path!r} must be an array")
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
        if isinstance(template, dict):
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
    return {name: _json_value(value) for name, value in rebuilt.items()}


def resolve_runtime_config(
    *,
    json_path: str | Path | None = None,
    json_data: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ResolvedRuntimeConfig:
    """Resolve ``CLI > JSON > defaults`` without mutating global defaults.

    CLI keys may be nested mappings or dotted paths such as
    ``{"arm.ip": "192.0.2.1"}``.  ``None`` CLI values mean "not supplied" and
    therefore never mask JSON/default values.
    """
    from dexmani_real.config import defaults

    if json_path is not None and json_data is not None:
        raise ValueError("provide json_path or json_data, not both")
    base = {name: _json_value(getattr(defaults, name)) for name in _SECTION_NAMES}

    file_overrides: Mapping[str, Any] = {}
    if json_path is not None:
        with Path(json_path).open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, Mapping):
            raise TypeError("runtime JSON root must be an object")
        file_overrides = loaded
    elif json_data is not None:
        file_overrides = json_data

    merged = _merge(base, file_overrides)
    merged = _merge(merged, _expand_dotted(cli_overrides))
    validated = _validated_defaults_snapshot(merged)
    canonical = json.dumps(validated, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    frozen = {name: _freeze(validated[name]) for name in _SECTION_NAMES}
    return ResolvedRuntimeConfig(**frozen, canonical_json=canonical, sha256=digest)
