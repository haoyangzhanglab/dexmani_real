"""Shared serialization utilities for dataclass round-trips.

Provides type-introspection helpers and value-conversion functions that
replace the duplicated ``_is_ndarray_annotation``, ``_is_tuple_annotation``,
``_convert_field_value``, and inline ``from_dict`` logic previously copied
across ``planning/types.py``, ``config/pipeline_config.py``, ``robot/types.py``,
``robot/xarm7/xarm7.py``, and ``robot/xhand/xhand.py``.

All functions in this module are PUBLIC (no underscore prefix) so the
dependent modules can import them directly.
"""

from __future__ import annotations

import dataclasses
import sys
import types as _types
import typing
from typing import Any, get_args, get_origin

import numpy as np

# ---------------------------------------------------------------------------
# Type-introspection helpers
# ---------------------------------------------------------------------------


def is_ndarray_annotation(tp: object) -> bool:
    """Check whether a type annotation represents ``np.ndarray``.

    Returns ``True`` for:
      - ``np.ndarray`` directly
      - ``Optional[np.ndarray]`` (i.e. ``np.ndarray | None`` or
        ``Optional[np.ndarray]``)
      - ``Union[np.ndarray, …]`` with ndarray as one branch

    Works on both ``typing.Union`` and the PEP 604 ``X | Y`` syntax
    (``types.UnionType``, Python 3.10+).

    Args:
        tp: A type annotation object (e.g. the result of
            ``typing.get_type_hints()`` or ``dataclasses.fields(f).type``).

    Returns:
        ``True`` if the annotation ultimately resolves to ``np.ndarray``
        (possibly wrapped in Optional/Union), ``False`` otherwise.

    Examples:
        >>> is_ndarray_annotation(np.ndarray)
        True
        >>> is_ndarray_annotation(np.ndarray | None)
        True
        >>> is_ndarray_annotation(tuple[float, ...])
        False
    """
    if tp is np.ndarray:
        return True

    origin = get_origin(tp)
    if origin is not None:
        # typing.Optional, typing.Union, etc.
        return any(is_ndarray_annotation(a) for a in get_args(tp))

    # Python 3.10+ PEP 604 UnionType (X | Y)
    if sys.version_info >= (3, 10) and isinstance(tp, _types.UnionType):
        return any(is_ndarray_annotation(a) for a in get_args(tp))

    return False


def is_tuple_annotation(tp: object) -> bool:
    """Check whether a type annotation represents a ``tuple`` type.

    Matches ``tuple``, ``tuple[T, ...]``, and ``tuple[T1, T2, ...]``.

    Args:
        tp: A type annotation object.

    Returns:
        ``True`` if ``get_origin(tp) is tuple``, ``False`` otherwise.

    Examples:
        >>> is_tuple_annotation(tuple[float, ...])
        True
        >>> is_tuple_annotation(tuple[int, int])
        True
        >>> is_tuple_annotation(list[float])
        False
    """
    return get_origin(tp) is tuple


# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------


def convert_field_value(val: object, target_tp: object) -> object:
    """Convert a serialized value back to the field's declared type.

    Handles three canonical conversions:

    1. **list → np.ndarray** — when the field annotation represents
       ``np.ndarray`` (see :func:`is_ndarray_annotation`), the list is
       converted via ``np.array(val, dtype=np.float64)``.
    2. **list → tuple** — when the field annotation represents ``tuple``
       (see :func:`is_tuple_annotation`), the list is converted via
       ``tuple(val)``.
    3. **dict → dataclass** — when the target type has a ``from_dict``
       classmethod, ``target_tp.from_dict(val)`` is called recursively.

    If ``val`` is ``None`` it is returned unchanged (handles ``Optional``
    fields). Any value that does not match the above rules is returned
    as-is (identity pass-through for scalars, bools, strings, etc.).

    Args:
        val: The deserialized value (typically from JSON). This will be
            a ``list`` for arrays/tuples, a ``dict`` for nested
            dataclasses, or a scalar.
        target_tp: The target type annotation (e.g. from
            ``typing.get_type_hints(Cls)``).

    Returns:
        The value converted to the declared type.

    Examples:
        >>> convert_field_value([1.0, 2.0, 3.0], np.ndarray)
        array([1., 2., 3.])

        >>> convert_field_value([15.0, 8.0], tuple[float, ...])
        (15.0, 8.0)

        >>> convert_field_value({"path_dt": 0.066}, PlanningProfile)
        PlanningProfile(path_dt=0.066, ...)
    """
    if val is None:
        return None

    if isinstance(val, list):
        if is_ndarray_annotation(target_tp):
            return np.array(val, dtype=np.float64)
        if is_tuple_annotation(target_tp):
            return tuple(val)

    if isinstance(val, dict) and hasattr(target_tp, "from_dict"):
        return target_tp.from_dict(val)

    return val


# ---------------------------------------------------------------------------
# Centralized ``from_dict`` implementation
# ---------------------------------------------------------------------------


def from_dict_helper(cls: type, d: dict[str, Any]) -> dict[str, object]:
    """Build a kwargs dict from a serialized dict for dataclass reconstruction.

    Iterates over every ``dataclasses.field()`` of *cls*, looks up the
    corresponding key in *d*, converts the value via
    :func:`convert_field_value` using the resolved type hint, and returns
    a ``dict[str, object]`` suitable for ``cls(**kwargs)``.

    This is the shared implementation behind all ``from_dict(cls, d)``
    classmethods in the codebase (``PlanningProfile``, ``TeleopProfile``,
    ``PipelineConfig``, ``RobotInterfaceConfig``, ``XArm7Config``,
    ``XHandConfig``).

    Fields that are not present in *d* are skipped (they keep their
    dataclass defaults). Extra keys in *d* that have no matching field
    are silently ignored (forward compatibility).

    Args:
        cls: The dataclass type to reconstruct.
        d: A dictionary (typically from JSON deserialization) whose keys
            correspond to field names of *cls*.

    Returns:
        A ``dict[str, object]`` that can be unpacked into ``cls(**…)``.

    Example:
        >>> kw = from_dict_helper(TeleopProfile, raw_dict)
        >>> profile = TeleopProfile(**kw)
    """
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {f.name: f.type for f in dataclasses.fields(cls)}
    kw: dict[str, object] = {}

    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        target = hints.get(f.name, f.type)
        kw[f.name] = convert_field_value(val, target)

    return kw


class FromDictMixin:
    """Mixin providing ``from_dict(cls, d)`` for dataclass deserialization.

    Usage::

        @dataclass
        class MyConfig(FromDictMixin):
            field1: float = 0.0

        cfg = MyConfig.from_dict({"field1": 1.5})

        # Hot-reload from file (P3.2):
        cfg = MyConfig.from_yaml("configs/profile.yaml")
        cfg = MyConfig.from_json("configs/profile.json")
    """

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Any:
        """Reconstruct from a serialized dict."""
        return cls(**from_dict_helper(cls, d))

    @classmethod
    def from_yaml(cls, path: str) -> Any:
        """Load configuration from a YAML file (hot-reload, P3.2).

        Supports nested dataclass fields — uses from_dict_helper internally
        which handles tuple/list/ndarray conversion, Enum lookup, and nested
        FromDictMixin dataclasses.
        """
        import yaml
        from pathlib import Path

        with open(Path(path), "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        if not isinstance(d, dict):
            raise TypeError(f"YAML file {path} must contain a mapping at the top level, " f"got {type(d).__name__}")
        return cls.from_dict(d)

    @classmethod
    def from_json(cls, path: str) -> Any:
        """Load configuration from a JSON file (hot-reload, P3.2).

        Same semantics as from_yaml but for JSON format.
        """
        import json
        from pathlib import Path

        with open(Path(path), "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise TypeError(f"JSON file {path} must contain a mapping at the top level, " f"got {type(d).__name__}")
        return cls.from_dict(d)
