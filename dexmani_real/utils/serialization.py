"""Strict dataclass deserialization helpers."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, get_args, get_origin

import numpy as np


def _is_ndarray_annotation(tp: object) -> bool:
    """Return whether an annotation contains ``np.ndarray``."""
    if tp is np.ndarray:
        return True

    origin = get_origin(tp)
    if origin is not None:
        return any(_is_ndarray_annotation(a) for a in get_args(tp))

    return False


def _is_tuple_annotation(tp: object) -> bool:
    """Return whether an annotation is a tuple."""
    return get_origin(tp) is tuple


def _convert_field_value(val: object, target_tp: object) -> object:
    """Convert a serialized value back to the field's declared type.

    Handles three canonical conversions:

    1. **list → np.ndarray** — when the field annotation represents
       ``np.ndarray`` (see :func:`_is_ndarray_annotation`), the list is
       converted via ``np.array(val, dtype=np.float64)``.
    2. **list → tuple** — when the field annotation represents ``tuple``
       (see :func:`_is_tuple_annotation`), the list is converted via
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
        >>> _convert_field_value([1.0, 2.0, 3.0], np.ndarray)
        array([1., 2., 3.])

        >>> _convert_field_value([15.0, 8.0], tuple[float, ...])
        (15.0, 8.0)

        >>> _convert_field_value({"path_dt": 0.066}, PlanningProfile)
        PlanningProfile(path_dt=0.066, ...)
    """
    if val is None:
        return None

    if isinstance(val, list):
        if _is_ndarray_annotation(target_tp):
            return np.array(val, dtype=np.float64)
        if _is_tuple_annotation(target_tp):
            return tuple(val)

    if isinstance(val, dict) and hasattr(target_tp, "from_dict"):
        return target_tp.from_dict(val)

    return val


def from_dict_helper(cls: type, d: dict[str, Any]) -> dict[str, object]:
    """Build dataclass kwargs while rejecting unknown serialized fields."""
    hints = typing.get_type_hints(cls)
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(d) - set(fields))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    kw: dict[str, object] = {}

    for name, field in fields.items():
        if name not in d:
            continue
        target = hints.get(name, field.type)
        kw[name] = _convert_field_value(d[name], target)

    return kw
