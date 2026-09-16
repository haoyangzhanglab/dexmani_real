"""Periodic recording deadline calculation."""

from __future__ import annotations

import numpy as np

_UINT64_MAX = int(np.iinfo(np.uint64).max)


def _require_uint64_integer(
    value: object,
    *,
    name: str,
    positive: bool,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    if result > _UINT64_MAX:
        raise ValueError(f"{name} exceeds uint64")
    return result


def next_periodic_deadline_ns(
    deadline_ns: int,
    period_ns: int,
    now_ns: int,
) -> int:
    """Advance an absolute cadence to its first deadline strictly after now."""
    deadline = _require_uint64_integer(
        deadline_ns,
        name="deadline_ns",
        positive=True,
    )
    period = _require_uint64_integer(period_ns, name="period_ns", positive=True)
    now = _require_uint64_integer(now_ns, name="now_ns", positive=True)
    if deadline > now:
        return deadline
    periods = (now - deadline) // period + 1
    offset = periods * period
    if offset > _UINT64_MAX or deadline > _UINT64_MAX - offset:
        raise ValueError("periodic deadline exceeds uint64")
    return deadline + offset
