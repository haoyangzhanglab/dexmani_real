"""Pure timing calculations for immutable learned-policy action grids."""

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


def first_future_step_index(
    logical_start_ns: int,
    step_dt_ns: int,
    now_ns: int,
    num_steps: int,
) -> int | None:
    """Return the first action index whose logical target is not in the past."""
    logical_start = _require_uint64_integer(
        logical_start_ns,
        name="logical_start_ns",
        positive=True,
    )
    step_dt = _require_uint64_integer(
        step_dt_ns,
        name="step_dt_ns",
        positive=True,
    )
    now = _require_uint64_integer(now_ns, name="now_ns", positive=True)
    count = _require_uint64_integer(num_steps, name="num_steps", positive=True)
    final_offset = (count - 1) * step_dt
    if final_offset > _UINT64_MAX or logical_start > _UINT64_MAX - final_offset:
        raise ValueError("action timeline exceeds uint64")
    if now <= logical_start:
        return 0
    index = (now - logical_start + step_dt - 1) // step_dt
    return int(index) if index < count else None


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
