"""Nominal control period and offline export timing tolerance."""

import math


def next_tick_ns(planned_ns, period_ns, now_ns):
    """Next future point of the absolute grid; never burst-send missed points."""
    return planned_ns + max(1, (now_ns - planned_ns) // period_ns + 1) * period_ns


def sampling_clock_ns(control_hz):
    """Integer grid and dt/4 export tolerance; not a control-stop threshold."""
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    dt_ns = round(1e9 / control_hz)
    if dt_ns <= 0:
        raise ValueError("control period must be at least one nanosecond")
    return dt_ns, dt_ns // 4
