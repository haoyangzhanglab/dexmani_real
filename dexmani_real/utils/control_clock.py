"""Initial research sampling budget shared by teleop and training admission."""

import math


def sampling_clock_ns(control_hz):
    """Integer grid and dt/4 sampling tolerance; not a measured hardware bound."""
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    dt_ns = round(1e9 / control_hz)
    if dt_ns <= 0:
        raise ValueError("control period must be at least one nanosecond")
    return dt_ns, dt_ns // 4
