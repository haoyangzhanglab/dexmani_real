"""绝对时钟调度与来源时效判断。"""

from __future__ import annotations

import math


def next_tick_ns(planned_ns, period_ns, now_ns):
    """跳到时钟网格的下一个未来时刻，不补发已错过的点。"""
    return planned_ns + max(1, (now_ns - planned_ns) // period_ns + 1) * period_ns


def sampling_clock_ns(control_hz):
    """将采样频率转换为有效整数周期；第二项为相邻采样间隔使用的四分之一周期容差。"""
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    dt_ns = round(1e9 / control_hz)
    if dt_ns <= 0:
        raise ValueError("control period must be at least one nanosecond")
    return dt_ns, dt_ns // 4


def sample_is_fresh(timestamp_ns, max_age_s, now_ns):
    stamp = int(timestamp_ns)
    return 0 < stamp <= now_ns and now_ns - stamp < int(max_age_s * 1e9)
