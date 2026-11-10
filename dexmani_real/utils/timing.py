"""绝对时钟调度与来源时效判断。"""

from __future__ import annotations

import math


def next_tick_ns(planned_ns, period_ns, now_ns):
    """跳到时钟网格的下一个未来时刻，不补发已错过的点。"""
    return planned_ns + max(1, (now_ns - planned_ns) // period_ns + 1) * period_ns


def sampling_clock_ns(control_hz: float) -> int:
    """验证采样频率并返回整数纳秒周期。"""
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    period_ns = 1e9 / control_hz
    if not math.isfinite(period_ns):
        raise ValueError("control period must be finite")
    dt_ns = round(period_ns)
    if dt_ns <= 0:
        raise ValueError("control period must be at least one nanosecond")
    return dt_ns


def sample_is_fresh(timestamp_ns, max_age_s, now_ns):
    stamp = int(timestamp_ns)
    return 0 < stamp <= now_ns and now_ns - stamp < int(max_age_s * 1e9)


def validate_hoi_skew_budget(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("联合 HOI 输入必须显式配置有限正数 hoi_max_skew_s")


def hoi_sources_within_skew(stamps, budget_s):
    return min(stamps) > 0 and max(stamps) - min(stamps) <= budget_s * 1e9
