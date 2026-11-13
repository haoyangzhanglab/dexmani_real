"""绝对时钟调度、来源时效判断与关节时间插值。"""

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


def validate_joint_gap(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("hoi_max_joint_gap_s 必须为有限正数")


def interpolate_joints(stamps, positions, targets, max_gap_s):
    """返回关节和有效标志；无法插值的位置为 NaN。

    调用方按连续区间提供历史；精确命中直接取样，否则要求前后样本间隔不超预算。
    """
    import numpy as np

    stamps = np.asarray(stamps, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    targets = np.atleast_1d(np.asarray(targets, dtype=np.int64))
    if (
        stamps.ndim != 1
        or not len(stamps)
        or positions.ndim != 2
        or len(positions) != len(stamps)
        or np.any(stamps <= 0)
        or np.any(np.diff(stamps) < 0)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Invalid joint history")
    duplicate = np.diff(stamps) == 0
    if np.any(np.any(positions[1:] != positions[:-1], axis=1) & duplicate):
        raise ValueError("Conflicting joint states at the same timestamp")
    keep = np.r_[True, ~duplicate]
    stamps, positions = stamps[keep], positions[keep]
    right = np.searchsorted(stamps, targets).clip(0, len(stamps) - 1)
    exact = stamps[right] == targets
    left = np.where(exact, right, np.maximum(0, right - 1))
    lo, hi = stamps[left], stamps[right]
    span_ns = hi - lo
    valid = (lo <= targets) & (targets <= hi) & (span_ns <= max_gap_s * 1e9)
    fraction = np.divide(targets - lo, span_ns, out=np.zeros(len(targets)), where=span_ns != 0)
    values = positions[left] + fraction[:, None] * (positions[right] - positions[left])
    values[~valid] = np.nan
    return values, valid
