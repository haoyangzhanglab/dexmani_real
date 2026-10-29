"""绝对时钟调度与来源时效判断。"""

from __future__ import annotations

import math
import time


class Rate:
    """按实际循环起点限频；超期立即返回，不追加等待或补跑旧周期。"""

    def __init__(self, hz: float) -> None:
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError(f"hz must be finite and positive, got {hz}")
        self.period = 1.0 / hz
        self.reset()

    def reset(self) -> None:
        self.next_deadline = time.perf_counter() + self.period

    def wait(self, *, service=None) -> None:
        now = time.perf_counter()
        remaining = self.next_deadline - now
        while remaining > 0:
            if service is not None:
                service()
                remaining = self.next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(min(0.005, remaining) if service is not None else remaining)
            now = time.perf_counter()
            remaining = self.next_deadline - now
        self.next_deadline = now + self.period


def next_tick_ns(planned_ns, period_ns, now_ns):
    """跳到时钟网格的下一个未来时刻，不补发已错过的点。"""
    return planned_ns + max(1, (now_ns - planned_ns) // period_ns + 1) * period_ns


def sampling_clock_ns(control_hz):
    """整数采样周期及网格容差；训练导出和严格回放分别选择时钟。"""
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    dt_ns = round(1e9 / control_hz)
    if dt_ns <= 0:
        raise ValueError("control period must be at least one nanosecond")
    return dt_ns, dt_ns // 4


def sample_is_fresh(timestamp_ns, max_age_s, now_ns=None):
    now = time.monotonic_ns() if now_ns is None else now_ns
    stamp = int(timestamp_ns)
    return 0 < stamp <= now and now - stamp <= int(max_age_s * 1e9)
