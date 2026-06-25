"""Rate manager with precise wait and per-stream statistics.

Provides:
  - RateManager: high-precision sleep using busy-wait hybrid,
    achieving < 1ms target error (vs ~15ms for time.sleep()).
  - StreamStats: per-stream drop rate, frame age, throughput tracking.

Ref: LeFranX Ruckig rate decoupling.
     BunnyVisionPro wait_until_next_control_signal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


# ── Rate Manager ──


class RateManager:
    """High-precision rate limiter with hybrid busy-wait + sleep.

    Unlike RateLimiter (which uses time.sleep()), RateManager uses a
    hybrid approach: sleep for 95% of the remaining time, then busy-wait
    for the final ~1ms. This achieves < 1ms target error on Linux with
    PREEMPT_RT or standard kernels.

    Usage:
        rm = RateManager(50.0)  # 50 Hz
        while running:
            do_work()
            rm.wait()
    """

    def __init__(self, target_hz: float) -> None:
        if target_hz <= 0:
            raise ValueError(f"target_hz must be positive, got {target_hz}")
        self.target_hz = float(target_hz)
        self.period = 1.0 / target_hz
        self._last_wake = time.perf_counter()

        self._total_cycles: int = 0
        self._overdue_cycles: int = 0
        self._total_sleep_err_s: float = 0.0
        self._max_sleep_err_s: float = 0.0
        self._overdue_throttle: int = 0

    def wait(self) -> None:
        """Sleep until the next control cycle boundary with precision.

        Hybrid strategy:
          1. Compute remaining time
          2. If > 2ms: time.sleep(remaining - 1ms)
          3. Busy-wait for the final precision window
        """
        elapsed = time.perf_counter() - self._last_wake
        remaining = self.period - elapsed

        if remaining > 0.002:  # > 2ms: sleep for bulk
            time.sleep(remaining - 0.001)
            # After sleep, busy-wait for precision
            while time.perf_counter() - self._last_wake < self.period:
                pass  # spin
        elif remaining > 0:
            # Short window: pure busy-wait
            while time.perf_counter() - self._last_wake < self.period:
                pass  # spin
        else:
            # Overdue: emit throttled warning
            self._overdue_cycles += 1
            if self._overdue_throttle <= 0:
                logger.warning(
                    "Control loop over budget: actual=%.1fms target=%.1fms "
                    "(overdue %d/%d cycles)",
                    elapsed * 1000, self.period * 1000,
                    self._overdue_cycles, self._total_cycles + 1,
                )
                self._overdue_throttle = 50
            else:
                self._overdue_throttle -= 1

        # Track sleep error
        actual_elapsed = time.perf_counter() - self._last_wake
        err = abs(actual_elapsed - self.period)
        self._total_sleep_err_s += err
        if err > self._max_sleep_err_s:
            self._max_sleep_err_s = err

        self._total_cycles += 1
        self._last_wake = time.perf_counter()

    def reset(self) -> None:
        self._last_wake = time.perf_counter()
        self._overdue_throttle = 0
        self._overdue_cycles = 0
        self._total_cycles = 0
        self._total_sleep_err_s = 0.0
        self._max_sleep_err_s = 0.0

    @property
    def mean_sleep_error_ms(self) -> float:
        if self._total_cycles == 0:
            return 0.0
        return (self._total_sleep_err_s / self._total_cycles) * 1000.0

    @property
    def max_sleep_error_ms(self) -> float:
        return self._max_sleep_err_s * 1000.0

    @property
    def overdue_ratio(self) -> float:
        if self._total_cycles == 0:
            return 0.0
        return self._overdue_cycles / self._total_cycles


# ── Stream Statistics ──


@dataclass
class StreamStats:
    """Per-stream statistics for monitoring and diagnostics.

    Tracks produce rate, consume rate, drop rate, and frame age.
    """

    name: str = "unknown"
    target_hz: float = 0.0

    # Counters
    produced: int = 0
    consumed: int = 0
    dropped: int = 0
    stale: int = 0  # frames consumed but too old

    # Timing
    last_produce_ts: float = 0.0
    last_consume_ts: float = 0.0
    last_interval_s: float = 0.0

    # Frame age EMA
    age_ema_alpha: float = 0.1
    age_ema_s: float = 0.0
    max_age_s: float = 0.0
    min_age_s: float = float("inf")

    # Windowed stats (last N frames)
    _age_window: list[float] = field(default_factory=list)
    WINDOW_SIZE: int = 100

    def record_produce(self) -> None:
        now = time.perf_counter()
        if self.last_produce_ts > 0:
            self.last_interval_s = now - self.last_produce_ts
        self.last_produce_ts = now
        self.produced += 1

    def record_consume(self, age_s: float) -> None:
        now = time.perf_counter()
        self.last_consume_ts = now
        self.consumed += 1

        # Age tracking
        if age_s > self.max_age_s:
            self.max_age_s = age_s
        if age_s < self.min_age_s:
            self.min_age_s = age_s
        self.age_ema_s = (
            self.age_ema_alpha * age_s + (1 - self.age_ema_alpha) * self.age_ema_s
        )
        self._age_window.append(age_s)
        if len(self._age_window) > self.WINDOW_SIZE:
            self._age_window.pop(0)

    def record_drop(self) -> None:
        self.dropped += 1

    def record_stale(self) -> None:
        self.stale += 1

    @property
    def drop_rate(self) -> float:
        total = self.produced
        if total == 0:
            return 0.0
        return self.dropped / total

    @property
    def stale_rate(self) -> float:
        total = self.consumed
        if total == 0:
            return 0.0
        return self.stale / total

    @property
    def observed_hz(self) -> float:
        total = self.produced
        if total < 2 or self.last_produce_ts == 0:
            return 0.0
        # Estimate from recent intervals
        if len(self._age_window) >= 2:
            return 1.0 / max(self.last_interval_s, 1e-9)
        return 0.0

    @property
    def mean_age_s(self) -> float:
        return self.age_ema_s

    @property
    def p99_age_s(self) -> float:
        if not self._age_window:
            return 0.0
        return float(np.percentile(self._age_window, 99))

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "produced": self.produced,
            "consumed": self.consumed,
            "dropped": self.dropped,
            "stale": self.stale,
            "drop_rate": round(self.drop_rate, 4),
            "stale_rate": round(self.stale_rate, 4),
            "observed_hz": round(self.observed_hz, 1),
            "mean_age_ms": round(self.mean_age_s * 1000, 1),
            "p99_age_ms": round(self.p99_age_s * 1000, 1),
            "max_age_ms": round(self.max_age_s * 1000, 1),
            "min_age_ms": round(self.min_age_s * 1000, 1),
        }


# ── Frame Drop Policy ──


@dataclass
class FrameDropPolicy:
    """Configuration for multi-tier frame drop behavior.

    Three tiers of staleness:
      - Tier 1 (Soft Stale, > soft_threshold_s): Linear interpolation (soft decel)
      - Tier 2 (Hard Stale, > hard_threshold_s): Hold position, mark TRACKING_LOST
      - Tier 3 (Emergency,   > emergency_threshold_s): Emergency stop

    Ref: DexUMI drop-oldest backpressure strategy.
    """

    soft_threshold_s: float = 0.1    # > 100ms → soft deceleration
    hard_threshold_s: float = 0.5    # > 500ms → hold position
    emergency_threshold_s: float = 1.0  # > 1s → emergency stop

    def classify(self, age_s: float) -> int:
        """Classify frame age into tier 0 (fresh), 1, 2, or 3 (emergency).

        Returns 0 for fresh frames, 1/2/3 for escalating staleness.
        """
        if age_s <= self.soft_threshold_s:
            return 0  # fresh
        if age_s <= self.hard_threshold_s:
            return 1  # soft stale
        if age_s <= self.emergency_threshold_s:
            return 2  # hard stale
        return 3  # emergency

    @property
    def tier_names(self) -> dict[int, str]:
        return {0: "FRESH", 1: "SOFT_STALE", 2: "HARD_STALE", 3: "EMERGENCY"}
