"""Absolute-deadline rate limiting with overrun accounting."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RateStats:
    """Immutable snapshot of one rate manager's accumulated loop health."""

    target_period_s: float
    loop_count: int
    last_work_duration_s: float
    max_work_duration_s: float
    deadline_overrun_count: int
    missed_slot_count: int
    long_block_reanchor_count: int
    elapsed_s: float
    actual_hz: float


class RateManager:
    """Rate limiter that preserves an absolute schedule without catch-up bursts."""

    def __init__(
        self,
        target_hz: float,
        *,
        label: str = "unnamed",
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not isinstance(target_hz, (int, float)) or not math.isfinite(float(target_hz)) or target_hz <= 0:
            raise ValueError(f"target_hz must be positive, got {target_hz}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("rate manager label must be a non-empty string")
        self.target_hz = float(target_hz)
        self.label = label.strip()
        self.period = 1.0 / target_hz
        self._clock = time.perf_counter if clock is None else clock
        self._sleep = time.sleep if sleep is None else sleep
        self._busy_wait = clock is None
        started = self._clock()
        self._started_at = started
        self._cycle_started_at = started
        self._last_completed_at = started
        self._next_deadline = started + self.period
        self._overdue_throttle: int = 0
        self._loop_count = 0
        self._last_work_duration_s = 0.0
        self._max_work_duration_s = 0.0
        self._deadline_overrun_count = 0
        self._missed_slot_count = 0
        self._long_block_reanchor_count = 0

    @property
    def stats(self) -> RateStats:
        """Return a read-only point-in-time copy of accumulated statistics."""
        elapsed = max(0.0, self._last_completed_at - self._started_at)
        return RateStats(
            target_period_s=self.period,
            loop_count=self._loop_count,
            last_work_duration_s=self._last_work_duration_s,
            max_work_duration_s=self._max_work_duration_s,
            deadline_overrun_count=self._deadline_overrun_count,
            missed_slot_count=self._missed_slot_count,
            long_block_reanchor_count=self._long_block_reanchor_count,
            elapsed_s=elapsed,
            actual_hz=(self._loop_count / elapsed if elapsed > 0.0 else 0.0),
        )

    def wait(self) -> None:
        """Sleep until the next absolute cycle deadline with precision.

        Deadlines advance by exactly one period per call (absolute schedule),
        so per-tick jitter does NOT accumulate as long-term drift — ticks stay
        locked to the recording time grid.

        Hybrid strategy:
          1. Compute remaining time to the deadline
          2. If > 2ms: time.sleep(remaining - 1ms)
          3. Busy-wait for the final precision window
        """
        now = self._clock()
        work_duration_s = max(0.0, now - self._cycle_started_at)
        self._last_work_duration_s = work_duration_s
        self._max_work_duration_s = max(self._max_work_duration_s, work_duration_s)
        remaining = self._next_deadline - now

        if remaining > 0:
            if remaining > 0.002:  # > 2ms: sleep for bulk
                self._sleep(remaining - 0.001)
            if self._busy_wait:
                while self._clock() < self._next_deadline:
                    pass  # spin
            else:
                final_remaining = self._next_deadline - self._clock()
                if final_remaining > 0.0:
                    self._sleep(final_remaining)
            self._next_deadline += self.period
        else:
            lateness = -remaining
            self._deadline_overrun_count += 1
            self._missed_slot_count += int((lateness + self.period * 1e-12) // self.period)

            if lateness > 1.0:
                self._long_block_reanchor_count += 1
                self._next_deadline = now + self.period
                self._overdue_throttle = 0
            else:
                # Short overrun: emit a throttled warning.
                if self._overdue_throttle <= 0:
                    logger.warning(
                        "Control loop over budget: loop=%s actual=%.1fms "
                        "target=%.1fms lateness=%.1fms missed_total=%d",
                        self.label,
                        (self.period - remaining) * 1000,
                        self.period * 1000,
                        lateness * 1000,
                        self._missed_slot_count,
                    )
                    self._overdue_throttle = 50
                else:
                    self._overdue_throttle -= 1

                if lateness >= self.period:
                    # Missed a full slot or more: re-anchor to now (no catch-up burst)
                    self._next_deadline = now + self.period
                else:
                    # Small overrun: keep the absolute grid, next tick absorbs it
                    self._next_deadline += self.period

        completed = self._clock()
        self._loop_count += 1
        self._last_completed_at = completed
        self._cycle_started_at = completed

    def reset(self) -> None:
        """Reset the deadline and overdue throttle to the current time.

        Call after a long blocking operation (e.g. return-to-home) so the
        next wait() does not see a stale deadline and log a spurious
        over-budget warning.
        """
        now = self._clock()
        self._next_deadline = now + self.period
        self._cycle_started_at = now
        self._overdue_throttle = 0
