"""Absolute-deadline loop-rate control with overrun accounting."""

from __future__ import annotations

import math
import time
from typing import Callable

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class LoopRate:
    """Rate limiter that preserves an absolute schedule without catch-up bursts.

    Actuator loops may need the final busy-wait precision window. Service loops
    can disable it to avoid competing for CPU time.
    """

    def __init__(
        self,
        target_hz: float,
        *,
        label: str = "unnamed",
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        busy_wait: bool | None = None,
        warn_on_overrun: bool = True,
    ) -> None:
        if (
            not isinstance(target_hz, (int, float))
            or not math.isfinite(float(target_hz))
            or target_hz <= 0
        ):
            raise ValueError(f"target_hz must be positive, got {target_hz}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("loop label must be a non-empty string")
        if busy_wait is not None and not isinstance(busy_wait, bool):
            raise ValueError("busy_wait must be a bool or None")
        if not isinstance(warn_on_overrun, bool):
            raise ValueError("warn_on_overrun must be a bool")
        self.label = label.strip()
        self.period = 1.0 / target_hz
        self._clock = time.perf_counter if clock is None else clock
        self._sleep = time.sleep if sleep is None else sleep
        # Injected clocks default to sleep-only; the real clock defaults to a final spin.
        self._busy_wait = clock is None if busy_wait is None else bool(busy_wait)
        self._warn_on_overrun = warn_on_overrun
        started = self._clock()
        self._next_deadline = started + self.period
        self._overdue_throttle: int = 0
        self._missed_slot_count = 0

    def wait(self) -> None:
        """Sleep until the next absolute cycle deadline with precision.

        Small overruns preserve the absolute schedule. Missing a full period
        re-anchors the next deadline to avoid catch-up bursts. This poll schedule
        does not define the recorder's or policy's logical time grid.

        Hybrid strategy:
          1. Compute remaining time to the deadline
          2. If > 2ms: time.sleep(remaining - 1ms)
          3. Spin or sleep for the final window according to busy_wait
        """
        now = self._clock()
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
            self._missed_slot_count += int((lateness + self.period * 1e-12) // self.period)

            if lateness > 1.0:
                self._next_deadline = now + self.period
                self._overdue_throttle = 0
            else:
                if self._warn_on_overrun and self._overdue_throttle <= 0:
                    logger.warning(
                        "Loop deadline missed: loop=%s period_ms=%.1f "
                        "deadline_lateness_ms=%.1f missed_total=%d",
                        self.label,
                        self.period * 1000,
                        lateness * 1000,
                        self._missed_slot_count,
                    )
                    self._overdue_throttle = 50
                elif self._warn_on_overrun:
                    self._overdue_throttle -= 1

                if lateness >= self.period:
                    # Missed a full slot or more: re-anchor to now (no catch-up burst)
                    self._next_deadline = now + self.period
                else:
                    # Small overrun: keep the absolute grid, next tick absorbs it
                    self._next_deadline += self.period

    def reset(self) -> None:
        """Reset the deadline and overdue throttle to the current time.

        Call after a long blocking operation (e.g. return-to-home) so the
        next wait() does not see a stale deadline and log a spurious
        over-budget warning.
        """
        now = self._clock()
        self._next_deadline = now + self.period
        self._overdue_throttle = 0
