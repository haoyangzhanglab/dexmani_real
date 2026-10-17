"""Absolute-deadline loop-rate control with overrun accounting."""

from __future__ import annotations

import logging
import math
import time


logger = logging.getLogger(__name__)


class LoopRate:
    """Sleep to absolute deadlines without catch-up bursts."""

    def __init__(
        self,
        target_hz: float,
        *,
        label: str,
    ) -> None:
        if not math.isfinite(target_hz) or target_hz <= 0:
            raise ValueError(f"target_hz must be positive, got {target_hz}")
        self.label = label
        self.period = 1.0 / target_hz
        self._next_deadline = time.perf_counter() + self.period
        self._overdue_throttle: int = 0
        self._missed_slot_count = 0

    def wait(self) -> None:
        """Keep the grid on small overruns, re-anchor after a missed period; this polling clock does not define data time."""
        now = time.perf_counter()
        remaining = self._next_deadline - now

        if remaining > 0:
            if remaining > 0.002:
                time.sleep(remaining - 0.001)
            final_remaining = self._next_deadline - time.perf_counter()
            if final_remaining > 0.0:
                time.sleep(final_remaining)
            self._next_deadline += self.period
        else:
            lateness = -remaining
            self._missed_slot_count += int((lateness + self.period * 1e-12) // self.period)

            if lateness > 1.0:
                self._next_deadline = now + self.period
                self._overdue_throttle = 0
            else:
                if self._overdue_throttle <= 0:
                    logger.warning(
                        "Loop deadline missed: loop=%s period_ms=%.1f "
                        "deadline_lateness_ms=%.1f missed_total=%d",
                        self.label,
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

    def reset(self) -> None:
        """Re-anchor deadline/warning state after blocking work such as HOME."""
        now = time.perf_counter()
        self._next_deadline = now + self.period
        self._overdue_throttle = 0
