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
        self.next_deadline = time.perf_counter() + self.period
        self.overdue_throttle: int = 0
        self.missed_slot_count = 0

    def wait(self) -> None:
        """Re-anchor after a missed period; this polling grid does not define data time."""
        now = time.perf_counter()
        remaining = self.next_deadline - now

        if remaining > 0:
            if remaining > 0.002:
                time.sleep(remaining - 0.001)
            final_remaining = self.next_deadline - time.perf_counter()
            if final_remaining > 0.0:
                time.sleep(final_remaining)
            self.next_deadline += self.period
        else:
            lateness = -remaining
            self.missed_slot_count += int((lateness + self.period * 1e-12) // self.period)

            if lateness > 1.0:
                self.next_deadline = now + self.period
                self.overdue_throttle = 0
            else:
                if self.overdue_throttle <= 0:
                    logger.warning(
                        "Loop deadline missed: loop=%s period_ms=%.1f "
                        "deadline_lateness_ms=%.1f missed_total=%d",
                        self.label,
                        self.period * 1000,
                        lateness * 1000,
                        self.missed_slot_count,
                    )
                    self.overdue_throttle = 50
                else:
                    self.overdue_throttle -= 1

                if lateness >= self.period:
                    # Missed a full slot or more: re-anchor to now (no catch-up burst)
                    self.next_deadline = now + self.period
                else:
                    # Small overrun: keep the absolute grid, next tick absorbs it
                    self.next_deadline += self.period

    def reset(self) -> None:
        """Re-anchor deadline/warning state after blocking work such as HOME."""
        now = time.perf_counter()
        self.next_deadline = now + self.period
        self.overdue_throttle = 0


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


def sample_is_fresh(timestamp_ns, max_age_s, now_ns=None):
    now = time.monotonic_ns() if now_ns is None else now_ns
    return 0 < int(timestamp_ns) <= now and now - int(timestamp_ns) <= int(max_age_s * 1e9)
