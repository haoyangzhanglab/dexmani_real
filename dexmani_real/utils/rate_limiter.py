"""Rate limiter for control loops.

Ensures consistent loop frequency by compensating for computation time,
rather than blindly sleeping for a fixed interval.

When the actual cycle time exceeds the target period, a throttled warning
is emitted (ref: BunnyVisionPro wait_until_next_control_signal).
"""

from __future__ import annotations

import time

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Rate limiter that compensates for computation time.

    Usage:
        limiter = RateLimiter(50.0)  # e.g. 50 Hz
        while True:
            do_work()
            limiter.wait()  # sleeps only the remaining time in this cycle

    CLAUDE.md Section 4.4 reference: control loops must use rate limiter, not plain time.sleep().
    """

    def __init__(self, target_hz: float) -> None:
        if target_hz <= 0:
            raise ValueError(f"target_hz must be positive, got {target_hz}")
        self.dt = 1.0 / target_hz
        self.last_wake = time.perf_counter()
        self._overdue_cycles: int = 0
        self._total_cycles: int = 0
        self._overdue_throttle: int = 0

    def wait(self) -> None:
        """Sleep until the next control cycle boundary.

        Emits a throttled warning when the cycle is over budget
        (ref: BunnyVisionPro xarm7_ability.py:233-244).
        """
        elapsed = time.perf_counter() - self.last_wake
        sleep_time = self.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif self._overdue_throttle <= 0:
            self._overdue_cycles += 1
            logger.warning(
                "Control loop over budget: actual=%.1fms target=%.1fms " "(overdue %s/%s cycles)",
                elapsed * 1000,
                self.dt * 1000,
                self._overdue_cycles,
                self._total_cycles + 1,
            )
            self._overdue_throttle = 50  # throttle: ~1 warning/s at the target rate
        else:
            self._overdue_cycles += 1
            self._overdue_throttle -= 1
        self._total_cycles += 1
        self.last_wake = time.perf_counter()

    def reset(self) -> None:
        """Reset the timer and overdue throttle to the current time."""
        self.last_wake = time.perf_counter()
        self._overdue_throttle = 0

    @property
    def target_hz(self) -> float:
        return 1.0 / self.dt

    @property
    def overdue_ratio(self) -> float:
        """Fraction of cycles that exceeded the target period."""
        if self._total_cycles == 0:
            return 0.0
        return self._overdue_cycles / self._total_cycles
