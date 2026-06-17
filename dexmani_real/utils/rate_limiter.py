"""Rate limiter for control loops.

Ensures consistent loop frequency by compensating for computation time,
rather than blindly sleeping for a fixed interval.
"""

from __future__ import annotations

import time


class RateLimiter:
    """Rate limiter that compensates for computation time.

    Usage:
        limiter = RateLimiter(50.0)  # 50 Hz
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

    def wait(self) -> None:
        """Sleep until the next control cycle boundary."""
        elapsed = time.perf_counter() - self.last_wake
        sleep_time = self.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_wake = time.perf_counter()

    def reset(self) -> None:
        """Reset the timer to the current time."""
        self.last_wake = time.perf_counter()

    @property
    def target_hz(self) -> float:
        return 1.0 / self.dt
