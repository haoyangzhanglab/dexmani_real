"""High-precision rate manager with hybrid busy-wait.

RateManager: high-precision sleep using busy-wait hybrid,
achieving < 1ms target error (vs ~15ms for time.sleep()).

Ref: LeFranX Ruckig rate decoupling.
     BunnyVisionPro wait_until_next_control_signal.
"""

from __future__ import annotations

import time

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
        rm = RateManager(50.0)  # e.g. 50 Hz
        while running:
            do_work()
            rm.wait()
    """

    def __init__(self, target_hz: float) -> None:
        if target_hz <= 0:
            raise ValueError(f"target_hz must be positive, got {target_hz}")
        self.target_hz = float(target_hz)
        self.period = 1.0 / target_hz
        self._next_deadline = time.perf_counter() + self.period
        self._overdue_throttle: int = 0

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
        now = time.perf_counter()
        remaining = self._next_deadline - now

        if remaining > 0:
            if remaining > 0.002:  # > 2ms: sleep for bulk
                time.sleep(remaining - 0.001)
            while time.perf_counter() < self._next_deadline:
                pass  # spin
            self._next_deadline += self.period
            return

        # Overdue.
        #
        # Long block (>1 s): deliberate blocking operation (e.g. return-to-home,
        # episode save).  Re-anchor silently — this is not a performance problem.
        if -remaining > 1.0:
            self._next_deadline = now + self.period
            self._overdue_throttle = 0
            return

        # Short overrun: emit throttled warning (every ~50 occurrences)
        if self._overdue_throttle <= 0:
            logger.warning(
                "Control loop over budget: actual=%.1fms target=%.1fms",
                (self.period - remaining) * 1000,
                self.period * 1000,
            )
            self._overdue_throttle = 50
        else:
            self._overdue_throttle -= 1

        if -remaining >= self.period:
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
        self._next_deadline = time.perf_counter() + self.period
        self._overdue_throttle = 0
