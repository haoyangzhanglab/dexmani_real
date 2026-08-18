"""Lightweight consecutive-event counter with threshold escalation.
"""

from __future__ import annotations

import math
from collections import deque


class RetryCounter:
    """Count consecutive events and signal when a threshold is reached.

    Usage::

        counter = RetryCounter(max_consecutive=5, label="servo")
        if fail:
            counter.inc()
            if counter.triggered:
                escalate()
        else:
            counter.reset()

    ``triggered`` is True when ``count >= max_consecutive`` (the counter
    keeps incrementing past the threshold — callers decide whether to stop).
    """

    __slots__ = ("max_consecutive", "label", "_count")

    def __init__(self, max_consecutive: int, label: str = "") -> None:
        self.max_consecutive = max_consecutive
        self.label = label
        self._count = 0

    def inc(self) -> int:
        """Increment and return the new count."""
        self._count += 1
        return self._count

    def reset(self) -> None:
        """Reset the counter to zero."""
        self._count = 0

    @property
    def count(self) -> int:
        """Current consecutive-event count."""
        return self._count

    @property
    def triggered(self) -> bool:
        """True when ``count >= max_consecutive``."""
        return self._count >= self.max_consecutive


class EventWindowCounter:
    """Count non-consecutive events inside a monotonic sliding window."""

    def __init__(self, max_events: int, window_s: float) -> None:
        if not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("event window threshold must be a positive integer")
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("event window duration must be finite and positive")
        self.max_events = int(max_events)
        self.window_s = float(window_s)
        self._timestamps: deque[float] = deque()
        self._last_timestamp = float("-inf")

    def record(self, timestamp_s: float) -> bool:
        """Record one event and return whether the window threshold is met."""
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("event window timestamps must be finite")
        if timestamp < self._last_timestamp:
            raise ValueError("event window timestamps must be monotonic")
        self._last_timestamp = timestamp
        cutoff = timestamp - self.window_s
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        self._timestamps.append(timestamp)
        return self.triggered

    @property
    def count(self) -> int:
        return len(self._timestamps)

    @property
    def triggered(self) -> bool:
        return len(self._timestamps) >= self.max_events
