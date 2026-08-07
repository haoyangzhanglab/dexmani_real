"""Lightweight consecutive-event counter with threshold escalation.

Shared by arm_loop and hand_loop — replaces scattered ``_consecutive_*``
integer counters + ``_*_RETRY_MAX`` constants with a single consistent
pattern.
"""

from __future__ import annotations


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
