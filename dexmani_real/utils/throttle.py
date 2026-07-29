"""ThrottledWarner — rate-limited warning logger (≤1 / interval_s pattern).

Used by arm_process, hand_process, and SeqlockRingBuffer to avoid log spam
from per-tick hot-path conditions (torn reads, stale state, producer mismatch).
"""

from __future__ import annotations

import time
from typing import Any

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class ThrottledWarner:
    """Callable that forwards to ``logger.warning`` at most once per *interval_s*.

    Typical interval: 5.0 s (same cadence as SeqlockRingBuffer torn-read warns).
    """

    def __init__(self, interval_s: float = 5.0) -> None:
        self._interval_ns = int(interval_s * 1e9)
        self._last_ns = 0

    def __call__(self, msg: str, *args: Any) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_ns < self._interval_ns:
            return
        self._last_ns = now_ns
        logger.warning(msg, *args)
