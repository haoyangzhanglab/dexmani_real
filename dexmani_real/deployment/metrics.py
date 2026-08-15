"""Deployment observability counters (execution doc §94).

A minimal, dependency-free counter/gauge registry with structured logging —
no Prometheus/OpenTelemetry (§94). Each worker process owns one :class:`Metrics`
instance and flushes it periodically so the H0–H6 hardware gates have live
numbers without any external collector. The counters are ordinary Python dicts:
each worker loop is single-threaded, so no locking is needed (a mutex would be
dead weight).

Counters are per-flush deltas (``flush`` resets them to zero); gauges hold the
last observed value (age/latency), which overwrites each tick.
"""

from __future__ import annotations

import time
from typing import Any

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# §94 counter/guard names — the single source of truth used by both the
# inference worker and the coordinator so they never diverge.
OBSERVATIONS_BUILT = "observations_built"
OBSERVATION_AGE_MS = "observation_age_ms"
OBSERVATION_SKEW_MS = "observation_skew_ms"
INFERENCE_MS = "inference_ms"
INFERENCE_FAILURES = "inference_failures"
PLANS_CREATED = "plans_created"
PLANS_SUPERSEDED = "plans_superseded"
PLANS_STALE = "plans_stale"
PLANS_GENERATION_DROPPED = "plans_generation_dropped"
PLAN_AGE_MS = "plan_age_ms"
ENDPOINTS_DUE = "endpoints_due"
ENDPOINTS_COALESCED = "endpoints_coalesced"
ENDPOINTS_PUBLISHED = "endpoints_published"
SAFETY_REJECTIONS = "safety_rejections"
POLICY_ABORTS = "policy_aborts"
COMMAND_SILENCE_ABORT = "command_silence_abort"

# Names whose value is a per-flush delta (increment-only); gauges are everything
# else and keep their last observed value across flushes.
_COUNTER_NAMES = frozenset(
    {
        OBSERVATIONS_BUILT,
        INFERENCE_FAILURES,
        PLANS_CREATED,
        PLANS_SUPERSEDED,
        PLANS_STALE,
        PLANS_GENERATION_DROPPED,
        ENDPOINTS_DUE,
        ENDPOINTS_COALESCED,
        ENDPOINTS_PUBLISHED,
        SAFETY_REJECTIONS,
        POLICY_ABORTS,
        COMMAND_SILENCE_ABORT,
    }
)


class Metrics:
    """Per-worker counter/gauge registry with structured flush logging."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, n: int = 1) -> None:
        """Add *n* to counter *name* (counters are per-flush deltas)."""
        self._counters[name] = self._counters.get(name, 0) + int(n)

    def observe(self, name: str, value: float) -> None:
        """Record the latest value of gauge *name* (overwrites)."""
        self._gauges[name] = float(value)

    def snapshot(self) -> dict[str, int | float]:
        """Return the current counters and gauges merged into one mapping."""
        result: dict[str, int | float] = {}
        result.update(self._counters)
        result.update(self._gauges)
        return result

    def flush(self, *, prefix: str = "metrics") -> None:
        """Log one structured ``key=value`` line and reset the per-flush counters.

        Gauges (age/latency) are not reset — they are instantaneous and simply
        keep their last observed value until the next ``observe``.
        """
        values = self.snapshot()
        if not values:
            return
        rendered = " ".join(f"{key}={value}" for key, value in sorted(values.items()))
        logger.info("%s: %s", prefix, rendered)
        for name in list(self._counters):
            self._counters[name] = 0


def flush_every(metrics: Metrics, *, last_ns: int, interval_s: float = 1.0, prefix: str = "metrics") -> int:
    """Flush *metrics* if ``interval_s`` has elapsed since *last_ns* (monotonic ns).

    Returns the monotonic timestamp to carry forward; a pure helper so the
    worker loops keep their throttle logic identical.
    """
    now_ns = time.monotonic_ns()
    if now_ns - last_ns >= int(interval_s * 1e9):
        metrics.flush(prefix=prefix)
        return now_ns
    return last_ns
