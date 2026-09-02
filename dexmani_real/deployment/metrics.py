"""Deployment observability counters and timing summaries.

A minimal, dependency-free counter/gauge registry with structured logging —
no Prometheus/OpenTelemetry. Each worker process owns one :class:`Metrics`
instance and flushes it periodically for live experiment diagnostics. The
counters are ordinary Python dicts:
each worker loop is single-threaded, so no locking is needed (a mutex would be
dead weight).

Counters are per-flush deltas (``flush`` resets them to zero); gauges hold the
last observed value (age/latency), which overwrites each tick.  Timing samples
are retained in a fixed-size window and summarized with the nearest-rank
p50/p95/p99 convention.
"""

from __future__ import annotations

import math
import time
from collections import deque

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Counter and guard names shared by the inference worker and coordinator.
OBSERVATIONS_BUILT = "observations_built"
OBSERVATION_WAIT_POINTCLOUD_GRID = "observation_wait_pointcloud_grid"
OBSERVATION_WAIT_POINTCLOUD_STALE = "observation_wait_pointcloud_stale"
OBSERVATION_WAIT_ARM_HISTORY = "observation_wait_arm_history"
OBSERVATION_WAIT_HAND_HISTORY = "observation_wait_hand_history"
OBSERVATION_WAIT_POINTCLOUD_HISTORY = "observation_wait_pointcloud_history"
OBSERVATION_WAIT_RGB_GRID = "observation_wait_rgb_grid"
OBSERVATION_WAIT_RGB_HISTORY = "observation_wait_rgb_history"
OBSERVATION_WAIT_GRID_ADVANCE = "observation_wait_grid_advance"
OBSERVATION_AGE_MS = "observation_age_ms"
OBSERVATION_SKEW_MS = "observation_skew_ms"
INFERENCE_MS = "inference_ms"
INFERENCE_FAILURES = "inference_failures"
PLANS_CREATED = "plans_created"
PLANS_SUPERSEDED = "plans_superseded"
PLANS_STALE = "plans_stale"
PLANS_GENERATION_DROPPED = "plans_generation_dropped"
PLANS_INGESTED = "plans_ingested"
PLANS_EVICTED = "plans_evicted"
PLAN_AGE_MS = "plan_age_ms"
USABLE_HORIZON_MS = "usable_horizon_ms"
ENDPOINTS_DUE = "endpoints_due"
ENDPOINTS_COALESCED = "endpoints_coalesced"
ENDPOINTS_PUBLISHED = "endpoints_published"
ENDPOINTS_VALIDATED = "endpoints_validated"
ENDPOINTS_COMMITTED = "endpoints_committed"
ACKNOWLEDGED = "acknowledged"
ACK_TIMEOUT = "ack_timeout"
ACK_FAILURE = "ack_failure"
ACK_LATENCY_MS = "ack_latency_ms"
ENDPOINTS_MOTION_DISCARDED = "endpoints_motion_discarded"
ENDPOINTS_STALE_DISCARDED = "endpoints_stale_discarded"
ENDPOINTS_TRANSIENT_DEFERRED = "endpoints_transient_deferred"
ENDPOINTS_FATAL_REJECTED = "endpoints_fatal_rejected"
IK_CHECKER_REJECTS = "ik_checker_rejects"
SAFETY_REJECTIONS = "safety_rejections"
HAND_PREFLIGHT_REJECTIONS = "hand_preflight_rejections"
HAND_POLICY_ENDPOINT_ROUNDOFF_CANONICALIZED = (
    "hand_policy_endpoint_roundoff_canonicalized"
)
POLICY_ABORTS = "policy_aborts"
COMMAND_SILENCE_ABORT = "command_silence_abort"

_TIMING_SAMPLE_CAPACITY = 256
_QUANTILE_POINTS = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))


def reject_counter_name(gate_code: str | None) -> str:
    """Per-operation counter name for a safety-gate rejection.

    ``gate_code`` is the machine-readable :class:`GateRejectCode` value string
    (``None`` for the aggregate). The space-separated reason is normalized to a
    snake_case counter name so each distinct rejection reason is attributed
    separately in the flush log rather than folding into one aggregate total.
    """
    if gate_code is None:
        return SAFETY_REJECTIONS
    return "safety_reject_" + gate_code.replace(" ", "_")


class Metrics:
    """Per-worker counter/gauge registry with structured flush logging."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = {}

    def begin_run(self) -> None:
        """Reset diagnostics at one B/epoch boundary.

        The interval counters are reset too: a new B epoch must not carry an
        unfinished previous-run logging interval into its first flush.
        """
        self._counters.clear()
        self._gauges.clear()
        self._samples.clear()

    def increment(self, name: str, n: int = 1) -> None:
        """Add *n* to counter *name* (counters are per-flush deltas)."""
        increment = int(n)
        self._counters[name] = self._counters.get(name, 0) + increment

    def observe(self, name: str, value: float) -> None:
        """Record the latest value of gauge *name* (overwrites)."""
        sample = float(value)
        if not math.isfinite(sample):
            raise ValueError("metric gauge value must be finite")
        self._gauges[name] = sample

    def observe_timing(self, name: str, value: float) -> None:
        """Record one finite timing sample in a fixed-size rolling window."""
        sample = float(value)
        if not math.isfinite(sample):
            raise ValueError("timing sample must be finite")
        samples = self._samples.get(name)
        if samples is None:
            samples = deque(maxlen=_TIMING_SAMPLE_CAPACITY)
            self._samples[name] = samples
        samples.append(sample)

    @staticmethod
    def _timing_summary(name: str, samples: deque[float]) -> dict[str, int | float]:
        """Return deterministic nearest-rank p50/p95/p99 for one bounded window."""
        ordered = sorted(samples)
        count = len(ordered)
        if count == 0:
            return {}
        result: dict[str, int | float] = {f"{name}_samples": count}
        for label, quantile in _QUANTILE_POINTS:
            index = max(0, math.ceil(quantile * count) - 1)
            result[f"{name}_{label}"] = ordered[index]
        return result

    def _timing_snapshot(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for name in sorted(self._samples):
            result.update(self._timing_summary(name, self._samples[name]))
        return result

    def snapshot(self) -> dict[str, int | float]:
        """Return the current counters and gauges merged into one mapping."""
        result: dict[str, int | float] = {}
        result.update(self._counters)
        result.update(self._gauges)
        result.update(self._timing_snapshot())
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


def flush_every(
    metrics: Metrics, *, last_ns: int, interval_s: float = 1.0, prefix: str = "metrics"
) -> int:
    """Flush *metrics* if ``interval_s`` has elapsed since *last_ns* (monotonic ns).

    Returns the monotonic timestamp to carry forward; a pure helper so the
    worker loops keep their throttle logic identical.
    """
    now_ns = time.monotonic_ns()
    if now_ns - last_ns >= int(interval_s * 1e9):
        metrics.flush(prefix=prefix)
        return now_ns
    return last_ns
