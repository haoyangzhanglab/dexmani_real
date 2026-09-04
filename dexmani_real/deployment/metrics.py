"""Deployment observability counters and timing summaries.

A minimal, dependency-free counter/gauge registry with structured logging —
no Prometheus/OpenTelemetry. Each worker process owns one :class:`Metrics`
instance and flushes it periodically for live experiment diagnostics. The
counters are ordinary Python dicts:
each worker loop is single-threaded, so no locking is needed (a mutex would be
dead weight).

Counters are per-flush deltas (``flush`` resets them to zero), while a second
counter view accumulates one policy episode. Gauges hold the last observed
value (age/latency), which overwrites each tick. Timing samples are retained in
a fixed-size episode window and summarized with the nearest-rank p50/p95/p99
convention.
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
CHUNKS_CREATED = "chunks_created"
CHUNKS_STALE = "chunks_stale"
CHUNKS_GENERATION_DROPPED = "chunks_generation_dropped"
CHUNKS_INGESTED = "chunks_ingested"
ENDPOINTS_DUE = "endpoints_due"
ENDPOINTS_PUBLISHED = "endpoints_published"
ENDPOINTS_VALIDATED = "endpoints_validated"
COMMAND_PROGRESS_TIMEOUT = "command_progress_timeout"
PUBLICATION_INTERVAL_MS = "publication_interval_ms"
ENDPOINT_SCHEDULE_LATENESS_MS = "endpoint_schedule_lateness_ms"
ENDPOINTS_MOTION_DISCARDED = "endpoints_motion_discarded"
ENDPOINTS_STALE_DISCARDED = "endpoints_stale_discarded"
ENDPOINTS_TRANSIENT_DEFERRED = "endpoints_transient_deferred"
ENDPOINTS_FATAL_REJECTED = "endpoints_fatal_rejected"
SAFETY_REJECTIONS = "safety_rejections"
HAND_PREFLIGHT_REJECTIONS = "hand_preflight_rejections"
HAND_POLICY_ENDPOINT_ROUNDOFF_CANONICALIZED = (
    "hand_policy_endpoint_roundoff_canonicalized"
)
POLICY_ABORTS = "policy_aborts"
COMMAND_SILENCE_ABORT = "command_silence_abort"
EPISODE_ACTION_STEPS = "episode_action_steps"
SUCCESSFUL_ACTION_STEPS = "successful_action_steps"
SAFETY_REJECTED_STEPS = "safety_rejected_steps"

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
        self._episode_counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = {}
        self._episode_generation: int | None = None
        self._episode_started_monotonic_ns: int | None = None

    def begin_episode(self, *, generation: int, started_monotonic_ns: int) -> None:
        """Reset diagnostics at one B/epoch boundary.

        The interval counters are reset too: a new B epoch must not carry an
        unfinished previous-run logging interval into its first flush.
        """
        if generation <= 0:
            raise ValueError("episode generation must be positive")
        if started_monotonic_ns <= 0:
            raise ValueError("episode start timestamp must be positive")
        self.begin_run()
        self._episode_generation = int(generation)
        self._episode_started_monotonic_ns = int(started_monotonic_ns)

    def begin_run(self) -> None:
        """Reset diagnostics for an epoch without opening an episode summary."""
        self._counters.clear()
        self._episode_counters.clear()
        self._gauges.clear()
        self._samples.clear()
        self._episode_generation = None
        self._episode_started_monotonic_ns = None

    def increment(self, name: str, n: int = 1) -> None:
        """Add *n* to counter *name* (counters are per-flush deltas)."""
        increment = int(n)
        self._counters[name] = self._counters.get(name, 0) + increment
        self._episode_counters[name] = self._episode_counters.get(name, 0) + increment

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

    def episode_snapshot(self) -> dict[str, int | float]:
        """Return counters and bounded timing summaries since the current B."""
        result: dict[str, int | float] = {}
        result.update(self._episode_counters)
        result.update(self._timing_snapshot())
        return result

    def log_episode_summary(self, *, status: str, reason: str) -> None:
        """Log one compact, in-memory diagnostic summary for the active episode.

        This intentionally writes no artifact. Calling it after an episode has
        already been summarized is a no-op, so competing shutdown paths cannot
        emit duplicate summaries.
        """
        generation = self._episode_generation
        started_ns = self._episode_started_monotonic_ns
        if generation is None or started_ns is None:
            return
        duration_s = max(0.0, (time.monotonic_ns() - started_ns) / 1e9)
        values = self.episode_snapshot()
        rendered = " ".join(f"{key}={value}" for key, value in sorted(values.items()))
        logger.info(
            "episode summary: generation=%d status=%s reason=%r duration_s=%.3f%s",
            generation,
            status,
            reason,
            duration_s,
            f" {rendered}" if rendered else "",
        )
        self._episode_generation = None
        self._episode_started_monotonic_ns = None

    def flush(self, *, prefix: str = "metrics", debug: bool = False) -> None:
        """Log one structured ``key=value`` line and reset the per-flush counters.

        Gauges (age/latency) are not reset — they are instantaneous and simply
        keep their last observed value until the next ``observe``. High-rate
        producers may emit at DEBUG while retaining identical counter resets.
        """
        values = self.snapshot()
        if not values:
            return
        rendered = " ".join(f"{key}={value}" for key, value in sorted(values.items()))
        log = logger.debug if debug else logger.info
        log("%s: %s", prefix, rendered)
        for name in list(self._counters):
            self._counters[name] = 0


def flush_every(
    metrics: Metrics,
    *,
    last_ns: int,
    interval_s: float = 1.0,
    prefix: str = "metrics",
    debug: bool = False,
) -> int:
    """Flush *metrics* if ``interval_s`` has elapsed since *last_ns* (monotonic ns).

    Returns the monotonic timestamp to carry forward; a pure helper so the
    worker loops keep their throttle logic identical.
    """
    now_ns = time.monotonic_ns()
    if now_ns - last_ns >= int(interval_s * 1e9):
        metrics.flush(prefix=prefix, debug=debug)
        return now_ns
    return last_ns
