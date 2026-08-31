"""Deployment observability counters and bounded timing summaries.

A minimal, dependency-free counter/gauge registry with structured logging —
no Prometheus/OpenTelemetry. Each worker process owns one :class:`Metrics`
instance and flushes it periodically so the H0–H6 hardware gates have live
numbers without any external collector. The counters are ordinary Python dicts:
each worker loop is single-threaded, so no locking is needed (a mutex would be
dead weight).

Counters are per-flush deltas (``flush`` resets them to zero); gauges hold the
last observed value (age/latency), which overwrites each tick.  Timing samples
are intentionally bounded, retained across flushes, and summarized with the
nearest-rank p50/p95/p99 convention.  They support an operator-readable shadow
receipt without introducing a metrics service or unbounded memory growth.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Mapping

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
ENDPOINTS_SHADOW_VALIDATED = "endpoints_shadow_validated"
ENDPOINTS_COMMITTED = "endpoints_committed"
COUPLED_COMMAND_WRITES = "coupled_command_writes"
SHADOW_COUPLED_WRITE_VIOLATIONS = "shadow_coupled_write_violations"
EXECUTE_ACKNOWLEDGED = "execute_acknowledged"
EXECUTE_ACK_TIMEOUT = "execute_ack_timeout"
EXECUTE_PUBLICATION_BOUND_REACHED = "execute_publication_bound_reached"
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
PHYSICAL_HOME_COMPLETED = "physical_home_completed"

_TIMING_SAMPLE_CAPACITY = 256
_QUANTILE_POINTS = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))

# else and keep their last observed value across flushes.
_COUNTER_NAMES = frozenset(
    {
        OBSERVATIONS_BUILT,
        OBSERVATION_WAIT_POINTCLOUD_GRID,
        OBSERVATION_WAIT_POINTCLOUD_STALE,
        OBSERVATION_WAIT_ARM_HISTORY,
        OBSERVATION_WAIT_HAND_HISTORY,
        OBSERVATION_WAIT_POINTCLOUD_HISTORY,
        OBSERVATION_WAIT_RGB_GRID,
        OBSERVATION_WAIT_RGB_HISTORY,
        OBSERVATION_WAIT_GRID_ADVANCE,
        INFERENCE_FAILURES,
        PLANS_CREATED,
        PLANS_SUPERSEDED,
        PLANS_STALE,
        PLANS_GENERATION_DROPPED,
        PLANS_INGESTED,
        PLANS_EVICTED,
        ENDPOINTS_DUE,
        ENDPOINTS_COALESCED,
        ENDPOINTS_PUBLISHED,
        ENDPOINTS_SHADOW_VALIDATED,
        ENDPOINTS_COMMITTED,
        COUPLED_COMMAND_WRITES,
        SHADOW_COUPLED_WRITE_VIOLATIONS,
        EXECUTE_ACKNOWLEDGED,
        EXECUTE_ACK_TIMEOUT,
        EXECUTE_PUBLICATION_BOUND_REACHED,
        ENDPOINTS_MOTION_DISCARDED,
        ENDPOINTS_STALE_DISCARDED,
        ENDPOINTS_TRANSIENT_DEFERRED,
        ENDPOINTS_FATAL_REJECTED,
        IK_CHECKER_REJECTS,
        SAFETY_REJECTIONS,
        HAND_PREFLIGHT_REJECTIONS,
        HAND_POLICY_ENDPOINT_ROUNDOFF_CANONICALIZED,
        POLICY_ABORTS,
        COMMAND_SILENCE_ABORT,
        PHYSICAL_HOME_COMPLETED,
        # Per-gate reject counters are derived from ``reject_counter_name``.
    }
)


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
        self._run_counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = {}

    def begin_run(self) -> None:
        """Reset the bounded per-run receipt values at one B/epoch boundary.

        The interval counters are reset too: a new B epoch must not carry an
        unfinished previous-run logging interval into its first flush.
        """
        self._counters.clear()
        self._run_counters.clear()
        self._gauges.clear()
        self._samples.clear()

    def increment(self, name: str, n: int = 1) -> None:
        """Add *n* to counter *name* (counters are per-flush deltas)."""
        increment = int(n)
        self._counters[name] = self._counters.get(name, 0) + increment
        self._run_counters[name] = self._run_counters.get(name, 0) + increment

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

    def run_snapshot(self) -> dict[str, int | float]:
        """Return non-resetting counters and bounded timing stats for one run."""
        result: dict[str, int | float] = {}
        result.update(self._run_counters)
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


def shadow_run_receipt_json(
    *,
    run_generation: int,
    reason: str,
    coupled_command_start_sequence: int,
    coupled_command_end_sequence: int,
    metrics: Mapping[str, int | float],
) -> str:
    """Render one canonical, auditable shadow-run receipt.

    The coupled-ring sequence is sampled when B enters RUNNING and again when
    the policy run leaves it.  A non-zero delta is an invariant violation, not
    a substitute for physical validation.  This function is pure so its shape
    and the zero-write arithmetic remain offline-testable.
    """
    if isinstance(run_generation, bool) or int(run_generation) <= 0:
        raise ValueError("run_generation must be a positive integer")
    if (
        isinstance(coupled_command_start_sequence, bool)
        or isinstance(coupled_command_end_sequence, bool)
        or int(coupled_command_start_sequence) < 0
        or int(coupled_command_end_sequence) < int(coupled_command_start_sequence)
    ):
        raise ValueError("coupled command sequences must be monotonic integers")
    if not isinstance(reason, str) or not reason:
        raise ValueError("reason must be a non-empty string")
    normalized_metrics: dict[str, int | float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("receipt metric values must be finite numbers")
        normalized_metrics[name] = (
            int(value) if isinstance(value, int) else float(value)
        )
    writes = int(coupled_command_end_sequence) - int(coupled_command_start_sequence)
    receipt = {
        "coupled_command_end_sequence": int(coupled_command_end_sequence),
        "coupled_command_start_sequence": int(coupled_command_start_sequence),
        "coupled_command_writes": writes,
        "execution_mode": "shadow",
        "metrics": normalized_metrics,
        "reason": reason,
        "run_generation": int(run_generation),
        "zero_coupled_command_writes": writes == 0,
    }
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)


def execute_run_receipt_json(
    *,
    run_generation: int,
    reason: str,
    coupled_command_start_sequence: int,
    coupled_command_end_sequence: int,
    max_published_endpoints: int,
    acknowledgement_timeout_s: float,
    acknowledged_action_id: int | None,
    completed: bool,
    provenance_json: str | None = None,
    metrics: Mapping[str, int | float],
) -> str:
    """Render the bounded H4 execute receipt without hiding physical writes."""
    if isinstance(max_published_endpoints, bool) or max_published_endpoints != 1:
        raise ValueError("H4 execute receipt requires exactly one publication bound")
    return bounded_execute_run_receipt_json(
        execution_mode="execute",
        run_generation=run_generation,
        reason=reason,
        coupled_command_start_sequence=coupled_command_start_sequence,
        coupled_command_end_sequence=coupled_command_end_sequence,
        max_published_endpoints=max_published_endpoints,
        acknowledgement_timeout_s=acknowledgement_timeout_s,
        acknowledged_action_id=acknowledged_action_id,
        completed=completed,
        provenance_json=provenance_json,
        metrics=metrics,
    )


def bounded_execute_run_receipt_json(
    *,
    execution_mode: str,
    run_generation: int,
    reason: str,
    coupled_command_start_sequence: int,
    coupled_command_end_sequence: int,
    max_published_endpoints: int,
    acknowledgement_timeout_s: float,
    acknowledged_action_id: int | None,
    completed: bool,
    provenance_json: str | None = None,
    metrics: Mapping[str, int | float],
) -> str:
    """Render one bounded physical-execution receipt."""
    if execution_mode not in {"execute", "task"}:
        raise ValueError("execution_mode must be 'execute' or 'task'")
    if (
        isinstance(max_published_endpoints, bool)
        or not isinstance(max_published_endpoints, int)
        or max_published_endpoints <= 0
    ):
        raise ValueError("max_published_endpoints must be a positive integer")
    if (
        isinstance(acknowledgement_timeout_s, bool)
        or not math.isfinite(float(acknowledgement_timeout_s))
        or acknowledgement_timeout_s <= 0.0
    ):
        raise ValueError("acknowledgement_timeout_s must be finite and positive")
    if acknowledged_action_id is not None and (
        isinstance(acknowledged_action_id, bool) or acknowledged_action_id <= 0
    ):
        raise ValueError("acknowledged_action_id must be positive or None")
    if not isinstance(completed, bool):
        raise TypeError("completed must be a boolean")
    provenance = None
    if provenance_json is not None:
        if not isinstance(provenance_json, str):
            raise TypeError("provenance_json must be a string or None")
        provenance = json.loads(provenance_json)
        if not isinstance(provenance, dict):
            raise ValueError("provenance_json must encode a JSON object")
    if isinstance(run_generation, bool) or int(run_generation) <= 0:
        raise ValueError("run_generation must be a positive integer")
    if (
        isinstance(coupled_command_start_sequence, bool)
        or isinstance(coupled_command_end_sequence, bool)
        or int(coupled_command_start_sequence) < 0
        or int(coupled_command_end_sequence) < int(coupled_command_start_sequence)
    ):
        raise ValueError("coupled command sequences must be monotonic integers")
    if not isinstance(reason, str) or not reason:
        raise ValueError("reason must be a non-empty string")
    normalized_metrics: dict[str, int | float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("receipt metric values must be finite numbers")
        normalized_metrics[name] = (
            int(value) if isinstance(value, int) else float(value)
        )
    writes = int(coupled_command_end_sequence) - int(coupled_command_start_sequence)
    if completed and writes != max_published_endpoints:
        raise ValueError(
            "completed execute receipt requires exactly the bounded publication count"
        )
    receipt = {
        "acknowledged_action_id": acknowledged_action_id,
        "acknowledgement_timeout_s": float(acknowledgement_timeout_s),
        "completed": completed,
        "coupled_command_end_sequence": int(coupled_command_end_sequence),
        "coupled_command_start_sequence": int(coupled_command_start_sequence),
        "coupled_command_writes": writes,
        "execution_mode": execution_mode,
        "max_published_endpoints": int(max_published_endpoints),
        "metrics": normalized_metrics,
        "outcome": "completed" if completed else "not_completed",
        "provenance": provenance,
        "reason": reason,
        "run_generation": int(run_generation),
        "within_publication_bound": writes <= int(max_published_endpoints),
    }
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
