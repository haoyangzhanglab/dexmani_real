"""Small, direct deployment diagnostics for control-quality investigation."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_SAMPLE_CAPACITY = 256


def _samples() -> deque[float]:
    return deque(maxlen=_SAMPLE_CAPACITY)


@dataclass
class PolicyStats:
    """Bounded timings and the few failure counts useful to an operator.

    Each deployment worker owns one instance. It deliberately has explicit
    fields instead of a named metric registry: these values are the supported
    diagnostics for learned-policy execution, not a generic observability API.
    """

    inference_latency_ms: deque[float] = field(default_factory=_samples)
    observation_age_ms: deque[float] = field(default_factory=_samples)
    observation_skew_ms: deque[float] = field(default_factory=_samples)
    schedule_lateness_ms: deque[float] = field(default_factory=_samples)
    publication_interval_ms: deque[float] = field(default_factory=_samples)
    safety_rejection_count: int = 0
    command_progress_timeout_count: int = 0
    ik_rejection_count: int = 0
    stale_prediction_count: int = 0

    @staticmethod
    def _append(samples: deque[float], value: float) -> None:
        sample = float(value)
        if not math.isfinite(sample):
            raise ValueError("deployment diagnostic sample must be finite")
        samples.append(sample)

    def observe_inference_latency_ms(self, value: float) -> None:
        self._append(self.inference_latency_ms, value)

    def observe_observation_age_ms(self, value: float) -> None:
        self._append(self.observation_age_ms, value)

    def observe_observation_skew_ms(self, value: float) -> None:
        self._append(self.observation_skew_ms, value)

    def observe_schedule_lateness_ms(self, value: float) -> None:
        self._append(self.schedule_lateness_ms, value)

    def observe_publication_interval_ms(self, value: float) -> None:
        self._append(self.publication_interval_ms, value)

    @staticmethod
    def _latest(samples: deque[float]) -> float | None:
        return samples[-1] if samples else None

    def snapshot(self) -> dict[str, int | float]:
        """Return latest timings and the failure counts since the last report."""
        result: dict[str, int | float] = {
            "safety_rejection_count": self.safety_rejection_count,
            "command_progress_timeout_count": self.command_progress_timeout_count,
        }
        optional_counts = {
            "ik_rejection_count": self.ik_rejection_count,
            "stale_prediction_count": self.stale_prediction_count,
        }
        result.update({name: count for name, count in optional_counts.items() if count})
        for name, samples in (
            ("inference_latency_ms", self.inference_latency_ms),
            ("observation_age_ms", self.observation_age_ms),
            ("observation_skew_ms", self.observation_skew_ms),
            ("schedule_lateness_ms", self.schedule_lateness_ms),
            ("publication_interval_ms", self.publication_interval_ms),
        ):
            latest = self._latest(samples)
            if latest is not None:
                result[name] = latest
        return result

    def flush(self, *, prefix: str, debug: bool = False) -> None:
        """Log current diagnostics and reset only interval failure counts."""
        rendered = " ".join(
            f"{key}={value}" for key, value in sorted(self.snapshot().items())
        )
        log = logger.debug if debug else logger.info
        log("%s: %s", prefix, rendered)
        self.safety_rejection_count = 0
        self.command_progress_timeout_count = 0
        self.ik_rejection_count = 0
        self.stale_prediction_count = 0


def flush_every(
    stats: PolicyStats,
    *,
    last_ns: int,
    interval_s: float = 1.0,
    prefix: str,
    debug: bool = False,
) -> int:
    """Log stats at a monotonic, bounded reporting cadence."""
    now_ns = time.monotonic_ns()
    if now_ns - last_ns >= int(interval_s * 1e9):
        stats.flush(prefix=prefix, debug=debug)
        return now_ns
    return last_ns
