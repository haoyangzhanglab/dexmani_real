"""Small, direct deployment diagnostics for control-quality investigation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyStats:
    """Latest control-quality diagnostics and cumulative rejection counts."""

    inference_latency_ms: float | None = None
    observation_age_ms: float | None = None
    observation_skew_ms: float | None = None
    schedule_lateness_ms: float | None = None
    publication_interval_ms: float | None = None
    skipped_prefix_steps: int | None = None
    safety_rejection_count: int = 0
    command_progress_timeout_count: int = 0
    ik_rejection_count: int = 0
    stale_prediction_count: int = 0

    def snapshot(self) -> dict[str, int | float]:
        """Return latest timings and cumulative counts for this rollout."""
        result: dict[str, int | float] = {
            "safety_rejection_count": self.safety_rejection_count,
            "command_progress_timeout_count": self.command_progress_timeout_count,
        }
        optional_counts = {
            "ik_rejection_count": self.ik_rejection_count,
            "stale_prediction_count": self.stale_prediction_count,
        }
        result.update({name: count for name, count in optional_counts.items() if count})
        for name in (
            "inference_latency_ms",
            "observation_age_ms",
            "observation_skew_ms",
            "schedule_lateness_ms",
            "publication_interval_ms",
            "skipped_prefix_steps",
        ):
            value = getattr(self, name)
            if value is not None and math.isfinite(value) and value >= 0:
                result[name] = value
        return result

    def flush(self, *, prefix: str, debug: bool = False) -> None:
        """Log without consuming the counts needed by the rollout result."""
        rendered = " ".join(
            f"{key}={value}" for key, value in sorted(self.snapshot().items())
        )
        log = logger.debug if debug else logger.info
        log("%s: %s", prefix, rendered)


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
