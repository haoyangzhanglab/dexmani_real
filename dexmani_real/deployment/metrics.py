"""Small, direct deployment diagnostics for control-quality investigation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyStats:
    """Latest control-quality diagnostics and cumulative rejection counts."""

    inference_latency_ms: float | None = None
    observation_age_ms: float | None = None
    observation_skew_ms: float | None = None
    publication_interval_ms: float | None = None
    publication_input_age_ms: float | None = None
    safety_rejection_count: int = 0
    ik_rejection_count: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def count_rejection(self, reason: str) -> None:
        """Count one rejected policy step without a second visible line.

        The rejection itself is already visible exactly once: a recoverable
        miss prints the chunk-boundary ``[DROP]`` line, and a contract
        violation logs critically. Reasons are summarized at episode end.
        """
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def log_summary(self) -> None:
        """Report episode counts without presenting old timings as live metrics."""
        logger.info(
            "policy summary: safety_rejections=%d ik_rejections=%d%s",
            self.safety_rejection_count,
            self.ik_rejection_count,
            " | " + "; ".join(
                f"{reason}: {count}" for reason, count in self.rejection_reasons.items()
            ) if self.rejection_reasons else "",
        )

    def snapshot(self) -> dict[str, int | float]:
        """Return latest timings and cumulative counts for this rollout."""
        result: dict[str, int | float] = {
            "safety_rejection_count": self.safety_rejection_count,
        }
        if self.ik_rejection_count:
            result["ik_rejection_count"] = self.ik_rejection_count
        for name in (
            "inference_latency_ms",
            "observation_age_ms",
            "observation_skew_ms",
            "publication_interval_ms",
            "publication_input_age_ms",
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
