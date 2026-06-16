"""VR frame quality check and timeout detection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackingQualityResult:
    ok: bool
    stale: bool = False
    tracking_lost: bool = False
    age_s: float = float("inf")
    lost_duration_s: float = 0.0


@dataclass
class TrackingQualityConfig:
    max_frame_age_s: float = 0.2
    ema_loss_timeout_s: float = 1.0  # continuous loss > this → emergency_stop


class TrackingQuality:
    """Per-frame VR tracking quality gate.

    Usage per _tick():
        result = tq.check(vr_frame)
        if not result.ok:
            # hold in place, skip IK/retarget
            ...
        if result.tracking_lost:
            # escalate to emergency_stop
            ...
    """

    def __init__(self, config: TrackingQualityConfig | None = None) -> None:
        self.config = config or TrackingQualityConfig()
        self._lost_since: float | None = None  # perf_counter when tracking was first lost

    def check(self, vr_frame: dict[str, Any] | None) -> TrackingQualityResult:
        now = time.perf_counter()

        if vr_frame is None:
            return self._handle_missing(now)

        age_s = self._frame_age(vr_frame)
        stale = age_s > self.config.max_frame_age_s

        if stale:
            result = self._handle_missing(now)
            result.age_s = age_s
            return result

        # Frame is fresh — clear lost state
        self._lost_since = None
        return TrackingQualityResult(ok=True, age_s=age_s)

    def reset(self) -> None:
        self._lost_since = None

    @property
    def is_lost(self) -> bool:
        return self._lost_since is not None

    def _handle_missing(self, now: float) -> TrackingQualityResult:
        if self._lost_since is None:
            self._lost_since = now
        lost_duration = now - self._lost_since
        tracking_lost = lost_duration > self.config.ema_loss_timeout_s
        return TrackingQualityResult(
            ok=False,
            stale=True,
            tracking_lost=tracking_lost,
            lost_duration_s=lost_duration,
        )

    @staticmethod
    def _frame_age(frame: dict[str, Any]) -> float:
        local_recv = frame.get("local_recv_ns")
        if local_recv is not None:
            return (time.monotonic_ns() - local_recv) * 1e-9
        return float("inf")


def example() -> None:
    tq = TrackingQuality()

    # Simulate fresh frame
    fresh = {
        "local_recv_ns": time.monotonic_ns(),
        "sequence_id": 42,
    }
    result = tq.check(fresh)
    print(f"fresh: ok={result.ok}, age={result.age_s:.4f}s")

    # Simulate None
    result = tq.check(None)
    print(f"none: ok={result.ok}, lost_duration={result.lost_duration_s:.4f}s")

    # Simulate stale
    stale = {
        "local_recv_ns": time.monotonic_ns() - int(0.3 * 1e9),
        "sequence_id": 40,
    }
    result = tq.check(stale)
    print(f"stale: ok={result.ok}, stale={result.stale}, tracking_lost={result.tracking_lost}")


if __name__ == "__main__":
    example()
