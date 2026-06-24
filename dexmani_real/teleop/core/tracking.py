"""VR frame quality check, timeout detection, and frame drop policy.

.. deprecated::
    This module is superseded by inlined VR tracking logic in
    TeleopController (controller.py:691-713).  FrameDropPolicy and
    TrackingQuality classes are retained for reference/documentation
    but are no longer imported or used by the controller.

    The inlined version provides the same three-tier staleness:
      - Tier 0 (FRESH):       age <= soft_threshold  → use frame normally
      - Tier 1 (SOFT_STALE):  soft < age <= hard      → hold position
      - Tier 2 (HARD_STALE):  hard < age <= emergency → soft deceleration
      - Tier 3 (EMERGENCY):   age > emergency         → immediate emergency stop

Ref: DexUMI drop-oldest backpressure.
     BunnyVisionPro FrameAge gate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameDropPolicy:
    """Three-tier frame staleness policy.

    Used by both TrackingQuality (real-time) and TimestampAligner (offline).
    """

    soft_threshold_s: float = 0.1      # Tier 1: soft deceleration
    hard_threshold_s: float = 0.5      # Tier 2: hold position
    emergency_threshold_s: float = 1.0  # Tier 3: emergency stop

    def classify(self, age_s: float) -> int:
        """Classify frame age into tier 0 (fresh), 1, 2, or 3 (emergency)."""
        if age_s <= self.soft_threshold_s:
            return 0
        if age_s <= self.hard_threshold_s:
            return 1
        if age_s <= self.emergency_threshold_s:
            return 2
        return 3

    @property
    def tier_names(self) -> dict[int, str]:
        return {0: "FRESH", 1: "SOFT_STALE", 2: "HARD_STALE", 3: "EMERGENCY"}


@dataclass
class TrackingQualityResult:
    ok: bool
    stale: bool = False
    tracking_lost: bool = False
    age_s: float = float("inf")
    lost_duration_s: float = 0.0
    tier: int = 0  # FrameDropPolicy tier (0-3)


@dataclass
class TrackingQualityConfig:
    drop_policy: FrameDropPolicy | None = None  # if None, uses defaults


class TrackingQuality:
    """Per-frame VR tracking quality gate with tier-based staleness.

    Usage per _tick():
        result = tq.check(vr_frame)
        if result.tier >= 1:  # SOFT_STALE
            # apply soft deceleration, skip IK/retarget
            ...
        if result.tracking_lost:  # tier >= 2
            # escalate to emergency_stop
            ...
    """

    def __init__(self, config: TrackingQualityConfig | None = None) -> None:
        self.config = config or TrackingQualityConfig()
        self._lost_since: float | None = None  # perf_counter when tracking was first lost
        self._drop_policy = self.config.drop_policy or FrameDropPolicy()

    def check(self, vr_frame: dict[str, Any] | None) -> TrackingQualityResult:
        now = time.perf_counter()

        if vr_frame is None:
            return self._handle_missing(now)

        age_s = self.frame_age(vr_frame)
        tier = self._drop_policy.classify(age_s)

        if tier >= 1:
            result = self._handle_missing(now)
            result.age_s = age_s
            result.tier = tier
            return result

        # Frame is fresh — clear lost state
        self._lost_since = None
        return TrackingQualityResult(ok=True, age_s=age_s, tier=0)

    def reset(self) -> None:
        self._lost_since = None

    @property
    def is_lost(self) -> bool:
        return self._lost_since is not None

    @property
    def drop_policy(self) -> FrameDropPolicy:
        return self._drop_policy

    def _handle_missing(self, now: float) -> TrackingQualityResult:
        if self._lost_since is None:
            self._lost_since = now
        lost_duration = now - self._lost_since
        tier = self._drop_policy.classify(lost_duration)
        tracking_lost = tier >= 2
        return TrackingQualityResult(
            ok=False,
            stale=True,
            tracking_lost=tracking_lost,
            lost_duration_s=lost_duration,
            tier=tier,
        )

    @staticmethod
    def frame_age(frame: dict[str, Any]) -> float:
        local_recv = frame.get("local_recv_ns")
        if local_recv is not None:
            return (time.monotonic_ns() - local_recv) * 1e-9
        return float("inf")

