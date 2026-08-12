"""Small, deterministic state transitions used by the teleoperation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dexmani_real.policy.runtime import ActionCandidate

HoldApplication = Literal["idle", "pending", "applied", "failed"]


def _home_handoff_state(*, candidate_present: bool, candidate_applied: bool) -> str:
    """Describe the hold candidate that homing is about to supersede."""
    if not candidate_present:
        return "idle"
    return "applied" if candidate_applied else "pending"


@dataclass
class ControlHold:
    """Track one run-scoped measured hold and its exact apply acknowledgement."""

    reason: str | None = None
    candidate: ActionCandidate | None = None
    record_candidate: ActionCandidate | None = None
    applied: bool = False
    applied_monotonic_ns: int = 0
    deadline_s: float | None = None

    @property
    def active(self) -> bool:
        return self.reason is not None

    @property
    def application_pending(self) -> bool:
        """Whether clean exit must wait for the correlated apply ACK."""
        return self.candidate is not None and not self.applied

    def pause_without_candidate(self, reason: str) -> None:
        """Invalidate local motion while feedback is too unhealthy to publish a hold."""
        self.reason = reason
        self.candidate = None
        self.record_candidate = None
        self.applied = False
        self.applied_monotonic_ns = 0
        self.deadline_s = None

    def relabel(self, reason: str) -> None:
        """Keep an existing hold but update why re-anchoring is required."""
        if not self.active:
            raise RuntimeError("cannot relabel an inactive control hold")
        self.reason = reason

    def begin(self, reason: str, candidate: ActionCandidate, *, deadline_s: float) -> None:
        """Begin waiting for the exact arm/hand candidate to be applied."""
        self.reason = reason
        self.candidate = candidate
        self.record_candidate = candidate
        self.applied = False
        self.applied_monotonic_ns = 0
        self.deadline_s = float(deadline_s)

    def observe_delivery(self, sent_at_s: float | None, *, now_s: float, hold_delivery_s: float = 0.15) -> HoldApplication:
        """Check whether the hold has had time to propagate through the workers.

        Without ACK protocol, we wait ``hold_delivery_s`` after the command was
        sent (enough for the arm queue to drain and both workers to apply).
        """
        if self.candidate is None or self.applied:
            return "idle"
        if sent_at_s is not None and now_s - sent_at_s >= hold_delivery_s:
            self.applied = True
            self.applied_monotonic_ns = int(sent_at_s * 1e9)
            self.deadline_s = None
            return "applied"
        if self.deadline_s is not None and now_s >= self.deadline_s:
            return "failed"
        return "pending"

    def take_record_candidate(self) -> ActionCandidate | None:
        """Return the hold candidate once for recording provenance."""
        candidate = self.record_candidate
        self.record_candidate = None
        return candidate

    def clear(self) -> tuple[str, str | None]:
        """Clear the hold and return its pre-clear handoff state and reason."""
        state = _home_handoff_state(candidate_present=self.candidate is not None, candidate_applied=self.applied)
        reason = self.reason
        self.reason = None
        self.candidate = None
        self.record_candidate = None
        self.applied = False
        self.applied_monotonic_ns = 0
        self.deadline_s = None
        return state, reason
