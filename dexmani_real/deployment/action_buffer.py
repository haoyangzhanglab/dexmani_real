"""Pure rolling scheduler for immutable learned-policy action chunks.

``ActionBuffer`` owns no clock, shared memory, hardware, planner, or safety
gate.  Its caller supplies logical monotonic timestamps and resolves the
returned endpoint through the normal candidate/publication boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.deployment.contracts import JointActionChunk


def _require_integer(value: object, *, name: str, positive: bool) -> int:
    """Return one exact integer, rejecting bools and non-positive values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _copy_chunk(chunk: JointActionChunk) -> JointActionChunk:
    """Reconstruct an independent chunk at the scheduler ownership boundary."""
    return JointActionChunk(
        arm_qpos=(
            None if chunk.arm_qpos is None else np.array(chunk.arm_qpos, copy=True)
        ),
        hand_qpos=(
            None if chunk.hand_qpos is None else np.array(chunk.hand_qpos, copy=True)
        ),
        target_monotonic_ns=np.array(chunk.target_monotonic_ns, copy=True),
        valid_mask=np.array(chunk.valid_mask, copy=True),
        ee_pos=None if chunk.ee_pos is None else np.array(chunk.ee_pos, copy=True),
        ee_rot6d=(
            None if chunk.ee_rot6d is None else np.array(chunk.ee_rot6d, copy=True)
        ),
    )


@dataclass(frozen=True)
class BufferedPlan:
    """One copied plan plus the immutable provenance/deadline it needs."""

    plan_id: int
    run_generation: int
    observation_id: int
    observation_anchor_ns: int
    observation_latest_source_ns: int
    inference_finished_ns: int
    deadline_ns: int
    chunk: JointActionChunk

    def __post_init__(self) -> None:
        plan_id = _require_integer(self.plan_id, name="plan_id", positive=True)
        generation = _require_integer(
            self.run_generation, name="run_generation", positive=False
        )
        observation_id = _require_integer(
            self.observation_id, name="observation_id", positive=True
        )
        anchor = _require_integer(
            self.observation_anchor_ns, name="observation_anchor_ns", positive=True
        )
        source = _require_integer(
            self.observation_latest_source_ns,
            name="observation_latest_source_ns",
            positive=True,
        )
        finished = _require_integer(
            self.inference_finished_ns,
            name="inference_finished_ns",
            positive=True,
        )
        deadline = _require_integer(self.deadline_ns, name="deadline_ns", positive=True)
        if not source <= anchor <= finished:
            raise ValueError(
                "BufferedPlan provenance must satisfy source <= anchor <= finished"
            )
        if not isinstance(self.chunk, JointActionChunk):
            raise TypeError("chunk must be a JointActionChunk")
        if np.any(np.asarray(self.chunk.target_monotonic_ns) <= 0):
            raise ValueError("chunk target_monotonic_ns must be positive")

        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "run_generation", generation)
        object.__setattr__(self, "observation_id", observation_id)
        object.__setattr__(self, "observation_anchor_ns", anchor)
        object.__setattr__(self, "observation_latest_source_ns", source)
        object.__setattr__(self, "inference_finished_ns", finished)
        object.__setattr__(self, "deadline_ns", deadline)
        object.__setattr__(self, "chunk", _copy_chunk(self.chunk))

    @property
    def identity(self) -> tuple[int, int]:
        """Ordering identity: newer observation first, then newer plan."""
        return (self.observation_id, self.plan_id)


@dataclass(frozen=True)
class EndpointToken:
    """Immutable capability for exactly one endpoint finalization.

    Its public fields are provenance for logs only.  ``ActionBuffer`` accepts
    only the exact issued object; callers obtain it from
    :meth:`ActionBuffer.peek_due`.
    """

    token_id: int
    run_generation: int
    target_ns: int
    plan_id: int
    observation_id: int
    step_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_id",
            _require_integer(self.token_id, name="token_id", positive=True),
        )
        object.__setattr__(
            self,
            "run_generation",
            _require_integer(
                self.run_generation, name="run_generation", positive=False
            ),
        )
        object.__setattr__(
            self,
            "target_ns",
            _require_integer(self.target_ns, name="target_ns", positive=True),
        )
        object.__setattr__(
            self,
            "plan_id",
            _require_integer(self.plan_id, name="plan_id", positive=True),
        )
        object.__setattr__(
            self,
            "observation_id",
            _require_integer(self.observation_id, name="observation_id", positive=True),
        )
        object.__setattr__(
            self,
            "step_index",
            _require_integer(self.step_index, name="step_index", positive=False),
        )


class BufferCoverage(str, Enum):
    """Whether a buffer has a currently evaluable endpoint."""

    DUE = "due"
    FUTURE = "future"
    EXHAUSTED = "exhausted"


class PushStatus(str, Enum):
    """Typed admission outcome for one plan."""

    ACCEPTED = "accepted"
    WRONG_GENERATION = "wrong_generation"
    DUPLICATE = "duplicate"
    STALE_IDENTITY = "stale_identity"
    DEADLINE_CLOSED = "deadline_closed"
    ALL_FINALIZED = "all_finalized"


@dataclass(frozen=True)
class PushResult:
    """Result of one bounded plan admission attempt."""

    status: PushStatus
    evicted_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.status, PushStatus):
            raise TypeError("status must be a PushStatus")
        if self.evicted_count < 0:
            raise ValueError("evicted_count must be non-negative")

    @property
    def accepted(self) -> bool:
        return self.status is PushStatus.ACCEPTED


@dataclass(frozen=True)
class PeekResult:
    """One pure selection result; only DUE carries a finalization token."""

    coverage: BufferCoverage
    plan: BufferedPlan | None = None
    step_index: int | None = None
    token: EndpointToken | None = None
    coalesced_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.coverage, BufferCoverage):
            raise TypeError("coverage must be a BufferCoverage")
        if self.coalesced_count < 0:
            raise ValueError("coalesced_count must be non-negative")
        due = self.coverage is BufferCoverage.DUE
        populated = (
            self.plan is not None
            and self.step_index is not None
            and self.token is not None
        )
        if due != populated:
            raise ValueError("DUE result must contain exactly one selected endpoint")
        if not due and self.coalesced_count != 0:
            raise ValueError("only a DUE result may coalesce endpoints")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("step_index must be non-negative")


def compute_max_buffered_plans(max_plan_age_s: float, inference_hz: float) -> int:
    """Compute the bounded scheduler capacity from Real-owned runtime values."""
    for name, value in (
        ("max_plan_age_s", max_plan_age_s),
        ("inference_hz", inference_hz),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite positive number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    return min(32, max(2, math.ceil(float(max_plan_age_s) * float(inference_hz)) + 2))


class ActionBuffer:
    """Bounded latest-wins scheduler with commit/discard watermarks.

    ``reset`` establishes the only generation admitted by this instance.  Each
    pushed plan receives stable endpoint tokens immediately; peeking is fully
    side-effect free, so a transient publication deferral returns the exact
    same token on the following tick.
    """

    def __init__(self, *, max_buffered_plans: int) -> None:
        capacity = _require_integer(
            max_buffered_plans, name="max_buffered_plans", positive=True
        )
        if not 2 <= capacity <= 32:
            raise ValueError("max_buffered_plans must be in [2, 32]")
        self._max_buffered_plans = capacity
        self._run_generation: int | None = None
        self._plans: dict[tuple[int, int], BufferedPlan] = {}
        self._tokens: dict[int, EndpointToken] = {}
        self._token_ids_by_plan: dict[tuple[int, int], tuple[int, ...]] = {}
        # A target remains owned by the newest accepted plan that covered it
        # even after that plan expires, while an older retained plan still has
        # the same target. This prevents exact-target resurrection.
        self._superseded_winner_by_target: dict[int, tuple[int, int]] = {}
        self._finalized_through_ns = 0
        self._max_seen_identity: tuple[int, int] | None = None
        self._next_token_id = 1

    @property
    def finalized_through_ns(self) -> int:
        """Largest target finalized by commit or motion discard this run."""
        return self._finalized_through_ns

    @property
    def plan_count(self) -> int:
        """Number of retained plans (always bounded by configured capacity)."""
        return len(self._plans)

    def reset(self, *, run_generation: int) -> None:
        """Discard every retained endpoint before a new run generation starts."""
        self._run_generation = _require_integer(
            run_generation, name="run_generation", positive=False
        )
        self._plans.clear()
        self._tokens.clear()
        self._token_ids_by_plan.clear()
        self._superseded_winner_by_target.clear()
        self._finalized_through_ns = 0
        self._max_seen_identity = None

    def push(self, plan: BufferedPlan, *, now_ns: int) -> PushResult:
        """Admit one plan or return its explicit non-mutating rejection reason."""
        now = _require_integer(now_ns, name="now_ns", positive=True)
        self._require_reset()
        if not isinstance(plan, BufferedPlan):
            raise TypeError("plan must be a BufferedPlan")
        self._prune_closed(now)

        if plan.run_generation != self._run_generation:
            return PushResult(PushStatus.WRONG_GENERATION)
        identity = plan.identity
        if identity in self._plans or identity == self._max_seen_identity:
            return PushResult(PushStatus.DUPLICATE)
        if self._max_seen_identity is not None and identity < self._max_seen_identity:
            return PushResult(PushStatus.STALE_IDENTITY)
        if now >= plan.deadline_ns:
            return PushResult(PushStatus.DEADLINE_CLOSED)
        if not self._has_admissible_endpoint(plan, now):
            return PushResult(PushStatus.ALL_FINALIZED)

        evicted_count = 0
        while len(self._plans) >= self._max_buffered_plans:
            oldest_identity = min(self._plans)
            self._remove_plan(oldest_identity)
            evicted_count += 1
        self._plans[identity] = plan
        self._token_ids_by_plan[identity] = self._allocate_plan_tokens(plan)
        self._register_supersession(plan)
        self._max_seen_identity = identity
        return PushResult(PushStatus.ACCEPTED, evicted_count=evicted_count)

    def peek_due(self, *, now_ns: int) -> PeekResult:
        """Choose one latest due endpoint without changing scheduler state."""
        now = _require_integer(now_ns, name="now_ns", positive=True)
        self._require_reset()
        entries = self._live_entries(now)
        due = [entry for entry in entries if entry[0] <= now]
        if not due:
            coverage = BufferCoverage.FUTURE if entries else BufferCoverage.EXHAUSTED
            return PeekResult(coverage)

        selected_target = max(entry[0] for entry in due)
        target_entries = [entry for entry in due if entry[0] == selected_target]
        _, plan, step_index, token = max(
            target_entries, key=lambda entry: entry[1].identity
        )
        coalesced_targets = {
            target for target, *_rest in due if target < selected_target
        }
        return PeekResult(
            BufferCoverage.DUE,
            plan=plan,
            step_index=step_index,
            token=token,
            coalesced_count=len(coalesced_targets),
        )

    def coverage(self, *, now_ns: int) -> BufferCoverage:
        """Report DUE/FUTURE/EXHAUSTED without selecting or finalizing an endpoint."""
        return self.peek_due(now_ns=now_ns).coverage

    def commit(self, token: EndpointToken) -> None:
        """Finalize a successfully validated/published endpoint and earlier due ones."""
        self._finalize(token)

    def discard(self, token: EndpointToken, *, reason_code: str) -> None:
        """Finalize one rejected motion endpoint without allowing stale fallback."""
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ValueError("reason_code must be a non-empty string")
        self._finalize(token)

    def _require_reset(self) -> None:
        if self._run_generation is None:
            raise RuntimeError("ActionBuffer.reset must be called before use")

    def _has_live_endpoint(self, plan: BufferedPlan, now_ns: int) -> bool:
        if plan.run_generation != self._run_generation or now_ns >= plan.deadline_ns:
            return False
        for index, target in enumerate(plan.chunk.target_monotonic_ns):
            if self._endpoint_is_live(plan, index):
                return True
        return False

    def _has_admissible_endpoint(self, plan: BufferedPlan, now_ns: int) -> bool:
        """Validate a new plan before it takes ownership of overlapping targets."""
        if plan.run_generation != self._run_generation or now_ns >= plan.deadline_ns:
            return False
        for step_index, target in enumerate(plan.chunk.target_monotonic_ns):
            target_ns = int(target)
            if (
                int(plan.chunk.valid_mask[step_index]) == 1
                and target_ns < plan.deadline_ns
                and target_ns > self._finalized_through_ns
            ):
                return True
        return False

    def _allocate_plan_tokens(self, plan: BufferedPlan) -> tuple[int, ...]:
        token_ids: list[int] = []
        for step_index, target in enumerate(plan.chunk.target_monotonic_ns):
            token = EndpointToken(
                token_id=self._next_token_id,
                run_generation=plan.run_generation,
                target_ns=int(target),
                plan_id=plan.plan_id,
                observation_id=plan.observation_id,
                step_index=step_index,
            )
            self._tokens[token.token_id] = token
            token_ids.append(token.token_id)
            self._next_token_id += 1
        return tuple(token_ids)

    def _live_entries(
        self, now_ns: int
    ) -> list[tuple[int, BufferedPlan, int, EndpointToken]]:
        entries: list[tuple[int, BufferedPlan, int, EndpointToken]] = []
        for identity, plan in self._plans.items():
            if (
                plan.run_generation != self._run_generation
                or now_ns >= plan.deadline_ns
            ):
                continue
            token_ids = self._token_ids_by_plan[identity]
            for step_index, target in enumerate(plan.chunk.target_monotonic_ns):
                target_ns = int(target)
                if not self._endpoint_is_live(plan, step_index):
                    continue
                entries.append(
                    (target_ns, plan, step_index, self._tokens[token_ids[step_index]])
                )
        return entries

    def _prune_closed(self, now_ns: int) -> None:
        """Forget plans whose remaining endpoints cannot become live again."""
        for identity, plan in tuple(self._plans.items()):
            if now_ns >= plan.deadline_ns or not self._has_live_endpoint(plan, now_ns):
                self._remove_plan(identity)

    def _remove_plan(self, identity: tuple[int, int]) -> None:
        removed = self._plans.pop(identity, None)
        for token_id in self._token_ids_by_plan.pop(identity, ()):
            self._tokens.pop(token_id, None)
        if removed is not None:
            self._cleanup_supersession(removed.chunk.target_monotonic_ns)

    def _finalize(self, token: EndpointToken) -> None:
        self._require_reset()
        if not isinstance(token, EndpointToken):
            raise RuntimeError("endpoint token is invalid")
        if token.run_generation != self._run_generation:
            raise RuntimeError("endpoint token generation is stale")
        if token.target_ns <= self._finalized_through_ns:
            raise RuntimeError("endpoint token was already finalized")
        issued = self._tokens.get(token.token_id)
        if issued is not token:
            raise RuntimeError("endpoint token is unknown or forged")
        identity = (token.observation_id, token.plan_id)
        plan = self._plans.get(identity)
        if plan is None:
            raise RuntimeError("endpoint token plan is no longer retained")
        if (
            token.step_index >= len(plan.chunk.target_monotonic_ns)
            or int(plan.chunk.target_monotonic_ns[token.step_index]) != token.target_ns
        ):
            raise RuntimeError("endpoint token does not match its plan endpoint")
        if not self._endpoint_is_live(plan, token.step_index):
            raise RuntimeError("endpoint token is no longer the latest target winner")

        self._finalized_through_ns = token.target_ns
        self._cleanup_supersession()
        self._prune_closed(now_ns=0)

    def _endpoint_is_live(self, plan: BufferedPlan, step_index: int) -> bool:
        """Return whether this endpoint remains the unfinalized target owner."""
        target_ns = int(plan.chunk.target_monotonic_ns[step_index])
        return (
            int(plan.chunk.valid_mask[step_index]) == 1
            and target_ns < plan.deadline_ns
            and target_ns > self._finalized_through_ns
            and self._superseded_winner_by_target.get(target_ns) == plan.identity
        )

    def _register_supersession(self, plan: BufferedPlan) -> None:
        """Make this accepted plan the permanent owner of every live target."""
        for step_index, target in enumerate(plan.chunk.target_monotonic_ns):
            target_ns = int(target)
            if (
                int(plan.chunk.valid_mask[step_index]) == 1
                and target_ns < plan.deadline_ns
                and target_ns > self._finalized_through_ns
            ):
                self._superseded_winner_by_target[target_ns] = plan.identity

    def _cleanup_supersession(self, targets: np.ndarray | None = None) -> None:
        """Retain ghost winners only while an older endpoint could resurrect."""
        candidate_targets = (
            tuple(self._superseded_winner_by_target)
            if targets is None
            else {int(target) for target in targets}
        )
        for target_ns in candidate_targets:
            if target_ns <= self._finalized_through_ns or not self._target_is_covered(
                target_ns
            ):
                self._superseded_winner_by_target.pop(target_ns, None)

    def _target_is_covered(self, target_ns: int) -> bool:
        """Whether any retained plan still has a structurally valid target."""
        for plan in self._plans.values():
            if plan.run_generation != self._run_generation:
                continue
            for step_index, target in enumerate(plan.chunk.target_monotonic_ns):
                if (
                    int(target) == target_ns
                    and int(plan.chunk.valid_mask[step_index]) == 1
                    and target_ns < plan.deadline_ns
                    and target_ns > self._finalized_through_ns
                ):
                    return True
        return False


__all__ = [
    "ActionBuffer",
    "BufferedPlan",
    "BufferCoverage",
    "EndpointToken",
    "PeekResult",
    "PushResult",
    "PushStatus",
    "compute_max_buffered_plans",
]
