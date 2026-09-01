"""Pure unit tests for rolling learned-policy endpoint scheduling."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from dexmani_real.deployment.action_buffer import (
    ActionBuffer,
    BufferCoverage,
    BufferedPlan,
    EndpointToken,
    PushStatus,
    compute_max_buffered_plans,
)
from dexmani_real.deployment.contracts import JointActionChunk


def _chunk(
    targets: tuple[int, ...], *, mask: tuple[int, ...] | None = None
) -> JointActionChunk:
    count = len(targets)
    return JointActionChunk(
        arm_qpos=np.zeros((count, 7), dtype=np.float64),
        hand_qpos=np.zeros((count, 12), dtype=np.float64),
        target_monotonic_ns=np.asarray(targets, dtype=np.uint64),
        valid_mask=np.asarray(
            (1,) * count if mask is None else mask,
            dtype=np.uint8,
        ),
    )


def _plan(
    plan_id: int,
    observation_id: int,
    targets: tuple[int, ...],
    *,
    generation: int = 7,
    deadline_ns: int | None = None,
    mask: tuple[int, ...] | None = None,
) -> BufferedPlan:
    return BufferedPlan(
        plan_id=plan_id,
        run_generation=generation,
        observation_id=observation_id,
        observation_anchor_ns=40,
        observation_latest_source_ns=30,
        inference_finished_ns=50,
        deadline_ns=(max(targets) + 1_000 if deadline_ns is None else deadline_ns),
        chunk=_chunk(targets, mask=mask),
    )


class ActionBufferTest(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = ActionBuffer(max_buffered_plans=4)
        self.buffer.reset(run_generation=7)

    def _scheduler_state(self) -> tuple[object, ...]:
        """Snapshot every private field that affects scheduling or bounds."""
        plans = tuple(
            (
                identity,
                id(plan),
                plan.deadline_ns,
                tuple(int(value) for value in plan.chunk.target_monotonic_ns),
                tuple(int(value) for value in plan.chunk.valid_mask),
            )
            for identity, plan in sorted(self.buffer._plans.items())
        )
        tokens = tuple(
            (
                token_id,
                id(token),
                token.run_generation,
                token.target_ns,
                token.plan_id,
                token.observation_id,
                token.step_index,
            )
            for token_id, token in sorted(self.buffer._tokens.items())
        )
        token_ids_by_plan = tuple(
            (identity, token_ids)
            for identity, token_ids in sorted(self.buffer._token_ids_by_plan.items())
        )
        return (
            self.buffer._max_buffered_plans,
            self.buffer._run_generation,
            self.buffer._next_token_id,
            self.buffer._finalized_through_ns,
            self.buffer._max_seen_identity,
            plans,
            tokens,
            token_ids_by_plan,
            tuple(sorted(self.buffer._superseded_winner_by_target.items())),
        )

    def test_same_target_uses_latest_observation_then_plan(self) -> None:
        first = _plan(1, 1, (100,))
        self.assertTrue(self.buffer.push(first, now_ns=1).accepted)
        self.assertEqual(
            self.buffer.push(first, now_ns=1).status,
            PushStatus.DUPLICATE,
        )
        self.assertTrue(self.buffer.push(_plan(2, 2, (100,)), now_ns=1).accepted)
        self.assertTrue(self.buffer.push(_plan(3, 2, (100,)), now_ns=1).accepted)

        result = self.buffer.peek_due(now_ns=100)

        self.assertEqual(result.coverage, BufferCoverage.DUE)
        assert result.plan is not None and result.token is not None
        self.assertEqual((result.plan.observation_id, result.plan.plan_id), (2, 3))
        self.assertEqual(result.token.target_ns, 100)
        self.assertEqual(result.coalesced_count, 0)

    def test_fresh_overlap_wins_while_uncovered_old_future_remains_fallback(
        self,
    ) -> None:
        old = _plan(1, 1, (100, 200))
        fresh = _plan(2, 2, (100,))
        self.buffer.push(old, now_ns=1)
        self.buffer.push(fresh, now_ns=1)

        overlap = self.buffer.peek_due(now_ns=100)
        assert overlap.plan is not None and overlap.token is not None
        self.assertEqual(overlap.plan.identity, fresh.identity)
        self.buffer.commit(overlap.token)

        self.assertEqual(self.buffer.coverage(now_ns=199), BufferCoverage.FUTURE)
        fallback = self.buffer.peek_due(now_ns=200)
        assert fallback.plan is not None and fallback.token is not None
        self.assertEqual(fallback.plan.identity, old.identity)
        self.assertEqual(fallback.token.target_ns, 200)

    def test_multiple_overdue_targets_coalesce_by_distinct_logical_target(self) -> None:
        self.buffer.push(_plan(1, 1, (100, 200, 300)), now_ns=1)

        result = self.buffer.peek_due(now_ns=350)

        assert result.token is not None
        self.assertEqual(result.token.target_ns, 300)
        self.assertEqual(result.coalesced_count, 2)
        self.buffer.commit(result.token)
        self.assertEqual(self.buffer.finalized_through_ns, 300)
        self.assertEqual(self.buffer.coverage(now_ns=350), BufferCoverage.EXHAUSTED)

    def test_discard_finalizes_target_without_older_same_target_fallback(self) -> None:
        self.buffer.push(_plan(1, 1, (100,)), now_ns=1)
        self.buffer.push(_plan(2, 2, (100,)), now_ns=1)
        selected = self.buffer.peek_due(now_ns=100)
        assert selected.token is not None

        self.buffer.discard(selected.token, reason_code="collision_transition")

        self.assertEqual(self.buffer.coverage(now_ns=100), BufferCoverage.EXHAUSTED)
        rejected = self.buffer.push(_plan(3, 3, (100,)), now_ns=1)
        self.assertEqual(rejected.status, PushStatus.ALL_FINALIZED)
        with self.assertRaisesRegex(RuntimeError, "already finalized"):
            self.buffer.discard(selected.token, reason_code="again")

    def test_transient_deferral_repeated_peek_is_stable(self) -> None:
        self.buffer.push(_plan(1, 1, (100,)), now_ns=1)

        first = self.buffer.peek_due(now_ns=100)
        second = self.buffer.peek_due(now_ns=100)

        self.assertEqual(first.coverage, BufferCoverage.DUE)
        self.assertIs(first.token, second.token)
        self.assertEqual(first.token, second.token)
        self.assertEqual(first.step_index, second.step_index)
        self.assertEqual(self.buffer.finalized_through_ns, 0)

    def test_peek_due_and_coverage_are_side_effect_free(self) -> None:
        self.assertTrue(self.buffer.push(_plan(1, 1, (100, 200)), now_ns=1).accepted)
        self.assertTrue(self.buffer.push(_plan(2, 2, (100,)), now_ns=1).accepted)
        before = self._scheduler_state()

        first = self.buffer.peek_due(now_ns=100)
        self.assertEqual(first.coverage, BufferCoverage.DUE)
        self.assertEqual(self.buffer.coverage(now_ns=50), BufferCoverage.FUTURE)
        self.assertEqual(self.buffer.coverage(now_ns=100), BufferCoverage.DUE)
        second = self.buffer.peek_due(now_ns=100)

        self.assertIs(first.token, second.token)
        self.assertEqual(before, self._scheduler_state())

    def test_expired_newer_exact_target_never_revives_old_endpoint(self) -> None:
        old = _plan(1, 1, (100,), deadline_ns=1_000)
        newer = _plan(2, 2, (100,), deadline_ns=150)
        self.buffer.push(old, now_ns=1)
        self.buffer.push(newer, now_ns=1)
        selected = self.buffer.peek_due(now_ns=100)
        assert selected.plan is not None
        self.assertEqual(selected.plan.identity, newer.identity)

        # A rejected late push still prunes the expired newer owner. Its
        # exact target must not fall back to the older plan.
        self.assertEqual(
            self.buffer.push(_plan(3, 3, (300,), deadline_ns=150), now_ns=150).status,
            PushStatus.DEADLINE_CLOSED,
        )
        self.assertEqual(self.buffer.coverage(now_ns=150), BufferCoverage.EXHAUSTED)

    def test_expired_newer_target_keeps_old_uncovered_future_as_fallback(self) -> None:
        old = _plan(1, 1, (100, 200), deadline_ns=1_000)
        newer = _plan(2, 2, (100,), deadline_ns=150)
        self.buffer.push(old, now_ns=1)
        self.buffer.push(newer, now_ns=1)
        self.buffer.push(_plan(3, 3, (300,), deadline_ns=150), now_ns=150)

        self.assertEqual(self.buffer.coverage(now_ns=150), BufferCoverage.FUTURE)
        fallback = self.buffer.peek_due(now_ns=200)
        assert fallback.plan is not None and fallback.token is not None
        self.assertEqual(fallback.plan.identity, old.identity)
        self.assertEqual(fallback.token.target_ns, 200)

    def test_due_future_and_exhausted_coverage(self) -> None:
        self.buffer.push(_plan(1, 1, (200,)), now_ns=1)
        self.assertEqual(self.buffer.coverage(now_ns=100), BufferCoverage.FUTURE)
        self.assertEqual(self.buffer.coverage(now_ns=200), BufferCoverage.DUE)

        selected = self.buffer.peek_due(now_ns=200)
        assert selected.token is not None
        self.buffer.commit(selected.token)

        self.assertEqual(self.buffer.coverage(now_ns=200), BufferCoverage.EXHAUSTED)

    def test_mask_deadline_and_watermark_filter_live_endpoints(self) -> None:
        masked = self.buffer.push(_plan(1, 1, (100,), mask=(0,)), now_ns=1)
        self.assertEqual(masked.status, PushStatus.ALL_FINALIZED)
        deadline_target = self.buffer.push(
            _plan(2, 2, (200,), deadline_ns=200), now_ns=1
        )
        self.assertEqual(deadline_target.status, PushStatus.ALL_FINALIZED)
        closed = self.buffer.push(_plan(3, 3, (300,), deadline_ns=1), now_ns=1)
        self.assertEqual(closed.status, PushStatus.DEADLINE_CLOSED)

        self.buffer.push(_plan(4, 4, (400,)), now_ns=1)
        token = self.buffer.peek_due(now_ns=400).token
        assert token is not None
        self.buffer.commit(token)
        watermark = self.buffer.push(_plan(5, 5, (400,)), now_ns=1)
        self.assertEqual(watermark.status, PushStatus.ALL_FINALIZED)

    def test_deadline_cuts_tail_without_mutating_transport_mask(self) -> None:
        transport_mask = (0, 0, 1, 1, 1, 1)
        plan = _plan(
            1,
            1,
            (100, 200, 300, 400, 500, 600),
            deadline_ns=450,
            mask=transport_mask,
        )

        self.assertEqual(self.buffer.push(plan, now_ns=1).status, PushStatus.ACCEPTED)
        first = self.buffer.peek_due(now_ns=300)
        assert first.token is not None
        self.assertEqual(first.token.target_ns, 300)
        self.buffer.commit(first.token)
        second = self.buffer.peek_due(now_ns=400)
        assert second.token is not None
        self.assertEqual(second.token.target_ns, 400)
        self.buffer.commit(second.token)
        self.assertEqual(self.buffer.coverage(now_ns=400), BufferCoverage.EXHAUSTED)
        np.testing.assert_array_equal(
            plan.chunk.valid_mask,
            np.asarray(transport_mask, dtype=np.uint8),
        )

    def test_deadline_eliminating_every_transport_valid_target_is_not_admitted(
        self,
    ) -> None:
        result = self.buffer.push(
            _plan(
                1,
                1,
                (100, 200, 300, 400),
                deadline_ns=300,
                mask=(0, 0, 1, 1),
            ),
            now_ns=1,
        )

        self.assertEqual(result.status, PushStatus.ALL_FINALIZED)
        self.assertEqual(self.buffer.coverage(now_ns=1), BufferCoverage.EXHAUSTED)

    def test_generation_reset_invalidates_tokens_and_wrong_generation_is_typed(
        self,
    ) -> None:
        self.buffer.push(_plan(1, 1, (100,)), now_ns=1)
        token = self.buffer.peek_due(now_ns=100).token
        assert token is not None

        self.buffer.reset(run_generation=8)
        with self.assertRaisesRegex(RuntimeError, "generation is stale"):
            self.buffer.commit(token)
        wrong_generation = self.buffer.push(_plan(2, 2, (200,), generation=7), now_ns=1)
        self.assertEqual(wrong_generation.status, PushStatus.WRONG_GENERATION)

    def test_capacity_eviction_and_late_identity_rejection_are_deterministic(
        self,
    ) -> None:
        buffer = ActionBuffer(max_buffered_plans=2)
        buffer.reset(run_generation=7)
        self.assertTrue(buffer.push(_plan(1, 1, (100,)), now_ns=1).accepted)
        self.assertTrue(buffer.push(_plan(3, 3, (300,)), now_ns=1).accepted)

        admitted = buffer.push(_plan(4, 4, (400,)), now_ns=1)

        self.assertTrue(admitted.accepted)
        self.assertEqual(admitted.evicted_count, 1)
        self.assertEqual(buffer.plan_count, 2)
        stale = buffer.push(_plan(2, 2, (200,)), now_ns=1)
        self.assertEqual(stale.status, PushStatus.STALE_IDENTITY)

    def test_buffered_plan_copies_the_chunk_boundary(self) -> None:
        source = _chunk((100,))
        plan = BufferedPlan(
            plan_id=1,
            run_generation=7,
            observation_id=1,
            observation_anchor_ns=40,
            observation_latest_source_ns=30,
            inference_finished_ns=50,
            deadline_ns=1_000,
            chunk=source,
        )

        self.assertIsNot(plan.chunk, source)
        assert plan.chunk.arm_qpos is not None and source.arm_qpos is not None
        self.assertFalse(np.shares_memory(plan.chunk.arm_qpos, source.arm_qpos))
        source.arm_qpos.flags.writeable = True
        source.arm_qpos[0, 0] = 9.0
        self.assertEqual(plan.chunk.arm_qpos[0, 0], 0.0)

    def test_stale_forged_and_double_tokens_fail_closed(self) -> None:
        old = _plan(1, 1, (100,))
        self.buffer.push(old, now_ns=1)
        old_token = self.buffer.peek_due(now_ns=100).token
        assert old_token is not None
        self.buffer.push(_plan(2, 2, (100,)), now_ns=1)
        with self.assertRaisesRegex(RuntimeError, "latest target winner"):
            self.buffer.commit(old_token)

        newest = self.buffer.peek_due(now_ns=100).token
        assert newest is not None
        with self.assertRaisesRegex(RuntimeError, "unknown or forged"):
            self.buffer.commit(replace(newest, plan_id=99))
        self.buffer.commit(newest)
        with self.assertRaisesRegex(RuntimeError, "already finalized"):
            self.buffer.commit(newest)
        with self.assertRaisesRegex(ValueError, "reason_code"):
            self.buffer.discard(newest, reason_code=" ")

    def test_token_is_opaque_against_structural_reconstruction(self) -> None:
        self.buffer.push(_plan(1, 1, (100,)), now_ns=1)
        selected = self.buffer.peek_due(now_ns=100)
        assert selected.token is not None
        reconstructed = EndpointToken(
            token_id=selected.token.token_id,
            run_generation=selected.token.run_generation,
            target_ns=selected.token.target_ns,
            plan_id=selected.token.plan_id,
            observation_id=selected.token.observation_id,
            step_index=selected.token.step_index,
        )
        self.assertEqual(reconstructed, selected.token)
        self.assertIsNot(reconstructed, selected.token)
        with self.assertRaisesRegex(RuntimeError, "unknown or forged"):
            self.buffer.commit(reconstructed)

        self.buffer.commit(selected.token)

    def test_reset_is_required_and_constructor_contracts_are_strict(self) -> None:
        fresh = ActionBuffer(max_buffered_plans=2)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            fresh.peek_due(now_ns=1)
        with self.assertRaisesRegex(ValueError, r"\[2, 32\]"):
            ActionBuffer(max_buffered_plans=1)
        with self.assertRaisesRegex(TypeError, "max_buffered_plans"):
            ActionBuffer(max_buffered_plans=True)
        with self.assertRaisesRegex(ValueError, "source <= anchor <= finished"):
            BufferedPlan(
                plan_id=1,
                run_generation=0,
                observation_id=1,
                observation_anchor_ns=20,
                observation_latest_source_ns=30,
                inference_finished_ns=40,
                deadline_ns=100,
                chunk=_chunk((50,)),
            )
        with self.assertRaisesRegex(TypeError, "JointActionChunk"):
            BufferedPlan(
                plan_id=1,
                run_generation=0,
                observation_id=1,
                observation_anchor_ns=30,
                observation_latest_source_ns=20,
                inference_finished_ns=40,
                deadline_ns=100,
                chunk=object(),  # type: ignore[arg-type]
            )

    def test_capacity_formula_is_bounded_and_rejects_invalid_values(self) -> None:
        self.assertEqual(compute_max_buffered_plans(1.0, 10.0), 12)
        self.assertEqual(compute_max_buffered_plans(0.01, 1.0), 3)
        self.assertEqual(compute_max_buffered_plans(100.0, 100.0), 32)
        for value in (True, 0.0, -1.0, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    compute_max_buffered_plans(value, 10.0)


if __name__ == "__main__":
    unittest.main()
