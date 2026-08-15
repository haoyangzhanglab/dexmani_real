"""P7: inference worker's plan publication and generation cancellation.

Locks the two correctness guarantees of the proposal boundary:

  - ``publish_plan`` writes the latest-wins plan ring only when the plan's
    ``run_generation`` still matches ``shared.run_generation`` at publish time;
    a generation advance drops the in-flight plan without relabeling (§70).
  - A bad model result (NaN/shape/ordering) surfaces as a ``ValueError`` at
    decode time — the worker's drop-not-crash abort signal (§80.2) — and an
    over-capacity chunk fails closed (§61).
"""

from __future__ import annotations

import sys
import time

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import make_arm_state_frame

from dexmani_real.deployment.contracts import InferenceContext, JointActionChunk
from dexmani_real.deployment.fake import FakeActionAdapter
from dexmani_real.deployment.worker import _read_state_history, publish_plan
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.utils.schema import MAX_POLICY_CHUNK_STEPS


def _context(run_generation: int, observation_id: int = 1) -> InferenceContext:
    return InferenceContext(
        run_generation=run_generation,
        observation_id=observation_id,
        observation_anchor_monotonic_ns=1_000_000_000,
        inference_started_monotonic_ns=1_000_000_100,
        inference_finished_monotonic_ns=1_000_001_000,
        step_dt_ns=62_500_000,
    )


def _chunk(n: int) -> JointActionChunk:
    arm = np.zeros((n, 7), dtype=np.float64)
    arm[:] = np.linspace(0.1, 0.1 + 0.001 * (n - 1), n)[:, None]
    hand = np.zeros((n, 12), dtype=np.float64)
    target = np.asarray(1_000_001_000, dtype=np.uint64) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(62_500_000)
    return JointActionChunk(
        arm_qpos=arm,
        hand_qpos=hand,
        target_monotonic_ns=target,
        valid_mask=np.ones(n, dtype=np.uint8),
    )


def main() -> int:
    shared = SharedStorage.create(prefix="check_inference_generation")
    try:
        shared.run_generation.value = 5

        # ── publish when generation matches ──
        ctx = _context(run_generation=5)
        chunk = _chunk(4)
        assert publish_plan(shared, plan_id=1, context=ctx, chunk=chunk) is True
        latest = shared.policy_plan_ring.read_latest()
        assert latest is not None
        assert int(latest[0]["plan_id"][0]) == 1
        assert int(latest[0]["run_generation"][0]) == 5
        assert int(latest[0]["num_steps"][0]) == 4
        assert int(latest[0]["arm_present"][0]) == 1
        assert int(latest[0]["hand_present"][0]) == 1

        # ── generation advance -> drop, no relabel, no write ──
        shared.run_generation.value = 6
        assert publish_plan(shared, plan_id=2, context=_context(run_generation=5), chunk=chunk) is False
        latest = shared.policy_plan_ring.read_latest()
        assert int(latest[0]["plan_id"][0]) == 1, "dropped plan must not overwrite the ring"

        # matching the new generation publishes again
        assert publish_plan(shared, plan_id=3, context=_context(run_generation=6), chunk=_chunk(2)) is True
        latest = shared.policy_plan_ring.read_latest()
        assert int(latest[0]["plan_id"][0]) == 3

        # ── over-capacity chunk fails closed (§61) ──
        try:
            publish_plan(shared, plan_id=4, context=_context(run_generation=6), chunk=_chunk(MAX_POLICY_CHUNK_STEPS + 1))
        except ValueError:
            pass
        else:
            raise AssertionError("over-capacity chunk must raise ValueError")

        # ── NaN model output -> decode raises (worker drops, not crash) ──
        nan_raw = {"arm_qpos": np.full((4, 7), np.nan), "hand_qpos": None}
        try:
            FakeActionAdapter().decode(nan_raw, context=_context(run_generation=6))
        except ValueError:
            pass
        else:
            raise AssertionError("NaN chunk must raise ValueError at decode")

    finally:
        shared.close()

    # ── causal history window: a future-publish frame is excluded ──
    shared2 = SharedStorage.create(prefix="check_inference_generation_hist")
    try:
        mid = np.full(7, 0.3, dtype=np.float64)
        good = make_arm_state_frame(mid)
        future = make_arm_state_frame(mid)
        now_ns = time.monotonic_ns()
        future["publish_monotonic_ns"][0] = now_ns + 10_000_000_000  # future publish
        shared2.arm_state_ring.write(future)
        shared2.arm_state_ring.write(good)

        window = _read_state_history(
            shared2.arm_state_ring,
            horizon=4,
            anchor_ns=now_ns,
            values_field="qpos",
        )
        assert window is not None
        assert window.values.shape == (1, 7), "only the causal frame may be selected"
        np.testing.assert_allclose(window.values[0], mid)
    finally:
        shared2.close()

    print("check_inference_generation: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
