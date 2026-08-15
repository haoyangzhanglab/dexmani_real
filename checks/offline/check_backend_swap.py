"""P12: backend replacement is a config-only change (§100).

Locks two properties of the swap boundary:

  1. The deployment core (coordinator / worker / loader / SafetyGate /
     SharedStorage) never names a concrete backend or integration — so swapping
     the model never requires touching those files.
  2. Two distinct backends (``FakePolicyBackend`` vs ``ZeroTargetPolicyBackend``)
     both run the *same* ``encode -> infer -> decode -> publish`` path and are
     both consumed by the *same* coordinator adoption/scheduling helpers, with
     the only difference between their resolved configs being ``backend_target``.
"""

from __future__ import annotations

import inspect
import sys

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.deployment.config import resolve_deployment_config
from dexmani_real.deployment.contracts import (
    InferenceContext,
    JointActionChunk,
    PolicyBackend,
)
from dexmani_real.deployment.loader import (
    load_action_adapter,
    load_backend,
    load_observation_adapter,
)
from dexmani_real.deployment.observation import FrameWindow, ObservationBatch
from dexmani_real.shm.ring_buffer import SharedMemoryRingBuffer
from dexmani_real.utils.schema import POLICY_PLAN_DTYPE

# The core modules a backend swap must never touch (§100).
import dexmani_real.deployment.coordinator as coordinator_mod
import dexmani_real.deployment.worker as worker_mod
import dexmani_real.deployment.loader as loader_mod
import dexmani_real.policy.safety as safety_mod
import dexmani_real.shm.shared_storage as shared_storage_mod

_FORBIDDEN = ("integrations", "deployment.fake", "FakePolicyBackend", "ZeroTargetPolicyBackend")


def _window(values: np.ndarray, start_ns: int) -> FrameWindow:
    t = values.shape[0]
    seq = np.arange(1, t + 1, dtype=np.uint64)
    src = np.arange(start_ns, start_ns + t, dtype=np.uint64)
    return FrameWindow(
        values=values,
        source_sequence=seq,
        source_monotonic_ns=src,
        publish_monotonic_ns=src + 1,
        valid_mask=np.ones(t, dtype=np.uint8),
    )


def _to_plan_frame(chunk: JointActionChunk, context: InferenceContext, plan_id: int):
    """Serialise a chunk the way ``worker.publish_plan`` does (shared dtype)."""
    from dexmani_real.shm.shared_storage import new_frame

    n = chunk.arm_qpos.shape[0]
    frame = new_frame(POLICY_PLAN_DTYPE)
    frame["plan_id"][0] = np.uint64(plan_id)
    frame["run_generation"][0] = np.uint64(context.run_generation)
    frame["observation_id"][0] = np.uint64(context.observation_id)
    frame["observation_anchor_monotonic_ns"][0] = np.uint64(context.observation_anchor_monotonic_ns)
    frame["inference_started_monotonic_ns"][0] = np.uint64(context.inference_started_monotonic_ns)
    frame["inference_finished_monotonic_ns"][0] = np.uint64(context.inference_finished_monotonic_ns)
    frame["num_steps"][0] = np.uint32(n)
    frame["arm_present"][0] = 1
    frame["hand_present"][0] = 1 if chunk.hand_qpos is not None else 0
    frame["target_monotonic_ns"][0, :n] = chunk.target_monotonic_ns
    frame["arm_qpos"][0, :n] = chunk.arm_qpos
    if chunk.hand_qpos is not None:
        frame["hand_qpos"][0, :n] = chunk.hand_qpos
    frame["valid_mask"][0, :n] = chunk.valid_mask
    return frame


def main() -> int:
    # ── 1. static: core modules never name a concrete backend/integration ──
    for mod in (coordinator_mod, worker_mod, loader_mod, safety_mod, shared_storage_mod):
        src = inspect.getsource(mod)
        for token in _FORBIDDEN:
            assert token not in src, f"{mod.__name__} references {token!r}; swap must not touch core"

    # ── 2. two configs differing only in backend_target ──
    base = {
        "observation_adapter_target": "dexmani_real.deployment.fake:FakeObservationAdapter",
        "action_adapter_target": "dexmani_real.deployment.fake:FakeActionAdapter",
    }
    a = resolve_deployment_config(
        data={"backend_target": "dexmani_real.deployment.fake:FakePolicyBackend", **base}
    )
    b = resolve_deployment_config(
        data={"backend_target": "_swap_fixtures:ZeroTargetPolicyBackend", **base}
    )
    assert a.sha256 != b.sha256, "distinct backends must yield distinct config identities"
    assert a.deployment.observation_adapter_target == b.deployment.observation_adapter_target

    # ── 3. both load through the same loader and conform ──
    backend_a = load_backend(a.deployment.backend_target, config=a.deployment)
    backend_b = load_backend(b.deployment.backend_target, config=b.deployment)
    assert isinstance(backend_a, PolicyBackend)
    assert isinstance(backend_b, PolicyBackend)
    assert backend_a is not backend_b

    obs_adapter = load_observation_adapter(a.deployment.observation_adapter_target, config=a.deployment)
    act_adapter = load_action_adapter(a.deployment.action_adapter_target, config=a.deployment)

    # ── 4. same observation -> two distinct, valid chunks ──
    arm_hist = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    batch = ObservationBatch(
        observation_id=1,
        run_generation=4,
        anchor_monotonic_ns=1_000_000_000,
        arm_history=_window(arm_hist, start_ns=1_000_000),
    )
    ctx = InferenceContext(
        run_generation=4,
        observation_id=1,
        observation_anchor_monotonic_ns=1_000_000_000,
        inference_started_monotonic_ns=1_000_000_100,
        inference_finished_monotonic_ns=1_000_001_000,
        step_dt_ns=100_000_000,
    )

    def _run(backend):
        backend.load()
        model_input = obs_adapter.encode(batch)
        raw = backend.infer(model_input)
        chunk = act_adapter.decode(raw, context=ctx)
        assert isinstance(chunk, JointActionChunk)
        return chunk

    chunk_a = _run(backend_a)
    chunk_b = _run(backend_b)
    # Genuinely different models: hold-at-current vs hold-at-zero. Horizons may
    # differ (8 vs 4), so compare the first commanded step, which always differs.
    assert not np.allclose(chunk_a.arm_qpos[0], chunk_b.arm_qpos[0])
    assert chunk_a.hand_qpos is None and chunk_b.hand_qpos is None

    # ── 5. both chunks publish to the same ring and are adopted identically ──
    ring = SharedMemoryRingBuffer.create_or_replace("check_backend_swap", dtype=POLICY_PLAN_DTYPE, maxlen=3)
    try:
        ring.write(_to_plan_frame(chunk_a, ctx, plan_id=1))
        ring.write(_to_plan_frame(chunk_b, ctx, plan_id=2))
        latest = ring.read_latest()[0][0]
        assert int(latest["plan_id"]) == 2, "latest-wins: second backend's plan is current"

        # The coordinator's adoption/scheduling helpers consume either backend
        # identically (no backend knowledge — only the shared plan dtype).
        for plan_id, chunk in ((1, chunk_a), (2, chunk_b)):
            rec = _to_plan_frame(chunk, ctx, plan_id=plan_id)[0]
            ok, reason = coordinator_mod._adoptable(
                rec,
                current_generation=4,
                last_observation_id=0,
                now_ns=ctx.inference_finished_monotonic_ns + 1,
                max_plan_age_ns=1_000_000_000,
                max_observation_age_ns=1_000_000_000,
            )
            assert ok, f"plan {plan_id} must be adoptable, got {reason}"
            n = int(rec["num_steps"])
            selected, _ = coordinator_mod._select_due_step(
                np.asarray(rec["target_monotonic_ns"][:n], dtype=np.uint64),
                np.asarray(rec["valid_mask"][:n], dtype=np.uint8),
                n,
                0,
                ctx.inference_finished_monotonic_ns + 4 * np.uint64(ctx.step_dt_ns),
            )
            assert selected is not None, f"plan {plan_id} must schedule a due endpoint"
    finally:
        ring.close()
        ring.unlink()

    backend_a.close()
    backend_b.close()

    print("check_backend_swap: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
