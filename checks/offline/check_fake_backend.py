"""P6: deterministic fake backend end-to-end (architecture gate §66).

Proves obs -> backend -> chunk -> plan-ring runs end-to-end without importing
torch or ``dexmani_policy``. The fake is the reference implementation of the
three deployment Protocols and the backend-swap fixture (P12).
"""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.deployment.config import resolve_deployment_config
from dexmani_real.deployment.contracts import (
    ActionAdapter,
    InferenceContext,
    JointActionChunk,
    ObservationAdapter,
    PolicyBackend,
)
from dexmani_real.deployment.fake import (
    FakeActionAdapter,
    FakeObservationAdapter,
    FakePolicyBackend,
)
from dexmani_real.deployment.loader import (
    load_action_adapter,
    load_backend,
    load_observation_adapter,
)
from dexmani_real.deployment.observation import FrameWindow, ObservationBatch
from dexmani_real.shm.ring_buffer import SharedMemoryRingBuffer
from dexmani_real.utils.schema import MAX_POLICY_CHUNK_STEPS, POLICY_PLAN_DTYPE


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


def main() -> int:
    # ── architecture gate: no torch / no dexmani_policy imported ──
    assert "torch" not in sys.modules
    assert "dexmani_policy" not in sys.modules

    # ── build a causal observation ──
    arm_hist = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71],
        ]
    )
    hand_hist = np.zeros((2, 12))
    batch = ObservationBatch(
        observation_id=7,
        run_generation=3,
        anchor_monotonic_ns=1_000_000_000,
        arm_history=_window(arm_hist, start_ns=1_000_000),
        hand_history=_window(hand_hist, start_ns=2_000_000),
    )

    # ── obs -> backend -> chunk ──
    obs_adapter = FakeObservationAdapter()
    backend = FakePolicyBackend(horizon=4, offset_rad=0.01)
    action_adapter = FakeActionAdapter()

    assert isinstance(obs_adapter, ObservationAdapter)
    assert isinstance(backend, PolicyBackend)
    assert isinstance(action_adapter, ActionAdapter)

    backend.load()
    model_input = obs_adapter.encode(batch)
    # encode returns the latest valid frame of each window
    np.testing.assert_allclose(model_input["arm_qpos"], arm_hist[1])
    np.testing.assert_allclose(model_input["hand_qpos"], np.zeros(12))

    raw = backend.infer(model_input)
    assert raw["arm_qpos"].shape == (4, 7)
    assert raw["hand_qpos"].shape == (4, 12)
    # deterministic: same input -> byte-identical output
    raw2 = backend.infer(model_input)
    np.testing.assert_array_equal(raw["arm_qpos"], raw2["arm_qpos"])

    context = InferenceContext(
        run_generation=3,
        observation_id=7,
        observation_anchor_monotonic_ns=1_000_000_000,
        inference_started_monotonic_ns=1_000_000_100,
        inference_finished_monotonic_ns=1_000_001_000,
        step_dt_ns=100_000_000,  # 100 ms
    )
    chunk = action_adapter.decode(raw, context=context)
    assert isinstance(chunk, JointActionChunk)
    assert chunk.arm_qpos.shape == (4, 7)
    assert chunk.hand_qpos is not None and chunk.hand_qpos.shape == (4, 12)
    assert int(chunk.valid_mask.sum()) == 4
    expected_targets = np.asarray(
        context.inference_finished_monotonic_ns, dtype=np.uint64
    ) + np.arange(1, 5, dtype=np.uint64) * np.uint64(context.step_dt_ns)
    np.testing.assert_array_equal(chunk.target_monotonic_ns, expected_targets)
    # step 0 = last arm + one offset step, monotonically increasing by offset
    np.testing.assert_allclose(chunk.arm_qpos[0], arm_hist[1] + 0.01)
    np.testing.assert_allclose(chunk.arm_qpos[3], arm_hist[1] + 0.04)

    # ── chunk -> plan ring round-trip ──
    ring = SharedMemoryRingBuffer.create_or_replace(
        "check_fake_backend", dtype=POLICY_PLAN_DTYPE, maxlen=3
    )
    try:
        frame = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
        n = chunk.arm_qpos.shape[0]
        assert n <= MAX_POLICY_CHUNK_STEPS
        frame["plan_id"][0] = 1
        frame["run_generation"][0] = context.run_generation
        frame["observation_id"][0] = context.observation_id
        frame["observation_anchor_monotonic_ns"][0] = context.observation_anchor_monotonic_ns
        frame["inference_started_monotonic_ns"][0] = context.inference_started_monotonic_ns
        frame["inference_finished_monotonic_ns"][0] = context.inference_finished_monotonic_ns
        frame["num_steps"][0] = n
        frame["arm_present"][0] = 1
        frame["hand_present"][0] = 1
        frame["target_monotonic_ns"][0, :n] = chunk.target_monotonic_ns
        frame["arm_qpos"][0, :n] = chunk.arm_qpos
        frame["hand_qpos"][0, :n] = chunk.hand_qpos
        frame["valid_mask"][0, :n] = chunk.valid_mask
        ring.write(frame)

        read = ring.read_latest()
        assert read is not None
        assert int(read[0]["num_steps"][0]) == n
        assert int(read[0]["arm_present"][0]) == 1
        assert int(read[0]["hand_present"][0]) == 1
        np.testing.assert_array_equal(read[0]["arm_qpos"][0, :n], chunk.arm_qpos)
        np.testing.assert_array_equal(
            read[0]["target_monotonic_ns"][0, :n], chunk.target_monotonic_ns
        )
    finally:
        ring.close()
        ring.unlink()

    # ── loader: fake resolves from module:symbol and conforms ──
    resolved = resolve_deployment_config(
        data={
            "backend_target": "dexmani_real.deployment.fake:FakePolicyBackend",
            "observation_adapter_target": "dexmani_real.deployment.fake:FakeObservationAdapter",
            "action_adapter_target": "dexmani_real.deployment.fake:FakeActionAdapter",
        }
    )
    backend_loaded = load_backend(resolved.deployment.backend_target, config=resolved.deployment)
    assert isinstance(backend_loaded, PolicyBackend)
    obs_loaded = load_observation_adapter(resolved.deployment.observation_adapter_target)
    assert isinstance(obs_loaded, ObservationAdapter)
    act_loaded = load_action_adapter(
        resolved.deployment.action_adapter_target, config=resolved.deployment
    )
    assert isinstance(act_loaded, ActionAdapter)

    print("check_fake_backend: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
