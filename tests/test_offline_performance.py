from __future__ import annotations

import gc
import time
import tracemalloc
import uuid

import numpy as np

from dexmani_real.policy.action_protocol import SafeCommandPublisher
from dexmani_real.policy.observation import CausalFrame, SnapshotBuilder
from dexmani_real.policy.runtime import ActionCandidate, ModalitySpec, ObservationSpec
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig


def _candidate(now_ns: int) -> ActionCandidate:
    return ActionCandidate(
        observation_id=1,
        session_generation=1,
        policy_epoch=1,
        action_id=1,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=now_ns + 100_000_000,
        valid_until_monotonic_ns=now_ns + 200_000_000,
        arm_qpos=np.zeros(7),
        chunk_id=1,
    )


def test_snapshot_and_commit_10000_ticks_have_bounded_python_memory() -> None:
    """Catch accidental per-tick retention without imposing an RSS assertion."""
    prefix = f"perf_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix=prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(2, 3, 3),
            camera_depth_shape=(2, 3),
            camera_pc_shape=(4, 6),
        ),
    )
    spec = ObservationSpec((ModalitySpec("arm_qpos", (7,), "float64"),))
    builder = SnapshotBuilder(spec, session_generation=1)
    source_ns = time.monotonic_ns()
    frames = {"arm_qpos": [CausalFrame(np.zeros(7), 1, source_ns, source_ns + 1)]}
    publisher = SafeCommandPublisher(shared)
    candidate = _candidate(source_ns)
    anchor_ns = source_ns + 2
    try:
        for _ in range(200):
            builder.build(anchor_monotonic_ns=anchor_ns, frames=frames)
            publisher.commit(candidate)
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for _ in range(10_000):
            snapshot = builder.build(anchor_monotonic_ns=anchor_ns, frames=frames)
            publisher.commit(candidate)
        del snapshot
        gc.collect()
        after = tracemalloc.take_snapshot()
        net_bytes = sum(stat.size_diff for stat in after.compare_to(before, "lineno"))
        assert net_bytes < 1024 * 1024
    finally:
        tracemalloc.stop()
        shared.close()


def test_snapshot_and_commit_loose_shared_runner_p99_budget() -> None:
    """A loose CI smoke check; fixed-host regression measurements remain separate."""
    prefix = f"perf_p99_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix=prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(2, 3, 3),
            camera_depth_shape=(2, 3),
            camera_pc_shape=(4, 6),
        ),
    )
    spec = ObservationSpec((ModalitySpec("arm_qpos", (7,), "float64"),))
    builder = SnapshotBuilder(spec, session_generation=1)
    source_ns = time.monotonic_ns()
    frames = {"arm_qpos": [CausalFrame(np.zeros(7), 1, source_ns, source_ns + 1)]}
    publisher = SafeCommandPublisher(shared)
    candidate = _candidate(source_ns)
    anchor_ns = source_ns + 2
    durations_ms: list[float] = []
    try:
        for _ in range(100):
            builder.build(anchor_monotonic_ns=anchor_ns, frames=frames)
            publisher.commit(candidate)
        for _ in range(1_000):
            started_ns = time.perf_counter_ns()
            builder.build(anchor_monotonic_ns=anchor_ns, frames=frames)
            publisher.commit(candidate)
            durations_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
        assert float(np.percentile(durations_ms, 99)) < 12.5
    finally:
        shared.close()
