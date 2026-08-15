"""P5: POLICY_PLAN_DTYPE, policy_plan_ring, and the inference identity.

Locks the fixed plan payload shape (§60), the latest-wins ring round-trip, and
the six-touchpoint "inference" identity sync so a new worker is consistently
named across schema, SharedStorage, and the safety heartbeat/readiness config.
"""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config.defaults import SafetyParams
from dexmani_real.shm.ring_buffer import SharedMemoryRingBuffer
from dexmani_real.shm.shared_storage import (
    HEARTBEAT_FIELDS,
    READY_FIELDS,
    SharedStorage,
)
from dexmani_real.utils.schema import MAX_POLICY_CHUNK_STEPS, POLICY_PLAN_DTYPE


def main() -> int:
    # ── dtype fields and shapes ──
    names = set(POLICY_PLAN_DTYPE.names or ())
    assert {
        "plan_id",
        "run_generation",
        "observation_id",
        "observation_anchor_monotonic_ns",
        "inference_started_monotonic_ns",
        "inference_finished_monotonic_ns",
        "num_steps",
        "arm_present",
        "hand_present",
        "target_monotonic_ns",
        "arm_qpos",
        "hand_qpos",
        "valid_mask",
    } <= names, "POLICY_PLAN_DTYPE is missing a required field"

    assert MAX_POLICY_CHUNK_STEPS == 32
    assert POLICY_PLAN_DTYPE["target_monotonic_ns"].shape == (MAX_POLICY_CHUNK_STEPS,)
    assert POLICY_PLAN_DTYPE["arm_qpos"].shape == (MAX_POLICY_CHUNK_STEPS, 7)
    assert POLICY_PLAN_DTYPE["hand_qpos"].shape == (MAX_POLICY_CHUNK_STEPS, 12)
    assert POLICY_PLAN_DTYPE["valid_mask"].shape == (MAX_POLICY_CHUNK_STEPS,)
    assert POLICY_PLAN_DTYPE["num_steps"].itemsize == 4, "num_steps must be u4"

    # ── ring round-trip + latest-wins ──
    ring = SharedMemoryRingBuffer.create_or_replace(
        "check_policy_plan", dtype=POLICY_PLAN_DTYPE, maxlen=3
    )
    try:
        for plan_id in (1, 2, 3):
            frame = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
            frame["plan_id"][0] = plan_id
            frame["run_generation"][0] = 7
            frame["observation_id"][0] = 100 + plan_id
            frame["num_steps"][0] = 2
            frame["arm_present"][0] = 1
            frame["hand_present"][0] = 0
            frame["target_monotonic_ns"][0, :2] = [1000, 2000]
            frame["arm_qpos"][0, :2] = np.arange(14).reshape(2, 7)
            frame["valid_mask"][0, :2] = [1, 0]
            ring.write(frame)

        latest = ring.read_latest()
        assert latest is not None
        assert int(latest[0]["plan_id"][0]) == 3, "latest-wins must expose the newest plan"

        history = ring.get_last_k(3)
        assert len(history) == 3, "ring must retain its full capacity"
        ids = [int(data["plan_id"][0]) for data, _ts, _seq in history]
        assert ids == [1, 2, 3], "get_last_k must return oldest-first"
        assert int(history[-1][0]["num_steps"][0]) == 2

        # A full short ring still returns verified frames, no torn reads.
        frame = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
        frame["plan_id"][0] = 4
        ring.write(frame)
        assert int(ring.read_latest()[0]["plan_id"][0]) == 4
        assert [int(d["plan_id"][0]) for d, _t, _s in ring.get_last_k(3)] == [2, 3, 4]
    finally:
        ring.close()
        ring.unlink()

    # ── inference identity: six-touchpoint sync ──
    assert HEARTBEAT_FIELDS[-1] == "inference"
    assert READY_FIELDS[-1] == "inference"
    safety = SafetyParams()  # raises if a subsystem is missing from the dicts
    assert "inference" in safety.heartbeat_timeouts
    assert "inference" in safety.readiness_timeouts_s
    assert safety.heartbeat_timeouts["inference"] == 5.0
    assert safety.readiness_timeouts_s["inference"] == 120.0

    # SharedStorage allocates the ring and the identity slots.
    shared = SharedStorage.create(prefix="check_policy_plan_storage")
    try:
        assert shared.policy_plan_ring is not None
        assert shared.policy_plan_ring.maxlen == 3
        shared.set_ready("inference")
        assert shared.is_ready("inference")
        shared.set_heartbeat("inference", 1.0)
        assert shared.get_heartbeat("inference") == 1.0
    finally:
        shared.close()

    print("check_policy_plan_dtype: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
