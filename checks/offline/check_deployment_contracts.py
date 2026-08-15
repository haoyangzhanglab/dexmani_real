"""P1: deployment contracts and data types are frozen, validated, read-only."""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401

from dexmani_real.deployment.contracts import (
    ActionAdapter,
    InferenceContext,
    JointActionChunk,
    ObservationAdapter,
    PolicyBackend,
)
from dexmani_real.deployment.observation import CameraWindow, FrameWindow, ObservationBatch
from dexmani_real.utils.schema import MAX_POLICY_CHUNK_STEPS


def _frame_window(t: int, feat: tuple[int, ...]) -> FrameWindow:
    return FrameWindow(
        values=np.zeros((t, *feat), dtype=np.float64),
        source_sequence=np.arange(t, dtype=np.uint64),
        source_monotonic_ns=np.arange(t, dtype=np.uint64) + 1,
        publish_monotonic_ns=np.arange(t, dtype=np.uint64) + 2,
        valid_mask=np.ones(t, dtype=np.uint8),
    )


def main() -> int:
    n = 4
    arm = np.zeros((n, 7), dtype=np.float64)
    hand = np.zeros((n, 12), dtype=np.float64)
    target = np.arange(n, dtype=np.uint64) * 10 + 1000
    mask = np.ones(n, dtype=np.uint8)

    # ── JointActionChunk: valid, read-only, hand-optional ──
    chunk = JointActionChunk(arm_qpos=arm, hand_qpos=hand, target_monotonic_ns=target, valid_mask=mask)
    assert chunk.arm_qpos.shape == (n, 7)
    assert chunk.hand_qpos.shape == (n, 12)
    try:
        chunk.arm_qpos[0, 0] = 1.0
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("JointActionChunk arrays must be read-only")

    arm_only = JointActionChunk(arm_qpos=arm, hand_qpos=None, target_monotonic_ns=target, valid_mask=mask)
    assert arm_only.hand_qpos is None

    # ── JointActionChunk: rejections (each must raise ValueError) ──
    bad_chunks = (
        dict(  # empty chunk
            arm_qpos=np.zeros((0, 7)),
            hand_qpos=None,
            target_monotonic_ns=np.zeros(0, dtype=np.uint64),
            valid_mask=np.zeros(0, dtype=np.uint8),
        ),
        dict(  # wrong arm feature dim
            arm_qpos=np.zeros((n, 6)),
            hand_qpos=None,
            target_monotonic_ns=target,
            valid_mask=mask,
        ),
        dict(  # wrong hand shape (does not match N)
            arm_qpos=arm,
            hand_qpos=np.zeros((n - 1, 12)),
            target_monotonic_ns=target,
            valid_mask=mask,
        ),
        dict(  # non-increasing target timestamps (uint64, must not underflow past the check)
            arm_qpos=arm,
            hand_qpos=None,
            target_monotonic_ns=np.array([100, 99, 98, 97], dtype=np.uint64),
            valid_mask=mask,
        ),
        dict(  # valid_mask outside {0, 1}
            arm_qpos=arm,
            hand_qpos=None,
            target_monotonic_ns=target,
            valid_mask=np.full(n, 7, dtype=np.uint8),
        ),
        dict(  # NaN arm target
            arm_qpos=np.full((n, 7), np.nan),
            hand_qpos=None,
            target_monotonic_ns=target,
            valid_mask=mask,
        ),
        dict(  # over transport capacity (MAX_POLICY_CHUNK_STEPS) — must fail, never truncate
            arm_qpos=np.zeros((MAX_POLICY_CHUNK_STEPS + 1, 7)),
            hand_qpos=None,
            target_monotonic_ns=np.arange(MAX_POLICY_CHUNK_STEPS + 1, dtype=np.uint64) * 10 + 1000,
            valid_mask=np.ones(MAX_POLICY_CHUNK_STEPS + 1, dtype=np.uint8),
        ),
    )
    for bad in bad_chunks:
        try:
            JointActionChunk(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid JointActionChunk must raise ValueError")

    # ── InferenceContext ──
    ctx = InferenceContext(
        run_generation=3,
        observation_id=7,
        observation_anchor_monotonic_ns=1000,
        inference_started_monotonic_ns=2000,
        inference_finished_monotonic_ns=3000,
        step_dt_ns=62_500_000,
    )
    assert ctx.inference_finished_monotonic_ns > ctx.inference_started_monotonic_ns
    try:
        InferenceContext(
            run_generation=0,
            observation_id=0,
            observation_anchor_monotonic_ns=1000,
            inference_started_monotonic_ns=3000,  # finish precedes start
            inference_finished_monotonic_ns=2000,
            step_dt_ns=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("inverse-time InferenceContext must raise ValueError")

    # ── ObservationBatch (nested per-modality windows) ──
    arm_win = _frame_window(3, (7,))
    hand_win = _frame_window(3, (12,))
    batch = ObservationBatch(
        observation_id=7, run_generation=3, anchor_monotonic_ns=10_000,
        arm_history=arm_win, hand_history=hand_win,
    )
    assert batch.arm_history.values.shape == (3, 7)
    assert batch.hand_history.values.shape == (3, 12)
    assert batch.tactile_history is None
    assert batch.camera_history is None

    # ── CameraWindow: receive layer + at least one sub-modality ──
    cam = CameraWindow(
        rgb=np.zeros((3, 64, 64, 3), dtype=np.uint8),
        source_sequence=np.arange(3, dtype=np.uint64),
        source_monotonic_ns=np.arange(3, dtype=np.uint64) + 1,
        receive_monotonic_ns=np.arange(3, dtype=np.uint64) + 2,
        publish_monotonic_ns=np.arange(3, dtype=np.uint64) + 3,
        valid_mask=np.ones(3, dtype=np.uint8),
    )
    assert cam.rgb.shape == (3, 64, 64, 3)
    try:
        CameraWindow()  # no sub-modality present
    except ValueError:
        pass
    else:
        raise AssertionError("empty CameraWindow must raise ValueError")

    # ── Protocols are runtime-checkable ──
    class _FakeBackend:
        def load(self) -> None: ...
        def reset(self, *, run_generation: int) -> None: ...
        def infer(self, model_input): return model_input
        def close(self) -> None: ...

    class _FakeObsAdapter:
        def encode(self, observation): return observation

    class _FakeActAdapter:
        def decode(self, raw_output, *, context): return raw_output

    assert isinstance(_FakeBackend(), PolicyBackend)
    assert isinstance(_FakeObsAdapter(), ObservationAdapter)
    assert isinstance(_FakeActAdapter(), ActionAdapter)

    print("check_deployment_contracts: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
