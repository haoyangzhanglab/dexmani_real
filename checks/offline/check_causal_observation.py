"""P2: causal readers select the newest frame within the causal cut.

Locks the extracted ``shm.causal_reader`` invariants:
  - core: ``0 < source <= publish <= anchor``, newest first
  - arm/hand modality readers reuse the core on the real state dtypes
  - camera: the extra receive layer ``0 < source <= receive <= publish <= anchor``
  - tactile: in-payload ``publish_monotonic_ns`` is absent, so the ring's own
    publish timestamp is the fallback
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

import _bootstrap  # noqa: F401

from dexmani_real.shm.causal_reader import (
    read_arm_state_causal,
    read_camera_frame_causal,
    read_causal_structured_frame,
    read_hand_state_causal,
    read_hand_tactile_causal,
)
from dexmani_real.shm.ring_buffer import SharedMemoryRingBuffer
from dexmani_real.utils.schema import (
    ARM_JOINT_SHAPE,
    ARM_STATE_DTYPE,
    HAND_JOINT_SHAPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
)

from _fakes import make_arm_state_frame, make_hand_state_frame


def _structured_ring(name: str, dtype: np.dtype, maxlen: int) -> SharedMemoryRingBuffer:
    return SharedMemoryRingBuffer.create_or_replace(name, dtype, maxlen=maxlen)


def main() -> int:
    # ── Core invariant: in-payload publish present ──
    core = np.dtype(
        [("source_monotonic_ns", "<u8"), ("publish_monotonic_ns", "<u8"), ("v", "<f8")]
    )
    ring = _structured_ring("check_causal_core", core, maxlen=3)
    try:
        for source, publish, v in ((100, 200, 1.0), (200, 300, 2.0), (300, 400, 3.0)):
            f = np.zeros(1, dtype=core)
            f["source_monotonic_ns"][0] = source
            f["publish_monotonic_ns"][0] = publish
            f["v"][0] = v
            ring.write(f)

        sel = read_causal_structured_frame(ring, source_field="source_monotonic_ns", anchor_monotonic_ns=350)
        assert sel is not None and sel[0]["v"][0] == 2.0, "anchor 350 must select frame 2"
        sel = read_causal_structured_frame(ring, source_field="source_monotonic_ns", anchor_monotonic_ns=500)
        assert sel is not None and sel[0]["v"][0] == 3.0, "anchor 500 must select frame 3"
        assert (
            read_causal_structured_frame(ring, source_field="source_monotonic_ns", anchor_monotonic_ns=150) is None
        ), "no frame may precede anchor 150"
    finally:
        ring.close()
        ring.unlink()

    # ── Core invariant: publish absent -> ring publish fallback (tactile pattern) ──
    src_only = np.dtype([("source_monotonic_ns", "<u8"), ("v", "<f8")])
    ring = _structured_ring("check_causal_fallback", src_only, maxlen=2)
    try:
        f = np.zeros(1, dtype=src_only)
        f["source_monotonic_ns"][0] = 1_000_000_000  # far in the past relative to ring publish
        f["v"][0] = 9.0
        ring.write(f)
        sel = read_causal_structured_frame(
            ring, source_field="source_monotonic_ns", anchor_monotonic_ns=10**18
        )
        assert sel is not None and sel[0]["v"][0] == 9.0, "fallback publish must be usable"
        assert (
            read_causal_structured_frame(ring, source_field="source_monotonic_ns", anchor_monotonic_ns=1) is None
        ), "source in the future of anchor must be skipped"
    finally:
        ring.close()
        ring.unlink()

    # ── Arm / hand modality readers (real state dtypes) ──
    arm_ring = _structured_ring("check_causal_arm", ARM_STATE_DTYPE, maxlen=3)
    hand_ring = _structured_ring("check_causal_hand", HAND_STATE_DTYPE, maxlen=3)
    tactile_ring = _structured_ring("check_causal_tactile", HAND_TACTILE_DTYPE, maxlen=2)
    shared = SimpleNamespace(
        arm_state_ring=arm_ring, hand_state_ring=hand_ring, hand_tactile_ring=tactile_ring
    )
    try:
        for seed, source, publish in ((1.0, 1000, 1100), (2.0, 2000, 2100)):
            af = make_arm_state_frame(np.full(ARM_JOINT_SHAPE, seed))
            af["source_monotonic_ns"][0] = source
            af["publish_monotonic_ns"][0] = publish
            arm_ring.write(af)

            hf = make_hand_state_frame(np.full(HAND_JOINT_SHAPE, seed))
            hf["source_monotonic_ns"][0] = source
            hf["publish_monotonic_ns"][0] = publish
            hand_ring.write(hf)

        arm = read_arm_state_causal(shared, anchor_monotonic_ns=2500)
        assert arm is not None and float(arm["qpos"][0][0]) == 2.0, "arm causal must pick newest (2.0)"
        hand = read_hand_state_causal(shared, anchor_monotonic_ns=2500)
        assert hand is not None and float(hand["qpos"][0][0]) == 2.0, "hand causal must pick newest (2.0)"
        arm = read_arm_state_causal(shared, anchor_monotonic_ns=1500)
        assert arm is not None and float(arm["qpos"][0][0]) == 1.0, "anchor 1500 must pick only frame 1"
        assert read_arm_state_causal(shared, anchor_monotonic_ns=500) is None, "arm: no frame before 500"
        assert read_hand_state_causal(shared, anchor_monotonic_ns=500) is None, "hand: no frame before 500"
        # anchor=None falls back to the latest frame.
        assert read_arm_state_causal(shared) is not None
        assert read_hand_state_causal(shared) is not None

        # Tactile: latest fallback + causal None.
        tf = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
        tf["source_monotonic_ns"][0] = 1_000_000_000
        tactile_ring.write(tf)
        assert read_hand_tactile_causal(shared) is not None, "tactile latest must be readable"
        assert read_hand_tactile_causal(shared, anchor_monotonic_ns=1) is None, "tactile future source skipped"
    finally:
        arm_ring.close()
        arm_ring.unlink()
        hand_ring.close()
        hand_ring.unlink()
        tactile_ring.close()
        tactile_ring.unlink()

    # ── Camera: extra receive layer ──
    cam = _fake_camera_ring()
    frame = read_camera_frame_causal(SimpleNamespace(camera_ring=cam), anchor_monotonic_ns=2500)
    assert frame is not None and frame["frame_number"] == 2, "camera causal must pick frame 2"
    assert (
        read_camera_frame_causal(SimpleNamespace(camera_ring=cam), anchor_monotonic_ns=1000) is None
    ), "camera: receive > anchor must be skipped"

    print("check_causal_observation: PASS")
    return 0


def _fake_camera_ring():
    """Minimal camera-ring double: 3 frames with source/receive/publish layers.

    Frame 1 (seq 1): source=1000 receive=1200 publish=1300  -> valid, within 2500
    Frame 2 (seq 2): source=2000 receive=2200 publish=2300  -> valid, within 2500 (newest)
    Frame 3 (seq 3): source=3000 receive=3200 publish=3300  -> receive > anchor 2500, skipped
    """

    hdr = np.dtype(
        [
            ("source_monotonic_ns", "<u8"),
            ("receive_monotonic_ns", "<u8"),
            ("publish_monotonic_ns", "<u8"),
            ("pointcloud_valid", "<u1"),
            ("pc_num_points", "<i4"),
            ("frame_number", "<u8"),
            ("timestamp", "<f8"),
            ("capture_monotonic_s", "<f8"),
            ("camera_generation", "<i4"),
            ("clock_reset", "<u1"),
            ("duplicate", "<u1"),
            ("frame_gap", "<u4"),
            ("backlog_s", "<f8"),
            ("camera_health", "<i4"),
            ("pc_source_point_count", "<i4"),
            ("pc_valid_depth_ratio", "<f8"),
            ("pc_padding_count", "<i4"),
        ]
    )

    class _Cam:
        maxlen = 3

        def __init__(self) -> None:
            self._headers = []
            self._payloads = {}
            for seq, (src, rcv, pub) in enumerate(
                ((1000, 1200, 1300), (2000, 2200, 2300), (3000, 3200, 3300)), start=1
            ):
                h = np.zeros(1, dtype=hdr)
                h["source_monotonic_ns"][0] = src
                h["receive_monotonic_ns"][0] = rcv
                h["publish_monotonic_ns"][0] = pub
                h["frame_number"][0] = seq
                h["pointcloud_valid"][0] = 1
                h["pc_num_points"][0] = 10
                self._headers.append((h, pub, seq))
                self._payloads[seq] = {
                    "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                    "depth": np.zeros((1, 1), dtype=np.uint16),
                    "pointcloud": np.zeros((10, 3), dtype=np.float32),
                }

        def get_last_metadata(self, k: int):
            return self._headers[-k:]

        def read_sequence(self, seq: int, modalities=()):
            return self._payloads.get(seq)

        def read_latest(self):
            return None

    return _Cam()


if __name__ == "__main__":
    sys.exit(main())
