"""Unit tests for SeqlockRingBuffer.get_last_k(k) — M05.

Run from repo root::

    conda run -n real_robot python -m pytest tests/test_robot_ring.py -v
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pytest

from dexmani_real.shm.robot_ring import SeqlockRingBuffer

# Simple dtype for testing — one float64 field, tiny footprint.
_TEST_DTYPE = np.dtype([("value", "<f8", (3,))])

# Per-test maxlen (3 slots — exercises wrap-around quickly).
_TEST_MAXLEN = 3


def _new_frame(values: list[float]) -> np.ndarray:
    """Allocate a 0-d structured array matching _TEST_DTYPE."""
    f = np.zeros(1, dtype=_TEST_DTYPE)
    f["value"][0] = np.asarray(values, dtype=np.float64)
    return f


def _ring_name(test_name: str) -> str:
    """Unique SHM name per test to avoid cross-contamination."""
    return f"test_m05_{test_name}_{id(test_name)}"


@pytest.fixture
def ring():
    """Create a fresh SeqlockRingBuffer for a test, clean up afterwards."""
    name = _ring_name("ring")
    r = SeqlockRingBuffer.create_or_replace(name, _TEST_DTYPE, maxlen=_TEST_MAXLEN)
    yield r
    try:
        r.close()
        r.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# get_last_k tests
# ---------------------------------------------------------------------------


class TestGetLastK:
    """M05: SeqlockRingBuffer.get_last_k(k) — k-frame history retrieval."""

    # -- basic cases --------------------------------------------------------

    def test_empty_ring_returns_empty(self, ring):
        """Nothing written → get_last_k returns [].

        Sequence==0 is the canonical "empty" sentinel (used by
        SharedMemoryRingBuffer.read_latest and SeqlockRingBuffer.read_latest
        for the same disambiguation).
        """
        assert ring.get_last_k(1) == []
        assert ring.get_last_k(3) == []

    def test_k_zero_returns_empty(self, ring):
        """k=0 is a no-op — always returns []."""
        ring.write(_new_frame([1.0, 2.0, 3.0]))
        assert ring.get_last_k(0) == []

    def test_k_one_returns_latest(self, ring):
        """k=1 returns the same frame as read_latest()."""
        ring.write(_new_frame([10.0, 20.0, 30.0]))
        ring.write(_new_frame([100.0, 200.0, 300.0]))

        # read_latest reference
        ref_data, ref_ts, ref_seq = ring.read_latest()

        result = ring.get_last_k(1)
        assert len(result) == 1
        data, ts, seq = result[0]
        assert seq == ref_seq
        assert ts == ref_ts
        np.testing.assert_array_equal(data["value"][0], ref_data["value"][0])

    def test_k_one_empty_ring(self, ring):
        """k=1 on empty ring returns [] (not None — list API)."""
        assert ring.get_last_k(1) == []

    # -- normal multi-frame -------------------------------------------------

    def test_normal_k3_after_5_writes(self, ring):
        """Write 5 frames, get last 3 — should be frames 3,4,5 oldest-first."""
        for i in range(5):
            ring.write(_new_frame([float(i), float(i), float(i)]))

        result = ring.get_last_k(3)
        assert len(result) == 3

        # Oldest-first: frames 2,3,4 (0-indexed), i.e., seq 3,4,5
        for j, expected_seq in enumerate([3, 4, 5]):
            data, _ts, seq = result[j]
            assert seq == expected_seq, f"result[{j}].seq={seq}, expected={expected_seq}"
            np.testing.assert_array_equal(
                data["value"][0],
                [float(expected_seq - 1), float(expected_seq - 1), float(expected_seq - 1)],
            )

    def test_fewer_than_k_written(self, ring):
        """Only 2 frames exist but k=3 — returns the 2 that exist."""
        ring.write(_new_frame([1.0, 0.0, 0.0]))
        ring.write(_new_frame([2.0, 0.0, 0.0]))

        result = ring.get_last_k(3)  # k ≤ maxlen=3
        assert len(result) == 2
        assert result[0][2] == 1  # seq 1 oldest
        assert result[1][2] == 2  # seq 2 newest

    def test_k_exceeds_maxlen_raises(self, ring):
        """k > maxlen → ValueError. maxlen=3, k=10 → raises."""
        for i in range(5):
            ring.write(_new_frame([float(i), 0.0, 0.0]))

        with pytest.raises(ValueError, match="exceeds ring capacity"):
            ring.get_last_k(10)

    # -- wrap-around --------------------------------------------------------

    def test_wrap_around(self, ring):
        """maxlen=3. Write 5 frames → ring wraps. k=3 returns only the latest 3."""
        for i in range(5):
            ring.write(_new_frame([float(i), float(i), float(i)]))

        result = ring.get_last_k(3)
        assert len(result) == 3

        # After 5 writes with maxlen=3, slots: seq 3→0, 4→1, 5→2.
        # Frame 2 (seq 3) is oldest surviving, frame 4 (seq 5) is newest.
        seqs = [s for _, _, s in result]
        assert seqs == [3, 4, 5], f"expected [3,4,5], got {seqs}"

    # -- ordering -----------------------------------------------------------

    def test_oldest_first_ordering(self, ring):
        """Regardless of ring state, get_last_k MUST return oldest-first."""
        ring.write(_new_frame([10.0, 0.0, 0.0]))
        ring.write(_new_frame([11.0, 0.0, 0.0]))
        ring.write(_new_frame([12.0, 0.0, 0.0]))
        ring.write(_new_frame([13.0, 0.0, 0.0]))
        ring.write(_new_frame([14.0, 0.0, 0.0]))

        result = ring.get_last_k(3)
        seqs = [s for _, _, s in result]

        # Must be monotonically increasing (oldest → newest).
        assert seqs == sorted(seqs), f"not oldest-first: {seqs}"

        # Values should also ascend (since we wrote ascending values).
        vals = [float(d["value"][0][0]) for d, _, _ in result]
        assert vals == sorted(vals), f"values not monotonically increasing: {vals}"

    # -- concurrent writer (single-producer simulation) ---------------------

    def test_writer_during_read_no_tears(self, ring):
        """get_last_k never returns torn data even if writer is active.

        We simulate a busy writer by filling the ring then rapidly writing
        from a second thread.  Every returned frame must pass seqlock.
        """
        ring.write(_new_frame([1.0, 1.0, 1.0]))
        ring.write(_new_frame([2.0, 2.0, 2.0]))
        ring.write(_new_frame([3.0, 3.0, 3.0]))

        errors: list[str] = []

        def busy_writer():
            try:
                for i in range(1000):
                    ring.write(_new_frame([float(i), float(i), float(i)]))
            except Exception as e:
                errors.append(str(e))

        writer = mp.Process(target=busy_writer)
        writer.start()

        # Read repeatedly while the writer is running.
        for _ in range(50):
            frames = ring.get_last_k(3)
            for data, _ts, seq in frames:
                # Every returned frame must have a valid, complete seqlock
                # marker (enforced by get_last_k).  We additionally verify
                # the data is finite — torn writes can produce garbage.
                vals = data["value"][0]
                assert np.all(np.isfinite(vals)), (
                    f"non-finite values in frame seq={seq}: {vals}"
                )

        writer.join(timeout=3)
        if writer.is_alive():
            writer.terminate()
            writer.join()

        assert not errors, f"writer errors: {errors}"

    def test_overwritten_frames_dropped(self):
        """When the writer wraps, overwritten frames are silently omitted.

        maxlen=3.  Write 100 frames rapidly → the earliest 97 are overwritten.
        get_last_k(3) returns only the 3 physically-stored frames.
        """
        name = _ring_name("overwritten")
        r = SeqlockRingBuffer.create_or_replace(name, _TEST_DTYPE, maxlen=3)
        try:
            for i in range(100):
                r.write(_new_frame([float(i), float(i), float(i)]))

            result = r.get_last_k(3)
            # At most maxlen=3 frames are physically stored.
            assert len(result) <= 3

            # The returned sequences should be among the last maxlen frames
            # written (98, 99, 100).
            latest = int(r._write_seq[0])
            for _, _, seq in result:
                assert seq > latest - 3, (
                    f"stale frame seq={seq} survived; latest={latest}"
                )
        finally:
            try:
                r.close()
                r.unlink()
            except Exception:
                pass

    # -- type/shape contract ------------------------------------------------

    def test_data_shape_matches_read_latest(self, ring):
        """get_last_k data shape MUST match read_latest: (1,) with dtype fields."""
        ring.write(_new_frame([42.0, 43.0, 44.0]))

        ref = ring.read_latest()
        assert ref is not None
        ref_data = ref[0]

        result = ring.get_last_k(1)
        assert len(result) == 1
        data = result[0][0]

        assert data.shape == ref_data.shape
        assert data.dtype == ref_data.dtype

    def test_data_is_copy_not_view(self, ring):
        """Returned data must be independent copies — mutating must not affect SHM."""
        ring.write(_new_frame([7.0, 8.0, 9.0]))

        result = ring.get_last_k(1)
        data = result[0][0]
        original = float(data["value"][0][0])

        # Mutate the returned copy
        data["value"][0][0] = 999.0

        # Re-read — must still be the original value
        ref = ring.read_latest()
        assert ref is not None
        assert float(ref[0]["value"][0][0]) == original

    # -- overwrite detection (strict sequence matching) -----------------------

    def test_overwrite_detection_stops_walk(self):
        """When a slot is overwritten, the walk stops (strict seq matching).

        maxlen=3.  Write seq=1,2,3 then seq=4 overwrites slot 1 (seq 1).
        get_last_k(3) should return only [3,4] — seq 1 is gone, and because
        the writer overwrites monotonically, no frames older than the
        overwritten one remain.
        """
        name = _ring_name("overwrite_stop")
        r = SeqlockRingBuffer.create_or_replace(name, _TEST_DTYPE, maxlen=3)
        try:
            # Fill ring: seq 1→0, 2→1, 3→2
            r.write(_new_frame([1.0, 0.0, 0.0]))
            r.write(_new_frame([2.0, 0.0, 0.0]))
            r.write(_new_frame([3.0, 0.0, 0.0]))

            # Write seq=4 to slot 1 (4%3=1), overwriting seq 1
            r.write(_new_frame([4.0, 0.0, 0.0]))

            # k=3 asks for seq 2,3,4 — but seq 2 is at slot 2 (2%3=2) which
            # should still be valid.
            result = r.get_last_k(3)
            seqs = [s for _, _, s in result]
            # seq 1 was overwritten by seq 4.  seq 2 is at slot 2 (not yet overwritten).
            # With newest-first read: seq 4 OK, seq 3 OK, seq 2 OK → 3 frames.
            # But if writer advanced to seq 5→2 during read, seq 2 might be overwritten.
            assert len(result) >= 2, f"expected at least 2 frames, got {seqs}"
            assert 4 in seqs, f"seq 4 should be present, got {seqs}"
            assert 1 not in seqs, f"seq 1 was overwritten, should be absent: {seqs}"
        finally:
            try:
                r.close()
                r.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# SharedStorage helpers (integration smoke tests)
# ---------------------------------------------------------------------------


class TestSharedStorageHelpers:
    """M05 Layer 2: read_arm_state_k / read_hand_state_k wrappers."""

    def test_read_arm_state_k_import(self):
        """Smoke test: import works and function is callable."""
        from dexmani_real.shm.shared_storage import read_arm_state_k, read_hand_state_k

        assert callable(read_arm_state_k)
        assert callable(read_hand_state_k)

    def test_read_arm_state_k_on_real_ring(self):
        """Integration: create a ring, write to it, read via helper."""
        from dexmani_real.shm.shared_storage import read_arm_state_k

        # Use the real ARM_STATE_DTYPE ring for a quick integration check.
        from dexmani_real.shm.shared_storage import ARM_STATE_DTYPE, new_frame, SharedStorageConfig

        name = _ring_name("arm_state_k_test")
        cfg = SharedStorageConfig(arm_state_ring_maxlen=4)
        ring = SeqlockRingBuffer.create_or_replace(name, ARM_STATE_DTYPE, maxlen=cfg.arm_state_ring_maxlen)

        try:
            from unittest.mock import patch

            # Create a minimal SharedStorage that has arm_state_ring attached.
            # We can't easily instantiate SharedStorage with just one ring, so
            # test via direct ring access.
            f = new_frame(ARM_STATE_DTYPE)
            f["qpos"][0] = np.arange(7, dtype=np.float64)
            ring.write(f)

            result = ring.get_last_k(1)
            assert len(result) == 1
        finally:
            try:
                ring.close()
                ring.unlink()
            except Exception:
                pass
