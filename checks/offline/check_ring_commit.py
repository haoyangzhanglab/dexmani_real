"""A9: ring timestamp is stamped after the payload, sequence after commit.

Verifies the deferred-commit seqlock protocol ``write()`` relies on —
``begin_write`` marks the slot odd with a *zero* timestamp, the payload is
copied, then ``stamp_timestamp`` records the commit time, then ``end_write``
publishes the even marker — and that a round trip yields a positive timestamp
with the sequence visible only after the write returns.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

import dexmani_real.shm.ring_buffer as ring_buffer_mod
from dexmani_real.shm.ring_buffer import (
    SeqlockSlot,
    SharedMemoryRingBuffer,
    _seqlock_even,
    _seqlock_odd,
)


def main() -> int:
    # ── Seqlock protocol: timestamp is deferred past the payload ────────
    buf = bytearray(32)
    slot = SeqlockSlot(buf, 0)
    seq = 7
    slot.begin_write(seq, 0)
    assert slot.marker == _seqlock_odd(seq), "begin_write must set the odd marker"
    assert slot.timestamp_ns == 0, "begin_write must NOT stamp the timestamp yet"
    stamp = 123_456_789
    slot.stamp_timestamp(stamp)
    assert slot.timestamp_ns == stamp, "stamp_timestamp must set the commit time"
    slot.end_write(seq)
    assert slot.marker == _seqlock_even(seq), "end_write must set the even marker"
    assert slot.verify(_seqlock_even(seq)), "complete even marker must verify"

    # ── write() wires the deferred-commit protocol in order ─────────────
    # Monkeypatch the SeqlockSlot class that write() looks up by module global,
    # then assert write() actually calls begin_write(seq, 0) -> payload ->
    # stamp_timestamp -> end_write.  This is the ordering fa2f2be fixed; a
    # regression that re-stamps the timestamp inside begin_write would be
    # caught here (events would show a nonzero begin_write arg and no
    # stamp_timestamp step).
    class _SpySeqlockSlot(SeqlockSlot):
        events: list = []

        def __init__(self, buf, slot_base):
            super().__init__(buf, slot_base)
            _SpySeqlockSlot.events.append("init")

        def begin_write(self, seq, now_ns):
            _SpySeqlockSlot.events.append(("begin_write", now_ns))
            super().begin_write(seq, now_ns)

        def stamp_timestamp(self, now_ns):
            _SpySeqlockSlot.events.append("stamp_timestamp")
            super().stamp_timestamp(now_ns)

        def end_write(self, seq):
            _SpySeqlockSlot.events.append("end_write")
            super().end_write(seq)

    original_slot = ring_buffer_mod.SeqlockSlot
    ring_buffer_mod.SeqlockSlot = _SpySeqlockSlot
    try:
        spy_dtype = np.dtype([("v", "<f8")])
        spy_ring = SharedMemoryRingBuffer.create_or_replace(
            "check_ring_commit_spy", spy_dtype, maxlen=2
        )
        try:
            frame0 = np.zeros(1, dtype=spy_dtype)
            frame0["v"][0] = 3.0
            _SpySeqlockSlot.events = []
            spy_ring.write(frame0)
            assert _SpySeqlockSlot.events == [
                "init",
                ("begin_write", 0),
                "stamp_timestamp",
                "end_write",
            ], f"write() must defer the timestamp: got {_SpySeqlockSlot.events}"
        finally:
            spy_ring.close()
            spy_ring.unlink()
    finally:
        ring_buffer_mod.SeqlockSlot = original_slot

    # ── Round trip: positive timestamp, sequence only after commit ──────
    dtype = np.dtype([("v", "<f8")])
    ring = SharedMemoryRingBuffer.create_or_replace("check_ring_commit_test", dtype, maxlen=3)
    try:
        assert ring.latest_sequence == 0, "sequence must start at 0"

        frame = np.zeros(1, dtype=dtype)
        frame["v"][0] = 1.0
        written_seq = ring.write(frame)
        assert written_seq == 1
        assert ring.latest_sequence == 1, "sequence must be visible after commit"

        latest = ring.read_latest()
        assert latest is not None, "a committed frame must be readable"
        data, ts, seq = latest
        assert seq == 1
        assert ts > 0, "committed frame must carry a positive timestamp"
        assert float(data["v"][0]) == 1.0

        # A second write must not move the timestamp backwards.
        frame2 = np.zeros(1, dtype=dtype)
        frame2["v"][0] = 2.0
        ring.write(frame2)
        latest2 = ring.read_latest()
        assert latest2 is not None
        assert latest2[2] == 2
        assert latest2[1] >= ts, "timestamps must be monotonically non-decreasing"
    finally:
        ring.close()
        ring.unlink()

    print("check_ring_commit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
