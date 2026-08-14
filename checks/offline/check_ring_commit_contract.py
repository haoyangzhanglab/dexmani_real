"""Offline check: ring commit/publish contract for ``SharedMemoryRingBuffer``.

Two invariants from the seqlock write protocol:

1. A producer publishes the logical sequence only *after* the slot payload is
   committed (even marker), so a reader that samples a fresh ``_write_seq`` can
   never mistake a half-written slot for the latest frame.
2. ``get_last_k`` skips a slot whose marker has already been overwritten by a
   newer sequence and keeps walking older sequences, rather than discarding the
   still-valid history and returning an empty list.

This check drives the ring deterministically in a single process:

- after 5 writes into a maxlen=3 ring, ``get_last_k(3)`` returns the last three
  committed sequences oldest-first (3, 4, 5), and ``get_last_k(1)`` returns only
  the newest (5);
- simulating a producer that published a newer sequence without committing its
  payload (by bumping ``_write_seq`` past the committed slots) yields the
  committed history (4, 5) instead of ``[]``.

Run from the repo root:
    conda run -n real_robot python checks/offline/check_ring_commit_contract.py
"""

from __future__ import annotations

import numpy as np

from dexmani_real.shm.ring_buffer import SharedMemoryRingBuffer


def _seqs(frames: list[tuple[np.ndarray, int, int]]) -> list[int]:
    return [seq for (_, _, seq) in frames]


def main() -> int:
    dtype = np.dtype([("v", "<i8")])
    ring = SharedMemoryRingBuffer.create_or_replace("check_ring_commit_contract", dtype, maxlen=3)
    try:
        for i in range(1, 6):
            frame = np.zeros(1, dtype=dtype)
            frame["v"][0] = i
            ring.write(frame)

        # Round-robin overwrite: only the last 3 sequences survive, oldest-first.
        assert _seqs(ring.get_last_k(3)) == [3, 4, 5], _seqs(ring.get_last_k(3))
        assert _seqs(ring.get_last_k(1)) == [5], _seqs(ring.get_last_k(1))

        # Simulate a producer that published a newer sequence (6) before
        # committing its payload: the newest slot is logically overwritten, so
        # get_last_k must skip it and return the committed 4 and 5 — not [].
        ring._write_seq[0] = np.uint64(6)
        assert _seqs(ring.get_last_k(3)) == [4, 5], _seqs(ring.get_last_k(3))

        print("OK: ring publishes sequence only after commit; get_last_k skips torn slots")
        return 0
    finally:
        ring.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
