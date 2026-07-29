"""Shared seqlock protocol helpers for lock-free cross-process ring buffers.

Used by ``CameraRingBuffer`` (ring_buffer.py) and ``SeqlockRingBuffer``
(robot_ring.py).  Both classes implement the same odd/even seqlock protocol
with different memory layouts; these functions encode the shared convention:

    - Logical sequence *seq* (1, 2, 3, …) — 0 = unwritten.
    - odd marker ``2*seq - 1`` — writer is mid-write.
    - even marker ``2*seq`` — frame complete.

Why odd/even, not just a seq1==seq2 re-check: a single sequence store before
the data leaves a legal TSO interleaving where the reader's seq1 load lands
between the writer's sequence store and its data stores — seq1 == seq2 == new
seq, and torn (or stale-data-with-new-timestamp) bytes pass verification.
With odd/even, an in-progress slot is always ODD, so a reader whose first
sample races the writer's opening sequence store sees an odd value and
rejects; a reader whose first sample is the final EVEN value is guaranteed
(by store ordering) to see all of that frame's data stores.
"""

from __future__ import annotations


def seqlock_odd(seq: int) -> int:
    """Encode logical *seq* as the odd (write-in-progress) marker: ``2*seq - 1``."""
    return 2 * seq - 1


def seqlock_even(seq: int) -> int:
    """Encode logical *seq* as the even (frame-complete) marker: ``2*seq``."""
    return 2 * seq


def seqlock_is_complete(marker: int) -> bool:
    """True when *marker* represents a complete (nonzero, even) frame."""
    return marker != 0 and (marker & 1) == 0


def seqlock_to_logical(marker: int) -> int:
    """Decode an even seqlock marker back to the logical sequence number."""
    return marker // 2
