"""Seqlock-protected shared memory ring buffer for robot state/command streams.

Subclasses ``SharedMemoryRingBuffer`` (ring_buffer.py) — the header/slot
layout, write-idx handling and lifecycle are INHERITED, so the block layout
stays byte-compatible by construction — but both the write and read paths use
a real odd/even seqlock, the torn-read protection required for robot control
(arm-hand-process-isolation plan §4.7, F2):

    write: slot.sequence = 2*seq-1 (ODD = write in progress) → ts + data →
    slot.sequence = 2*seq (EVEN = complete). The global header counter and
    the value ``write()`` returns stay the LOGICAL sequence ``seq`` (0 =
    unwritten), so RPC correlation (cmd_seq), the hand echo
    (last_cmd_seq/last_processed_seq) and the SharedMemoryRingBuffer
    "no writes" disambiguation all keep working unchanged.

    read:  seq1 = slot.sequence → copy ts + data → seq2 = slot.sequence →
    accept only if seq1 == seq2, nonzero AND EVEN (a complete frame whose
    data stores all preceded seq1's store). Odd / mismatched / zero means
    the writer is mid-overwrite → retry once with a fresh write_idx → still
    torn → return last-good cache if any, else None. Never propagate
    half-written data. Torn-read warnings are throttled (≤1 / 5 s).

Why odd/even, not just a seq1==seq2 re-check: a single sequence store BEFORE
the data leaves a legal TSO interleaving where the reader's seq1 load lands
between the writer's sequence store and its data stores — seq1 == seq2 == new
seq, and torn (or stale-data-with-new-timestamp) bytes pass verification.
With odd/even, an in-progress slot is always ODD, so a reader whose first
sample races the writer's opening sequence store sees an odd value and
rejects; a reader whose first sample is the final EVEN value is guaranteed
(by store ordering) to see all of that frame's data stores.

Note on plain-reader compatibility: the header and slot LAYOUT are identical
to SharedMemoryRingBuffer (inherited), but the slot ``sequence`` field now
carries odd/even markers, so a plain SharedMemoryRingBuffer reader would
accept an odd (mid-write) sequence — robot-ring consumers must attach with
SeqlockRingBuffer (plan §10.4: every reader does its own seqlock re-check).

The seqlock requirement mirrors CameraRingBuffer.read_latest (ring_buffer.py:
461-467); maxlen=3 @30Hz covers only ~100 ms, so a single IK spike in the main
loop can wrap the writer around mid-read — protection is a requirement, not
insurance.

Usage:
    # Producer (arm control process)
    ring = SeqlockRingBuffer("dexmani_arm_state", ARM_STATE_DTYPE, maxlen=3, create=True)
    ring.write(state_record)

    # Consumer (main process façade)
    ring = SeqlockRingBuffer("dexmani_arm_state", ARM_STATE_DTYPE, maxlen=3, create=False)
    latest = ring.read_latest()  # (data copy shaped (1,), timestamp_ns, seq) or None
    if latest is not None and is_fresh(latest[1], timeout_s=0.06): ...
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

from dexmani_real.shm.ring_buffer import SharedMemoryRingBuffer
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class SeqlockRingBuffer(SharedMemoryRingBuffer):
    """Lock-free FILO ring buffer with an odd/even seqlock-verified read path.

    Layout of the shared memory block (inherited from SharedMemoryRingBuffer):
        [0:8)     write_idx  (uint64, atomic — only producer writes, consumer reads)
        [8:16)    sequence   (uint64, monotonic LOGICAL counter — 0 = unwritten)
        [16:24)   slot_size  (uint64, bytes per slot)
        [24:32)   maxlen     (uint64, number of slots)
        [32:64)   padding (32 bytes to cache-line-align the data region)
        [64:)     N slots, each of size slot_size
                    Each slot: [timestamp_ns: uint64, seq: uint64, data: ...]
                    Slot ``seq`` carries the seqlock marker: ODD (2*seq-1)
                    while the frame is being written, EVEN (2*seq) once
                    complete, where ``seq`` is the logical sequence.

    The offsets, header init, write_idx handling, ``frame_age_ns()``,
    ``latest_sequence`` and close/unlink are inherited from
    SharedMemoryRingBuffer, so the block layout is byte-compatible by
    construction. ``write()`` is overridden to publish the odd/even markers
    around the data, and ``read_latest()`` adds seqlock verification: the
    slot's sequence is sampled before and after the copy; anything but
    seq1 == seq2 != 0 and EVEN (writer mid-overwrite, or unwritten slot)
    means the copy is torn and is discarded.

    Torn-read handling (plan §4.7):
        1. retry once, re-reading write_idx (the writer may have advanced to
           a newer, consistent slot);
        2. still torn → return the last-good cached frame if one exists,
           otherwise None; a throttled warning is logged (≤1 / 5 s).
    Half-written data is never returned.
    """

    # Torn-read warning throttle: at most one warning per 5 s per buffer.
    _TORN_WARN_INTERVAL_NS = 5 * 1_000_000_000

    def __init__(
        self,
        name: str,
        dtype: np.dtype,
        maxlen: int = 3,
        create: bool = True,
        stale_cleanup: bool = True,
    ) -> None:
        """Initialize or attach to a named shared memory ring buffer.

        Args:
            name: Unique name for the shared memory block (e.g. "dexmani_arm_state").
            dtype: Numpy dtype for each slot's data payload.
            maxlen: Number of slots in the ring buffer.
            create: If True, create the shared memory block; if False, attach
                    to an existing one.
            stale_cleanup: If True and ``create`` hits FileExistsError, unlink
                    the leftover block from a crashed run and recreate
                    (camera_process.py stale-SHM pattern, plan §5.1 — a stale
                    arm_target is an immediate-motion hazard). If False,
                    FileExistsError propagates.
        """
        if create and stale_cleanup:
            try:
                super().__init__(name, dtype, maxlen=maxlen, create=True)
            except FileExistsError:
                # Leftover block from a run that died without unlink. Stale
                # targets/commands are worse than stale camera frames — drop
                # and recreate (plan §5.1, D5).
                logger.warning(
                    "SeqlockRingBuffer '%s' already exists (stale from a previous run) " "— unlinking and recreating.",
                    name,
                )
                stale = shared_memory.SharedMemory(name=name)
                stale.close()
                stale.unlink()
                super().__init__(name, dtype, maxlen=maxlen, create=True)
        else:
            super().__init__(name, dtype, maxlen=maxlen, create=create)

        # Last-good frame cache for torn-read fallback (plan §4.7).
        self._last_good: tuple[np.ndarray, int, int] | None = None
        self._last_torn_warn_ns = 0

        logger.debug(
            "SeqlockRingBuffer(name=%s, slot_size=%d, maxlen=%d, total=%d, create=%s)",
            name,
            self._slot_size,
            maxlen,
            self._total_size,
            create,
        )

    # ------------------------------------------------------------------
    # Producer API (odd/even seqlock around the data)
    # ------------------------------------------------------------------

    def write(self, data: np.ndarray) -> int:
        """Write one frame into the ring buffer (producer-side).

        Seqlock protocol: the slot sequence is set to the ODD value
        ``2*seq-1`` (write in progress) BEFORE the timestamp/data stores and
        to the EVEN value ``2*seq`` (frame complete) AFTER them, so a
        concurrent reader can distinguish a complete frame from a mid-write
        one (x86_64 TSO keeps the stores in program order). The global
        header counter holds the LOGICAL sequence ``seq``.

        Overwrites the oldest slot. Returns the new LOGICAL sequence number.

        Args:
            data: A 0-d or 1-d structured array matching self.dtype.
        """
        now_ns = time.monotonic_ns()

        # Increment the logical sequence (global counter; 0 = unwritten).
        seq = int(self._write_seq[0]) + 1
        self._write_seq[0] = np.uint64(seq)

        # Compute slot index
        idx = seq % self.maxlen

        # Write to slot: ODD marker → ts + data → EVEN marker.
        slot = self._data_buf[idx]
        slot["sequence"] = np.uint64(2 * seq - 1)  # odd: write in progress
        slot["timestamp_ns"] = np.uint64(now_ns)
        slot["data"] = data
        slot["sequence"] = np.uint64(2 * seq)  # even: frame complete

        # Atomic write of write_idx (aligned uint64 store on x86_64)
        self._write_idx_view()[0] = np.uint64(idx)

        return seq

    # ------------------------------------------------------------------
    # Consumer API (seqlock-verified, plan §4.7)
    # ------------------------------------------------------------------

    def read_latest(self) -> tuple[np.ndarray, int, int] | None:  # type: ignore[override]
        """Read the most recently written frame (consumer-side).

        Seqlock protocol: sample the slot sequence before and after copying
        ts + data; accept only when both samples agree AND are nonzero AND
        EVEN. An odd value (writer mid-write), a mismatch (writer wrapped
        around and overwrote the slot mid-copy) or a zero (unwritten slot)
        means the copy is torn and is discarded. On a torn read, retry once
        with a fresh write_idx; if still torn, return the last-good cached
        frame (never half-written data), or None if no good frame was ever
        read.

        Returns:
            (data copy shaped (1,), timestamp_ns, logical sequence) — the
            data copy matches the record shape produced by ``new_frame()`` —
            or None only when nothing has been written (and no last-good
            exists).
        """
        for _attempt in range(2):
            idx = int(self._write_idx_view()[0])
            slot = self._data_buf[idx]
            seq1 = int(slot["sequence"])

            # Nothing written yet (same disambiguation as SharedMemoryRingBuffer:
            # slot 0 seq==0 is ambiguous between zero writes and a mid-write
            # slot 0 — the global LOGICAL sequence counter resolves it).
            if seq1 == 0 and idx == 0 and int(self._write_seq[0]) == 0:
                return None

            ts = int(slot["timestamp_ns"])
            data = slot["data"].copy().reshape(1)
            seq2 = int(slot["sequence"])

            # A complete frame: both samples equal, written (nonzero) and
            # EVEN. Because the writer stores the even marker AFTER the data,
            # observing it on the first sample guarantees all of that frame's
            # data stores are visible; agreement on the second sample
            # guarantees no later overwrite disturbed the copy.
            if seq1 == seq2 and seq1 != 0 and (seq1 & 1) == 0:
                self._last_good = (data, ts, seq1 // 2)
                return self._last_good
            # Torn (writer mid-overwrite: odd or mismatched) or unwritten
            # slot — retry with a fresh write_idx; the writer may have
            # advanced to a consistent slot.

        # Still torn after one retry: fall back to last-good, never propagate
        # half-written data. Throttled warning (≤1 / 5 s).
        self._warn_torn_read()
        return self._last_good

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _warn_torn_read(self) -> None:
        """Log a torn-read fallback warning, throttled to ≤1 / 5 s."""
        now_ns = time.monotonic_ns()
        if now_ns - self._last_torn_warn_ns < self._TORN_WARN_INTERVAL_NS:
            return
        self._last_torn_warn_ns = now_ns
        logger.warning(
            "SeqlockRingBuffer '%s': torn read persisted after retry; returning %s.",
            self.name,
            "last-good cached frame" if self._last_good is not None else "None",
        )


def is_fresh(ts_ns: int, timeout_s: float, now_ns: int | None = None) -> bool:
    """Return True if a frame timestamp is within ``timeout_s`` of ``now_ns``.

    Args:
        ts_ns: Frame timestamp in ``time.monotonic_ns()`` units (slot timestamp_ns).
        timeout_s: Freshness budget in seconds.
        now_ns: Current monotonic time; defaults to ``time.monotonic_ns()``.
                Injectable for fake-clock tests.

    Invalid timestamps (ts_ns <= 0, i.e. never written) are never fresh.
    """
    if ts_ns <= 0:
        return False
    if now_ns is None:
        now_ns = time.monotonic_ns()
    return (now_ns - ts_ns) <= int(timeout_s * 1e9)
