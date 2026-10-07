"""Shared-memory ring buffer with latest-sample reads.

One serialized writer publishes with odd/even markers; readers take owned copies.
See SharedMemoryRingBuffer for platform and memory-ordering limits.
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from dexmani_real.utils.log import get_logger


def _seqlock_odd(seq: int) -> int:
    """Encode logical *seq* as the odd (write-in-progress) marker: ``2*seq - 1``."""
    return 2 * seq - 1


def _seqlock_even(seq: int) -> int:
    """Encode logical *seq* as the even (frame-complete) marker: ``2*seq``."""
    return 2 * seq


def seqlock_is_complete(marker: int) -> bool:
    """True when *marker* represents a complete (nonzero, even) frame."""
    return marker != 0 and (marker & 1) == 0


def seqlock_to_logical(marker: int) -> int:
    """Decode an even seqlock marker back to the logical sequence number."""
    return marker // 2


class SeqlockSlot:
    """Odd/even seqlock for a single slot's ``[timestamp_ns, sequence]`` prefix.

    Non-owning view of the first 16 bytes, shared by sensor and camera rings.
    Writers mark odd with timestamp 0, copy the payload, stamp commit time,
    then publish even. Readers accept matching nonzero even markers sampled
    before and after copying; timestamps describe commit, not copy start.
    """

    def __init__(self, buf: Any, slot_base: int) -> None:
        """Bind to the ``[timestamp_ns, sequence]`` prefix at *slot_base* of *buf*."""
        self._ts_seq: np.ndarray[Any, np.dtype[np.uint64]] = np.ndarray(
            (2,), dtype=np.uint64, buffer=buf, offset=slot_base
        )

    @property
    def marker(self) -> int:
        """The current sequence marker (odd = writer active, even = complete)."""
        return int(self._ts_seq[1])

    @property
    def timestamp_ns(self) -> int:
        return int(self._ts_seq[0])

    def begin_write(self, seq: int, now_ns: int) -> None:
        """Mark writer-active (odd), then store *now_ns* as the timestamp.

        Deferred-commit writers pass ``0`` here and stamp the real commit time
        later via :meth:`stamp_timestamp`; the timestamp must not be trusted
        until :meth:`end_write` publishes the even marker.
        """
        self._ts_seq[1] = np.uint64(_seqlock_odd(seq))
        self._ts_seq[0] = np.uint64(now_ns)

    def end_write(self, seq: int) -> None:
        """Mark the slot complete (even)."""
        self._ts_seq[1] = np.uint64(_seqlock_even(seq))

    def stamp_timestamp(self, now_ns: int) -> None:
        """Stamp the timestamp after the payload commit (deferred-commit writes)."""
        self._ts_seq[0] = np.uint64(now_ns)

    def verify(self, marker_before: int) -> bool:
        """Accept an unchanged, complete marker after copying the payload."""
        return marker_before == self.marker and seqlock_is_complete(marker_before)


logger = get_logger(__name__)

TORN_WARN_INTERVAL_NS = 5 * 1_000_000_000


class SharedMemoryRingBuffer:
    """Shared-memory ring with odd/even markers for consistent reads.

    Layout of the shared memory block:
        [0:8)     write_idx  (uint64 — only producer writes, consumer reads)
        [8:16)    sequence   (uint64, monotonic counter)
        [16:24)   slot_size  (uint64, bytes per slot)
        [24:32)   maxlen     (uint64, number of slots)
        [32:64)   padding (32 bytes to cache-line-align the data region)
        [64:)     N slots, each of size slot_size
                    Each slot: [timestamp_ns: uint64, seq: uint64, data: ...]

    write_idx identifies the latest published slot.
    Used on Linux x86_64 with one serialized writer. NumPy uint64 access
    does not specify acquire/release memory ordering; this is not a portable
    lock-free guarantee. Multiple writer processes must hold their own
    cross-process write lock.

    Latest readers may skip overwritten frames; consumers check freshness.
    """

    _OFF_WRITE_IDX = 0
    _OFF_SEQUENCE = 8
    _OFF_SLOT_SIZE = 16
    _OFF_MAXLEN = 24
    _HEADER_SIZE = 64  # cache-line aligned start of data region

    def __init__(
        self,
        name: str,
        dtype: np.dtype,
        maxlen: int = 3,
        create: bool = True,
    ) -> None:
        """Create a named ring of maxlen dtype payloads, or attach when create=False."""
        self.name = name
        self.dtype = np.dtype(dtype)
        self.maxlen = maxlen

        self._slot_dtype = np.dtype(
            [("timestamp_ns", "<u8"), ("sequence", "<u8"), ("data", self.dtype)]
        )
        self._slot_size = self._slot_dtype.itemsize

        self._total_size = self._HEADER_SIZE + maxlen * self._slot_size

        if create:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=self._total_size)
        else:
            self._shm = shared_memory.SharedMemory(name=name)

        self._header: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
            (self._HEADER_SIZE,), dtype=np.uint8, buffer=self._shm.buf, offset=0
        )

        self._data_buf: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
            (maxlen,),
            dtype=self._slot_dtype,
            buffer=self._shm.buf,
            offset=self._HEADER_SIZE,
        )

        if create:
            self._init_header()

        self._write_seq: np.ndarray[Any, np.dtype[np.uint64]] = np.ndarray(
            (1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_SEQUENCE
        )
        self._last_good: tuple[np.ndarray, int, int] | None = None
        self._last_torn_warn_ns = 0

        logger.debug(
            "SharedMemoryRingBuffer(name=%s, slot_size=%d, maxlen=%d, total=%d, create=%s)",
            name,
            self._slot_size,
            maxlen,
            self._total_size,
            create,
        )

    def write(self, data: np.ndarray) -> int:
        """Publish a (1,) array matching self.dtype and return its sequence number.

        Overwrites the oldest slot.
        """
        if not isinstance(data, np.ndarray) or data.shape != (1,) or data.dtype != self.dtype:
            raise ValueError(
                f"ring {self.name!r} requires ndarray shape=(1,) dtype={self.dtype}; "
                f"got shape={getattr(data, 'shape', None)} "
                f"dtype={getattr(data, 'dtype', None)}"
            )
        seq = int(self._write_seq[0]) + 1

        idx = seq % self.maxlen

        # Mark the slot incomplete before writing; readers accept matching even markers.
        slot = self._data_buf[idx]
        seqlock = SeqlockSlot(self._shm.buf, self._HEADER_SIZE + idx * self._slot_size)
        seqlock.begin_write(seq, 0)
        slot["data"] = data
        seqlock.stamp_timestamp(time.monotonic_ns())
        seqlock.end_write(seq)

        self._write_seq[0] = np.uint64(seq)

        self._write_idx_view()[0] = np.uint64(idx)

        return seq

    def read_latest(self) -> tuple[np.ndarray, int, int] | None:
        """Return a verified ``(data, timestamp_ns, logical_sequence)`` frame."""
        for _attempt in range(2):
            idx = int(self._write_idx_view()[0])
            slot = self._data_buf[idx]
            seqlock = SeqlockSlot(self._shm.buf, self._HEADER_SIZE + idx * self._slot_size)
            marker1 = seqlock.marker
            if marker1 == 0 and idx == 0 and int(self._write_seq[0]) == 0:
                return None
            timestamp_ns = seqlock.timestamp_ns
            data = slot["data"].copy().reshape(1)
            if seqlock.verify(marker1):
                self._last_good = (data, timestamp_ns, seqlock_to_logical(marker1))
                return self._last_good
        self._warn_torn_read()
        return self._last_good

    def close(self) -> None:
        """Close the shared memory file descriptor (does NOT destroy)."""
        self._shm.close()

    def unlink(self) -> None:
        """Destroy the shared memory block (only call from creator process)."""
        self._shm.unlink()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize by identity so ``spawn`` children attach to the block.

        ``SharedMemory`` memoryviews and NumPy views themselves are not a safe
        pickle transport.  Reconstructing them from the named block also keeps
        parent and child resource ownership explicit.
        """
        # Preserve the dtype object during spawn reconstruction to retain alignment.
        return {"name": self.name, "dtype": self.dtype, "maxlen": self.maxlen}

    def __setstate__(self, state: dict[str, Any]) -> None:
        type(self).__init__(
            self,
            state["name"],
            np.dtype(state["dtype"]),
            maxlen=int(state["maxlen"]),
            create=False,
        )

    def _init_header(self) -> None:
        """Initialize the header region with zeros."""
        self._header[:] = 0
        np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_SLOT_SIZE)[0] = (
            np.uint64(self._slot_size)
        )
        np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAXLEN)[0] = (
            np.uint64(self.maxlen)
        )

    def _write_idx_view(self) -> np.ndarray:
        """Return a writeable view of the write_idx as a uint64 array of length 1."""
        return np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_WRITE_IDX)

    def _warn_torn_read(self) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_torn_warn_ns < TORN_WARN_INTERVAL_NS:
            return
        self._last_torn_warn_ns = now_ns
        logger.warning(
            "Shared-memory ring %s had a persistent torn read; returning %s",
            self.name,
            "last-good frame" if self._last_good is not None else "None",
        )
