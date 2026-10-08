"""Shared-memory ring buffer with latest-sample reads.

One serialized writer publishes with odd/even markers; readers take owned copies.
See SharedMemoryRingBuffer for platform and memory-ordering limits.
"""

from __future__ import annotations

import hashlib
import logging
import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np


def seqlock_is_complete(marker: int) -> bool:
    """True when *marker* represents a complete (nonzero, even) frame."""
    return marker != 0 and (marker & 1) == 0


def seqlock_to_logical(marker: int) -> int:
    """Decode an even seqlock marker back to the logical sequence number."""
    return marker // 2


class SeqlockSlot:
    """Odd/even seqlock for a single slot's ``[timestamp_ns, sequence]`` prefix.

    Non-owning view of the first 16 bytes of a ring slot.
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

    def begin_write(self, seq: int) -> None:
        """Mark writer-active before copying; publication time remains unknown."""
        self._ts_seq[1] = np.uint64(2 * seq - 1)
        self._ts_seq[0] = np.uint64(0)

    def end_write(self, seq: int) -> None:
        """Mark the slot complete (even)."""
        self._ts_seq[1] = np.uint64(2 * seq)

    def stamp_timestamp(self, now_ns: int) -> None:
        """Set publication time after the payload copy, before publishing even."""
        self._ts_seq[0] = np.uint64(now_ns)

    def verify(self, marker_before: int) -> bool:
        """Accept an unchanged, complete marker after copying the payload."""
        return marker_before == self.marker and seqlock_is_complete(marker_before)


logger = logging.getLogger(__name__)

TORN_WARN_INTERVAL_NS = 5 * 1_000_000_000


class SharedMemoryRingBuffer:
    """Fixed-session storage with one serialized writer and owned sample copies.

    Used on Linux x86_64. NumPy uint64 stores do not specify acquire/release
    ordering; this is not a portable lock-free guarantee. Only the creator
    unlinks, after all readers/writers have exited.
    """

    _HEADER_SIZE = 64

    def __init__(self, name: str, dtype: np.dtype, maxlen: int = 3, create: bool = True):
        if type(maxlen) is not int or maxlen <= 0:
            raise ValueError("ring capacity must be a positive integer")
        self.name, self.dtype, self.maxlen = name, np.dtype(dtype), maxlen
        if self.dtype.hasobject:
            raise ValueError("shared-memory payload must not contain Python objects")
        self._slot_dtype = np.dtype(
            [("timestamp_ns", "<u8"), ("sequence", "<u8"), ("data", self.dtype)],
            align=True,
        )
        self._slot_size = self._slot_dtype.itemsize
        self._total_size = self._HEADER_SIZE + maxlen * self._slot_size
        # Equal byte counts can still describe different pixel shapes or field layouts.
        dtype_digest = hashlib.sha256(repr(self.dtype.descr).encode("utf-8")).digest()
        layout = (self._slot_size, maxlen, *np.frombuffer(dtype_digest, dtype="<u8"))
        self._shm = shared_memory.SharedMemory(
            name=name, create=create, size=self._total_size if create else 0
        )
        try:
            if self._shm.size != self._total_size:
                raise ValueError("shared memory size differs from the supplied ring layout")
            self._header = np.ndarray((8,), dtype=np.uint64, buffer=self._shm.buf)
            if create:
                self._header[:] = (0, 0, *layout)
            elif tuple(self._header[2:]) != layout:
                raise ValueError("attached ring payload dtype, slot size or capacity differs")
            self._data_buf = np.ndarray(
                (maxlen,), dtype=self._slot_dtype, buffer=self._shm.buf, offset=self._HEADER_SIZE
            )
        except BaseException:
            self._shm.close()
            if create:
                self._shm.unlink()
            raise
        self._last_good = None
        self._last_torn_warn_ns = 0

    def write(self, data: np.ndarray) -> int:
        if not isinstance(data, np.ndarray) or data.shape != (1,) or data.dtype != self.dtype:
            raise ValueError(f"ring {self.name!r} requires ndarray (1,) dtype={self.dtype}")
        return self._publish(data)

    def write_fields(self, **fields) -> int:
        """Copy existing arrays directly into a slot, without a full-frame staging copy."""
        if self.dtype.names is None or set(fields) != set(self.dtype.names):
            raise ValueError("ring fields do not match the payload dtype")
        values = {}
        for name, value in fields.items():
            field_dtype = self.dtype.fields[name][0]
            dtype, shape = field_dtype.subdtype or (field_dtype, ())
            array = np.asarray(value)
            if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
                raise ValueError(f"ring field {name!r} must be contiguous {dtype} {shape}")
            values[name] = array
        return self._publish(values)

    def _publish(self, values) -> int:
        sequence = int(self._header[1]) + 1
        index = sequence % self.maxlen
        seqlock = SeqlockSlot(self._shm.buf, self._HEADER_SIZE + index * self._slot_size)
        seqlock.begin_write(sequence)
        if isinstance(values, np.ndarray):
            self._data_buf[index]["data"] = values[0]
        else:
            for name, value in values.items():
                self._data_buf[index]["data"][name] = value
        seqlock.stamp_timestamp(time.monotonic_ns())
        seqlock.end_write(sequence)
        self._header[1] = np.uint64(sequence)
        self._header[0] = np.uint64(index)
        return sequence

    def _copy_slot(self, index: int, expected: int | None = None):
        if not 0 <= index < self.maxlen:
            return None
        seqlock = SeqlockSlot(self._shm.buf, self._HEADER_SIZE + index * self._slot_size)
        marker = seqlock.marker
        if not seqlock_is_complete(marker):
            return None
        sequence = seqlock_to_logical(marker)
        if expected is not None and sequence != expected:
            return None
        stamp = seqlock.timestamp_ns
        data = self._data_buf[index]["data"].copy().reshape(1)
        return (data, stamp, sequence) if seqlock.verify(marker) else None

    def read_latest(self) -> tuple[np.ndarray, int, int] | None:
        """Try twice, returning the last verified sample on a persistent torn read."""
        for _ in range(2):
            index = int(self._header[0])
            if index == 0 and int(self._header[1]) == 0:
                return None
            sample = self._copy_slot(index)
            if sample is not None:
                self._last_good = sample
                return sample
        self._warn_torn_read()
        return self._last_good

    def read_latest_uncached(self) -> tuple[np.ndarray, int, int] | None:
        """Copy the latest sample once; return None when incomplete or overwritten."""
        return self._copy_slot(int(self._header[0]))

    @property
    def latest_sequence(self) -> int:
        index = int(self._header[0])
        if not 0 <= index < self.maxlen:
            return 0
        marker = SeqlockSlot(self._shm.buf, self._HEADER_SIZE + index * self._slot_size).marker
        return seqlock_to_logical(marker) if seqlock_is_complete(marker) else 0

    def read_sequence(self, sequence: int) -> tuple[np.ndarray, int, int] | None:
        if sequence <= 0:
            return None
        return self._copy_slot(sequence % self.maxlen, expected=sequence)

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

    def __getstate__(self):
        return {"name": self.name, "dtype": self.dtype, "maxlen": self.maxlen}

    def __setstate__(self, state):
        self.__init__(state["name"], state["dtype"], maxlen=state["maxlen"], create=False)

    def _warn_torn_read(self):
        now = time.monotonic_ns()
        if now - self._last_torn_warn_ns >= TORN_WARN_INTERVAL_NS:
            self._last_torn_warn_ns = now
            logger.warning("ring %s torn read; returning last-good sample or None", self.name)
