"""Lock-free shared-memory ring buffer with latest, history, and sequence reads.

Uses multiprocessing.shared_memory for zero-copy cross-process communication.
Each ring has one serialized writer at a time and may have multiple readers.
Odd/even markers prevent readers from accepting a slot while its payload is
being overwritten.

     DexUMI drop-oldest backpressure via FILO semantics.

Usage:
    buf = SharedMemoryRingBuffer("vr_frames", VR_FRAME_DTYPE, maxlen=3, create=True)
    buf.write(vr_frame_array)

    buf = SharedMemoryRingBuffer("vr_frames", VR_FRAME_DTYPE, maxlen=3, create=False)
    latest = buf.read_latest()  # (data copy, timestamp_ns, logical sequence) or None
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

    Every slot in both ``SharedMemoryRingBuffer`` and ``CameraRingBuffer`` is
    laid out as ``[timestamp_ns: u8, sequence: u8, <payload>]``.  A producer
    marks the slot writer-active (odd) with a *zero* placeholder timestamp via
    :meth:`begin_write`, copies the payload, stamps the true commit time via
    :meth:`stamp_timestamp`, and finally calls :meth:`end_write` (even marker),
    so a reader can never observe a half-written frame as complete and the
    timestamp reflects the commit time rather than the start of the payload
    copy.  Readers sample :attr:`marker`
    before and after reading the payload and accept the frame only when
    :meth:`verify` confirms both samples agree on a complete (nonzero, even)
    marker.

    Instances are cheap, non-owning views over a slot's first 16 bytes and may
    be created per access without concern.
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
        """Return True when the marker is unchanged and complete after the payload read.

        ``marker_before`` is the marker sampled before reading the payload; the
        marker is re-sampled now.  A torn read (marker changed mid-read) or a
        writer-active (odd) slot fails the check.
        """
        return marker_before == self.marker and seqlock_is_complete(marker_before)


logger = get_logger(__name__)

# Torn-read warning throttle: at most one warning per 5 s per buffer.
TORN_WARN_INTERVAL_NS = 5 * 1_000_000_000


class SharedMemoryRingBuffer:
    """Lock-free, odd/even-seqlock FILO ring buffer in shared memory.

    Layout of the shared memory block:
        [0:8)     write_idx  (uint64, atomic — only producer writes, consumer reads)
        [8:16)    sequence   (uint64, monotonic counter)
        [16:24)   slot_size  (uint64, bytes per slot)
        [24:32)   maxlen     (uint64, number of slots)
        [32:64)   padding (32 bytes to cache-line-align the data region)
        [64:)     N slots, each of size slot_size
                    Each slot: [timestamp_ns: uint64, seq: uint64, data: ...]

    The write_idx always points to the most recently written slot index
    (0..maxlen-1). The consumer reads write_idx to find the latest frame.
    Because only one serialized writer updates write_idx, this is safe on
    x86_64 without CAS. Callers with multiple writer processes must hold their
    own cross-process write lock.

    Latest/history readers use FILO semantics: old frames are silently
    overwritten (drop-oldest backpressure). A sequence-addressed reader can
    instead ownership-copy one known resident frame. That form is for bounded
    FIFO consumers that track their own acknowledgement sequence.
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
        """Initialize or attach to a named shared memory ring buffer.

        Args:
            name: Unique name for the shared memory block (e.g. "vr_frames").
            dtype: Numpy dtype for each slot's data payload.
            maxlen: Number of slots in the ring buffer.
            create: If True, create the shared memory block; if False, attach
                    to an existing one.
        """
        self.name = name
        self.dtype = np.dtype(dtype)
        self.maxlen = maxlen

        # Slot layout: timestamp_ns (u8) + sequence (u8) + data
        self._slot_dtype = np.dtype(
            [("timestamp_ns", "<u8"), ("sequence", "<u8"), ("data", self.dtype)]
        )
        self._slot_size = self._slot_dtype.itemsize

        self._total_size = self._HEADER_SIZE + maxlen * self._slot_size

        if create:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total_size
            )
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
        self._last_torn_warn_k_ns = 0

        logger.debug(
            "SharedMemoryRingBuffer(name=%s, slot_size=%d, maxlen=%d, total=%d, create=%s)",
            name,
            self._slot_size,
            maxlen,
            self._total_size,
            create,
        )

    def write(self, data: np.ndarray) -> int:
        """Write one frame into the ring buffer (producer-side).

        Overwrites the oldest slot. Returns the new sequence number.

        Args:
            data: A one-element structured array matching ``self.dtype`` exactly.
        """
        if (
            not isinstance(data, np.ndarray)
            or data.shape != (1,)
            or data.dtype != self.dtype
        ):
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
            seqlock = SeqlockSlot(
                self._shm.buf, self._HEADER_SIZE + idx * self._slot_size
            )
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

    def read_sequence(self, sequence: int) -> tuple[np.ndarray, int, int] | None:
        """Ownership-copy one exact, still-resident logical sequence.

        Unlike :meth:`get_last_k`, this method never scans or copies unrelated
        history slots. It is therefore suitable for a bounded FIFO consumer
        that already knows the next sequence it must acknowledge. ``None``
        means that the requested sequence is not resident or could not be read
        consistently; callers that require losslessness must not skip ahead.
        """
        if sequence <= 0:
            return None

        slot = self._data_buf[sequence % self.maxlen]
        seqlock = SeqlockSlot(
            self._shm.buf,
            self._HEADER_SIZE + (sequence % self.maxlen) * self._slot_size,
        )
        for _attempt in range(2):
            marker1 = seqlock.marker
            if (
                not seqlock_is_complete(marker1)
                or seqlock_to_logical(marker1) != sequence
            ):
                return None
            timestamp_ns = seqlock.timestamp_ns
            data = slot["data"].copy().reshape(1)
            if seqlock.verify(marker1) and seqlock_to_logical(marker1) == sequence:
                return data, timestamp_ns, sequence
        return None

    def get_last_k(self, k: int) -> list[tuple[np.ndarray, int, int]]:
        """Return up to ``k`` independently verified frames, oldest first."""
        if k <= 0:
            return []
        if k > self.maxlen:
            raise ValueError(f"k ({k}) exceeds ring capacity maxlen ({self.maxlen})")
        latest_seq = int(self._write_seq[0])
        if latest_seq == 0:
            return []
        frames: list[tuple[np.ndarray, int, int]] = []
        dropped = False
        count = min(k, latest_seq)
        first_seq = latest_seq - count + 1
        for target_seq in range(first_seq, latest_seq + 1):
            slot = self._data_buf[target_seq % self.maxlen]
            seqlock = SeqlockSlot(
                self._shm.buf,
                self._HEADER_SIZE + (target_seq % self.maxlen) * self._slot_size,
            )
            accepted = False
            for _attempt in range(2):
                marker1 = seqlock.marker
                if not seqlock_is_complete(marker1):
                    continue
                if seqlock_to_logical(marker1) != target_seq:
                    # Skip slots overwritten during the read.
                    dropped = True
                    break
                timestamp_ns = seqlock.timestamp_ns
                data = slot["data"].copy().reshape(1)
                if (
                    seqlock.verify(marker1)
                    and seqlock_to_logical(marker1) == target_seq
                ):
                    frames.append((data, timestamp_ns, target_seq))
                    accepted = True
                    break
            if not accepted:
                dropped = True
        if dropped:
            self._warn_torn_read_k(k, len(frames))
        return frames

    @property
    def latest_sequence(self) -> int:
        return int(self._write_seq[0])

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
        np.ndarray(
            (1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_SLOT_SIZE
        )[0] = np.uint64(self._slot_size)
        np.ndarray(
            (1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAXLEN
        )[0] = np.uint64(self.maxlen)

    def _write_idx_view(self) -> np.ndarray:
        """Return a writeable view of the write_idx as a uint64 array of length 1."""
        return np.ndarray(
            (1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_WRITE_IDX
        )

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

    def _warn_torn_read_k(self, k: int, recovered: int) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_torn_warn_k_ns < TORN_WARN_INTERVAL_NS:
            return
        self._last_torn_warn_k_ns = now_ns
        logger.warning(
            "Shared-memory ring %s recovered %d/%d history frames",
            self.name,
            recovered,
            k,
        )
