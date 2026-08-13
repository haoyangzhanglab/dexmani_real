"""Lock-free FILO ring buffer in shared memory.

Uses multiprocessing.shared_memory for zero-copy cross-process communication.
Each ring has one producer and may have multiple readers. Odd/even markers
prevent readers from accepting a slot while its payload is being overwritten.

     DexUMI drop-oldest backpressure via FILO semantics.

Usage:
    # Producer process
    buf = SharedMemoryRingBuffer("vr_frames", VR_FRAME_DTYPE, maxlen=3, create=True)
    buf.write(vr_frame_array)

    # Consumer process
    buf = SharedMemoryRingBuffer("vr_frames", VR_FRAME_DTYPE, maxlen=3, create=False)
    latest = buf.read_latest()  # (data copy, timestamp_ns, logical sequence) or None
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from dexmani_real.utils.schema import CAMERA_FRAME_HEADER_DTYPE
from dexmani_real.utils.log import get_logger

# ---------------------------------------------------------------------------
# Seqlock protocol helpers (odd/even markers for lock-free torn-read defence)
# ---------------------------------------------------------------------------


def _seqlock_odd(seq: int) -> int:
    """Encode logical *seq* as the odd (write-in-progress) marker: ``2*seq - 1``."""
    return 2 * seq - 1


def _seqlock_even(seq: int) -> int:
    """Encode logical *seq* as the even (frame-complete) marker: ``2*seq``."""
    return 2 * seq


def _seqlock_is_complete(marker: int) -> bool:
    """True when *marker* represents a complete (nonzero, even) frame."""
    return marker != 0 and (marker & 1) == 0


def _seqlock_to_logical(marker: int) -> int:
    """Decode an even seqlock marker back to the logical sequence number."""
    return marker // 2


class SeqlockSlot:
    """Odd/even seqlock for a single slot's ``[timestamp_ns, sequence]`` prefix.

    Every slot in both ``SharedMemoryRingBuffer`` and ``CameraRingBuffer`` is
    laid out as ``[timestamp_ns: u8, sequence: u8, <payload>]``.  A producer
    calls :meth:`begin_write` (odd marker, then timestamp) before writing the
    payload and :meth:`end_write` (even marker) after, so a reader can never
    observe a half-written frame as complete.  Readers sample :attr:`marker`
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
        """Mark writer-active (odd), then stamp the timestamp — in that order."""
        self._ts_seq[1] = np.uint64(_seqlock_odd(seq))
        self._ts_seq[0] = np.uint64(now_ns)

    def end_write(self, seq: int) -> None:
        """Mark the slot complete (even)."""
        self._ts_seq[1] = np.uint64(_seqlock_even(seq))

    def verify(self, marker_before: int) -> bool:
        """Return True when the marker is unchanged and complete after the payload read.

        ``marker_before`` is the marker sampled before reading the payload; the
        marker is re-sampled now.  A torn read (marker changed mid-read) or a
        writer-active (odd) slot fails the check.
        """
        return marker_before == self.marker and _seqlock_is_complete(marker_before)


logger = get_logger(__name__)

# Torn-read warning throttle: at most one warning per 5 s per buffer.
# Shared by CameraRingBuffer and SharedMemoryRingBuffer.
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
    Because only the producer writes write_idx and only the consumer reads
    it, this is safe on x86_64 without CAS.

    FILO semantics: the buffer always returns the latest frame; old frames
    are silently overwritten (drop-oldest backpressure).
    """

    # Offset constants
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
        self._slot_dtype = np.dtype([("timestamp_ns", "<u8"), ("sequence", "<u8"), ("data", self.dtype)])
        self._slot_size = self._slot_dtype.itemsize

        # Total shared memory size
        self._total_size = self._HEADER_SIZE + maxlen * self._slot_size

        if create:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=self._total_size)
        else:
            self._shm = shared_memory.SharedMemory(name=name)

        # Map the header first (needed by _init_header)
        self._header: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
            (self._HEADER_SIZE,), dtype=np.uint8, buffer=self._shm.buf, offset=0
        )

        # Map the data region as a numpy array
        self._data_buf: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
            (maxlen,), dtype=self._slot_dtype, buffer=self._shm.buf, offset=self._HEADER_SIZE
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

    @classmethod
    def create_or_replace(cls, name: str, dtype: np.dtype, maxlen: int = 3) -> "SharedMemoryRingBuffer":
        """Create a ring, unlinking a stale block left by a crashed run."""
        try:
            return cls(name, dtype, maxlen=maxlen, create=True)
        except FileExistsError:
            logger.warning("Shared-memory ring %s already exists; replacing stale block", name)
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
            return cls(name, dtype, maxlen=maxlen, create=True)

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def write(self, data: np.ndarray) -> int:
        """Write one frame into the ring buffer (producer-side).

        Overwrites the oldest slot. Returns the new sequence number.

        Args:
            data: A 0-d or 1-d structured array matching self.dtype.
        """
        now_ns = time.monotonic_ns()

        # Increment the logical sequence.
        seq = int(self._write_seq[0]) + 1
        self._write_seq[0] = np.uint64(seq)

        # Compute slot index
        idx = seq % self.maxlen

        # Mark the slot incomplete before touching its payload, then publish an
        # even completion marker. Readers accept only two matching even reads.
        slot = self._data_buf[idx]
        seqlock = SeqlockSlot(self._shm.buf, self._HEADER_SIZE + idx * self._slot_size)
        seqlock.begin_write(seq, now_ns)
        slot["data"] = data
        seqlock.end_write(seq)

        # Atomic write of write_idx (aligned uint64 store on x86_64)
        self._write_idx_view()[0] = np.uint64(idx)

        return seq

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

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
                self._last_good = (data, timestamp_ns, _seqlock_to_logical(marker1))
                return self._last_good
        self._warn_torn_read()
        return self._last_good

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
        for offset in range(min(k, latest_seq)):
            target_seq = latest_seq - offset
            slot = self._data_buf[target_seq % self.maxlen]
            seqlock = SeqlockSlot(self._shm.buf, self._HEADER_SIZE + (target_seq % self.maxlen) * self._slot_size)
            accepted = False
            for _attempt in range(2):
                marker1 = seqlock.marker
                if not _seqlock_is_complete(marker1):
                    continue
                if _seqlock_to_logical(marker1) != target_seq:
                    frames.reverse()
                    if dropped:
                        self._warn_torn_read_k(k, len(frames))
                    return frames
                timestamp_ns = seqlock.timestamp_ns
                data = slot["data"].copy().reshape(1)
                if seqlock.verify(marker1) and _seqlock_to_logical(marker1) == target_seq:
                    frames.append((data, timestamp_ns, target_seq))
                    accepted = True
                    break
            if not accepted:
                dropped = True
        frames.reverse()
        if dropped:
            self._warn_torn_read_k(k, len(frames))
        return frames

    def frame_age_ns(self) -> int:
        """Return age of the latest frame in nanoseconds, or -1 if no frame."""
        idx = int(self._write_idx_view()[0])
        slot = self._data_buf[idx]

        if slot["sequence"] == 0 and idx == 0 and int(self._write_seq[0]) == 0:
            return -1

        return time.monotonic_ns() - int(slot["timestamp_ns"])

    @property
    def latest_sequence(self) -> int:
        return int(self._write_seq[0])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        # Pickle the dtype object itself. ``dtype.descr`` materializes implicit
        # alignment gaps as synthetic ``fN`` fields, so an aligned structured
        # dtype no longer compares/casts equal after spawn reconstruction.
        return {"name": self.name, "dtype": self.dtype, "maxlen": self.maxlen}

    def __setstate__(self, state: dict[str, Any]) -> None:
        type(self).__init__(
            self, state["name"], np.dtype(state["dtype"]), maxlen=int(state["maxlen"]), create=False
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_header(self) -> None:
        """Initialize the header region with zeros."""
        self._header[:] = 0
        # Write slot_size and maxlen for introspection
        np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_SLOT_SIZE)[0] = np.uint64(
            self._slot_size
        )
        np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAXLEN)[0] = np.uint64(self.maxlen)

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


class CameraRingBuffer:
    """Shared memory ring buffer for large camera frames (~1.5MB each).

    Uses a different layout than SharedMemoryRingBuffer because camera
    frames contain variable-size RGB and depth arrays. Each slot stores:
      - Header: CAMERA_FRAME_HEADER_DTYPE (metadata)
      - RGB raw bytes
      - Depth raw bytes
      - Optional pointcloud block (fixed-size float32 (N, 6), pc_shape != None)

    Layout of shared memory:
        [0:8)     write_idx  (uint64, atomic)
        [8:16)    sequence   (uint64)
        [16:24)   max_rgb_bytes (uint64, max RGB bytes per frame)
        [24:32)   max_depth_bytes (uint64, max depth bytes per frame)
        [32:40)   max_pc_bytes (uint64, pointcloud bytes per frame, 0 = none)
        [40:64)   padding
        [64:)     N slots, each of:
                    [0:8)   timestamp_ns (uint64)
                    [8:16)  sequence (uint64)
                    [16:16+header.itemsize) CAMERA_FRAME_HEADER_DTYPE
                    [...:...+max_rgb_bytes) RGB data
                    [...:...+max_depth_bytes) Depth data
                    [...:...+max_pc_bytes) Pointcloud data (float32)
    """

    _OFF_WRITE_IDX = 0
    _OFF_SEQUENCE = 8
    _OFF_MAX_RGB = 16
    _OFF_MAX_DEPTH = 24
    _OFF_MAX_PC = 32
    _HEADER_SIZE = 64  # cache-line aligned

    def __init__(
        self,
        name: str,
        rgb_shape: tuple[int, int, int] | None = None,
        depth_shape: tuple[int, int] | None = None,
        maxlen: int = 5,
        create: bool = True,
        pc_shape: tuple[int, int] | None = None,
    ) -> None:
        self.name = name
        self.maxlen = maxlen

        # Per-slot layout
        self._slot_header_size = 8 + 8 + CAMERA_FRAME_HEADER_DTYPE.itemsize

        if create:
            if rgb_shape is None or depth_shape is None:
                raise ValueError("rgb_shape and depth_shape are required when create=True")
            self._rgb_shape: tuple[int, int, int] | None = rgb_shape
            self._depth_shape: tuple[int, int] | None = depth_shape
            self._pc_shape: tuple[int, int] | None = pc_shape
            self._max_rgb_bytes = rgb_shape[0] * rgb_shape[1] * rgb_shape[2]
            self._max_depth_bytes = depth_shape[0] * depth_shape[1] * 2
            self._max_pc_bytes = pc_shape[0] * pc_shape[1] * 4 if pc_shape else 0
            self._slot_size = self._slot_header_size + self._max_rgb_bytes + self._max_depth_bytes + self._max_pc_bytes
            self._total_size = self._HEADER_SIZE + maxlen * self._slot_size

            self._shm = shared_memory.SharedMemory(name=name, create=True, size=self._total_size)
            self._init_header()
        else:
            # Attach to existing SHM — read dimensions from the header.
            self._shm = shared_memory.SharedMemory(name=name)
            self._max_rgb_bytes = int(
                np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAX_RGB)[0]
            )
            self._max_depth_bytes = int(
                np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAX_DEPTH)[0]
            )
            self._max_pc_bytes = int(
                np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAX_PC)[0]
            )
            self._slot_size = self._slot_header_size + self._max_rgb_bytes + self._max_depth_bytes + self._max_pc_bytes
            self._total_size = self._HEADER_SIZE + maxlen * self._slot_size
            # Reconstruct shapes from byte counts for attach-mode consumers.
            self._rgb_shape = None
            self._depth_shape = None
            self._pc_shape = (self._max_pc_bytes // (6 * 4), 6) if self._max_pc_bytes > 0 else None

        self._last_torn_warn_ns = 0

        self._write_seq: np.ndarray[Any, np.dtype[np.uint64]] = np.ndarray(
            (1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_SEQUENCE
        )

        logger.debug(
            "CameraRingBuffer(name=%s, slot_size=%d, maxlen=%d, total=%dMB, create=%s)",
            name,
            self._slot_size,
            maxlen,
            self._total_size / (1024 * 1024),
            create,
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "maxlen": self.maxlen,
            "rgb_shape": self._rgb_shape,
            "depth_shape": self._depth_shape,
            "pc_shape": self._pc_shape,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        CameraRingBuffer.__init__(self, state["name"], maxlen=int(state["maxlen"]), create=False)
        rgb_shape = state["rgb_shape"]
        depth_shape = state["depth_shape"]
        pc_shape = state["pc_shape"]
        self._rgb_shape = (int(rgb_shape[0]), int(rgb_shape[1]), int(rgb_shape[2])) if rgb_shape is not None else None
        self._depth_shape = (int(depth_shape[0]), int(depth_shape[1])) if depth_shape is not None else None
        self._pc_shape = (int(pc_shape[0]), int(pc_shape[1])) if pc_shape is not None else None

    def write(
        self,
        header: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        pointcloud: np.ndarray | None = None,
    ) -> int:
        """Write a camera frame into the ring buffer.

        Args:
            header: 1-d array of CAMERA_FRAME_HEADER_DTYPE (1 element).
            rgb: Raw RGB bytes (uint8 array, flattened).
            depth: Raw depth bytes (uint16 array, flattened).
            pointcloud: Contiguous float32 array matching pc_shape (pass a
                zeros block when no valid cloud); ignored when the buffer was
                created without pc_shape.
        Returns:
            New sequence number.
        """
        if not isinstance(header, np.ndarray) or header.shape != (1,) or header.dtype != CAMERA_FRAME_HEADER_DTYPE:
            raise ValueError(
                f"camera header must have shape (1,) and dtype {CAMERA_FRAME_HEADER_DTYPE}, "
                f"got shape={getattr(header, 'shape', None)} dtype={getattr(header, 'dtype', None)}"
            )
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2:] != (3,):
            raise ValueError(f"camera rgb must be (H, W, 3) uint8, got shape={getattr(rgb, 'shape', None)}")
        if not isinstance(depth, np.ndarray) or depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"camera depth must be (H, W) uint16, got shape={getattr(depth, 'shape', None)}")
        if not rgb.flags.c_contiguous or not depth.flags.c_contiguous:
            raise ValueError("camera rgb/depth payloads must be C-contiguous")
        if self._rgb_shape is not None and rgb.shape != self._rgb_shape:
            raise ValueError(f"camera rgb shape {rgb.shape} does not match ring capacity {self._rgb_shape}")
        if self._depth_shape is not None and depth.shape != self._depth_shape:
            raise ValueError(f"camera depth shape {depth.shape} does not match ring capacity {self._depth_shape}")
        if rgb.nbytes != self._max_rgb_bytes or depth.nbytes != self._max_depth_bytes:
            raise ValueError("camera rgb/depth payload must exactly fill its configured ring capacity")
        if self._max_pc_bytes > 0:
            if pointcloud is None or pointcloud.dtype != np.float32 or pointcloud.shape != self._pc_shape:
                raise ValueError(
                    f"camera pointcloud must be {self._pc_shape} float32, "
                    f"got shape={getattr(pointcloud, 'shape', None)} dtype={getattr(pointcloud, 'dtype', None)}"
                )
            if pointcloud.nbytes != self._max_pc_bytes:
                raise ValueError("camera pointcloud payload must exactly fill its configured ring capacity")
            if not pointcloud.flags.c_contiguous:
                raise ValueError("camera pointcloud payload must be C-contiguous")

        h = header[0]
        if int(h["rgb_size"]) != rgb.nbytes or int(h["depth_size"]) != depth.nbytes:
            raise ValueError("camera header byte sizes do not match payloads")
        if (int(h["rgb_shape_h"]), int(h["rgb_shape_w"]), int(h["rgb_shape_c"])) != rgb.shape:
            raise ValueError("camera header RGB shape does not match payload")
        if (int(h["depth_shape_h"]), int(h["depth_shape_w"])) != depth.shape:
            raise ValueError("camera header depth shape does not match payload")
        pc_num_points = int(h["pc_num_points"])
        if self._pc_shape is not None and not 0 <= pc_num_points <= self._pc_shape[0]:
            raise ValueError(f"pc_num_points={pc_num_points} exceeds ring capacity {self._pc_shape[0]}")
        if self._pc_shape is None and (pointcloud is not None or pc_num_points != 0):
            raise ValueError("pointcloud payload/header provided to a ring without pointcloud capacity")
        if bool(h["pointcloud_valid"]) != (pc_num_points > 0):
            raise ValueError("pointcloud_valid must agree with pc_num_points")
        valid_depth_ratio = float(h["pc_valid_depth_ratio"])
        padding_count = int(h["pc_padding_count"])
        if not np.isfinite(valid_depth_ratio) or not 0.0 <= valid_depth_ratio <= 1.0:
            raise ValueError("pc_valid_depth_ratio must be finite and in [0, 1]")
        if self._pc_shape is not None and not 0 <= padding_count <= self._pc_shape[0]:
            raise ValueError("pc_padding_count exceeds ring capacity")

        now_ns = time.monotonic_ns()
        seq = int(self._write_seq[0]) + 1
        self._write_seq[0] = np.uint64(seq)

        idx = seq % self.maxlen
        slot_base = self._HEADER_SIZE + idx * self._slot_size

        # ── Seqlock write protocol: odd→data→even ──
        # Write an odd marker BEFORE the payload (including timestamp) so
        # concurrent readers see "writer active" and bail out.
        seqlock = SeqlockSlot(self._shm.buf, slot_base)
        seqlock.begin_write(seq, now_ns)

        # Write the fixed camera transport header.
        header_offset = slot_base + 16
        header_dest: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
            (1,), dtype=CAMERA_FRAME_HEADER_DTYPE, buffer=self._shm.buf, offset=header_offset
        )
        header_dest[0] = header[0]
        header_dest["publish_monotonic_ns"][0] = np.uint64(now_ns)

        # Write RGB bytes
        rgb_offset = header_offset + CAMERA_FRAME_HEADER_DTYPE.itemsize
        rgb_len = rgb.nbytes
        rgb_dest: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
            (rgb_len,), dtype=np.uint8, buffer=self._shm.buf, offset=rgb_offset
        )
        rgb_dest[:] = rgb.ravel()[:rgb_len]

        # Write depth bytes
        depth_offset = rgb_offset + self._max_rgb_bytes
        depth_len = depth.nbytes
        depth_dest: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
            (depth_len,), dtype=np.uint8, buffer=self._shm.buf, offset=depth_offset
        )
        depth_dest[:] = depth.view(np.uint8).ravel()[:depth_len]

        # Write pointcloud bytes (fixed-size block; validity is header-flagged)
        if self._max_pc_bytes > 0 and pointcloud is not None:
            pc_offset = depth_offset + self._max_depth_bytes
            pc_len = pointcloud.nbytes
            pc_dest: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
                (pc_len,), dtype=np.uint8, buffer=self._shm.buf, offset=pc_offset
            )
            pc_dest[:] = pointcloud.view(np.uint8).ravel()[:pc_len]

        # ── Seqlock: write even marker — payload is now consistent ──
        seqlock.end_write(seq)

        self._write_idx_view()[0] = np.uint64(idx)
        return seq

    def read_latest(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int] | None:
        """Read the latest camera frame.

        Returns (header, rgb, depth, pointcloud, sequence) or None if no frame
        available. ``pointcloud`` is a shaped float32 copy (pc_shape), or None
        when the buffer was created without pc_shape. All returned arrays are
        copies.
        """
        idx = int(self._write_idx_view()[0])

        slot_base = self._HEADER_SIZE + idx * self._slot_size

        # Read timestamp + sequence
        seqlock = SeqlockSlot(self._shm.buf, slot_base)
        slot_seq = seqlock.marker

        if slot_seq == 0 and idx == 0 and int(self._write_seq[0]) == 0:
            return None

        # Read header
        header_offset = slot_base + 16
        header: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
            (1,), dtype=CAMERA_FRAME_HEADER_DTYPE, buffer=self._shm.buf, offset=header_offset
        ).copy()

        # Read RGB — validate size and shape against known maxima to guard
        # against torn reads where the producer is mid-write and header fields
        # contain garbage values that would create an out-of-bounds ndarray view
        # or cause reshape to fail with mismatched dimensions.
        h = header[0]
        rgb_size = int(h["rgb_size"])
        rgb_h, rgb_w, rgb_c = int(h["rgb_shape_h"]), int(h["rgb_shape_w"]), int(h["rgb_shape_c"])
        if rgb_size > self._max_rgb_bytes or rgb_size <= 0 or rgb_h * rgb_w * rgb_c != rgb_size:
            now_ns = time.monotonic_ns()
            if now_ns - self._last_torn_warn_ns >= TORN_WARN_INTERVAL_NS:
                self._last_torn_warn_ns = now_ns
                logger.warning(
                    "CameraRingBuffer read_latest: torn or corrupt RGB header "
                    "(rgb_size=%d, shape=%dx%dx%d, max=%d), discarding",
                    rgb_size,
                    rgb_h,
                    rgb_w,
                    rgb_c,
                    self._max_rgb_bytes,
                )
            return None
        rgb_offset = header_offset + CAMERA_FRAME_HEADER_DTYPE.itemsize
        rgb: np.ndarray[Any, np.dtype[np.uint8]] = (
            np.ndarray((rgb_size,), dtype=np.uint8, buffer=self._shm.buf, offset=rgb_offset)
            .copy()
            .reshape((rgb_h, rgb_w, rgb_c))
        )

        # Read depth — same torn-read guard (size + shape consistency)
        depth_size = int(h["depth_size"])
        depth_h, depth_w = int(h["depth_shape_h"]), int(h["depth_shape_w"])
        if depth_size > self._max_depth_bytes or depth_size <= 0 or depth_h * depth_w * 2 != depth_size:
            now_ns = time.monotonic_ns()
            if now_ns - self._last_torn_warn_ns >= TORN_WARN_INTERVAL_NS:
                self._last_torn_warn_ns = now_ns
                logger.warning(
                    "CameraRingBuffer read_latest: torn or corrupt depth header "
                    "(depth_size=%d, shape=%dx%d, max=%d), discarding",
                    depth_size,
                    depth_h,
                    depth_w,
                    self._max_depth_bytes,
                )
            return None
        depth_offset = rgb_offset + self._max_rgb_bytes
        depth = (
            np.ndarray((depth_size,), dtype=np.uint8, buffer=self._shm.buf, offset=depth_offset)
            .copy()
            .view(np.uint16)
            .reshape((depth_h, depth_w))
        )

        # Read pointcloud block — fixed size, so no torn-read size guard needed
        # beyond the seqlock re-check below.
        pointcloud = None
        if self._max_pc_bytes > 0 and self._pc_shape is not None:
            pc_offset = depth_offset + self._max_depth_bytes
            pointcloud = (
                np.ndarray((self._max_pc_bytes,), dtype=np.uint8, buffer=self._shm.buf, offset=pc_offset)
                .copy()
                .view(np.float32)
                .reshape(self._pc_shape)
            )

        # ── Seqlock: reject writer-active or torn reads ──
        # odd seq → writer is mid-write; re-read mismatch → overwritten during read.
        if not seqlock.verify(slot_seq):
            return None

        return header, rgb, depth, pointcloud, _seqlock_to_logical(slot_seq)

    def get_last_metadata(self, k: int) -> list[tuple[np.ndarray, int, int]]:
        """Return up to *k* verified camera headers without copying payloads.

        Results are oldest-first ``(header, publish_monotonic_ns, sequence)``.
        This is the cheap first phase of causal camera selection.
        """
        if k <= 0:
            return []
        if k > self.maxlen:
            raise ValueError(f"k ({k}) exceeds ring capacity maxlen ({self.maxlen})")
        latest_seq = int(self._write_seq[0])
        if latest_seq == 0:
            return []
        result: list[tuple[np.ndarray, int, int]] = []
        for sequence in range(max(1, latest_seq - min(k, latest_seq) + 1), latest_seq + 1):
            idx = sequence % self.maxlen
            slot_base = self._HEADER_SIZE + idx * self._slot_size
            seqlock = SeqlockSlot(self._shm.buf, slot_base)
            marker1 = seqlock.marker
            if not _seqlock_is_complete(marker1) or _seqlock_to_logical(marker1) != sequence:
                continue
            publish_ns = seqlock.timestamp_ns
            header: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
                (1,), dtype=CAMERA_FRAME_HEADER_DTYPE, buffer=self._shm.buf, offset=slot_base + 16
            ).copy()
            if seqlock.verify(marker1):
                result.append((header, publish_ns, sequence))
        return result

    def read_sequence(
        self,
        sequence: int,
        *,
        modalities: tuple[str, ...] = ("rgb", "depth", "pointcloud"),
    ) -> dict[str, np.ndarray] | None:
        """Copy selected payloads for one still-resident verified sequence."""
        allowed = {"rgb", "depth", "pointcloud"}
        unknown = set(modalities) - allowed
        if unknown:
            raise ValueError(f"unknown camera modalities: {sorted(unknown)}")
        if sequence <= 0:
            return None
        idx = sequence % self.maxlen
        slot_base = self._HEADER_SIZE + idx * self._slot_size
        seqlock = SeqlockSlot(self._shm.buf, slot_base)
        marker1 = seqlock.marker
        if not _seqlock_is_complete(marker1) or _seqlock_to_logical(marker1) != sequence:
            return None
        header_offset = slot_base + 16
        header: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
            (1,), dtype=CAMERA_FRAME_HEADER_DTYPE, buffer=self._shm.buf, offset=header_offset
        ).copy()
        h = header[0]
        output: dict[str, np.ndarray] = {"header": header}
        rgb_offset = header_offset + CAMERA_FRAME_HEADER_DTYPE.itemsize
        if "rgb" in modalities:
            rgb_shape = (int(h["rgb_shape_h"]), int(h["rgb_shape_w"]), int(h["rgb_shape_c"]))
            size = int(h["rgb_size"])
            if size <= 0 or size > self._max_rgb_bytes or int(np.prod(rgb_shape)) != size:
                return None
            output["rgb"] = (
                np.ndarray((size,), dtype=np.uint8, buffer=self._shm.buf, offset=rgb_offset).copy().reshape(rgb_shape)
            )
        depth_offset = rgb_offset + self._max_rgb_bytes
        if "depth" in modalities:
            depth_shape = (int(h["depth_shape_h"]), int(h["depth_shape_w"]))
            size = int(h["depth_size"])
            if size <= 0 or size > self._max_depth_bytes or int(np.prod(depth_shape)) * 2 != size:
                return None
            output["depth"] = (
                np.ndarray((size,), dtype=np.uint8, buffer=self._shm.buf, offset=depth_offset)
                .copy()
                .view(np.uint16)
                .reshape(depth_shape)
            )
        if "pointcloud" in modalities and self._max_pc_bytes > 0 and self._pc_shape is not None:
            pc_offset = depth_offset + self._max_depth_bytes
            output["pointcloud"] = (
                np.ndarray((self._max_pc_bytes,), dtype=np.uint8, buffer=self._shm.buf, offset=pc_offset)
                .copy()
                .view(np.float32)
                .reshape(self._pc_shape)
            )
        if not seqlock.verify(marker1):
            return None
        return output

    def frame_age_ns(self) -> int:
        """Return age of the latest frame in nanoseconds, or -1 if no frame."""
        idx = int(self._write_idx_view()[0])
        slot_base = self._HEADER_SIZE + idx * self._slot_size
        seqlock = SeqlockSlot(self._shm.buf, slot_base)
        slot_seq = seqlock.marker
        if slot_seq == 0 and idx == 0 and int(self._write_seq[0]) == 0:
            return -1
        return time.monotonic_ns() - seqlock.timestamp_ns

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

    def _init_header(self) -> None:
        """Zero-initialize the header region and write metadata."""
        header_view: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
            (self._HEADER_SIZE,), dtype=np.uint8, buffer=self._shm.buf, offset=0
        )
        header_view[:] = 0
        np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAX_RGB)[0] = np.uint64(
            self._max_rgb_bytes
        )
        np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAX_DEPTH)[0] = np.uint64(
            self._max_depth_bytes
        )
        np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_MAX_PC)[0] = np.uint64(
            self._max_pc_bytes
        )

    def _write_idx_view(self) -> np.ndarray:
        return np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=self._OFF_WRITE_IDX)
