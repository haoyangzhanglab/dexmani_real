"""Lock-free FILO ring buffer in shared memory.

Uses multiprocessing.shared_memory for zero-copy cross-process communication.
Single producer (writer), single consumer (reader) — safe without CAS on
x86_64 because aligned uint64 stores are atomic at the instruction level.

Ref: ManiUniCon lock-free shared memory pattern (main.py:163-170).
     DexUMI drop-oldest backpressure via FILO semantics.

Usage:
    # Producer process
    buf = SharedMemoryRingBuffer("vr_frames", VR_FRAME_DTYPE, maxlen=3, create=True)
    buf.write(vr_frame_array)

    # Consumer process
    buf = SharedMemoryRingBuffer("vr_frames", VR_FRAME_DTYPE, maxlen=3, create=False)
    latest = buf.read_latest()
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np

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


logger = get_logger(__name__)


class SharedMemoryRingBuffer:
    """Lock-free FILO ring buffer in shared memory.

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

        logger.debug(
            "SharedMemoryRingBuffer(name=%s, slot_size=%d, maxlen=%d, total=%d, create=%s)",
            name,
            self._slot_size,
            maxlen,
            self._total_size,
            create,
        )

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

        # Increment sequence
        seq = int(self._write_seq[0]) + 1
        self._write_seq[0] = np.uint64(seq)

        # Compute slot index
        idx = seq % self.maxlen

        # Write to slot
        slot = self._data_buf[idx]
        slot["timestamp_ns"] = np.uint64(now_ns)
        slot["sequence"] = np.uint64(seq)
        slot["data"] = data

        # Atomic write of write_idx (aligned uint64 store on x86_64)
        self._write_idx_view()[0] = np.uint64(idx)

        return seq

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def read_latest(self) -> np.ndarray | None:
        """Read the most recently written frame (consumer-side).

        Returns a copy of the latest data as a numpy array, or None if no
        frame has been written yet.
        """
        idx = int(self._write_idx_view()[0])

        # Check if any frame has been written
        slot = self._data_buf[idx]
        if slot["sequence"] == 0 and idx == 0:
            # Ambiguous: either no writes or exactly one write to slot 0.
            # Disambiguate using the global sequence counter.
            if int(self._write_seq[0]) == 0:
                return None

        # Return a copy of the data (safe for consumer to hold)
        return slot["data"].copy()

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
                    [16:80) CAMERA_FRAME_HEADER_DTYPE (64 bytes)
                    [80:80+max_rgb_bytes) RGB data
                    [80+max_rgb_bytes:...+max_depth_bytes) Depth data
                    [...:...+max_pc_bytes) Pointcloud data (float32)
    """

    _OFF_WRITE_IDX = 0
    _OFF_SEQUENCE = 8
    _OFF_MAX_RGB = 16
    _OFF_MAX_DEPTH = 24
    _OFF_MAX_PC = 32
    _HEADER_SIZE = 64  # cache-line aligned

    # Torn-read warning throttle: ≤1 / 5 s (matching SeqlockRingBuffer).
    _TORN_WARN_INTERVAL_NS = 5 * 1_000_000_000

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
            # Reconstruct shapes from byte counts (used only for logging).
            self._rgb_shape = None
            self._depth_shape = None
            self._pc_shape = None

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

    @classmethod
    def attach(cls, name: str) -> "CameraRingBuffer":
        """Attach to an existing CameraRingBuffer by name (no shape params needed).

        Reads max byte sizes from the existing SHM header, so the attach-mode
        caller does not need to duplicate the shapes used at create time.
        """
        return cls(name=name, create=False)

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
        now_ns = time.monotonic_ns()
        seq = int(self._write_seq[0]) + 1
        self._write_seq[0] = np.uint64(seq)

        idx = seq % self.maxlen
        slot_base = self._HEADER_SIZE + idx * self._slot_size

        # ── Seqlock write protocol: odd→data→even ──
        # Write an odd marker BEFORE the payload (including timestamp) so
        # concurrent readers see "writer active" and bail out.
        ts_arr: np.ndarray[Any, np.dtype[np.uint64]] = np.ndarray(
            (2,), dtype=np.uint64, buffer=self._shm.buf, offset=slot_base
        )
        ts_arr[1] = np.uint64(_seqlock_odd(seq))  # odd: writer active — MUST be first
        ts_arr[0] = np.uint64(now_ns)

        # Write camera header (64 bytes)
        header_offset = slot_base + 16
        header_dest: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
            (1,), dtype=CAMERA_FRAME_HEADER_DTYPE, buffer=self._shm.buf, offset=header_offset
        )
        header_dest[0] = header[0]

        # Write RGB bytes
        rgb_offset = header_offset + CAMERA_FRAME_HEADER_DTYPE.itemsize
        rgb_len = min(rgb.nbytes, self._max_rgb_bytes)
        rgb_dest: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
            (rgb_len,), dtype=np.uint8, buffer=self._shm.buf, offset=rgb_offset
        )
        rgb_dest[:] = rgb.ravel()[:rgb_len]

        # Write depth bytes
        depth_offset = rgb_offset + self._max_rgb_bytes
        depth_len = min(depth.nbytes, self._max_depth_bytes)
        depth_dest: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
            (depth_len,), dtype=np.uint8, buffer=self._shm.buf, offset=depth_offset
        )
        depth_dest[:] = depth.view(np.uint8).ravel()[:depth_len]

        # Write pointcloud bytes (fixed-size block; validity is header-flagged)
        if self._max_pc_bytes > 0 and pointcloud is not None:
            pc_offset = depth_offset + self._max_depth_bytes
            pc_len = min(pointcloud.nbytes, self._max_pc_bytes)
            pc_dest: np.ndarray[Any, np.dtype[np.uint8]] = np.ndarray(
                (pc_len,), dtype=np.uint8, buffer=self._shm.buf, offset=pc_offset
            )
            pc_dest[:] = pointcloud.view(np.uint8).ravel()[:pc_len]

        # ── Seqlock: write even marker — payload is now consistent ──
        ts_arr[1] = np.uint64(_seqlock_even(seq))  # even: writer done

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
        ts_arr: np.ndarray[Any, np.dtype[np.uint64]] = np.ndarray(
            (2,), dtype=np.uint64, buffer=self._shm.buf, offset=slot_base
        )
        slot_seq = int(ts_arr[1])

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
            if now_ns - self._last_torn_warn_ns >= self._TORN_WARN_INTERVAL_NS:
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
            if now_ns - self._last_torn_warn_ns >= self._TORN_WARN_INTERVAL_NS:
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
        if not _seqlock_is_complete(slot_seq):
            return None
        ts_arr_check: np.ndarray[Any, np.dtype[np.uint64]] = np.ndarray(
            (2,), dtype=np.uint64, buffer=self._shm.buf, offset=slot_base
        )
        if int(ts_arr_check[1]) != slot_seq:
            return None

        return header, rgb, depth, pointcloud, _seqlock_to_logical(slot_seq)

    def frame_age_ns(self) -> int:
        """Return age of the latest frame in nanoseconds, or -1 if no frame."""
        idx = int(self._write_idx_view()[0])
        slot_base = self._HEADER_SIZE + idx * self._slot_size
        ts_arr: np.ndarray[Any, np.dtype[np.uint64]] = np.ndarray(
            (2,), dtype=np.uint64, buffer=self._shm.buf, offset=slot_base
        )
        slot_seq = int(ts_arr[1])
        if slot_seq == 0 and idx == 0 and int(self._write_seq[0]) == 0:
            return -1
        return time.monotonic_ns() - int(ts_arr[0])

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


# Camera frame header dtype (moved from layouts.py — Phase 2.7)
CAMERA_FRAME_HEADER_DTYPE = np.dtype(
    [
        ("timestamp", "<f8"),
        ("frame_number", "<u8"),
        ("rgb_size", "<u8"),
        ("depth_size", "<u8"),
        ("rgb_shape_h", "<u4"),
        ("rgb_shape_w", "<u4"),
        ("rgb_shape_c", "<u4"),
        ("depth_shape_h", "<u4"),
        ("depth_shape_w", "<u4"),
        ("pc_num_points", "<u4"),
        ("camera_health", "<u1"),
        ("pad", "<u1", (7,)),
    ],
    align=True,
)
