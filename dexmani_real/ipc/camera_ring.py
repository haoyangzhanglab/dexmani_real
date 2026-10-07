"""Fixed-session RGB/Z16 layout with the same seqlock assumptions as ring.py.

The initialization header owns dimensions and capacity, including for independent
attach/spawn consumers. Host source time stays in each frame header;
SeqlockSlot separately stamps publication completion. NumPy stores are not a
portable acquire/release memory-order guarantee.
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

from dexmani_real.ipc.ring import SeqlockSlot, seqlock_is_complete, seqlock_to_logical
from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE


class CameraRingBuffer:
    # uint64: write index, logical sequence, RGB H/W, depth H/W, capacity, layout tag.
    _HEADER_SIZE = 64
    _LAYOUT_TAG = 0x43414D3037

    def __init__(self, name, rgb_shape=None, depth_shape=None, maxlen=5, create=True):
        self.name = name
        if create:
            if (
                rgb_shape is None
                or depth_shape is None
                or len(rgb_shape) != 3
                or rgb_shape[2] != 3
                or len(depth_shape) != 2
                or tuple(rgb_shape[:2]) != tuple(depth_shape)
                or any(type(v) is not int or v <= 0 for v in (*rgb_shape, *depth_shape, maxlen))
            ):
                raise ValueError(
                    "camera layout requires positive aligned RGB HWC/Z16 HW and capacity"
                )
            layout = (*rgb_shape[:2], *depth_shape, maxlen, self._LAYOUT_TAG)
            self._set_layout(layout)
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=self._total_size)
            np.ndarray((8,), dtype=np.uint64, buffer=self._shm.buf)[:] = (0, 0, *layout)
        else:
            self._shm = shared_memory.SharedMemory(name=name)
            try:
                layout = tuple(
                    int(v) for v in np.ndarray((8,), dtype=np.uint64, buffer=self._shm.buf)[2:]
                )
                self._set_layout(layout)
                if self._shm.size != self._total_size:
                    raise ValueError(
                        "camera shared memory size does not match initialization layout"
                    )
                if rgb_shape is not None and tuple(rgb_shape) != self._rgb_shape:
                    raise ValueError("attached camera RGB layout differs")
                if depth_shape is not None and tuple(depth_shape) != self._depth_shape:
                    raise ValueError("attached camera depth layout differs")
            except BaseException:
                self._shm.close()
                raise
        self._write_seq = np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf, offset=8)

    def _set_layout(self, layout):
        h, w, dh, dw, capacity, tag = layout
        if tag != self._LAYOUT_TAG or min(h, w, dh, dw, capacity) <= 0 or (h, w) != (dh, dw):
            raise ValueError("invalid fixed camera initialization layout")
        self._rgb_shape, self._depth_shape = (h, w, 3), (dh, dw)
        self.maxlen = capacity
        self._max_rgb_bytes, self._max_depth_bytes = h * w * 3, dh * dw * 2
        self._slot_header_size = 16 + CAMERA_FRAME_HEADER_DTYPE.itemsize
        self._depth_offset = (self._slot_header_size + self._max_rgb_bytes + 1) // 2 * 2
        # Keep uint16 depth and uint64 seqlock markers aligned for odd image sizes too.
        self._slot_size = (self._depth_offset + self._max_depth_bytes + 7) // 8 * 8
        self._total_size = self._HEADER_SIZE + capacity * self._slot_size

    def __getstate__(self):
        return {"name": self.name}

    def __setstate__(self, state):
        self.__init__(state["name"], create=False)

    def _write_idx_view(self):
        return np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf)

    def write(self, header, rgb, depth):
        if (
            not isinstance(header, np.ndarray)
            or header.shape != (1,)
            or header.dtype != CAMERA_FRAME_HEADER_DTYPE
        ):
            raise ValueError("camera header must match CAMERA_FRAME_HEADER_DTYPE (1,)")
        for array, shape, dtype in (
            (rgb, self._rgb_shape, np.uint8),
            (depth, self._depth_shape, np.uint16),
        ):
            if (
                not isinstance(array, np.ndarray)
                or array.shape != shape
                or array.dtype != dtype
                or not array.flags.c_contiguous
            ):
                raise ValueError("camera payload must match fixed shape/dtype and be C-contiguous")
        seq = int(self._write_seq[0]) + 1
        idx = seq % self.maxlen
        base = self._HEADER_SIZE + idx * self._slot_size
        seqlock = SeqlockSlot(self._shm.buf, base)
        seqlock.begin_write(seq, 0)
        np.ndarray((1,), dtype=CAMERA_FRAME_HEADER_DTYPE, buffer=self._shm.buf, offset=base + 16)[
            :
        ] = header
        offset = base + self._slot_header_size
        np.ndarray(self._rgb_shape, dtype=np.uint8, buffer=self._shm.buf, offset=offset)[:] = rgb
        np.ndarray(
            self._depth_shape,
            dtype=np.uint16,
            buffer=self._shm.buf,
            offset=base + self._depth_offset,
        )[:] = depth
        seqlock.stamp_timestamp(time.monotonic_ns())
        seqlock.end_write(seq)
        self._write_seq[0] = np.uint64(seq)
        self._write_idx_view()[0] = np.uint64(idx)
        return seq

    def _copy_slot(self, idx, modalities, expected=None):
        if not 0 <= idx < self.maxlen:
            return None
        base = self._HEADER_SIZE + idx * self._slot_size
        seqlock = SeqlockSlot(self._shm.buf, base)
        marker = seqlock.marker
        if not seqlock_is_complete(marker):
            return None
        sequence = seqlock_to_logical(marker)
        if expected is not None and sequence != expected:
            return None
        output = {
            "header": np.ndarray(
                (1,), dtype=CAMERA_FRAME_HEADER_DTYPE, buffer=self._shm.buf, offset=base + 16
            ).copy()
        }
        offset = base + self._slot_header_size
        if "rgb" in modalities:
            output["rgb"] = np.ndarray(
                self._rgb_shape, dtype=np.uint8, buffer=self._shm.buf, offset=offset
            ).copy()
        if "depth" in modalities:
            output["depth"] = np.ndarray(
                self._depth_shape,
                dtype=np.uint16,
                buffer=self._shm.buf,
                offset=base + self._depth_offset,
            ).copy()
        return (output, sequence) if seqlock.verify(marker) else None

    def read_latest(self):
        result = self._copy_slot(int(self._write_idx_view()[0]), ("rgb", "depth"))
        if result is None:
            return None
        data, sequence = result
        return data["header"], data["rgb"], data["depth"], sequence

    @property
    def latest_sequence(self):
        idx = int(self._write_idx_view()[0])
        if not 0 <= idx < self.maxlen:
            return 0
        marker = SeqlockSlot(self._shm.buf, self._HEADER_SIZE + idx * self._slot_size).marker
        return seqlock_to_logical(marker) if seqlock_is_complete(marker) else 0

    def read_sequence(self, sequence, *, modalities=("rgb", "depth")):
        if set(modalities) - {"rgb", "depth"}:
            raise ValueError(f"unknown camera modalities: {modalities}")
        if sequence <= 0:
            return None
        result = self._copy_slot(sequence % self.maxlen, modalities, expected=sequence)
        return result[0] if result is not None else None

    def close(self):
        self._shm.close()

    def unlink(self):
        self._shm.unlink()
