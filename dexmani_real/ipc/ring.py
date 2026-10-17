"""Single-writer sensor storage passed directly to spawned session workers."""

import logging
import time

import numpy as np


logger = logging.getLogger(__name__)
_TORN_WARN_INTERVAL_NS = 5_000_000_000


class SharedMemoryRingBuffer:
    """Owned reader copies using odd/even publication markers on Linux x86_64.

    NumPy stores provide no portable acquire/release guarantee. RawArray keeps
    storage alive in every process that receives it, without named attachment.
    """

    _HEADER_SIZE = 16

    def __init__(self, ctx, dtype: np.dtype, maxlen: int):
        if type(maxlen) is not int or maxlen <= 0:
            raise ValueError("ring capacity must be a positive integer")
        self.dtype, self.maxlen = np.dtype(dtype), maxlen
        if not self.dtype.names or self.dtype.hasobject:
            raise ValueError("shared-memory payload must have fields without Python objects")
        self._slot_dtype = np.dtype([
            ("timestamp_ns", "<u8"), ("sequence", "<u8"), ("data", self.dtype),
        ], align=True)
        self._buffer = ctx.RawArray("b", self._HEADER_SIZE + maxlen * self._slot_dtype.itemsize)
        self._last_good = None
        self._last_torn_warn_ns = 0

    @property
    def _header(self):
        return np.frombuffer(self._buffer, dtype=np.uint64, count=2)

    @property
    def _slots(self):
        # Construct local views after spawn; pickling an ndarray would copy its data.
        return np.frombuffer(self._buffer, dtype=self._slot_dtype,
                             count=self.maxlen, offset=self._HEADER_SIZE)

    def write(self, **fields) -> int:
        """Validate, then copy fields directly into the next unpublished slot."""
        if fields.keys() != self.dtype.fields.keys():
            raise ValueError("ring fields do not match the payload dtype")
        values = {}
        for name, value in fields.items():
            field_dtype = self.dtype.fields[name][0]
            dtype, shape = field_dtype.subdtype or (field_dtype, ())
            array = np.asarray(value)
            # Exact shape/dtype prevent NumPy from broadcasting or casting missing data.
            if array.dtype != dtype or array.shape != shape:
                raise ValueError(f"ring field {name!r} must be {dtype} {shape}")
            values[name] = array
        header = self._header
        sequence = int(header[1]) + 1
        index = sequence % self.maxlen
        slot = self._slots[index]
        slot["sequence"] = 2 * sequence - 1
        slot["timestamp_ns"] = 0
        for name, value in values.items():
            slot["data"][name] = value
        slot["timestamp_ns"] = time.monotonic_ns()
        slot["sequence"] = 2 * sequence
        header[1] = sequence
        header[0] = index
        return sequence

    def _copy_slot(self, index, expected=None):
        slot = self._slots[index]
        marker = int(slot["sequence"])
        if marker == 0 or marker & 1:
            return None
        sequence = marker // 2
        if expected is not None and sequence != expected:
            return None
        stamp = int(slot["timestamp_ns"])
        data = slot["data"].copy().reshape(1)
        return (data, stamp, sequence) if marker == int(slot["sequence"]) else None

    def read_latest(self, *, use_last_good=False) -> tuple[np.ndarray, int, int] | None:
        """Cached readers retry twice; fallback retains its original source/publication clocks."""
        for _ in range(2 if use_last_good else 1):
            sample = self._copy_slot(int(self._header[0]))
            if sample is not None:
                if use_last_good:
                    self._last_good = sample
                return sample
        if use_last_good and int(self._header[1]) != 0:
            now = time.monotonic_ns()
            if now - self._last_torn_warn_ns >= _TORN_WARN_INTERVAL_NS:
                self._last_torn_warn_ns = now
                logger.warning("sensor ring torn read; returning last verified sample or None")
            return self._last_good
        return None

    @property
    def latest_sequence(self) -> int:
        marker = int(self._slots[int(self._header[0])]["sequence"])
        return marker // 2 if marker and not marker & 1 else 0

    def read_sequence(self, sequence: int) -> tuple[np.ndarray, int, int] | None:
        if sequence <= 0:
            return None
        return self._copy_slot(sequence % self.maxlen, expected=sequence)
