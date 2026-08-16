"""Timestamp-aligned recording buffer for multi-rate sensor streams.

Assigns each incoming data point to a fixed-dt grid segment, supports explicit
wall-time re-anchors between unsampled segments, pre-allocates NumPy arrays,
and flushes aligned data in bulk at episode stop.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

__all__ = ["BufferAddResult", "FillReason", "TimestampAlignedBuffer"]

_INTERNAL_DATASET_NAMES = frozenset(
    {"flag_sample_valid", "source_sample_index", "source_timestamp", "fill_reason"}
)


class FillReason(IntEnum):
    SOURCE = 0
    CAUSAL_HOLD_LAST = 1
    LEADING_PLACEHOLDER = 2


@dataclass(frozen=True)
class BufferAddResult:
    """Exact storage effect of one :meth:`TimestampAlignedBuffer.add` call."""

    previous_size: int
    size: int
    source_written: bool
    capacity_reached: bool

    @property
    def slots_written(self) -> int:
        return self.size - self.previous_size


def _get_accumulate_timestamp_idxs(
    timestamps: list[float] | float,
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: int = 0,
) -> tuple[list[int], list[int], int]:
    """Map raw timestamps to discrete grid indices.

    Assigns each source to the first grid deadline that is not earlier than its
    timestamp. Exact-boundary samples stay on that boundary; later samples in
    the same interval are dropped (first wins). Missing deadlines are never
    associated with a future source sample; :class:`TimestampAlignedBuffer`
    fills them from a past source or an explicit leading placeholder.

    Args:
        timestamps: Sorted list of timestamps (or a single float).
        start_time: Reference start time for the grid.
        dt: Grid interval in seconds.
        eps: Small epsilon to avoid floating-point boundary issues.
        next_global_idx: The next expected global index (carried across calls).

    Returns:
        local_idxs: Indices into ``timestamps`` to use for each assigned slot.
        global_idxs: The global grid index for each assigned slot.
        next_global_idx: Updated value for the next call.
    """
    if isinstance(timestamps, float):
        timestamps = [timestamps]

    local_idxs: list[int] = []
    global_idxs: list[int] = []

    for local_idx, ts in enumerate(timestamps):
        if not np.isfinite(ts):
            continue
        # The grid is a causal deadline, not the left edge of an arrival-time
        # bucket. eps keeps floating-point representations of exact boundaries
        # on that boundary while every truly later source moves to the next one.
        global_idx = math.ceil((ts - start_time) / dt - eps)
        if global_idx < 0:
            continue

        if global_idx < next_global_idx:
            continue
        local_idxs.append(local_idx)
        global_idxs.append(global_idx)
        next_global_idx = global_idx + 1

    return local_idxs, global_idxs, next_global_idx


class TimestampAlignedBuffer:
    """Pre-allocated buffer that aligns streaming data to a fixed-dt time grid.

    Within one segment, data is assigned to slots ``start_time + k*dt``. The
    caller may re-anchor the next global index after an intentional unsampled
    interval; prior slots and timestamps remain unchanged. The first call to
    ``add()`` inspects the data dict to determine shapes and dtypes, then
    pre-allocates numpy arrays of shape ``(max_record_steps,) + value.shape``.

    When the data source is slower than 1/dt, missed slots causally hold the
    latest *past* sample. Episode-leading slots with no past source use an
    explicit zero/NaN placeholder. When faster, the first value before each
    deadline wins (later values assigned to the same deadline are dropped).

    Camera frames should NOT be routed through this buffer (they are too large
    for pre-allocation); use the existing per-frame HDF5 path instead.
    """

    def __init__(
        self,
        start_time: float,
        dt: float,
        max_record_steps: int,
        eps: float = 1e-5,
    ) -> None:
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.max_record_steps = max_record_steps

        self._data_buffer: dict[str, np.ndarray] | None = None
        self._timestamp_buffer: np.ndarray | None = None
        self._size: int = 0
        self._recording_stopped: bool = False
        self._next_global_idx: int = 0
        self._next_source_index: int = 0
        self._last_source_data: dict[str, Any] | None = None
        self._last_source_index: int = -1
        self._last_source_timestamp: float = float("nan")

    # -- public properties ----------------------------------------------------

    @property
    def data(self) -> dict[str, np.ndarray]:
        """Aligned data arrays trimmed to actual used size."""
        if self._timestamp_buffer is None:
            return {}
        return {key: arr[: self._size] for key, arr in self._data_buffer.items()}  # type: ignore[union-attr]

    @property
    def timestamps(self) -> np.ndarray:
        """Aligned timestamp array trimmed to actual used size."""
        if self._timestamp_buffer is None:
            return np.array([], dtype=np.float64)
        return self._timestamp_buffer[: self._size]

    @property
    def size(self) -> int:
        """Number of occupied grid slots."""
        return self._size

    def __len__(self) -> int:
        return self._size

    # -- core methods ---------------------------------------------------------

    def reanchor(self, next_source_timestamp: float) -> None:
        """Start the next contiguous storage slot at a new wall-time anchor.

        Command-silent pauses intentionally contain no recording samples. A
        controller generation change therefore re-anchors the remaining grid
        instead of representing the pause as a run of causal hold-last slots.
        Existing timestamps are never rewritten.
        """
        timestamp = float(next_source_timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("re-anchor timestamp must be finite")
        if self._recording_stopped:
            raise RuntimeError("cannot re-anchor a full timestamp buffer")
        if self._size > 0 and self._timestamp_buffer is not None:
            previous_timestamp = float(self._timestamp_buffer[self._size - 1])
            if timestamp <= previous_timestamp:
                raise ValueError("re-anchor timestamp must be newer than the previous stored slot")
        self.start_time = timestamp - self._next_global_idx * self.dt

    def add(
        self,
        data: Mapping[str, np.ndarray | np.generic | float | int],
        timestamp: float,
    ) -> BufferAddResult:
        """Add one multi-stream data point at the first causal grid deadline.

        On the first call the buffer allocates arrays based on the keys, shapes,
        and dtypes found in *data*.  Subsequent calls must supply the same keys.
        """
        previous_size = self._size
        if self._recording_stopped:
            return BufferAddResult(previous_size, self._size, False, True)

        # Validate before advancing source/grid indices so a malformed frame
        # cannot partially mutate buffer state. The first accepted layout is
        # the contract for every subsequent call.
        self._validate_input_layout(data)

        source_index = self._next_source_index
        self._next_source_index += 1
        previous_next_idx = self._next_global_idx
        _, global_idxs, next_global_idx = _get_accumulate_timestamp_idxs(
            timestamps=timestamp,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self._next_global_idx,
        )
        self._next_global_idx = next_global_idx

        if len(global_idxs) == 0:
            return BufferAddResult(previous_size, self._size, False, self._recording_stopped)

        source_global_idx = global_idxs[0]
        gap_idxs = list(range(previous_next_idx, source_global_idx))
        write_idxs = gap_idxs + [source_global_idx]

        # Check capacity — truncate to valid slots if last index overflows.
        max_required = source_global_idx + 1
        if max_required > self.max_record_steps:
            # Keep only the indices that fit within the pre-allocated arrays.
            write_idxs = [g for g in write_idxs if g < self.max_record_steps]
            if not write_idxs:
                self._recording_stopped = True
                return BufferAddResult(previous_size, self._size, False, True)
            n_dropped = max_required - self.max_record_steps
            max_required = write_idxs[-1] + 1
            logger.warning(
                "TimestampAlignedBuffer: reached max_record_steps=%d — "
                "truncated %d/%d slots, stopping after this batch",
                self.max_record_steps,
                n_dropped,
                n_dropped + len(write_idxs),
            )
            self._recording_stopped = True

        # Lazy allocation on first call
        if self._data_buffer is None:
            self._allocate(data)

        assert self._data_buffer is not None
        # Fill gaps only from an already observed source. No future sample may
        # write a slot whose grid time precedes that sample's source time.
        for grid_idx in gap_idxs:
            if grid_idx >= self.max_record_steps:
                break
            if self._last_source_data is not None:
                for key, value in self._last_source_data.items():
                    if key in self._data_buffer:
                        self._data_buffer[key][grid_idx] = value
                self._data_buffer["source_sample_index"][grid_idx] = self._last_source_index
                self._data_buffer["source_timestamp"][grid_idx] = self._last_source_timestamp
                self._data_buffer["fill_reason"][grid_idx] = int(FillReason.CAUSAL_HOLD_LAST)
            else:
                self._data_buffer["source_sample_index"][grid_idx] = -1
                self._data_buffer["source_timestamp"][grid_idx] = np.nan
                self._data_buffer["fill_reason"][grid_idx] = int(FillReason.LEADING_PLACEHOLDER)

        if source_global_idx < self.max_record_steps:
            for key, value in data.items():
                if key in self._data_buffer:
                    self._data_buffer[key][source_global_idx] = value
            self._data_buffer["flag_sample_valid"][source_global_idx] = True
            self._data_buffer["source_sample_index"][source_global_idx] = source_index
            self._data_buffer["source_timestamp"][source_global_idx] = timestamp
            self._data_buffer["fill_reason"][source_global_idx] = int(FillReason.SOURCE)
            self._last_source_data = {
                key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
                for key, value in data.items()
            }
            self._last_source_index = source_index
            self._last_source_timestamp = timestamp

        if self._timestamp_buffer is not None:
            # Assign grid-aligned synthetic timestamps so every slot gets
            # a unique, strictly-monotonic value even when the data source
            # back-fills or stalls.  Without this, back-filled slots all
            # share the original timestamp → duplicate timestamps in the
            # HDF5 → broken temporal alignment for downstream training.
            for gidx in write_idxs:
                self._timestamp_buffer[gidx] = self.start_time + gidx * self.dt

        self._size = max_required
        return BufferAddResult(
            previous_size=previous_size,
            size=self._size,
            source_written=source_global_idx < self.max_record_steps,
            capacity_reached=self._recording_stopped or self._size >= self.max_record_steps,
        )

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _field_layout(key: str, value: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
        if isinstance(value, np.ndarray):
            return value.shape, value.dtype
        if isinstance(value, (float, int, np.generic)):
            scalar = np.asarray(value)
            return scalar.shape, scalar.dtype
        raise TypeError(
            f"TimestampAlignedBuffer field {key!r} must be a numpy array or numeric scalar, "
            f"got {type(value).__name__}"
        )

    def _validate_input_layout(self, data: Mapping[str, Any]) -> None:
        """Require stable source keys, shapes, and dtypes across ``add`` calls."""

        reserved = sorted(set(data) & _INTERNAL_DATASET_NAMES)
        if reserved:
            raise ValueError(
                f"TimestampAlignedBuffer input collides with internal fields: {reserved}"
            )

        actual_layout = {
            key: self._field_layout(key, value) for key, value in data.items()
        }
        if self._data_buffer is None:
            return

        expected_keys = set(self._data_buffer) - _INTERNAL_DATASET_NAMES
        actual_keys = set(actual_layout)
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        if missing or unexpected:
            raise ValueError(
                "TimestampAlignedBuffer field keys differ from the initial layout: "
                f"missing={missing}, unexpected={unexpected}"
            )

        for key in sorted(expected_keys):
            actual_shape, actual_dtype = actual_layout[key]
            target = self._data_buffer[key]
            expected_shape = target.shape[1:]
            expected_dtype = target.dtype
            if actual_shape != expected_shape:
                raise ValueError(
                    f"TimestampAlignedBuffer field {key!r} shape {actual_shape} "
                    f"does not match initial shape {expected_shape}"
                )
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"TimestampAlignedBuffer field {key!r} dtype {actual_dtype} "
                    f"does not match initial dtype {expected_dtype}"
                )

    def _allocate(self, data: Mapping[str, Any]) -> None:
        """Pre-allocate numpy arrays based on the shape/dtype of each field."""
        self._data_buffer = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                shape = (self.max_record_steps,) + value.shape
                if np.issubdtype(value.dtype, np.floating):
                    self._data_buffer[key] = np.full(shape, np.nan, dtype=value.dtype)
                else:
                    self._data_buffer[key] = np.zeros(shape, dtype=value.dtype)
            elif isinstance(value, (float, int, np.generic)):
                scalar = np.asarray(value)
                if np.issubdtype(scalar.dtype, np.floating):
                    self._data_buffer[key] = np.full(
                        self.max_record_steps, np.nan, dtype=scalar.dtype
                    )
                else:
                    self._data_buffer[key] = np.zeros(
                        self.max_record_steps, dtype=scalar.dtype
                    )
            else:
                # ``_validate_input_layout`` runs immediately before allocation.
                raise TypeError(
                    f"TimestampAlignedBuffer field {key!r} has unsupported type "
                    f"{type(value).__name__}"
                )

        self._data_buffer["flag_sample_valid"] = np.zeros(self.max_record_steps, dtype=bool)
        self._data_buffer["source_sample_index"] = np.full(self.max_record_steps, -1, dtype=np.int64)
        self._data_buffer["source_timestamp"] = np.full(self.max_record_steps, np.nan, dtype=np.float64)
        self._data_buffer["fill_reason"] = np.full(
            self.max_record_steps, int(FillReason.LEADING_PLACEHOLDER), dtype=np.uint8
        )
        self._timestamp_buffer = np.zeros(self.max_record_steps, dtype=np.float64)
