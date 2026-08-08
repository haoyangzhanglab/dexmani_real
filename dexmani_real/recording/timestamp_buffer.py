"""Timestamp-aligned recording buffer for multi-rate sensor streams.

Assigns each incoming data point to a fixed dt time grid (start_time + k*dt),
pre-allocates numpy arrays, and flushes aligned data in bulk at episode stop.

Ref: ManiUniCon TimestampAlignedBuffer (maniunicon/utils/timestamp_accumulator.py).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

__all__ = ["TimestampAlignedBuffer"]


def _get_accumulate_timestamp_idxs(
    timestamps: list[float] | float,
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: int = 0,
) -> tuple[list[int], list[int], int]:
    """Map raw timestamps to discrete grid indices.

    For each dt window [start_time + k*dt, start_time + (k+1)*dt), assigns the
    first timestamp falling into that window (later samples in the same window
    are dropped — first wins).  When a window has no timestamp (dropped frame),
    it is back-filled by the NEXT arriving sample's index.

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
        # eps*dt ensures ts == start_time + k*dt lands exactly in slot k
        global_idx = math.floor((ts - start_time) / dt + eps)
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

    Data is assigned to discrete slots ``start_time + k*dt``.  The first call to
    ``add()`` inspects the data dict to determine shapes and dtypes, then
    pre-allocates numpy arrays of shape ``(max_record_steps,) + value.shape``.

    When the data source is slower than 1/dt, missed slots are back-filled by
    the next arriving sample.  When faster, the first value in each window wins
    (later ones are dropped).

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

    @property
    def recording_stopped(self) -> bool:
        """True when max_record_steps has been reached."""
        return self._recording_stopped

    def __len__(self) -> int:
        return self._size

    # -- core methods ---------------------------------------------------------

    def add(self, data: dict[str, np.ndarray | float | int], timestamp: float) -> None:
        """Add one multi-stream data point, assigning it to the nearest grid slot.

        On the first call the buffer allocates arrays based on the keys, shapes,
        and dtypes found in *data*.  Subsequent calls must supply the same keys.
        """
        if self._recording_stopped:
            return

        _, global_idxs, next_global_idx = _get_accumulate_timestamp_idxs(
            timestamps=timestamp,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self._next_global_idx,
        )
        self._next_global_idx = next_global_idx

        if len(global_idxs) == 0:
            return

        # Check capacity — truncate to valid slots if last index overflows.
        max_required = global_idxs[-1] + 1
        if max_required > self.max_record_steps:
            # Keep only the indices that fit within the pre-allocated arrays.
            keep = [i for i, g in enumerate(global_idxs) if g < self.max_record_steps]
            if not keep:
                self._recording_stopped = True
                return
            n_dropped = len(global_idxs) - len(keep)
            global_idxs = [global_idxs[i] for i in keep]
            max_required = global_idxs[-1] + 1
            logger.warning(
                "TimestampAlignedBuffer: reached max_record_steps=%d — "
                "truncated %d/%d slots, stopping after this batch",
                self.max_record_steps,
                n_dropped,
                n_dropped + len(keep),
            )
            self._recording_stopped = True

        # Lazy allocation on first call
        if self._data_buffer is None:
            self._allocate(data)

        # Write data into pre-allocated slots
        for key, value in data.items():
            if key in self._data_buffer:  # type: ignore[operator]
                self._data_buffer[key][global_idxs] = value  # type: ignore[index]
        self._data_buffer["flag_sample_valid"][global_idxs] = True  # type: ignore[index]

        if self._timestamp_buffer is not None:
            # Assign grid-aligned synthetic timestamps so every slot gets
            # a unique, strictly-monotonic value even when the data source
            # back-fills or stalls.  Without this, back-filled slots all
            # share the original timestamp → duplicate timestamps in the
            # HDF5 → broken temporal alignment for downstream training.
            for gidx in global_idxs:
                self._timestamp_buffer[gidx] = self.start_time + gidx * self.dt

        self._size = max_required

    # -- internal helpers -----------------------------------------------------

    def _allocate(self, data: dict[str, Any]) -> None:
        """Pre-allocate numpy arrays based on the shape/dtype of each field."""
        self._data_buffer = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                shape = (self.max_record_steps,) + value.shape
                self._data_buffer[key] = np.zeros(shape, dtype=value.dtype)
            elif isinstance(value, (float, int)):
                self._data_buffer[key] = np.zeros(self.max_record_steps, dtype=type(value))
            else:
                logger.warning(
                    "TimestampAlignedBuffer: skipping key=%r (unsupported type %s)",
                    key,
                    type(value).__name__,
                )

        self._data_buffer["flag_sample_valid"] = np.zeros(self.max_record_steps, dtype=bool)
        self._timestamp_buffer = np.zeros(self.max_record_steps, dtype=np.float64)
