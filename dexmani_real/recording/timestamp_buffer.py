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

__all__ = ["TimestampAlignedBuffer", "get_accumulate_timestamp_idxs"]


def get_accumulate_timestamp_idxs(
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

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats

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

        _, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamp,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self._next_global_idx,
        )
        self._next_global_idx = next_global_idx

        if len(global_idxs) == 0:
            return

        # Check capacity
        max_required = global_idxs[-1] + 1
        if max_required > self.max_record_steps:
            logger.warning(
                "TimestampAlignedBuffer: reached max_record_steps=%d — stopping",
                self.max_record_steps,
            )
            self._recording_stopped = True
            return

        # Lazy allocation on first call
        if self._data_buffer is None:
            self._allocate(data)

        # Write data into pre-allocated slots
        for key, value in data.items():
            if key in self._data_buffer:  # type: ignore[operator]
                self._data_buffer[key][global_idxs] = value  # type: ignore[index]

        if self._timestamp_buffer is not None:
            self._timestamp_buffer[global_idxs] = timestamp

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

        self._timestamp_buffer = np.zeros(self.max_record_steps, dtype=np.float64)
