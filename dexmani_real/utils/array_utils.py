"""Array construction utilities."""

from __future__ import annotations

import numpy as np

__all__ = ["nan_array", "safe_resize"]


def nan_array(shape: int | tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
    """Create an array filled with NaN.

    Centralized factory for the ``np.full(shape, np.nan, dtype=np.float64)``
    pattern repeated across the codebase.  Ensures consistent dtype and NaN fill.
    """
    return np.full(shape, np.nan, dtype=dtype)


def safe_resize(value, expected_size: int) -> np.ndarray:
    """Convert value to float64 array of exactly `expected_size` elements.

    - None → NaN-filled array of `expected_size`.
    - Shorter array → NaN-padded to `expected_size`.
    - Longer array → truncated to first `expected_size` elements.
    """
    if value is None:
        return nan_array(expected_size)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size >= expected_size:
        return arr[:expected_size]
    out = nan_array(expected_size)
    out[: arr.size] = arr
    return out
