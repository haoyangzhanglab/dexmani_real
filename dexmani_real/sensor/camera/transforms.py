"""Pure deterministic RGB-D resize transforms for online and offline use."""

from __future__ import annotations

import numpy as np


def resize_rgb(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize one RGB frame using the fixed downsampling contract."""

    import cv2

    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError(
            f"RGB frame must be uint8 [H,W,3], got {value.shape} {value.dtype}"
        )
    if height <= 0 or width <= 0:
        raise ValueError("target RGB height and width must be positive")
    if value.shape[:2] == (height, width):
        return np.ascontiguousarray(value)
    interpolation = (
        cv2.INTER_AREA
        if height <= value.shape[0] and width <= value.shape[1]
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(value, (width, height), interpolation=interpolation)
    return np.ascontiguousarray(resized, dtype=np.uint8)
