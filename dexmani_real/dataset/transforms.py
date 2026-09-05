"""Deterministic numerical transforms for the processed episode view."""

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


def resize_depth(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize native RealSense Z16 depth without inventing intermediate values."""

    import cv2

    value = np.asarray(frame)
    if value.ndim != 2 or value.dtype != np.uint16:
        raise ValueError(
            f"depth frame must be uint16 [H,W], got {value.shape} {value.dtype}"
        )
    if height <= 0 or width <= 0:
        raise ValueError("target depth height and width must be positive")
    if value.shape == (height, width):
        return np.ascontiguousarray(value)
    resized = cv2.resize(value, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(resized, dtype=np.uint16)


def resize_camera_intrinsic(
    camera_k: np.ndarray,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Scale a row-major pinhole K for a resize with no crop."""

    value = np.asarray(camera_k, dtype=np.float64)
    if value.shape == (9,):
        value = value.reshape(3, 3)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(
            "camera_intrinsic must be a finite 3x3 matrix or length-9 vector"
        )
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError(
            "camera intrinsic source and target dimensions must be positive"
        )
    if not np.allclose(value[2], np.array([0.0, 0.0, 1.0]), rtol=0.0, atol=1e-8):
        raise ValueError(
            "camera_intrinsic must have the canonical pinhole last row [0,0,1]"
        )
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    result = value.copy()
    result[0, 0] *= scale_x
    result[0, 2] *= scale_x
    result[1, 1] *= scale_y
    result[1, 2] *= scale_y
    return result.astype(np.float32).reshape(9)
