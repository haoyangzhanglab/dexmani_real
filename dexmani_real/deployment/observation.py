"""Process-local immutable observation windows for the deployment runtime.

These types never enter SharedStorage and therefore carry no
IPC dtype. They are the ``ObservationAdapter`` input contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


POLICY_OBSERVATION_FIELDS = frozenset(
    {
        "arm_qpos",
        "hand_qpos",
        "hand_current",
        "hand_tactile_sum",
        "hand_tactile_force",
        "arm_joint_position",
        "hand_joint_position",
        "hand_joint_torque",
        "fingertip_force",
        "xhand_tactile",
    }
)


def parse_observation_fields(spec: str) -> tuple[str, ...]:
    """Parse and validate the comma-separated policy observation contract."""
    if not isinstance(spec, str):
        raise TypeError("observation_fields must be a comma-separated string")
    fields = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not fields:
        raise ValueError("observation_fields must contain at least one field")
    unknown = sorted(set(fields) - POLICY_OBSERVATION_FIELDS)
    if unknown:
        raise ValueError(f"unknown observation field(s): {', '.join(unknown)}")
    if len(set(fields)) != len(fields):
        raise ValueError("observation_fields must not contain duplicates")
    return fields


def _freeze(
    arr: Any,
    *,
    name: str,
    dtype: Any = None,
) -> np.ndarray | None:
    """Return a read-only C-order copy of *arr* (or None), validating finiteness.

    Finiteness is checked on the raw input (before any integer dtype cast) so a
    NaN timestamp cannot silently collapse to 0/INT_MAX.
    """
    if arr is None:
        return None
    raw = np.asarray(arr)
    if raw.dtype.kind in "fc" and not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} contains NaN/Inf")
    out = np.array(raw, dtype=dtype, copy=True, order="C")
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class FrameWindow:
    """Oldest-first window of frames from one ring + aligned per-frame metadata.

    ``values`` is the feature tensor with leading axis = number of frames ``T``
    (arm ``[T,7]``, hand ``[T,12]``, tactile-sum ``[T,5,3]``). The metadata
    arrays are all ``[T]`` and aligned to that same axis. ``valid_mask[i] == 0``
    marks a padding slot whose frame must be ignored by consumers.
    """

    values: np.ndarray
    source_sequence: np.ndarray
    source_monotonic_ns: np.ndarray
    publish_monotonic_ns: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        values = _freeze(self.values, name="FrameWindow.values")
        if values.ndim < 2:
            raise ValueError("FrameWindow.values must be [T, ...]")
        t = values.shape[0]
        for name in ("source_sequence", "source_monotonic_ns", "publish_monotonic_ns"):
            arr = _freeze(getattr(self, name), name=f"FrameWindow.{name}", dtype=np.uint64)
            if arr is None or arr.shape != (t,):
                raise ValueError(f"FrameWindow.{name} must be a ({t},) uint64 array")
            object.__setattr__(self, name, arr)
        mask = _freeze(self.valid_mask, name="FrameWindow.valid_mask", dtype=np.uint8)
        if mask is None or mask.shape != (t,):
            raise ValueError(f"FrameWindow.valid_mask must be a ({t},) uint8 array")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("FrameWindow.valid_mask must be 0 or 1")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", mask)


@dataclass(frozen=True)
class CameraWindow:
    """Oldest-first camera frame window (rgb/depth/pointcloud) + metadata.

    Camera frames add a ``receive_monotonic_ns`` layer between source and
    publish (``source <= receive <= publish``). Sub-modality arrays share the
    leading ``T`` axis; at least one sub-modality must be present.
    """

    rgb: np.ndarray | None = None  # [T, H, W, 3] uint8
    depth: np.ndarray | None = None  # [T, H, W] uint16
    pointcloud: np.ndarray | None = None  # [T, N, 3] float32
    source_sequence: np.ndarray | None = None
    source_monotonic_ns: np.ndarray | None = None
    receive_monotonic_ns: np.ndarray | None = None
    publish_monotonic_ns: np.ndarray | None = None
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        mods = [m for m in ("rgb", "depth", "pointcloud") if getattr(self, m) is not None]
        if not mods:
            raise ValueError("CameraWindow requires at least one of rgb/depth/pointcloud")
        t = np.asarray(getattr(self, mods[0])).shape[0]
        for m in mods:
            arr = _freeze(getattr(self, m), name=f"CameraWindow.{m}")
            if arr.ndim < 2 or arr.shape[0] != t:
                raise ValueError(f"CameraWindow.{m} must be [T, ...] with leading axis {t}")
            object.__setattr__(self, m, arr)
        for name in ("source_sequence", "source_monotonic_ns", "receive_monotonic_ns", "publish_monotonic_ns"):
            arr = _freeze(getattr(self, name), name=f"CameraWindow.{name}", dtype=np.uint64)
            if arr is None or arr.shape != (t,):
                raise ValueError(f"CameraWindow.{name} must be a ({t},) uint64 array")
            object.__setattr__(self, name, arr)
        mask = _freeze(self.valid_mask, name="CameraWindow.valid_mask", dtype=np.uint8)
        if mask is None or mask.shape != (t,):
            raise ValueError(f"CameraWindow.valid_mask must be a ({t},) uint8 array")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("CameraWindow.valid_mask must be 0 or 1")
        object.__setattr__(self, "valid_mask", mask)


@dataclass(frozen=True)
class ObservationBatch:
    """One causal observation assembled from the arm/hand/tactile/camera rings.

    Immutable and process-local. ``arm_history``/``hand_history``/
    ``hand_current_history``/``hand_tactile_sum_history``/``tactile_history``/
    ``camera_history`` are ``FrameWindow``/``CameraWindow`` (or None when the
    modality is absent or not required by the adapter); each carries its own
    source/publish metadata because the rings advance independently.
    ``anchor_monotonic_ns`` is the causal cut: no frame published after the
    anchor may be included.
    """

    observation_id: int
    run_generation: int
    anchor_monotonic_ns: int

    arm_history: FrameWindow | None = None
    hand_history: FrameWindow | None = None
    tactile_history: FrameWindow | None = None
    camera_history: CameraWindow | None = None
    # Appended after the original fields to preserve positional construction
    # compatibility for existing adapters and offline callers.
    hand_current_history: FrameWindow | None = None
    hand_tactile_sum_history: FrameWindow | None = None

    def __post_init__(self) -> None:
        if self.observation_id < 0 or self.run_generation < 0:
            raise ValueError("observation_id and run_generation must be non-negative")
        if self.anchor_monotonic_ns <= 0:
            raise ValueError("anchor_monotonic_ns must be positive")
