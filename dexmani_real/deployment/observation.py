"""Process-local immutable observation windows for the deployment runtime.

These types never enter RuntimeChannels and therefore carry no
IPC dtype. They are the ``PolicyRuntime`` input contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.ipc.schema import validate_point_cloud_array

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
        "point_cloud",
        "rgb",
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


def freeze_array(
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
    target_dtype = np.dtype(dtype) if dtype is not None else None
    if target_dtype is not None and target_dtype.kind in "iu":
        if raw.dtype.kind not in "iu":
            raise TypeError(f"{name} must contain integers before {target_dtype} cast")
        info = np.iinfo(target_dtype)
        if raw.size and (np.any(raw < info.min) or np.any(raw > info.max)):
            raise ValueError(f"{name} is out of range for {target_dtype}")
    elif target_dtype is not None and target_dtype.kind == "b":
        if raw.dtype.kind != "b":
            raise TypeError(f"{name} must contain booleans before bool cast")
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
        values = freeze_array(self.values, name="FrameWindow.values")
        if values is None:
            raise ValueError("FrameWindow.values must not be None")
        if values.ndim < 2:
            raise ValueError("FrameWindow.values must be [T, ...]")
        t = values.shape[0]
        for name in ("source_sequence", "source_monotonic_ns", "publish_monotonic_ns"):
            arr = freeze_array(
                getattr(self, name), name=f"FrameWindow.{name}", dtype=np.uint64
            )
            if arr is None or arr.shape != (t,):
                raise ValueError(f"FrameWindow.{name} must be a ({t},) uint64 array")
            object.__setattr__(self, name, arr)
        mask = freeze_array(
            self.valid_mask, name="FrameWindow.valid_mask", dtype=np.uint8
        )
        if mask is None or mask.shape != (t,):
            raise ValueError(f"FrameWindow.valid_mask must be a ({t},) uint8 array")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("FrameWindow.valid_mask must be 0 or 1")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", mask)


@dataclass(frozen=True)
class PointCloudFrame:
    """One fixed-size xArm-base point cloud and its causal provenance."""

    values: np.ndarray  # [N, 6] float32
    source_camera_sequence: int
    source_monotonic_ns: int
    publish_monotonic_ns: int
    camera_generation: int

    def __post_init__(self) -> None:
        raw_values = np.asarray(self.values)
        values = validate_point_cloud_array(
            raw_values,
            num_points=raw_values.shape[0] if raw_values.ndim == 2 else 1,
            label="PointCloudFrame.values",
        )
        values = np.array(values, copy=True, order="C")
        values.flags.writeable = False
        provenance = (
            self.source_camera_sequence,
            self.source_monotonic_ns,
            self.publish_monotonic_ns,
            self.camera_generation,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in provenance
        ):
            raise ValueError("PointCloudFrame provenance values must be positive")
        if int(self.source_monotonic_ns) > int(self.publish_monotonic_ns):
            raise ValueError("PointCloudFrame source time cannot exceed publish time")
        object.__setattr__(self, "values", values)
        for name in (
            "source_camera_sequence",
            "source_monotonic_ns",
            "publish_monotonic_ns",
            "camera_generation",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))


@dataclass(frozen=True)
class RgbFrame:
    """One raw RGB camera frame and its causal provenance."""

    values: np.ndarray  # [H, W, 3] uint8, RGB, [0, 255]
    source_camera_sequence: int
    source_monotonic_ns: int
    publish_monotonic_ns: int
    camera_generation: int

    def __post_init__(self) -> None:
        values = freeze_array(self.values, name="RgbFrame.values", dtype=np.uint8)
        if values is None or values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("RgbFrame.values must be uint8 [H, W, 3]")
        provenance = (
            self.source_camera_sequence,
            self.source_monotonic_ns,
            self.publish_monotonic_ns,
            self.camera_generation,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in provenance
        ):
            raise ValueError("RgbFrame provenance values must be positive")
        if int(self.source_monotonic_ns) > int(self.publish_monotonic_ns):
            raise ValueError("RgbFrame source time cannot exceed publish time")
        object.__setattr__(self, "values", values)
        for name in (
            "source_camera_sequence",
            "source_monotonic_ns",
            "publish_monotonic_ns",
            "camera_generation",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))


@dataclass(frozen=True)
class ObservationBatch:
    """One causal observation assembled from state and point-cloud rings.

    Immutable and process-local. ``arm_history``/``hand_history``/
    ``hand_current_history``/``hand_tactile_sum_history``/``tactile_history``
    are ``FrameWindow`` values. ``pointcloud`` is the latest causally valid
    ``PointCloudFrame``. Optional modalities are None when not requested.
    ``anchor_monotonic_ns`` is the causal cut: no frame published after the
    anchor may be included.
    """

    observation_id: int
    run_generation: int
    run_started_monotonic_ns: int
    anchor_monotonic_ns: int
    latest_source_monotonic_ns: int
    logical_step_monotonic_ns: int

    arm_history: FrameWindow | None = None
    hand_history: FrameWindow | None = None
    tactile_history: FrameWindow | None = None
    hand_current_history: FrameWindow | None = None
    hand_tactile_sum_history: FrameWindow | None = None
    pointcloud: PointCloudFrame | None = None
    # Oldest-first causal window of recent point-cloud frames; ``pointcloud`` is
    # the latest (and last element) when non-empty.  ``point_cloud`` models use
    # this window for their per-step point-cloud history.
    pointcloud_history: tuple[PointCloudFrame, ...] = ()
    # Raw camera RGB frames on the same causal control grid as state and, when
    # requested jointly, the point-cloud history.
    rgb_history: tuple[RgbFrame, ...] = ()

    def __post_init__(self) -> None:
        if self.observation_id < 0 or self.run_generation < 0:
            raise ValueError("observation_id and run_generation must be non-negative")
        if (
            min(
                self.run_started_monotonic_ns,
                self.anchor_monotonic_ns,
                self.latest_source_monotonic_ns,
                self.logical_step_monotonic_ns,
            )
            <= 0
        ):
            raise ValueError("observation timestamps must be positive")
        if not (
            self.run_started_monotonic_ns
            <= self.latest_source_monotonic_ns
            <= self.logical_step_monotonic_ns
            <= self.anchor_monotonic_ns
        ):
            raise ValueError(
                "observation time order must be epoch <= source <= logical step <= cut"
            )
        windows = {
            "arm_history": self.arm_history,
            "hand_history": self.hand_history,
            "hand_current_history": self.hand_current_history,
            "hand_tactile_sum_history": self.hand_tactile_sum_history,
            "tactile_history": self.tactile_history,
        }
        for name, window in windows.items():
            if window is None:
                continue
            valid = window.valid_mask == 1
            sources = window.source_monotonic_ns[valid]
            publishes = window.publish_monotonic_ns[valid]
            if np.any(sources < self.run_started_monotonic_ns) or np.any(
                (sources > publishes) | (publishes > self.anchor_monotonic_ns)
            ):
                raise ValueError(f"{name} crosses the observation causal cut")
        history = self.pointcloud_history
        if history:
            if not all(isinstance(frame, PointCloudFrame) for frame in history):
                raise TypeError(
                    "pointcloud_history must contain PointCloudFrame values"
                )
            sources = [frame.source_monotonic_ns for frame in history]
            sequences = [frame.source_camera_sequence for frame in history]
            generations = {frame.camera_generation for frame in history}
            if any(
                frame.source_monotonic_ns < self.run_started_monotonic_ns
                or frame.publish_monotonic_ns > self.anchor_monotonic_ns
                for frame in history
            ):
                raise ValueError(
                    "pointcloud history crosses the observation causal cut"
                )
            if any(right <= left for left, right in zip(sources, sources[1:])):
                raise ValueError("pointcloud source times must be strictly increasing")
            if any(right <= left for left, right in zip(sequences, sequences[1:])):
                raise ValueError(
                    "pointcloud camera sequences must be strictly increasing"
                )
            if len(generations) != 1:
                raise ValueError("pointcloud history crosses a camera generation")
            latest = history[-1]
            if self.pointcloud is None or (
                self.pointcloud.source_camera_sequence,
                self.pointcloud.source_monotonic_ns,
                self.pointcloud.publish_monotonic_ns,
                self.pointcloud.camera_generation,
            ) != (
                latest.source_camera_sequence,
                latest.source_monotonic_ns,
                latest.publish_monotonic_ns,
                latest.camera_generation,
            ):
                raise ValueError("pointcloud must match the last history frame")
            if sources[-1] != self.latest_source_monotonic_ns:
                raise ValueError("latest_source_monotonic_ns must match pointcloud")
            for name, window in windows.items():
                if window is not None and window.values.shape[0] != len(history):
                    raise ValueError(
                        f"{name} must align one-to-one with pointcloud_history"
                    )
        rgb_history = self.rgb_history
        if rgb_history:
            if not all(isinstance(frame, RgbFrame) for frame in rgb_history):
                raise TypeError("rgb_history must contain RgbFrame values")
            sources = [frame.source_monotonic_ns for frame in rgb_history]
            sequences = [frame.source_camera_sequence for frame in rgb_history]
            generations = {frame.camera_generation for frame in rgb_history}
            if any(
                frame.source_monotonic_ns < self.run_started_monotonic_ns
                or frame.publish_monotonic_ns > self.anchor_monotonic_ns
                for frame in rgb_history
            ):
                raise ValueError("rgb history crosses the observation causal cut")
            if any(right <= left for left, right in zip(sources, sources[1:])):
                raise ValueError("rgb source times must be strictly increasing")
            if any(right <= left for left, right in zip(sequences, sequences[1:])):
                raise ValueError("rgb camera sequences must be strictly increasing")
            if len(generations) != 1:
                raise ValueError("rgb history crosses a camera generation")
            if history:
                paired = zip(history, rgb_history, strict=True)
                if len(history) != len(rgb_history) or any(
                    (
                        pointcloud.source_camera_sequence,
                        pointcloud.source_monotonic_ns,
                        pointcloud.camera_generation,
                    )
                    != (
                        rgb.source_camera_sequence,
                        rgb.source_monotonic_ns,
                        rgb.camera_generation,
                    )
                    for pointcloud, rgb in paired
                ):
                    raise ValueError(
                        "rgb history must match pointcloud camera provenance"
                    )
            else:
                if sources[-1] != self.latest_source_monotonic_ns:
                    raise ValueError("latest_source_monotonic_ns must match rgb")
                for name, window in windows.items():
                    if window is not None and window.values.shape[0] != len(
                        rgb_history
                    ):
                        raise ValueError(
                            f"{name} must align one-to-one with rgb_history"
                        )
