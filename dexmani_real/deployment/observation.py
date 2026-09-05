"""Process-local observation windows for the deployment runtime.

These types never enter RuntimeChannels and therefore carry no
IPC dtype. They are the ``PolicyRuntime`` input contract.

Shared-memory readers already take ownership copies.  These containers validate
those process-local arrays without copying their payloads again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from dexmani_real.ipc.schema import validate_point_cloud_array

_POLICY_MODALITIES = frozenset(
    {"joint_state", "point_cloud", "rgb", "contact_force", "fingertip_points"}
)


@dataclass(frozen=True)
class PolicyObservation:
    """Narrow NumPy boundary passed to a Policy runtime.

    Mapping insertion order is the validated Policy modality order. Arrays are
    C-contiguous, writeable, inference-process-owned model inputs.
    """

    observation_id: int
    run_generation: int
    anchor_monotonic_ns: int
    latest_source_monotonic_ns: int
    logical_step_monotonic_ns: int
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if type(self.observation_id) is not int or self.observation_id <= 0:
            raise ValueError("observation_id must be a positive integer")
        if type(self.run_generation) is not int or self.run_generation < 0:
            raise ValueError("run_generation must be a non-negative integer")
        for name in (
            "anchor_monotonic_ns",
            "latest_source_monotonic_ns",
            "logical_step_monotonic_ns",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if not (
            0
            < self.latest_source_monotonic_ns
            <= self.logical_step_monotonic_ns
            <= self.anchor_monotonic_ns
        ):
            raise ValueError("PolicyObservation timestamps are inconsistent")
        if not isinstance(self.arrays, Mapping):
            raise TypeError("PolicyObservation arrays must be a mapping")
        modalities = tuple(self.arrays)
        if (
            not modalities
            or len(set(modalities)) != len(modalities)
            or not set(modalities) <= _POLICY_MODALITIES
            or "joint_state" not in modalities
        ):
            raise ValueError("PolicyObservation modalities are invalid")
        horizon: int | None = None
        for name in modalities:
            arr = np.asarray(self.arrays[name])
            expected_dtype = np.uint8 if name == "rgb" else np.float32
            if arr.dtype != np.dtype(expected_dtype):
                raise TypeError(f"{name} must have dtype {np.dtype(expected_dtype)}")
            _validate_finite(arr, name=f"PolicyObservation.{name}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            if not arr.flags.writeable:
                raise ValueError(f"{name} must be writeable")
            expected_tail = {
                "joint_state": (19,),
                "contact_force": (5, 3),
                "fingertip_points": (5, 3),
            }.get(name)
            if arr.ndim < 2 or (
                expected_tail is not None and arr.shape[1:] != expected_tail
            ):
                raise ValueError(f"{name} has invalid shape {arr.shape}")
            if name == "point_cloud" and (arr.ndim != 3 or arr.shape[2] != 6):
                raise ValueError("point_cloud must be [T, N, 6]")
            if name == "rgb" and (arr.ndim != 4 or arr.shape[3] != 3):
                raise ValueError("rgb must be [T, H, W, 3]")
            if name == "point_cloud" and arr.shape[1] <= 0:
                raise ValueError("point_cloud N must be positive")
            if name == "rgb" and min(arr.shape[1:3]) <= 0:
                raise ValueError("rgb H and W must be positive")
            if horizon is None:
                horizon = int(arr.shape[0])
            elif arr.shape[0] != horizon:
                raise ValueError("PolicyObservation modalities must share T")
        if horizon is None or horizon <= 0:
            raise ValueError("PolicyObservation requires a non-empty history")


def _validate_finite(array: np.ndarray, *, name: str) -> None:
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")


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
        if self.values is None:
            raise ValueError("FrameWindow.values must not be None")
        values = np.asarray(self.values)
        _validate_finite(values, name="FrameWindow.values")
        if values.ndim < 2:
            raise ValueError("FrameWindow.values must be [T, ...]")
        t = values.shape[0]
        for name in ("source_sequence", "source_monotonic_ns", "publish_monotonic_ns"):
            arr = np.asarray(getattr(self, name))
            if arr.dtype != np.uint64:
                raise TypeError(f"FrameWindow.{name} must have dtype uint64")
            if arr.shape != (t,):
                raise ValueError(f"FrameWindow.{name} must be a ({t},) uint64 array")
            object.__setattr__(self, name, arr)
        mask = np.asarray(self.valid_mask)
        if mask.dtype != np.uint8:
            raise TypeError("FrameWindow.valid_mask must have dtype uint8")
        if mask.shape != (t,):
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
        values = np.asarray(self.values)
        if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
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

    Process-local. ``arm_history``/``hand_history`` and
    ``hand_tactile_sum_history`` are sensor-value ``FrameWindow`` values.
    ``hand_tactile_provenance_history`` contains only the unit-code proof
    aligned to tactile sums; the full tactile tensor never enters deployment.
    ``pointcloud`` is the latest causally valid ``PointCloudFrame``. Optional
    modalities are None when not requested.
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
    hand_tactile_sum_history: FrameWindow | None = None
    hand_tactile_provenance_history: FrameWindow | None = None
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
            "hand_tactile_sum_history": self.hand_tactile_sum_history,
            "hand_tactile_provenance_history": self.hand_tactile_provenance_history,
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
