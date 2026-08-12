"""Reader for the current transactional HDF5 episode format.

Reads camera frames from DexMani episodes. Non-camera datasets
(arm_qpos, hand_qpos, flags, etc.) are accessed directly through
:attr:`h5f` — a merged view of the episode HDF5 sidecars.

An episode is one published directory containing ``data.h5``, ``depth.h5``,
``pointcloud.h5`` and ``rgb.mp4``. Older flat HDF5 files and pre-v16 episode
directories intentionally require an external migration tool.

Usage::

    with EpisodeReader("episode_001") as reader:
        # Non-camera data — direct merged-h5py access
        arm_qpos = reader.h5f["arm_qpos"][:]

        # Camera data
        rgb_frame  = reader.read_camera_frame("rgb", 42)
        all_depth  = reader.read_camera_all("depth")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.recording.timestamp_buffer import FillReason
from dexmani_real.recording.video_codec import VideoDecoder
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class ValidityState(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class EpisodeTiming:
    """Timing metadata from a schema-v16 episode."""

    rate_hz: float
    grid_dt_s: float
    grid_duration_s: float
    wall_duration_s: float
    non_sampled_duration_s: float


class MergedH5File:
    """Transparent merged view of ``data.h5`` and camera HDF5 sidecars.

    Camera keys are routed to their sidecar files; everything else goes to
    the data file.  ``"rgb"`` is handled by
    :class:`VideoDecoder` and is **not** present in either file.
    """

    __slots__ = ("_data", "_sidecars")

    def __init__(self, data_h5f: h5py.File, sidecars: dict[str, h5py.File] | None = None) -> None:
        self._data = data_h5f
        self._sidecars = sidecars or {}

    def __getitem__(self, key: str) -> Any:
        sidecar = self._sidecars.get(key)
        if sidecar is not None and key in sidecar:
            return sidecar[key]
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        sidecar = self._sidecars.get(key)
        return key in self._data or (sidecar is not None and key in sidecar)

    def keys(self) -> list[str]:
        ks = list(self._data.keys())
        for sidecar in self._sidecars.values():
            ks.extend(k for k in sidecar.keys() if k not in ks)
        return ks

    def __iter__(self):
        return iter(self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def close(self) -> None:
        self._data.close()
        seen: set[int] = set()
        for sidecar in self._sidecars.values():
            if id(sidecar) not in seen:
                sidecar.close()
                seen.add(id(sidecar))


class EpisodeReader:
    """Read camera frames from DexMani episodes.

    :attr:`h5f` returns a merged dict-like view over ``data.h5`` and camera
    sidecars so downstream code can access datasets by key
    (``f["arm_qpos"]``, ``f["depth"]``, ``f["pointcloud"]``).
    """

    def __init__(self, h5_path: str | Path) -> None:
        self._path = Path(h5_path)
        self._closed = False
        self._cache: dict[str, np.ndarray] = {}
        if not self._path.is_dir():
            raise ValueError(f"episode must be a schema-v16 directory: {self._path}")

        paths = {
            "data": self._path / "data.h5",
            "depth": self._path / "depth.h5",
            "pointcloud": self._path / "pointcloud.h5",
            "rgb": self._path / "rgb.mp4",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"schema-v16 episode is missing required files {missing}: {self._path}")

        self._data_h5f = h5py.File(paths["data"], "r")
        depth_h5f = h5py.File(paths["depth"], "r")
        pointcloud_h5f = h5py.File(paths["pointcloud"], "r")
        self._h5f = MergedH5File(
            self._data_h5f,
            {"depth": depth_h5f, "pointcloud": pointcloud_h5f},
        )
        self._rgb_decoder = VideoDecoder(paths["rgb"])
        schema_version = self.schema_version
        if schema_version != 16:
            self.close()
            raise ValueError(
                f"unsupported episode schema v{schema_version}; expected v16 "
                "(migrate historical episodes outside the runtime)"
            )

    # -- public properties ------------------------------------------------

    @property
    def h5f(self) -> MergedH5File:
        """Merged view of ``data.h5`` and camera HDF5 sidecars.

        ``f["rgb"]`` raises ``KeyError`` — use :meth:`read_camera_frame`
        or :meth:`read_camera_all` for RGB frames (MP4 decoding).
        """
        return self._h5f

    @property
    def h5_path(self) -> Path:
        return self._path

    @property
    def schema_version(self) -> int:
        meta = self._h5f.get("meta")
        return 0 if meta is None else int(meta.attrs.get("schema_version", 0) or 0)

    @property
    def validity(self) -> ValidityState:
        """Return whether the current schema-v16 episode is internally consistent."""
        meta = self._h5f.get("meta")
        if meta is None:
            return ValidityState.INVALID
        required = {
            "timestamp",
            "flag_sample_valid",
            "source_sample_index",
            "source_timestamp",
            "fill_reason",
            "observation_id",
            "observation_anchor_monotonic_ns",
            "arm_source_sequence",
            "hand_source_sequence",
            "vr_source_sequence",
            "camera_source_sequence",
            "arm_source_monotonic_ns",
            "hand_source_monotonic_ns",
            "vr_source_monotonic_ns",
            "camera_source_monotonic_ns",
            "arm_publish_monotonic_ns",
            "hand_publish_monotonic_ns",
            "vr_publish_monotonic_ns",
            "camera_publish_monotonic_ns",
            "observation_valid",
            "observation_source_age_s",
            "observation_source_skew_s",
            "observation_source_receive_monotonic_ns",
            "observation_history_valid_mask",
            "action_id",
            "action_created_monotonic_ns",
            "action_target_monotonic_ns",
            "action_valid_until_monotonic_ns",
            "action_arm_joint",
            "action_hand_joint",
            "action_arm_joint_raw",
            "action_hand_joint_raw",
            "flag_action_queued",
            "tactile_fresh",
            "tactile_source_monotonic_ns",
            "tactile_calibrated",
            "tactile_unit_code",
        }
        if not required.issubset(self._h5f.keys()):
            return ValidityState.INVALID
        frame_count = int(meta.attrs.get("num_frames", -1))
        if frame_count < 0 or any(int(self._h5f[key].shape[0]) != frame_count for key in required):
            return ValidityState.INVALID
        config_hash = str(meta.attrs.get("resolved_config_sha256", ""))
        if len(config_hash) != 64:
            return ValidityState.INVALID
        if not bool(meta.attrs.get("success", False)) or str(meta.attrs.get("camera_writer_error", "")):
            return ValidityState.INVALID
        if self._rgb_decoder is None or "depth" not in self._h5f or "pointcloud" not in self._h5f:
            return ValidityState.INVALID
        depth = self._h5f["depth"]
        pointcloud = self._h5f["pointcloud"]
        height = int(meta.attrs.get("camera_encoding_height", -1))
        width = int(meta.attrs.get("camera_encoding_width", -1))
        if (
            depth.shape != (frame_count, height, width)
            or depth.dtype != np.dtype(np.uint16)
            or depth.shape[0] != frame_count
            or pointcloud.ndim != 3
            or pointcloud.shape[0] != frame_count
            or pointcloud.shape[-1] != 6
            or pointcloud.dtype != np.dtype(np.float32)
        ):
            return ValidityState.INVALID
        try:
            if self._rgb_decoder.count_decoded_frames() != frame_count:
                return ValidityState.INVALID
        except Exception:
            logger.warning("failed to decode schema-v16 RGB stream", exc_info=True)
            return ValidityState.INVALID
        timestamps = np.asarray(self._h5f["timestamp"][:], dtype=np.float64)
        sample_valid = np.asarray(self._h5f["flag_sample_valid"][:], dtype=bool)
        source_indices = np.asarray(self._h5f["source_sample_index"][:], dtype=np.int64)
        source_timestamps = np.asarray(self._h5f["source_timestamp"][:], dtype=np.float64)
        fill_reasons = np.asarray(self._h5f["fill_reason"][:], dtype=np.uint8)
        scalar_shapes = (timestamps, sample_valid, source_indices, source_timestamps, fill_reasons)
        if any(values.shape != (frame_count,) for values in scalar_shapes):
            return ValidityState.INVALID
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
            return ValidityState.INVALID

        known_fill_reasons = np.isin(fill_reasons, [int(reason) for reason in FillReason])
        if not np.all(known_fill_reasons):
            return ValidityState.INVALID
        is_source = fill_reasons == int(FillReason.SOURCE)
        is_hold = fill_reasons == int(FillReason.CAUSAL_HOLD_LAST)
        is_leading = fill_reasons == int(FillReason.LEADING_PLACEHOLDER)
        source_defined = source_indices >= 0
        source_time_finite = np.isfinite(source_timestamps)
        if (
            np.any(is_source != sample_valid)
            or np.any((is_source | is_hold) != source_defined)
            or np.any((is_source | is_hold) != source_time_finite)
            or np.any(is_leading & (sample_valid | source_defined | source_time_finite))
        ):
            return ValidityState.INVALID
        # All non-placeholder values must be causally available at their grid
        # deadline. The small tolerance only covers float64 representation of
        # large monotonic timestamps; it is far below a control tick.
        if np.any(source_defined & (source_timestamps > timestamps + 1e-7)):
            return ValidityState.INVALID
        defined_indices = source_indices[source_defined]
        if defined_indices.size > 1 and np.any(np.diff(defined_indices) < 0):
            return ValidityState.INVALID
        if not np.any(sample_valid):
            return ValidityState.INVALID

        anchors = np.asarray(self._h5f["observation_anchor_monotonic_ns"][:], dtype=np.uint64)
        source_sequences = np.column_stack(
            [
                np.asarray(self._h5f[f"{name}_source_sequence"][:], dtype=np.uint64)
                for name in ("arm", "hand", "vr", "camera")
            ]
        )
        observation_source_ns = np.column_stack(
            [
                np.asarray(self._h5f[f"{name}_source_monotonic_ns"][:], dtype=np.uint64)
                for name in ("arm", "hand", "vr", "camera")
            ]
        )
        observation_publish_ns = np.column_stack(
            [
                np.asarray(self._h5f[f"{name}_publish_monotonic_ns"][:], dtype=np.uint64)
                for name in ("arm", "hand", "vr", "camera")
            ]
        )
        observation_receive_ns = np.asarray(self._h5f["observation_source_receive_monotonic_ns"][:], dtype=np.uint64)
        observation_history_valid = np.asarray(self._h5f["observation_history_valid_mask"][:], dtype=bool)
        if observation_history_valid.shape != (frame_count, 4, 1):
            return ValidityState.INVALID
        valid_sources = observation_history_valid[:, :, 0]
        if (
            anchors.shape != (frame_count,)
            or source_sequences.shape != (frame_count, 4)
            or observation_source_ns.shape != (frame_count, 4)
            or observation_receive_ns.shape != (frame_count, 4)
            or observation_publish_ns.shape != (frame_count, 4)
            or np.any(anchors == 0)
        ):
            return ValidityState.INVALID
        causal_chain = (
            (source_sequences > 0)
            & (observation_source_ns > 0)
            & (observation_source_ns <= observation_receive_ns)
            & (observation_receive_ns <= observation_publish_ns)
            & (observation_publish_ns <= anchors[:, None])
        )
        if np.any(valid_sources & ~causal_chain):
            return ValidityState.INVALID
        tactile_fresh = np.asarray(self._h5f["tactile_fresh"][:], dtype=bool)
        tactile_source_ns = np.asarray(self._h5f["tactile_source_monotonic_ns"][:], dtype=np.uint64)
        if np.any(tactile_fresh & ((tactile_source_ns == 0) | (tactile_source_ns > anchors))):
            return ValidityState.INVALID

        observation_ids = np.asarray(self._h5f["observation_id"][:], dtype=np.uint64)
        action_ids = np.asarray(self._h5f["action_id"][:], dtype=np.uint64)
        action_created_ns = np.asarray(self._h5f["action_created_monotonic_ns"][:], dtype=np.uint64)
        action_target_ns = np.asarray(self._h5f["action_target_monotonic_ns"][:], dtype=np.uint64)
        action_valid_until_ns = np.asarray(self._h5f["action_valid_until_monotonic_ns"][:], dtype=np.uint64)
        queued = np.asarray(self._h5f["flag_action_queued"][:], dtype=bool)
        held = (
            np.asarray(self._h5f["flag_held"][:], dtype=bool)
            if "flag_held" in self._h5f
            else np.zeros(frame_count, dtype=bool)
        )
        # A held source sample may intentionally publish no new action. Active
        # samples require an action identity, and queue progress may never
        # claim the zero-action sentinel.
        if np.any(observation_ids[sample_valid] == 0) or np.any(sample_valid & ~held & (action_ids == 0)):
            return ValidityState.INVALID
        if np.any(queued & (action_ids == 0)):
            return ValidityState.INVALID
        action_timing_valid = (
            (action_created_ns > 0)
            & (action_created_ns <= action_target_ns)
            & (action_target_ns <= action_valid_until_ns)
        )
        if np.any(queued & ~action_timing_valid):
            return ValidityState.INVALID
        action_arrays = (
            ("action_arm_joint", (frame_count, *ARM_JOINT_SHAPE)),
            ("action_hand_joint", (frame_count, *HAND_JOINT_SHAPE)),
            ("action_arm_joint_raw", (frame_count, *ARM_JOINT_SHAPE)),
            ("action_hand_joint_raw", (frame_count, *HAND_JOINT_SHAPE)),
        )
        for name, expected_shape in action_arrays:
            values = np.asarray(self._h5f[name][:], dtype=np.float64)
            if values.shape != expected_shape or np.any(~np.isfinite(values[sample_valid])):
                return ValidityState.INVALID
        if np.any(sample_valid & ~held & ~queued):
            return ValidityState.INVALID
        return ValidityState.VALID

    def require_valid(self, purpose: str = "training/live replay") -> None:
        state = self.validity
        if state is not ValidityState.VALID:
            raise ValueError(
                f"episode validity is {state.value}; {purpose} requires schema-v16 VALID data "
                "(the runtime only accepts schema-v16 episodes)"
            )

    @property
    def timing(self) -> EpisodeTiming:
        """Return timing recorded by the schema-v16 fixed control grid."""
        meta = self._h5f.get("meta")
        attrs = meta.attrs if meta is not None else {}

        def _positive(value: Any) -> float | None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if np.isfinite(result) and result > 0 else None

        timestamps = np.asarray(self._h5f["timestamp"][:], dtype=np.float64)
        timestamp_dt_s: float | None = None
        timestamp_duration_s: float | None = None
        if timestamps is not None and timestamps.size >= 2:
            finite = timestamps[np.isfinite(timestamps)]
            if finite.size >= 2:
                positive_deltas = np.diff(finite)
                positive_deltas = positive_deltas[positive_deltas > 0]
                if positive_deltas.size:
                    timestamp_dt_s = float(np.median(positive_deltas))
                span = float(finite[-1] - finite[0])
                if span >= 0:
                    timestamp_duration_s = span

        control_hz = _positive(attrs.get("control_hz"))
        grid_dt_s = _positive(attrs.get("grid_dt_s"))
        if control_hz is None or grid_dt_s is None:
            raise ValueError("schema-v16 episode has invalid control-grid metadata")
        rate_hz = control_hz
        explicit_grid_duration_s = attrs.get("grid_duration_s")
        try:
            grid_duration_s = float(explicit_grid_duration_s) if explicit_grid_duration_s is not None else float("nan")
        except (TypeError, ValueError):
            grid_duration_s = float("nan")
        if not np.isfinite(grid_duration_s) or grid_duration_s < 0:
            if timestamp_duration_s is None:
                raise ValueError("schema-v16 episode has invalid grid_duration_s")
            grid_duration_s = timestamp_duration_s

        wall_duration_s = float(attrs.get("wall_duration_s", grid_duration_s))
        if not np.isfinite(wall_duration_s) or wall_duration_s < 0:
            wall_duration_s = grid_duration_s
        non_sampled_duration_s = float(attrs.get("non_sampled_duration_s", max(0.0, wall_duration_s - grid_duration_s)))
        if not np.isfinite(non_sampled_duration_s) or non_sampled_duration_s < 0:
            non_sampled_duration_s = max(0.0, wall_duration_s - grid_duration_s)

        return EpisodeTiming(
            rate_hz=rate_hz,
            grid_dt_s=grid_dt_s,
            grid_duration_s=grid_duration_s,
            wall_duration_s=wall_duration_s,
            non_sampled_duration_s=non_sampled_duration_s,
        )

    # -- camera queries ---------------------------------------------------

    def read_camera_frame(self, key: str, index: int) -> np.ndarray:
        """Read a single camera frame by index.

        The schema-v16 MP4 sidecar must have exactly one frame per grid slot.
        """
        if key == "rgb" and self._rgb_decoder is not None:
            n = self._rgb_decoder.frame_count
            if n == 0:
                raise ValueError(f"MP4 file contains no frames: {self._path}")
            return self._rgb_decoder.read_frame(index)
        if key in self._h5f:
            return np.asarray(self._h5f[key][index])
        raise KeyError(f"Camera dataset '{key}' not found in {self._path}")

    def read_camera_all(self, key: str) -> np.ndarray:
        """Read all camera frames. Cached after the first call.

        Returns a ``(T, ...)`` array (``uint8`` for RGB, ``uint16`` for depth).
        """
        if key in self._cache:
            return self._cache[key]

        if key == "rgb" and self._rgb_decoder is not None:
            data = self._rgb_decoder.read_all()
            grid_len = int(self._h5f["meta"].attrs.get("num_frames", 0))
            if grid_len != data.shape[0]:
                raise ValueError(f"schema-v16 RGB length {data.shape[0]} does not match grid length {grid_len}")
        elif key in self._h5f:
            data = np.asarray(self._h5f[key][:])
        else:
            raise KeyError(f"Camera dataset '{key}' not found in {self._path}")

        self._cache[key] = data
        return data

    # -- context manager --------------------------------------------------

    def close(self) -> None:
        """Close all files and decoders. Idempotent."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._cache.clear()
        if self._rgb_decoder is not None:
            self._rgb_decoder.close()
            self._rgb_decoder = None
        if hasattr(self, "_h5f") and self._h5f is not None:
            self._h5f.close()
            self._h5f = None  # type: ignore[assignment]

    def __enter__(self) -> "EpisodeReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
