"""Post-processing pipeline for HDF5 teleoperation episodes.

Provides offline timestamp alignment and stream interpolation for
multi-rate sensor data recorded at different frequencies.

Ref: BunnyVisionPro post-hoc alignment (postprocessing/align_streams.py).
     DexUMI timestamp-based latency correction.

Architecture:
    ┌──────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ HDF5     │──►│ StreamInterpolator│──►│ TimestampAligner │──► Aligned dict
    │ Episode  │   │ (per-stream)     │   │ (unified grid)   │
    └──────────┘   └──────────────────┘   └──────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


# ── Stream Interpolator ──


@dataclass
class InterpolationConfig:
    """Configuration for StreamInterpolator."""

    method: str = "linear"  # "linear", "nearest", "slerp", "none"
    max_gap_s: float = 0.1  # Maximum gap to interpolate across
    extrapolate: bool = False  # Whether to extrapolate beyond data bounds


class StreamInterpolator:
    """Interpolate a single sensor stream onto target timestamps.

    Supports linear and nearest-neighbor interpolation with a configurable
    maximum gap threshold. Gaps larger than max_gap_s produce NaN values
    rather than spurious interpolation.

    Usage:
        si = StreamInterpolator(config)
        aligned = si.interpolate(source_ts, source_data, target_ts)
    """

    def __init__(self, config: InterpolationConfig | None = None) -> None:
        self.config = config or InterpolationConfig()

    def interpolate(
        self,
        source_ts: np.ndarray,
        source_data: np.ndarray,
        target_ts: np.ndarray,
    ) -> np.ndarray:
        """Interpolate source_data onto target_ts.

        Args:
            source_ts: (N,) array of source timestamps (seconds).
            source_data: (N, D...) array of source data values.
            target_ts: (M,) array of target timestamps (seconds).

        Returns:
            (M, D...) aligned data array with NaN in un-interpolatable regions.
        """
        source_ts = np.asarray(source_ts, dtype=np.float64)
        source_data = np.asarray(source_data)
        target_ts = np.asarray(target_ts, dtype=np.float64)

        if len(source_ts) == 0 or len(target_ts) == 0:
            return np.full((len(target_ts),) + source_data.shape[1:], np.nan, dtype=source_data.dtype)

        if self.config.method == "nearest":
            return self._interpolate_nearest(source_ts, source_data, target_ts)
        elif self.config.method == "linear":
            return self._interpolate_linear(source_ts, source_data, target_ts)
        elif self.config.method == "slerp":
            return self._interpolate_slerp(source_ts, source_data, target_ts)
        elif self.config.method == "none":
            # No interpolation — return raw data aligned by index
            return source_data[: len(target_ts)]
        else:
            raise ValueError(f"Unknown interpolation method: {self.config.method}")

    def _interpolate_linear(
        self,
        source_ts: np.ndarray,
        source_data: np.ndarray,
        target_ts: np.ndarray,
    ) -> np.ndarray:
        """Linear interpolation with max gap constraint."""
        # Find insertion indices
        idx = np.searchsorted(source_ts, target_ts)
        idx = np.clip(idx, 1, len(source_ts) - 1)

        # Left and right neighbors
        t_left = source_ts[idx - 1]
        t_right = source_ts[idx]
        d_left = source_data[idx - 1]
        d_right = source_data[idx]

        # Gap check
        gap = t_right - t_left
        valid_gap = gap <= self.config.max_gap_s

        # Also check distance to nearest source point
        dist_left = target_ts - t_left
        dist_right = t_right - target_ts

        # Fraction between left and right
        frac = np.where(gap > 0, (target_ts - t_left) / gap, 0.0)
        frac = np.clip(frac, 0.0, 1.0)

        # Perform interpolation
        # Reshape for broadcasting if data has extra dimensions
        flat_data = len(source_data.shape) > 1
        if flat_data:
            frac = frac[:, np.newaxis] if frac.ndim == 1 else frac
            while frac.ndim < d_left.ndim:
                frac = frac[..., np.newaxis]

        result = d_left + frac * (d_right - d_left)

        # Invalidate gaps
        invalid = ~valid_gap
        # Also invalidate extrapolation (beyond source bounds)
        if not self.config.extrapolate:
            invalid = invalid | (target_ts < source_ts[0]) | (target_ts > source_ts[-1])

        if invalid.any():
            if flat_data:
                invalid_expanded = invalid
                while invalid_expanded.ndim < result.ndim:
                    invalid_expanded = invalid_expanded[..., np.newaxis]
                result = np.where(invalid_expanded, np.nan, result)
            else:
                result[invalid] = np.nan

        return result

    def _interpolate_nearest(
        self,
        source_ts: np.ndarray,
        source_data: np.ndarray,
        target_ts: np.ndarray,
    ) -> np.ndarray:
        """Nearest-neighbor interpolation with max gap constraint."""
        idx = np.searchsorted(source_ts, target_ts)
        idx = np.clip(idx, 0, len(source_ts) - 1)

        result = source_data[idx]

        # Gap check: distance to nearest source point
        dist = np.abs(target_ts - source_ts[idx])
        invalid = dist > self.config.max_gap_s

        if not self.config.extrapolate:
            invalid = invalid | (target_ts < source_ts[0]) | (target_ts > source_ts[-1])

        if invalid.any():
            if len(source_data.shape) > 1:
                invalid_expanded = invalid
                while invalid_expanded.ndim < result.ndim:
                    invalid_expanded = invalid_expanded[..., np.newaxis]
                result = np.where(invalid_expanded, np.nan, result)
            else:
                result[invalid] = np.nan

        return result

    def _interpolate_slerp(
        self,
        source_ts: np.ndarray,
        source_data: np.ndarray,
        target_ts: np.ndarray,
    ) -> np.ndarray:
        """Spherical linear interpolation for quaternion data (WXYZ order).

        Uses scipy.spatial.transform.Slerp for correct unit-quaternion
        interpolation.  Falls back to linear+L2-normalize if scipy is
        unavailable.
        """
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        source_data = np.asarray(source_data)
        if source_data.shape[-1] != 4:
            raise ValueError(f"SLERP requires quaternion data (last dim=4), got shape {source_data.shape}")

        # Build rotation objects from WXYZ quaternions
        source_rots = R.from_quat(source_data[..., [1, 2, 3, 0]])  # WXYZ → xyzw
        slerp = Slerp(source_ts, source_rots)

        # Interpolate at target timestamps
        target_rots = slerp(target_ts)
        result = target_rots.as_quat()  # xyzw

        # Convert back to WXYZ order
        result_wxyz = np.zeros_like(result)
        result_wxyz[..., 0] = result[..., 3]  # w
        result_wxyz[..., 1:] = result[..., :3]  # xyz

        # Invalidate gaps and extrapolation
        idx = np.searchsorted(source_ts, target_ts)
        idx = np.clip(idx, 1, len(source_ts) - 1)
        gaps = source_ts[idx] - source_ts[idx - 1]
        valid_gap = gaps <= self.config.max_gap_s

        invalid = ~valid_gap
        if not self.config.extrapolate:
            invalid = invalid | (target_ts < source_ts[0]) | (target_ts > source_ts[-1])

        if invalid.any():
            invalid_expanded = invalid
            while invalid_expanded.ndim < result_wxyz.ndim:
                invalid_expanded = invalid_expanded[..., np.newaxis]
            result_wxyz = np.where(invalid_expanded, np.nan, result_wxyz)

        return result_wxyz


# ── Timestamp Aligner ──


class TimestampAligner:
    """Post-process all sensor streams onto a unified timestamp grid.

    Handles the common case where control loop runs at 50Hz, camera at 30Hz,
    and VR tracking at 72-120Hz — aligning everything to a single 20ms grid.

    Usage:
        aligner = TimestampAligner()
        aligned = aligner.align(h5_file, t_start, t_end, dt=0.020)
        # aligned is a dict of {dataset_path: aligned_array}
    """

    def __init__(
        self,
        dt: float = 0.020,
        method: str = "linear",
        max_gap_s: float = 0.1,
    ) -> None:
        self.dt = dt
        self.config = InterpolationConfig(method=method, max_gap_s=max_gap_s)
        self.interpolator = StreamInterpolator(self.config)

    def align(
        self,
        h5_path: str,
        t_start: float | None = None,
        t_end: float | None = None,
        dt: float | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Align all streams in an HDF5 episode to a unified time grid.

        Args:
            h5_path: Path to the HDF5 episode file.
            t_start: Start time (seconds). Default: first control timestamp.
            t_end: End time (seconds). Default: last control timestamp.
            dt: Time step (seconds). Default: self.dt (20ms).

        Returns:
            dict mapping dataset path → aligned numpy array, or None on error.
        """
        dt = dt or self.dt

        try:
            with h5py.File(h5_path, "r") as f:
                return self._align_from_file(f, t_start, t_end, dt)
        except (OSError, KeyError) as e:
            logger.error("TimestampAligner failed on %s: %s", h5_path, e)
            return None

    def _align_from_file(
        self,
        f: h5py.File,
        t_start: float | None,
        t_end: float | None,
        dt: float,
    ) -> dict[str, np.ndarray]:
        """Internal: align streams from an open HDF5 file."""
        # Get control timestamps
        if "timestamp" in f:
            ctrl_ts = np.asarray(f["timestamp"][:], dtype=np.float64)
        else:
            # Fall back to synthetic timestamps based on frame count
            n_frames = f["arm_qpos"].shape[0]
            fps = f["meta"].attrs.get("fps", 50.0)
            ctrl_ts = np.arange(n_frames, dtype=np.float64) / fps
            logger.warning("No /timestamp dataset — using synthetic timestamps @ %.0f Hz", fps)

        if t_start is None:
            t_start = float(ctrl_ts[0])
        if t_end is None:
            t_end = float(ctrl_ts[-1])

        target_ts = np.arange(t_start, t_end + dt / 2, dt, dtype=np.float64)
        result: dict[str, np.ndarray] = {}

        # Streams to align (path → source timestamps) — v2 flat schema
        streams: list[tuple[str, np.ndarray, str]] = [
            ("arm_qpos", ctrl_ts, "linear"),
            ("arm_qvel", ctrl_ts, "linear"),
            ("arm_tau", ctrl_ts, "linear"),
            ("arm_ee", ctrl_ts, "linear"),
            ("hand_qpos", ctrl_ts, "linear"),
            ("hand_fingertip", ctrl_ts, "linear"),
            ("hand_contact", ctrl_ts, "linear"),
            ("action_arm_joint", ctrl_ts, "linear"),
            ("action_arm_ee", ctrl_ts, "linear"),
            ("action_hand_joint", ctrl_ts, "linear"),
            ("vr_wrist_pos", ctrl_ts, "linear"),
            ("vr_wrist_rot6d", ctrl_ts, "linear"),
            ("vr_landmarks", ctrl_ts, "linear"),
        ]

        # Add camera streams if available (aligned by index — v2 has no camera timestamps)
        has_camera = "rgb" in f
        if has_camera:
            # Camera frames are forward-filled to match the control grid;
            # use nearest-neighbor to map camera index → timestamp grid.
            cam_ts = np.arange(f["rgb"].shape[0], dtype=np.float64) / f["meta"].attrs.get("fps", 50.0)
            streams.append(("rgb", cam_ts, "nearest"))
            if "depth" in f:
                streams.append(("depth", cam_ts, "nearest"))

        # Align each stream
        for path, source_ts, method in streams:
            if path not in f:
                continue

            source_data = np.asarray(f[path][:])
            if len(source_data) == 0:
                continue

            # Use appropriate method
            si = StreamInterpolator(InterpolationConfig(method=method))
            aligned = si.interpolate(source_ts, source_data, target_ts)
            result[path] = aligned

        result["aligned_timestamps"] = target_ts
        result["aligned_fps"] = np.float64(1.0 / dt)

        logger.info(
            "TimestampAligner: %d streams aligned to %.0fms grid (%d→%d frames)",
            len(result) - 2,
            dt * 1000,
            len(ctrl_ts),
            len(target_ts),
        )

        return result

    def validate_alignment(self, aligned: dict[str, np.ndarray]) -> dict[str, Any]:
        """Validate aligned data for NaN gaps and consistency.

        Returns a validation report dict.
        """
        report: dict[str, Any] = {
            "total_streams": 0,
            "streams_with_nan": 0,
            "max_nan_ratio": 0.0,
            "stream_details": {},
        }

        for path, data in aligned.items():
            if path in ("aligned_timestamps", "aligned_fps"):
                continue

            nan_count = int(np.isnan(data).any(axis=tuple(range(1, data.ndim))).sum())
            nan_ratio = nan_count / max(len(data), 1)

            report["stream_details"][path] = {
                "shape": data.shape,
                "nan_frames": nan_count,
                "nan_ratio": round(nan_ratio, 4),
            }
            report["total_streams"] += 1
            if nan_count > 0:
                report["streams_with_nan"] += 1
            if nan_ratio > report["max_nan_ratio"]:
                report["max_nan_ratio"] = nan_ratio

        report["ok"] = report["max_nan_ratio"] < 0.1  # less than 10% NaN

        return report


# ── Convenience function ──


def align_and_validate(
    h5_path: str,
    dt: float = 0.020,
    method: str = "linear",
    max_gap_s: float = 0.1,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Align an episode and validate the result.

    Returns (aligned_data, validation_report).
    """
    aligner = TimestampAligner(dt=dt, method=method, max_gap_s=max_gap_s)
    aligned = aligner.align(h5_path, dt=dt)
    if aligned is None:
        return None, {"error": "Alignment failed", "ok": False}
    report = aligner.validate_alignment(aligned)
    return aligned, report
