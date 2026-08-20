"""Transactional raw v18 episode serialization from owned ``EpisodeFrame`` rows.

State, action, VR, and camera rows are causally aligned to the policy grid.
The recorder owns HDF5/video sidecars and verifies them before atomically
publishing an episode; it neither reads shared memory nor controls hardware.
"""

from __future__ import annotations

__all__ = ["EpisodeRecorder", "StopResult"]

import atexit
import json
import os
import shutil
import tempfile
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.defaults import camera
from dexmani_real.recording.camera_stream_writer import (
    CameraStreamWriter,
    CameraStreamWriterConfig,
)
from dexmani_real.recording.episode_frame import EpisodeFrame, build_episode_frame
from dexmani_real.recording.episode_schema import (
    ARM_SENT_MARKER,
    EPISODE_SCHEMA_VERSION,
    SEMANTIC_META_ATTRS_V17,
    validate_data_layout_v17,
    validate_source_frame_keys_v17,
)
from dexmani_real.recording.timestamp_buffer import TimestampAlignedBuffer
from dexmani_real.recording.transaction import atomic_publish
from dexmani_real.recording.video_codec import VideoDecoder
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_RUNTIME_QUALITY_METRIC_NAMES = frozenset(
    {"hand_read_error_count", "hand_overcurrent_count"}
)

DEFAULT_MAX_RECORD_FRAMES: int = 10000
SCHEMA_VERSION = EPISODE_SCHEMA_VERSION
_CAMERA_WRITER_CLOSE_TIMEOUT_S = 60.0
_PREVIOUS_EPISODE_STOP_TIMEOUT_S = 15.0
_PROCESS_EXIT_STOP_TIMEOUT_S = 60.0


# The episode-stop thread is daemonized so forced exits do not block shutdown.
_LIVE_RECORDERS: weakref.WeakSet = weakref.WeakSet()


def _flush_all_recorders() -> None:
    for rec in list(_LIVE_RECORDERS):  # snapshot — WeakSet may mutate under GC
        try:
            if rec._recording:
                rec.stop_episode(success=False, reason="atexit")
            if not rec.join_stop(timeout=_PROCESS_EXIT_STOP_TIMEOUT_S):
                logger.error("recorder did not finish before interpreter shutdown")
        except Exception:
            logger.warning(
                "recorder cleanup failed during interpreter shutdown", exc_info=True
            )


atexit.register(_flush_all_recorders)


def _episode_quality_metrics(
    datasets: dict[str, Any],
    *,
    frame_count: int,
    control_hz: float,
    runtime_metrics: dict[str, int],
) -> dict[str, int]:
    """Summarize persisted frame flags plus runtime-only event counters."""
    if frame_count < 0 or not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("invalid episode quality dimensions")
    unknown_metrics = set(runtime_metrics) - _RUNTIME_QUALITY_METRIC_NAMES
    if unknown_metrics:
        raise ValueError(f"unknown runtime quality metrics: {sorted(unknown_metrics)}")
    normalized_runtime_metrics = {
        name: int(value) for name, value in runtime_metrics.items()
    }
    if any(value < 0 for value in normalized_runtime_metrics.values()):
        raise ValueError("runtime quality metrics must be non-negative")
    if frame_count == 0:
        return {
            "ik_hold_frame_count": 0,
            "camera_invalid_frame_count": 0,
            "observation_invalid_frame_count": 0,
            "sample_invalid_frame_count": 0,
            "safety_reject_frame_count": 0,
            "command_quiescence_count": 0,
            **normalized_runtime_metrics,
        }

    def _bool_dataset(name: str) -> np.ndarray:
        if name not in datasets:
            raise KeyError(f"required quality dataset missing: {name}")
        return np.asarray(datasets[name][:frame_count], dtype=bool)

    held = _bool_dataset("flag_held")
    ik_ok = _bool_dataset("flag_ik_ok")
    observation_valid = _bool_dataset("observation_valid")
    camera_fresh = _bool_dataset("flag_camera_fresh")
    sample_valid = _bool_dataset("flag_sample_valid")
    safety_reject = _bool_dataset("flag_safety_reject")
    if "timestamp" not in datasets:
        raise KeyError("required quality dataset missing: timestamp")
    timestamps = np.asarray(datasets["timestamp"][:frame_count], dtype=np.float64)
    return {
        "ik_hold_frame_count": int(np.count_nonzero(held & ~ik_ok)),
        "camera_invalid_frame_count": int(np.count_nonzero(~camera_fresh)),
        "observation_invalid_frame_count": int(np.count_nonzero(~observation_valid)),
        "sample_invalid_frame_count": int(np.count_nonzero(~sample_valid)),
        "safety_reject_frame_count": int(np.count_nonzero(safety_reject)),
        "command_quiescence_count": int(
            np.count_nonzero(np.diff(timestamps) > (1.5 / control_hz))
        ),
        **normalized_runtime_metrics,
    }


@dataclass
class StopResult:
    """Outcome of a background stop_episode daemon, returned by poll_stop()."""

    done: bool
    """True when the daemon finished (success or crash)."""
    error: str | None = None
    """Error message if the daemon crashed, None otherwise."""
    success: bool = False
    """Whether stop_episode was called with success=True (save vs discard)."""
    path: str | None = None
    """Episode directory path, or None if no stop was pending."""
    frame_count: int = 0
    """Frame count at the moment stop_episode was called."""


class EpisodeRecorder:
    """Records teleoperation episodes to HDF5 files.

    Lifecycle: start_episode() → add_frame() × N → stop_episode()
    """

    def __init__(
        self,
        data_dir: str,
        max_frames: int = DEFAULT_MAX_RECORD_FRAMES,
        control_hz: float = 16.0,
        min_frames: int = 50,
        arm_sent_stream: bool = False,
        camera_writer_config: CameraStreamWriterConfig | None = None,
        resolved_config_hash: str | None = None,
        provenance: dict[str, str] | None = None,
    ) -> None:
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        if resolved_config_hash is None or len(resolved_config_hash) != 64:
            raise ValueError("EpisodeRecorder requires a resolved config SHA-256")
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.control_hz = float(control_hz)
        self.min_frames = int(min_frames)
        self._resolved_config_hash = resolved_config_hash
        self._provenance = dict(provenance or {})

        self.arm_sent_stream: bool = bool(arm_sent_stream)

        self._file: Any = (
            None  # h5py.File | None — data.h5 (control streams + metadata)
        )
        self._camera_writer: CameraStreamWriter | None = None
        self._camera_writer_metrics: dict[str, float | int] = {}
        self._last_camera_payload: tuple[np.ndarray, np.ndarray] | None = None
        self._camera_writer_config = camera_writer_config or CameraStreamWriterConfig(
            rgb_shape=camera.rgb_shape,
            depth_shape=camera.depth_shape,
            fps=self.control_hz,
            queue_size=camera.writer_queue_size,
        )
        if not np.isclose(self._camera_writer_config.fps, self.control_hz):
            raise ValueError("camera writer fps must match recorder control_hz")
        self._frame_count: int = 0
        self._recording: bool = False
        self._max_frames_reached: bool = False
        self._start_time: float | None = None
        self._episode_dir: str | None = None  # episode_XXX/ directory
        self._temp_dir: str | None = None  # .tmp_episode_XXX/ directory
        self._datasets: dict[str, Any] = {}
        self._pending_meta: dict[str, Any] = {}
        self._runtime_quality_metrics: dict[str, int] = {}

        self._buffer: TimestampAlignedBuffer | None = None
        self._flush_interval: int = max(1, int(round(10.0 * self.control_hz)))
        self._flushed_frames: int = 0

        self._skip_initial_frames: int = 0
        self._skipped_so_far: int = 0
        self._last_control_run_generation: int | None = None

        self._stop_thread: threading.Thread | None = None

        # Error from the last stop operation; callers read it after join_stop().
        self._stop_error: str | None = None

        self._stop_success: bool = False
        self._stop_path: str | None = None
        self._stop_frame_count: int = 0

        _LIVE_RECORDERS.add(self)  # atexit flush net

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def set_runtime_quality_metrics(self, metrics: dict[str, int]) -> None:
        """Attach non-negative episode-level counters before stop."""
        if not self._recording:
            raise RuntimeError("runtime quality metrics require an active episode")
        normalized: dict[str, int] = {}
        for name, value in metrics.items():
            if name not in _RUNTIME_QUALITY_METRIC_NAMES:
                raise ValueError(f"unknown runtime quality metric: {name!r}")
            count = int(value)
            if count < 0:
                raise ValueError(
                    f"runtime quality metric {name!r} must be non-negative"
                )
            normalized[name] = count
        self._runtime_quality_metrics = normalized

    @property
    def max_frames_reached(self) -> bool:
        return self._max_frames_reached

    @property
    def stop_error(self) -> str | None:
        """Error message from the last _stop_episode_impl, or None if clean."""
        return self._stop_error

    @property
    def camera_writer_error(self) -> str | None:
        """Latched camera sidecar error requiring episode discard."""
        return self._camera_writer.error if self._camera_writer is not None else None

    def start_episode(
        self,
        task_label: str = "",
        operator: str = "",
        calib: CameraCalib | None = None,
        camera_K: np.ndarray | None = None,
        camera_name: str | None = None,
        camera_serial: str | None = None,
        depth_scale: float | None = None,
        camera_metadata: dict[str, Any] | None = None,
        skip_initial_frames: int = 0,
    ) -> bool:
        if not self.join_stop(timeout=_PREVIOUS_EPISODE_STOP_TIMEOUT_S):
            if self._stop_error is not None:
                logger.warning(
                    "Previous stop crashed (%s) — state was reset, allowing new start",
                    self._stop_error,
                )
                self._stop_thread = None
            else:
                logger.error(
                    "Previous episode still flushing — refusing to start a new one"
                )
                return False

        if self._recording:
            return False

        self.data_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ep_dir = self.data_dir / f"episode_{stamp}"
        tmp_dir = self.data_dir / f".tmp_episode_{stamp}"
        dedup = 1
        while ep_dir.exists() or tmp_dir.exists():  # same-second collision → suffix
            ep_dir = self.data_dir / f"episode_{stamp}_{dedup}"
            tmp_dir = self.data_dir / f".tmp_episode_{stamp}_{dedup}"
            dedup += 1

        self._episode_dir = str(ep_dir)
        self._temp_dir = str(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)
        self._frame_count = 0
        self._max_frames_reached = False
        self._start_time = time.perf_counter()
        self._recording = True
        self._stop_error = None
        self._datasets = {}
        self._flushed_frames = 0

        self._skip_initial_frames = max(
            0, min(int(skip_initial_frames), self.max_frames - 1)
        )
        self._skipped_so_far = 0
        self._last_control_run_generation = None

        self._pending_meta = {
            "task_label": task_label,
            "operator": operator,
            "calib": calib,
            "camera_K": camera_K,
            "camera_name": camera_name,
            "camera_serial": camera_serial,
            "depth_scale": depth_scale,
            "camera_metadata": camera_metadata,
            "skip_initial_frames": self._skip_initial_frames,
        }

        # Defer data.h5 creation until the first flush or stop.
        dt = 1.0 / self.control_hz
        self._buffer = TimestampAlignedBuffer(
            start_time=self._start_time,
            dt=dt,
            max_record_steps=self.max_frames,
            # Only tolerate floating-point error at an exact grid boundary.
            eps=1e-5,
        )
        self._file = None
        self._camera_writer = CameraStreamWriter(tmp_dir, self._camera_writer_config)
        self._last_camera_payload = None
        return True

    def _write_meta_attrs(self, meta: h5py.Group) -> None:
        """Write deferred metadata attributes to the HDF5 meta group."""
        p = self._pending_meta
        meta.attrs["task_label"] = p.get("task_label", "")
        meta.attrs["operator"] = p.get("operator", "")
        meta.attrs["control_hz"] = (
            self.control_hz
        )  # nominal grid rate; dt = 1/control_hz
        meta.attrs["fps"] = self.control_hz
        if self._resolved_config_hash is not None:
            meta.attrs["resolved_config_sha256"] = self._resolved_config_hash
        for key, value in sorted(self._provenance.items()):
            meta.attrs[f"provenance_{key}"] = str(value)

        # Additive metadata for fields whose numeric layout remains unchanged.
        for key, semantic_value in SEMANTIC_META_ATTRS_V17.items():
            meta.attrs[key] = semantic_value

        if self.arm_sent_stream:
            meta.attrs[ARM_SENT_MARKER] = True

        self._write_camera_meta_attrs(meta)

        meta.attrs["skip_initial_frames"] = int(p.get("skip_initial_frames", 0))
        camera_metadata = p.get("camera_metadata") or {}
        for key, val in camera_metadata.items():
            meta.attrs[key] = val

        video = self._camera_writer_config.video
        meta.attrs["camera_writer_queue_size"] = self._camera_writer_config.queue_size
        meta.attrs["camera_encoding_codec"] = video.codec
        meta.attrs["camera_encoding_crf"] = video.crf
        meta.attrs["camera_encoding_preset"] = video.preset
        meta.attrs["camera_encoding_pixel_format"] = video.pixel_format
        meta.attrs["camera_encoding_width"] = self._camera_writer_config.rgb_shape[1]
        meta.attrs["camera_encoding_height"] = self._camera_writer_config.rgb_shape[0]
        meta.attrs["camera_encoding_fps"] = self._camera_writer_config.fps
        meta.attrs["camera_health_taxonomy_json"] = json.dumps(
            {
                "0": "OK",
                "1": "CLOCK_RESET",
                "2": "DUPLICATE",
                "3": "FRAME_GAP",
                "4": "BACKLOG",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        meta.attrs["camera_depth_storage"] = "uint16/gzip-1"

    def _write_camera_meta_attrs(self, meta: h5py.Group) -> None:
        """Camera identity/geometry attrs from _pending_meta (None entries skipped).

        Idempotent — _stop_episode_impl re-runs it so values supplied late
        still reach /meta after the initial lazy write.
        """
        p = self._pending_meta
        calib = p.get("calib")
        camera_name = p.get("camera_name")
        camera_serial = p.get("camera_serial")
        if camera_serial is not None:
            meta.attrs["camera_serial"] = str(camera_serial)
        if camera_name is not None:
            meta.attrs["camera_name"] = str(camera_name)
        if calib is not None and camera_name is not None:
            # Verify a supplied serial against the named calibration entry.
            calib_meta = calib.to_meta_dict(camera_name, expected_serial=camera_serial)
            meta.attrs["camera_serial"] = calib_meta.get("camera_serial", "")
            meta.attrs["camera_type"] = calib_meta.get("camera_type", "")
            if "camera_T_world_camera" in calib_meta:
                meta.attrs["camera_T_world_camera"] = calib_meta[
                    "camera_T_world_camera"
                ]
            if "camera_T_eef_camera" in calib_meta:
                meta.attrs["camera_T_eef_camera"] = calib_meta["camera_T_eef_camera"]

        camera_K = p.get("camera_K")
        if camera_K is not None:
            meta.attrs["camera_K"] = camera_K.flatten().tolist()

        # Raw uint16 depth units in meters (L515: 0.00025) — without this,
        # offline consumers cannot convert /depth correctly.
        depth_scale = p.get("depth_scale")
        if depth_scale is not None:
            meta.attrs["depth_scale"] = float(depth_scale)

    def add_frame(
        self,
        state: RobotState,
        action: RobotAction,
        vr_frame: Mapping[str, object],
        camera_frame: Mapping[str, object] | None = None,
        signals: Mapping[str, object] | None = None,
        arm_qpos_sent: np.ndarray | None = None,
        diagnostics: Mapping[str, object] | None = None,
        control_run_generation: int = 0,
    ) -> bool:
        """Compatibility adapter from component inputs to :class:`EpisodeFrame`."""
        if not self._accept_source_frame():
            return False
        frame = build_episode_frame(
            state,
            action,
            vr_frame,
            camera_frame=camera_frame,
            signals=signals,
            arm_qpos_sent=arm_qpos_sent,
            diagnostics=diagnostics,
            control_run_generation=control_run_generation,
            arm_sent_stream=self.arm_sent_stream,
        )
        return self._add_episode_frame(frame)

    def add_episode_frame(self, frame: EpisodeFrame) -> bool:
        """Align and serialize one already-normalized recording frame."""
        if not self._accept_source_frame():
            return False
        return self._add_episode_frame(frame)

    def _accept_source_frame(self) -> bool:
        """Apply lifecycle, capacity, and initial-skip admission policy."""
        if not self._recording or self._buffer is None:
            return False

        if self._frame_count >= self.max_frames:
            logger.warning(
                "Episode reached max_frames=%d, auto-stopping.", self.max_frames
            )
            self._max_frames_reached = True
            return False

        if self._skipped_so_far < self._skip_initial_frames:
            self._skipped_so_far += 1
            return False
        return True

    def _add_episode_frame(self, frame: EpisodeFrame) -> bool:
        """Write an admitted typed frame to the aligned control grid."""
        assert self._buffer is not None
        ts = frame.timestamp_s
        run_generation = frame.control_run_generation
        # The first source and each quiescence boundary start a wall-time segment.
        if (
            self._last_control_run_generation is None
            or run_generation != self._last_control_run_generation
        ):
            self._buffer.reanchor(ts)

        data = frame.data
        source_layout_errors = validate_source_frame_keys_v17(
            set(data), arm_sent_stream=self.arm_sent_stream
        )
        if source_layout_errors:
            raise RuntimeError(
                "episode source frame mismatch: " + "; ".join(source_layout_errors)
            )

        add_result = self._buffer.add(data, timestamp=ts)
        if add_result.source_written:
            self._last_control_run_generation = run_generation

        self._frame_count = add_result.size
        prev_size = add_result.previous_size
        k = add_result.slots_written  # grid slots advanced (usually 1; 0 = dup bucket)

        # A live camera observation may cross skipped deadlines; only its causal slot is fresh.
        if k > 0:
            self._update_aligned_causality(slice(prev_size, self._buffer.size))

        if self._buffer.size - self._flushed_frames >= self._flush_interval:
            self._ensure_hdf5()
            self._flush_buffered()

        if k > 0:
            if not self._submit_aligned_camera_frames(frame, prev_size):
                return False

        if add_result.capacity_reached:
            self._max_frames_reached = True
            logger.info(
                "Episode reached max_frames=%d after aligned camera submission",
                self.max_frames,
            )
            return False
        return True

    def _update_aligned_causality(self, new_slice: slice) -> None:
        """Recompute causal metadata for source and synthetic grid slots."""
        assert self._buffer is not None
        buffer_data = self._buffer.data
        source_valid = np.asarray(
            buffer_data["flag_sample_valid"][new_slice], dtype=bool
        )
        grid_anchor_ns = np.rint(self._buffer.timestamps[new_slice] * 1e9).astype(
            np.uint64
        )
        buffer_data["observation_anchor_monotonic_ns"][new_slice] = grid_anchor_ns
        history_valid = np.asarray(
            buffer_data["observation_history_valid_mask"][new_slice, :, 0],
            dtype=bool,
        )
        source_monotonic_ns = np.column_stack(
            [
                buffer_data[f"{name}_source_monotonic_ns"][new_slice]
                for name in ("arm", "hand", "vr", "camera")
            ]
        ).astype(np.uint64)
        source_age_s = np.full(history_valid.shape, np.nan, dtype=np.float64)
        causal = history_valid & (source_monotonic_ns <= grid_anchor_ns[:, None])
        age_delta_ns = grid_anchor_ns[:, None].astype(
            np.float64
        ) - source_monotonic_ns.astype(np.float64)
        source_age_s[causal] = age_delta_ns[causal] / 1e9
        buffer_data["observation_source_age_s"][new_slice] = source_age_s
        buffer_data["observation_valid"][new_slice] &= source_valid
        buffer_data["tactile_fresh"][new_slice] &= source_valid
        buffer_data["flag_camera_fresh"][new_slice] &= source_valid

        # Synthetic slots inherit the effective target but do not claim a send.
        hold_slots = ~source_valid
        buffer_data["flag_action_queued"][new_slice] &= source_valid
        for name in (
            "action_id",
            "action_created_monotonic_ns",
            "action_target_monotonic_ns",
            "action_valid_until_monotonic_ns",
        ):
            buffer_data[name][new_slice][hold_slots] = 0

    def _submit_aligned_camera_frames(
        self, frame: EpisodeFrame, previous_size: int
    ) -> bool:
        """Submit one payload per newly materialized causal grid slot."""
        assert self._buffer is not None
        writer = self._camera_writer
        if writer is None:
            logger.error("EpisodeRecorder: camera writer missing during add_frame")
            return False
        current_payload = self._camera_payload(frame)
        zero_payload = self._camera_payload(None)
        sample_valid_slots = self._buffer.data["flag_sample_valid"][
            previous_size : self._buffer.size
        ]
        for sample_valid in sample_valid_slots:
            if sample_valid:
                payload = current_payload
                self._last_camera_payload = (
                    np.array(current_payload[0], copy=True),
                    np.array(current_payload[1], copy=True),
                )
            else:
                payload = self._last_camera_payload or zero_payload
            if not writer.submit(*payload):
                return False
        return True

    def _camera_payload(
        self, frame: EpisodeFrame | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return shape-stable camera arrays, using explicit zero placeholders."""
        cfg = self._camera_writer_config
        if frame is None:
            return (
                np.zeros(cfg.rgb_shape, dtype=np.uint8),
                np.zeros(cfg.depth_shape, dtype=np.uint16),
            )

        rgb = frame.camera_rgb
        depth = frame.camera_depth
        if rgb is None:
            rgb = np.zeros(cfg.rgb_shape, dtype=np.uint8)
        if depth is None:
            depth = np.zeros(cfg.depth_shape, dtype=np.uint16)
        return rgb, depth

    def _ensure_hdf5(self) -> None:
        """Lazily create ``data.h5`` in the temp directory."""
        if self._file is not None:
            return
        if self._temp_dir is None:
            raise RuntimeError("EpisodeRecorder: _temp_dir is None during HDF5 open")
        self._file = h5py.File(str(Path(self._temp_dir) / "data.h5"), "w")
        self._write_meta_attrs(self._file.create_group("meta"))

    def _flush_buffered(self) -> None:
        """Write buffered non-camera streams to HDF5, keeping datasets resizable.

        Called periodically during recording (every ``_flush_interval`` frames)
        and finally at :meth:`stop_episode`.  On first call the datasets are
        created with ``maxshape=(None, ...)``; subsequent calls resize and
        append only the new frames.
        """
        if self._buffer is None or self._buffer.size == self._flushed_frames:
            return

        self._ensure_hdf5()
        buf_data = self._buffer.data
        buf_size = self._buffer.size
        new_start = self._flushed_frames

        # Validate the schema before creating or extending HDF5 datasets.
        buffer_shapes = {name: tuple(values.shape) for name, values in buf_data.items()}
        buffer_dtypes = {name: values.dtype for name, values in buf_data.items()}
        buffer_shapes["timestamp"] = tuple(self._buffer.timestamps.shape)
        buffer_dtypes["timestamp"] = self._buffer.timestamps.dtype
        layout_errors = validate_data_layout_v17(
            buffer_shapes,
            buffer_dtypes,
            frame_count=buf_size,
            arm_sent_stream=self.arm_sent_stream,
        )
        if layout_errors:
            raise RuntimeError(
                "episode recorder buffer mismatch: " + "; ".join(layout_errors)
            )

        for h5_key, arr in buf_data.items():
            if h5_key not in self._datasets:
                self._datasets[h5_key] = self._file.create_dataset(
                    h5_key,
                    data=arr[:buf_size].copy(),
                    maxshape=(None,) + arr.shape[1:],
                    dtype=arr.dtype,
                    compression="gzip",
                )
            else:
                ds = self._datasets[h5_key]
                ds.resize(buf_size, axis=0)
                ds[new_start:buf_size] = arr[new_start:buf_size]

        ts = self._buffer.timestamps
        if "timestamp" not in self._datasets:
            self._datasets["timestamp"] = self._file.create_dataset(
                "timestamp",
                data=ts[:buf_size].copy(),
                maxshape=(None,),
                dtype=np.float64,
                compression="gzip",
            )
        else:
            ts_ds = self._datasets["timestamp"]
            ts_ds.resize(buf_size, axis=0)
            ts_ds[new_start:buf_size] = ts[new_start:buf_size]

        self._flushed_frames = buf_size

    def stop_episode(self, success: bool = True, reason: str = "") -> str | None:
        """Signal end of episode; return path immediately, flush in background.

        The heavy work (camera-writer drain, buffer flush, metadata write,
        file close) runs on a joinable thread so the control
        loop stays responsive.  Callers must join_stop() before relying on
        the file (or before process exit).

        Args:
            success: stored as /meta success.
            reason: stored as /meta stop_reason; empty → "max_frames" when
                    the episode hit the frame cap, else "manual".
        """
        if not self._recording:
            return None

        truncated = self._max_frames_reached

        # Mark stopped before spawning the thread so add_frame() rejects new frames.
        self._recording = False
        self._max_frames_reached = False
        path = self._episode_dir

        # Snapshot stop metadata BEFORE spawning the worker (it overwrites
        # self._frame_count during _stop_episode_impl_inner).
        self._stop_success = success
        self._stop_path = path
        self._stop_frame_count = self._frame_count
        runtime_quality_metrics = dict(self._runtime_quality_metrics)

        t = threading.Thread(
            target=self._stop_episode_impl,
            args=(success, reason, truncated, runtime_quality_metrics),
            daemon=False,
            name="episode-stop",
        )
        t.start()
        self._stop_thread = t
        return path

    def join_stop(self, timeout: float = 30.0) -> bool:
        """Wait for the background stop daemon (HDF5 fully written + closed).

        Returns True when the flush completed cleanly.  Returns False on timeout
        (thread still alive — handle KEPT so start_episode() keeps refusing) OR
        when the stop thread crashed (handle cleared, _stop_error set — caller
        must inspect stop_error to distinguish).

        The entry MUST consult stop_error after a join_stop() that returned True:
        a True from a crashed thread means "no pending flush" (the daemon is dead
        and can't be re-joined), NOT "file written successfully".
        """
        if not np.isfinite(timeout) or timeout < 0:
            raise ValueError("episode stop timeout must be finite and non-negative")
        t = self._stop_thread
        if t is None:
            return self._stop_error is None
        if t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(
                    "episode-stop still flushing after %.0fs — keeping handle", timeout
                )
                return False
        ok = self._stop_error is None
        if self._stop_error is not None:
            logger.error("episode-stop failed: %s", self._stop_error)
        self._stop_thread = None
        self._stop_success = False
        self._stop_path = None
        self._stop_frame_count = 0
        return ok

    def poll_stop(self) -> StopResult:
        """Non-blocking check: has the background stop daemon finished?

        Safe to call once per configured control-grid tick. Returns immediately
        — never blocks on I/O. After the first call that returns
        ``done=True``, the internal state is reset and subsequent calls
        return a clean sentinel (``done=True, path=None``) until the next
        ``stop_episode()``.

        The terminal payload is consumptive: only the first completed poll
        carries its path, frame count, success flag, and error.
        """
        t = self._stop_thread
        if t is None:
            return StopResult(
                done=True,
                error=self._stop_error,
                success=self._stop_success,
                path=self._stop_path,
                frame_count=self._stop_frame_count,
            )
        if t.is_alive():
            return StopResult(done=False)
        result = StopResult(
            done=True,
            error=self._stop_error,
            success=self._stop_success,
            path=self._stop_path,
            frame_count=self._stop_frame_count,
        )
        self._stop_thread = None
        self._stop_error = None
        self._stop_success = False
        self._stop_path = None
        self._stop_frame_count = 0
        return result

    def _stop_episode_impl(
        self,
        success: bool,
        reason: str,
        truncated: bool,
        runtime_quality_metrics: dict[str, int],
    ) -> None:
        """Background: finalize sidecars, flush buffers, write metadata, and publish.

        ENOSPC / OSError at any h5py call site is captured into ``_stop_error``
        so ``join_stop()`` and RecorderIO report failure instead of publishing
        or announcing a truncated episode.
        """
        try:
            self._stop_episode_impl_inner(
                success, reason, truncated, runtime_quality_metrics
            )
        except Exception as exc:
            self._stop_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "stop_episode failed: %s — HDF5 may be truncated", self._stop_error
            )
            try:
                self._write_aborted_manifest(reason=reason, error=self._stop_error)
            except Exception:
                logger.error(
                    "failed to publish aborted episode manifest", exc_info=True
                )
            try:
                if self._camera_writer is not None:
                    self._camera_writer.close(timeout=5.0)
            except Exception:
                logger.warning(
                    "camera writer cleanup failed after episode stop error",
                    exc_info=True,
                )
            self._camera_writer = None
            try:
                if self._file is not None:
                    self._file.close()
            except Exception:
                logger.warning(
                    "HDF5 cleanup failed after episode stop error", exc_info=True
                )
            self._file = None
        finally:
            # Always clean up the temp directory and reset state after stopping.
            _tmp = self._temp_dir
            if _tmp is not None:
                self._discard_temp_files(_tmp)
            self._reset_episode_state()

    def _stop_episode_impl_inner(
        self,
        success: bool,
        reason: str,
        truncated: bool,
        runtime_quality_metrics: dict[str, int],
    ) -> None:
        """Inner body of _stop_episode_impl — extracted so the try/except wrapper
        can reset state on any exception without duplicating the reset list."""
        duration = time.perf_counter() - (self._start_time or 0.0)

        writer = self._camera_writer
        if writer is None:
            raise RuntimeError("camera writer missing at episode stop")
        writer.close(timeout=_CAMERA_WRITER_CLOSE_TIMEOUT_S)
        camera_frame_count = writer.frame_count
        self._camera_writer_metrics = writer.metrics
        self._camera_writer = None

        self._flush_buffered()
        buf_size = self._buffer.size if self._buffer is not None else 0
        self._ensure_hdf5()

        if camera_frame_count != buf_size:
            raise RuntimeError(
                f"camera/control grid length mismatch: camera={camera_frame_count}, control={buf_size}"
            )

        self._buffer = None
        self._frame_count = buf_size
        quality_metrics = _episode_quality_metrics(
            self._datasets,
            frame_count=self._frame_count,
            control_hz=self.control_hz,
            runtime_metrics=runtime_quality_metrics,
        )

        _had_rgb = camera_frame_count > 0

        if self._file is not None:
            meta = self._file["meta"]
            grid_dt_s = 1.0 / self.control_hz
            grid_duration_s = max(0, self._frame_count - 1) * grid_dt_s
            meta.attrs["schema_version"] = SCHEMA_VERSION
            # ``duration`` remains wall-clock time; explicit grid fields keep
            # pauses and other non-sampled time distinct from the control rate.
            meta.attrs["duration"] = duration
            meta.attrs["wall_duration_s"] = duration
            meta.attrs["grid_duration_s"] = grid_duration_s
            meta.attrs["grid_dt_s"] = grid_dt_s
            meta.attrs["non_sampled_duration_s"] = max(0.0, duration - grid_duration_s)
            meta.attrs["num_frames"] = self._frame_count
            meta.attrs["success"] = success
            meta.attrs["fps"] = self.control_hz
            meta.attrs["wall_fps"] = (
                self._frame_count / duration if duration > 0 else self.control_hz
            )
            meta.attrs["min_frames_met"] = self._frame_count >= self.min_frames
            meta.attrs["has_camera"] = _had_rgb
            meta.attrs["has_timestamps"] = "timestamp" in self._datasets
            meta.attrs["camera_stream_frames"] = camera_frame_count
            meta.attrs["camera_writer_error"] = ""
            for metric_name, metric_value in self._camera_writer_metrics.items():
                meta.attrs[metric_name] = metric_value
            for metric_name, metric_value in quality_metrics.items():
                meta.attrs[metric_name] = int(metric_value)
            meta.attrs["truncated"] = bool(truncated)
            meta.attrs["stop_reason"] = reason or (
                "max_frames" if truncated else "manual"
            )
            # Backfill camera metadata if the child connected after the lazy write.
            self._write_camera_meta_attrs(meta)

        if self._file is not None:
            self._file.close()
        self._file = None
        # Atomically rename the temporary directory into its final location.
        _final = self._episode_dir
        _tmp = self._temp_dir
        if _tmp is not None and _final is not None:
            if success:
                self._validate_and_sync_temp_episode(Path(_tmp), self._frame_count)
                atomic_publish(_tmp, _final)
                logger.info(
                    "Episode quality: path=%s frames=%d ik_hold=%d camera_invalid=%d "
                    "observation_invalid=%d "
                    "sample_invalid=%d safety_reject=%d quiescence=%d "
                    "hand_read_errors=%d hand_overcurrent=%d",
                    _final,
                    self._frame_count,
                    quality_metrics["ik_hold_frame_count"],
                    quality_metrics["camera_invalid_frame_count"],
                    quality_metrics["observation_invalid_frame_count"],
                    quality_metrics["sample_invalid_frame_count"],
                    quality_metrics["safety_reject_frame_count"],
                    quality_metrics["command_quiescence_count"],
                    quality_metrics.get("hand_read_error_count", 0),
                    quality_metrics.get("hand_overcurrent_count", 0),
                )
            else:
                self._write_aborted_manifest(reason=reason or "discarded", error="")
                self._discard_temp_files(_tmp)

        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        """Reset all mutable episode state to defaults (called from both the
        success path and the crash-handler in _stop_episode_impl)."""
        self._datasets.clear()
        self._recording = False
        self._max_frames_reached = False
        self._frame_count = 0
        self._start_time = None
        self._episode_dir = None
        self._temp_dir = None
        self._buffer = None
        self._flushed_frames = 0
        self._camera_writer = None
        self._camera_writer_metrics = {}
        self._last_camera_payload = None
        self._last_control_run_generation = None
        self._runtime_quality_metrics = {}

    # ── Atomic file finalisation ──────────────────────────────────────

    def _write_aborted_manifest(self, *, reason: str, error: str) -> Path:
        """Persist only small failure provenance; never retain partial payloads."""
        episode_name = Path(self._episode_dir or "aborted_episode_unknown").name
        target = self.data_dir / f"{episode_name}.aborted.json"
        suffix = 1
        while target.exists():
            target = self.data_dir / f"{episode_name}.{suffix}.aborted.json"
            suffix += 1
        payload = {
            "episode": episode_name,
            "status": "aborted",
            "reason": reason,
            "error": error,
            "frame_count_before_abort": int(self._frame_count),
            "created_wall_time_ns": time.time_ns(),
            "resolved_config_sha256": self._resolved_config_hash or "",
        }
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.tmp-", dir=self.data_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    payload,
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temp_name, target)
            dir_fd = os.open(self.data_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return target

    @staticmethod
    def _validate_and_sync_temp_episode(temp_dir: Path, expected_frames: int) -> None:
        """Reopen/decode all three modalities, then fsync before publication."""
        paths = {
            "data": temp_dir / "data.h5",
            "depth": temp_dir / "depth.h5",
            "rgb": temp_dir / "rgb.mp4",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f"episode finalization missing modalities: {missing}")
        with h5py.File(paths["data"], "r") as data_h5:
            meta = data_h5.get("meta")
            if meta is None or int(meta.attrs.get("num_frames", -1)) != expected_frames:
                raise RuntimeError("data.h5 frame count metadata mismatch")
            dataset_shapes = {
                key: tuple(dataset.shape)
                for key, dataset in data_h5.items()
                if isinstance(dataset, h5py.Dataset)
            }
            dataset_dtypes = {
                key: dataset.dtype
                for key, dataset in data_h5.items()
                if isinstance(dataset, h5py.Dataset)
            }
            layout_errors = validate_data_layout_v17(
                dataset_shapes,
                dataset_dtypes,
                frame_count=expected_frames,
                arm_sent_stream=bool(meta.attrs.get(ARM_SENT_MARKER, False)),
            )
            if layout_errors:
                raise RuntimeError(
                    "data.h5 episode layout mismatch: " + "; ".join(layout_errors)
                )
        for key in ("depth",):
            with h5py.File(paths[key], "r") as sidecar:
                if key not in sidecar or int(sidecar[key].shape[0]) != expected_frames:
                    raise RuntimeError(f"{key} sidecar length mismatch")
        with VideoDecoder(paths["rgb"]) as decoder:
            decoded_frames = decoder.count_decoded_frames()
            if decoded_frames != expected_frames:
                raise RuntimeError(
                    f"RGB decoded frame count {decoded_frames} != {expected_frames}"
                )
        for path in paths.values():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        dir_fd = os.open(temp_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    @staticmethod
    def _discard_temp_files(tmp: str) -> None:
        """Remove temp directory and all contents. Never raises."""
        shutil.rmtree(tmp, ignore_errors=True)
