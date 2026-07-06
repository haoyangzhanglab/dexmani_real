"""EpisodeRecorder — HDF5-based teleoperation data recorder.

Supports two write modes:
  - Per-frame append (default):  h5py.Dataset.resize() on every add_frame().
  - Timestamp-aligned buffer:    all non-camera streams aligned to a unified
    dt=20ms time grid at record time, flushed to HDF5 in bulk at stop_episode().
"""

from __future__ import annotations

__all__ = ["EpisodeRecorder"]

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.pipeline_config import DEFAULT_MAX_RECORD_FRAMES
from dexmani_real.recording.timestamp_buffer import TimestampAlignedBuffer
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class EpisodeRecorder:
    """Records teleoperation episodes to HDF5 files.

    Lifecycle: start_episode() → add_frame() × N → stop_episode()
    """

    def __init__(
        self,
        data_dir: str,
        max_frames: int = DEFAULT_MAX_RECORD_FRAMES,
        use_timestamp_buffer: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.use_timestamp_buffer = use_timestamp_buffer

        self._file: h5py.File | None = None
        self._frame_count: int = 0
        self._recording: bool = False
        self._max_frames_reached: bool = False
        self._start_time: float | None = None
        self._episode_path: str | None = None
        self._datasets: dict[str, Any] = {}

        # Timestamp-aligned buffer (only when use_timestamp_buffer=True)
        self._buffer: TimestampAlignedBuffer | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def max_frames_reached(self) -> bool:
        return self._max_frames_reached

    def start_episode(
        self,
        task_label: str = "",
        operator: str = "",
        tags: list[str] | None = None,
        calib: CameraCalib | None = None,
        camera_K: np.ndarray | None = None,
        camera_name: str | None = None,
    ) -> bool:
        if self._recording:
            return False

        idx = 0
        while (self.data_dir / f"episode_{idx:03d}.h5").exists():
            idx += 1

        path = self.data_dir / f"episode_{idx:03d}.h5"
        self._episode_path = str(path)
        self._frame_count = 0
        self._max_frames_reached = False
        self._start_time = time.perf_counter()
        self._recording = True
        self._datasets = {}

        # Store metadata for deferred write (buffer mode) or immediate (append mode)
        self._pending_meta: dict[str, Any] = {
            "task_label": task_label,
            "operator": operator,
            "tags": tags,
            "calib": calib,
            "camera_K": camera_K,
            "camera_name": camera_name,
        }

        if self.use_timestamp_buffer:
            # Defer HDF5 creation to stop_episode(); start the aligned buffer.
            dt = 1.0 / 50.0
            buffer_steps = self.max_frames + 100  # margin for grid-boundary alignment
            self._buffer = TimestampAlignedBuffer(
                start_time=self._start_time,
                dt=dt,
                max_record_steps=buffer_steps,
            )
            self._file = None
            self._camera_datasets_created: bool = False
            return True

        # Per-frame append mode: create HDF5 immediately.
        self._file = h5py.File(str(path), "w")
        self._write_meta_attrs(self._file.create_group("meta"))
        self._file.create_group("obs")
        self._file.create_group("action")
        self._file.create_group("vr")

        return True

    def _write_meta_attrs(self, meta: h5py.Group) -> None:
        """Write deferred metadata attributes to the HDF5 meta group."""
        p = self._pending_meta
        meta.attrs["task_label"] = p.get("task_label", "")
        meta.attrs["operator"] = p.get("operator", "")
        tags = p.get("tags")
        meta.attrs["tags"] = ",".join(tags) if tags else ""
        meta.attrs["fps"] = 50.0

        calib = p.get("calib")
        camera_name = p.get("camera_name")
        if calib is not None and camera_name is not None:
            calib_meta = calib.to_meta_dict(camera_name)
            meta.attrs["camera_serial"] = calib_meta.get("camera_serial", "")
            meta.attrs["camera_type"] = calib_meta.get("camera_type", "")
            if "camera_T_base_camera" in calib_meta:
                meta.attrs["camera_T_base_camera"] = calib_meta["camera_T_base_camera"]
            if "camera_T_eef_camera" in calib_meta:
                meta.attrs["camera_T_eef_camera"] = calib_meta["camera_T_eef_camera"]

        camera_K = p.get("camera_K")
        if camera_K is not None:
            meta.attrs["camera_K"] = camera_K.flatten().tolist()

    def add_frame(
        self,
        state,
        action,
        vr_frame: dict[str, Any],
        camera_frame: dict[str, Any] | None = None,
        T_base_eef: np.ndarray | None = None,
        camera_frames: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        if not self._recording:
            return False

        if self._frame_count >= self.max_frames:
            logger.warning("Episode reached max_frames=%d, auto-stopping.", self.max_frames)
            self._max_frames_reached = True
            return False

        # ── Buffer mode: accumulate non-camera streams, write camera per-frame ──
        if self.use_timestamp_buffer:
            return self._add_frame_buffered(
                state, action, vr_frame, camera_frame, T_base_eef, camera_frames
            )

        # ── Per-frame append mode (original path) ──
        return self._add_frame_append(
            state, action, vr_frame, camera_frame, T_base_eef, camera_frames
        )

    def _add_frame_buffered(
        self,
        state,
        action,
        vr_frame: dict[str, Any],
        camera_frame: dict[str, Any] | None,
        T_base_eef: np.ndarray | None,
        camera_frames: dict[str, dict[str, Any]] | None,
    ) -> bool:
        """Buffer mode: align non-camera streams to time grid, write camera per-frame."""
        assert self._buffer is not None

        # Build flat data dict for the aligned buffer
        ts = time.perf_counter()
        data: dict[str, np.ndarray | float] = {
            "obs/arm_qpos": np.asarray(state.arm_qpos, dtype=np.float64),
            "obs/arm_qvel": np.asarray(state.arm_qvel, dtype=np.float64),
            "obs/arm_tau": np.asarray(state.arm_tau, dtype=np.float64),
            "obs/eef_pos": np.asarray(state.eef_pos, dtype=np.float64),
            "obs/eef_quat": np.asarray(state.eef_quat_wxyz, dtype=np.float64),
            "obs/hand_qpos": np.asarray(state.hand_qpos, dtype=np.float64),
            "obs/hand_tactile_sum": np.asarray(state.hand_tactile_sum, dtype=np.float64),
            "obs/hand_tactile_force": np.asarray(state.hand_tactile_force, dtype=np.float64),
            "action/arm_qpos": np.asarray(action.arm_qpos_cmd, dtype=np.float64),
            "action/hand_qpos": np.asarray(action.hand_qpos_cmd, dtype=np.float64),
            "vr/wrist_pos": np.asarray(vr_frame["wrist_pos"], dtype=np.float64),
            "vr/wrist_quat": np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64),
            "vr/landmarks": np.asarray(vr_frame["landmarks"], dtype=np.float64),
            "vr_timestamps": float((vr_frame.get("local_recv_ns") or 0) * 1e-9),
        }

        self._buffer.add(data, timestamp=ts)

        # Check buffer capacity
        if self._buffer.recording_stopped:
            self._max_frames_reached = True
            return False

        self._frame_count = self._buffer.size

        # Camera data: create HDF5 lazily on first camera frame, write per-frame
        has_camera = (
            camera_frame is not None
            or (camera_frames is not None and len(camera_frames) > 0)
        )
        if has_camera:
            self._ensure_hdf5_for_camera()
            return self._add_frame_append(
                state, action, vr_frame, camera_frame, T_base_eef, camera_frames,
                increment_frame_count=False,  # buffer tracks frame_count from buffer.size
            )
        return True

    def _ensure_hdf5_for_camera(self) -> None:
        """Lazily create the HDF5 file (in buffer mode) for per-frame camera writes."""
        if self._file is not None:
            return
        assert self._episode_path is not None
        self._file = h5py.File(str(self._episode_path), "w")
        self._write_meta_attrs(self._file.create_group("meta"))
        self._file.create_group("obs")
        self._file.create_group("action")
        self._file.create_group("vr")
        self._camera_datasets_created = False

    def _add_frame_append(
        self,
        state,
        action,
        vr_frame: dict[str, Any],
        camera_frame: dict[str, Any] | None,
        T_base_eef: np.ndarray | None,
        camera_frames: dict[str, dict[str, Any]] | None,
        increment_frame_count: bool = True,
    ) -> bool:
        """Original per-frame append logic (used in both append and buffer modes)."""
        assert self._file is not None

        # Observation
        self._append_or_create("obs/arm_qpos", state.arm_qpos)
        self._append_or_create("obs/arm_qvel", state.arm_qvel)
        self._append_or_create("obs/arm_tau", state.arm_tau)
        self._append_or_create("obs/eef_pos", state.eef_pos)
        self._append_or_create("obs/eef_quat", state.eef_quat_wxyz)
        self._append_or_create("obs/hand_qpos", state.hand_qpos)
        self._append_or_create("obs/hand_tactile_sum", state.hand_tactile_sum)
        self._append_or_create("obs/hand_tactile_force", state.hand_tactile_force)

        # Action
        self._append_or_create("action/arm_qpos", action.arm_qpos_cmd)
        self._append_or_create("action/hand_qpos", action.hand_qpos_cmd)

        # VR
        self._append_or_create("vr/wrist_pos", vr_frame["wrist_pos"])
        self._append_or_create("vr/wrist_quat", vr_frame["wrist_quat_wxyz"])
        self._append_or_create("vr/landmarks", vr_frame["landmarks"])

        # Timestamps
        ts_arr = np.array([time.perf_counter()], dtype=np.float64)
        if "timestamps" not in self._datasets:
            self._datasets["timestamps"] = self._file.create_dataset(
                "timestamps",
                data=ts_arr,
                maxshape=(None,),
                chunks=True,
                dtype=np.float64,
            )
        else:
            self._resize_append("timestamps", ts_arr)

        vr_ts = np.array([(vr_frame.get("local_recv_ns") or 0) * 1e-9], dtype=np.float64)
        if "vr_timestamps" not in self._datasets:
            self._datasets["vr_timestamps"] = self._file.create_dataset(
                "vr_timestamps",
                data=vr_ts,
                maxshape=(None,),
                chunks=True,
                dtype=np.float64,
            )
        else:
            self._resize_append("vr_timestamps", vr_ts)

        # Camera extrinsics
        if T_base_eef is not None:
            calib = self._file["meta"].attrs
            camera_type = calib.get("camera_type", "")
            if camera_type == "eye_in_hand":
                T_eef_camera_raw = calib.get("camera_T_eef_camera", None)
                if T_eef_camera_raw is not None:
                    T_eef_camera = np.array(T_eef_camera_raw, dtype=np.float64).reshape(4, 4)
                    T_base_camera = T_base_eef @ T_eef_camera
                else:
                    T_base_camera = T_base_eef
            else:
                T_base_camera = T_base_eef
            self._append_or_create("camera/extrinsics", T_base_camera)

        # Camera frames (single-camera)
        if camera_frame is not None:
            rgb = camera_frame.get("rgb")
            depth = camera_frame.get("depth")
            ts = camera_frame.get("timestamp", 0.0)
            if "camera/rgb" not in self._datasets:
                if rgb is not None:
                    self._datasets["camera/rgb"] = self._file.create_dataset(
                        "camera/rgb",
                        data=rgb[np.newaxis, ...],
                        maxshape=(None,) + rgb.shape,
                        chunks=True,
                        dtype=rgb.dtype,
                    )
                if depth is not None:
                    self._datasets["camera/depth"] = self._file.create_dataset(
                        "camera/depth",
                        data=depth[np.newaxis, ...],
                        maxshape=(None,) + depth.shape,
                        chunks=True,
                        dtype=depth.dtype,
                    )
                self._datasets["camera/timestamps"] = self._file.create_dataset(
                    "camera/timestamps",
                    data=[ts],
                    maxshape=(None,),
                    chunks=True,
                    dtype=np.float64,
                )
            else:
                if rgb is not None:
                    self._resize_append("camera/rgb", rgb)
                if depth is not None:
                    self._resize_append("camera/depth", depth)
                self._resize_append("camera/timestamps", np.array([ts]))

        # Multi-camera frames
        if camera_frames:
            for cam_name, cam_frame in camera_frames.items():
                if cam_frame is None:
                    continue
                safe_name = str(cam_name).replace("/", "_").replace("\\", "_")
                rgb = cam_frame.get("rgb")
                depth = cam_frame.get("depth")
                ts = cam_frame.get("timestamp", 0.0)

                rgb_key = f"camera/{safe_name}/rgb"
                depth_key = f"camera/{safe_name}/depth"
                ts_key = f"camera/{safe_name}/timestamps"

                if rgb_key not in self._datasets:
                    if rgb is not None and hasattr(rgb, "shape"):
                        self._datasets[rgb_key] = self._file.create_dataset(
                            rgb_key,
                            data=rgb[np.newaxis, ...],
                            maxshape=(None,) + rgb.shape,
                            chunks=True,
                            dtype=rgb.dtype if rgb.dtype == np.uint8 else np.uint8,
                        )
                    if depth is not None and hasattr(depth, "shape"):
                        self._datasets[depth_key] = self._file.create_dataset(
                            depth_key,
                            data=depth[np.newaxis, ...],
                            maxshape=(None,) + depth.shape,
                            chunks=True,
                            dtype=depth.dtype if depth.dtype == np.uint16 else np.uint16,
                        )
                    self._datasets[ts_key] = self._file.create_dataset(
                        ts_key,
                        data=np.array([ts]),
                        maxshape=(None,),
                        chunks=True,
                        dtype=np.float64,
                    )
                else:
                    if rgb is not None and hasattr(rgb, "shape"):
                        self._resize_append(rgb_key, rgb)
                    if depth is not None and hasattr(depth, "shape"):
                        self._resize_append(depth_key, depth)
                    self._resize_append(ts_key, np.array([ts]))

        if increment_frame_count:
            self._frame_count += 1
        return True

    def stop_episode(self, success: bool = True) -> str | None:
        if not self._recording:
            return None

        duration = time.perf_counter() - (self._start_time or 0.0)

        # ── Buffer mode: flush aligned data to HDF5 ──
        if self.use_timestamp_buffer and self._buffer is not None:
            buf_data = self._buffer.data
            buf_size = self._buffer.size
            # Ensure HDF5 file exists (may not if no camera frames were recorded)
            self._ensure_hdf5_for_camera()
            if self._file is None:
                # No camera frames at all — create HDF5 now for the buffer data
                assert self._episode_path is not None
                self._file = h5py.File(str(self._episode_path), "w")
                self._write_meta_attrs(self._file.create_group("meta"))
                self._file.create_group("obs")
                self._file.create_group("action")
                self._file.create_group("vr")

            # Write buffered non-camera streams as static datasets
            for key, arr in buf_data.items():
                # Determine group from key prefix
                if key.startswith("obs/"):
                    h5_key = key
                elif key.startswith("action/"):
                    h5_key = key
                elif key.startswith("vr_timestamps"):
                    h5_key = "vr_timestamps"
                elif key.startswith("vr/"):
                    h5_key = key
                else:
                    h5_key = key

                # Only write if not already written by camera path
                if h5_key not in self._datasets:
                    self._datasets[h5_key] = self._file.create_dataset(
                        h5_key, data=arr, dtype=arr.dtype
                    )

            # Write aligned timestamps
            if "timestamps" not in self._datasets:
                self._datasets["timestamps"] = self._file.create_dataset(
                    "timestamps", data=self._buffer.timestamps, dtype=np.float64
                )

            self._buffer = None
            self._frame_count = buf_size

        # ── Write final metadata (both modes) ──
        if self._file is not None:
            meta = self._file["meta"]
            meta.attrs["duration"] = duration
            meta.attrs["num_frames"] = self._frame_count
            meta.attrs["success"] = success
            meta.attrs["fps"] = self._frame_count / duration if duration > 0 else 50.0
            meta.attrs["min_frames_met"] = self._frame_count >= 50
            meta.attrs["has_camera"] = "camera/timestamps" in self._file
            meta.attrs["has_timestamps"] = "timestamps" in self._file

        path = self._episode_path
        if self._file is not None:
            self._file.close()
        self._file = None
        self._datasets.clear()
        self._recording = False
        self._max_frames_reached = False
        self._frame_count = 0
        self._start_time = None
        self._episode_path = None
        self._buffer = None
        return path

    def _append_or_create(self, key: str, data: np.ndarray) -> None:
        arr = np.asarray(data)
        if key not in self._datasets:
            self._datasets[key] = self._file.create_dataset(
                key,
                data=arr[np.newaxis, ...],
                maxshape=(None,) + arr.shape,
                chunks=True,
                dtype=arr.dtype,
            )
        else:
            self._resize_append(key, arr)

    def _resize_append(self, key: str, data: np.ndarray) -> None:
        ds = self._datasets[key]
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = data
