"""EpisodeRecorder — HDF5-based teleoperation data recorder.

Single write path: state/action/vr streams are aligned to a fixed dt=20ms time
grid at record time (TimestampAlignedBuffer) and flushed to HDF5 in bulk at
stop_episode(); camera frames are streamed per-frame but kept length-aligned to
the same grid, so every dataset is index-aligned by construction.
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
from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d
from dexmani_real.recording.timestamp_buffer import TimestampAlignedBuffer
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 3  # v3: /meta camera_T_base_camera → camera_T_world_camera (world frame)


class EpisodeRecorder:
    """Records teleoperation episodes to HDF5 files.

    Lifecycle: start_episode() → add_frame() × N → stop_episode()
    """

    def __init__(
        self,
        data_dir: str,
        max_frames: int = DEFAULT_MAX_RECORD_FRAMES,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames

        self._file: h5py.File | None = None
        self._frame_count: int = 0
        self._recording: bool = False
        self._max_frames_reached: bool = False
        self._start_time: float | None = None
        self._episode_path: str | None = None
        self._datasets: dict[str, Any] = {}

        # Record-time aligned buffer for non-camera streams.
        self._buffer: TimestampAlignedBuffer | None = None

        # Camera length-alignment: cache last frame to forward-fill on None reads
        # / dropped grid slots so camera datasets stay index-aligned with the grid.
        self._cam_seen: bool = False
        self._last_camera_frame: dict[str, Any] | None = None
        self._last_camera_frames: dict[str, dict[str, Any]] | None = None
        self._last_T_base_eef: np.ndarray | None = None

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
        self._cam_seen = False
        self._last_camera_frame = None
        self._last_camera_frames = None
        self._last_T_base_eef = None

        # Store metadata for deferred write (HDF5 is created lazily).
        self._pending_meta: dict[str, Any] = {
            "task_label": task_label,
            "operator": operator,
            "tags": tags,
            "calib": calib,
            "camera_K": camera_K,
            "camera_name": camera_name,
        }

        # Defer HDF5 creation to the first camera frame / stop_episode();
        # start the record-time aligned buffer for non-camera streams.
        dt = 1.0 / 50.0
        buffer_steps = self.max_frames + 100  # margin for grid-boundary alignment
        self._buffer = TimestampAlignedBuffer(
            start_time=self._start_time,
            dt=dt,
            max_record_steps=buffer_steps,
        )
        self._file = None
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
            # If the live camera serial was supplied, verify it matches the named
            # calibration entry — a wrong camera_name would otherwise silently
            # embed the wrong extrinsics/serial into the dataset.
            calib_meta = calib.to_meta_dict(camera_name, expected_serial=p.get("camera_serial"))
            meta.attrs["camera_serial"] = calib_meta.get("camera_serial", "")
            meta.attrs["camera_type"] = calib_meta.get("camera_type", "")
            if "camera_T_world_camera" in calib_meta:
                meta.attrs["camera_T_world_camera"] = calib_meta["camera_T_world_camera"]
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
        signals: dict[str, Any] | None = None,
    ) -> bool:
        if not self._recording or self._buffer is None:
            return False

        if self._frame_count >= self.max_frames:
            logger.warning("Episode reached max_frames=%d, auto-stopping.", self.max_frames)
            self._max_frames_reached = True
            return False

        # ── Non-camera streams → record-time aligned buffer ──
        sig = signals or {}

        def _make_action_ee() -> np.ndarray:
            pos = action.target_eef_pos
            rot6d = action.target_eef_rot6d
            p = np.asarray(pos, dtype=np.float64) if pos is not None else np.full(3, np.nan)
            r = np.asarray(rot6d, dtype=np.float64) if rot6d is not None else np.full(6, np.nan)
            return np.concatenate([p, r])

        data: dict[str, np.ndarray | float] = {
            # ── Observables ──
            "arm_qpos": np.asarray(state.arm_qpos, dtype=np.float64),
            "arm_ee": np.concatenate([state.eef_pos, state.eef_rot6d]).astype(np.float64),
            "arm_qvel": np.asarray(state.arm_qvel, dtype=np.float64),
            "arm_tau": np.asarray(state.arm_tau, dtype=np.float64),
            "hand_qpos": np.asarray(state.hand_qpos, dtype=np.float64),
            "hand_fingertip": np.asarray(state.fingertip_pos, dtype=np.float64),
            "hand_contact": np.asarray(state.hand_tactile_sum, dtype=np.float64),
            # ── Actions ──
            "action_arm_joint": np.asarray(action.arm_qpos_cmd, dtype=np.float64),
            "action_arm_ee": _make_action_ee(),
            "action_hand_joint": np.asarray(action.hand_qpos_cmd, dtype=np.float64),
            # ── Flags ──
            "flag_ik_ok": bool(sig.get("ik_ok", False)),
            "flag_retarget_ok": bool(sig.get("retarget_ok", False)),
            "flag_held": bool(sig.get("held", False)),
            # ── VR ──
            "vr_wrist_pos": np.asarray(vr_frame["wrist_pos"], dtype=np.float64),
            "vr_wrist_rot6d": quat_wxyz_to_rot6d(
                np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)
            ),
            "vr_landmarks": np.asarray(vr_frame["landmarks"], dtype=np.float64),
        }

        prev_size = self._buffer.size
        self._buffer.add(data, timestamp=float(state.timestamp))

        if self._buffer.recording_stopped:
            self._max_frames_reached = True
            return False

        self._frame_count = self._buffer.size
        k = self._buffer.size - prev_size  # grid slots advanced (usually 1; 0 = dup bucket)

        # ── Camera streams → per-frame HDF5, forward-filled to stay length-aligned ──
        has_camera_now = camera_frame is not None or bool(camera_frames)
        if has_camera_now:
            self._cam_seen = True
            self._last_camera_frame = camera_frame
            self._last_camera_frames = camera_frames
            self._last_T_base_eef = T_base_eef

        if self._cam_seen and k > 0:
            self._ensure_hdf5()
            self._append_camera(
                self._last_camera_frame,
                self._last_T_base_eef,
                self._last_camera_frames,
                repeat=k,
            )
        return True

    def _ensure_hdf5(self) -> None:
        """Lazily create the HDF5 file (flat schema — no groups)."""
        if self._file is not None:
            return
        assert self._episode_path is not None
        self._file = h5py.File(str(self._episode_path), "w")
        self._write_meta_attrs(self._file.create_group("meta"))

    def _append_camera(
        self,
        camera_frame: dict[str, Any] | None,
        T_base_eef: np.ndarray | None,
        camera_frames: dict[str, dict[str, Any]] | None,
        repeat: int = 1,
    ) -> None:
        """Append camera frame(s) `repeat` times (forward-fill on gaps)."""
        assert self._file is not None
        for _ in range(max(1, repeat)):
            # Single-camera
            if camera_frame is not None:
                rgb = camera_frame.get("rgb")
                depth = camera_frame.get("depth")
                if "rgb" not in self._datasets:
                    if rgb is not None:
                        self._datasets["rgb"] = self._file.create_dataset(
                            "rgb",
                            data=rgb[np.newaxis, ...],
                            maxshape=(None,) + rgb.shape,
                            chunks=True,
                            dtype=rgb.dtype,
                        )
                    if depth is not None:
                        self._datasets["depth"] = self._file.create_dataset(
                            "depth",
                            data=depth[np.newaxis, ...],
                            maxshape=(None,) + depth.shape,
                            chunks=True,
                            dtype=depth.dtype,
                        )
                else:
                    if rgb is not None:
                        self._resize_append("rgb", rgb)
                    if depth is not None:
                        self._resize_append("depth", depth)

            # Multi-camera
            if camera_frames:
                for cam_name, cam_frame in camera_frames.items():
                    if cam_frame is None:
                        continue
                    safe = str(cam_name).replace("/", "_").replace("\\", "_")
                    rgb = cam_frame.get("rgb")
                    depth = cam_frame.get("depth")
                    rgb_key = f"{safe}_rgb"
                    depth_key = f"{safe}_depth"
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
                    else:
                        if rgb is not None and hasattr(rgb, "shape"):
                            self._resize_append(rgb_key, rgb)
                        if depth is not None and hasattr(depth, "shape"):
                            self._resize_append(depth_key, depth)

    def stop_episode(self, success: bool = True) -> str | None:
        if not self._recording:
            return None

        duration = time.perf_counter() - (self._start_time or 0.0)

        # ── Flush aligned non-camera streams to HDF5 ──
        if self._buffer is not None:
            buf_data = self._buffer.data
            buf_size = self._buffer.size
            self._ensure_hdf5()

            # Write buffered streams as static datasets.
            for h5_key, arr in buf_data.items():
                if h5_key not in self._datasets:
                    self._datasets[h5_key] = self._file.create_dataset(
                        h5_key, data=arr, dtype=arr.dtype
                    )

            # Write aligned timestamp
            self._datasets["timestamp"] = self._file.create_dataset(
                "timestamp", data=self._buffer.timestamps, dtype=np.float64
            )

            # Camera streams are forward-filled; warn on drift.
            for key in ("rgb", "depth"):
                ds = self._datasets.get(key)
                if ds is not None and ds.shape[0] != buf_size:
                    logger.warning(
                        "%s length %d != grid length %d (alignment drift)",
                        key, ds.shape[0], buf_size,
                    )

            self._buffer = None
            self._frame_count = buf_size

        # ── Write final metadata ──
        if self._file is not None:
            meta = self._file["meta"]
            meta.attrs["schema_version"] = SCHEMA_VERSION
            meta.attrs["duration"] = duration
            meta.attrs["num_frames"] = self._frame_count
            meta.attrs["success"] = success
            meta.attrs["fps"] = self._frame_count / duration if duration > 0 else 50.0
            meta.attrs["min_frames_met"] = self._frame_count >= 50
            meta.attrs["has_camera"] = "rgb" in self._file
            meta.attrs["has_timestamps"] = "timestamp" in self._file

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
        self._cam_seen = False
        self._last_camera_frame = None
        self._last_camera_frames = None
        self._last_T_base_eef = None
        return path

    def _resize_append(self, key: str, data: np.ndarray) -> None:
        ds = self._datasets[key]
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = data
