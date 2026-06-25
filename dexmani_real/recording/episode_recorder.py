"""EpisodeRecorder — HDF5-based teleoperation data recorder (simplified).

Uses direct per-frame h5py.Dataset.resize() write mode.
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
        self._file = h5py.File(str(path), "w")
        self._episode_path = str(path)
        self._frame_count = 0
        self._max_frames_reached = False
        self._start_time = time.perf_counter()
        self._recording = True
        self._datasets = {}

        meta = self._file.create_group("meta")
        meta.attrs["task_label"] = task_label
        meta.attrs["operator"] = operator
        meta.attrs["tags"] = ",".join(tags) if tags else ""
        meta.attrs["fps"] = 50.0

        if calib is not None and camera_name is not None:
            calib_meta = calib.to_meta_dict(camera_name)
            meta.attrs["camera_serial"] = calib_meta.get("camera_serial", "")
            meta.attrs["camera_type"] = calib_meta.get("camera_type", "")
            if "camera_T_base_camera" in calib_meta:
                meta.attrs["camera_T_base_camera"] = calib_meta["camera_T_base_camera"]
            if "camera_T_eef_camera" in calib_meta:
                meta.attrs["camera_T_eef_camera"] = calib_meta["camera_T_eef_camera"]
        if camera_K is not None:
            meta.attrs["camera_K"] = camera_K.flatten().tolist()

        self._file.create_group("obs")
        self._file.create_group("action")
        self._file.create_group("vr")

        return True

    def add_frame(
        self,
        state,
        action,
        vr_frame: dict[str, Any],
        camera_frame: dict[str, Any] | None = None,
        T_base_eef: np.ndarray | None = None,
        camera_frames: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        if not self._recording or self._file is None:
            return False

        if self._frame_count >= self.max_frames:
            logger.warning("Episode reached max_frames=%d, auto-stopping.", self.max_frames)
            self._max_frames_reached = True
            return False

        # Observation
        self._append_or_create("obs/arm_qpos", state.arm_qpos)
        self._append_or_create("obs/arm_qvel", state.arm_qvel)
        self._append_or_create("obs/arm_tau", state.arm_tau)
        self._append_or_create("obs/eef_pos", state.eef_pos)
        self._append_or_create("obs/eef_quat", state.eef_quat_wxyz)
        self._append_or_create("obs/hand_qpos", state.hand_qpos)
        self._append_or_create("obs/hand_tactile_sum", state.hand_tactile_sum)

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
                "timestamps", data=ts_arr, maxshape=(None,), chunks=True, dtype=np.float64,
            )
        else:
            self._resize_append("timestamps", ts_arr)

        vr_ts = np.array([vr_frame.get("local_recv_ns", 0) * 1e-9], dtype=np.float64)
        if "vr_timestamps" not in self._datasets:
            self._datasets["vr_timestamps"] = self._file.create_dataset(
                "vr_timestamps", data=vr_ts, maxshape=(None,), chunks=True, dtype=np.float64,
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
                        "camera/rgb", data=rgb[np.newaxis, ...],
                        maxshape=(None,) + rgb.shape, chunks=True, dtype=rgb.dtype,
                    )
                if depth is not None:
                    self._datasets["camera/depth"] = self._file.create_dataset(
                        "camera/depth", data=depth[np.newaxis, ...],
                        maxshape=(None,) + depth.shape, chunks=True, dtype=depth.dtype,
                    )
                self._datasets["camera/timestamps"] = self._file.create_dataset(
                    "camera/timestamps", data=[ts], maxshape=(None,), chunks=True, dtype=np.float64,
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
                            rgb_key, data=rgb[np.newaxis, ...],
                            maxshape=(None,) + rgb.shape, chunks=True,
                            dtype=rgb.dtype if rgb.dtype == np.uint8 else np.uint8,
                        )
                    if depth is not None and hasattr(depth, "shape"):
                        self._datasets[depth_key] = self._file.create_dataset(
                            depth_key, data=depth[np.newaxis, ...],
                            maxshape=(None,) + depth.shape, chunks=True,
                            dtype=depth.dtype if depth.dtype == np.uint16 else np.uint16,
                        )
                    self._datasets[ts_key] = self._file.create_dataset(
                        ts_key, data=np.array([ts]), maxshape=(None,), chunks=True, dtype=np.float64,
                    )
                else:
                    if rgb is not None and hasattr(rgb, "shape"):
                        self._resize_append(rgb_key, rgb)
                    if depth is not None and hasattr(depth, "shape"):
                        self._resize_append(depth_key, depth)
                    self._resize_append(ts_key, np.array([ts]))

        self._frame_count += 1
        return True

    def stop_episode(self, success: bool = True) -> str | None:
        if not self._recording or self._file is None:
            return None

        duration = time.perf_counter() - (self._start_time or 0.0)
        meta = self._file["meta"]
        meta.attrs["duration"] = duration
        meta.attrs["num_frames"] = self._frame_count
        meta.attrs["success"] = success
        meta.attrs["fps"] = self._frame_count / duration if duration > 0 else 0.0
        meta.attrs["min_frames_met"] = self._frame_count >= 50
        meta.attrs["has_camera"] = "camera/timestamps" in self._file
        meta.attrs["has_timestamps"] = "timestamps" in self._file

        path = self._episode_path
        self._file.close()
        self._file = None
        self._datasets.clear()
        self._recording = False
        self._max_frames_reached = False
        self._frame_count = 0
        self._start_time = None
        self._episode_path = None
        return path

    def _append_or_create(self, key: str, data: np.ndarray) -> None:
        arr = np.asarray(data)
        if key not in self._datasets:
            self._datasets[key] = self._file.create_dataset(
                key, data=arr[np.newaxis, ...], maxshape=(None,) + arr.shape,
                chunks=True, dtype=arr.dtype,
            )
        else:
            self._resize_append(key, arr)

    def _resize_append(self, key: str, data: np.ndarray) -> None:
        ds = self._datasets[key]
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = data
