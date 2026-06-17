"""EpisodeRecorder — HDF5-based teleoperation data recorder.

HDF5 structure (per recording-spec.md):

    episode_000.h5
      /meta: task_label, operator, tags, duration, fps,
             num_frames, num_valid_frames, success,
             camera_serial, camera_type,
             camera_K,                             # [fx, fy, cx, cy]
             camera_T_base_camera | camera_T_eef_camera,  # 4x4 flat
             retargeting_config, pipeline_snapshot
      /obs/arm_qpos(7)  arm_qvel(7)  arm_tau(7)  eef_pos(3)  eef_quat(4)
      /obs/hand_qpos(12)  hand_current(12)  hand_tactile_sum(5,3)  hand_temperature(12)
      /action/arm_qpos(7)  hand_qpos(12)
      /vr/wrist_pos(3)  wrist_quat(4)  landmarks(21,3)
      /quality_flags(T,) uint16
      /camera/rgb(T,H,W,3)  depth(T,H,W)  timestamps(T)
      /camera/K(3,3)                        # 内参矩阵
      /camera/extrinsics(T,4,4)             # T_base_camera，逐帧外参
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.robot.interface import RobotAction, RobotState


class EpisodeRecorder:
    """Records teleoperation episodes to HDF5 files.

    Lifecycle: start_episode() → add_frame() × N → stop_episode()
    """

    def __init__(
        self,
        data_dir: str,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._file: h5py.File | None = None
        self._frame_count: int = 0
        self._recording: bool = False
        self._start_time: float | None = None
        self._episode_path: str | None = None

        # Expandable datasets — use chunked + resizable storage
        self._datasets: dict[str, Any] = {}

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start_episode(
        self,
        task_label: str = "",
        operator: str = "",
        tags: list[str] | None = None,
        calib: CameraCalib | None = None,
        camera_K: np.ndarray | None = None,
        camera_name: str | None = None,
    ) -> bool:
        """Create a new HDF5 episode file and write initial metadata."""
        if self._recording:
            return False

        # Find next episode index
        idx = 0
        while (self.data_dir / f"episode_{idx:03d}.h5").exists():
            idx += 1

        path = self.data_dir / f"episode_{idx:03d}.h5"
        self._file = h5py.File(str(path), "w")
        self._episode_path = str(path)
        self._frame_count = 0
        self._start_time = time.perf_counter()
        self._recording = True
        self._datasets = {}

        # ── /meta ──
        meta = self._file.create_group("meta")
        meta.attrs["task_label"] = task_label
        meta.attrs["operator"] = operator
        meta.attrs["tags"] = ",".join(tags) if tags else ""
        meta.attrs["fps"] = 50.0  # will be updated at stop

        # Camera calibration metadata
        if calib is not None and camera_name is not None:
            calib_meta = calib.to_meta_dict(camera_name)
            meta.attrs["camera_serial"] = calib_meta.get("camera_serial", "")
            meta.attrs["camera_type"] = calib_meta.get("camera_type", "")
            if "camera_T_base_camera" in calib_meta:
                meta.attrs["camera_T_base_camera"] = calib_meta["camera_T_base_camera"]
            if "camera_T_eef_camera" in calib_meta:
                meta.attrs["camera_T_eef_camera"] = calib_meta["camera_T_eef_camera"]

        if camera_K is not None:
            meta.attrs["camera_K"] = camera_K.flatten().tolist()  # [fx, fy, cx, cy]

        # ── Create groups ──
        self._file.create_group("obs")
        self._file.create_group("action")
        self._file.create_group("vr")

        return True

    def add_frame(
        self,
        state: RobotState,
        action: RobotAction,
        vr_frame: dict[str, Any],
        quality_flags: int,
        camera_frame: dict[str, Any] | None = None,
        T_base_eef: np.ndarray | None = None,
    ) -> bool:
        """Append one frame to the HDF5 file."""
        if not self._recording or self._file is None:
            return False

        self._append_or_create("obs/arm_qpos", state.arm_qpos)
        self._append_or_create("obs/arm_qvel", state.arm_qvel)
        self._append_or_create("obs/arm_tau", state.arm_tau)
        self._append_or_create("obs/eef_pos", state.eef_pos)
        self._append_or_create("obs/eef_quat", state.eef_quat_wxyz)
        self._append_or_create("obs/hand_qpos", state.hand_qpos)
        self._append_or_create("obs/hand_current", state.hand_current)
        self._append_or_create("obs/hand_tactile_sum", state.hand_tactile_sum)
        self._append_or_create("obs/hand_temperature", state.hand_temperature)

        self._append_or_create("action/arm_qpos", action.arm_qpos_cmd)
        self._append_or_create("action/hand_qpos", action.hand_qpos_cmd)

        self._append_or_create("vr/wrist_pos", vr_frame["wrist_pos"])
        self._append_or_create("vr/wrist_quat", vr_frame["wrist_quat_wxyz"])
        self._append_or_create("vr/landmarks", vr_frame["landmarks"])

        qf_arr = np.array([quality_flags], dtype=np.uint16)
        if "quality_flags" not in self._datasets:
            self._datasets["quality_flags"] = self._file.create_dataset(
                "quality_flags", data=qf_arr,
                maxshape=(None,), chunks=True, dtype=np.uint16,
            )
        else:
            self._resize_append("quality_flags", qf_arr)

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

        # Camera frames (RGB + depth)
        if camera_frame is not None:
            rgb = camera_frame.get("rgb")
            depth = camera_frame.get("depth")
            ts = camera_frame.get("timestamp", 0.0)

            if "camera/rgb" not in self._datasets:
                if rgb is not None:
                    maxshape = (None,) + rgb.shape
                    self._datasets["camera/rgb"] = self._file.create_dataset(
                        "camera/rgb", data=rgb[np.newaxis, ...],
                        maxshape=maxshape, chunks=True, dtype=rgb.dtype,
                    )
                if depth is not None:
                    maxshape = (None,) + depth.shape
                    self._datasets["camera/depth"] = self._file.create_dataset(
                        "camera/depth", data=depth[np.newaxis, ...],
                        maxshape=maxshape, chunks=True, dtype=depth.dtype,
                    )
                self._datasets["camera/timestamps"] = self._file.create_dataset(
                    "camera/timestamps", data=[ts],
                    maxshape=(None,), chunks=True, dtype=np.float64,
                )
            else:
                if rgb is not None:
                    self._resize_append("camera/rgb", rgb)
                if depth is not None:
                    self._resize_append("camera/depth", depth)
                self._resize_append("camera/timestamps", np.array([ts]))

        self._frame_count += 1
        return True

    def stop_episode(self, success: bool = True) -> str | None:
        """Close the HDF5 file and finalize metadata. Returns file path."""
        if not self._recording or self._file is None:
            return None

        duration = time.perf_counter() - (self._start_time or 0.0)

        meta = self._file["meta"]
        meta.attrs["duration"] = duration
        meta.attrs["num_frames"] = self._frame_count
        meta.attrs["success"] = success
        meta.attrs["fps"] = self._frame_count / duration if duration > 0 else 0.0

        # num_valid_frames
        if "quality_flags" in self._file:
            from dexmani_real.recording.quality_flags import ALL_GOOD_MASK

            qf = np.asarray(self._file["quality_flags"][:], dtype=np.uint16)
            valid = int(np.sum((qf & np.uint16(ALL_GOOD_MASK)) == np.uint16(ALL_GOOD_MASK)))
            meta.attrs["num_valid_frames"] = valid

        path = self._episode_path
        self._file.close()
        self._file = None
        self._datasets.clear()
        self._recording = False
        self._frame_count = 0
        self._start_time = None
        self._episode_path = None
        return path

    def _append_or_create(self, key: str, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if key not in self._datasets:
            maxshape = (None,) + arr.shape
            self._datasets[key] = self._file.create_dataset(
                key, data=arr[np.newaxis, ...],
                maxshape=maxshape, chunks=True, dtype=arr.dtype,
            )
        else:
            self._resize_append(key, arr)

    def _resize_append(self, key: str, data: np.ndarray) -> None:
        ds = self._datasets[key]
        n = ds.shape[0]
        ds.resize(n + 1, axis=0)
        ds[n] = data
