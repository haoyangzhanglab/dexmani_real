"""InMemoryFrameBuffer — pre-allocated numpy ring buffer for batch HDF5 writing.

Replaces the per-frame h5py.Dataset.resize() pattern with O(1) numpy
array assignment + batch flush every BATCH_SIZE frames.

Camera frames are written directly to HDF5 in batches without full
in-memory caching (avoiding 9+ GB memory for 10000 RGB frames).

Performance comparison:
    Current (per-frame resize): ~15× ds.resize() per frame
    New (batch flush):         arr[i] = val per frame, 15× resize per 100 frames

Ref: BunnyVisionPro batched HDF5 write pattern.
     DexUMI Zarr storage pipeline.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import h5py
import numpy as np

from dexmani_real.log import get_logger

logger = get_logger(__name__)

# ── Dataset specifications ──
# Format: (hdf5_path, shape_per_frame, dtype)
_DATASET_SPECS: list[tuple[str, tuple[int, ...], np.dtype]] = [
    # Observation
    ("obs/arm_qpos", (7,), np.float64),
    ("obs/arm_qvel", (7,), np.float64),
    ("obs/arm_tau", (7,), np.float64),
    ("obs/eef_pos", (3,), np.float64),
    ("obs/eef_quat", (4,), np.float64),
    ("obs/hand_qpos", (12,), np.float64),
    ("obs/hand_current", (12,), np.float64),
    ("obs/hand_tactile_sum", (5, 3), np.float64),
    ("obs/hand_temperature", (12,), np.float64),
    # Action
    ("action/arm_qpos", (7,), np.float64),
    ("action/hand_qpos", (12,), np.float64),
    # VR
    ("vr/wrist_pos", (3,), np.float64),
    ("vr/wrist_quat", (4,), np.float64),
    ("vr/landmarks", (21, 3), np.float64),
    # Quality
    ("quality_flags", (1,), np.uint16),
    # Camera extrinsics
    ("camera/extrinsics", (4, 4), np.float64),
    # Timestamps
    ("timestamps", (1,), np.float64),
    ("vr_timestamps", (1,), np.float64),
]


class InMemoryFrameBuffer:
    """Pre-allocated numpy buffer with batch HDF5 flush.

    Every BATCH_SIZE frames (≈2s at 50Hz), the buffer is flushed to HDF5
    via a single set of dataset.resize() + dataset.write() calls.
    Camera data is NOT cached in memory — it is written directly to HDF5
    in batches.

    Lifecycle:
        buf = InMemoryFrameBuffer(max_frames=10000, h5_file=h5_file)
        for ...:
            buf.add_frame(state, action, vr_frame, quality_flags, ...)
        buf.flush_all()
        buf.close()
    """

    BATCH_SIZE: ClassVar[int] = 100

    def __init__(
        self,
        max_frames: int = 10000,
        h5_file: h5py.File | None = None,
    ) -> None:
        self.max_frames = max_frames
        self._h5_file = h5_file

        # Per-spec numpy buffers
        self._buffers: dict[str, np.ndarray] = {}
        self._buffer_mask: np.ndarray | None = None  # tracks which rows are populated

        # HDF5 datasets (created lazily on first flush)
        self._datasets: dict[str, h5py.Dataset] = {}

        # Camera batch accumulator (written directly, not cached)
        self._camera_rgb_batch: list[np.ndarray] = []
        self._camera_depth_batch: list[np.ndarray] = []
        self._camera_ts_batch: list[float] = []
        self._camera_written: bool = False  # track if camera datasets exist

        # Multi-camera batch accumulators: name → { rgb: [...], depth: [...], ts: [...] }
        self._mc_rgb_batches: dict[str, list[np.ndarray | None]] = {}
        self._mc_depth_batches: dict[str, list[np.ndarray | None]] = {}
        self._mc_ts_batches: dict[str, list[float]] = {}
        self._mc_written: set[str] = set()  # track which camera names have datasets

        # Frame counter
        self._frame_count: int = 0
        self._total_written: int = 0  # frames flushed to HDF5

        # Pre-allocate all numpy buffers
        self._allocate()

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _allocate(self) -> None:
        """Pre-allocate numpy arrays for all non-camera datasets."""
        for path, shape, dtype in _DATASET_SPECS:
            full_shape = (self.max_frames,) + shape
            self._buffers[path] = np.zeros(full_shape, dtype=dtype)
        self._buffer_mask = np.zeros(self.max_frames, dtype=bool)

        total_bytes = sum(b.nbytes for b in self._buffers.values())
        logger.debug(
            "InMemoryFrameBuffer allocated: max_frames=%d total=%.1f MB batch_size=%d",
            self.max_frames, total_bytes / (1024 * 1024), self.BATCH_SIZE,
        )

    # ------------------------------------------------------------------
    # Frame append (O(1))
    # ------------------------------------------------------------------

    def add_frame(
        self,
        state: Any,          # RobotState
        action: Any,         # RobotAction
        vr_frame: dict[str, Any],
        quality_flags: int,
        camera_frame: dict[str, Any] | None = None,
        T_base_eef: np.ndarray | None = None,
        timestamp_ns: int | None = None,
        vr_timestamp_ns: int | None = None,
        camera_frames: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        """Append one frame to the in-memory buffer. O(1) numpy assignment.

        Args:
            state: Current robot state.
            action: Computed robot action.
            vr_frame: VR tracking frame dict.
            quality_flags: Per-frame quality flags bitmask.
            camera_frame: Single camera frame (backward compat).
            T_base_eef: 4x4 base→EEF transform.
            timestamp_ns: Control loop timestamp in nanoseconds.
            vr_timestamp_ns: VR frame receive timestamp in nanoseconds.
            camera_frames: Multi-camera frames dict (name → frame dict).

        Returns True if the frame was buffered, False if the buffer is full.
        """
        if self._frame_count >= self.max_frames:
            logger.warning("InMemoryFrameBuffer full: %d frames", self._frame_count)
            return False

        i = self._frame_count

        # Observation
        self._assign("obs/arm_qpos", i, state.arm_qpos)
        self._assign("obs/arm_qvel", i, state.arm_qvel)
        self._assign("obs/arm_tau", i, state.arm_tau)
        self._assign("obs/eef_pos", i, state.eef_pos)
        self._assign("obs/eef_quat", i, state.eef_quat_wxyz)
        self._assign("obs/hand_qpos", i, state.hand_qpos)
        self._assign("obs/hand_current", i, state.hand_current)
        self._assign("obs/hand_tactile_sum", i, state.hand_tactile_sum)
        self._assign("obs/hand_temperature", i, state.hand_temperature)

        # Action
        self._assign("action/arm_qpos", i, action.arm_qpos_cmd)
        self._assign("action/hand_qpos", i, action.hand_qpos_cmd)

        # VR
        self._assign("vr/wrist_pos", i, vr_frame["wrist_pos"])
        self._assign("vr/wrist_quat", i, vr_frame["wrist_quat_wxyz"])
        self._assign("vr/landmarks", i, vr_frame["landmarks"])

        # Quality flags
        self._buffers["quality_flags"][i] = np.uint16(quality_flags)

        # Camera extrinsics
        if T_base_eef is not None:
            self._assign("camera/extrinsics", i, T_base_eef)

        # Timestamps
        if timestamp_ns is not None:
            self._buffers["timestamps"][i] = np.float64(timestamp_ns * 1e-9)
        else:
            self._buffers["timestamps"][i] = np.float64(time.perf_counter())
        if vr_timestamp_ns is not None:
            self._buffers["vr_timestamps"][i] = np.float64(vr_timestamp_ns * 1e-9)

        self._buffer_mask[i] = True
        self._frame_count += 1

        # Camera frames: accumulate for direct batch write
        if camera_frame is not None:
            rgb = camera_frame.get("rgb")
            depth = camera_frame.get("depth")
            ts = camera_frame.get("timestamp", 0.0)
            if rgb is not None:
                self._camera_rgb_batch.append(np.asarray(rgb, dtype=np.uint8))
            if depth is not None:
                self._camera_depth_batch.append(np.asarray(depth, dtype=np.uint16))
            self._camera_ts_batch.append(ts)
        else:
            # Placeholder for missing camera frame (will be skipped in flush)
            self._camera_rgb_batch.append(None)
            self._camera_depth_batch.append(None)
            self._camera_ts_batch.append(-1.0)

        # Multi-camera frames: accumulate per-camera
        if camera_frames:
            for cam_name, cam_frame in camera_frames.items():
                safe_name = str(cam_name).replace("/", "_").replace("\\", "_")
                if safe_name not in self._mc_rgb_batches:
                    self._mc_rgb_batches[safe_name] = [None] * self._frame_count
                    self._mc_depth_batches[safe_name] = [None] * self._frame_count
                    self._mc_ts_batches[safe_name] = [-1.0] * self._frame_count
                # Extend to match current frame count if needed
                while len(self._mc_rgb_batches[safe_name]) <= i:
                    self._mc_rgb_batches[safe_name].append(None)
                    self._mc_depth_batches[safe_name].append(None)
                    self._mc_ts_batches[safe_name].append(-1.0)

                if cam_frame is not None:
                    rgb = cam_frame.get("rgb")
                    depth = cam_frame.get("depth")
                    ts = cam_frame.get("timestamp", 0.0)
                    self._mc_rgb_batches[safe_name][i] = (
                        np.asarray(rgb, dtype=np.uint8) if rgb is not None else None
                    )
                    self._mc_depth_batches[safe_name][i] = (
                        np.asarray(depth, dtype=np.uint16) if depth is not None else None
                    )
                    self._mc_ts_batches[safe_name][i] = ts

        # Auto-flush when batch is full
        if self._frame_count > 0 and self._frame_count % self.BATCH_SIZE == 0:
            self.flush_batch()

        return True

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush_batch(self) -> int:
        """Flush newly buffered frames (since last flush) to HDF5.

        Returns the number of frames flushed.
        """
        if self._h5_file is None:
            return 0

        new_since_last = self._frame_count - self._total_written
        if new_since_last <= 0:
            return 0

        # Determine the slice to write
        start = self._total_written
        end = self._frame_count
        n = end - start

        self._flush_datasets(start, end, n)
        self._flush_camera_batch(start, end)
        self._flush_multi_camera_batches(start, end)

        self._total_written = end
        return n

    def flush_all(self) -> int:
        """Flush all remaining frames. Call at episode end.

        Returns total frames flushed in this call.
        """
        return self.flush_batch()

    def _flush_datasets(self, start: int, end: int, n: int) -> None:
        """Resize and write all non-camera datasets for frames [start:end)."""
        for path, shape, dtype in _DATASET_SPECS:
            buf = self._buffers[path]
            data = buf[start:end]

            if path not in self._datasets:
                # First write: create resizable dataset
                full_shape = (None,) + shape
                self._datasets[path] = self._h5_file.create_dataset(
                    path, data=data, maxshape=full_shape, chunks=True, dtype=dtype,
                )
            else:
                ds = self._datasets[path]
                current_len = ds.shape[0]
                ds.resize(current_len + n, axis=0)
                ds[current_len:] = data

    def _flush_camera_batch(self, start: int, end: int) -> None:
        """Write camera frames directly to HDF5 in a batch."""
        if not self._camera_rgb_batch:
            return

        # Extract valid camera frames in [start:end)
        rgb_slice = self._camera_rgb_batch[start:end]
        depth_slice = self._camera_depth_batch[start:end]
        ts_slice = self._camera_ts_batch[start:end]

        # Filter out None entries (missing frames)
        valid_rgb = [r for r in rgb_slice if r is not None]
        valid_depth = [d for d in depth_slice if d is not None]
        valid_ts = [t for t, r in zip(ts_slice, rgb_slice) if r is not None]

        if not valid_rgb:
            return

        # Stack into a single array per channel
        rgb_stack = np.stack(valid_rgb, axis=0)
        depth_stack = np.stack(valid_depth, axis=0)
        ts_arr = np.array(valid_ts, dtype=np.float64)

        if not self._camera_written:
            # First batch: create datasets
            self._datasets["camera/rgb"] = self._h5_file.create_dataset(
                "camera/rgb", data=rgb_stack,
                maxshape=(None,) + rgb_stack.shape[1:],
                chunks=True, dtype=np.uint8,
            )
            self._datasets["camera/depth"] = self._h5_file.create_dataset(
                "camera/depth", data=depth_stack,
                maxshape=(None,) + depth_stack.shape[1:],
                chunks=True, dtype=np.uint16,
            )
            self._datasets["camera/timestamps"] = self._h5_file.create_dataset(
                "camera/timestamps", data=ts_arr,
                maxshape=(None,), chunks=True, dtype=np.float64,
            )
            self._camera_written = True
        else:
            # Append to existing datasets
            for key, stack in [
                ("camera/rgb", rgb_stack),
                ("camera/depth", depth_stack),
            ]:
                ds = self._datasets[key]
                n_old = ds.shape[0]
                ds.resize(n_old + stack.shape[0], axis=0)
                ds[n_old:] = stack
            ts_ds = self._datasets["camera/timestamps"]
            n_old = ts_ds.shape[0]
            ts_ds.resize(n_old + len(ts_arr), axis=0)
            ts_ds[n_old:] = ts_arr

    def _flush_multi_camera_batches(self, start: int, end: int) -> None:
        """Write multi-camera frames to HDF5 in per-camera batch."""
        if not self._mc_rgb_batches:
            return

        for cam_name in list(self._mc_rgb_batches.keys()):
            rgb_batch = self._mc_rgb_batches.get(cam_name, [])
            depth_batch = self._mc_depth_batches.get(cam_name, [])
            ts_batch = self._mc_ts_batches.get(cam_name, [])

            if len(rgb_batch) <= end:
                continue

            rgb_slice = rgb_batch[start:end]
            depth_slice = depth_batch[start:end]
            ts_slice = ts_batch[start:end]

            # Filter out None entries
            valid_rgb = [r for r in rgb_slice if r is not None]
            valid_depth = [d for d in depth_slice if d is not None]
            valid_ts = [t for t, r in zip(ts_slice, rgb_slice) if r is not None]

            if not valid_rgb:
                continue

            rgb_stack = np.stack(valid_rgb, axis=0)
            depth_stack = np.stack(valid_depth, axis=0)
            ts_arr = np.array(valid_ts, dtype=np.float64)

            rgb_key = f"camera/{cam_name}/rgb"
            depth_key = f"camera/{cam_name}/depth"
            ts_key = f"camera/{cam_name}/timestamps"

            if cam_name not in self._mc_written:
                self._datasets[rgb_key] = self._h5_file.create_dataset(
                    rgb_key, data=rgb_stack,
                    maxshape=(None,) + rgb_stack.shape[1:],
                    chunks=True, dtype=np.uint8,
                )
                self._datasets[depth_key] = self._h5_file.create_dataset(
                    depth_key, data=depth_stack,
                    maxshape=(None,) + depth_stack.shape[1:],
                    chunks=True, dtype=np.uint16,
                )
                self._datasets[ts_key] = self._h5_file.create_dataset(
                    ts_key, data=ts_arr,
                    maxshape=(None,), chunks=True, dtype=np.float64,
                )
                self._mc_written.add(cam_name)
            else:
                for key, stack in [
                    (rgb_key, rgb_stack),
                    (depth_key, depth_stack),
                ]:
                    ds = self._datasets[key]
                    n_old = ds.shape[0]
                    ds.resize(n_old + stack.shape[0], axis=0)
                    ds[n_old:] = stack
                ts_ds = self._datasets[ts_key]
                n_old = ts_ds.shape[0]
                ts_ds.resize(n_old + len(ts_arr), axis=0)
                ts_ds[n_old:] = ts_arr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assign(self, path: str, idx: int, data: np.ndarray) -> None:
        """Assign a numpy array to the buffer at the given index."""
        arr = np.asarray(data, dtype=np.float64)
        # Squeeze 1-dim arrays into scalar shapes if needed
        self._buffers[path][idx] = arr

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def written_count(self) -> int:
        return self._total_written

    @property
    def memory_usage_mb(self) -> float:
        """Total memory used by numpy buffers in MB."""
        return sum(b.nbytes for b in self._buffers.values()) / (1024 * 1024)

    def close(self) -> None:
        """Release in-memory buffers (does not close HDF5 file)."""
        self._buffers.clear()
        self._datasets.clear()
        self._camera_rgb_batch.clear()
        self._camera_depth_batch.clear()
        self._camera_ts_batch.clear()
        self._mc_rgb_batches.clear()
        self._mc_depth_batches.clear()
        self._mc_ts_batches.clear()
        self._mc_written.clear()
        self._buffer_mask = None
        self._frame_count = 0
        self._total_written = 0
