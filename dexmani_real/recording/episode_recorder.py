"""EpisodeRecorder — HDF5-based teleoperation data recorder.

Single write path: state/action/vr streams are aligned to a fixed dt=1/control_hz
time grid at record time (TimestampAlignedBuffer) and flushed to HDF5 in bulk at
stop_episode(); camera frames are streamed per-frame but kept length-aligned to
the same grid, so every dataset is index-aligned by construction.
"""

from __future__ import annotations

__all__ = ["EpisodeRecorder"]

import atexit
import queue
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.recording.collection_config import DEFAULT_MAX_RECORD_FRAMES
from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d
from dexmani_real.recording.timestamp_buffer import TimestampAlignedBuffer
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 8  # v8: +truncated/stop_reason/cam_frames_dropped/cam_items_written meta; rgb/depth codec lzf; v7: rate-parameterized grid (control_hz meta attr; dt/fps derived, no 50Hz hardcode); v6: +/flag_camera_fresh (camera stall marker); camera streams grid-index-aligned; v5: /pointcloud (T,N,6) + has_pointcloud + pc_* meta; /depth gated by L515 validity

CAMERA_FRESH_TIMEOUT_S = 0.2  # flag_camera_fresh: max age of the last *new* camera frame (~6 frames @30fps)

# ── atexit safety net ──
# The episode-stop / cam-writer threads are daemons: if an entry point exits
# without join_stop() (estop path, second Ctrl-C inside a finally prompt),
# the interpreter kills them mid-flush and truncates the HDF5.  One hook
# joins every live recorder at interpreter exit.  No-op on SIGTERM/SIGKILL.
_LIVE_RECORDERS: weakref.WeakSet = weakref.WeakSet()


def _flush_all_recorders() -> None:
    for rec in list(_LIVE_RECORDERS):  # snapshot — WeakSet may mutate under GC
        try:
            if rec._recording:
                rec.stop_episode(success=False, reason="atexit")
            rec.join_stop(timeout=60.0)
        except Exception:
            pass  # never raise during interpreter shutdown


atexit.register(_flush_all_recorders)


class EpisodeRecorder:
    """Records teleoperation episodes to HDF5 files.

    Lifecycle: start_episode() → add_frame() × N → stop_episode()
    """

    def __init__(
        self,
        data_dir: str,
        max_frames: int = DEFAULT_MAX_RECORD_FRAMES,
        control_hz: float = 50.0,
        min_frames: int = 50,
    ) -> None:
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.control_hz = float(control_hz)
        self.min_frames = int(min_frames)

        self._file: h5py.File | None = None
        self._frame_count: int = 0
        self._recording: bool = False
        self._max_frames_reached: bool = False
        self._start_time: float | None = None
        self._episode_path: str | None = None
        self._datasets: dict[str, Any] = {}

        # Record-time aligned buffer for non-camera streams + periodic flush.
        self._buffer: TimestampAlignedBuffer | None = None
        self._flush_interval: int = max(1, int(round(10.0 * self.control_hz)))  # frames (~10s)
        self._flushed_frames: int = 0

        # Skip the first N add_frame() calls per episode (begin-transition noise).
        # The grid is re-anchored to the first accepted frame's timestamp so the
        # dropped frames leave no forward-filled gap.
        self._skip_initial_frames: int = 0
        self._skipped_so_far: int = 0
        self._grid_anchored: bool = False

        # Camera length-alignment: cache last frame to forward-fill on None reads
        # / dropped grid slots so camera datasets stay index-aligned with the grid.
        self._cam_seen: bool = False
        self._last_camera_frame: dict[str, Any] | None = None
        self._last_camera_frames: dict[str, dict[str, Any]] | None = None
        self._last_T_base_eef: np.ndarray | None = None

        # Camera freshness: last seen frame_number (tuple for multi-cam) and the
        # state.timestamp at which it last changed — drives /flag_camera_fresh.
        # A stalled camera keeps re-serving the same shm frame, so an unchanged
        # frame_number (not a missing frame) is the stall signal.
        self._cam_last_frame_number: object | None = None
        self._cam_last_change_ts: float = 0.0

        # Background camera writer: queue → thread → HDF5.
        # Keeps the 50 Hz hot path at queue-push cost (~µs) instead of
        # HDF5 resize + lzf compression (~2–3 ms).
        self._cam_queue: queue.Queue = queue.Queue(maxsize=200)
        self._cam_writer: threading.Thread | None = None
        self._cam_writer_stop: threading.Event = threading.Event()
        self._cam_written: int = 0
        self._cam_dropped: int = 0  # enqueue-side drops (writer backlog) — see add_frame
        self._hdf5_lock: threading.Lock = threading.Lock()

        # Deferred flush: when True the background writer thread will call
        # _flush_buffered() on its next iteration instead of blocking the
        # 50 Hz hot path with HDF5 gzip + resize + fsync.
        self._flush_pending: bool = False

        # Pending async stop_episode thread (None = no pending stop).
        # Guarded by start_episode() to prevent overlapping episodes.
        self._stop_thread: threading.Thread | None = None

        _LIVE_RECORDERS.add(self)  # atexit flush net

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
        depth_scale: float | None = None,
        record_config: dict | None = None,
        skip_initial_frames: int = 0,
    ) -> bool:
        # Wait for any pending async stop_episode to finish — it owns
        # _file, _buffer, _datasets, _pending_meta and will reset them on
        # completion.  Refuse to start while it is still alive: proceeding
        # would let the old stop thread clobber the new episode's state.
        if not self.join_stop(timeout=10.0):
            logger.error("Previous episode still flushing — refusing to start a new one")
            return False

        if self._recording:
            return False

        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.data_dir / f"episode_{stamp}.h5"
        dedup = 1
        while path.exists():  # same-second collision → suffix, never overwrite
            path = self.data_dir / f"episode_{stamp}_{dedup}.h5"
            dedup += 1

        self._episode_path = str(path)
        self._frame_count = 0
        self._max_frames_reached = False
        self._start_time = time.perf_counter()
        self._recording = True
        self._datasets = {}
        self._flushed_frames = 0
        self._flush_pending = False
        self._cam_seen = False
        self._last_camera_frame = None
        self._last_camera_frames = None
        self._last_T_base_eef = None
        self._cam_last_frame_number = None
        self._cam_last_change_ts = 0.0

        # Skip-initial-frames gate — clamp below max_frames so we never drop all.
        self._skip_initial_frames = max(0, min(int(skip_initial_frames), self.max_frames - 1))
        self._skipped_so_far = 0
        self._grid_anchored = False

        # Store metadata for deferred write (HDF5 is created lazily).
        self._pending_meta: dict[str, Any] = {
            "task_label": task_label,
            "operator": operator,
            "tags": tags,
            "calib": calib,
            "camera_K": camera_K,
            "camera_name": camera_name,
            "depth_scale": depth_scale,
            "record_config": record_config,
            "skip_initial_frames": self._skip_initial_frames,
        }

        # Defer HDF5 creation to the first camera frame / stop_episode();
        # start the record-time aligned buffer for non-camera streams.
        dt = 1.0 / self.control_hz
        buffer_steps = self.max_frames + 100  # margin for grid-boundary alignment
        self._buffer = TimestampAlignedBuffer(
            start_time=self._start_time,
            dt=dt,
            max_record_steps=buffer_steps,
            # Ticks fire on slot boundaries (absolute-deadline RateManager), so
            # samples land at k*dt ± scheduling jitter relative to the first-frame
            # anchor.  eps=0.5 → round-to-nearest slot: absorbs ±dt/2 jitter
            # instead of dup-dropping samples that land marginally early.
            eps=0.5,
        )
        self._file = None
        self._start_cam_writer()
        return True

    def _write_meta_attrs(self, meta: h5py.Group) -> None:
        """Write deferred metadata attributes to the HDF5 meta group."""
        p = self._pending_meta
        meta.attrs["task_label"] = p.get("task_label", "")
        meta.attrs["operator"] = p.get("operator", "")
        tags = p.get("tags")
        meta.attrs["tags"] = ",".join(tags) if tags else ""
        meta.attrs["control_hz"] = self.control_hz  # nominal grid rate; dt = 1/control_hz
        meta.attrs["fps"] = self.control_hz

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

        # Raw uint16 depth units in meters (L515: 0.00025) — without this,
        # offline consumers cannot convert /depth correctly.
        depth_scale = p.get("depth_scale")
        if depth_scale is not None:
            meta.attrs["depth_scale"] = float(depth_scale)

        # Collection-config snapshot (control mode, EMA alphas, delta clips) —
        # essential for downstream reproducibility.  Values are pre-sanitized to
        # h5py-compatible scalars/strings by the controller.
        meta.attrs["skip_initial_frames"] = int(p.get("skip_initial_frames", 0))
        record_config = p.get("record_config") or {}
        for key, val in record_config.items():
            meta.attrs[key] = val

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

        # ── Skip initial frames (begin-transition pose noise) ──
        # Return False (not recorded) so CollectionLoop's frame counter stays
        # consistent with the HDF5 num_frames — skipped frames touch no buffer.
        if self._skipped_so_far < self._skip_initial_frames:
            self._skipped_so_far += 1
            return False

        # Re-anchor the aligned grid to the first accepted frame so the dropped
        # frames leave no forward-filled gap (buffer forward-fills from start_time).
        if not self._grid_anchored:
            ts0 = float(state.timestamp)
            if np.isfinite(ts0):
                self._buffer.start_time = ts0
                self._buffer._next_global_idx = 0
            self._grid_anchored = True

        # ── Non-camera streams → record-time aligned buffer ──
        sig = signals or {}

        # ── Camera freshness (recorder-side; no per-script signal plumbing) ──
        # frame_number is monotonic per camera (shm/layouts.py header); a stall
        # keeps re-serving the same shm frame, so an unchanged number means no
        # new data. Multi-cam: any camera advancing counts as fresh.
        ts = float(state.timestamp)
        if camera_frame is not None:
            fresh_token = camera_frame.get("frame_number")
        elif camera_frames:
            fresh_token = tuple(
                f.get("frame_number") if f is not None else None for f in camera_frames.values()
            )
        else:
            fresh_token = None
        if fresh_token is not None and fresh_token != self._cam_last_frame_number:
            self._cam_last_frame_number = fresh_token
            self._cam_last_change_ts = ts
        flag_camera_fresh = (
            self._cam_last_frame_number is not None
            and (ts - self._cam_last_change_ts) <= CAMERA_FRESH_TIMEOUT_S
        )

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
            "flag_camera_fresh": flag_camera_fresh,
            # ── VR ──
            "vr_wrist_pos": np.asarray(vr_frame["wrist_pos"], dtype=np.float64),
            "vr_wrist_rot6d": quat_wxyz_to_rot6d(np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)),
            "vr_landmarks": np.asarray(vr_frame["landmarks"], dtype=np.float64),
        }

        prev_size = self._buffer.size
        self._buffer.add(data, timestamp=float(state.timestamp))

        if self._buffer.recording_stopped:
            self._max_frames_reached = True
            return False

        self._frame_count = self._buffer.size
        k = self._buffer.size - prev_size  # grid slots advanced (usually 1; 0 = dup bucket)

        # ── Periodic flush: signal background writer thread instead of
        # blocking the 50 Hz hot path with HDF5 gzip + resize + fsync.
        if self._buffer.size - self._flushed_frames >= self._flush_interval:
            self._flush_pending = True

        # ── Camera streams → background writer thread ──
        # Queue push (~µs) keeps the hot path fast; HDF5 compression +
        # resize run in a daemon thread.  Forward-fill at stop_episode
        # handles minor k>1 jitter.
        has_camera_now = camera_frame is not None or bool(camera_frames)
        if has_camera_now:
            self._cam_seen = True
            self._last_camera_frame = camera_frame
            self._last_camera_frames = camera_frames
            self._last_T_base_eef = T_base_eef

        if self._cam_seen and k > 0:
            self._ensure_hdf5()
            # camera.poll_latest_frame() returns fresh dicts each call —
            # no deep copy needed; the queue reference keeps arrays alive
            # until the writer thread consumes them.
            if self._cam_queue.qsize() < self._cam_queue.maxsize - 1:
                # target_len = grid length after this add: the writer fills every
                # camera dataset up to it, so dropped/backlogged slots self-heal
                # and dataset index == grid index at all times.
                item = (
                    self._last_camera_frame,
                    self._last_camera_frames,
                    self._buffer.size,
                )
                try:
                    self._cam_queue.put_nowait(item)
                except queue.Full:  # defensive — headroom check above keeps ≥2 slots free
                    self._count_cam_drop()
            else:
                # Writer backlog: drop rather than block the control loop.  The
                # next accepted item backfills the gap with *its* frame, so the
                # dataset stays index-aligned but this content is lost — count it.
                self._count_cam_drop()
        return True

    def _count_cam_drop(self) -> None:
        """Count a camera frame dropped at enqueue; warn on the first and every 100th."""
        self._cam_dropped += 1
        if self._cam_dropped == 1 or self._cam_dropped % 100 == 0:
            logger.warning(
                "Camera writer backlog — dropped %d camera frame(s) so far",
                self._cam_dropped,
            )

    def _ensure_hdf5(self) -> None:
        """Lazily create the HDF5 file (flat schema — no groups)."""
        if self._file is not None:
            return
        assert self._episode_path is not None
        self._file = h5py.File(str(self._episode_path), "w")
        self._write_meta_attrs(self._file.create_group("meta"))

    # ── Background camera writer ──────────────────────────────────────────

    def _start_cam_writer(self) -> None:
        self._cam_writer_stop.clear()
        self._cam_written = 0
        self._cam_dropped = 0
        self._cam_writer = threading.Thread(
            target=self._cam_writer_loop, daemon=True, name="episode-cam-writer"
        )
        self._cam_writer.start()

    def _cam_writer_loop(self) -> None:
        """Background thread: pop camera frames from queue, write to HDF5.

        Also handles deferred buffer flushes (signalled by the hot path via
        ``_flush_pending``) so HDF5 gzip + resize + fsync never blocks the
        50 Hz control loop.
        """
        _hdf5_flush_every_n = 100  # HDF5 metadata flush (cheap)
        while not self._cam_writer_stop.is_set():
            # ── Deferred buffer flush (gzip + resize + fsync) ──
            # _flush_buffered() manages its own _hdf5_lock internally —
            # do NOT wrap it in another lock acquisition (threading.Lock
            # is non-reentrant; nesting would deadlock the writer thread).
            if self._flush_pending or (
                self._buffer is not None
                and self._buffer.size - self._flushed_frames >= self._flush_interval
            ):
                self._flush_buffered()
                self._flush_pending = False

            try:
                item = self._cam_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:  # sentinel
                break
            with self._hdf5_lock:
                self._append_camera(*item)
                self._cam_written += 1
                if self._cam_written % _hdf5_flush_every_n == 0:
                    self._file.flush()

        # Final flush before draining (buffer is static during drain).
        # _flush_buffered() manages its own _hdf5_lock internally.
        if self._buffer is not None and self._buffer.size > self._flushed_frames:
            self._flush_buffered()

        # Drain remaining camera items after stop signal
        while True:
            try:
                item = self._cam_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            with self._hdf5_lock:
                self._append_camera(*item)
                self._cam_written += 1

    # ── Camera HDF5 write (called from background writer OR inline at stop) ─

    def _append_camera(
        self,
        camera_frame: dict[str, Any] | None,
        camera_frames: dict[str, dict[str, Any]] | None,
        target_len: int,
    ) -> None:
        """Fill every camera dataset up to ``target_len`` grid slots.

        ``target_len`` is the aligned-grid length at enqueue time, so dataset
        index == grid index: loop slower than the grid (k>1), dropped queue
        items, a camera appearing mid-episode, or per-dataset slot skips all
        show up as "dataset shorter than target" and are healed by the next
        item — backfilled slots repeat the current frame, matching the
        TimestampAlignedBuffer forward-fill convention for non-camera streams.

        Called from the background writer thread (normal operation) or inline
        (drain at stop).  Caller must hold ``self._hdf5_lock``.
        """
        assert self._file is not None
        # Single-camera
        if camera_frame is not None:
            rgb = camera_frame.get("rgb")
            depth = camera_frame.get("depth")
            if rgb is not None:
                if "rgb" not in self._datasets:
                    # lzf (h5py built-in): ~5x faster than gzip-1 — the single
                    # writer thread must sustain the full grid rate or frames
                    # drop silently (gzip-1 measured ~12 items/s ceiling, v8).
                    self._datasets["rgb"] = self._file.create_dataset(
                        "rgb",
                        data=rgb[np.newaxis, ...],
                        maxshape=(None,) + rgb.shape,
                        chunks=True,
                        dtype=rgb.dtype,
                        compression="lzf",
                    )
                self._fill_to("rgb", rgb, target_len)
            if depth is not None:
                if "depth" not in self._datasets:
                    self._datasets["depth"] = self._file.create_dataset(
                        "depth",
                        data=depth[np.newaxis, ...],
                        maxshape=(None,) + depth.shape,
                        chunks=True,
                        dtype=depth.dtype if depth.dtype == np.uint16 else np.uint16,  # Z16 guard
                        compression="lzf",
                    )
                self._fill_to("depth", depth, target_len)

            # Fixed-size world-frame pointcloud (xyz + rgb), computed online
            # in the CameraProcess child — see sensor/pointcloud_processor.py.
            # gzip level 1: float32 FPS-downsampled coords compress ~2-3x,
            # fast enough for the background writer thread.
            pc = camera_frame.get("pointcloud")
            if pc is not None and camera_frame.get("pointcloud_valid", True):
                if "pointcloud" not in self._datasets:
                    self._datasets["pointcloud"] = self._file.create_dataset(
                        "pointcloud",
                        data=pc[np.newaxis, ...],
                        maxshape=(None,) + pc.shape,
                        chunks=(1,) + pc.shape,
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=1,
                    )
                self._fill_to("pointcloud", pc, target_len)

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
                if rgb is not None and hasattr(rgb, "shape"):
                    if rgb_key not in self._datasets:
                        self._datasets[rgb_key] = self._file.create_dataset(
                            rgb_key,
                            data=rgb[np.newaxis, ...],
                            maxshape=(None,) + rgb.shape,
                            chunks=True,
                            dtype=rgb.dtype if rgb.dtype == np.uint8 else np.uint8,
                            compression="lzf",
                        )
                    self._fill_to(rgb_key, rgb, target_len)
                if depth is not None and hasattr(depth, "shape"):
                    if depth_key not in self._datasets:
                        self._datasets[depth_key] = self._file.create_dataset(
                            depth_key,
                            data=depth[np.newaxis, ...],
                            maxshape=(None,) + depth.shape,
                            chunks=True,
                            dtype=depth.dtype if depth.dtype == np.uint16 else np.uint16,
                            compression="lzf",
                        )
                    self._fill_to(depth_key, depth, target_len)

    def _flush_buffered(self) -> None:
        """Write buffered non-camera streams to HDF5, keeping datasets resizable.

        Called periodically during recording (every ``_flush_interval`` frames)
        and finally at :meth:`stop_episode`.  On first call the datasets are
        created with ``maxshape=(None, ...)``; subsequent calls resize and
        append only the new frames.
        """
        if self._buffer is None or self._buffer.size == self._flushed_frames:
            return

        with self._hdf5_lock:
            self._ensure_hdf5()
            buf_data = self._buffer.data
            buf_size = self._buffer.size
            new_start = self._flushed_frames

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

            # Timestamp (stored separately from buf_data)
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

            # Push HDF5 metadata (chunk B-tree, resized shapes) to disk so a hard
            # crash (SIGKILL/segfault) loses at most one flush interval.
            self._file.flush()

    def stop_episode(self, success: bool = True, reason: str = "") -> str | None:
        """Signal end of episode; return path immediately, flush in background.

        The heavy HDF5 work (buffer flush, camera forward-fill, compression,
        metadata write, file close) runs on a daemon thread so the control
        loop stays responsive.  Callers must join_stop() before relying on
        the file (or before process exit).

        Args:
            success: stored as /meta success.
            reason: stored as /meta stop_reason; empty → "max_frames" when
                    the episode hit the frame cap, else "manual".
        """
        if not self._recording:
            return None

        # Capture the truncation flag before the reset below — drives /meta truncated.
        truncated = self._max_frames_reached

        # Mark as stopped *before* spawning the thread — add_frame() must
        # reject new frames immediately.  All data is already in _buffer
        # and _cam_queue; the background thread will drain both.
        self._recording = False
        self._max_frames_reached = False
        path = self._episode_path

        t = threading.Thread(
            target=self._stop_episode_impl,
            args=(success, reason, truncated),
            daemon=True,
            name="episode-stop",
        )
        t.start()
        self._stop_thread = t
        return path

    def join_stop(self, timeout: float = 30.0) -> bool:
        """Wait for the background stop daemon (HDF5 fully written + closed).

        Returns True when no flush is pending anymore.  On timeout the thread
        handle is KEPT so start_episode() keeps refusing to overlap — dropping
        it would let a late-finishing flush clobber a new episode's state.
        """
        t = self._stop_thread
        if t is None:
            return True
        if t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("episode-stop still flushing after %.0fs — keeping handle", timeout)
                return False
        self._stop_thread = None
        return True

    def _stop_episode_impl(self, success: bool, reason: str, truncated: bool) -> None:
        """Background: flush buffers, forward-fill cameras, write meta, close HDF5."""
        duration = time.perf_counter() - (self._start_time or 0.0)

        # ── Stop background camera writer ──
        # 1. Signal stop + push sentinel so the thread exits its loop.
        self._cam_writer_stop.set()
        try:
            self._cam_queue.put_nowait(None)
        except queue.Full:
            pass  # queue is full; sentinel won't fit — drain below instead

        # 2. Join the writer thread.
        if self._cam_writer is not None and self._cam_writer.is_alive():
            self._cam_writer.join(timeout=5.0)
            if self._cam_writer.is_alive():
                logger.warning("Camera writer thread did not exit within 5s")

        # 3. Safety drain: pick up anything the writer may have missed.
        while True:
            try:
                leftover = self._cam_queue.get_nowait()
            except queue.Empty:
                break
            if leftover is None:
                continue
            with self._hdf5_lock:
                self._append_camera(*leftover)
                self._cam_written += 1

        # ── Flush remaining buffered non-camera streams ──
        self._flush_buffered()
        buf_size = self._buffer.size if self._buffer is not None else 0

        # ── Camera forward-fill: pad camera datasets to match grid length ──
        # Camera frames are written by the background thread at ~50 Hz cadence,
        # but when k > 1 (minor timing jitter) the camera dataset falls slightly
        # behind the non-camera grid.  Forward-fill the last frame to keep every
        # dataset index-aligned.
        if self._file is not None and buf_size > 0:
            with self._hdf5_lock:
                for key in list(self._datasets.keys()):
                    if not (
                        key in ("rgb", "depth", "pointcloud")
                        or key.endswith("_rgb")
                        or key.endswith("_depth")
                    ):
                        continue
                    ds = self._datasets[key]
                    cam_len = ds.shape[0]
                    if 0 < cam_len < buf_size:
                        gap = buf_size - cam_len
                        logger.debug("camera tail-pad %s: +%d slots", key, gap)
                        last_frame = ds[cam_len - 1 : cam_len]  # keep dims: (1, ...)
                        repeats = np.repeat(last_frame, gap, axis=0)
                        ds.resize(buf_size, axis=0)
                        ds[cam_len:buf_size] = repeats

        self._buffer = None
        self._frame_count = buf_size

        # ── Write final metadata ──
        if self._file is not None:
            with self._hdf5_lock:
                meta = self._file["meta"]
                meta.attrs["schema_version"] = SCHEMA_VERSION
                meta.attrs["duration"] = duration
                meta.attrs["num_frames"] = self._frame_count
                meta.attrs["success"] = success
                meta.attrs["fps"] = self._frame_count / duration if duration > 0 else self.control_hz
                meta.attrs["min_frames_met"] = self._frame_count >= self.min_frames
                meta.attrs["has_camera"] = "rgb" in self._file
                meta.attrs["has_pointcloud"] = "pointcloud" in self._file
                meta.attrs["has_timestamps"] = "timestamp" in self._file
                meta.attrs["truncated"] = bool(truncated)
                meta.attrs["stop_reason"] = reason or ("max_frames" if truncated else "manual")
                # Enqueue-side camera drops (writer backlog): the target_len
                # backfill keeps datasets index-aligned, so these counters are
                # the only disk-side record that camera content was lost.
                meta.attrs["cam_frames_dropped"] = int(self._cam_dropped)
                meta.attrs["cam_items_written"] = int(self._cam_written)

        if self._cam_dropped > 0:
            total = self._cam_dropped + self._cam_written
            logger.warning(
                "Episode dropped %d/%d camera frames (%.1f%%) at the writer queue — "
                "rgb/depth content is forward-filled at those slots",
                self._cam_dropped,
                total,
                100.0 * self._cam_dropped / max(1, total),
            )

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
        self._flushed_frames = 0
        self._cam_seen = False
        self._last_camera_frame = None
        self._last_camera_frames = None
        self._last_T_base_eef = None
        self._cam_last_frame_number = None
        self._cam_last_change_ts = 0.0
        self._cam_writer = None
        self._cam_written = 0
        self._cam_dropped = 0

    def _fill_to(self, key: str, data: np.ndarray, target_len: int) -> None:
        """Fill dataset ``key`` up to ``target_len`` rows with ``data`` (broadcast).

        No-op when already at/past target — idempotent across the stop-time
        drain where queued items carry non-decreasing targets.
        """
        ds = self._datasets[key]
        n = ds.shape[0]
        if n < target_len:
            ds.resize(target_len, axis=0)
            ds[n:target_len] = data
