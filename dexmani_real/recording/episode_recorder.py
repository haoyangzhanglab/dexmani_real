"""EpisodeRecorder — HDF5-based teleoperation data recorder.

Single write path: state/action/vr streams are aligned to a fixed dt=1/control_hz
time grid at record time (TimestampAlignedBuffer) and flushed to HDF5 in bulk at
stop_episode(); camera frames are accumulated in memory and written to HDF5 in
periodic batches (~10 s), then tail-padded to grid length at stop — every dataset
is index-aligned by construction.

Ref: ManiUniCon accumulate-then-dump pattern (replay_buffer.py).
"""

from __future__ import annotations

__all__ = ["EpisodeRecorder", "StopResult"]

import atexit
import os
import shutil
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d
from dexmani_real.recording.timestamp_buffer import TimestampAlignedBuffer
from dexmani_real.recording.video_codec import VideoEncoder

DEFAULT_MAX_RECORD_FRAMES: int = 10000

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION_V11 = 11  # v11: +flag_ik_attempted +flag_frame_status (always recorded)


# ── atexit safety net ──
# The episode-stop thread is a daemon: if an entry point exits without
# join_stop() (estop path, second Ctrl-C inside a finally prompt), the
# interpreter kills it mid-flush and truncates the HDF5.  One hook joins
# every live recorder at interpreter exit.  No-op on SIGTERM/SIGKILL.
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
    ) -> None:
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.control_hz = float(control_hz)
        self.min_frames = int(min_frames)

        # Opt-in additive stream (schema v9): the delta-clipped arm joint command
        # actually forwarded to the SDK each tick, as opposed to action_arm_joint
        # (the IK target).  Off → byte-identical v8 behavior; nothing is wired to
        # pass the kwarg yet, so the flag alone is inert.
        self.arm_sent_stream: bool = bool(arm_sent_stream)



        self._file: Any = None  # h5py.File | None — data.h5 (non-camera + pointcloud)
        self._depth_file: Any = None  # h5py.File | None — depth.h5 (uint16, gzip-1)
        self._rgb_encoder: VideoEncoder | None = None  # MP4 streaming encoder for RGB
        self._frame_count: int = 0
        self._recording: bool = False
        self._max_frames_reached: bool = False
        self._start_time: float | None = None
        self._episode_dir: str | None = None  # episode_XXX/ directory
        self._temp_dir: str | None = None  # .tmp_episode_XXX/ directory
        self._datasets: dict[str, Any] = {}
        self._datasets_depth: dict[str, Any] = {}  # depth-only datasets in _depth_file

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

        # Camera frame accumulation: unique frames are collected in memory and
        # written to HDF5 in periodic batches (~every _flush_interval grid slots).
        # Forward-fill is deferred to stop_episode() — each frame is stored once
        # with its grid-end index in _cam_grid_end (accumulated across flushes).
        self._cam_frames: list[dict[str, Any]] = []
        self._cam_grid_end: list[int] = []

        # Pending async stop_episode thread (None = no pending stop).
        # Guarded by start_episode() to prevent overlapping episodes.
        self._stop_thread: threading.Thread | None = None

        # Error from the last _stop_episode_impl (ENOSPC, etc.) — set inside
        # the daemon thread; callers poll via stop_error after join_stop().
        self._stop_error: str | None = None

        # Non-blocking stop tracking (harvested by poll_stop()).
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

    @property
    def max_frames_reached(self) -> bool:
        return self._max_frames_reached

    @property
    def stop_error(self) -> str | None:
        """Error message from the last _stop_episode_impl, or None if clean."""
        return self._stop_error

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
        # A crashed stop (ENOSPC, etc.) has already reset state — allow
        # a new start after logging the error.
        if not self.join_stop(timeout=15.0):
            if self._stop_error is not None:
                logger.warning(
                    "Previous stop crashed (%s) — state was reset, allowing new start",
                    self._stop_error,
                )
                self._stop_thread = None
            else:
                logger.error("Previous episode still flushing — refusing to start a new one")
                return False

        if self._recording:
            return False

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
        self._datasets_depth = {}
        self._flushed_frames = 0
        self._cam_frames = []
        self._cam_grid_end = []

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
        self._depth_file = None
        self._rgb_encoder = None
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

        # Opt-in schema v9 marker — written only when enabled so the default
        # v8 file keeps its exact meta layout.
        if self.arm_sent_stream:
            meta.attrs["arm_sent_stream"] = True

        self._write_camera_meta_attrs(meta)

        # Collection-config snapshot (control mode, EMA alphas, delta clips) —
        # essential for downstream reproducibility.  Values are pre-sanitized to
        # h5py-compatible scalars/strings by the controller.
        meta.attrs["skip_initial_frames"] = int(p.get("skip_initial_frames", 0))
        record_config = p.get("record_config") or {}
        for key, val in record_config.items():
            meta.attrs[key] = val

    def _write_camera_meta_attrs(self, meta: h5py.Group) -> None:
        """Camera identity/geometry attrs from _pending_meta (None entries skipped).

        Idempotent — _stop_episode_impl re-runs it so values supplied late
        still reach /meta after the initial lazy write.
        """
        p = self._pending_meta
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

    def add_frame(
        self,
        state,
        action,
        vr_frame: dict[str, Any],
        camera_frame: dict[str, Any] | None = None,
        signals: dict[str, Any] | None = None,
        arm_qpos_sent: np.ndarray | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> bool:
        if not self._recording or self._buffer is None:
            return False

        if self._frame_count >= self.max_frames:
            logger.warning("Episode reached max_frames=%d, auto-stopping.", self.max_frames)
            self._max_frames_reached = True
            return False

        # ── Skip initial frames (begin-transition pose noise) ──
        # Return False (not recorded) so the caller's frame counter stays
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
        ts = float(state.timestamp)

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
            "hand_tactile_force": np.asarray(state.hand_tactile_force, dtype=np.float64),
            "hand_tactile_contact": np.asarray(state.hand_tactile_contact, dtype=bool),
            "hand_tipboard_err": np.asarray(state.hand_tipboard_err, dtype=np.int32),
            "hand_commboard_err": np.asarray(state.hand_commboard_err, dtype=np.int32),
            "hand_jointboard_err": np.asarray(state.hand_jointboard_err, dtype=np.int32),
            "hand_current": np.asarray(state.hand_current, dtype=np.float64) if state.hand_current is not None else np.full(12, np.nan),
            # ── Connection status ──
            # Distinguishes "physically disconnected (NaN qpos + connected=False)"
            # from "connected but read failed (NaN qpos + connected=True)".
            "arm_connected": bool(state.arm_connected),
            "hand_connected": bool(state.hand_connected),
            # ── Actions ──
            "action_arm_joint": np.asarray(action.arm_qpos_cmd, dtype=np.float64),
            "action_arm_ee": _make_action_ee(),
            "action_hand_joint": np.asarray(action.hand_qpos_cmd, dtype=np.float64),
            # ── Flags ──
            "flag_ik_ok": bool(sig.get("ik_ok", False)),
            "flag_ik_attempted": bool(sig.get("ik_attempted", True)),  # default True: normal frames
            "flag_retarget_ok": bool(sig.get("retarget_ok", False)),
            "flag_held": bool(sig.get("held", False)),
            "flag_safety_reject": bool(sig.get("flag_safety_reject", False)),
            "camera_health": int(camera_frame.get("camera_health", 0) if camera_frame is not None else 0),
            # ── Frame quality (schema v11) ──
            # 0=ok, 1=held (gate reject), 2=ik_fail, 3=safety_reject
            "flag_frame_status": int(sig.get("frame_status", 0)),
            # ── VR ──
            "vr_wrist_pos": np.asarray(vr_frame["wrist_pos"], dtype=np.float64),
            "vr_wrist_rot6d": quat_wxyz_to_rot6d(np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)),
            "vr_landmarks": np.asarray(vr_frame["landmarks"], dtype=np.float64),
        }
        # ── Opt-in sent-command stream (schema v9) ──
        # None (kwarg unset) → zeros: the TimestampAlignedBuffer forward-fills
        # the slot from this row, consistent with the action_arm_ee NaN-on-missing
        # convention for optional action streams.  Gated on the constructor flag so
        # an accidental kwarg can never add a dataset to a default (v8) recording.
        if self.arm_sent_stream:
            sent = np.asarray(arm_qpos_sent, dtype=np.float64) if arm_qpos_sent is not None else np.full(7, np.nan)
            data["action_arm_joint_sent"] = sent

        # ── Diagnostics (v10): continuous telemetry — auto-discovered by _flush_buffered ──
        if diagnostics:
            for key, val in diagnostics.items():
                data[key] = np.asarray(val, dtype=np.float64)

        prev_size = self._buffer.size
        self._buffer.add(data, timestamp=float(state.timestamp))

        if self._buffer.recording_stopped:
            self._max_frames_reached = True
            return False

        self._frame_count = self._buffer.size
        k = self._buffer.size - prev_size  # grid slots advanced (usually 1; 0 = dup bucket)

        # ── Periodic non-camera flush: write buffered streams to HDF5 ──
        if self._buffer.size - self._flushed_frames >= self._flush_interval:
            self._ensure_hdf5()
            self._flush_buffered()

        # ── Camera streams → accumulate in memory (ManiUniCon pattern) ──
        # Depth + pointcloud: collected in memory, written at stop time.
        # RGB: streamed to MP4 encoder during recording (no stop-time dump).
        if camera_frame is not None and k > 0:
            self._cam_frames.append(camera_frame)
            self._cam_grid_end.append(self._buffer.size)
            # ── RGB → MP4 streaming encode (lazy encoder init on first frame) ──
            rgb = camera_frame.get("rgb")
            if rgb is not None:
                if self._rgb_encoder is None:
                    assert self._temp_dir is not None
                    h, w = rgb.shape[:2]
                    mp4_path = Path(self._temp_dir) / "rgb.mp4"
                    self._rgb_encoder = VideoEncoder(mp4_path, fps=self.control_hz, width=w, height=h)
                self._rgb_encoder.write_frame(rgb)
        elif self._cam_grid_end and k > 0:
            self._cam_grid_end[-1] = self._buffer.size

        return True

    # ── Camera batch write (stop-time only) ────────────────────────────

    def _write_all_camera_frames(self) -> None:
        """Write depth + pointcloud to HDF5 in one pass at stop time.

        RGB is streamed to MP4 during recording via ``_rgb_encoder`` —
        no stop-time bulk write needed.  Depth goes to ``depth.h5``
        (gzip-1, pre-allocated); pointcloud stays in ``data.h5``.
        """
        if not self._cam_frames:
            return
        self._ensure_hdf5()

        # ── Collect depth frames for pre-allocated batch write ──
        depth_frames: list[np.ndarray] = []
        for camera_frame in self._cam_frames:
            depth = camera_frame.get("depth")
            if depth is not None:
                depth_frames.append(np.asarray(depth, dtype=np.uint16))

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
                else:
                    ds_pc = self._datasets["pointcloud"]
                    n = ds_pc.shape[0]
                    ds_pc.resize(n + 1, axis=0)
                    ds_pc[n] = pc

        # ── Depth → depth.h5: pre-allocate + batch write (gzip-1, ~32% smaller than lzf) ──
        if depth_frames:
            depth_shape = (len(depth_frames),) + depth_frames[0].shape
            ds = self._depth_file.create_dataset(
                "depth",
                shape=depth_shape,
                maxshape=(None,) + depth_frames[0].shape,
                dtype=np.uint16,
                chunks=(1,) + depth_frames[0].shape,
                compression="gzip",
                compression_opts=1,
            )
            for i, frame in enumerate(depth_frames):
                ds[i] = frame
            self._datasets_depth["depth"] = ds

    def _forward_fill_cameras(self, buf_size: int) -> None:
        """Forward-fill camera datasets to match the grid length.

        Each camera frame was stored as a single row during recording (one
        per unique frame).  ``_cam_grid_end`` maps each frame index to its
        grid span: frame *i* covers grid slots ``[start_i, end_i)`` where
        ``start_i`` is ``_cam_grid_end[i-1]`` (or 0) and ``end_i`` is
        ``_cam_grid_end[i]``.  The last frame also covers the tail
        ``[_cam_grid_end[-1], buf_size)``.

        Fills right-to-left so source rows are never overwritten before they
        are read.

        A dataset may have fewer rows than ``_cam_grid_end`` entries when a
        modality is missing for the first few frames (e.g. pointcloud before
        the first valid cloud).  The mapping ``i → max(0, i - offset)``
        handles this: early frames without data forward-fill from the first
        available row.
        """
        if not self._cam_grid_end or buf_size == 0 or self._file is None:
            return

        M = len(self._cam_grid_end)
        # RGB is in MP4 (no forward-fill needed — encoder writes every grid slot).
        for key in ("depth", "pointcloud"):
            datasets = self._datasets_depth if key == "depth" else self._datasets
            if key not in datasets:
                continue
            ds = datasets[key]
            cam_len = ds.shape[0]
            if cam_len == 0:
                continue

            ds.resize(buf_size, axis=0)
            offset = M - cam_len  # frames at the start without this modality

            # Fill right-to-left: _cam_grid_end[i] >= i guarantees source
            # row i (at dataset position i) sits to the left of the span
            # it fills, so later frames' writes never corrupt it.
            for i in range(M - 1, -1, -1):
                start = self._cam_grid_end[i - 1] if i > 0 else 0
                end = self._cam_grid_end[i]
                if end > start:
                    src = max(0, i - offset)
                    ds[start:end] = ds[src]

            # Tail: last frame covers the remainder of the grid.
            tail_start = self._cam_grid_end[-1]
            if tail_start < buf_size:
                ds[tail_start:buf_size] = ds[cam_len - 1]

    def _ensure_hdf5(self) -> None:
        """Lazily create ``data.h5`` + ``depth.h5`` in the temp directory.

        The data file carries all non-camera streams + pointcloud + /meta;
        the depth file holds only uint16 depth frames (gzip-1 compression).
        """
        if self._file is not None:
            return
        assert self._temp_dir is not None
        self._file = h5py.File(str(Path(self._temp_dir) / "data.h5"), "w")
        self._write_meta_attrs(self._file.create_group("meta"))
        self._depth_file = h5py.File(str(Path(self._temp_dir) / "depth.h5"), "w")

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
        # and _cam_frames; the stop daemon will flush both.
        self._recording = False
        self._max_frames_reached = False
        path = self._episode_dir

        # Snapshot stop metadata BEFORE spawning daemon (daemon overwrites
        # self._frame_count during _stop_episode_impl_inner).
        self._stop_success = success
        self._stop_path = path
        self._stop_frame_count = self._frame_count

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

        Returns True when the flush completed cleanly.  Returns False on timeout
        (thread still alive — handle KEPT so start_episode() keeps refusing) OR
        when the stop thread crashed (handle cleared, _stop_error set — caller
        must inspect stop_error to distinguish).

        The entry MUST consult stop_error after a join_stop() that returned True:
        a True from a crashed thread means "no pending flush" (the daemon is dead
        and can't be re-joined), NOT "file written successfully".
        """
        t = self._stop_thread
        if t is None:
            if self._stop_error is not None:
                # Thread already joined (dead from crash) in a prior call.
                return False
            return True
        if t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("episode-stop still flushing after %.0fs — keeping handle", timeout)
                return False
        # Thread finished — check whether it crashed.
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

        Safe to call at 16 Hz from the main control loop.  Returns immediately
        — never blocks on I/O.  After the first call that returns
        ``done=True``, the internal state is reset and subsequent calls
        return a clean sentinel (``done=True, path=None``) until the next
        ``stop_episode()``.

        Idempotent: once a stop completes, repeated calls return the same
        result (cached in the first returned ``StopResult``, then cleared).
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
        # Thread finished — harvest result and reset for the next stop.
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

    def _stop_episode_impl(self, success: bool, reason: str, truncated: bool) -> None:
        """Background: flush buffers, forward-fill cameras, write meta, close HDF5.

        ENOSPC / OSError at any h5py call site kills this daemon thread silently.
        The try/except captures the error into _stop_error so join_stop() and the
        entry can report failure instead of printing "已保存" for a truncated file.
        """
        try:
            self._stop_episode_impl_inner(success, reason, truncated)
        except Exception as exc:
            self._stop_error = f"{type(exc).__name__}: {exc}"
            logger.error("stop_episode failed: %s — HDF5 may be truncated", self._stop_error)
            # Best-effort close: metadata may flush partial B-tree updates.
            try:
                if self._rgb_encoder is not None:
                    self._rgb_encoder.close()
            except Exception:
                pass
            self._rgb_encoder = None
            try:
                if self._file is not None:
                    self._file.close()
            except Exception:
                pass
            self._file = None
            try:
                if self._depth_file is not None:
                    self._depth_file.close()
            except Exception:
                pass
            self._depth_file = None
            # Clean up temp directory if it still exists.
            _tmp = self._temp_dir
            if _tmp is not None:
                self._discard_temp_files(_tmp)
            self._reset_episode_state()

    def _stop_episode_impl_inner(self, success: bool, reason: str, truncated: bool) -> None:
        """Inner body of _stop_episode_impl — extracted so the try/except wrapper
        can reset state on any exception without duplicating the reset list."""
        duration = time.perf_counter() - (self._start_time or 0.0)

        # ── Write all accumulated camera frames (accumulate-then-dump) ──
        # Frames were collected in memory during recording with zero disk I/O
        # on the 16 Hz hot path.  Written once here at episode end.
        self._write_all_camera_frames()

        # ── Flush remaining buffered non-camera streams ──
        self._flush_buffered()
        buf_size = self._buffer.size if self._buffer is not None else 0

        # ── Camera forward-fill: broadcast unique frames across the grid ──
        # Each camera frame was stored as a single row during recording;
        # _cam_grid_end maps rows → grid spans.  One pass at stop time
        # replaces the per-batch _fill_to() + tail-pad online machinery.
        self._forward_fill_cameras(buf_size)

        self._buffer = None
        self._frame_count = buf_size

        # ── Close RGB encoder (streaming MP4) ──
        _had_rgb = self._rgb_encoder is not None
        if self._rgb_encoder is not None:
            self._rgb_encoder.close()
            self._rgb_encoder = None

        # ── Write final metadata ──
        if self._file is not None:
            meta = self._file["meta"]
            meta.attrs["schema_version"] = SCHEMA_VERSION_V11
            meta.attrs["duration"] = duration
            meta.attrs["num_frames"] = self._frame_count
            meta.attrs["success"] = success
            meta.attrs["fps"] = self._frame_count / duration if duration > 0 else self.control_hz
            meta.attrs["min_frames_met"] = self._frame_count >= self.min_frames
            meta.attrs["has_camera"] = _had_rgb
            meta.attrs["has_pointcloud"] = "pointcloud" in self._datasets
            meta.attrs["has_timestamps"] = "timestamp" in self._datasets
            meta.attrs["truncated"] = bool(truncated)
            meta.attrs["stop_reason"] = reason or ("max_frames" if truncated else "manual")
            # Camera meta backfill: the initial lazy write may have run
            # before the camera child finished connect — re-write so late
            # values land in the file.
            self._write_camera_meta_attrs(meta)

        if self._file is not None:
            self._file.close()
        self._file = None
        if self._depth_file is not None:
            self._depth_file.close()
            self._depth_file = None

        # ── Atomic finalise ──
        # success: rename temp dir → final dir.
        # discard:  remove temp dir.
        _final = self._episode_dir
        _tmp = self._temp_dir
        if _tmp is not None and _final is not None:
            if success:
                self._rename_temp_to_final(_tmp, _final)
            else:
                self._discard_temp_files(_tmp)

        self._reset_episode_state()
    def _reset_episode_state(self) -> None:
        """Reset all mutable episode state to defaults (called from both the
        success path and the crash-handler in _stop_episode_impl)."""
        self._datasets.clear()
        self._datasets_depth.clear()
        self._recording = False
        self._max_frames_reached = False
        self._frame_count = 0
        self._start_time = None
        self._episode_dir = None
        self._temp_dir = None
        self._buffer = None
        self._flushed_frames = 0
        self._cam_frames = []
        self._cam_grid_end = []

    # ── Atomic file finalisation ──────────────────────────────────────

    @staticmethod
    def _try_rename(src: str, dst: str) -> None:
        """Rename *src* → *dst*; fall back to copy+remove on cross-device error."""
        try:
            os.rename(src, dst)
        except OSError:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                shutil.rmtree(src)
            else:
                shutil.copy2(src, dst)
                os.unlink(src)

    @classmethod
    def _discard_temp_files(cls, tmp: str) -> None:
        """Remove temp directory and all contents. Never raises."""
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except OSError:
            pass

    def _rename_temp_to_final(self, tmp: str, final: str) -> None:
        """Rename temp directory to final episode directory."""
        self._try_rename(tmp, final)



