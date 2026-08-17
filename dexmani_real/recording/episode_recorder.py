"""Transactional HDF5 v16 episode serialization.

State, action, VR, and camera rows are written one-for-one on the policy grid.
The recorder verifies all sidecars before atomically publishing an episode.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.config.defaults import camera
from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d
from dexmani_real.recording.camera_stream_writer import CameraStreamWriter, CameraStreamWriterConfig
from dexmani_real.recording.episode_schema import (
    ARM_SENT_DATASET,
    ARM_SENT_MARKER,
    EPISODE_SCHEMA_VERSION,
    SEMANTIC_META_ATTRS_V17,
    normalize_diagnostics_v17,
    validate_data_layout_v17,
    validate_source_frame_keys_v17,
)
from dexmani_real.recording.timestamp_buffer import TimestampAlignedBuffer
from dexmani_real.recording.transaction import atomic_publish
from dexmani_real.recording.video_codec import VideoDecoder
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

logger = get_logger(__name__)

DEFAULT_MAX_RECORD_FRAMES: int = 10000
SCHEMA_VERSION = EPISODE_SCHEMA_VERSION
_CAMERA_WRITER_CLOSE_TIMEOUT_S = 60.0
_PREVIOUS_EPISODE_STOP_TIMEOUT_S = 15.0
_PROCESS_EXIT_STOP_TIMEOUT_S = 60.0


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
            if not rec.join_stop(timeout=_PROCESS_EXIT_STOP_TIMEOUT_S):
                logger.error("recorder did not finish before interpreter shutdown")
        except Exception:
            logger.warning("recorder cleanup failed during interpreter shutdown", exc_info=True)


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
        camera_writer_config: CameraStreamWriterConfig | None = None,
        resolved_config_hash: str | None = None,
        provenance: dict[str, str] | None = None,
    ) -> None:
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        if resolved_config_hash is None or len(resolved_config_hash) != 64:
            raise ValueError("EpisodeRecorder requires a resolved config SHA-256")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.control_hz = float(control_hz)
        self.min_frames = int(min_frames)
        self._resolved_config_hash = resolved_config_hash
        self._provenance = dict(provenance or {})

        # Opt-in stream: the safe arm command actually forwarded to the worker,
        # kept distinct from the accepted candidate in action_arm_joint.
        self.arm_sent_stream: bool = bool(arm_sent_stream)

        self._file: Any = None  # h5py.File | None — data.h5 (control streams + metadata)
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
        self._pending_meta: dict[str, Any] = {}  # deferred metadata until HDF5 created

        # Record-time aligned buffer for non-camera streams + periodic flush.
        self._buffer: TimestampAlignedBuffer | None = None
        self._flush_interval: int = max(1, int(round(10.0 * self.control_hz)))  # frames (~10s)
        self._flushed_frames: int = 0

        # Skip the first N add_frame() calls per episode (begin-transition noise).
        # The grid is re-anchored to the first accepted frame's timestamp so the
        # dropped frames leave no causal hold-last gap.
        self._skip_initial_frames: int = 0
        self._skipped_so_far: int = 0
        self._last_control_run_generation: int | None = None

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

    @property
    def camera_writer_error(self) -> str | None:
        """Latched camera sidecar error requiring episode discard."""
        return self._camera_writer.error if self._camera_writer is not None else None

    @staticmethod
    def _build_action_ee(action) -> "np.ndarray":
        """Build a (9,) array [eef_pos(3), eef_rot6d(6)] from a RobotAction."""
        pos = action.target_eef_pos
        rot6d = action.target_eef_rot6d
        p = np.asarray(pos, dtype=np.float64) if pos is not None else np.full(3, np.nan)
        r = np.asarray(rot6d, dtype=np.float64) if rot6d is not None else np.full(6, np.nan)
        return np.concatenate([p, r])

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
        # Wait for any pending async stop_episode to finish — it owns
        # _file, _buffer, _datasets, _pending_meta and will reset them on
        # completion.  Refuse to start while it is still alive: proceeding
        # would let the old stop thread clobber the new episode's state.
        # A crashed stop (ENOSPC, etc.) has already reset state — allow
        # a new start after logging the error.
        if not self.join_stop(timeout=_PREVIOUS_EPISODE_STOP_TIMEOUT_S):
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
        self._flushed_frames = 0

        # Skip-initial-frames gate — clamp below max_frames so we never drop all.
        self._skip_initial_frames = max(0, min(int(skip_initial_frames), self.max_frames - 1))
        self._skipped_so_far = 0
        self._last_control_run_generation = None

        # Store metadata for deferred write (HDF5 is created lazily).
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

        # Defer data.h5 creation to the first periodic flush / stop_episode();
        # start the record-time aligned buffer for non-camera streams.
        dt = 1.0 / self.control_hz
        self._buffer = TimestampAlignedBuffer(
            start_time=self._start_time,
            dt=dt,
            max_record_steps=self.max_frames,
            # Only tolerate floating-point representations of an exact grid
            # boundary. Scheduling jitter after a deadline must move the source
            # to the next slot; rounding it backward would violate causality.
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
        meta.attrs["control_hz"] = self.control_hz  # nominal grid rate; dt = 1/control_hz
        meta.attrs["fps"] = self.control_hz
        if self._resolved_config_hash is not None:
            meta.attrs["resolved_config_sha256"] = self._resolved_config_hash
        for key, value in sorted(self._provenance.items()):
            meta.attrs[f"provenance_{key}"] = str(value)

        # Additive, self-describing semantics for fields whose numeric layout
        # remains unchanged in v16. Historical readers may ignore these attrs.
        for key, semantic_value in SEMANTIC_META_ATTRS_V17.items():
            meta.attrs[key] = semantic_value

        # The conditional sent-command dataset and this marker must agree.
        # The marker remains absent when the optional v16 stream is disabled.
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
            {"0": "OK", "1": "CLOCK_RESET", "2": "DUPLICATE", "3": "FRAME_GAP", "4": "BACKLOG"},
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
            # If the live camera serial was supplied, verify it matches the named
            # calibration entry — a wrong camera_name would otherwise silently
            # embed the wrong extrinsics/serial into the dataset.
            calib_meta = calib.to_meta_dict(camera_name, expected_serial=camera_serial)
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
        control_run_generation: int = 0,
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

        # ── Non-camera streams → record-time aligned buffer ──
        sig = signals or {}
        diagnostic_values = normalize_diagnostics_v17(diagnostics)

        ts = float(state.timestamp)
        run_generation = int(control_run_generation)
        if run_generation < 0:
            raise ValueError("control_run_generation must be non-negative")
        # The first accepted source and every command-quiescence boundary start
        # a new wall-time segment. Storage remains contiguous, while the real
        # pause is retained as a timestamp jump instead of synthetic actions.
        if self._last_control_run_generation is None or run_generation != self._last_control_run_generation:
            self._buffer.reanchor(ts)

        camera_health = int(camera_frame.get("camera_health", 1)) if camera_frame is not None else 1
        camera_fresh = bool(camera_frame.get("camera_fresh", False)) if camera_frame is not None else False

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
            "hand_current": (
                np.asarray(state.hand_current, dtype=np.float64)
                if state.hand_current is not None
                else np.full(HAND_JOINT_SHAPE, np.nan)
            ),
            # ── Connection status ──
            # Distinguishes "physically disconnected (NaN qpos + connected=False)"
            # from "connected but read failed (NaN qpos + connected=True)".
            "arm_connected": bool(state.arm_connected),
            "hand_connected": bool(state.hand_connected),
            # ── Hand health/compatibility flags ──
            "hand_qpos_stale": bool(state.hand_qpos_stale),
            "hand_error_state": bool(state.hand_error_state),
            # Arm command timing.
            "arm_last_cmd_seq": int(state.arm_last_cmd_seq),
            "arm_last_cmd_queue_latency_s": float(state.arm_last_cmd_queue_latency_s),
            "arm_last_cmd_apply_latency_s": float(state.arm_last_cmd_apply_latency_s),
            "arm_last_cmd_sdk_duration_s": float(state.arm_last_cmd_sdk_duration_s),
            "arm_last_cmd_is_hold": bool(state.arm_last_cmd_is_hold),
            # ── Actions ──
            "action_arm_joint": np.asarray(action.arm_qpos_cmd, dtype=np.float64),
            "action_arm_ee": self._build_action_ee(action),
            "action_hand_joint": np.asarray(action.hand_qpos_cmd, dtype=np.float64),
            # ── Causal/action protocol provenance ──
            "observation_id": int(sig.get("observation_id", 0)),
            "observation_anchor_monotonic_ns": int(sig.get("observation_anchor_monotonic_ns", 0)),
            "arm_source_sequence": int(sig.get("arm_source_sequence", 0)),
            "hand_source_sequence": int(sig.get("hand_source_sequence", 0)),
            "vr_source_sequence": int(sig.get("vr_source_sequence", 0)),
            "camera_source_sequence": int(sig.get("camera_source_sequence", 0)),
            "arm_source_monotonic_ns": int(sig.get("arm_source_monotonic_ns", 0)),
            "hand_source_monotonic_ns": int(sig.get("hand_source_monotonic_ns", 0)),
            "vr_source_monotonic_ns": int(sig.get("vr_source_monotonic_ns", 0)),
            "camera_source_monotonic_ns": int(
                sig.get(
                    "camera_source_monotonic_ns",
                    camera_frame.get("source_monotonic_ns", 0) if camera_frame is not None else 0,
                )
            ),
            "arm_publish_monotonic_ns": int(sig.get("arm_publish_monotonic_ns", 0)),
            "hand_publish_monotonic_ns": int(sig.get("hand_publish_monotonic_ns", 0)),
            "vr_publish_monotonic_ns": int(sig.get("vr_publish_monotonic_ns", 0)),
            "camera_publish_monotonic_ns": int(
                sig.get(
                    "camera_publish_monotonic_ns",
                    camera_frame.get("publish_monotonic_ns", 0) if camera_frame is not None else 0,
                )
            ),
            "observation_source_receive_monotonic_ns": np.asarray(
                sig.get("observation_source_receive_monotonic_ns", np.zeros(4)), dtype=np.uint64
            ),
            "observation_source_age_s": np.asarray(
                sig.get("observation_source_age_s", np.full(4, np.nan)), dtype=np.float64
            ),
            "observation_source_skew_s": np.asarray(
                sig.get("observation_source_skew_s", np.full(4, np.nan)), dtype=np.float64
            ),
            "observation_history_valid_mask": np.asarray(
                sig.get("observation_history_valid_mask", np.zeros((4, 1), dtype=bool)), dtype=bool
            ),
            "observation_valid": bool(sig.get("observation_valid", False)),
            "observation_skew_s": float(sig.get("observation_skew_s", np.nan)),
            "action_id": int(sig.get("action_id", 0)),
            "action_created_monotonic_ns": int(sig.get("action_created_monotonic_ns", 0)),
            "action_target_monotonic_ns": int(sig.get("action_target_monotonic_ns", 0)),
            "action_valid_until_monotonic_ns": int(sig.get("action_valid_until_monotonic_ns", 0)),
            "action_arm_joint_raw": np.asarray(sig.get("action_arm_joint_raw", action.arm_qpos_cmd), dtype=np.float64),
            "flag_action_queued": bool(sig.get("action_queued", False)),
            "tactile_fresh": bool(sig.get("tactile_fresh", False)),
            "tactile_source_monotonic_ns": int(sig.get("tactile_source_monotonic_ns", 0)),
            "tactile_calibrated": bool(sig.get("tactile_calibrated", False)),
            "tactile_unit_code": int(sig.get("tactile_unit_code", 0)),
            "pointcloud_valid_depth_ratio": float(sig.get("pointcloud_valid_depth_ratio", np.nan)),
            # ── Flags ──
            "flag_ik_ok": bool(sig.get("ik_ok", False)),
            "flag_ik_attempted": bool(sig.get("ik_attempted", True)),  # default True: normal frames
            "flag_retarget_ok": bool(sig.get("retarget_ok", False)),
            "flag_held": bool(sig.get("held", False)),
            "flag_safety_reject": bool(sig.get("flag_safety_reject", False)),
            "camera_health": camera_health,
            "flag_camera_fresh": camera_fresh,
            "camera_frame_number": int(camera_frame.get("frame_number", 0)) if camera_frame is not None else 0,
            "camera_ring_sequence": int(camera_frame.get("ring_sequence", 0)) if camera_frame is not None else 0,
            "camera_device_timestamp_s": (
                float(camera_frame.get("device_timestamp_s", np.nan)) if camera_frame is not None else np.nan
            ),
            "camera_capture_monotonic_s": (
                float(camera_frame.get("capture_monotonic_s", np.nan)) if camera_frame is not None else np.nan
            ),
            "camera_age_s": float(camera_frame.get("camera_age_s", np.nan)) if camera_frame is not None else np.nan,
            "camera_generation": int(camera_frame.get("camera_generation", 0)) if camera_frame is not None else 0,
            "camera_clock_reset": bool(camera_frame.get("clock_reset", False)) if camera_frame is not None else False,
            "camera_duplicate": bool(camera_frame.get("duplicate", False)) if camera_frame is not None else False,
            "camera_frame_gap": int(camera_frame.get("frame_gap", 0)) if camera_frame is not None else 0,
            "camera_backlog_s": (float(camera_frame.get("backlog_s", np.nan)) if camera_frame is not None else np.nan),
            # ── Frame quality (schema v11) ──
            # 0=ok, 1=held (gate reject), 2=ik_fail, 3=safety_reject
            "flag_frame_status": int(sig.get("frame_status", 0)),
            # ── VR ──
            "vr_wrist_pos": np.asarray(vr_frame["wrist_pos"], dtype=np.float64),
            "vr_wrist_rot6d": quat_wxyz_to_rot6d(np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)),
            "vr_landmarks": np.asarray(vr_frame["landmarks"], dtype=np.float64),
            # ── Optional policy diagnostics (NaN when unavailable) ──
            "tracking_error": np.nan,
            "ik_solve_time_ms": np.nan,
            "target_pos_before_clamp": np.full(3, np.nan),
            "head_quat_wxyz": np.full(4, np.nan),
            "target_eef_pos_raw": np.full(3, np.nan),
            "target_eef_rot6d_raw": np.full(6, np.nan),
            "action_hand_joint_raw": np.asarray(action.hand_qpos_cmd, dtype=np.float64),
            "policy_map_time_ms": np.nan,
            "hand_retarget_time_ms": np.nan,
            "transition_check_time_ms": np.nan,
            "policy_compute_time_ms": np.nan,
        }
        # ── Conditional sent-command stream (schema v16) ──
        # None (kwarg unset) → NaN placeholder for this source sample; causal
        # alignment may only hold it into later slots, never backward-fill an
        # earlier slot. Gated on the constructor flag so
        # an accidental kwarg can never add a dataset when the stream is disabled.
        if self.arm_sent_stream:
            sent = (
                np.asarray(arm_qpos_sent, dtype=np.float64)
                if arm_qpos_sent is not None
                else np.full(ARM_JOINT_SHAPE, np.nan)
            )
            data[ARM_SENT_DATASET] = sent

        # Diagnostics are fixed-schema value overrides, never an extension
        # mechanism. ``normalize_diagnostics_v17`` rejects unknown keys,
        # reserved-field collisions, and incorrect tail shapes.
        data.update(diagnostic_values)
        source_layout_errors = validate_source_frame_keys_v17(set(data), arm_sent_stream=self.arm_sent_stream)
        if source_layout_errors:
            raise RuntimeError("schema-v16 source frame mismatch: " + "; ".join(source_layout_errors))

        add_result = self._buffer.add(data, timestamp=ts)
        if add_result.source_written:
            self._last_control_run_generation = run_generation

        self._frame_count = add_result.size
        prev_size = add_result.previous_size
        k = add_result.slots_written  # grid slots advanced (usually 1; 0 = dup bucket)

        # A single live camera observation may advance across several skipped
        # grid deadlines. Only its causal source slot can be fresh; earlier
        # synthetic slots retain shape but carry false validity.
        if k > 0:
            new_slice = slice(prev_size, self._buffer.size)
            source_valid = np.asarray(self._buffer.data["flag_sample_valid"][new_slice], dtype=bool)
            # Policy supplies its exact monotonic grid deadline as
            # ``state.timestamp``. Give every synthesized gap slot its own
            # deadline rather than retaining the previous source's anchor.
            grid_anchor_ns = np.rint(self._buffer.timestamps[new_slice] * 1e9).astype(np.uint64)
            self._buffer.data["observation_anchor_monotonic_ns"][new_slice] = grid_anchor_ns
            history_valid = np.asarray(self._buffer.data["observation_history_valid_mask"][new_slice, :, 0], dtype=bool)
            source_monotonic_ns = np.column_stack(
                [
                    self._buffer.data[f"{name}_source_monotonic_ns"][new_slice]
                    for name in ("arm", "hand", "vr", "camera")
                ]
            ).astype(np.uint64)
            source_age_s = np.full(history_valid.shape, np.nan, dtype=np.float64)
            causal = history_valid & (source_monotonic_ns <= grid_anchor_ns[:, None])
            source_age_delta_ns = grid_anchor_ns[:, None].astype(np.float64) - source_monotonic_ns.astype(np.float64)
            source_age_s[causal] = source_age_delta_ns[causal] / 1e9
            self._buffer.data["observation_source_age_s"][new_slice] = source_age_s
            self._buffer.data["observation_valid"][new_slice] &= source_valid
            self._buffer.data["tactile_fresh"][new_slice] &= source_valid
            self._buffer.data["flag_camera_fresh"][new_slice] &= source_valid
            # Synthetic gap/hold slots inherit the last source's effective
            # target but must not claim a send event: clear the action-queue
            # flag and zero action identity/timing on non-source slots so
            # replay does not republish commands that were never sent.
            hold_slots = ~source_valid
            self._buffer.data["flag_action_queued"][new_slice] &= source_valid
            for name in (
                "action_id",
                "action_created_monotonic_ns",
                "action_target_monotonic_ns",
                "action_valid_until_monotonic_ns",
            ):
                self._buffer.data[name][new_slice][hold_slots] = 0

        # ── Periodic non-camera flush: write buffered streams to HDF5 ──
        if self._buffer.size - self._flushed_frames >= self._flush_interval:
            self._ensure_hdf5()
            self._flush_buffered()

        # ── Camera streams → bounded background writer ──
        # A complete RGB-D payload is submitted for every grid slot, including
        # stale RGB/depth.  This keeps every sidecar exactly aligned with
        # data.h5 without retaining image arrays in EpisodeRecorder memory.
        if k > 0:
            current_payload = self._camera_payload(camera_frame)
            zero_payload = self._camera_payload(None)
            writer = self._camera_writer
            if writer is None:
                logger.error("EpisodeRecorder: camera writer missing during add_frame")
                return False
            sample_valid_slots = self._buffer.data["flag_sample_valid"][prev_size : self._buffer.size]
            for sample_valid_slot in sample_valid_slots:
                if sample_valid_slot:
                    payload = current_payload
                    self._last_camera_payload = (
                        np.array(current_payload[0], copy=True),
                        np.array(current_payload[1], copy=True),
                    )
                else:
                    payload = self._last_camera_payload or zero_payload
                rgb, depth = payload
                if not writer.submit(rgb, depth):
                    return False

        if add_result.capacity_reached:
            self._max_frames_reached = True
            logger.info("Episode reached max_frames=%d after aligned camera submission", self.max_frames)
            return False
        return True

    def _camera_payload(self, camera_frame: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
        """Return shape-stable camera arrays, using explicit zero placeholders."""
        cfg = self._camera_writer_config
        if camera_frame is None:
            return (
                np.zeros(cfg.rgb_shape, dtype=np.uint8),
                np.zeros(cfg.depth_shape, dtype=np.uint16),
            )

        rgb = camera_frame.get("rgb")
        depth = camera_frame.get("depth")
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

        # Validate the complete in-memory schema before creating or extending
        # any HDF5 dataset. Timestamp is stored outside ``buf_data`` but belongs
        # to the same 96/97-field contract.
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
            raise RuntimeError("schema-v16 recorder buffer mismatch: " + "; ".join(layout_errors))

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

        # Capture the truncation flag before the reset below — drives /meta truncated.
        truncated = self._max_frames_reached

        # Mark as stopped *before* spawning the thread — add_frame() must
        # reject new frames immediately.  All data is already in _buffer
        # and the camera writer queue; the stop daemon will flush both.
        self._recording = False
        self._max_frames_reached = False
        path = self._episode_dir

        # Snapshot stop metadata BEFORE spawning the worker (it overwrites
        # self._frame_count during _stop_episode_impl_inner).
        self._stop_success = success
        self._stop_path = path
        self._stop_frame_count = self._frame_count

        t = threading.Thread(
            target=self._stop_episode_impl,
            args=(success, reason, truncated),
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
            # A prior call may already have joined a crashed stop thread.
            return self._stop_error is None
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
        """Background: finalize sidecars, flush buffers, write metadata, and publish.

        ENOSPC / OSError at any h5py call site is captured into ``_stop_error``
        so ``join_stop()`` and RecorderIO report failure instead of publishing
        or announcing a truncated episode.
        """
        try:
            self._stop_episode_impl_inner(success, reason, truncated)
        except Exception as exc:
            self._stop_error = f"{type(exc).__name__}: {exc}"
            logger.error("stop_episode failed: %s — HDF5 may be truncated", self._stop_error)
            try:
                self._write_aborted_manifest(reason=reason, error=self._stop_error)
            except Exception:
                logger.error("failed to publish aborted episode manifest", exc_info=True)
            # Best-effort close: metadata may flush partial B-tree updates.
            try:
                if self._camera_writer is not None:
                    self._camera_writer.close(timeout=5.0)
            except Exception:
                logger.warning("camera writer cleanup failed after episode stop error", exc_info=True)
            self._camera_writer = None
            try:
                if self._file is not None:
                    self._file.close()
            except Exception:
                logger.warning("HDF5 cleanup failed after episode stop error", exc_info=True)
            self._file = None
        finally:
            # Always clean up temp directory and reset state, even if the
            # except handler above also failed (e.g. file close() raised).
            _tmp = self._temp_dir
            if _tmp is not None:
                self._discard_temp_files(_tmp)
            self._reset_episode_state()

    def _stop_episode_impl_inner(self, success: bool, reason: str, truncated: bool) -> None:
        """Inner body of _stop_episode_impl — extracted so the try/except wrapper
        can reset state on any exception without duplicating the reset list."""
        duration = time.perf_counter() - (self._start_time or 0.0)

        # ── Drain/finalize camera writer before publishing the episode ──
        writer = self._camera_writer
        if writer is None:
            raise RuntimeError("camera writer missing at episode stop")
        writer.close(timeout=_CAMERA_WRITER_CLOSE_TIMEOUT_S)
        camera_frame_count = writer.frame_count
        self._camera_writer_metrics = writer.metrics
        self._camera_writer = None

        # ── Flush remaining buffered non-camera streams ──
        self._flush_buffered()
        buf_size = self._buffer.size if self._buffer is not None else 0
        self._ensure_hdf5()

        if camera_frame_count != buf_size:
            raise RuntimeError(f"camera/control grid length mismatch: camera={camera_frame_count}, control={buf_size}")

        self._buffer = None
        self._frame_count = buf_size

        _had_rgb = camera_frame_count > 0

        # ── Write final metadata ──
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
            meta.attrs["wall_fps"] = self._frame_count / duration if duration > 0 else self.control_hz
            meta.attrs["min_frames_met"] = self._frame_count >= self.min_frames
            meta.attrs["has_camera"] = _had_rgb
            meta.attrs["has_timestamps"] = "timestamp" in self._datasets
            meta.attrs["camera_stream_frames"] = camera_frame_count
            meta.attrs["camera_writer_error"] = ""
            for metric_name, metric_value in self._camera_writer_metrics.items():
                meta.attrs[metric_name] = metric_value
            meta.attrs["truncated"] = bool(truncated)
            meta.attrs["stop_reason"] = reason or ("max_frames" if truncated else "manual")
            # Camera meta backfill: the initial lazy write may have run
            # before the camera child finished connect — re-write so late
            # values land in the file.
            self._write_camera_meta_attrs(meta)

        if self._file is not None:
            self._file.close()
        self._file = None
        # ── Atomic finalise ──
        # success: rename temp dir → final dir.
        # discard:  remove temp dir.
        _final = self._episode_dir
        _tmp = self._temp_dir
        if _tmp is not None and _final is not None:
            if success:
                self._validate_and_sync_temp_episode(Path(_tmp), self._frame_count)
                atomic_publish(_tmp, _final)
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
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=self.data_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
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
                raise RuntimeError("data.h5 schema-v17 layout mismatch: " + "; ".join(layout_errors))
        for key in ("depth",):
            with h5py.File(paths[key], "r") as sidecar:
                if key not in sidecar or int(sidecar[key].shape[0]) != expected_frames:
                    raise RuntimeError(f"{key} sidecar length mismatch")
        with VideoDecoder(paths["rgb"]) as decoder:
            decoded_frames = decoder.count_decoded_frames()
            if decoded_frames != expected_frames:
                raise RuntimeError(f"RGB decoded frame count {decoded_frames} != {expected_frames}")
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
