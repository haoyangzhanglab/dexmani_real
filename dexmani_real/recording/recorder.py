"""Non-blocking local recording; one writer thread owns each Raw episode."""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.config.hardware import CameraParams
from dexmani_real.recording.storage.hdf5_writer import EpisodeDataWriter
from dexmani_real.recording.storage.schema import DATASET_SPECS, RAW_FORMAT
from dexmani_real.recording.storage.video import VideoEncoder
from dexmani_real.sensor.camera.geometry import RGBDGeometry, validate_aligned_depth_distortion
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.geometry import validate_rigid_transform, validate_unit_quaternion_wxyz
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0
_COLLECTION_SOURCES = frozenset({"teleop", "policy_rollout"})
_STOP = object()


class RecordingError(RuntimeError):
    """Required recording failed; the control owner must revoke motion."""


class RecordingBackpressureError(RecordingError):
    """The bounded sink cannot sustain the experiment's control rate."""


def snapshot_recording_metadata(shared, runtime, *, collection_source):
    """Resolve physical calibration at START, never during frame submission."""
    serial = shared.camera_serial.value.rstrip(b"\x00").decode("utf-8")
    if not serial.strip():
        raise ValueError("recording START requires a nonempty camera serial")
    geometry = RGBDGeometry.from_dict(json.loads(shared.camera_geometry.value.decode("utf-8")))
    transform = None
    try:
        calibration = CameraExtrinsics()
        name = calibration.resolve_name_by_serial(serial)
        if calibration.to_meta_dict(name, expected_serial=serial)["camera_type"] == "eye_to_hand":
            transform = calibration.get_extrinsics(name)
        else:
            logger.warning("Recording without static eye-to-hand extrinsics")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning("Recording without camera extrinsics: %s", exc)
    return dict(
        collection_source=collection_source,
        camera_geometry=geometry,
        camera_T_xarm_base_from_color=transform,
        depth_scale=float(shared.camera_depth_scale.value),
        handbase_position_eef_m=runtime.hand.T_eef_handbase_pos_xyz,
        handbase_quat_eef_wxyz=runtime.hand.T_eef_handbase_quat_wxyz,
    )


def _validate_explicit_episode_name(episode_name: str) -> None:
    """Require a plain directory name that stays inside the recorder data dir."""
    if (
        type(episode_name) is not str
        or not episode_name
        or episode_name in {".", ".."}
        or "/" in episode_name
        or "\\" in episode_name
        or episode_name.startswith(".")
    ):
        raise ValueError(
            "episode_name must be a plain non-empty directory name without "
            f"path separators: {episode_name!r}"
        )


def _validate_hand_mount(
    handbase_position_eef_m: object,
    handbase_quat_eef_wxyz: object,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        position = np.asarray(handbase_position_eef_m, dtype=np.float64)
        quaternion = np.asarray(handbase_quat_eef_wxyz, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("hand mount must be numeric") from exc
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("handbase_position_eef_m must have shape (3,) and finite values")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("handbase_quat_eef_wxyz must have shape (4,) and finite values")
    validate_unit_quaternion_wxyz(quaternion, name="handbase_quat_eef_wxyz")
    return position.copy(), quaternion.copy()


class AsyncEpisodeRecorder:
    """One control-thread producer and one FIFO writer, with no recording IPC.

    Construction is resource-free. START and finalization may block, so callers
    must keep motion revoked throughout those boundaries. Frames transfer owned,
    immutable camera references; the producer must not mutate them after submit.
    """

    def __init__(
        self,
        data_dir,
        control_hz=16.0,
        rgb_shape=None,
        video_config=None,
        execution_path="synchronous_direct_sdk_v1",
    ):
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("recording control_hz must be finite and positive")
        self.data_dir = Path(data_dir)
        self.control_hz = float(control_hz)
        self._rgb_shape = tuple(rgb_shape or CameraParams().rgb_shape)
        if len(self._rgb_shape) != 3 or self._rgb_shape[2] != 3 or min(self._rgb_shape) <= 0:
            raise ValueError("recording rgb_shape must be positive HWC with 3 channels")
        self._video_config = video_config
        self.execution_path = execution_path
        self.policy_trace = None
        self._thread = None
        self._queue = Queue(maxsize=16)
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._abort = False
        self._recording = False
        self._save = False
        self._reason = ""
        self._temp_dir = None
        self._episode_dir = None
        self._handles_released = True
        self._frame_count = 0
        self._written_frames = 0
        self._saved = False

    @property
    def is_recording(self):
        return self._recording

    @property
    def frame_count(self):
        """Number of accepted rows, retained until the next START."""
        return self._frame_count

    @property
    def episode_path(self):
        return self._episode_dir

    @property
    def resources_released(self):
        return (self._thread is None or not self._thread.is_alive()) and self._handles_released

    @property
    def accepting_frames(self):
        return self._recording and self._error is None

    def _store_error(self, error):
        if self._error is None:
            self._error = error
        self._abort = True

    def check_error(self):
        if self._error is not None:
            raise RecordingError(f"Recording failed: {self._error}") from self._error

    def _snapshot_start_metadata(
        self,
        *,
        task_label: str,
        collection_source: str,
        camera_geometry: RGBDGeometry,
        camera_T_xarm_base_from_color: object,
        depth_scale: float,
        handbase_position_eef_m: object,
        handbase_quat_eef_wxyz: object,
    ) -> dict[str, object]:
        if not isinstance(task_label, str) or not task_label.strip():
            raise ValueError("task_label must be a non-empty string")
        if not isinstance(collection_source, str) or collection_source not in _COLLECTION_SOURCES:
            raise ValueError(
                f"collection_source must be one of {sorted(_COLLECTION_SOURCES)}, "
                f"got {collection_source!r}"
            )
        if not isinstance(camera_geometry, RGBDGeometry):
            raise TypeError("camera_geometry must be an RGBDGeometry instance")
        color = camera_geometry.color
        validate_aligned_depth_distortion(color.distortion_model)
        expected_rgb_shape = (color.height, color.width, 3)
        if self._rgb_shape != expected_rgb_shape:
            raise ValueError(
                "aligned RGB-D geometry must match recording shape: "
                f"expected rgb={self._rgb_shape}, "
                f"depth={self._rgb_shape[:2]}; "
                f"got rgb={expected_rgb_shape}, depth={expected_rgb_shape[:2]}"
            )
        try:
            depth_scale_value = float(depth_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("depth_scale must be numeric") from exc
        if not np.isfinite(depth_scale_value) or depth_scale_value <= 0.0:
            raise ValueError("depth_scale must be finite and positive")
        transform = (
            None
            if camera_T_xarm_base_from_color is None
            else validate_rigid_transform(
                camera_T_xarm_base_from_color, label="camera_T_xarm_base_from_color"
            )
        )
        handbase_position, handbase_quaternion = _validate_hand_mount(
            handbase_position_eef_m,
            handbase_quat_eef_wxyz,
        )
        return {
            "task_label": task_label,
            "collection_source": collection_source,
            "camera_payload_mode": "depth_to_color_aligned_rgbd",
            "camera_color_width": color.width,
            "camera_color_height": color.height,
            "camera_color_intrinsics": color.matrix().reshape(-1).copy(),
            "camera_color_distortion_model": color.distortion_model,
            "camera_color_distortion_coeffs": np.asarray(
                color.distortion_coeffs, dtype=np.float64
            ).copy(),
            **(
                {"camera_T_xarm_base_from_color": transform.reshape(-1).copy()}
                if transform is not None
                else {"camera_extrinsics_missing": True}
            ),
            "depth_scale": depth_scale_value,
            "handbase_position_eef_m": handbase_position,
            "handbase_quat_eef_wxyz": handbase_quaternion,
        }

    def start_episode(
        self,
        *,
        task_label,
        collection_source,
        camera_geometry,
        camera_T_xarm_base_from_color,
        depth_scale,
        handbase_position_eef_m,
        handbase_quat_eef_wxyz,
        episode_name=None,
    ):
        self.check_error()
        if self._recording or not self.resources_released:
            raise RecordingError("previous recording still owns resources")
        try:
            metadata = self._snapshot_start_metadata(
                task_label=task_label,
                collection_source=collection_source,
                camera_geometry=camera_geometry,
                camera_T_xarm_base_from_color=camera_T_xarm_base_from_color,
                depth_scale=depth_scale,
                handbase_position_eef_m=handbase_position_eef_m,
                handbase_quat_eef_wxyz=handbase_quat_eef_wxyz,
            )
            if episode_name is not None:
                _validate_explicit_episode_name(episode_name)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            stale = [
                p.name
                for p in self.data_dir.glob(".tmp_[!.]*")
                if p.is_dir() and not p.is_symlink()
            ]
            if stale:
                logger.warning(
                    "Unpublished recording staging remains (inspect manually): %s",
                    sorted(stale)[:5],
                )
            name = episode_name or f"episode_{time.strftime('%Y%m%d_%H%M%S')}"
            suffix = 0
            while True:
                candidate = name if suffix == 0 else f"{name}_{suffix}"
                destination = self.data_dir / candidate
                staging = self.data_dir / f".tmp_{candidate}"
                if not target_is_occupied(destination) and not target_is_occupied(staging):
                    break
                if episode_name is not None:
                    raise FileExistsError(f"explicit episode name already exists: {destination}")
                suffix += 1
            staging.mkdir(exist_ok=False)
            self._temp_dir, self._episode_dir = staging, destination
            self._queue = Queue(maxsize=16)
            self._ready.clear()
            self._abort = self._save = self._saved = False
            self._reason = ""
            self.policy_trace = None
            self._frame_count = self._written_frames = 0
            self._thread = threading.Thread(
                target=self._write_episode, args=(metadata,), name="episode-writer", daemon=False
            )
            self._thread.start()
            if not self._ready.wait(RECORDER_START_TIMEOUT_S):
                raise TimeoutError("recording START timed out")
            self.check_error()
            self._recording = True
            return True
        except BaseException as exc:
            self._store_error(exc)
            # A thread stuck in native I/O retains its staging and handles. Never
            # release them from another thread or accept another episode.
            try:
                if self._thread is not None and self._thread.ident is not None:
                    self._thread.join(timeout=RECORDER_STOP_TIMEOUT_S)
                else:
                    self._thread = None
            except Exception:
                logger.exception("Recording START cleanup also failed")
            if not isinstance(exc, Exception):
                raise
            raise RecordingError(f"recording START failed: {exc}") from exc

    def add_frame(self, frame):
        """Submit owned references to the bounded writer queue."""
        self.check_error()
        if not self.accepting_frames:
            return
        try:
            self._queue.put_nowait(frame)
        except Full as exc:
            error = RecordingBackpressureError("recording queue full (capacity=16)")
            self._store_error(error)
            raise error from exc
        self._frame_count += 1

    def save_episode(self, reason="manual"):
        return self._finish(save=True, reason=reason)

    def discard_episode(self, reason="discard"):
        self._finish(save=False, reason=reason)

    def close(self):
        """Save an unfinished prefix; callers must already have revoked motion."""
        self._finish(save=True, reason="close")

    def _finish(self, *, save, reason):
        self._recording = False
        thread = self._thread
        if thread is None:
            self.check_error()
            return None
        self._save = save and self._error is None
        self._reason = reason
        deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
        if thread.is_alive() and not self._abort:
            while thread.is_alive():
                try:
                    self._queue.put_nowait(_STOP)
                    break
                except Full:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    thread.join(timeout=min(0.01, remaining))
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            self._store_error(
                TimeoutError("recording finalization timed out; staging remains owned")
            )
        self.check_error()
        if not self.resources_released:
            raise RecordingError("recording retained unreleased resources")
        self._thread = None
        return self._episode_dir if self._saved else None

    def _write_episode(self, metadata):
        video = data = None
        rows = []
        missing_tactile = {"hand_contact": 0, "hand_tactile_force": 0}
        last_selected = last_dispatch_detail = None
        staging = self._temp_dir
        self._handles_released = False

        def initial_meta(meta):
            for name, value in metadata.items():
                meta.attrs[name] = value
            meta.attrs["control_hz"] = self.control_hz
            meta.attrs["execution_path"] = self.execution_path
            meta.attrs["observation_action_pairing"] = "control_tick_input_and_attempted_targets"
            meta.attrs["robot_timestamp_source"] = "host_monotonic_read_completion"
            meta.attrs["camera_timestamp_source"] = "host_monotonic_camera_queue_return"
            meta.attrs["time_missing_value"] = 0
            meta.attrs["dispatch_status_codes"] = (
                "0:not_called,1:accepted,2:crc_unconfirmed,3:rejected,4:unknown"
            )

        def flush_rows():
            if rows:
                data.append(
                    {
                        name: np.asarray([row[name] for row in rows], dtype=spec.dtype)
                        for name, spec in DATASET_SPECS.items()
                    }
                )
                rows.clear()

        try:
            # Assign each resource before opening it, so partial START failures
            # retain an owner that can close it on this same thread.
            data = EpisodeDataWriter(
                staging / "data.h5",
                write_initial_meta=initial_meta,
                depth_shape=self._rgb_shape[:2],
            )
            data.open()
            height, width, _ = self._rgb_shape
            video = VideoEncoder(
                staging / "rgb.mp4",
                config=self._video_config,
                fps=self.control_hz,
                width=width,
                height=height,
            )
            video.open()
            self._ready.set()
            while not self._abort:
                try:
                    frame = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                if frame is _STOP:
                    break
                if set(frame.data) != DATASET_SPECS.keys():
                    raise ValueError("episode frame fields do not match Raw")
                if frame.camera_rgb is None or frame.camera_depth is None:
                    raise ValueError("recording requires RGB-D for every row")
                if frame.camera_rgb.shape != self._rgb_shape or frame.camera_rgb.dtype != np.uint8:
                    raise ValueError("RGB shape or dtype mismatch")
                for name in missing_tactile:
                    missing_tactile[name] += int(not np.isfinite(frame.data[name]).all())
                last_selected = frame.selected
                last_dispatch_detail = frame.dispatch_detail
                video.write_frame(frame.camera_rgb)
                data.append_depth(frame.camera_depth)
                rows.append(frame.data)
                self._written_frames += 1
                if len(rows) >= 32:
                    flush_rows()
            if not self._abort:
                flush_rows()

                def final_meta(meta):
                    meta.attrs["format"] = RAW_FORMAT
                    meta.attrs["num_frames"] = self._written_frames
                    meta.attrs["termination_reason"] = self._reason
                    if self.policy_trace is not None:
                        meta.create_dataset(
                            "policy_trace", data=json.dumps(self.policy_trace, allow_nan=False)
                        )
                    for name, count in missing_tactile.items():
                        meta.attrs[f"{name}_missing_rows"] = count
                        if count:
                            logger.warning(
                                "Recording %s missing: %d/%d rows",
                                name,
                                count,
                                self._written_frames,
                            )
                    if last_dispatch_detail is not None:
                        meta.attrs["final_dispatch_result"] = json.dumps(
                            last_dispatch_detail, allow_nan=False
                        )
                    if last_selected is not None:
                        meta.attrs["final_selected_targets"] = json.dumps(
                            last_selected, allow_nan=False
                        )

                data.update_meta(final_meta)
            video.close()
            data.close()
            self._handles_released = True
            if self._save and not self._abort and self._written_frames:
                if (
                    self._written_frames != self._frame_count
                    or video.frame_count != self._written_frames
                    or data.depth_frames != self._written_frames
                ):
                    raise RuntimeError("recording accepted/written/camera row count mismatch")
                atomic_publish(staging, self._episode_dir, cancelled=lambda: self._abort)
                self._saved = True
                logger.info(
                    "Episode saved: %s frames=%d reason=%s",
                    self._episode_dir,
                    self._written_frames,
                    self._reason,
                )
            else:
                logger.info(
                    "Episode discarded: %s frames=%d reason=%s",
                    self._episode_dir,
                    self._written_frames,
                    self._reason or "recording_failure",
                )
        except BaseException as exc:
            self._store_error(exc)
            logger.exception("Recording failed; staging retained: %s", staging)
        finally:
            # Report failure before cleanup, which may itself block on storage.
            self._ready.set()
            released = True
            for resource in (video, data):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as exc:
                        released = False
                        self._store_error(exc)
                        logger.error("Recording resource close failed", exc_info=True)
            self._handles_released = released
            if released and self._error is None:
                try:
                    if staging.exists():
                        shutil.rmtree(staging)
                    self._temp_dir = None
                except BaseException as exc:
                    self._store_error(exc)
                    logger.error("Recording staging cleanup failed: %s", staging, exc_info=True)
            # Release references to unconsumed images on failure without touching SHM.
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
