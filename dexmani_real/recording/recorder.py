"""Nonblocking control-row submission and one streaming writer per Raw episode."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import threading
import time

import h5py
import numpy as np

from dexmani_real.config.hardware import CameraParams
from dexmani_real.recording.storage.schema import DATASET_SPECS, RAW_FORMAT
from dexmani_real.recording.storage.video import VideoEncoder
from dexmani_real.robot.commands import DispatchStatus
from dexmani_real.sensor.camera.geometry import RGBDGeometry, validate_aligned_depth_distortion
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.geometry import validate_rigid_transform
from dexmani_real.utils.control_clock import sampling_clock_ns

logger = logging.getLogger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0
_COLLECTION_SOURCES = frozenset({"teleop", "policy_rollout"})


class RecordingError(RuntimeError):
    """Required recording failed; the control owner must revoke motion."""


@dataclass(frozen=True)
class _Finish:
    save: bool
    reason: str
    details: object


def snapshot_recording_metadata(shared, *, collection_source, camera_calibration):
    """Select the connected serial from the session snapshot, without file I/O."""
    serial = shared.sensors.camera_serial.value.decode("utf-8")
    if not serial.strip():
        raise ValueError("recording START requires a nonempty camera serial")
    geometry = RGBDGeometry.from_dict(json.loads(shared.sensors.camera_geometry.value.decode("utf-8")))
    transform = None
    if camera_calibration is None:
        logger.warning("Recording without camera extrinsics: session has no calibration")
    else:
        try:
            name = camera_calibration.resolve_name_by_serial(serial)
            if (
                camera_calibration.to_meta_dict(name, expected_serial=serial)["camera_type"]
                == "eye_to_hand"
            ):
                transform = camera_calibration.get_extrinsics(name)
            else:
                logger.warning("Recording without static eye-to-hand extrinsics")
        except (KeyError, ValueError) as exc:
            logger.warning("Recording without camera extrinsics: %s", exc)
    return dict(
        collection_source=collection_source,
        camera_geometry=geometry,
        camera_T_xarm_base_from_color=transform,
        depth_scale=float(shared.sensors.camera_depth_scale.value),
    )


def _raw_row(row, command, result, timestamp):
    arm, hand, camera = (
        row.arm[0],
        None if row.hand is None else row.hand[0],
        row.camera,
    )
    if camera is None:
        raise ValueError("recording requires RGB-D from the current observation")
    contact_valid = hand is not None and bool(hand["tactile_aggregate_valid"])
    dense_valid = hand is not None and bool(hand["tactile_dense_valid"])
    values = {
        "timestamp": timestamp,
        "dispatch_status": (int(result.arm), int(result.hand)) if result is not None else (0, 0),
        "arm_qpos": arm["qpos"],
        "arm_qvel": arm["qvel"],
        "arm_effort": arm["effort"],
        "hand_qpos": hand["qpos"] if hand is not None else np.full(12, np.nan),
        "hand_current": hand["current"] if hand is not None else np.full(12, np.nan),
        "hand_contact": hand["tactile_aggregate"] if contact_valid else np.full((5, 3), np.nan),
        "hand_tactile_force": hand["tactile_dense"]
        if dense_valid
        else np.full((5, 120, 3), np.nan),
        "action_arm_joint_target": (
            command.arm_qpos
            if command is not None
            and command.arm_qpos is not None
            and result is not None
            and result.arm != DispatchStatus.NOT_CALLED
            else np.full(7, np.nan)
        ),
        "action_hand_joint_target": (
            command.hand_qpos
            if command is not None
            and command.hand_qpos is not None
            and result is not None
            and result.hand != DispatchStatus.NOT_CALLED
            else np.full(12, np.nan)
        ),
    }
    # Runtime flags may fail independently of joint telemetry. Never label a
    # nonfinite payload usable, or serialize an invalid payload as finite zeros.
    for name, valid in (("hand_contact", contact_valid), ("hand_tactile_force", dense_valid)):
        if valid and not np.isfinite(values[name]).all():
            raise ValueError(f"runtime tactile validity disagrees with {name} payload")
    if any(status not in range(5) for status in values["dispatch_status"]):
        raise ValueError("dispatch_status must contain status codes 0 through 4")
    data = {}
    for name, value in values.items():
        spec = DATASET_SPECS[name]
        array = np.asarray(value, dtype=spec.dtype)
        if array.shape != spec.tail_shape:
            raise ValueError(f"Raw {name}: expected {spec.tail_shape}, got {array.shape}")
        data[name] = array
    return data


class AsyncEpisodeRecorder:
    """Owned immutable rows cross one FIFO; all encoding and file I/O stay in its writer.

    START and finalization may wait. Callers must revoke motion before those
    boundaries, and must stop required recording after check_error() fails.
    """

    def __init__(self, data_dir, control_hz=16.0, rgb_shape=None, video_config=None):
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("recording control_hz must be finite and positive")
        self.data_dir = Path(data_dir)
        self.control_hz = float(control_hz)
        self._rgb_shape = tuple(rgb_shape or CameraParams().rgb_shape)
        if len(self._rgb_shape) != 3 or self._rgb_shape[2] != 3 or min(self._rgb_shape) <= 0:
            raise ValueError("recording rgb_shape must be positive HWC with 3 channels")
        self._video_config = video_config
        self._thread = None
        self._queue = Queue(maxsize=16)
        self._ready = threading.Event()
        self._cancelled = threading.Event()
        self._error = None
        self._recording = False
        self._temp_dir = self._episode_dir = self._published_path = None
        self._submitted_frames = self._written_frames = 0

    @property
    def is_recording(self):
        return self._recording

    @property
    def accepting_frames(self):
        return self._recording and self._error is None

    @property
    def staging_path(self):
        return self._temp_dir

    @property
    def written_frames(self):
        return self._written_frames

    def _store_error(self, error):
        if self._error is None:
            self._error = error

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
        return {
            "task_label": task_label,
            "collection_source": collection_source,
            **(dict(
                sampling_period_ns=sampling_clock_ns(self.control_hz)[0],
                sampling_tolerance_ns=sampling_clock_ns(self.control_hz)[1],
            ) if collection_source == "teleop" else {}),
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
                else {}
            ),
            "depth_scale": depth_scale_value,
        }

    def start_episode(
        self, *, task_label, collection_source, camera_geometry,
        camera_T_xarm_base_from_color, depth_scale, episode_name=None,
    ):
        self.check_error()
        if self._recording or (self._thread is not None and self._thread.is_alive()):
            raise RecordingError("previous recording still owns resources")
        self._temp_dir = self._episode_dir = self._published_path = None
        self._submitted_frames = self._written_frames = 0
        try:
            metadata = self._snapshot_start_metadata(
                task_label=task_label, collection_source=collection_source,
                camera_geometry=camera_geometry,
                camera_T_xarm_base_from_color=camera_T_xarm_base_from_color,
                depth_scale=depth_scale,
            )
            if episode_name is not None and (
                type(episode_name) is not str or not episode_name
                or episode_name.startswith(".") or "/" in episode_name or "\\" in episode_name
            ):
                raise ValueError("episode_name must be a plain non-empty directory name")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            name = episode_name or f"episode_{time.strftime('%Y%m%d_%H%M%S')}"
            suffix = 0
            while True:
                candidate = name if suffix == 0 else f"{name}_{suffix}"
                destination, staging = self.data_dir / candidate, self.data_dir / f".tmp_{candidate}"
                if not target_is_occupied(destination) and not target_is_occupied(staging):
                    break
                if episode_name is not None:
                    raise FileExistsError(f"explicit episode name already exists: {destination}")
                suffix += 1
            staging.mkdir(exist_ok=False)
            self._temp_dir, self._episode_dir = staging, destination
            self._queue = Queue(maxsize=16)
            self._ready.clear()
            self._cancelled.clear()
            self._thread = threading.Thread(
                target=self._write_episode, args=(metadata,), name="episode-writer", daemon=False
            )
            self._thread.start()
            if not self._ready.wait(RECORDER_START_TIMEOUT_S):
                raise TimeoutError("recording START timed out")
            self.check_error()
            self._recording = True
        except BaseException as exc:
            self._store_error(exc)
            self._cancelled.set()
            if self._thread is not None and self._thread.ident is not None:
                self._thread.join(timeout=RECORDER_STOP_TIMEOUT_S)
            if not isinstance(exc, Exception):
                raise
            raise RecordingError(f"recording START failed: {exc}") from exc

    def add_frame(self, row, command=None, result=None):
        """Submit already-owned immutable snapshots without conversion or I/O."""
        self.check_error()
        if not self._recording:
            return None
        try:
            self._queue.put_nowait((row, command, result))
        except Full as exc:
            error = RecordingError("recording queue full (capacity=16)")
            self._store_error(error)
            raise error from exc
        self._submitted_frames += 1

    def save_episode(self, reason="manual", *, details=None):
        return self._finish(save=True, reason=reason, details=details)

    def discard_episode(self, reason="discard"):
        self._finish(save=False, reason=reason)

    def close(self):
        """Save the active prefix after the caller has revoked motion."""
        self._finish(save=True, reason="close")

    def _finish(self, *, save, reason, details=None):
        was_recording, self._recording = self._recording, False
        thread = self._thread
        if thread is None:
            self.check_error()
            return None
        deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
        interrupted = None
        try:
            if was_recording and thread.is_alive() and not self._cancelled.is_set():
                try:
                    finish = _Finish(save, reason, deepcopy(details))
                except BaseException as exc:
                    self._store_error(exc)
                    finish = _Finish(False, reason, None)
                    if not isinstance(exc, Exception):
                        interrupted = exc
                self._queue.put(finish, timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as exc:
            self._store_error(exc)
            self._cancelled.set()
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            try:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException as exc:
                self._store_error(exc)
                self._cancelled.set()
                if not isinstance(exc, Exception):
                    interrupted = exc
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self._store_error(TimeoutError("recording finalization timed out; staging remains owned"))
                self._cancelled.set()
        if interrupted is not None:
            raise interrupted
        self.check_error()
        self._thread = None
        return self._published_path

    def _write_episode(self, metadata):
        staging, destination = self._temp_dir, self._episode_dir
        finish, rows = None, []
        first_stamp = previous_stamp = None
        try:
            with ExitStack() as resources:
                data = resources.enter_context(h5py.File(staging / "data.h5", "w"))
                meta = data.create_group("meta")
                meta.attrs.update(metadata)
                meta.attrs.update(control_hz=self.control_hz,
                                  timestamp_source="host_monotonic_relative",
                                  dispatch_status_source="recorded")
                datasets = {
                    name: data.create_dataset(name, shape=(0, *spec.tail_shape),
                                              maxshape=(None, *spec.tail_shape), dtype=spec.dtype)
                    for name, spec in DATASET_SPECS.items()
                }
                datasets["depth"] = data.create_dataset(
                    "depth", shape=(0, *self._rgb_shape[:2]), maxshape=(None, *self._rgb_shape[:2]),
                    chunks=(1, *self._rgb_shape[:2]), dtype=np.uint16, compression="gzip", compression_opts=1,
                )
                height, width, _ = self._rgb_shape
                video = VideoEncoder(staging / "rgb.mp4", config=self._video_config,
                                     fps=self.control_hz, width=width, height=height)
                resources.callback(video.close)
                video.open()
                self._ready.set()

                def flush():
                    if not rows:
                        return
                    arrays = {name: np.stack([row[name] for row in rows]) for name in datasets}
                    # Once an append begins, a failing batch is staging evidence;
                    # retrying it could duplicate a partially written batch.
                    rows.clear()
                    end = self._written_frames + len(arrays["timestamp"])
                    for name, values in arrays.items():
                        datasets[name].resize(end, axis=0)
                        datasets[name][self._written_frames:end] = values
                    self._written_frames = end

                try:
                    while not self._cancelled.is_set():
                        try:
                            item = self._queue.get(timeout=0.05)
                        except Empty:
                            continue
                        if isinstance(item, _Finish):
                            finish = item
                            break
                        row, command, result = item
                        stamp = int(row.observation_timestamp_ns)
                        if stamp <= 0 or (previous_stamp is not None and stamp <= previous_stamp):
                            raise ValueError("recording observation timestamps must be positive and increasing")
                        if first_stamp is None:
                            first_stamp = stamp
                        # Integer subtraction retains short intervals even for a large host clock.
                        values = _raw_row(row, command, result, (stamp - first_stamp) / 1e9)
                        rgb, depth = row.camera["rgb"], row.camera["depth"]
                        if rgb.shape != self._rgb_shape or rgb.dtype != np.uint8:
                            raise ValueError("RGB shape or dtype mismatch")
                        if depth.shape != self._rgb_shape[:2] or depth.dtype != np.uint16:
                            raise ValueError("depth shape or dtype mismatch")
                        video.write_frame(rgb)
                        rows.append(dict(values, depth=depth))
                        previous_stamp = stamp
                        if len(rows) >= 32:
                            flush()
                except BaseException as exc:
                    # Publish the failure before cleanup, which may block on native I/O.
                    self._store_error(exc)
                flush()
                reason = finish.reason if finish is not None else "recording_failure"
                details = list(finish.details or []) if finish is not None else []
                if self._error is not None:
                    details.append(dict(stage="recording", exception_type=type(self._error).__name__,
                                        message=str(self._error)))
                meta.attrs.update(format=RAW_FORMAT, num_frames=self._written_frames,
                                  termination_reason=reason)
                if details:
                    meta.create_dataset("termination_details", data=json.dumps(details, allow_nan=False))
                if self._error is None and (self._written_frames != self._submitted_frames
                                           or video.frame_count != self._written_frames):
                    raise RuntimeError("recording accepted/written/camera row count mismatch")
            if self._error is None and not self._cancelled.is_set():
                if finish is not None and finish.save and self._written_frames:
                    atomic_publish(staging, destination, cancelled=self._cancelled.is_set)
                    self._published_path = destination
                    logger.info("Episode saved: %s frames=%d reason=%s", destination,
                                self._written_frames, finish.reason)
                else:
                    shutil.rmtree(staging)
                self._temp_dir = None
        except BaseException as exc:
            self._store_error(exc)
            logger.exception("Recording failed; staging retained: %s", staging)
        finally:
            self._ready.set()
            if self._error is not None:
                logger.error("Recording staging retained: %s frames=%d", staging, self._written_frames)
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
