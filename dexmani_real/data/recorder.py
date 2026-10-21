"""Nonblocking control-row submission and one streaming writer per Raw episode."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

from dexmani_real.data.raw.schema import validate_execution_boundary
from dexmani_real.data.raw.writer import RawEpisodeWriter
from dexmani_real.sensor.camera.geometry import CameraIntrinsics
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0
COLLECTION_SOURCES = frozenset({"teleop", "policy_rollout"})


class RecordingError(RuntimeError):
    """Required recording failed; the control owner must revoke motion."""


@dataclass(frozen=True)
class CaptureFinishRequest:
    save: bool
    reason: str
    details: object
    timing: object = None
    arm_execution: object = None


@dataclass(frozen=True)
class QueryTimingRecord:
    values: dict


def snapshot_recording_metadata(sensors, *, camera_calibration, hand):
    """Build physical HDF5 attributes from the active camera and consumed tactile semantics."""
    serial = sensors.camera_serial.value.decode("utf-8")
    if not serial.strip():
        raise ValueError("recording START requires a nonempty camera serial")
    color = CameraIntrinsics(**json.loads(sensors.camera_intrinsics.value.decode("utf-8")))
    transform = None
    if camera_calibration is None:
        logger.warning("Recording without camera extrinsics: session has no calibration")
    else:
        try:
            transform = camera_calibration.for_serial(serial)
        except KeyError as exc:
            logger.warning("Recording without camera extrinsics: %s", exc)
    depth_scale = float(sensors.camera_depth_scale.value)
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    metadata = {
        "camera_payload_mode": "depth_to_color_aligned_rgbd",
        "camera_color_width": color.width,
        "camera_color_height": color.height,
        "camera_color_intrinsics": color.matrix().reshape(-1),
        "camera_color_distortion_model": color.distortion_model,
        "camera_color_distortion_coeffs": np.asarray(color.distortion_coeffs, dtype=np.float64),
        "depth_scale": depth_scale,
    }
    if transform is not None:
        metadata["camera_T_xarm_base_from_color"] = transform.reshape(-1)
    if hand is not None:
        metadata.update(hand.get_tactile_reset_metadata())
        metadata.update(
            hand_tactile_processing="sdk_native_bounded_previous_valid_vector_hold",
            hand_tactile_hold_max_age_s=float(hand.cfg.feedback_max_age_s),
        )
    return metadata


class AsyncEpisodeRecorder:
    """One FIFO writer owns snapshots and I/O; revoke motion before blocking finalization."""

    def __init__(self, data_dir, *, control_hz):
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("recording control_hz must be finite and positive")
        self.data_dir = Path(data_dir)
        self.control_hz = float(control_hz)
        self.thread = None
        self.queue = Queue(maxsize=16)
        self.ready = threading.Event()
        self.cancelled = threading.Event()
        self.error = None
        self.recording = False
        self.staging_dir = self.episode_dir = self.published_path = None
        self.submitted_frames = 0
        self.writer = None

    @property
    def is_recording(self):
        return self.recording

    @property
    def staging_path(self):
        return self.staging_dir

    @property
    def written_frames(self):
        return self.writer.frame_count if self.writer is not None else 0

    def store_error(self, error):
        if self.error is None:
            self.error = error

    def check_error(self):
        if self.error is not None:
            raise RecordingError(f"Recording failed: {self.error}") from self.error

    def start_episode(
        self,
        *,
        task_label,
        collection_source,
        metadata,
        arm_execution,
        arm_control_metadata,
        episode_name=None,
    ):
        self.check_error()
        if self.recording or (self.thread is not None and self.thread.is_alive()):
            raise RecordingError("previous recording still owns resources")
        self.staging_dir = self.episode_dir = self.published_path = None
        self.submitted_frames = 0
        self.writer = None
        try:
            arm_execution, arm_control_metadata = deepcopy(arm_execution), deepcopy(
                arm_control_metadata
            )
            validate_execution_boundary(arm_execution)
            if not isinstance(task_label, str) or not task_label.strip():
                raise ValueError("task_label must be a non-empty string")
            if collection_source not in COLLECTION_SOURCES:
                raise ValueError(f"collection_source must be one of {sorted(COLLECTION_SOURCES)}")
            metadata = dict(
                deepcopy(metadata), task_label=task_label, collection_source=collection_source
            )
            if collection_source == "teleop":
                period, tolerance = sampling_clock_ns(self.control_hz)
                metadata.update(sampling_period_ns=period, sampling_tolerance_ns=tolerance)
            if episode_name is not None and (
                type(episode_name) is not str
                or not episode_name
                or episode_name.startswith(".")
                or "/" in episode_name
                or "\\" in episode_name
            ):
                raise ValueError("episode_name must be a plain non-empty directory name")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            name = episode_name or f"episode_{time.strftime('%Y%m%d_%H%M%S')}"
            suffix = 0
            while True:
                candidate = name if suffix == 0 else f"{name}_{suffix}"
                destination, staging = (
                    self.data_dir / candidate,
                    self.data_dir / f".tmp_{candidate}",
                )
                if not target_is_occupied(destination) and not target_is_occupied(staging):
                    break
                if episode_name is not None:
                    raise FileExistsError(f"explicit episode name already exists: {destination}")
                suffix += 1
            staging.mkdir(exist_ok=False)
            self.staging_dir, self.episode_dir = staging, destination
            self.queue = Queue(maxsize=16)
            self.ready.clear()
            self.cancelled.clear()
            self.thread = threading.Thread(
                target=self.write_episode,
                args=(metadata, arm_execution, arm_control_metadata),
                name="episode-writer",
                daemon=False,
            )
            self.thread.start()
            if not self.ready.wait(RECORDER_START_TIMEOUT_S):
                raise TimeoutError("recording START timed out")
            self.check_error()
            self.recording = True
        except BaseException as exc:
            self.store_error(exc)
            self.cancelled.set()
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=RECORDER_STOP_TIMEOUT_S)
            if not isinstance(exc, Exception):
                raise
            raise RecordingError(f"recording START failed: {exc}") from exc

    def enqueue(self, item):
        self.check_error()
        if not self.recording:
            return False
        try:
            self.queue.put_nowait(item)
        except Full as exc:
            error = RecordingError("recording queue full (capacity=16)")
            self.store_error(error)
            raise error from exc
        return True

    def add_frame(self, row, command=None, result=None, *, timing=None):
        """Submit already-owned immutable snapshots without conversion or I/O."""
        if self.enqueue((row, command, result, dict(timing or {}))):
            self.submitted_frames += 1

    def record_query(self, values):
        """Queue a top-level snapshot; callers must leave nested observations unchanged."""
        self.enqueue(QueryTimingRecord(dict(values)))

    def save_episode(self, reason="manual", *, arm_execution, details=None, timing=None):
        return self.finish(
            save=True, reason=reason, details=details, timing=timing, arm_execution=arm_execution
        )

    def discard_episode(self, reason="discard"):
        self.finish(save=False, reason=reason)

    def close(self):
        """Join the writer; an active capture without an end snapshot retains staging."""
        self.finish(save=True, reason="close")

    def finish(self, *, save, reason, details=None, timing=None, arm_execution=None):
        was_recording, self.recording = self.recording, False
        thread = self.thread
        if thread is None:
            self.check_error()
            return None
        deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
        interrupted = None
        try:
            if was_recording and thread.is_alive() and not self.cancelled.is_set():
                try:
                    finish = CaptureFinishRequest(
                        save, reason, deepcopy(details), deepcopy(timing), deepcopy(arm_execution)
                    )
                except BaseException as exc:
                    self.store_error(exc)
                    finish = CaptureFinishRequest(False, reason, None)
                    if not isinstance(exc, Exception):
                        interrupted = exc
                self.queue.put(finish, timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as exc:
            self.store_error(exc)
            self.cancelled.set()
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            try:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException as exc:
                self.store_error(exc)
                self.cancelled.set()
                if not isinstance(exc, Exception):
                    interrupted = exc
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self.store_error(
                    TimeoutError("recording finalization timed out; staging remains owned")
                )
                self.cancelled.set()
        if interrupted is not None:
            raise interrupted
        self.check_error()
        self.thread = None
        return self.published_path

    def write_episode(self, metadata, arm_start, arm_control_metadata):
        staging, destination = self.staging_dir, self.episode_dir
        finish = None
        try:
            self.writer = writer = RawEpisodeWriter(staging, control_hz=self.control_hz)
            try:
                writer.open(metadata, arm_start, arm_control_metadata)
                self.ready.set()
                try:
                    while not self.cancelled.is_set():
                        try:
                            item = self.queue.get(timeout=0.05)
                        except Empty:
                            continue
                        if isinstance(item, CaptureFinishRequest):
                            finish = item
                            break
                        if isinstance(item, QueryTimingRecord):
                            writer.write_query(item.values)
                        else:
                            writer.append_frame(*item)
                except BaseException as exc:
                    # Notify control before flushing or closing potentially blocking native I/O.
                    self.store_error(exc)
                writer.flush()
                writer.write_execution_end(finish.arm_execution if finish is not None else None)
                if finish is not None and finish.save:
                    try:
                        validate_execution_boundary(finish.arm_execution, stopped=True)
                    except (KeyError, TypeError, ValueError) as exc:
                        self.store_error(exc)
                reason = finish.reason if finish is not None else "recording_failure"
                details = list(finish.details or []) if finish is not None else []
                if self.error is not None:
                    details.append(
                        dict(
                            stage="recording",
                            exception_type=type(self.error).__name__,
                            message=str(self.error),
                        )
                    )
                writer.write_termination(
                    reason,
                    details,
                    finish.timing if finish is not None else None,
                    expected_frames=self.submitted_frames if self.error is None else None,
                )
            finally:
                writer.close()
            if self.error is None and not self.cancelled.is_set() and finish is not None:
                if finish.save:
                    # Zero-row captures retain failure evidence until explicit discard.
                    atomic_publish(staging, destination, cancelled=self.cancelled.is_set)
                    self.published_path = destination
                    logger.info("Raw 已保存：%s（%d 帧）。", destination.name, self.written_frames)
                    logger.debug(
                        "Episode saved: %s frames=%d reason=%s",
                        destination,
                        self.written_frames,
                        finish.reason,
                    )
                else:
                    shutil.rmtree(staging)
                self.staging_dir = None
        except BaseException as exc:
            self.store_error(exc)
            logger.exception("Recording failed; staging retained: %s", staging)
        finally:
            self.ready.set()
            if self.error is not None:
                logger.error(
                    "Recording staging retained: %s frames=%d", staging, self.written_frames
                )
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
