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
from types import SimpleNamespace

import numpy as np

from dexmani_real.data.raw.schema import unwritten_dispatch_detail
from dexmani_real.data.raw.writer import RawEpisodeWriter
from dexmani_real.sensor.camera.geometry import CameraIntrinsics
from dexmani_real.utils.atomic_io import atomic_json_dump, atomic_publish, target_is_occupied

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
    stop_result: str = "unconfirmed"
    technical_status: str = "failed"


def frame_dispatch_detail(row, command, result):
    if result is None:
        return None
    return unwritten_dispatch_detail(
        row.observation_timestamp_ns, (result.arm, result.hand),
        command.arm_qpos if command is not None else None,
        command.hand_qpos if command is not None else None,
    )


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
        "camera_serial": serial,
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
            hand_tactile_units="sdk_native_unknown_physical_scale",
            hand_tactile_axes="sdk_sensor_axes_not_world_frame",
            hand_tactile_finger_order="thumb,index,middle,ring,pinky",
            hand_tactile_hold_max_age_s=float(hand.cfg.feedback_max_age_s),
        )
    return metadata


class AsyncEpisodeRecorder:
    """单次 START 的 FIFO writer；阻塞收尾前由调用方撤销运动授权。"""

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
        self.io_failed = False
        self.failed_dispatches = []
        self.is_recording = False
        self.start_attempted = False
        self.finish_request = None
        self.staging_path = self.episode_dir = self.published_path = None
        self.submitted_frames = 0
        self.writer = None

    @property
    def written_frames(self):
        return self.writer.frame_count if self.writer is not None else 0

    @property
    def writer_pending(self):
        return self.thread is not None and self.thread.is_alive()

    def store_error(self, error, *, io=False):
        if io:
            self.io_failed = True
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
        episode_name=None,
    ):
        if self.start_attempted or self.finish_request is not None:
            raise RecordingError("recorder permits only one START attempt")
        self.start_attempted = True
        try:
            if not isinstance(task_label, str) or not task_label.strip():
                raise ValueError("task_label must be a non-empty string")
            if collection_source not in COLLECTION_SOURCES:
                raise ValueError(f"collection_source must be one of {sorted(COLLECTION_SOURCES)}")
            metadata = dict(
                deepcopy(metadata), task_label=task_label, collection_source=collection_source
            )
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
            self.staging_path, self.episode_dir = staging, destination
            self.thread = threading.Thread(
                target=self.write_episode,
                args=(metadata,),
                name="episode-writer",
                daemon=False,
            )
            self.thread.start()
            if not self.ready.wait(RECORDER_START_TIMEOUT_S):
                raise TimeoutError("recording START timed out")
            self.check_error()
            self.is_recording = True
        except BaseException as exc:
            self.store_error(exc)
            self.cancelled.set()
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=RECORDER_STOP_TIMEOUT_S)
            if not isinstance(exc, Exception):
                raise
            raise RecordingError(f"recording START failed: {exc}") from exc

    def add_frame(self, row, command=None, result=None):
        """Snapshot mutable inputs once; live observations already own read-only arrays."""
        def owned(value):
            if isinstance(value, np.ndarray) and value.flags.writeable:
                value = value.copy()
                value.flags.writeable = False
            return value

        try:
            self.check_error()
            if not self.is_recording:
                return
            snapshot = SimpleNamespace(
                arm=owned(row.arm), hand=owned(row.hand),
                camera=(
                    {name: owned(value) for name, value in row.camera.items()}
                    if row.camera is not None else None
                ),
                observation_timestamp_ns=row.observation_timestamp_ns,
                control_timestamp_ns=row.control_timestamp_ns,
            )
            self.queue.put_nowait((snapshot, command, result))
            self.submitted_frames += 1
        except (Full, RecordingError) as exc:
            # Queue saturation may follow a real dispatch. Keep that bounded fact
            # even though its RGB-D row never entered the accepted prefix.
            detail = frame_dispatch_detail(row, command, result)
            if detail is not None:
                self.failed_dispatches.append(detail)
            if isinstance(exc, Full):
                error = RecordingError("recording queue full (capacity=16)")
                self.store_error(error)
                raise error from exc
            raise

    def save_episode(self, reason="manual", *, stop_result, technical_status="normal", details=None):
        return self.finish(
            save=True, reason=reason, details=details,
            stop_result=stop_result, technical_status=technical_status,
        )

    def discard_episode(self, reason="discard"):
        self.finish(save=False, reason=reason)

    def close(self):
        """Join an active writer without inventing a physical stop confirmation."""
        self.finish(save=True, reason="close")

    def finish(
        self, *, save, reason, details=None, stop_result="unconfirmed", technical_status="failed"
    ):
        was_recording, self.is_recording = self.is_recording, False
        thread = self.thread
        deadline = time.monotonic() + RECORDER_STOP_TIMEOUT_S
        interrupted = None
        try:
            if self.finish_request is None:
                try:
                    self.finish_request = CaptureFinishRequest(
                        save, reason, deepcopy(details), stop_result, technical_status
                    )
                except BaseException as exc:
                    self.store_error(exc, io=True)
                    self.finish_request = CaptureFinishRequest(True, reason, None)
                    if not isinstance(exc, Exception):
                        interrupted = exc
            if was_recording and self.writer_pending and not self.cancelled.is_set():
                self.queue.put(self.finish_request, timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as exc:
            self.store_error(exc)
            self.cancelled.set()
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            if thread is not None and thread.ident is not None:
                try:
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                except BaseException as exc:
                    self.store_error(exc)
                    self.cancelled.set()
                    if not isinstance(exc, Exception):
                        interrupted = exc
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if self.writer_pending:
                    self.store_error(
                        TimeoutError("recording finalization timed out; staging remains owned")
                    )
                    self.cancelled.set()
        if not self.writer_pending:
            # Writer 的末次 drain 之后仍可能有已越过 check_error 的提交。
            self.drain_unwritten_frames()
        if interrupted is not None:
            raise interrupted
        if not self.writer_pending and not save:
            if self.staging_path is not None:
                shutil.rmtree(self.staging_path)
                self.staging_path = None
            return None
        if self.staging_path is not None and not self.writer_pending and self.error is not None:
            # The failing encoder/HDF5 may already be closed when the last send
            # returns. Only unpublished staging receives this bounded evidence.
            dispatches = [*self.failed_dispatches]
            if self.writer is not None:
                dispatches.extend(self.writer.unwritten_dispatches)
            unique = []
            for detail in dispatches:
                if detail not in unique:
                    unique.append(detail)
            failure_path = self.staging_path / "capture_failure.json"
            if not failure_path.exists():
                atomic_json_dump(
                    dict(
                        reason=self.finish_request.reason, technical_status="failed",
                        stop_result=self.finish_request.stop_result,
                        error=str(self.error), unwritten_dispatches=unique,
                    ),
                    failure_path, overwrite=False,
                )
        if self.io_failed or self.published_path is None:
            self.check_error()
        return self.published_path

    def write_episode(self, metadata):
        staging, destination = self.staging_path, self.episode_dir
        finish = None
        try:
            self.writer = writer = RawEpisodeWriter(staging, control_hz=self.control_hz)
            try:
                writer.open(metadata)
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
                        writer.append_frame(*item)
                except BaseException as exc:
                    # Notify control before flushing or closing potentially blocking native I/O.
                    self.store_error(exc, io=True)
                    detail = frame_dispatch_detail(*item)
                    if detail is not None:
                        self.failed_dispatches.append(detail)
                writer.flush()
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
                details.extend(self.failed_dispatches)
                writer.write_termination(
                    reason, details,
                    expected_frames=self.submitted_frames if not self.io_failed else None,
                    stop_result=finish.stop_result if finish is not None else "unconfirmed",
                    technical_status=(
                        finish.technical_status
                        if finish is not None and self.error is None else "failed"
                    ),
                )
            finally:
                writer.close()
            if not self.io_failed and not self.cancelled.is_set() and finish is not None:
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
                    self.staging_path = None
        except BaseException as exc:
            self.store_error(exc, io=True)
            logger.exception("Recording failed; staging retained: %s", staging)
        finally:
            self.ready.set()
            if self.error is not None and self.published_path is None:
                logger.error(
                    "Recording staging retained: %s frames=%d", staging, self.written_frames
                )
            self.drain_unwritten_frames()

    def drain_unwritten_frames(self):
        """Retain bounded queued dispatches after the writer stops consuming rows."""
        while True:
            try:
                item = self.queue.get_nowait()
            except Empty:
                break
            if not isinstance(item, CaptureFinishRequest):
                detail = frame_dispatch_detail(*item)
                if detail is not None:
                    self.failed_dispatches.append(detail)
