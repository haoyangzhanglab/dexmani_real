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

from dexmani_real.recording.storage.schema import (
    DATASET_SPECS,
    RAW_FORMAT,
    ARM_EXECUTION_FORMAT,
    TIMING_ROW_DTYPE,
    validate_execution_boundary,
)
from dexmani_real.recording.storage.video import VideoEncoder
from dexmani_real.robot.commands import DispatchStatus
from dexmani_real.runtime.observation import observation_timing
from dexmani_real.sensor.camera.geometry import RGBDGeometry
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)
RECORDER_START_TIMEOUT_S = 10.0
RECORDER_STOP_TIMEOUT_S = 60.0
_COLLECTION_SOURCES = frozenset({"teleop", "policy_rollout"})


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


def snapshot_recording_metadata(runtime_state, *, camera_calibration, hand):
    """Build physical HDF5 attributes from the active camera and consumed tactile semantics."""
    serial = runtime_state.sensors.camera_serial.value.decode("utf-8")
    if not serial.strip():
        raise ValueError("recording START requires a nonempty camera serial")
    geometry = RGBDGeometry.from_dict(
        json.loads(runtime_state.sensors.camera_geometry.value.decode("utf-8"))
    )
    transform = None
    if camera_calibration is None:
        logger.warning("Recording without camera extrinsics: session has no calibration")
    else:
        try:
            transform = camera_calibration.for_serial(serial)
        except KeyError as exc:
            logger.warning("Recording without camera extrinsics: %s", exc)
    color = geometry.color
    depth_scale = float(runtime_state.sensors.camera_depth_scale.value)
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


def raw_row(row, command, result, timestamp):
    arm, hand, camera = (
        row.arm[0],
        None if row.hand is None else row.hand[0],
        row.camera,
    )
    if camera is None:
        raise ValueError("recording requires RGB-D from the current observation")
    values = {
        "timestamp": timestamp,
        "dispatch_status": (int(result.arm), int(result.hand)) if result is not None else (0, 0),
        "arm_qpos": arm["qpos"],
        "arm_qvel": arm["qvel"],
        "arm_effort": arm["effort"],
        "hand_qpos": hand["qpos"] if hand is not None else np.full(12, np.nan),
        "hand_current": hand["current"] if hand is not None else np.full(12, np.nan),
        "hand_contact": hand["tactile_aggregate"] if hand is not None else np.full((5, 3), np.nan),
        "hand_tactile_force": (
            hand["tactile_dense"] if hand is not None else np.full((5, 120, 3), np.nan)
        ),
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
    if any(status not in range(6) for status in values["dispatch_status"]):
        raise ValueError("dispatch_status must contain status codes 0 through 5")
    data = {}
    for name, value in values.items():
        spec = DATASET_SPECS[name]
        array = np.asarray(value, dtype=spec.dtype)
        if array.shape != spec.tail_shape:
            raise ValueError(f"Raw {name}: expected {spec.tail_shape}, got {array.shape}")
        data[name] = array
    return data


class AsyncEpisodeRecorder:
    """One FIFO writer owns snapshots and I/O; revoke motion before blocking finalization."""

    def __init__(self, data_dir, *, control_hz):
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("recording control_hz must be finite and positive")
        self.data_dir = Path(data_dir)
        self.control_hz = float(control_hz)
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
    def staging_path(self):
        return self._temp_dir

    @property
    def written_frames(self):
        return self._written_frames

    def store_error(self, error):
        if self._error is None:
            self._error = error

    def check_error(self):
        if self._error is not None:
            raise RecordingError(f"Recording failed: {self._error}") from self._error

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
        if self._recording or (self._thread is not None and self._thread.is_alive()):
            raise RecordingError("previous recording still owns resources")
        self._temp_dir = self._episode_dir = self._published_path = None
        self._submitted_frames = self._written_frames = 0
        try:
            arm_execution, arm_control_metadata = deepcopy(arm_execution), deepcopy(
                arm_control_metadata
            )
            validate_execution_boundary(arm_execution)
            if not isinstance(task_label, str) or not task_label.strip():
                raise ValueError("task_label must be a non-empty string")
            if collection_source not in _COLLECTION_SOURCES:
                raise ValueError(f"collection_source must be one of {sorted(_COLLECTION_SOURCES)}")
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
            self._temp_dir, self._episode_dir = staging, destination
            self._queue = Queue(maxsize=16)
            self._ready.clear()
            self._cancelled.clear()
            self._thread = threading.Thread(
                target=self.write_episode,
                args=(metadata, arm_execution, arm_control_metadata),
                name="episode-writer",
                daemon=False,
            )
            self._thread.start()
            if not self._ready.wait(RECORDER_START_TIMEOUT_S):
                raise TimeoutError("recording START timed out")
            self.check_error()
            self._recording = True
        except BaseException as exc:
            self.store_error(exc)
            self._cancelled.set()
            if self._thread is not None and self._thread.ident is not None:
                self._thread.join(timeout=RECORDER_STOP_TIMEOUT_S)
            if not isinstance(exc, Exception):
                raise
            raise RecordingError(f"recording START failed: {exc}") from exc

    def enqueue(self, item):
        self.check_error()
        if not self._recording:
            return False
        try:
            self._queue.put_nowait(item)
        except Full as exc:
            error = RecordingError("recording queue full (capacity=16)")
            self.store_error(error)
            raise error from exc
        return True

    def add_frame(self, row, command=None, result=None, *, timing=None):
        """Submit already-owned immutable snapshots without conversion or I/O."""
        if self.enqueue((row, command, result, dict(timing or {}))):
            self._submitted_frames += 1

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
                    finish = CaptureFinishRequest(
                        save, reason, deepcopy(details), deepcopy(timing), deepcopy(arm_execution)
                    )
                except BaseException as exc:
                    self.store_error(exc)
                    finish = CaptureFinishRequest(False, reason, None)
                    if not isinstance(exc, Exception):
                        interrupted = exc
                self._queue.put(finish, timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as exc:
            self.store_error(exc)
            self._cancelled.set()
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            try:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException as exc:
                self.store_error(exc)
                self._cancelled.set()
                if not isinstance(exc, Exception):
                    interrupted = exc
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self.store_error(
                    TimeoutError("recording finalization timed out; staging remains owned")
                )
                self._cancelled.set()
        if interrupted is not None:
            raise interrupted
        self.check_error()
        self._thread = None
        return self._published_path

    def write_episode(self, metadata, arm_start, arm_control_metadata):
        staging, destination = self._temp_dir, self._episode_dir
        finish, buffered = None, 0
        query_indices = {}
        first_stamp = previous_stamp = None
        try:
            with ExitStack() as resources:
                data = resources.enter_context(h5py.File(staging / "data.h5", "w"))
                meta = data.create_group("meta")
                meta.attrs.update(metadata)
                meta.attrs.update(
                    control_hz=self.control_hz,
                    timestamp_source="host_monotonic_relative",
                    dispatch_status_source="recorded",
                )
                datasets = {
                    name: data.create_dataset(
                        name,
                        shape=(0, *spec.tail_shape),
                        maxshape=(None, *spec.tail_shape),
                        dtype=spec.dtype,
                    )
                    for name, spec in DATASET_SPECS.items()
                }
                meta.attrs.update(
                    arm_control_mode=1,
                    arm_action_semantics="q_goal_submitted",
                    arm_execution_format=ARM_EXECUTION_FORMAT,
                )
                execution = data.create_group("arm_execution")
                execution.attrs.update(arm_control_metadata)
                execution.create_dataset("start", data=json.dumps(arm_start, allow_nan=False))

                timing = data.create_group("timing")
                timing.attrs.update(clock="host_monotonic_ns", missing_value=-1)
                timing.attrs.update(
                    camera_timestamp_source="oldest_channel_first_host_receive_monotonic",
                )
                row_times = timing.create_dataset(
                    "rows", shape=(0,), maxshape=(None,), dtype=TIMING_ROW_DTYPE
                )
                queries = timing.create_dataset(
                    "queries", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype()
                )
                rgb_shape = (metadata["camera_color_height"], metadata["camera_color_width"], 3)
                datasets["depth"] = data.create_dataset(
                    "depth",
                    shape=(0, *rgb_shape[:2]),
                    maxshape=(None, *rgb_shape[:2]),
                    chunks=(1, *rgb_shape[:2]),
                    dtype=np.uint16,
                    compression="gzip",
                    compression_opts=1,
                )
                # Copy writer-owned columns once so depth views do not retain full RGB-D snapshots.
                batch_size = 32
                arrays = {
                    name: np.empty((batch_size, *dataset.shape[1:]), dtype=dataset.dtype)
                    for name, dataset in datasets.items()
                }
                timing_rows = np.empty(batch_size, dtype=TIMING_ROW_DTYPE)
                height, width, _ = rgb_shape
                video = VideoEncoder(
                    staging / "rgb.mp4", fps=self.control_hz, width=width, height=height
                )
                resources.callback(video.close)
                video.open()
                self._ready.set()

                def flush():
                    nonlocal buffered
                    if not buffered:
                        return
                    # Failed appends retain staging; retries could duplicate a partially written batch.
                    count, buffered = buffered, 0
                    end = self._written_frames + count
                    for name, values in arrays.items():
                        datasets[name].resize(end, axis=0)
                        datasets[name][self._written_frames : end] = values[:count]
                    row_times.resize(end, axis=0)
                    row_times[self._written_frames : end] = timing_rows[:count]
                    self._written_frames = end

                try:
                    while not self._cancelled.is_set():
                        try:
                            item = self._queue.get(timeout=0.05)
                        except Empty:
                            continue
                        if isinstance(item, CaptureFinishRequest):
                            finish = item
                            break
                        if isinstance(item, QueryTimingRecord):
                            values = item.values
                            key = (values["run_id"], values["query_id"])
                            index = query_indices.get(key)
                            if index is None:
                                index = len(queries)
                                query_indices[key] = index
                                queries.resize(index + 1, axis=0)
                            queries[index] = json.dumps(values, allow_nan=False)
                            continue
                        row, command, result, execution_timing = item
                        stamp = int(row.observation_timestamp_ns)
                        if stamp <= 0 or (previous_stamp is not None and stamp <= previous_stamp):
                            raise ValueError(
                                "recording observation timestamps must be positive and increasing"
                            )
                        if first_stamp is None:
                            first_stamp = stamp
                        # Integer subtraction retains short intervals even for a large host clock.
                        values = raw_row(row, command, result, (stamp - first_stamp) / 1e9)
                        rgb, depth = row.camera["rgb"], row.camera["depth"]
                        if rgb.shape != rgb_shape or rgb.dtype != np.uint8:
                            raise ValueError("RGB shape or dtype mismatch")
                        if depth.shape != rgb_shape[:2] or depth.dtype != np.uint16:
                            raise ValueError("depth shape or dtype mismatch")
                        evidence = dict(observation_timing(row), **execution_timing)
                        evidence["command_seq"] = -1
                        if result is not None and result.arm == DispatchStatus.SUBMITTED:
                            run_id, sequence = result.command_id
                            if evidence.get("run_id", run_id) != run_id:
                                raise ValueError("dispatch/timing run_id mismatch")
                            evidence.update(run_id=run_id, command_seq=sequence)
                        unknown = evidence.keys() - set(TIMING_ROW_DTYPE.names)
                        if unknown:
                            raise ValueError(f"Unknown timing row fields: {sorted(unknown)}")
                        for name, value in values.items():
                            arrays[name][buffered] = value
                        arrays["depth"][buffered] = depth
                        timing_rows[buffered] = tuple(
                            evidence.get(name, -1) for name in TIMING_ROW_DTYPE.names
                        )
                        video.write_frame(rgb)
                        buffered += 1
                        previous_stamp = stamp
                        if buffered == batch_size:
                            flush()
                except BaseException as exc:
                    # Publish the failure before cleanup, which may block on native I/O.
                    self.store_error(exc)
                flush()
                if finish is not None and finish.arm_execution is not None:
                    execution.create_dataset(
                        "end", data=json.dumps(finish.arm_execution, allow_nan=False)
                    )
                if finish is not None and finish.save:
                    try:
                        validate_execution_boundary(finish.arm_execution, stopped=True)
                    except (KeyError, TypeError, ValueError) as exc:
                        self.store_error(exc)
                reason = finish.reason if finish is not None else "recording_failure"
                details = list(finish.details or []) if finish is not None else []
                if self._error is not None:
                    details.append(
                        dict(
                            stage="recording",
                            exception_type=type(self._error).__name__,
                            message=str(self._error),
                        )
                    )
                meta.attrs.update(
                    format=RAW_FORMAT, num_frames=self._written_frames, termination_reason=reason
                )
                if finish is not None and finish.timing is not None:
                    timing.create_dataset(
                        "termination", data=json.dumps(finish.timing, allow_nan=False)
                    )
                if len(row_times) != self._written_frames:
                    raise RuntimeError("timing/core row count mismatch")
                if details:
                    meta.create_dataset(
                        "termination_details", data=json.dumps(details, allow_nan=False)
                    )
                if self._error is None and (
                    self._written_frames != self._submitted_frames
                    or video.frame_count != self._written_frames
                ):
                    raise RuntimeError("recording accepted/written/camera row count mismatch")
            if self._error is None and not self._cancelled.is_set() and finish is not None:
                if finish.save:
                    # Zero-row captures retain attempted dispatch/failure evidence; only explicit discard deletes them.
                    atomic_publish(staging, destination, cancelled=self._cancelled.is_set)
                    self._published_path = destination
                    logger.info("Raw 已保存：%s（%d 帧）。", destination.name, self._written_frames)
                    logger.debug(
                        "Episode saved: %s frames=%d reason=%s",
                        destination,
                        self._written_frames,
                        finish.reason,
                    )
                else:
                    shutil.rmtree(staging)
                self._temp_dir = None
        except BaseException as exc:
            self.store_error(exc)
            logger.exception("Recording failed; staging retained: %s", staging)
        finally:
            self._ready.set()
            if self._error is not None:
                logger.error(
                    "Recording staging retained: %s frames=%d", staging, self._written_frames
                )
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
