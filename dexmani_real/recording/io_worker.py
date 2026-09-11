"""Dedicated recorder worker with a bounded shared-memory sample ring.

Policy owns episode boundaries, the configured control grid, and sample
contents. This module first ownership-copies each fixed shared-memory record
into an ``EpisodeFrame``, then owns serialization, non-blocking finalization,
camera encoding, HDF5 writes, verification and transactional publication.
Large camera arrays never travel through an ``mp.Queue``; they occupy fixed
slots in a seqlock ring.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.recording.client import (
    RECORDER_STOP_TIMEOUT_S,
    RecordingFinished,
    RecordingStarted,
    StartRecording,
    StopRecording,
)
from dexmani_real.recording.frame import decode_record_sample
from dexmani_real.recording.recorder import (
    EpisodeFinalizationError,
    EpisodeRecorder,
    normalize_provenance_metadata,
)
from dexmani_real.recording.storage.camera_writer import CameraStreamWriterConfig
from dexmani_real.sensor.camera.geometry import RGBDGeometry
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


@dataclass
class _PendingFinalization:
    save: bool
    reason: str
    path: str
    frame_count: int
    started_monotonic_s: float
    forced_error: str = ""
    truncated: bool = False
    thread: threading.Thread | None = None
    results: Queue = field(default_factory=Queue)


@dataclass(frozen=True)
class RecorderIOConfig:
    data_dir: str
    max_frames: int
    control_hz: float
    min_frames: int
    camera_calibration: CameraExtrinsics = field(default_factory=CameraExtrinsics)
    poll_hz: float = 128.0
    writer_queue_size: int = 8
    provenance: Mapping[str, str] = field(default_factory=dict)
    max_frames_stop_reason: str = "max_frames"

    def __post_init__(self) -> None:
        if (
            self.max_frames <= 0
            or self.min_frames < 0
            or not np.isfinite(self.control_hz)
            or not np.isfinite(self.poll_hz)
            or self.control_hz <= 0
            or self.poll_hz <= 0
            or self.writer_queue_size <= 0
        ):
            raise ValueError("invalid RecorderIO capacity/rate configuration")
        if not isinstance(self.camera_calibration, CameraExtrinsics):
            raise TypeError(
                "camera_calibration must be a preloaded CameraExtrinsics snapshot"
            )
        if not self.max_frames_stop_reason.strip():
            raise ValueError("max_frames_stop_reason must be non-empty")
        object.__setattr__(
            self,
            "provenance",
            normalize_provenance_metadata(self.provenance),
        )


def _shared_text(value: bytes, *, default: str | None) -> str | None:
    encoded = value.rstrip(b"\x00")
    return encoded.decode("utf-8") if encoded else default


def _camera_geometry_from_shared(camera_geometry_json: str) -> RGBDGeometry:
    """Decode the camera worker's immutable native RGB-D geometry snapshot."""
    try:
        raw_geometry = json.loads(camera_geometry_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("camera_geometry_json is not valid JSON") from exc
    if not isinstance(raw_geometry, dict):
        raise RuntimeError("camera_geometry_json must contain a JSON object")
    try:
        return RGBDGeometry.from_dict(raw_geometry)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("camera_geometry_json is malformed") from exc


def _build_start_metadata(
    shared: Any,
    *,
    task_label: str,
    operator: str,
    episode_name: str | None,
    calibration: CameraExtrinsics,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    """Snapshot only essential recording metadata at the immutable START boundary."""
    depth_scale = (
        float(shared.camera_depth_scale.value)
        if shared.camera_depth_scale.value != 0.0
        else None
    )
    camera_serial = _shared_text(shared.camera_serial.value, default=None)
    camera_geometry_json = (
        _shared_text(shared.camera_geometry.value, default="{}") or "{}"
    )
    camera_geometry = _camera_geometry_from_shared(camera_geometry_json)
    try:
        camera_name = (
            calibration.resolve_name_by_serial(camera_serial) if camera_serial else None
        )
    except (KeyError, FileNotFoundError):
        camera_name = None
        logger.warning(
            "Camera serial %s not found in cameras.json — no extrinsics in /meta",
            camera_serial,
        )

    return {
        "task_label": task_label,
        "operator": operator,
        "episode_name": episode_name,
        "calib": calibration,
        "camera_geometry": camera_geometry,
        "camera_name": camera_name,
        "camera_serial": camera_serial,
        "depth_scale": depth_scale,
        "provenance": dict(provenance),
    }


def _create_episode_recorder(shared: Any, config: RecorderIOConfig) -> EpisodeRecorder:
    """Build the process-owned recorder from the shared sample layout."""
    sample_dtype = shared.record_sample_ring.dtype
    rgb_dims = sample_dtype.fields["camera_rgb"][0].shape
    depth_dims = sample_dtype.fields["camera_depth"][0].shape
    rgb_shape = (int(rgb_dims[0]), int(rgb_dims[1]), int(rgb_dims[2]))
    depth_shape = (int(depth_dims[0]), int(depth_dims[1]))
    return EpisodeRecorder(
        data_dir=config.data_dir,
        max_frames=config.max_frames,
        control_hz=config.control_hz,
        min_frames=config.min_frames,
        camera_writer_config=CameraStreamWriterConfig(
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            fps=config.control_hz,
            queue_size=config.writer_queue_size,
        ),
    )


@dataclass
class _RecorderIOSession:
    """Own one recording and the sample FIFO; queues carry only boundaries."""

    shared: Any
    config: RecorderIOConfig
    recorder: EpisodeRecorder
    last_sample_sequence: int = 0
    pending_stop: StopRecording | None = None
    pending_finalization: _PendingFinalization | None = None
    fatal: bool = False

    @classmethod
    def create(cls, shared: Any, config: RecorderIOConfig, recorder: EpisodeRecorder):
        return cls(
            shared, config, recorder, int(shared.recorder_consumed_sequence.value)
        )

    @property
    def should_run(self) -> bool:
        return bool(
            self.shared.is_running.value
            or self.pending_finalization is not None
            or self.recorder.is_recording
        )

    def _send_result(self, result: RecordingStarted | RecordingFinished) -> None:
        self.shared.record_result_q.put(result, timeout=0.1)

    def _handle_start(self, control: StartRecording) -> None:
        if self.fatal or self.pending_finalization is not None or self.recorder.is_recording:
            raise RuntimeError("START while previous recording is active")
        try:
            if (
                control.start_sequence
                != int(self.shared.record_sample_ring.latest_sequence) + 1
            ):
                raise RuntimeError("START must begin after the last committed sample")
            self.last_sample_sequence = control.start_sequence - 1
            self.shared.recorder_consumed_sequence.value = self.last_sample_sequence
            metadata = _build_start_metadata(
                self.shared,
                task_label=control.task,
                operator=control.operator,
                episode_name=control.episode_name,
                calibration=self.config.camera_calibration,
                provenance=self.config.provenance,
            )
            if not self.recorder.start_episode(**metadata):
                raise RuntimeError("EpisodeRecorder refused start")
        except Exception as exc:
            logger.error("RecorderIO start failed", exc_info=True)
            if self.recorder.is_recording:
                self._begin_finalization(
                    save=False, reason="start_error", error=str(exc)
                )
                return
            if not self.recorder.resources_released:
                raise RuntimeError("episode start retained unreleased resources") from exc
            self._send_result(
                RecordingFinished(
                    saved=False,
                    path=None,
                    frame_count=0,
                    reason="start_error",
                    error=str(exc),
                )
            )
            return
        self._send_result(
            RecordingStarted(
                path=self.recorder.episode_path,
                max_frames=self.config.max_frames,
                max_frames_stop_reason=self.config.max_frames_stop_reason,
            )
        )

    def _begin_finalization(self, *, save: bool, reason: str, error: str = "") -> None:
        if self.pending_finalization is not None or not self.recorder.is_recording:
            return
        pending = _PendingFinalization(
            save=save,
            reason=reason,
            path=self.recorder.episode_path or "",
            frame_count=self.recorder.frame_count,
            truncated=self.recorder.max_frames_reached,
            started_monotonic_s=time.monotonic(),
            forced_error=error,
        )
        self.pending_stop = None
        self.pending_finalization = pending
        pending.thread = threading.Thread(
            target=self._finish_episode,
            args=(pending,),
            name="recorder-finalizer",
            daemon=False,
        )
        pending.thread.start()

    def _finish_episode(self, pending: _PendingFinalization) -> None:
        # Only this thread accesses the recorder until the main thread reaps it.
        path = pending.path
        error = None
        try:
            path = self.recorder.finish_episode(pending.save, pending.reason)
            if path is None:
                raise RuntimeError("active episode missing during finalization")
        except Exception as exc:
            error = exc
            logger.error("RecorderIO finalization failed", exc_info=True)
        pending.results.put((path, error, self.recorder.resources_released))

    def _fail_samples(self, reason: str, error: str) -> None:
        """Doom the episode before releasing any unconsumed sample capacity."""
        logger.error("RecorderIO %s: %s", reason, error)
        self._begin_finalization(save=False, reason=reason, error=error)
        self.last_sample_sequence = int(self.shared.record_sample_ring.latest_sequence)
        self.shared.recorder_consumed_sequence.value = self.last_sample_sequence

    def _handle_stop(self, control: StopRecording) -> None:
        # A late STOP after an asynchronous writer failure is not another result.
        if self.pending_finalization is not None or not self.recorder.is_recording:
            return
        latest = int(self.shared.record_sample_ring.latest_sequence)
        if not self.last_sample_sequence <= control.through_sequence <= latest:
            self._fail_samples(
                "invalid_stop_boundary", "STOP boundary is outside committed samples"
            )
            return
        self.pending_stop = control

    def _drain_samples(self) -> None:
        """Read consecutive owned samples, never beyond a received STOP boundary."""
        if self.pending_finalization is not None or not self.recorder.is_recording:
            return
        ring = self.shared.record_sample_ring
        latest = int(ring.latest_sequence)
        if latest - self.last_sample_sequence > ring.maxlen:
            self._fail_samples(
                "sample_ring_overflow", "unconsumed sample ring rows were overwritten"
            )
            return
        through = (
            latest if self.pending_stop is None else self.pending_stop.through_sequence
        )
        for expected in range(self.last_sample_sequence + 1, through + 1):
            result = ring.read_sequence(expected)
            if result is None or int(result[2]) != expected:
                self._fail_samples(
                    "sample_sequence_unavailable",
                    f"sample sequence {expected} unavailable",
                )
                return
            try:
                # Decode copies arrays into the recorder-owned frame before ACK.
                frame = decode_record_sample(result[0][0])
            except Exception as exc:
                self._fail_samples("sample_decode_error", str(exc))
                return
            self.last_sample_sequence = expected
            self.shared.recorder_consumed_sequence.value = expected
            try:
                before = self.recorder.frame_count
                added = self.recorder.add_episode_frame(frame)
                if self.recorder.camera_writer_error:
                    raise RuntimeError(self.recorder.camera_writer_error)
                # The last allowed row is accepted but returns False to signal capacity.
                accepted_last = (
                    self.recorder.max_frames_reached
                    and self.recorder.frame_count
                    == before + 1
                    == self.config.max_frames
                )
                if not added and not accepted_last:
                    raise RuntimeError("recorder rejected a source row at capacity")
            except Exception as exc:
                self._fail_samples("sample_write_error", str(exc))
                return
        stop = self.pending_stop
        if stop is not None and self.last_sample_sequence == stop.through_sequence:
            error = (
                f"recording aborted: {stop.reason}"
                if stop.reason
                in {"sample_ring_overflow", "camera_writer_error", "camera_stall"}
                else ""
            )
            self._begin_finalization(
                save=stop.save and not error, reason=stop.reason, error=error
            )

    def _poll_finalization(self) -> None:
        pending = self.pending_finalization
        if pending is None:
            return
        thread = pending.thread
        if thread is None:
            raise RuntimeError("episode finalizer did not start")
        if time.monotonic() - pending.started_monotonic_s >= RECORDER_STOP_TIMEOUT_S:
            self.fatal = True
            self.shared.error_state.value = True
            raise RuntimeError("episode finalization timed out")
        if thread.is_alive():
            return
        thread.join(timeout=0)
        try:
            path, error, released = pending.results.get_nowait()
        except Empty as exc:
            raise RuntimeError("episode finalizer returned no result") from exc
        if not released:
            self.fatal = True
            self.shared.error_state.value = True
            raise RuntimeError("episode finalizer retained unreleased resources")
        if error is not None and not isinstance(error, EpisodeFinalizationError):
            raise RuntimeError("unexpected episode finalizer failure") from error
        error = str(error) if error is not None else pending.forced_error
        self._send_result(
            RecordingFinished(
                saved=pending.save and not error,
                path=path or pending.path,
                frame_count=pending.frame_count,
                reason=pending.reason,
                error=error or None,
                min_frames_met=pending.frame_count >= self.config.min_frames,
            )
        )
        self.pending_finalization = None

    def shutdown(self) -> bool:
        """Use the same finalizer and its original deadline during worker exit."""
        if self.pending_finalization is None and self.recorder.is_recording:
            self._begin_finalization(save=False, reason="recorder_process_shutdown")
        pending = self.pending_finalization
        if pending is not None:
            thread = pending.thread
            if thread is None or thread.ident is None:
                return False
            remaining_s = max(
                0.0,
                RECORDER_STOP_TIMEOUT_S
                - (time.monotonic() - pending.started_monotonic_s),
            )
            thread.join(timeout=remaining_s)
            if thread.is_alive():
                self.fatal = True
                self.shared.error_state.value = True
                return False
        return self.recorder.resources_released

    def step(self) -> None:
        self.shared.set_heartbeat("recorder", time.monotonic())
        try:
            control = self.shared.record_control_q.get_nowait()
        except Empty:
            control = None
        if isinstance(control, StartRecording):
            self._handle_start(control)
        elif isinstance(control, StopRecording):
            self._handle_stop(control)
        elif control is not None:
            raise RuntimeError(f"unknown recorder command: {type(control).__name__}")
        if self.pending_finalization is not None:
            self._poll_finalization()
            return
        if (
            not self.shared.is_running.value
            and self.recorder.is_recording
            and self.pending_stop is None
        ):
            self._begin_finalization(save=False, reason="runtime_shutdown")
        self._drain_samples()
        if (
            self.pending_finalization is None
            and self.recorder.is_recording
            and self.recorder.camera_writer_error
        ):
            self._fail_samples("camera_writer_error", self.recorder.camera_writer_error)
        self._poll_finalization()


def recorder_io_loop(shared: Any, config: RecorderIOConfig) -> None:
    """Long-lived recorder process with supervised failures and one result owner."""
    recorder = None
    session = None
    crashed = False
    try:
        recorder = _create_episode_recorder(shared, config)
        session = _RecorderIOSession.create(shared, config, recorder)
        shared.set_heartbeat("recorder", time.monotonic())
        shared.set_ready("recorder")
        limiter = LoopRate(
            config.poll_hz, label="recorder", busy_wait=False, warn_on_overrun=False
        )
        while session.should_run:
            session.step()
            limiter.wait()
    except Exception:
        crashed = True
        if session is not None:
            session.fatal = True
        shared.error_state.value = True
        logger.error("RecorderIO process crashed", exc_info=True)
    finally:
        if session is not None:
            try:
                if not session.shutdown():
                    crashed = True
                    shared.error_state.value = True
            except Exception:
                crashed = True
                shared.error_state.value = True
                logger.error("RecorderIO shutdown failed", exc_info=True)
        logger.info("RecorderIO exited")
    if crashed or (session is not None and session.fatal):
        raise RuntimeError("RecorderIO exited with a recording failure")
