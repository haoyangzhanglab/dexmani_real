"""Copy shared-memory samples into EpisodeFrame rows and serialize raw episodes.

The worker encodes RGB-D, writes HDF5, and verifies files before atomic
publication. Camera arrays use fixed seqlock ring slots rather than mp.Queue.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.recording.client import (
    RecordingResult,
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
from dexmani_real.runtime.safety import RunEndReason, SafetyState, _revoke_motion_locked
from dexmani_real.sensor.camera.geometry import RGBDGeometry
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecorderWorkerConfig:
    data_dir: str
    control_hz: float
    min_frames: int
    camera_calibration: CameraExtrinsics
    poll_hz: float = 128.0
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.min_frames < 0
            or not np.isfinite(self.control_hz)
            or not np.isfinite(self.poll_hz)
            or self.control_hz <= 0
            or self.poll_hz <= 0
        ):
            raise ValueError("invalid RecorderIO rate/min_frames configuration")
        if not isinstance(self.camera_calibration, CameraExtrinsics):
            raise TypeError("camera_calibration must be a preloaded CameraExtrinsics snapshot")
        object.__setattr__(
            self,
            "provenance",
            normalize_provenance_metadata(self.provenance),
        )


def _warn_stale_staging(data_dir: str) -> None:
    # Both timestamp and explicit episode names use .tmp_<non-hidden episode name>.
    staging = sorted(
        path.name
        for path in Path(data_dir).glob(".tmp_[!.]*")
        if path.is_dir() and not path.is_symlink()
    )
    if staging:
        logger.warning(
            "Found %d stale recording staging directories in %s (up to 5 shown): %s. "
            "They are not published raw episodes and may remain after forced termination "
            "or interrupted recording I/O. They will not be automatically deleted or "
            "recovered; inspect them and handle or remove them manually.",
            len(staging),
            data_dir,
            ", ".join(staging[:5]),
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
    """Snapshot recording metadata when START is received."""
    depth_scale = (
        float(shared.camera_depth_scale.value) if shared.camera_depth_scale.value != 0.0 else None
    )
    camera_serial = _shared_text(shared.camera_serial.value, default=None)
    camera_geometry_json = _shared_text(shared.camera_geometry.value, default="{}") or "{}"
    camera_geometry = _camera_geometry_from_shared(camera_geometry_json)
    try:
        camera_name = calibration.resolve_name_by_serial(camera_serial) if camera_serial else None
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


def _create_episode_recorder(shared: Any, config: RecorderWorkerConfig) -> EpisodeRecorder:
    sample_dtype = shared.record_sample_ring.dtype
    rgb_dims = sample_dtype.fields["camera_rgb"][0].shape
    depth_dims = sample_dtype.fields["camera_depth"][0].shape
    rgb_shape = (int(rgb_dims[0]), int(rgb_dims[1]), int(rgb_dims[2]))
    depth_shape = (int(depth_dims[0]), int(depth_dims[1]))
    return EpisodeRecorder(
        data_dir=config.data_dir,
        control_hz=config.control_hz,
        min_frames=config.min_frames,
        camera_writer_config=CameraStreamWriterConfig(
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            fps=config.control_hz,
        ),
    )


def _revoke_recording_motion(shared: Any) -> None:
    with shared.motion_lock:
        shared.is_running.value = False
        if int(shared.safety_state.value) == int(SafetyState.RUNNING):
            _revoke_motion_locked(shared, SafetyState.ARMED, reason=RunEndReason.RECORDING_FAILURE)


@dataclass
class _RecorderIOSession:
    shared: Any
    config: RecorderWorkerConfig
    recorder: EpisodeRecorder
    last_sample_sequence: int = 0
    pending_stop: StopRecording | None = None
    fatal: bool = False

    @property
    def should_run(self) -> bool:
        return not self.fatal and bool(self.shared.is_running.value or self.recorder.is_recording)

    def _send_result(self, result: RecordingStarted | RecordingResult) -> None:
        self.shared.record_result_q.put(result, timeout=0.1)

    def _handle_start(self, control: StartRecording) -> None:
        if self.fatal or self.recorder.is_recording:
            raise RuntimeError("START while previous recording is active")
        try:
            if control.start_sequence != int(self.shared.record_sample_ring.latest_sequence) + 1:
                raise RuntimeError("START must begin after the last committed sample")
            self.last_sample_sequence = control.start_sequence - 1
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
            self.fatal = True
            _revoke_recording_motion(self.shared)
            if self.recorder.is_recording:
                self._begin_finalization(save=False, reason="start_error", error=str(exc))
                return
            if not self.recorder.resources_released:
                raise RuntimeError("episode start retained unreleased resources") from exc
            try:
                self._send_result(RecordingResult(reason="start_error", error=str(exc)))
            except Exception:
                logger.error("RecorderIO could not report START failure", exc_info=True)
            raise
        self._send_result(RecordingStarted(path=self.recorder.episode_path))

    def _begin_finalization(self, *, save: bool, reason: str, error: str = "") -> None:
        if not self.recorder.is_recording:
            return
        path = self.recorder.episode_path
        frame_count = self.recorder.frame_count
        self.pending_stop = None
        published = False
        try:
            self.recorder.finish_episode(save and not error, reason)
            published = self.recorder.last_finish_saved
        except EpisodeFinalizationError as exc:
            if error:
                logger.error("RecorderIO cleanup after %s failed", error, exc_info=True)
            else:
                error = str(exc)
        if error:
            self.fatal = True
            _revoke_recording_motion(self.shared)
        if not self.recorder.resources_released:
            raise RuntimeError(error or "episode retained unreleased resources")
        try:
            self._send_result(
                RecordingResult(
                    saved=published,
                    path=path,
                    frame_count=frame_count,
                    reason=reason,
                    error=error or None,
                    min_frames_met=frame_count >= self.config.min_frames,
                )
            )
        except Exception:
            if not error:
                raise
            logger.error("RecorderIO could not report recording failure", exc_info=True)
        if error:
            raise RuntimeError(error)

    def _fail_samples(self, reason: str, error: str) -> None:
        logger.error("RecorderIO %s: %s", reason, error)
        self.fatal = True
        # Revoke before failed storage cleanup can block.
        _revoke_recording_motion(self.shared)
        self._begin_finalization(save=False, reason=reason, error=error)

    def _handle_stop(self, control: StopRecording) -> None:
        # A late or duplicate STOP cannot change the first result or cutoff.
        if not self.recorder.is_recording or self.pending_stop is not None:
            return
        latest = int(self.shared.record_sample_ring.latest_sequence)
        if not self.last_sample_sequence <= control.through_sequence <= latest:
            self._fail_samples(
                "invalid_stop_boundary", "STOP boundary is outside committed samples"
            )
            return
        self.recorder.had_pause = control.had_pause
        self.recorder.technical_status = control.technical_status
        if not control.save:
            # Discarded rows still occupy global sequence numbers across episodes.
            self.last_sample_sequence = control.through_sequence
            self._begin_finalization(save=False, reason=control.reason)
            return
        self.pending_stop = control

    def _drain_samples(self) -> None:
        if not self.recorder.is_recording:
            return
        ring = self.shared.record_sample_ring
        latest = int(ring.latest_sequence)
        if latest - self.last_sample_sequence > ring.maxlen:
            self._fail_samples(
                "sample_ring_overflow", "unconsumed sample ring rows were overwritten"
            )
            return
        through = latest if self.pending_stop is None else self.pending_stop.through_sequence
        for expected in range(self.last_sample_sequence + 1, through + 1):
            result = ring.read_sequence(expected)
            if result is None or int(result[2]) != expected:
                self._fail_samples(
                    "sample_sequence_unavailable", f"sample sequence {expected} unavailable"
                )
                return
            try:
                frame = decode_record_sample(result[0][0])
            except Exception as exc:
                self._fail_samples("sample_decode_error", str(exc))
                return
            self.last_sample_sequence = expected
            try:
                if not self.recorder.add_frame(frame):
                    raise RuntimeError("recorder rejected a source row")
            except Exception as exc:
                self._fail_samples("sample_write_error", str(exc))
                return
        stop = self.pending_stop
        if stop is not None and self.last_sample_sequence == stop.through_sequence:
            self._begin_finalization(save=True, reason=stop.reason)

    def shutdown(self) -> bool:
        if self.recorder.is_recording:
            if self.fatal:
                self._begin_finalization(
                    save=False, reason="recorder_failure", error="RecorderIO failed during capture"
                )
            else:
                if self.pending_stop is None:
                    self._handle_stop(
                        StopRecording(
                            save=True,
                            reason="runtime_shutdown",
                            through_sequence=int(self.shared.record_sample_ring.latest_sequence),
                            technical_status="invalid",
                        )
                    )
                self._drain_samples()
        return self.recorder.resources_released

    def step(self) -> None:
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
        if not self.shared.is_running.value:
            self.shutdown()
        else:
            self._drain_samples()


def run_recorder_worker(shared: Any, config: RecorderWorkerConfig) -> None:
    """Record episodes until shutdown, reporting worker failures to supervision."""
    session = None
    failure = None
    try:
        _warn_stale_staging(config.data_dir)
        recorder = _create_episode_recorder(shared, config)
        session = _RecorderIOSession(shared, config, recorder)
        shared.recorder_ready.set()
        limiter = LoopRate(config.poll_hz, label="recorder", busy_wait=False, warn_on_overrun=False)
        while session.should_run:
            session.step()
            limiter.wait()
    except Exception as exc:
        failure = exc
        _revoke_recording_motion(shared)
        if session is not None:
            session.fatal = True
        logger.error("RecorderIO process crashed", exc_info=True)
    finally:
        if session is not None:
            try:
                if not session.shutdown():
                    raise RuntimeError("RecorderIO retained unreleased resources")
            except Exception as exc:
                if failure is None:
                    failure = exc
                _revoke_recording_motion(shared)
                logger.error("RecorderIO shutdown failed", exc_info=True)
        shared.recorder_ready.clear()
        logger.info("RecorderIO exited")
    if failure is not None:
        raise failure
    if session is not None and session.fatal:
        raise RuntimeError("RecorderIO exited with a recording failure")
