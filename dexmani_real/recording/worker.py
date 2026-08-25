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
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.ipc.schema import (
    RECORD_STATUS_DTYPE,
    RECORD_STATUS_TEXT_BYTES,
    RECORD_STOP_REASON_BYTES,
)
from dexmani_real.recording.camera_writer import CameraStreamWriterConfig
from dexmani_real.recording.client import (
    RECORDER_STOP_TIMEOUT_S,
    RecorderCommand,
    RecorderPhase,
    bounded_control_text,
)
from dexmani_real.recording.frame import decode_record_sample
from dexmani_real.recording.recorder import EpisodeRecorder
from dexmani_real.recording.recorder import StopResult as EpisodeStopResult
from dexmani_real.sensor.camera_geometry import RGBDGeometry
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

logger = get_logger(__name__)

STATUS_TEXT_BYTES = RECORD_STATUS_TEXT_BYTES


@dataclass
class _PendingFinalization:
    generation: int
    save: bool
    reason: str
    path: str
    frame_count: int
    started_monotonic_s: float
    forced_error: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class RecorderIOConfig:
    data_dir: str
    max_frames: int
    control_hz: float
    min_frames: int
    resolved_config_sha256: str
    camera_calibration: CameraCalib = field(default_factory=CameraCalib)
    provenance: tuple[tuple[str, str], ...] = ()
    poll_hz: float = 128.0
    writer_queue_size: int = 8

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
        if len(self.resolved_config_sha256) != 64:
            raise ValueError("RecorderIO requires the resolved config SHA-256")
        if not isinstance(self.camera_calibration, CameraCalib):
            raise TypeError("camera_calibration must be a preloaded CameraCalib snapshot")
        provenance_names = [name for name, _value in self.provenance]
        if len(set(provenance_names)) != len(provenance_names):
            raise ValueError("RecorderIO provenance keys must be unique")
        if any(not name or not value for name, value in self.provenance):
            raise ValueError("RecorderIO provenance keys and values must be non-empty")


def _control_text(record: np.void, field: str) -> str:
    """Decode a null-padded fixed control-plane text field."""
    return bytes(record[field]).rstrip(b"\x00").decode("utf-8", errors="replace")


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
    calibration: CameraCalib,
) -> dict[str, Any]:
    """Snapshot only essential recording metadata at the immutable START boundary."""
    depth_scale = (
        float(shared.camera_depth_scale.value)
        if shared.camera_depth_scale.value != 0.0
        else None
    )
    camera_serial = _shared_text(shared.camera_serial.value, default=None)
    camera_firmware = (
        _shared_text(shared.camera_firmware.value, default="unknown") or "unknown"
    )
    camera_sdk_version = (
        _shared_text(shared.camera_sdk_version.value, default="unknown") or "unknown"
    )
    camera_profile_json = (
        _shared_text(shared.camera_profile.value, default="{}") or "{}"
    )
    camera_geometry_json = (
        _shared_text(shared.camera_geometry.value, default="{}") or "{}"
    )
    camera_geometry = _camera_geometry_from_shared(camera_geometry_json)
    arm_identity_json = (
        _shared_text(
            shared.arm_device_identity.value, default='{"status":"unavailable"}'
        )
        or '{"status":"unavailable"}'
    )
    hand_identity_json = _shared_text(
        shared.hand_device_identity.value, default='{"status":"unavailable"}'
    )
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
        "calib": calibration,
        "camera_geometry": camera_geometry,
        "camera_name": camera_name,
        "camera_serial": camera_serial,
        "depth_scale": depth_scale,
        "camera_metadata": {
            "camera_firmware": camera_firmware,
            "camera_sdk_version": camera_sdk_version,
            "camera_actual_profile_json": camera_profile_json,
            "camera_payload_mode": "depth_to_color_aligned_rgbd",
            "camera_geometry_json": camera_geometry_json,
            "arm_device_identity_json": arm_identity_json,
            "hand_device_identity_json": hand_identity_json or '{"status":"disabled"}',
        },
    }


def _bounded_text(value: str) -> tuple[bytes, int]:
    payload = value.encode("utf-8")[:STATUS_TEXT_BYTES]
    return payload, len(payload)


def _publish_status(
    shared: Any,
    phase: RecorderPhase,
    generation: int,
    *,
    frame_count: int = 0,
    error: str = "",
    path: str = "",
    saved: bool = False,
    reason: str = "",
    min_frames_met: bool = False,
    failure_count: int = 0,
) -> None:
    frame = np.zeros(1, dtype=RECORD_STATUS_DTYPE)
    error_bytes, error_length = _bounded_text(error)
    path_bytes, path_length = _bounded_text(path)
    reason_bytes = bounded_control_text(
        reason, capacity=RECORD_STOP_REASON_BYTES, field="status reason"
    )
    frame["phase"][0] = int(phase)
    frame["saved"][0] = int(saved)
    frame["min_frames_met"][0] = int(min_frames_met)
    frame["generation"][0] = generation
    frame["frame_count"][0] = frame_count
    frame["failure_count"][0] = failure_count
    frame["updated_monotonic_ns"][0] = time.monotonic_ns()
    frame["reason_length"][0] = len(reason_bytes)
    frame["reason"][0] = reason_bytes
    frame["error_length"][0] = error_length
    frame["error"][0] = error_bytes
    frame["path_length"][0] = path_length
    frame["path"][0] = path_bytes
    shared.record_status_ring.write(frame)


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
        arm_sent_stream=True,
        resolved_config_hash=config.resolved_config_sha256,
        provenance=dict(config.provenance),
        camera_writer_config=CameraStreamWriterConfig(
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            fps=config.control_hz,
            queue_size=config.writer_queue_size,
        ),
    )


@dataclass
class _RecorderIOSession:
    """Single owner of one RecorderIO transaction, sequence, and finalize state.

    START/STOP controls are generation-ordered.  Samples are decoded only for
    the active transaction, and process shutdown continues polling a pending
    finalize so an in-flight episode is reported as completed, discarded, or
    failed rather than silently abandoned.
    """

    shared: Any
    config: RecorderIOConfig
    recorder: EpisodeRecorder
    active_generation: int = 0
    last_control_sequence: int = 0
    last_sample_sequence: int = 0
    pending_finalization: _PendingFinalization | None = None
    failure_count: int = 0
    sample_backlog_high_watermark: int = 0
    sample_read_failure_count: int = 0

    @classmethod
    def create(
        cls,
        shared: Any,
        config: RecorderIOConfig,
        recorder: EpisodeRecorder,
    ) -> "_RecorderIOSession":
        return cls(
            shared=shared,
            config=config,
            recorder=recorder,
            last_sample_sequence=int(shared.recorder_consumed_sequence.value),
        )

    @property
    def should_run(self) -> bool:
        return bool(
            self.shared.is_running.value
            or self.recorder.is_recording
            or self.pending_finalization is not None
        )

    def _read_control(self) -> np.void | None:
        result = self.shared.record_control_ring.read_latest()
        if result is None:
            return None
        control_data, _publish_ns, sequence = result
        if sequence == self.last_control_sequence:
            return None
        self.last_control_sequence = int(sequence)
        return control_data[0]

    def _begin_finalization(
        self,
        *,
        generation: int,
        save: bool,
        reason: str,
        forced_error: str = "",
    ) -> None:
        if self.pending_finalization is not None or not self.recorder.is_recording:
            return
        frame_count = self.recorder.frame_count
        path = self.recorder.stop_episode(success=save, reason=reason) or ""
        self.pending_finalization = _PendingFinalization(
            generation=generation,
            save=save,
            reason=reason,
            path=path,
            frame_count=frame_count,
            started_monotonic_s=time.monotonic(),
            forced_error=forced_error,
        )
        _publish_status(
            self.shared,
            RecorderPhase.FINALIZING,
            generation,
            frame_count=frame_count,
            path=path,
            reason=reason,
            min_frames_met=frame_count >= self.config.min_frames,
            failure_count=self.failure_count,
        )

    def _handle_start(self, control: np.void) -> None:
        generation = int(control["generation"])
        try:
            if self.recorder.is_recording or self.pending_finalization is not None:
                raise RuntimeError("previous recorder transaction is still active")
            if generation <= self.active_generation:
                raise RuntimeError(
                    "recorder START generation must increase monotonically"
                )
            metadata = _build_start_metadata(
                self.shared,
                task_label=_control_text(control, "task_label"),
                operator=_control_text(control, "operator"),
                calibration=self.config.camera_calibration,
            )
            if not self.recorder.start_episode(**metadata):
                raise RuntimeError("EpisodeRecorder refused start")
            self.sample_backlog_high_watermark = 0
            self.sample_read_failure_count = 0
            self.active_generation = generation
            _publish_status(
                self.shared,
                RecorderPhase.RECORDING,
                generation,
                failure_count=self.failure_count,
            )
        except Exception as exc:
            logger.error("RecorderIO start failed", exc_info=True)
            self.failure_count += 1
            _publish_status(
                self.shared,
                RecorderPhase.ERROR,
                generation,
                error=str(exc),
                reason="start_error",
                failure_count=self.failure_count,
            )

    def _discard_unrecoverable_samples(
        self,
        *,
        latest_sequence: int,
        reason: str,
        stop_reason: str,
    ) -> None:
        """Fail the active transaction, then release stale producer capacity."""
        self.sample_read_failure_count += 1
        if self.recorder.is_recording:
            logger.error("RecorderIO %s", reason)
            self._begin_finalization(
                generation=self.active_generation,
                save=False,
                reason=stop_reason,
                forced_error=reason,
            )
        else:
            logger.warning("RecorderIO discarded stale samples: %s", reason)
        # Once the transaction is doomed (or no transaction is active),
        # acknowledge through the snapshot so a later episode cannot be blocked
        # behind stale slots from the discarded generation.
        self.last_sample_sequence = latest_sequence
        self.shared.recorder_consumed_sequence.value = latest_sequence

    def _drain_samples(self) -> None:
        """Ownership-copy only consecutive unacknowledged samples before STOP.

        ``record_sample_ring`` carries full RGB-D payloads. Scanning its whole
        history at every poll needlessly copied already-consumed image slots and
        raced the producer while it recycled the oldest one. The recorder is a
        FIFO consumer, so it snapshots the producer sequence and reads only
        the exact next sequence(s) it still owns.
        """
        ring = self.shared.record_sample_ring
        latest_sequence = int(ring.latest_sequence)
        pending_count = latest_sequence - self.last_sample_sequence
        if pending_count <= 0:
            return

        previous_high_watermark = self.sample_backlog_high_watermark
        self.sample_backlog_high_watermark = max(
            self.sample_backlog_high_watermark, pending_count
        )
        backlog_warning_threshold = max(1, ring.maxlen - 1)
        if (
            pending_count >= backlog_warning_threshold
            and pending_count > previous_high_watermark
        ):
            logger.warning(
                "RecorderIO sample backlog high: pending=%d capacity=%d",
                pending_count,
                ring.maxlen,
            )

        if pending_count > ring.maxlen:
            self._discard_unrecoverable_samples(
                latest_sequence=latest_sequence,
                reason=(
                    "sample ring overflow: "
                    f"expected {self.last_sample_sequence + 1}, "
                    f"latest {latest_sequence}, capacity {ring.maxlen}"
                ),
                stop_reason="sample_ring_overflow",
            )
            return

        for expected_sequence in range(
            self.last_sample_sequence + 1, latest_sequence + 1
        ):
            result = ring.read_sequence(expected_sequence)
            if result is None:
                # A producer is forbidden from overwriting an unacknowledged
                # slot. Therefore a committed snapshot sequence that cannot be
                # copied is a loss of recorder ownership, not a history miss
                # that this FIFO reader may skip.
                self._discard_unrecoverable_samples(
                    latest_sequence=latest_sequence,
                    reason=(
                        f"sample sequence {expected_sequence} unavailable "
                        f"from snapshot latest={latest_sequence}"
                    ),
                    stop_reason="sample_sequence_unavailable",
                )
                return

            data, _publish_ns, sequence = result
            if sequence != expected_sequence:
                self._discard_unrecoverable_samples(
                    latest_sequence=latest_sequence,
                    reason=(
                        f"sample sequence mismatch: expected {expected_sequence}, "
                        f"received {sequence}"
                    ),
                    stop_reason="sample_sequence_unavailable",
                )
                return

            # This private copy is now owned by RecorderIO, so the producer may
            # safely reuse its shared-memory slot while serialization continues.
            self.last_sample_sequence = sequence
            self.shared.recorder_consumed_sequence.value = sequence
            record = data[0]
            if (
                not self.recorder.is_recording
                or int(record["generation"]) != self.active_generation
            ):
                continue
            try:
                frame = decode_record_sample(
                    record,
                    arm_sent_stream=self.recorder.arm_sent_stream,
                )
                added = self.recorder.add_episode_frame(frame)
                if self.recorder.camera_writer_error:
                    raise RuntimeError(self.recorder.camera_writer_error)
                if not added and self.recorder.max_frames_reached:
                    self._begin_finalization(
                        generation=self.active_generation,
                        save=True,
                        reason="max_frames",
                    )
            except Exception as exc:
                logger.error("RecorderIO sample write failed", exc_info=True)
                self._begin_finalization(
                    generation=self.active_generation,
                    save=False,
                    reason="sample_write_error",
                    forced_error=str(exc),
                )
                return

    def _handle_stop(self, control: np.void) -> None:
        generation = int(control["generation"])
        if generation == self.active_generation and self.recorder.is_recording:
            stop_reason = _control_text(control, "stop_reason") or "manual"
            forced_error = (
                f"recording aborted: {stop_reason}"
                if stop_reason
                in {"sample_ring_overflow", "camera_writer_error", "camera_stall"}
                else ""
            )
            self._begin_finalization(
                generation=generation,
                save=bool(control["save"]),
                reason=stop_reason,
                forced_error=forced_error,
            )
        elif generation != self.active_generation:
            logger.warning(
                "RecorderIO ignored STOP for generation %d (active=%d)",
                generation,
                self.active_generation,
            )

    def _handle_runtime_shutdown(self) -> None:
        if (
            not self.shared.is_running.value
            and self.recorder.is_recording
            and self.pending_finalization is None
        ):
            self._begin_finalization(
                generation=self.active_generation,
                save=False,
                reason="runtime_shutdown",
            )

    def _poll_finalization(self) -> None:
        pending = self.pending_finalization
        if pending is None:
            return
        stop_result: EpisodeStopResult = self.recorder.poll_stop()
        elapsed_s = time.monotonic() - pending.started_monotonic_s
        if (
            not stop_result.done
            and elapsed_s >= RECORDER_STOP_TIMEOUT_S
            and not pending.timed_out
        ):
            pending.timed_out = True
            self.failure_count += 1
            logger.error(
                "RecorderIO episode finalization exceeded %.1fs",
                RECORDER_STOP_TIMEOUT_S,
            )
            _publish_status(
                self.shared,
                RecorderPhase.FINALIZING,
                pending.generation,
                frame_count=pending.frame_count,
                error="episode finalization timed out",
                path=pending.path,
                reason=pending.reason,
                min_frames_met=pending.frame_count >= self.config.min_frames,
                failure_count=self.failure_count,
            )
        if not stop_result.done:
            return

        error = stop_result.error or pending.forced_error
        if pending.timed_out:
            error = error or "episode finalization timed out"
        if error and not pending.timed_out:
            self.failure_count += 1
        phase = RecorderPhase.ERROR if error else RecorderPhase.COMPLETED
        saved = pending.save and not error
        path = stop_result.path or pending.path
        frame_count = stop_result.frame_count or pending.frame_count
        self.pending_finalization = None
        logger.info(
            "RecorderIO sample transport: max_backlog=%d/%d read_failures=%d",
            self.sample_backlog_high_watermark,
            self.shared.record_sample_ring.maxlen,
            self.sample_read_failure_count,
        )
        _publish_status(
            self.shared,
            phase,
            pending.generation,
            frame_count=frame_count,
            error=error or "",
            path=path or "",
            saved=saved,
            reason=pending.reason,
            min_frames_met=frame_count >= self.config.min_frames,
            failure_count=self.failure_count,
        )

    def step(self) -> None:
        """Process one bounded control/sample/finalization iteration."""
        self.shared.set_heartbeat("recorder", time.monotonic())
        control = self._read_control()
        if (
            control is not None
            and RecorderCommand(int(control["command"])) is RecorderCommand.START
        ):
            self._handle_start(control)
        self._drain_samples()
        if (
            control is not None
            and RecorderCommand(int(control["command"])) is RecorderCommand.STOP
        ):
            self._handle_stop(control)
        self._handle_runtime_shutdown()
        self._poll_finalization()


def _shutdown_episode_recorder(recorder: EpisodeRecorder) -> None:
    """Discard active work and reap any pending stop thread at process exit."""
    if recorder.is_recording:
        recorder.stop_episode(
            success=False,
            reason="recorder_process_shutdown",
        )
    if recorder.join_stop(timeout=RECORDER_STOP_TIMEOUT_S):
        return
    if recorder.stop_error:
        logger.error(
            "RecorderIO process-shutdown finalization failed: %s",
            recorder.stop_error,
        )
    else:
        logger.error("RecorderIO timed out during process-shutdown finalization")


def recorder_io_loop(shared: Any, config: RecorderIOConfig) -> None:
    """Long-lived process target. Recording errors never latch robot FAULT."""
    recorder: EpisodeRecorder | None = None
    session: _RecorderIOSession | None = None
    crashed = False
    try:
        logger.debug("RecorderIO: LOADING")
        recorder = _create_episode_recorder(shared, config)
        session = _RecorderIOSession.create(shared, config, recorder)
        _publish_status(shared, RecorderPhase.READY, 0, failure_count=0)
        logger.debug("RecorderIO: READY")
        shared.set_heartbeat("recorder", time.monotonic())
        shared.set_ready("recorder")
        # RecorderIO owns no actuator commands. Periodic batch persistence may
        # exceed one poll period without threatening data ownership; backlog,
        # sequence continuity, and writer failures are the actual boundaries.
        limiter = LoopRate(
            config.poll_hz,
            label="recorder",
            busy_wait=False,
            warn_on_overrun=False,
        )

        while session.should_run:
            session.step()
            limiter.wait()
    except Exception:
        crashed = True
        logger.error("RecorderIO process crashed", exc_info=True)
        if session is not None:
            session.failure_count += 1
            failure_count = session.failure_count
            active_generation = session.active_generation
        else:
            failure_count = 1
            active_generation = 0
        _publish_status(
            shared,
            RecorderPhase.ERROR,
            active_generation,
            error="RecorderIO process crashed",
            reason="process_crash",
            failure_count=failure_count,
        )
    finally:
        if recorder is not None:
            _shutdown_episode_recorder(recorder)
        active_generation = session.active_generation if session is not None else 0
        failure_count = session.failure_count if session is not None else int(crashed)
        _publish_status(
            shared,
            RecorderPhase.STOPPED,
            active_generation,
            failure_count=failure_count,
        )
        if not crashed:
            logger.debug("RecorderIO: STOPPED")
        logger.info("RecorderIO exited")
