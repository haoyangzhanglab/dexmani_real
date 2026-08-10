"""Dedicated RecorderIO process with a bounded shared-memory sample ring.

Policy owns episode boundaries, the 16 Hz grid, and sample contents.  This
module owns only serialization, camera encoding, HDF5 writes, verification and
transactional publication.  Large camera arrays never travel through an
``mp.Queue``; they occupy fixed slots in a seqlock ring.
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.ipc.schema import (
    RECORD_CONTROL_DTYPE,
    RECORD_CONTROL_JSON_BYTES,
    RECORD_SAMPLE_JSON_BYTES,
    RECORD_STATUS_DTYPE,
    RECORD_STATUS_TEXT_BYTES,
    make_record_sample_dtype,
)
from dexmani_real.recording.camera_stream_writer import CameraStreamWriterConfig
from dexmani_real.recording.episode_recorder import EpisodeRecorder, StopResult
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

CONTROL_JSON_BYTES = RECORD_CONTROL_JSON_BYTES
SAMPLE_JSON_BYTES = RECORD_SAMPLE_JSON_BYTES
STATUS_TEXT_BYTES = RECORD_STATUS_TEXT_BYTES


class RecorderCommand(IntEnum):
    START = 1
    STOP = 2


class RecorderPhase(IntEnum):
    INIT = 0
    READY = 1
    RECORDING = 2
    STOPPING = 3
    COMPLETED = 4
    ERROR = 5
    STOPPED = 6


@dataclass(frozen=True)
class RecorderIOConfig:
    data_dir: str
    max_frames: int
    control_hz: float
    min_frames: int
    resolved_config_json: str
    resolved_config_sha256: str
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
        provenance_names = [name for name, _value in self.provenance]
        if len(set(provenance_names)) != len(provenance_names):
            raise ValueError("RecorderIO provenance keys must be unique")
        if any(not name or not value for name, value in self.provenance):
            raise ValueError("RecorderIO provenance keys and values must be non-empty")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"RecorderIO JSON cannot encode {type(value).__name__}")


def _encode_json(value: Any, *, capacity: int) -> tuple[bytes, int]:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    if len(payload) > capacity:
        raise ValueError(f"RecorderIO JSON payload {len(payload)} exceeds fixed capacity {capacity}")
    return payload, zlib.crc32(payload) & 0xFFFFFFFF


def _decode_json(record: np.void, *, capacity: int) -> dict[str, Any]:
    length = int(record["json_length"])
    if length < 0 or length > capacity:
        raise ValueError("RecorderIO JSON length is out of bounds")
    payload = bytes(record["json_payload"])[:length]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != int(record["json_crc32"]):
        raise ValueError("RecorderIO JSON CRC mismatch")
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError("RecorderIO JSON root must be an object")
    return decoded


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
) -> None:
    frame = np.zeros(1, dtype=RECORD_STATUS_DTYPE)
    error_bytes, error_length = _bounded_text(error)
    path_bytes, path_length = _bounded_text(path)
    frame["phase"][0] = int(phase)
    frame["saved"][0] = int(saved)
    frame["generation"][0] = generation
    frame["frame_count"][0] = frame_count
    frame["updated_monotonic_ns"][0] = time.monotonic_ns()
    frame["error_length"][0] = error_length
    frame["error"][0] = error_bytes
    frame["path_length"][0] = path_length
    frame["path"][0] = path_bytes
    shared.record_status_ring.write(frame)


class RecorderClient:
    """Policy-side owner of recording decisions and fixed sample construction."""

    def __init__(self, shared: Any) -> None:
        self.shared = shared
        self._generation = 0
        self._frame_count = 0
        self._recording = False
        self._stop_requested = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def camera_writer_error(self) -> str | None:
        status = self._read_status()
        if status is None or int(status["generation"]) != self._generation:
            return None
        return self._text(status, "error") if int(status["phase"]) == int(RecorderPhase.ERROR) else None

    def _read_status(self) -> np.void | None:
        result = self.shared.record_status_ring.read_latest()
        return result[0][0] if result is not None else None

    @staticmethod
    def _text(status: np.void, field: str) -> str:
        length = int(status[f"{field}_length"])
        return bytes(status[field])[:length].decode("utf-8", errors="replace")

    def _write_control(self, command: RecorderCommand, payload: dict[str, Any], *, save: bool = False) -> None:
        encoded, crc = _encode_json(payload, capacity=CONTROL_JSON_BYTES)
        frame = np.zeros(1, dtype=RECORD_CONTROL_DTYPE)
        frame["command"][0] = int(command)
        frame["generation"][0] = self._generation
        frame["save"][0] = int(save)
        frame["created_monotonic_ns"][0] = time.monotonic_ns()
        frame["json_length"][0] = len(encoded)
        frame["json_crc32"][0] = crc
        frame["json_payload"][0] = encoded
        self.shared.record_control_ring.write(frame)

    def start_episode(self, **kwargs: Any) -> bool:
        if self._recording or not self.shared.recorder_ready.is_set():
            return False
        self._generation += 1
        metadata = dict(kwargs)
        calib = metadata.pop("calib", None)
        camera_name = metadata.get("camera_name")
        camera_serial = metadata.get("camera_serial")
        if calib is not None and camera_name is not None:
            calib_metadata = calib.to_meta_dict(camera_name, expected_serial=camera_serial)
            camera_metadata = dict(metadata.get("camera_metadata") or {})
            camera_metadata.update(calib_metadata)
            metadata["camera_metadata"] = camera_metadata
        self._write_control(RecorderCommand.START, metadata)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and self.shared.is_running.value:
            status = self._read_status()
            if status is not None and int(status["generation"]) == self._generation:
                phase = RecorderPhase(int(status["phase"]))
                if phase is RecorderPhase.RECORDING:
                    self._recording = True
                    self._stop_requested = False
                    self._frame_count = 0
                    return True
                if phase is RecorderPhase.ERROR:
                    return False
            time.sleep(0.005)
        return False

    def add_frame(
        self,
        state: RobotState,
        action: RobotAction,
        vr_frame: dict[str, Any],
        camera_frame: dict[str, Any] | None = None,
        signals: dict[str, Any] | None = None,
        arm_qpos_sent: np.ndarray | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> bool:
        if not self._recording:
            return False
        latest = int(self.shared.record_sample_ring.latest_sequence)
        consumed = int(self.shared.recorder_consumed_sequence.value)
        if latest - consumed >= self.shared.record_sample_ring.maxlen:
            logger.error("RecorderIO sample ring overflow — aborting episode")
            self.stop_episode(success=False, reason="sample_ring_overflow")
            return False

        dtype = self.shared.record_sample_ring.dtype
        frame = np.zeros(1, dtype=dtype)
        frame["generation"][0] = self._generation
        frame["sample_sequence"][0] = self._frame_count + 1
        for name in ("arm_qpos", "arm_qvel", "arm_tau", "eef_pos", "eef_quat_wxyz", "eef_rot6d", "hand_qpos"):
            frame[name][0] = getattr(state, name)
        frame["hand_current"][0] = state.hand_current if state.hand_current is not None else np.full(12, np.nan)
        for name in (
            "hand_tactile_sum",
            "hand_tactile_force",
            "hand_tactile_contact",
            "hand_tipboard_err",
            "hand_commboard_err",
            "hand_jointboard_err",
            "fingertip_pos",
        ):
            frame[name][0] = getattr(state, name)
        for name in ("hand_qpos_stale", "arm_connected", "hand_connected", "hand_error_state", "arm_last_cmd_is_hold"):
            frame[name][0] = int(bool(getattr(state, name)))
        frame["state_timestamp"][0] = state.timestamp
        frame["arm_last_cmd_seq"][0] = state.arm_last_cmd_seq
        frame["arm_last_cmd_queue_latency_s"][0] = state.arm_last_cmd_queue_latency_s
        frame["arm_last_cmd_apply_latency_s"][0] = state.arm_last_cmd_apply_latency_s
        frame["arm_last_cmd_sdk_duration_s"][0] = state.arm_last_cmd_sdk_duration_s
        frame["action_arm_qpos"][0] = action.arm_qpos_cmd
        frame["action_hand_qpos"][0] = action.hand_qpos_cmd
        frame["action_target_eef_pos"][0] = (
            action.target_eef_pos if action.target_eef_pos is not None else np.full(3, np.nan)
        )
        frame["action_target_eef_rot6d"][0] = (
            action.target_eef_rot6d if action.target_eef_rot6d is not None else np.full(6, np.nan)
        )
        frame["vr_wrist_pos"][0] = vr_frame["wrist_pos"]
        frame["vr_wrist_quat_wxyz"][0] = vr_frame["wrist_quat_wxyz"]
        frame["vr_landmarks"][0] = vr_frame["landmarks"]
        frame["vr_head_quat_wxyz"][0] = vr_frame.get("head_quat_wxyz", np.full(4, np.nan))
        if camera_frame is not None:
            frame["camera_present"][0] = 1
            frame["camera_rgb"][0] = camera_frame.get("rgb", np.zeros(frame["camera_rgb"][0].shape, np.uint8))
            frame["camera_depth"][0] = camera_frame.get("depth", np.zeros(frame["camera_depth"][0].shape, np.uint16))
            frame["camera_pointcloud"][0] = camera_frame.get(
                "pointcloud", np.zeros(frame["camera_pointcloud"][0].shape, np.float32)
            )
        small_payload = {
            "camera": {
                key: value
                for key, value in (camera_frame or {}).items()
                if key not in {"header", "rgb", "depth", "pointcloud"}
            },
            "signals": signals or {},
            "arm_qpos_sent": arm_qpos_sent,
            "diagnostics": diagnostics or {},
        }
        encoded, crc = _encode_json(small_payload, capacity=SAMPLE_JSON_BYTES)
        frame["json_length"][0] = len(encoded)
        frame["json_crc32"][0] = crc
        frame["json_payload"][0] = encoded
        self.shared.record_sample_ring.write(frame)
        self._frame_count += 1
        return True

    def stop_episode(self, success: bool = True, reason: str = "") -> str | None:
        if not self._recording or self._stop_requested:
            return None
        self._write_control(RecorderCommand.STOP, {"reason": reason}, save=success)
        self._recording = False
        self._stop_requested = True
        return None

    def poll_stop(self) -> StopResult:
        if not self._stop_requested:
            return StopResult(done=False)
        status = self._read_status()
        if status is None or int(status["generation"]) != self._generation:
            return StopResult(done=False)
        phase = RecorderPhase(int(status["phase"]))
        if phase not in (RecorderPhase.COMPLETED, RecorderPhase.ERROR):
            return StopResult(done=False)
        self._stop_requested = False
        error = self._text(status, "error") or None
        path = self._text(status, "path") or None
        return StopResult(
            done=True,
            error=error,
            success=phase is RecorderPhase.COMPLETED and error is None and bool(status["saved"]),
            path=path,
            frame_count=int(status["frame_count"]),
        )

    def join_stop(self, timeout: float | None = None) -> bool:
        if not self._stop_requested:
            return True
        deadline = time.monotonic() + (60.0 if timeout is None else float(timeout))
        while time.monotonic() < deadline:
            status = self._read_status()
            if status is not None and int(status["generation"]) == self._generation:
                if RecorderPhase(int(status["phase"])) in (RecorderPhase.COMPLETED, RecorderPhase.ERROR):
                    self._stop_requested = False
                    return True
            time.sleep(0.01)
        return False


def _unpack_sample(record: np.void) -> tuple[RobotState, RobotAction, dict[str, Any], dict[str, Any]]:
    payload = _decode_json(record, capacity=SAMPLE_JSON_BYTES)
    state = RobotState(
        arm_qpos=np.array(record["arm_qpos"], copy=True),
        arm_qvel=np.array(record["arm_qvel"], copy=True),
        arm_tau=np.array(record["arm_tau"], copy=True),
        eef_pos=np.array(record["eef_pos"], copy=True),
        eef_quat_wxyz=np.array(record["eef_quat_wxyz"], copy=True),
        eef_rot6d=np.array(record["eef_rot6d"], copy=True),
        hand_qpos=np.array(record["hand_qpos"], copy=True),
        hand_tactile_sum=np.array(record["hand_tactile_sum"], copy=True),
        hand_tactile_force=np.array(record["hand_tactile_force"], copy=True),
        hand_tactile_contact=np.asarray(record["hand_tactile_contact"], dtype=bool),
        hand_tipboard_err=np.array(record["hand_tipboard_err"], copy=True),
        hand_commboard_err=np.array(record["hand_commboard_err"], copy=True),
        hand_jointboard_err=np.array(record["hand_jointboard_err"], copy=True),
        hand_qpos_stale=bool(record["hand_qpos_stale"]),
        fingertip_pos=np.array(record["fingertip_pos"], copy=True),
        arm_connected=bool(record["arm_connected"]),
        hand_connected=bool(record["hand_connected"]),
        timestamp=float(record["state_timestamp"]),
        hand_current=np.array(record["hand_current"], copy=True),
        hand_error_state=bool(record["hand_error_state"]),
        arm_last_cmd_seq=int(record["arm_last_cmd_seq"]),
        arm_last_cmd_queue_latency_s=float(record["arm_last_cmd_queue_latency_s"]),
        arm_last_cmd_apply_latency_s=float(record["arm_last_cmd_apply_latency_s"]),
        arm_last_cmd_sdk_duration_s=float(record["arm_last_cmd_sdk_duration_s"]),
        arm_last_cmd_is_hold=bool(record["arm_last_cmd_is_hold"]),
    )
    action = RobotAction(
        arm_qpos_cmd=np.array(record["action_arm_qpos"], copy=True),
        hand_qpos_cmd=np.array(record["action_hand_qpos"], copy=True),
        target_eef_pos=np.array(record["action_target_eef_pos"], copy=True),
        target_eef_rot6d=np.array(record["action_target_eef_rot6d"], copy=True),
    )
    vr_frame = {
        "wrist_pos": np.array(record["vr_wrist_pos"], copy=True),
        "wrist_quat_wxyz": np.array(record["vr_wrist_quat_wxyz"], copy=True),
        "landmarks": np.array(record["vr_landmarks"], copy=True),
        "head_quat_wxyz": np.array(record["vr_head_quat_wxyz"], copy=True),
    }
    camera_frame = dict(payload.get("camera") or {})
    if bool(record["camera_present"]):
        camera_frame.update(
            rgb=np.array(record["camera_rgb"], copy=True),
            depth=np.array(record["camera_depth"], copy=True),
            pointcloud=np.array(record["camera_pointcloud"], copy=True),
        )
    return state, action, vr_frame, {**payload, "camera": camera_frame}


def recorder_io_loop(shared: Any, config: RecorderIOConfig) -> None:
    """Long-lived process target. Recording errors never latch robot FAULT."""
    from dexmani_real.runtime.status import ComponentPhase, FaultCode
    from dexmani_real.shm.shared_storage import publish_component_status

    recorder: EpisodeRecorder | None = None
    active_generation = 0
    last_control_sequence = 0
    last_sample_sequence = int(shared.recorder_consumed_sequence.value)
    pending_stop: tuple[int, bool, str] | None = None
    crashed = False
    try:
        publish_component_status(shared, "recorder", ComponentPhase.LOADING)
        sample_dtype = shared.record_sample_ring.dtype
        rgb_dims = sample_dtype.fields["camera_rgb"][0].shape
        depth_dims = sample_dtype.fields["camera_depth"][0].shape
        pointcloud_dims = sample_dtype.fields["camera_pointcloud"][0].shape
        rgb_shape = (int(rgb_dims[0]), int(rgb_dims[1]), int(rgb_dims[2]))
        depth_shape = (int(depth_dims[0]), int(depth_dims[1]))
        pointcloud_shape = (int(pointcloud_dims[0]), int(pointcloud_dims[1]))
        recorder = EpisodeRecorder(
            data_dir=config.data_dir,
            max_frames=config.max_frames,
            control_hz=config.control_hz,
            min_frames=config.min_frames,
            arm_sent_stream=True,
            resolved_config_json=config.resolved_config_json,
            resolved_config_hash=config.resolved_config_sha256,
            provenance=dict(config.provenance),
            camera_writer_config=CameraStreamWriterConfig(
                rgb_shape=rgb_shape,
                depth_shape=depth_shape,
                pointcloud_shape=pointcloud_shape,
                fps=config.control_hz,
                queue_size=config.writer_queue_size,
            ),
        )
        _publish_status(shared, RecorderPhase.READY, 0)
        publish_component_status(shared, "recorder", ComponentPhase.READY)
        shared.recorder_heartbeat_s.value = time.monotonic()
        shared.recorder_ready.set()
        limiter = RateManager(config.poll_hz)

        while shared.is_running.value or recorder.is_recording:
            shared.recorder_heartbeat_s.value = time.monotonic()
            control_result = shared.record_control_ring.read_latest()
            control = None
            control_sequence = last_control_sequence
            if control_result is not None:
                control_data, _publish_ns, control_sequence = control_result
                if control_sequence != last_control_sequence:
                    control = control_data[0]
                    last_control_sequence = control_sequence

            if control is not None and RecorderCommand(int(control["command"])) is RecorderCommand.START:
                generation = int(control["generation"])
                try:
                    metadata = _decode_json(control, capacity=CONTROL_JSON_BYTES)
                    camera_K = metadata.get("camera_K")
                    if camera_K is not None:
                        metadata["camera_K"] = np.asarray(camera_K, dtype=np.float64)
                    if not recorder.start_episode(**metadata):
                        raise RuntimeError("EpisodeRecorder refused start")
                    active_generation = generation
                    pending_stop = None
                    _publish_status(shared, RecorderPhase.RECORDING, generation)
                except Exception as exc:
                    logger.error("RecorderIO start failed", exc_info=True)
                    _publish_status(shared, RecorderPhase.ERROR, generation, error=str(exc))

            # Drain all still-resident samples oldest-first before honoring STOP.
            samples = shared.record_sample_ring.get_last_k(shared.record_sample_ring.maxlen)
            unseen = [(data, seq) for data, _publish_ns, seq in samples if seq > last_sample_sequence]
            if unseen and unseen[0][1] > last_sample_sequence + 1 and recorder.is_recording:
                reason = f"sample ring overflow: expected {last_sample_sequence + 1}, found {unseen[0][1]}"
                logger.error("RecorderIO %s", reason)
                recorder.stop_episode(success=False, reason="sample_ring_overflow")
                recorder.join_stop(timeout=60.0)
                _publish_status(shared, RecorderPhase.ERROR, active_generation, error=reason)
            for data, sequence in unseen:
                last_sample_sequence = sequence
                shared.recorder_consumed_sequence.value = sequence
                record = data[0]
                if not recorder.is_recording or int(record["generation"]) != active_generation:
                    continue
                try:
                    state, action, vr_frame, payload = _unpack_sample(record)
                    camera_frame = payload["camera"] or None
                    if (
                        not recorder.add_frame(
                            state,
                            action,
                            vr_frame,
                            camera_frame=camera_frame,
                            signals=payload.get("signals") or {},
                            arm_qpos_sent=(
                                np.asarray(payload["arm_qpos_sent"], dtype=np.float64)
                                if payload.get("arm_qpos_sent") is not None
                                else None
                            ),
                            diagnostics=payload.get("diagnostics") or {},
                        )
                        and recorder.camera_writer_error
                    ):
                        raise RuntimeError(recorder.camera_writer_error)
                except Exception as exc:
                    logger.error("RecorderIO sample write failed", exc_info=True)
                    recorder.stop_episode(success=False, reason="sample_write_error")
                    recorder.join_stop(timeout=60.0)
                    _publish_status(shared, RecorderPhase.ERROR, active_generation, error=str(exc))
                    break

            if control is not None and RecorderCommand(int(control["command"])) is RecorderCommand.STOP:
                generation = int(control["generation"])
                metadata = _decode_json(control, capacity=CONTROL_JSON_BYTES)
                pending_stop = (generation, bool(control["save"]), str(metadata.get("reason", "")))

            # An unexpected policy/main shutdown may arrive before the policy
            # can publish STOP.  Do not keep the RecorderIO child alive merely
            # because an episode is active: drain every resident sample, close
            # all codecs/HDF5 handles, and retain only the aborted manifest.
            if not shared.is_running.value and recorder.is_recording and pending_stop is None:
                pending_stop = (active_generation, False, "runtime_shutdown")

            if pending_stop is not None and pending_stop[0] == active_generation and recorder.is_recording:
                generation, save, reason = pending_stop
                completed_frame_count = recorder.frame_count
                _publish_status(shared, RecorderPhase.STOPPING, generation, frame_count=completed_frame_count)
                path = recorder.stop_episode(success=save, reason=reason)
                recorder.join_stop(timeout=60.0)
                error = recorder.stop_error or ""
                phase = RecorderPhase.ERROR if error else RecorderPhase.COMPLETED
                _publish_status(
                    shared,
                    phase,
                    generation,
                    frame_count=completed_frame_count,
                    error=error,
                    path=path or "",
                    saved=save and not error,
                )
                pending_stop = None

            limiter.wait()
    except Exception:
        crashed = True
        logger.error("RecorderIO process crashed", exc_info=True)
        _publish_status(shared, RecorderPhase.ERROR, active_generation, error="RecorderIO process crashed")
        publish_component_status(
            shared,
            "recorder",
            ComponentPhase.FAULT,
            fault_code=FaultCode.RECORDING_ABORTED,
            detail="RecorderIO process crashed; robot fault is not latched",
        )
    finally:
        if recorder is not None and recorder.is_recording:
            recorder.stop_episode(success=False, reason="recorder_process_shutdown")
            recorder.join_stop(timeout=60.0)
        _publish_status(shared, RecorderPhase.STOPPED, active_generation)
        if not crashed:
            publish_component_status(shared, "recorder", ComponentPhase.STOPPED)
        logger.info("RecorderIO exited")
