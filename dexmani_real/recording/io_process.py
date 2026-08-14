"""Dedicated RecorderIO process with a bounded shared-memory sample ring.

Policy owns episode boundaries, the configured control grid, and sample
contents. This module owns serialization, non-blocking finalization, camera
encoding, HDF5 writes, verification and transactional publication. Large
camera arrays never travel through an ``mp.Queue``; they occupy fixed slots in
a seqlock ring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.recording.camera_stream_writer import CameraStreamWriterConfig
from dexmani_real.recording.episode_recorder import EpisodeRecorder, StopResult as EpisodeStopResult
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.schema import (
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    RECORD_CONTROL_DTYPE,
    RECORD_OPERATOR_BYTES,
    RECORD_STATUS_DTYPE,
    RECORD_STATUS_TEXT_BYTES,
    RECORD_STOP_REASON_BYTES,
    RECORD_TASK_LABEL_BYTES,
)

logger = get_logger(__name__)

STATUS_TEXT_BYTES = RECORD_STATUS_TEXT_BYTES
_RECORDER_STOP_TIMEOUT_S = 60.0
_STOP_POLL_INTERVAL_S = 0.01


class RecorderCommand(IntEnum):
    START = 1
    STOP = 2


class RecorderPhase(IntEnum):
    READY = 1
    RECORDING = 2
    FINALIZING = 3
    COMPLETED = 4
    ERROR = 5
    STOPPED = 6


@dataclass(frozen=True)
class RecorderStopResult:
    """Policy-visible outcome of one recorder transaction."""

    done: bool
    phase: RecorderPhase | None = None
    generation: int = 0
    saved: bool = False
    error: str | None = None
    path: str | None = None
    frame_count: int = 0
    reason: str = ""
    min_frames_met: bool = False
    failure_count: int = 0

    @property
    def success(self) -> bool:
        """Compatibility alias: only a successfully published save is success."""
        return self.done and self.phase is RecorderPhase.COMPLETED and self.saved and self.error is None


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


def _bounded_control_text(value: str, *, capacity: int, field: str) -> bytes:
    """Encode a control-plane text field without an unbounded JSON side channel."""
    payload = value.encode("utf-8")
    if len(payload) > capacity:
        raise ValueError(f"RecorderIO {field} exceeds fixed capacity {capacity}")
    return payload


def _control_text(record: np.void, field: str) -> str:
    """Decode a null-padded fixed control-plane text field."""
    return bytes(record[field]).rstrip(b"\x00").decode("utf-8", errors="replace")


def _shared_text(value: bytes, *, default: str | None) -> str | None:
    encoded = value.rstrip(b"\x00")
    return encoded.decode("utf-8") if encoded else default


def _build_start_metadata(shared: Any, *, task_label: str, operator: str) -> dict[str, Any]:
    """Snapshot only essential recording metadata at the immutable START boundary."""
    camera_K_values = list(shared.camera_K)
    camera_K = (
        np.asarray(camera_K_values, dtype=np.float64).reshape(3, 3)
        if any(value != 0.0 for value in camera_K_values)
        else None
    )
    depth_scale = float(shared.camera_depth_scale.value) if shared.camera_depth_scale.value != 0.0 else None
    camera_serial = _shared_text(shared.camera_serial.value, default=None)
    camera_firmware = _shared_text(shared.camera_firmware.value, default="unknown") or "unknown"
    camera_sdk_version = _shared_text(shared.camera_sdk_version.value, default="unknown") or "unknown"
    camera_profile_json = _shared_text(shared.camera_profile.value, default="{}") or "{}"
    arm_identity_json = (
        _shared_text(shared.arm_device_identity.value, default='{"status":"unavailable"}')
        or '{"status":"unavailable"}'
    )
    hand_identity_json = _shared_text(shared.hand_device_identity.value, default='{"status":"unavailable"}')
    calibration = CameraCalib()
    try:
        camera_name = calibration.resolve_name_by_serial(camera_serial) if camera_serial else None
    except (KeyError, FileNotFoundError):
        camera_name = None
        logger.warning("Camera serial %s not found in cameras.json — no extrinsics in /meta", camera_serial)

    return {
        "task_label": task_label,
        "operator": operator,
        "calib": calibration,
        "camera_K": camera_K,
        "camera_name": camera_name,
        "camera_serial": camera_serial,
        "depth_scale": depth_scale,
        "camera_metadata": {
            "camera_firmware": camera_firmware,
            "camera_sdk_version": camera_sdk_version,
            "camera_actual_profile_json": camera_profile_json,
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
    reason_bytes = _bounded_control_text(reason, capacity=RECORD_STOP_REASON_BYTES, field="status reason")
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


def _write_sample_metadata(
    frame: np.ndarray,
    *,
    action: RobotAction,
    camera_frame: dict[str, Any] | None,
    signals: dict[str, Any] | None,
    arm_qpos_sent: np.ndarray | None,
    diagnostics: dict[str, Any] | None,
) -> None:
    """Populate the fixed recorder metadata fields at the policy/IO boundary."""
    signal_data = signals or {}
    diagnostic_data = diagnostics or {}

    frame["arm_qpos_sent"][0] = (
        np.asarray(arm_qpos_sent, dtype=np.float64)
        if arm_qpos_sent is not None
        else np.full(ARM_JOINT_SHAPE, np.nan)
    )
    uint64_fields = (
        "observation_id",
        "observation_anchor_monotonic_ns",
        "arm_source_sequence",
        "hand_source_sequence",
        "vr_source_sequence",
        "camera_source_sequence",
        "arm_source_monotonic_ns",
        "hand_source_monotonic_ns",
        "vr_source_monotonic_ns",
        "camera_source_monotonic_ns",
        "arm_publish_monotonic_ns",
        "hand_publish_monotonic_ns",
        "vr_publish_monotonic_ns",
        "camera_publish_monotonic_ns",
        "action_id",
        "action_created_monotonic_ns",
        "action_target_monotonic_ns",
        "action_valid_until_monotonic_ns",
        "tactile_source_monotonic_ns",
    )
    for name in uint64_fields:
        frame[name][0] = int(signal_data.get(name, 0))
    bool_fields = {
        "observation_valid": "observation_valid",
        "flag_action_queued": "action_queued",
        "tactile_fresh": "tactile_fresh",
        "tactile_calibrated": "tactile_calibrated",
        "flag_ik_ok": "ik_ok",
        "flag_ik_attempted": "ik_attempted",
        "flag_retarget_ok": "retarget_ok",
        "flag_held": "held",
        "flag_safety_reject": "flag_safety_reject",
    }
    for field_name, signal_name in bool_fields.items():
        default = field_name == "flag_ik_attempted"
        frame[field_name][0] = int(bool(signal_data.get(signal_name, default)))
    frame["tactile_unit_code"][0] = int(signal_data.get("tactile_unit_code", 0))
    frame["pointcloud_source_point_count"][0] = int(signal_data.get("pointcloud_source_point_count", 0))
    frame["pointcloud_padding_count"][0] = int(signal_data.get("pointcloud_padding_count", 0))
    frame["flag_frame_status"][0] = int(signal_data.get("frame_status", 0))
    frame["observation_source_receive_monotonic_ns"][0] = np.asarray(
        signal_data.get("observation_source_receive_monotonic_ns", np.zeros(4)), dtype=np.uint64
    )
    frame["observation_source_age_s"][0] = np.asarray(
        signal_data.get("observation_source_age_s", np.full(4, np.nan)), dtype=np.float64
    )
    frame["observation_source_skew_s"][0] = np.asarray(
        signal_data.get("observation_source_skew_s", np.full(4, np.nan)), dtype=np.float64
    )
    frame["observation_history_valid_mask"][0] = np.asarray(
        signal_data.get("observation_history_valid_mask", np.zeros((4, 1), dtype=bool)), dtype=np.uint8
    )
    for name in (
        "observation_skew_s",
        "pointcloud_valid_depth_ratio",
    ):
        frame[name][0] = float(signal_data.get(name, np.nan))
    frame["action_arm_joint_raw"][0] = np.asarray(
        signal_data.get("action_arm_joint_raw", action.arm_qpos_cmd), dtype=np.float64
    )
    frame["action_hand_joint_raw"][0] = np.asarray(
        diagnostic_data.get("action_hand_joint_raw", action.hand_qpos_cmd), dtype=np.float64
    )

    cam = camera_frame or {}
    frame["camera_health"][0] = int(cam.get("camera_health", 1))
    frame["camera_fresh"][0] = int(bool(cam.get("camera_fresh", False)))
    frame["pointcloud_valid"][0] = int(bool(cam.get("pointcloud_valid", False)))
    camera_integer_fields = {
        "camera_frame_number": "frame_number",
        "camera_ring_sequence": "ring_sequence",
        "camera_generation": "camera_generation",
    }
    for field_name, camera_name in camera_integer_fields.items():
        frame[field_name][0] = int(cam.get(camera_name, 0))
    frame["camera_clock_reset"][0] = int(bool(cam.get("clock_reset", False)))
    frame["camera_duplicate"][0] = int(bool(cam.get("duplicate", False)))
    frame["camera_frame_gap"][0] = int(cam.get("frame_gap", 0))
    for name, source_name in (
        ("camera_device_timestamp_s", "device_timestamp_s"),
        ("camera_capture_monotonic_s", "capture_monotonic_s"),
        ("camera_age_s", "camera_age_s"),
        ("camera_backlog_s", "backlog_s"),
    ):
        frame[name][0] = float(cam.get(source_name, np.nan))

    for name in (
        "tracking_error",
        "ik_solve_time_ms",
        "policy_map_time_ms",
        "hand_retarget_time_ms",
        "transition_check_time_ms",
        "policy_compute_time_ms",
    ):
        frame[name][0] = float(diagnostic_data.get(name, np.nan))
    for name, shape in (
        ("target_pos_before_clamp", (3,)),
        ("head_quat_wxyz", (4,)),
        ("target_eef_pos_raw", (3,)),
        ("target_eef_rot6d_raw", (6,)),
    ):
        frame[name][0] = np.asarray(diagnostic_data.get(name, np.full(shape, np.nan)), dtype=np.float64)


class RecorderClient:
    """Policy-side owner of recording decisions and fixed sample construction."""

    def __init__(self, shared: Any) -> None:
        self.shared = shared
        self._generation = 0
        self._frame_count = 0
        self._recording = False
        self._stop_requested = False
        self._last_poll_status_sequence = 0
        self._last_stop_result: RecorderStopResult | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def stop_pending(self) -> bool:
        return self._stop_requested

    @property
    def last_stop_result(self) -> RecorderStopResult | None:
        return self._last_stop_result

    @property
    def camera_writer_error(self) -> str | None:
        status = self._read_status()
        if status is None or int(status["generation"]) != self._generation:
            return None
        return self._text(status, "error") if int(status["phase"]) == int(RecorderPhase.ERROR) else None

    def _read_status_with_sequence(self) -> tuple[np.void, int] | None:
        result = self.shared.record_status_ring.read_latest()
        return (result[0][0], int(result[2])) if result is not None else None

    def _read_status(self) -> np.void | None:
        result = self._read_status_with_sequence()
        return result[0] if result is not None else None

    @staticmethod
    def _text(status: np.void, field: str) -> str:
        length = int(status[f"{field}_length"])
        return bytes(status[field])[:length].decode("utf-8", errors="replace")

    def _write_control(
        self,
        command: RecorderCommand,
        *,
        save: bool = False,
        task_label: str = "",
        operator: str = "",
        stop_reason: str = "",
    ) -> None:
        frame = np.zeros(1, dtype=RECORD_CONTROL_DTYPE)
        frame["command"][0] = int(command)
        frame["generation"][0] = self._generation
        frame["save"][0] = int(save)
        frame["created_monotonic_ns"][0] = time.monotonic_ns()
        frame["task_label"][0] = _bounded_control_text(
            task_label, capacity=RECORD_TASK_LABEL_BYTES, field="task_label"
        )
        frame["operator"][0] = _bounded_control_text(operator, capacity=RECORD_OPERATOR_BYTES, field="operator")
        frame["stop_reason"][0] = _bounded_control_text(
            stop_reason, capacity=RECORD_STOP_REASON_BYTES, field="stop_reason"
        )
        self.shared.record_control_ring.write(frame)

    def start_episode(self, *, task_label: str = "", operator: str = "") -> bool:
        if self._recording or self._stop_requested or not self.shared.is_ready("recorder"):
            return False
        try:
            _bounded_control_text(task_label, capacity=RECORD_TASK_LABEL_BYTES, field="task_label")
            _bounded_control_text(operator, capacity=RECORD_OPERATOR_BYTES, field="operator")
        except ValueError:
            logger.error("RecorderIO start metadata exceeds its fixed control boundary", exc_info=True)
            return False
        self._generation += 1
        self._last_stop_result = None
        self._write_control(RecorderCommand.START, task_label=task_label, operator=operator)
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
            # Starting is bounded but may span more than one supervisor tick.
            # RecorderClient is policy-owned, so keep that owner's heartbeat live.
            self.shared.set_heartbeat("policy", time.monotonic())
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
        frame["control_run_generation"][0] = int(self.shared.run_generation.value)
        frame["sample_sequence"][0] = self._frame_count + 1
        for name in ("arm_qpos", "arm_qvel", "arm_tau", "eef_pos", "eef_quat_wxyz", "eef_rot6d", "hand_qpos"):
            frame[name][0] = getattr(state, name)
        frame["hand_current"][0] = (
            state.hand_current if state.hand_current is not None else np.full(HAND_JOINT_SHAPE, np.nan)
        )
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
        _write_sample_metadata(
            frame,
            action=action,
            camera_frame=camera_frame,
            signals=signals,
            arm_qpos_sent=arm_qpos_sent,
            diagnostics=diagnostics,
        )
        self.shared.record_sample_ring.write(frame)
        self._frame_count += 1
        return True

    def stop_episode(self, success: bool = True, reason: str = "") -> str | None:
        if not self._recording or self._stop_requested:
            return None
        self._write_control(RecorderCommand.STOP, save=success, stop_reason=reason)
        self._recording = False
        self._stop_requested = True
        return None

    def _status_result(self, status: np.void, *, done: bool) -> RecorderStopResult:
        phase = RecorderPhase(int(status["phase"]))
        error = self._text(status, "error") or None
        path = self._text(status, "path") or None
        return RecorderStopResult(
            done=done,
            phase=phase,
            generation=int(status["generation"]),
            saved=phase is RecorderPhase.COMPLETED and error is None and bool(status["saved"]),
            error=error,
            path=path,
            frame_count=int(status["frame_count"]),
            reason=self._text(status, "reason"),
            min_frames_met=bool(status["min_frames_met"]),
            failure_count=int(status["failure_count"]),
        )

    def poll_stop(self) -> RecorderStopResult:
        """Return each newly published recorder status once, including max-stop."""
        status_result = self._read_status_with_sequence()
        if status_result is None:
            return RecorderStopResult(done=False)
        status, sequence = status_result
        if sequence == self._last_poll_status_sequence or int(status["generation"]) != self._generation:
            return RecorderStopResult(done=False)
        self._last_poll_status_sequence = sequence
        phase = RecorderPhase(int(status["phase"]))
        if phase is RecorderPhase.FINALIZING:
            self._recording = False
            self._stop_requested = True
            return self._status_result(status, done=False)
        if phase not in (RecorderPhase.COMPLETED, RecorderPhase.ERROR):
            return RecorderStopResult(done=False, phase=phase, generation=self._generation)
        self._recording = False
        self._stop_requested = False
        result = self._status_result(status, done=True)
        self._last_stop_result = result
        return result

    def join_stop(self, timeout: float | None = None) -> RecorderStopResult:
        """Wait for a terminal status without conflating ERROR with success."""
        if not self._stop_requested:
            return self._last_stop_result or RecorderStopResult(done=True)
        timeout_s = _RECORDER_STOP_TIMEOUT_S if timeout is None else float(timeout)
        if not np.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("recorder stop timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self._read_status()
            if status is not None and int(status["generation"]) == self._generation:
                phase = RecorderPhase(int(status["phase"]))
                if phase in (RecorderPhase.COMPLETED, RecorderPhase.ERROR):
                    self._recording = False
                    self._stop_requested = False
                    result = self._status_result(status, done=True)
                    self._last_stop_result = result
                    return result
            self.shared.set_heartbeat("policy", time.monotonic())
            time.sleep(_STOP_POLL_INTERVAL_S)
        return RecorderStopResult(
            done=False,
            phase=RecorderPhase.FINALIZING,
            generation=self._generation,
            reason="finalization_timeout",
        )


def _unpack_sample(
    record: np.void,
) -> tuple[
    RobotState,
    RobotAction,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    dict[str, Any],
]:
    """Copy one fixed shared-memory sample into RecorderIO-owned values."""
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
    camera_frame: dict[str, Any] = {
        "camera_health": int(record["camera_health"]),
        "camera_fresh": bool(record["camera_fresh"]),
        "pointcloud_valid": bool(record["pointcloud_valid"]),
        "frame_number": int(record["camera_frame_number"]),
        "ring_sequence": int(record["camera_ring_sequence"]),
        "device_timestamp_s": float(record["camera_device_timestamp_s"]),
        "capture_monotonic_s": float(record["camera_capture_monotonic_s"]),
        "camera_age_s": float(record["camera_age_s"]),
        "camera_generation": int(record["camera_generation"]),
        "clock_reset": bool(record["camera_clock_reset"]),
        "duplicate": bool(record["camera_duplicate"]),
        "frame_gap": int(record["camera_frame_gap"]),
        "backlog_s": float(record["camera_backlog_s"]),
    }
    if bool(record["camera_present"]):
        camera_frame.update(
            rgb=np.array(record["camera_rgb"], copy=True),
            depth=np.array(record["camera_depth"], copy=True),
            pointcloud=np.array(record["camera_pointcloud"], copy=True),
        )
    signal_names = (
        "observation_id",
        "observation_anchor_monotonic_ns",
        "arm_source_sequence",
        "hand_source_sequence",
        "vr_source_sequence",
        "camera_source_sequence",
        "arm_source_monotonic_ns",
        "hand_source_monotonic_ns",
        "vr_source_monotonic_ns",
        "camera_source_monotonic_ns",
        "arm_publish_monotonic_ns",
        "hand_publish_monotonic_ns",
        "vr_publish_monotonic_ns",
        "camera_publish_monotonic_ns",
        "observation_source_receive_monotonic_ns",
        "observation_source_age_s",
        "observation_source_skew_s",
        "observation_history_valid_mask",
        "observation_valid",
        "observation_skew_s",
        "action_id",
        "action_created_monotonic_ns",
        "action_target_monotonic_ns",
        "action_valid_until_monotonic_ns",
        "action_arm_joint_raw",
        "tactile_fresh",
        "tactile_source_monotonic_ns",
        "tactile_calibrated",
        "tactile_unit_code",
        "pointcloud_source_point_count",
        "pointcloud_valid_depth_ratio",
        "pointcloud_padding_count",
        "flag_ik_ok",
        "flag_ik_attempted",
        "flag_retarget_ok",
        "flag_held",
        "flag_safety_reject",
    )

    def _copy_field(name: str) -> Any:
        value = record[name]
        return np.array(value, copy=True) if np.asarray(value).ndim else value.item()

    signals = {name: _copy_field(name) for name in signal_names}
    signals["action_queued"] = bool(record["flag_action_queued"])
    signals["frame_status"] = int(record["flag_frame_status"])
    diagnostics = {
        name: _copy_field(name)
        for name in (
            "tracking_error",
            "ik_solve_time_ms",
            "target_pos_before_clamp",
            "head_quat_wxyz",
            "target_eef_pos_raw",
            "target_eef_rot6d_raw",
            "action_hand_joint_raw",
            "policy_map_time_ms",
            "hand_retarget_time_ms",
            "transition_check_time_ms",
            "policy_compute_time_ms",
        )
    }
    return state, action, vr_frame, camera_frame, signals, np.array(record["arm_qpos_sent"], copy=True), diagnostics


def recorder_io_loop(shared: Any, config: RecorderIOConfig) -> None:
    """Long-lived process target. Recording errors never latch robot FAULT."""
    from dexmani_real.runtime.status import ComponentPhase, FaultCode
    from dexmani_real.shm.shared_storage import publish_component_status

    recorder: EpisodeRecorder | None = None
    active_generation = 0
    last_control_sequence = 0
    last_sample_sequence = int(shared.recorder_consumed_sequence.value)
    pending_finalization: _PendingFinalization | None = None
    failure_count = 0
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
        _publish_status(shared, RecorderPhase.READY, 0, failure_count=failure_count)
        publish_component_status(shared, "recorder", ComponentPhase.READY)
        shared.set_heartbeat("recorder", time.monotonic())
        shared.set_ready("recorder")
        limiter = RateManager(config.poll_hz)

        def _begin_finalization(
            *,
            generation: int,
            save: bool,
            reason: str,
            forced_error: str = "",
        ) -> None:
            nonlocal pending_finalization
            if pending_finalization is not None:
                return
            if not recorder.is_recording:
                return
            frame_count = recorder.frame_count
            path = recorder.stop_episode(success=save, reason=reason) or ""
            pending_finalization = _PendingFinalization(
                generation=generation,
                save=save,
                reason=reason,
                path=path,
                frame_count=frame_count,
                started_monotonic_s=time.monotonic(),
                forced_error=forced_error,
            )
            _publish_status(
                shared,
                RecorderPhase.FINALIZING,
                generation,
                frame_count=frame_count,
                path=path,
                reason=reason,
                min_frames_met=frame_count >= config.min_frames,
                failure_count=failure_count,
            )

        while shared.is_running.value or recorder.is_recording or pending_finalization is not None:
            shared.set_heartbeat("recorder", time.monotonic())
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
                    if recorder.is_recording or pending_finalization is not None:
                        raise RuntimeError("previous recorder transaction is still active")
                    if generation <= active_generation:
                        raise RuntimeError("recorder START generation must increase monotonically")
                    metadata = _build_start_metadata(
                        shared,
                        task_label=_control_text(control, "task_label"),
                        operator=_control_text(control, "operator"),
                    )
                    if not recorder.start_episode(**metadata):
                        raise RuntimeError("EpisodeRecorder refused start")
                    active_generation = generation
                    _publish_status(
                        shared,
                        RecorderPhase.RECORDING,
                        generation,
                        failure_count=failure_count,
                    )
                except Exception as exc:
                    logger.error("RecorderIO start failed", exc_info=True)
                    failure_count += 1
                    _publish_status(
                        shared,
                        RecorderPhase.ERROR,
                        generation,
                        error=str(exc),
                        reason="start_error",
                        failure_count=failure_count,
                    )

            # Drain all still-resident samples oldest-first before honoring STOP.
            samples = shared.record_sample_ring.get_last_k(shared.record_sample_ring.maxlen)
            unseen = [(data, seq) for data, _publish_ns, seq in samples if seq > last_sample_sequence]
            if unseen and unseen[0][1] > last_sample_sequence + 1 and recorder.is_recording:
                reason = f"sample ring overflow: expected {last_sample_sequence + 1}, found {unseen[0][1]}"
                logger.error("RecorderIO %s", reason)
                _begin_finalization(
                    generation=active_generation,
                    save=False,
                    reason="sample_ring_overflow",
                    forced_error=reason,
                )
            for data, sequence in unseen:
                last_sample_sequence = sequence
                shared.recorder_consumed_sequence.value = sequence
                record = data[0]
                if not recorder.is_recording or int(record["generation"]) != active_generation:
                    continue
                try:
                    state, action, vr_frame, camera_frame, signals, arm_qpos_sent, diagnostics = _unpack_sample(record)
                    added = recorder.add_frame(
                        state,
                        action,
                        vr_frame,
                        # Keep typed camera quality metadata even when the
                        # image payload is absent; EpisodeRecorder fills
                        # shape-stable zero arrays for that grid slot.
                        camera_frame=camera_frame,
                        signals=signals,
                        arm_qpos_sent=arm_qpos_sent,
                        diagnostics=diagnostics,
                        control_run_generation=int(record["control_run_generation"]),
                    )
                    if recorder.camera_writer_error:
                        raise RuntimeError(recorder.camera_writer_error)
                    if not added and recorder.max_frames_reached:
                        _begin_finalization(
                            generation=active_generation,
                            save=True,
                            reason="max_frames",
                        )
                except Exception as exc:
                    logger.error("RecorderIO sample write failed", exc_info=True)
                    _begin_finalization(
                        generation=active_generation,
                        save=False,
                        reason="sample_write_error",
                        forced_error=str(exc),
                    )
                    break

            if control is not None and RecorderCommand(int(control["command"])) is RecorderCommand.STOP:
                generation = int(control["generation"])
                if generation == active_generation and recorder.is_recording:
                    stop_reason = _control_text(control, "stop_reason") or "manual"
                    forced_error = (
                        f"recording aborted: {stop_reason}"
                        if stop_reason in {"sample_ring_overflow", "camera_writer_error", "camera_stall"}
                        else ""
                    )
                    _begin_finalization(
                        generation=generation,
                        save=bool(control["save"]),
                        reason=stop_reason,
                        forced_error=forced_error,
                    )
                elif generation != active_generation:
                    logger.warning(
                        "RecorderIO ignored STOP for generation %d (active=%d)",
                        generation,
                        active_generation,
                    )

            # An unexpected policy/main shutdown may arrive before the policy
            # can publish STOP.  Do not keep the RecorderIO child alive merely
            # because an episode is active: drain every resident sample, close
            # all codecs/HDF5 handles, and retain only the aborted manifest.
            if not shared.is_running.value and recorder.is_recording and pending_finalization is None:
                _begin_finalization(
                    generation=active_generation,
                    save=False,
                    reason="runtime_shutdown",
                )

            if pending_finalization is not None:
                stop_result: EpisodeStopResult = recorder.poll_stop()
                elapsed_s = time.monotonic() - pending_finalization.started_monotonic_s
                if (
                    not stop_result.done
                    and elapsed_s >= _RECORDER_STOP_TIMEOUT_S
                    and not pending_finalization.timed_out
                ):
                    pending_finalization.timed_out = True
                    failure_count += 1
                    logger.error(
                        "RecorderIO episode finalization exceeded %.1fs",
                        _RECORDER_STOP_TIMEOUT_S,
                    )
                    _publish_status(
                        shared,
                        # Timeout is a sticky session failure, but it is not a
                        # terminal transaction state while the stop thread is
                        # still alive. Keeping FINALIZING prevents a new START
                        # from overtaking ownership of its recorder state.
                        RecorderPhase.FINALIZING,
                        pending_finalization.generation,
                        frame_count=pending_finalization.frame_count,
                        error="episode finalization timed out",
                        path=pending_finalization.path,
                        reason=pending_finalization.reason,
                        min_frames_met=pending_finalization.frame_count >= config.min_frames,
                        failure_count=failure_count,
                    )
                if stop_result.done:
                    error = stop_result.error or pending_finalization.forced_error
                    if pending_finalization.timed_out:
                        error = error or "episode finalization timed out"
                    if error and not pending_finalization.timed_out:
                        failure_count += 1
                    phase = RecorderPhase.ERROR if error else RecorderPhase.COMPLETED
                    saved = pending_finalization.save and not error
                    path = stop_result.path or pending_finalization.path
                    frame_count = stop_result.frame_count or pending_finalization.frame_count
                    generation = pending_finalization.generation
                    reason = pending_finalization.reason
                    pending_finalization = None
                    _publish_status(
                        shared,
                        phase,
                        generation,
                        frame_count=frame_count,
                        error=error or "",
                        path=path or "",
                        saved=saved,
                        reason=reason,
                        min_frames_met=frame_count >= config.min_frames,
                        failure_count=failure_count,
                    )

            limiter.wait()
    except Exception:
        crashed = True
        logger.error("RecorderIO process crashed", exc_info=True)
        failure_count += 1
        _publish_status(
            shared,
            RecorderPhase.ERROR,
            active_generation,
            error="RecorderIO process crashed",
            reason="process_crash",
            failure_count=failure_count,
        )
        publish_component_status(
            shared,
            "recorder",
            ComponentPhase.FAULT,
            fault_code=FaultCode.RECORDING_ABORTED,
            detail="RecorderIO process crashed; robot fault is not latched",
        )
    finally:
        if recorder is not None:
            if recorder.is_recording:
                recorder.stop_episode(
                    success=False,
                    reason="recorder_process_shutdown",
                )
            # Also reap a stop thread that was already pending when an
            # unexpected exception escaped the polling loop. The thread owns
            # HDF5/camera state and must not be abandoned at process exit.
            if not recorder.join_stop(timeout=_RECORDER_STOP_TIMEOUT_S):
                if recorder.stop_error:
                    logger.error(
                        "RecorderIO process-shutdown finalization failed: %s",
                        recorder.stop_error,
                    )
                else:
                    logger.error(
                        "RecorderIO timed out during process-shutdown finalization"
                    )
        _publish_status(
            shared,
            RecorderPhase.STOPPED,
            active_generation,
            failure_count=failure_count,
        )
        if not crashed:
            publish_component_status(shared, "recorder", ComponentPhase.STOPPED)
        logger.info("RecorderIO exited")
