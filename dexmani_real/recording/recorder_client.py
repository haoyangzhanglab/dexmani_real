"""Policy-side recorder client and shared control-plane protocol types.

``RecorderClient`` is the policy-side owner of recording decisions and fixed
sample construction for the RecorderIO transport. ``DirectRecorderClient``
uses the same sample contract with a policy-local ``EpisodeRecorder``. This
module also holds the control-plane types shared with the RecorderIO process
(``recorder_io_loop`` in ``io_process.py``), which imports them from here. The
module never imports ``io_process``, keeping that dependency one-way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from dexmani_real.recording.episode_schema import normalize_diagnostics_v17
from dexmani_real.recording.camera_stream_writer import CameraStreamWriterConfig
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.recording.start_metadata import build_start_metadata
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import (
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    RECORD_CONTROL_DTYPE,
    RECORD_OPERATOR_BYTES,
    RECORD_STOP_REASON_BYTES,
    RECORD_TASK_LABEL_BYTES,
)

logger = get_logger(__name__)

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


def _bounded_control_text(value: str, *, capacity: int, field: str) -> bytes:
    """Encode a control-plane text field without an unbounded JSON side channel."""
    payload = value.encode("utf-8")
    if len(payload) > capacity:
        raise ValueError(f"RecorderIO {field} exceeds fixed capacity {capacity}")
    return payload


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
    diagnostic_data = normalize_diagnostics_v17(diagnostics)

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


class DirectRecorderClient:
    """Policy-local adapter over :class:`EpisodeRecorder`.

    This is the simplified recording mechanism: policy still constructs the
    exact same full sample and the existing ``EpisodeRecorder`` still writes
    schema-v17 HDF5 plus RGB/depth sidecars.  Only the fixed shared-memory
    sample/control/status transport and the dedicated RecorderIO process are
    removed from the hot path.
    """

    def __init__(
        self,
        shared: Any,
        *,
        data_dir: str,
        max_frames: int,
        control_hz: float,
        min_frames: int,
        resolved_config_sha256: str,
        align_mode: str,
        provenance: dict[str, str] | None = None,
        rgb_shape: tuple[int, int, int],
        depth_shape: tuple[int, int],
        writer_queue_size: int,
    ) -> None:
        self.shared = shared
        self._recorder = EpisodeRecorder(
            data_dir=data_dir,
            max_frames=max_frames,
            control_hz=control_hz,
            min_frames=min_frames,
            arm_sent_stream=True,
            resolved_config_hash=resolved_config_sha256,
            provenance=provenance,
            camera_writer_config=CameraStreamWriterConfig(
                rgb_shape=rgb_shape,
                depth_shape=depth_shape,
                fps=control_hz,
                queue_size=writer_queue_size,
            ),
        )
        self._align_mode = align_mode
        self._recording = False
        self._stop_pending = False
        self._stop_save = False
        self._stop_reason = ""
        self._stop_path: str | None = None
        self._stop_frame_count = 0
        self._last_stop_result: RecorderStopResult | None = None

    @property
    def frame_count(self) -> int:
        return self._recorder.frame_count

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def stop_pending(self) -> bool:
        return self._stop_pending

    @property
    def camera_writer_error(self) -> str | None:
        return self._recorder.camera_writer_error

    def start_episode(self, *, task_label: str = "", operator: str = "") -> bool:
        if self._recording:
            return False
        if self._stop_pending:
            result = self.join_stop(timeout=_RECORDER_STOP_TIMEOUT_S)
            if not result.done:
                return False
        if not self._recorder.join_stop(timeout=_RECORDER_STOP_TIMEOUT_S) and self._recorder.stop_error is None:
            logger.error("direct recorder: previous episode did not finish: %s", self._recorder.stop_error)
            return False
        try:
            metadata = build_start_metadata(
                self.shared,
                task_label=task_label,
                operator=operator,
                align_mode=self._align_mode,
            )
            started = self._recorder.start_episode(**metadata)
        except Exception:
            logger.error("direct recorder: episode start failed", exc_info=True)
            return False
        if started:
            self._recording = True
            self._last_stop_result = None
        return started

    def add_frame(
        self,
        state: Any,
        action: RobotAction,
        vr_frame: dict[str, Any],
        camera_frame: dict[str, Any] | None = None,
        signals: dict[str, Any] | None = None,
        arm_qpos_sent: np.ndarray | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> bool:
        if not self._recording:
            return False
        try:
            added = self._recorder.add_frame(
                state,
                action,
                vr_frame,
                camera_frame=camera_frame,
                signals=signals,
                arm_qpos_sent=arm_qpos_sent,
                diagnostics=diagnostics,
                control_run_generation=int(self.shared.run_generation.value),
            )
        except Exception as exc:
            logger.error("direct recorder: sample write failed", exc_info=True)
            self.stop_episode(success=False, reason="sample_write_error")
            return False
        if not added and self._recorder.max_frames_reached:
            self.stop_episode(success=True, reason="max_frames")
        return added

    def stop_episode(self, success: bool = True, reason: str = "") -> str | None:
        if not self._recording or self._stop_pending:
            return None
        self._recording = False
        self._stop_pending = True
        self._stop_save = bool(success)
        self._stop_reason = reason or ("max_frames" if self._recorder.max_frames_reached else "manual")
        self._stop_frame_count = self.frame_count
        self._stop_path = self._recorder.stop_episode(success=success, reason=reason)
        return self._stop_path

    def _stop_result(self, result: Any, *, done: bool) -> RecorderStopResult:
        error = result.error or self._recorder.stop_error
        return RecorderStopResult(
            done=done,
            phase=RecorderPhase.ERROR if error else (RecorderPhase.COMPLETED if done else RecorderPhase.FINALIZING),
            saved=bool(done and self._stop_save and not error),
            error=error,
            path=result.path or self._stop_path,
            frame_count=int(result.frame_count or self._stop_frame_count or self.frame_count),
            reason=self._stop_reason,
            min_frames_met=int(result.frame_count or self._stop_frame_count or self.frame_count)
            >= self._recorder.min_frames,
        )

    def poll_stop(self) -> RecorderStopResult:
        if not self._stop_pending:
            return self._last_stop_result or RecorderStopResult(done=True)
        result = self._recorder.poll_stop()
        if not result.done:
            return self._stop_result(result, done=False)
        mapped = self._stop_result(result, done=True)
        self._stop_pending = False
        self._last_stop_result = mapped
        return mapped

    def join_stop(self, timeout: float | None = None) -> RecorderStopResult:
        if not self._stop_pending:
            return self._last_stop_result or RecorderStopResult(done=True)
        timeout_s = _RECORDER_STOP_TIMEOUT_S if timeout is None else float(timeout)
        if not np.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("recorder stop timeout must be finite and non-negative")
        done = self._recorder.join_stop(timeout=timeout_s)
        if not done:
            return self._stop_result(self._recorder.poll_stop(), done=False)
        result = self._recorder.poll_stop()
        mapped = self._stop_result(result, done=True)
        self._stop_pending = False
        self._last_stop_result = mapped
        return mapped
