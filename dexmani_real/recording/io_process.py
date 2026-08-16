"""Dedicated RecorderIO process with a bounded shared-memory sample ring.

Policy owns episode boundaries, the configured control grid, and sample
contents. This module owns serialization, non-blocking finalization, camera
encoding, HDF5 writes, verification and transactional publication. Large
camera arrays never travel through an ``mp.Queue``; they occupy fixed slots in
a seqlock ring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.recording.camera_stream_writer import CameraStreamWriterConfig
from dexmani_real.recording.episode_recorder import EpisodeRecorder, StopResult as EpisodeStopResult
from dexmani_real.recording.recorder_client import (
    RecorderCommand,
    RecorderPhase,
    _RECORDER_STOP_TIMEOUT_S,
    _bounded_control_text,
)
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager
from dexmani_real.utils.schema import (
    RECORD_STATUS_DTYPE,
    RECORD_STATUS_TEXT_BYTES,
    RECORD_STOP_REASON_BYTES,
)

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
    align_mode: str
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
        if self.align_mode != "depth_to_color":
            raise ValueError(
                "RecorderIO production recording requires align_mode='depth_to_color' "
                "so camera_K and T_world_camera share the color optical frame"
            )
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


def _camera_geometry_from_profile(
    camera_profile_json: str,
    *,
    configured_align_mode: str,
    camera_K: np.ndarray,
) -> tuple[str, str, str]:
    """Validate actual camera geometry and return alignment/frame labels.

    Production calibration is defined in the color optical frame.  Requiring
    the actual camera profile to agree with the already-validated recorder
    configuration prevents metadata from labelling a depth-frame K/extrinsic
    pair as color-frame geometry.
    """
    if configured_align_mode != "depth_to_color":
        raise ValueError("production camera metadata requires align_mode='depth_to_color'")
    try:
        profile = json.loads(camera_profile_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("camera_actual_profile_json is not valid JSON") from exc
    if not isinstance(profile, dict):
        raise RuntimeError("camera_actual_profile_json must contain a JSON object")

    actual_align_mode = str(profile.get("align_mode", ""))
    common_viewport = str(profile.get("common_viewport", ""))
    output_optical_frame = str(profile.get("output_optical_frame", ""))
    if actual_align_mode != configured_align_mode:
        raise RuntimeError(
            "camera alignment does not match RecorderIO configuration: "
            f"actual={actual_align_mode!r}, configured={configured_align_mode!r}"
        )
    if common_viewport != "color" or output_optical_frame != "camera_color_optical":
        raise RuntimeError(
            "production camera profile must use the color common viewport and "
            "camera_color_optical output frame"
        )
    output_intrinsics = profile.get("output_intrinsics")
    if not isinstance(output_intrinsics, dict):
        raise RuntimeError("camera_actual_profile_json is missing output_intrinsics")
    try:
        profile_K = np.array(
            [
                [float(output_intrinsics["fx"]), 0.0, float(output_intrinsics["cx"])],
                [0.0, float(output_intrinsics["fy"]), float(output_intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("camera output_intrinsics are malformed") from exc
    if not np.allclose(profile_K, camera_K, rtol=1e-6, atol=1e-6):
        raise RuntimeError("camera_K does not match the actual common-viewport intrinsics")
    return actual_align_mode, common_viewport, output_optical_frame


def _build_start_metadata(
    shared: Any,
    *,
    task_label: str,
    operator: str,
    align_mode: str,
) -> dict[str, Any]:
    """Snapshot only essential recording metadata at the immutable START boundary."""
    camera_K_values = list(shared.camera_K)
    try:
        camera_K = np.asarray(camera_K_values, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("camera_K is unavailable or malformed at recorder START") from exc
    if (
        not np.all(np.isfinite(camera_K))
        or camera_K[0, 0] <= 0.0
        or camera_K[1, 1] <= 0.0
        or not np.allclose(camera_K[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
    ):
        raise RuntimeError("camera_K is unavailable or malformed at recorder START")
    depth_scale = float(shared.camera_depth_scale.value) if shared.camera_depth_scale.value != 0.0 else None
    camera_serial = _shared_text(shared.camera_serial.value, default=None)
    camera_firmware = _shared_text(shared.camera_firmware.value, default="unknown") or "unknown"
    camera_sdk_version = _shared_text(shared.camera_sdk_version.value, default="unknown") or "unknown"
    camera_profile_json = _shared_text(shared.camera_profile.value, default="{}") or "{}"
    actual_align_mode, common_viewport, output_optical_frame = _camera_geometry_from_profile(
        camera_profile_json,
        configured_align_mode=align_mode,
        camera_K=camera_K,
    )
    camera_pointcloud_config_json = (
        _shared_text(shared.camera_pointcloud_config.value, default="{}") or "{}"
    )
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
            "camera_alignment_mode": actual_align_mode,
            "camera_common_viewport": common_viewport,
            "camera_K_optical_frame": output_optical_frame,
            "camera_output_optical_frame": output_optical_frame,
            "camera_pointcloud_config_json": camera_pointcloud_config_json,
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
    recorder: EpisodeRecorder | None = None
    active_generation = 0
    last_control_sequence = 0
    last_sample_sequence = int(shared.recorder_consumed_sequence.value)
    pending_finalization: _PendingFinalization | None = None
    failure_count = 0
    crashed = False
    try:
        logger.debug("RecorderIO: LOADING")
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
        logger.debug("RecorderIO: READY")
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
                        align_mode=config.align_mode,
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
            logger.debug("RecorderIO: STOPPED")
        logger.info("RecorderIO exited")
