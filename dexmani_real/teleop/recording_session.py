"""Build episode-start metadata and bounded operator recording decisions."""

from __future__ import annotations

import hashlib
import time
from enum import Enum, auto
from typing import Any

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class QuitRecordingDecision(Enum):
    SAVE = auto()
    DISCARD = auto()
    SAVE_AND_HOME = auto()
    ESTOP = auto()
    SHUTDOWN = auto()
    TIMEOUT = auto()


def await_quit_recording_decision(
    shared: SharedStorage,
    keyboard: KeyboardHandler,
    *,
    timeout_s: float,
) -> QuitRecordingDecision:
    """Wait for the bounded save/discard decision while keeping policy health live."""
    deadline_s = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline_s:
        if shared.estop_request.value:
            return QuitRecordingDecision.ESTOP
        shared.policy_heartbeat_s.value = time.monotonic()
        for signal in keyboard.poll(timeout=0.1):
            if signal is ControlSignal.STOP:
                return QuitRecordingDecision.SAVE
            if signal is ControlSignal.DISCARD:
                return QuitRecordingDecision.DISCARD
            if signal is ControlSignal.HOME:
                return QuitRecordingDecision.SAVE_AND_HOME
            if signal is ControlSignal.EMERGENCY_STOP:
                shared.estop_request.value = True
                return QuitRecordingDecision.ESTOP
        if shared.estop_request.value:
            return QuitRecordingDecision.ESTOP
        if not shared.is_running.value:
            return QuitRecordingDecision.SHUTDOWN
    return QuitRecordingDecision.TIMEOUT


def _identity_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _shared_text(value: bytes, *, default: str | None) -> str | None:
    encoded = value.rstrip(b"\x00")
    return encoded.decode("utf-8") if encoded else default


def build_episode_start_kwargs(
    shared: SharedStorage,
    config: TeleopConfig,
    *,
    hand_available: bool,
) -> dict[str, Any]:
    """Snapshot device identity, calibration, and control provenance for one episode."""
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
        _shared_text(shared.arm_device_identity.value, default='{"status":"unavailable"}') or '{"status":"unavailable"}'
    )
    hand_identity_default = '{"status":"unavailable"}' if config.hand_enabled else '{"status":"disabled"}'
    hand_identity_json = (
        _shared_text(shared.hand_device_identity.value, default=hand_identity_default) or hand_identity_default
    )

    calibration = CameraCalib()
    try:
        camera_name = calibration.resolve_name_by_serial(camera_serial) if camera_serial else None
    except (KeyError, FileNotFoundError):
        camera_name = None
        logger.warning("Camera serial %s not found in cameras.json — no extrinsics in /meta", camera_serial)

    return {
        "task_label": config.task_label,
        "operator": config.operator,
        "calib": calibration,
        "camera_K": camera_K,
        "camera_name": camera_name,
        "camera_serial": camera_serial,
        "depth_scale": depth_scale,
        "camera_metadata": {
            "camera_config_width": config.camera_width,
            "camera_config_height": config.camera_height,
            "camera_config_fps": config.camera_fps,
            "camera_align_mode": config.camera_align_mode,
            "camera_max_frame_age_s": config.camera_max_frame_age_s,
            "camera_recording_stall_abort_s": config.camera_recording_stall_abort_s,
            "camera_firmware": camera_firmware,
            "camera_sdk_version": camera_sdk_version,
            "camera_actual_profile_json": camera_profile_json,
            "camera_serial_sha256": _identity_hash(camera_serial or "unknown"),
            "camera_firmware_sha256": _identity_hash(camera_firmware),
            "camera_sdk_version_sha256": _identity_hash(camera_sdk_version),
            "camera_actual_profile_sha256": _identity_hash(camera_profile_json),
        },
        "record_config": {
            "ema_alpha_pos": config.ema_alpha_pos,
            "ema_alpha_rot": config.ema_alpha_rot,
            "joint_max_acc": config.joint_max_acc_deg_s2,
            "joint_max_speed": config.joint_max_speed_deg_s,
            "arm_loop_hz": config.arm_loop_hz,
            "jerk_management": "unmanaged",
            "contact_stall_enabled": config.contact_stall_enabled,
            "contact_stall_table_z_surface_m": config.contact_stall_table_z_surface_m,
            "contact_stall_table_context_height_m": config.contact_stall_table_context_height_m,
            "contact_stall_min_downward_target_m": config.contact_stall_min_downward_target_m,
            "contact_stall_tracking_error_rad": config.contact_stall_tracking_error_rad,
            "contact_stall_max_closing_speed_rad_s": config.contact_stall_max_closing_speed_rad_s,
            "hand_available": hand_available,
            "hand_retargeting_type": config.hand_retargeting_type,
            "hand_output_smoothing_alpha": config.hand_output_smoothing_alpha,
            "hand_ramp_duration_s": config.hand_ramp_duration_s,
            "begin_motion_gate_timeout_s": config.begin_motion_gate_timeout_s,
            "hand_feedback_bound_tolerance_rad": config.hand_feedback_bound_tolerance_rad,
            "arm_device_identity_json": arm_identity_json,
            "arm_device_identity_sha256": _identity_hash(arm_identity_json),
            "hand_device_identity_json": hand_identity_json,
            "hand_device_identity_sha256": _identity_hash(hand_identity_json),
        },
    }
