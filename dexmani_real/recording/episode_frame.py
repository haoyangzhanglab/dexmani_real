"""Immutable ownership boundary between recording samples and serialization.

Each frame copies arrays out of its producer-owned storage, exposes a
read-only mapping and read-only arrays, and is therefore safe to retain while
the shared-memory rings continue being overwritten.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d
from dexmani_real.recording.episode_schema import (
    ARM_SENT_DATASET,
    normalize_diagnostics,
)
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

EpisodeValue = np.ndarray | np.generic | float | int | bool


@dataclass(frozen=True)
class EpisodeFrame:
    """One immutable recorder-owned source frame ready for causal alignment."""

    timestamp_s: float
    control_run_generation: int
    data: Mapping[str, EpisodeValue]
    camera_rgb: np.ndarray | None = None
    camera_depth: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamp_s = float(self.timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("episode frame timestamp must be finite")
        generation = int(self.control_run_generation)
        if generation < 0:
            raise ValueError("control_run_generation must be non-negative")
        copied_data: dict[str, EpisodeValue] = {}
        for name, value in self.data.items():
            if isinstance(value, np.ndarray):
                copied_value = np.array(value, copy=True)
                copied_value.setflags(write=False)
                copied_data[name] = copied_value
            else:
                copied_data[name] = value
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "control_run_generation", generation)
        object.__setattr__(self, "data", MappingProxyType(copied_data))
        if self.camera_rgb is not None:
            camera_rgb = np.array(self.camera_rgb, copy=True)
            camera_rgb.setflags(write=False)
            object.__setattr__(self, "camera_rgb", camera_rgb)
        if self.camera_depth is not None:
            camera_depth = np.array(self.camera_depth, copy=True)
            camera_depth.setflags(write=False)
            object.__setattr__(self, "camera_depth", camera_depth)


def _action_eef(action: RobotAction) -> np.ndarray:
    position = (
        np.asarray(action.target_eef_pos, dtype=np.float64)
        if action.target_eef_pos is not None
        else np.full(3, np.nan)
    )
    rotation_6d = (
        np.asarray(action.target_eef_rot6d, dtype=np.float64)
        if action.target_eef_rot6d is not None
        else np.full(6, np.nan)
    )
    return np.concatenate([position, rotation_6d])


def _scalar_int(value: object) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"expected an integer-compatible scalar, got {array.shape}")
    return int(array.item())


def _scalar_float(value: object) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"expected a float-compatible scalar, got {array.shape}")
    return float(array.item())


def build_episode_frame(
    state: RobotState,
    action: RobotAction,
    vr_frame: Mapping[str, object],
    *,
    camera_frame: Mapping[str, object] | None = None,
    signals: Mapping[str, object] | None = None,
    arm_qpos_sent: np.ndarray | None = None,
    diagnostics: Mapping[str, object] | None = None,
    control_run_generation: int = 0,
    arm_sent_stream: bool,
) -> EpisodeFrame:
    """Build one schema-shaped frame from the recording inputs."""
    signal = signals or {}
    camera = camera_frame or {}
    diagnostic_values = normalize_diagnostics(diagnostics)
    camera_source_ns = signal.get(
        "camera_source_monotonic_ns", camera.get("source_monotonic_ns", 0)
    )
    camera_publish_ns = signal.get(
        "camera_publish_monotonic_ns", camera.get("publish_monotonic_ns", 0)
    )
    data: dict[str, EpisodeValue] = {
        "arm_qpos": np.asarray(state.arm_qpos, dtype=np.float64),
        "arm_ee": np.concatenate([state.eef_pos, state.eef_rot6d]).astype(np.float64),
        "arm_qvel": np.asarray(state.arm_qvel, dtype=np.float64),
        "arm_tau": np.asarray(state.arm_tau, dtype=np.float64),
        "hand_qpos": np.asarray(state.hand_qpos, dtype=np.float64),
        "hand_fingertip": np.asarray(state.fingertip_pos, dtype=np.float64),
        "hand_contact": np.asarray(state.hand_tactile_sum, dtype=np.float64),
        "hand_tactile_force": np.asarray(state.hand_tactile_force, dtype=np.float64),
        "hand_tactile_contact": np.asarray(state.hand_tactile_contact, dtype=bool),
        "hand_tipboard_err": np.asarray(state.hand_tipboard_err, dtype=np.int32),
        "hand_commboard_err": np.asarray(state.hand_commboard_err, dtype=np.int32),
        "hand_jointboard_err": np.asarray(state.hand_jointboard_err, dtype=np.int32),
        "hand_current": (
            np.asarray(state.hand_current, dtype=np.float64)
            if state.hand_current is not None
            else np.full(HAND_JOINT_SHAPE, np.nan)
        ),
        "arm_connected": bool(state.arm_connected),
        "hand_connected": bool(state.hand_connected),
        "hand_qpos_stale": bool(state.hand_qpos_stale),
        "arm_last_cmd_seq": int(state.arm_last_cmd_seq),
        "arm_last_cmd_is_hold": bool(state.arm_last_cmd_is_hold),
        "action_arm_joint": np.asarray(action.arm_qpos_cmd, dtype=np.float64),
        "action_arm_ee": _action_eef(action),
        "action_hand_joint": np.asarray(action.hand_qpos_cmd, dtype=np.float64),
        "observation_id": _scalar_int(signal.get("observation_id", 0)),
        "observation_anchor_monotonic_ns": _scalar_int(
            signal.get("observation_anchor_monotonic_ns", 0)
        ),
        "arm_source_sequence": _scalar_int(signal.get("arm_source_sequence", 0)),
        "hand_source_sequence": _scalar_int(signal.get("hand_source_sequence", 0)),
        "vr_source_sequence": _scalar_int(signal.get("vr_source_sequence", 0)),
        "camera_source_sequence": _scalar_int(signal.get("camera_source_sequence", 0)),
        "arm_source_monotonic_ns": _scalar_int(
            signal.get("arm_source_monotonic_ns", 0)
        ),
        "hand_source_monotonic_ns": _scalar_int(
            signal.get("hand_source_monotonic_ns", 0)
        ),
        "vr_source_monotonic_ns": _scalar_int(signal.get("vr_source_monotonic_ns", 0)),
        "camera_source_monotonic_ns": _scalar_int(camera_source_ns),
        "arm_publish_monotonic_ns": _scalar_int(
            signal.get("arm_publish_monotonic_ns", 0)
        ),
        "hand_publish_monotonic_ns": _scalar_int(
            signal.get("hand_publish_monotonic_ns", 0)
        ),
        "vr_publish_monotonic_ns": _scalar_int(
            signal.get("vr_publish_monotonic_ns", 0)
        ),
        "camera_publish_monotonic_ns": _scalar_int(camera_publish_ns),
        "observation_source_receive_monotonic_ns": np.asarray(
            signal.get("observation_source_receive_monotonic_ns", np.zeros(4)),
            dtype=np.uint64,
        ),
        "observation_source_age_s": np.asarray(
            signal.get("observation_source_age_s", np.full(4, np.nan)),
            dtype=np.float64,
        ),
        "observation_source_skew_s": np.asarray(
            signal.get("observation_source_skew_s", np.full(4, np.nan)),
            dtype=np.float64,
        ),
        "observation_history_valid_mask": np.asarray(
            signal.get(
                "observation_history_valid_mask",
                np.zeros((4, 1), dtype=bool),
            ),
            dtype=bool,
        ),
        "observation_valid": bool(signal.get("observation_valid", False)),
        "observation_skew_s": _scalar_float(signal.get("observation_skew_s", np.nan)),
        "action_id": _scalar_int(signal.get("action_id", 0)),
        "action_created_monotonic_ns": _scalar_int(
            signal.get("action_created_monotonic_ns", 0)
        ),
        "action_target_monotonic_ns": _scalar_int(
            signal.get("action_target_monotonic_ns", 0)
        ),
        "action_valid_until_monotonic_ns": _scalar_int(
            signal.get("action_valid_until_monotonic_ns", 0)
        ),
        "action_arm_joint_raw": np.asarray(
            signal.get("action_arm_joint_raw", action.arm_qpos_cmd),
            dtype=np.float64,
        ),
        "flag_action_queued": bool(signal.get("action_queued", False)),
        "tactile_fresh": bool(signal.get("tactile_fresh", False)),
        "tactile_source_monotonic_ns": _scalar_int(
            signal.get("tactile_source_monotonic_ns", 0)
        ),
        "tactile_calibrated": bool(signal.get("tactile_calibrated", False)),
        "tactile_unit_code": _scalar_int(signal.get("tactile_unit_code", 0)),
        "pointcloud_valid_depth_ratio": _scalar_float(
            signal.get("pointcloud_valid_depth_ratio", np.nan)
        ),
        "flag_ik_ok": bool(signal.get("ik_ok", False)),
        "flag_ik_attempted": bool(signal.get("ik_attempted", True)),
        "flag_retarget_ok": bool(signal.get("retarget_ok", False)),
        "flag_held": bool(signal.get("held", False)),
        "flag_safety_reject": bool(signal.get("flag_safety_reject", False)),
        "camera_health": _scalar_int(camera.get("camera_health", 1)),
        "flag_camera_fresh": bool(camera.get("camera_fresh", False)),
        "camera_depth_frame_number": _scalar_int(camera.get("depth_frame_number", 0)),
        "camera_color_frame_number": _scalar_int(camera.get("color_frame_number", 0)),
        "camera_ring_sequence": _scalar_int(camera.get("ring_sequence", 0)),
        "camera_depth_device_timestamp_s": _scalar_float(
            camera.get("depth_device_timestamp_s", np.nan)
        ),
        "camera_color_device_timestamp_s": _scalar_float(
            camera.get("color_device_timestamp_s", np.nan)
        ),
        "camera_wait_return_monotonic_ns": _scalar_int(
            camera.get("wait_return_monotonic_ns", 0)
        ),
        "camera_payload_ready_monotonic_ns": _scalar_int(
            camera.get("payload_ready_monotonic_ns", 0)
        ),
        "camera_depth_timestamp_domain": _scalar_int(
            camera.get("depth_timestamp_domain", 0)
        ),
        "camera_color_timestamp_domain": _scalar_int(
            camera.get("color_timestamp_domain", 255)
        ),
        "camera_age_s": _scalar_float(camera.get("camera_age_s", np.nan)),
        "camera_generation": _scalar_int(camera.get("camera_generation", 0)),
        "camera_clock_reset": bool(camera.get("clock_reset", False)),
        "camera_duplicate": bool(camera.get("duplicate", False)),
        "camera_frame_gap": _scalar_int(camera.get("frame_gap", 0)),
        "camera_backlog_s": _scalar_float(camera.get("backlog_s", np.nan)),
        "camera_delivery_delay_above_floor_s": _scalar_float(
            camera.get("delivery_delay_above_floor_s", camera.get("backlog_s", np.nan))
        ),
        "flag_frame_status": _scalar_int(signal.get("frame_status", 0)),
        "vr_wrist_pos": np.asarray(vr_frame["wrist_pos"], dtype=np.float64),
        "vr_wrist_rot6d": quat_wxyz_to_rot6d(
            np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)
        ),
        "vr_landmarks": np.asarray(vr_frame["landmarks"], dtype=np.float64),
        "tracking_error": np.nan,
        "ik_solve_time_ms": np.nan,
        "target_pos_before_clamp": np.full(3, np.nan),
        "head_quat_wxyz": np.full(4, np.nan),
        "target_eef_pos_raw": np.full(3, np.nan),
        "target_eef_rot6d_raw": np.full(6, np.nan),
        "action_hand_joint_raw": np.asarray(action.hand_qpos_cmd, dtype=np.float64),
        "policy_map_time_ms": np.nan,
        "hand_retarget_time_ms": np.nan,
        "transition_check_time_ms": np.nan,
        "policy_compute_time_ms": np.nan,
    }
    if arm_sent_stream:
        data[ARM_SENT_DATASET] = (
            np.asarray(arm_qpos_sent, dtype=np.float64)
            if arm_qpos_sent is not None
            else np.full(ARM_JOINT_SHAPE, np.nan)
        )
    data.update(diagnostic_values)

    rgb_value = camera.get("rgb")
    depth_value = camera.get("depth")
    return EpisodeFrame(
        timestamp_s=float(state.timestamp),
        control_run_generation=control_run_generation,
        data=data,
        camera_rgb=(None if rgb_value is None else np.asarray(rgb_value)),
        camera_depth=(None if depth_value is None else np.asarray(depth_value)),
    )


def _copy_record_field(record: np.void, name: str) -> EpisodeValue:
    value = record[name]
    return np.array(value, copy=True) if np.asarray(value).ndim else value.item()


def decode_record_sample(
    record: np.void, *, arm_sent_stream: bool = True
) -> EpisodeFrame:
    """Copy one fixed shared-memory record into an owned episode frame."""
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
        arm_last_cmd_seq=int(record["arm_last_cmd_seq"]),
        arm_last_cmd_is_hold=bool(record["arm_last_cmd_is_hold"]),
    )
    action = RobotAction(
        arm_qpos_cmd=np.array(record["action_arm_qpos"], copy=True),
        hand_qpos_cmd=np.array(record["action_hand_qpos"], copy=True),
        target_eef_pos=np.array(record["action_target_eef_pos"], copy=True),
        target_eef_rot6d=np.array(record["action_target_eef_rot6d"], copy=True),
    )
    vr_frame: dict[str, object] = {
        "wrist_pos": np.array(record["vr_wrist_pos"], copy=True),
        "wrist_quat_wxyz": np.array(record["vr_wrist_quat_wxyz"], copy=True),
        "landmarks": np.array(record["vr_landmarks"], copy=True),
    }
    camera_frame: dict[str, object] = {
        "camera_health": int(record["camera_health"]),
        "camera_fresh": bool(record["camera_fresh"]),
        "depth_frame_number": int(record["camera_depth_frame_number"]),
        "color_frame_number": int(record["camera_color_frame_number"]),
        "ring_sequence": int(record["camera_ring_sequence"]),
        "depth_device_timestamp_s": float(record["camera_depth_device_timestamp_s"]),
        "color_device_timestamp_s": float(record["camera_color_device_timestamp_s"]),
        "wait_return_monotonic_ns": int(record["camera_wait_return_monotonic_ns"]),
        "payload_ready_monotonic_ns": int(record["camera_payload_ready_monotonic_ns"]),
        "depth_timestamp_domain": int(record["camera_depth_timestamp_domain"]),
        "color_timestamp_domain": int(record["camera_color_timestamp_domain"]),
        "camera_age_s": float(record["camera_age_s"]),
        "camera_generation": int(record["camera_generation"]),
        "clock_reset": bool(record["camera_clock_reset"]),
        "duplicate": bool(record["camera_duplicate"]),
        "frame_gap": int(record["camera_frame_gap"]),
        "backlog_s": float(record["camera_backlog_s"]),
        "delivery_delay_above_floor_s": float(
            record["camera_delivery_delay_above_floor_s"]
        ),
    }
    if bool(record["camera_present"]):
        camera_frame.update(
            rgb=np.array(record["camera_rgb"], copy=True),
            depth=np.array(record["camera_depth"], copy=True),
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
        "pointcloud_valid_depth_ratio",
    )
    signals: dict[str, object] = {
        name: _copy_record_field(record, name) for name in signal_names
    }
    for signal_name, field_name in (
        ("action_queued", "flag_action_queued"),
        ("ik_ok", "flag_ik_ok"),
        ("ik_attempted", "flag_ik_attempted"),
        ("retarget_ok", "flag_retarget_ok"),
        ("held", "flag_held"),
        ("flag_safety_reject", "flag_safety_reject"),
    ):
        signals[signal_name] = bool(record[field_name])
    signals["frame_status"] = int(record["flag_frame_status"])
    diagnostics = {
        name: _copy_record_field(record, name)
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
    return build_episode_frame(
        state,
        action,
        vr_frame,
        camera_frame=camera_frame,
        signals=signals,
        arm_qpos_sent=np.array(record["arm_qpos_sent"], copy=True),
        diagnostics=diagnostics,
        control_run_generation=int(record["control_run_generation"]),
        arm_sent_stream=arm_sent_stream,
    )


__all__ = [
    "EpisodeFrame",
    "EpisodeValue",
    "build_episode_frame",
    "decode_record_sample",
]
