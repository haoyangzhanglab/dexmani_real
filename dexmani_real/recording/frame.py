"""Owned raw-v25 source rows assembled at the recording boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d
from dexmani_real.recording.sample import EpisodeAction, EpisodeState
from dexmani_real.recording.storage.schema import (
    DATASET_SPECS,
    SOURCE_FRAME_DATASET_NAMES,
)

EpisodeValue = np.ndarray | np.generic | float | int | bool


@dataclass
class EpisodeFrame:
    """One owned source row; builders copy producer arrays before retaining them."""

    timestamp_s: float
    data: dict[str, EpisodeValue]
    camera_rgb: np.ndarray | None = None
    camera_depth: np.ndarray | None = None


def episode_source_values(
    state: EpisodeState,
    action: EpisodeAction,
    vr_frame: Mapping[str, object],
    *,
    camera_frame: Mapping[str, object] | None = None,
    signals: Mapping[str, object] | None = None,
    arm_qpos_sent: np.ndarray | None = None,
) -> dict[str, EpisodeValue]:
    """Map source values; the destination owns copying them before retention."""
    signal = signals or {}
    camera = camera_frame or {}
    if arm_qpos_sent is None:
        raise ValueError("recording requires the explicit submitted arm target")
    return {
        "arm_qpos": state.arm_qpos,
        "arm_qvel": state.arm_qvel,
        "arm_tau": state.arm_tau,
        "hand_qpos": state.hand_qpos,
        "hand_current": state.hand_current,
        "hand_contact": state.hand_tactile_sum,
        "hand_tactile_force": state.hand_tactile_force,
        "arm_connected": state.arm_connected,
        "hand_connected": state.hand_connected,
        "hand_qpos_stale": state.hand_qpos_stale,
        "arm_last_cmd_seq": state.arm_last_cmd_seq,
        "action_arm_joint_sent": arm_qpos_sent,
        "action_hand_joint": action.hand_qpos_cmd,
        "action_arm_ee": np.concatenate(
            (
                (
                    action.target_eef_pos
                    if action.target_eef_pos is not None
                    else np.full(3, np.nan)
                ),
                (
                    action.target_eef_rot6d
                    if action.target_eef_rot6d is not None
                    else np.full(6, np.nan)
                ),
            )
        ),
        "flag_action_queued": signal.get("action_queued", False),
        "flag_frame_status": signal.get("frame_status", 0),
        "tracking_error": signal.get("tracking_error", np.nan),
        "observation_anchor_monotonic_ns": signal.get(
            "observation_anchor_monotonic_ns", 0
        ),
        "observation_valid": signal.get("observation_valid", False),
        "arm_source_monotonic_ns": signal.get("arm_source_monotonic_ns", 0),
        "hand_source_monotonic_ns": signal.get("hand_source_monotonic_ns", 0),
        "tactile_source_monotonic_ns": signal.get("tactile_source_monotonic_ns", 0),
        "vr_source_monotonic_ns": signal.get("vr_source_monotonic_ns", 0),
        "camera_source_monotonic_ns": signal.get(
            "camera_source_monotonic_ns", camera.get("source_monotonic_ns", 0)
        ),
        "tactile_fresh": signal.get("tactile_fresh", False),
        "tactile_calibrated": signal.get("tactile_calibrated", False),
        "tactile_unit_code": signal.get("tactile_unit_code", 0),
        "flag_camera_fresh": camera.get("camera_fresh", False),
        "camera_depth_frame_number": camera.get("depth_frame_number", 0),
        "camera_color_frame_number": camera.get("color_frame_number", 0),
        "policy_observation_arm_qpos": signal.get(
            "policy_observation_arm_qpos", np.full(7, np.nan)
        ),
        "policy_observation_hand_qpos": signal.get(
            "policy_observation_hand_qpos", np.full(12, np.nan)
        ),
        "policy_observation_valid": signal.get("policy_observation_valid", False),
        "vr_wrist_pos": vr_frame["wrist_pos"],
        "vr_wrist_rot6d": quat_wxyz_to_rot6d(
            np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)
        ),
        "vr_landmarks": vr_frame["landmarks"],
        "head_quat_wxyz": vr_frame.get(
            "head_quat_wxyz", signal.get("head_quat_wxyz", np.full(4, np.nan))
        ),
    }


def build_episode_frame(
    state: EpisodeState,
    action: EpisodeAction,
    vr_frame: Mapping[str, object],
    *,
    camera_frame: Mapping[str, object] | None = None,
    signals: Mapping[str, object] | None = None,
    arm_qpos_sent: np.ndarray | None = None,
) -> EpisodeFrame:
    """Copy a source sample for direct recorder retention."""
    values = episode_source_values(
        state,
        action,
        vr_frame,
        camera_frame=camera_frame,
        signals=signals,
        arm_qpos_sent=arm_qpos_sent,
    )
    camera = camera_frame or {}
    data = {
        name: np.array(value, dtype=DATASET_SPECS[name].dtype, copy=True)
        for name, value in values.items()
    }
    return EpisodeFrame(
        timestamp_s=float(state.timestamp),
        data=data,
        camera_rgb=(
            np.array(camera["rgb"], copy=True)
            if camera.get("rgb") is not None
            else None
        ),
        camera_depth=(
            np.array(camera["depth"], copy=True)
            if camera.get("depth") is not None
            else None
        ),
    )


def decode_record_sample(record: np.void) -> EpisodeFrame:
    """Copy the sample ring before its producer can overwrite the slot."""
    present = bool(record["camera_present"])
    return EpisodeFrame(
        timestamp_s=float(record["timestamp"]),
        data={
            name: np.array(
                record[name],
                dtype=DATASET_SPECS[name].dtype,
                copy=True,
            )
            for name in SOURCE_FRAME_DATASET_NAMES
        },
        camera_rgb=np.array(record["camera_rgb"], copy=True) if present else None,
        camera_depth=np.array(record["camera_depth"], copy=True) if present else None,
    )
