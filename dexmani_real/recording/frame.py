"""Owned control-step source rows assembled at the recording boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time

import numpy as np

from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d
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


def build_episode_frame(
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    action: Mapping[str, np.ndarray],
    vr_frame: Mapping[str, object],
    *,
    timestamp_s: float | None = None,
    camera_frame: Mapping[str, object] | None = None,
    signals: Mapping[str, object] | None = None,
) -> EpisodeFrame:
    """Copy a control sample directly into the raw episode's field names.

    Action keys are action_arm_joint_sent, action_hand_joint, action_arm_ee.
    Joint targets are those actually submitted, while action_arm_ee retains
    Cartesian intent. Invalid tactile is NaN, never a false zero-contact sample.
    Arm/hand payloads and timestamps refer to the caller's one selected sample.
    """
    arm = arm_state[0] if arm_state is not None else None
    hand = hand_state[0] if hand_state is not None else None
    contact_valid = hand is not None and bool(hand["tactile_aggregate_valid"])
    dense_valid = hand is not None and bool(hand["tactile_dense_valid"])
    signal = signals or {}
    camera = camera_frame or {}
    values = {
        "arm_qpos": arm["qpos"] if arm is not None else np.full(7, np.nan),
        "arm_qvel": arm["qvel"] if arm is not None else np.full(7, np.nan),
        "arm_tau": arm["tau"] if arm is not None else np.full(7, np.nan),
        "hand_qpos": hand["qpos"] if hand is not None else np.full(12, np.nan),
        "hand_current": hand["current"] if hand is not None else np.full(12, np.nan),
        "hand_contact": hand["tactile_aggregate"] if contact_valid else np.full((5, 3), np.nan),
        "hand_contact_valid": contact_valid,
        "hand_tactile_force": hand["tactile_dense"] if dense_valid else np.full((5, 120, 3), np.nan),
        "hand_tactile_force_valid": dense_valid,
        "arm_connected": bool(arm["connected"]) if arm is not None else False,
        "hand_connected": bool(hand["connected"]) if hand is not None else False,
        "hand_qpos_stale": bool(hand["qpos_stale"]) if hand is not None else False,
        "action_arm_joint_sent": action["action_arm_joint_sent"],
        "action_hand_joint": action["action_hand_joint"],
        "action_arm_ee": action["action_arm_ee"],
        "flag_action_queued": signal.get("action_queued", False),
        "flag_frame_status": signal.get("frame_status", 0),
        "tracking_error": signal.get("tracking_error", np.nan),
        "observation_anchor_monotonic_ns": signal.get(
            "observation_anchor_monotonic_ns", 0
        ),
        "observation_valid": signal.get("observation_valid", False),
        "arm_source_monotonic_ns": signal.get("arm_source_monotonic_ns", 0),
        "hand_source_monotonic_ns": signal.get("hand_source_monotonic_ns", 0),
        "vr_source_monotonic_ns": signal.get("vr_source_monotonic_ns", 0),
        "camera_source_monotonic_ns": signal.get(
            "camera_source_monotonic_ns", camera.get("source_monotonic_ns", 0)
        ),
        "flag_camera_fresh": camera.get("camera_fresh", False),
        "camera_health": camera.get("camera_health", 0),
        "camera_depth_frame_number": camera.get("depth_frame_number", 0),
        "camera_color_frame_number": camera.get("color_frame_number", 0),
        "vr_wrist_pos": vr_frame["wrist_pos"],
        "vr_wrist_rot6d": quat_wxyz_to_rot6d(
            np.asarray(vr_frame["wrist_quat_wxyz"], dtype=np.float64)
        ),
        "vr_landmarks": vr_frame["landmarks"],
        "head_quat_wxyz": vr_frame.get(
            "head_quat_wxyz", signal.get("head_quat_wxyz", np.full(4, np.nan))
        ),
    }
    data = {
        name: np.array(value, dtype=DATASET_SPECS[name].dtype, copy=True)
        for name, value in values.items()
    }
    return EpisodeFrame(
        timestamp_s=time.perf_counter() if timestamp_s is None else float(timestamp_s),
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
