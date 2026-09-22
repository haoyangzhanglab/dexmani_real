"""Research rows copied from the same observation used by the controller."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d
from dexmani_real.recording.storage.schema import (
    DATASET_SPECS,
    FRAME_OK,
    SOURCE_FRAME_DATASET_NAMES,
)


@dataclass
class EpisodeFrame:
    timestamp_s: float
    data: dict
    camera_rgb: np.ndarray | None = None
    camera_depth: np.ndarray | None = None


def build_episode_frame(
    row,
    command=None,
    *,
    action_timestamp_ns=0,
    frame_status=FRAME_OK,
    arm_eef_intent=None,
):
    arm, hand, vr, camera = (
        row.arm[0],
        None if row.hand is None else row.hand[0],
        row.vr,
        row.camera,
    )
    contact_valid = hand is not None and bool(hand["tactile_aggregate_valid"])
    dense_valid = hand is not None and bool(hand["tactile_dense_valid"])
    values = {
        "arm_qpos": arm["qpos"],
        "arm_qvel": arm["qvel"],
        "arm_effort": arm["effort"],
        "hand_qpos": hand["qpos"] if hand is not None else np.full(12, np.nan),
        "hand_current": hand["current"] if hand is not None else np.full(12, np.nan),
        "hand_contact": hand["tactile_aggregate"] if contact_valid else np.full((5, 3), np.nan),
        "hand_contact_valid": contact_valid,
        "hand_tactile_force": hand["tactile_dense"]
        if dense_valid
        else np.full((5, 120, 3), np.nan),
        "hand_tactile_force_valid": dense_valid,
        "action_arm_joint_target": (
            command.arm_qpos
            if command is not None and command.arm_qpos is not None
            else np.full(7, np.nan)
        ),
        "action_hand_joint_target": (
            command.hand_qpos
            if command is not None and command.hand_qpos is not None
            else np.full(12, np.nan)
        ),
        "arm_eef_intent": arm_eef_intent if arm_eef_intent is not None else np.full(9, np.nan),
        "flag_frame_status": frame_status,
        "observation_timestamp_ns": row.observation_timestamp_ns,
        "action_timestamp_ns": action_timestamp_ns,
        "arm_timestamp_ns": arm["timestamp_ns"],
        "hand_timestamp_ns": hand["timestamp_ns"] if hand is not None else 0,
        "vr_timestamp_ns": vr["recv_ts_ns"] if vr is not None else 0,
        "camera_timestamp_ns": camera["timestamp_ns"] if camera is not None else 0,
        "camera_depth_frame_number": camera["depth_frame_number"] if camera else 0,
        "camera_color_frame_number": camera["color_frame_number"] if camera else 0,
        "vr_wrist_pos": vr["wrist_pos"] if vr else np.full(3, np.nan),
        "vr_wrist_rot6d": quat_wxyz_to_rot6d(vr["wrist_quat_wxyz"]) if vr else np.full(6, np.nan),
        "vr_landmarks": vr["landmarks"] if vr else np.full((21, 3), np.nan),
        "head_quat_wxyz": vr["head_quat_wxyz"] if vr else np.full(4, np.nan),
    }
    return EpisodeFrame(
        row.observation_timestamp_ns / 1e9,
        {k: np.array(v, dtype=DATASET_SPECS[k].dtype, copy=True) for k, v in values.items()},
        camera["rgb"] if camera else None,
        camera["depth"] if camera else None,
    )


def decode_record_sample(record):
    present = bool(record["camera_present"])
    return EpisodeFrame(
        float(record["timestamp"]),
        {
            name: np.array(record[name], dtype=DATASET_SPECS[name].dtype, copy=True)
            for name in SOURCE_FRAME_DATASET_NAMES
        },
        np.array(record["camera_rgb"], copy=True) if present else None,
        np.array(record["camera_depth"], copy=True) if present else None,
    )
