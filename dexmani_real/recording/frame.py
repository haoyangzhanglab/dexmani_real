"""Research rows copied from the same observation used by the controller."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.recording.storage.schema import DATASET_SPECS


@dataclass
class EpisodeFrame:
    data: dict
    camera_rgb: np.ndarray
    camera_depth: np.ndarray


def build_episode_frame(
    row,
    command=None,
    *,
    frame_valid: bool,
):
    arm, hand, camera = (
        row.arm[0],
        None if row.hand is None else row.hand[0],
        row.camera,
    )
    if camera is None:
        raise ValueError("recording requires RGB-D from the current observation")
    contact_valid = hand is not None and bool(hand["tactile_aggregate_valid"])
    dense_valid = hand is not None and bool(hand["tactile_dense_valid"])
    values = {
        "arm_qpos": arm["qpos"],
        "arm_qvel": arm["qvel"],
        "arm_effort": arm["effort"],
        "hand_qpos": hand["qpos"] if hand is not None else np.full(12, np.nan),
        "hand_current": hand["current"] if hand is not None else np.full(12, np.nan),
        "hand_contact": hand["tactile_aggregate"] if contact_valid else np.full((5, 3), np.nan),
        "hand_tactile_force": hand["tactile_dense"]
        if dense_valid
        else np.full((5, 120, 3), np.nan),
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
        "frame_valid": frame_valid,
    }
    # Runtime flags may fail independently of joint telemetry. Never label a
    # nonfinite payload usable, or serialize an invalid payload as finite zeros.
    for name, valid in (("hand_contact", contact_valid), ("hand_tactile_force", dense_valid)):
        if valid and not np.isfinite(values[name]).all():
            raise ValueError(f"runtime tactile validity disagrees with {name} payload")
    return EpisodeFrame(
        {k: np.array(v, dtype=DATASET_SPECS[k].dtype, copy=True) for k, v in values.items()},
        camera["rgb"],
        camera["depth"],
    )


def decode_record_sample(record):
    return EpisodeFrame(
        {
            name: np.array(record[name], dtype=DATASET_SPECS[name].dtype, copy=True)
            for name in DATASET_SPECS
        },
        np.array(record["camera_rgb"], copy=True),
        np.array(record["camera_depth"], copy=True),
    )
