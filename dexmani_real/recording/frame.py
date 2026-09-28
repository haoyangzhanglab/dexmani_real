"""Research rows copied from the same observation used by the controller."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.recording.storage.schema import DATASET_SPECS


@dataclass(frozen=True)
class EpisodeFrame:
    data: dict
    camera_rgb: np.ndarray
    camera_depth: np.ndarray
    dispatch: tuple[int, int] = (0, 0)
    selected: dict | None = None
    dispatch_detail: dict | None = None


def build_episode_frame(
    row,
    command=None,
    result=None,
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
            if command is not None
            and command.arm_qpos is not None
            and result is not None
            and result.arm != 0
            else np.full(7, np.nan)
        ),
        "action_hand_joint_target": (
            command.hand_qpos
            if command is not None
            and command.hand_qpos is not None
            and result is not None
            and result.hand != 0
            else np.full(12, np.nan)
        ),
    }
    # Runtime flags may fail independently of joint telemetry. Never label a
    # nonfinite payload usable, or serialize an invalid payload as finite zeros.
    for name, valid in (("hand_contact", contact_valid), ("hand_tactile_force", dense_valid)):
        if valid and not np.isfinite(values[name]).all():
            raise ValueError(f"runtime tactile validity disagrees with {name} payload")
    data = {k: np.array(v, dtype=DATASET_SPECS[k].dtype, copy=True) for k, v in values.items()}
    for value in data.values():
        value.flags.writeable = False
    return EpisodeFrame(
        data,
        camera["rgb"],
        camera["depth"],
        (int(result.arm), int(result.hand)) if result is not None else (0, 0),
        {
            name: (None if getattr(command, name) is None else getattr(command, name).tolist())
            for name in ("arm_qpos", "hand_qpos")
        }
        if command is not None
        else None,
        {
            "arm_code": result.arm_code,
            "hand_status": None if result.hand_status is None else result.hand_status.value,
        }
        if result is not None
        else None,
    )
