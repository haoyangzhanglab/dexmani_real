"""按帧索引比较两次运行的关节与末端误差。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.geometry.kinematics import ArmFK, eef_poses

if TYPE_CHECKING:
    from dexmani_real.replay.trajectory import Trajectory


def joint_errors(original, replayed):
    error = np.rad2deg(replayed - original)
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    return {
        "mae_per_joint_deg": mae.tolist(),
        "rmse_per_joint_deg": rmse.tolist(),
        "mae_overall_deg": float(mae.mean()),
        "pooled_rmse_deg": float(np.sqrt(np.mean(error**2))),
    }


def compute_metrics(trajectory: Trajectory, replay_data: dict) -> dict:
    arm = np.asarray(replay_data["arm_qpos"])
    hand = np.asarray(replay_data["hand_qpos"])
    for name, values, joints in (("arm_qpos", arm, 7), ("hand_qpos", hand, 12)):
        if (
            values.ndim != 2
            or values.shape[1] != joints
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"replay {name} must be finite real [rows,{joints}]")
    if len(arm) != len(hand):
        raise ValueError("replay arm and hand measurements must have the same row count")
    count = min(len(trajectory), len(arm))
    metrics = {
        "episode_path": trajectory.episode_path,
        "task_label": trajectory.task_label,
        "original_frames": len(trajectory),
        "replayed_frames": len(arm),
        "matching_frames": count,
        "arm_joint": None,
        "hand_joint": None,
        "eef_position": None,
        "eef_orientation": None,
    }
    if not count:
        return metrics

    arm, hand = arm[:count], hand[:count]
    metrics["arm_joint"] = joint_errors(trajectory.arm_qpos[:count], arm)
    metrics["hand_joint"] = joint_errors(trajectory.hand_qpos[:count], hand)

    fk = ArmFK()
    original = eef_poses(trajectory.arm_qpos[:count], arm_fk=fk)
    replayed = eef_poses(arm, arm_fk=fk)
    position_error = np.linalg.norm(replayed[:, :3, 3] - original[:, :3, 3], axis=1) * 1000
    trace = np.sum(original[:, :3, :3] * replayed[:, :3, :3], axis=(1, 2))
    rotation_error = np.rad2deg(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
    metrics["eef_position"] = {
        "mean_error_mm": float(position_error.mean()),
        "max_error_mm": float(position_error.max()),
        "rmse_error_mm": float(np.sqrt(np.mean(position_error**2))),
    }
    metrics["eef_orientation"] = {
        "mean_error_deg": float(rotation_error.mean()),
        "max_error_deg": float(rotation_error.max()),
    }
    return metrics
