"""按帧索引比较两次运行的关节与末端误差。"""

from pathlib import Path

import numpy as np

from dexmani_real.geometry.kinematics import ArmFK, eef_pose_sequence
from dexmani_real.replay.data import Trajectory
from dexmani_real.utils.atomic_io import atomic_json_dump, atomic_publish


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
    count = min(len(trajectory), len(replay_data["arm_qpos"]))
    metrics = {
        "episode_path": trajectory.episode_path,
        "task_label": trajectory.task_label,
        "original_frames": len(trajectory),
        "replayed_frames": len(replay_data["arm_qpos"]),
        "matching_frames": count,
        "arm_joint": None,
        "hand_joint": None,
        "eef_position": None,
        "eef_orientation": None,
    }
    if not count:
        return metrics

    arm = replay_data["arm_qpos"][:count]
    hand = replay_data["hand_qpos"][:count]
    metrics["arm_joint"] = joint_errors(trajectory.arm_qpos[:count], arm)
    metrics["hand_joint"] = joint_errors(trajectory.hand_qpos[:count], hand)

    fk = ArmFK()
    original = eef_pose_sequence(trajectory.arm_qpos[:count], arm_fk=fk)
    replayed = eef_pose_sequence(arm, arm_fk=fk)
    position_error = np.linalg.norm(replayed[:, :3] - original[:, :3], axis=1) * 1000
    # FK 的 rot6d 已是正交单位列，无需逐帧重复归一化。
    first_a, second_a = original[:, 3:6], original[:, 6:9]
    first_b, second_b = replayed[:, 3:6], replayed[:, 6:9]
    trace = np.sum(first_a * first_b + second_a * second_b, axis=1)
    trace += np.sum(np.cross(first_a, second_a) * np.cross(first_b, second_b), axis=1)
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


def evaluate_replay(trajectory, replay_data, *, output_dir):
    if replay_data is None:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # 先保存实测数据，离线 FK 或指标计算失败时仍可重新评估。
    temporary = output / "replay_data.tmp.npz"
    np.savez_compressed(temporary, **replay_data)
    atomic_publish(temporary, output / "replay_data.npz")

    metrics = compute_metrics(trajectory, replay_data)
    atomic_json_dump(metrics, output / "metrics.json", ensure_ascii=False)
    print(f"Replay: {metrics['matching_frames']}/{metrics['original_frames']} matching frames")
    for name in ("arm_joint", "hand_joint", "eef_position", "eef_orientation"):
        print(f"  {name}: {metrics[name]}")
