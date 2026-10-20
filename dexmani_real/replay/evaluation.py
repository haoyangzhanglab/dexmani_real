"""Offline replay pose derivation, consistency metrics and result persistence."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.pose import rot6d_to_rotmat
from dexmani_real.replay.trajectory import TrajectoryData
from dexmani_real.robot.model import ARM_JOINT_SHAPE
from dexmani_real.utils.atomic_io import atomic_json_dump

ALIGNMENT_WINDOW_S = 0.4
MIN_ALIGNMENT_OFFSET_FRAMES = 6
MIN_ALIGNMENT_OVERLAP_FRAMES = 10
MIN_ALIGNMENT_SEQUENCE_FRAMES = 20
TARGET_GAP_PERCENTILE = 95.0


def geodesic_distance_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Geodesic angular distance between two rotation matrices in degrees."""
    cos_angle = 0.5 * (np.trace(rotation_a @ rotation_b.T) - 1.0)
    return float(np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def compute_metrics(
    trajectory: TrajectoryData, replay_data: dict, *, hand_start_duration_s=0.0
) -> dict:
    """Return JSON-ready cross-run errors by frame index, not hardware response delays."""
    fps = trajectory.fps
    original_frames = len(trajectory.arm_qpos)
    replayed_frames = len(replay_data["arm_qpos"])
    count = min(original_frames, replayed_frames)
    metrics = {
        "hand_start": {
            "requested_ramp_duration_s": hand_start_duration_s,
            "target": "first_recorded_action" if hand_start_duration_s > 0 else "none",
            "included_in_replay_rows": False,
        },
        "episode_path": trajectory.episode_path,
        "task_label": trajectory.task_label,
        "original_frames": original_frames,
        "replayed_frames": replayed_frames,
        "matching_frames": count,
        "valid_frames": {"arm": 0, "hand": 0, "eef_position": 0, "eef_rotation": 0},
        "arm_joint": {
            "mae_per_joint_deg": [None] * ARM_JOINT_SHAPE[0],
            "rmse_per_joint_deg": [None] * ARM_JOINT_SHAPE[0],
            "mae_overall_deg": None,
            "mean_joint_rmse_deg": None,
        },
        "eef_position": {"mean_error_mm": None, "max_error_mm": None, "rmse_error_mm": None},
        "eef_orientation": {"mean_error_deg": None, "max_error_deg": None},
        "trajectory_alignment": {
            "per_active_joint_offset_frames": {},
            "median_offset_frames": None,
        },
    }
    if count == 0:
        return metrics

    orig_q, rep_q = trajectory.arm_qpos[:count], replay_data["arm_qpos"][:count]
    valid = np.isfinite(orig_q).all(axis=1) & np.isfinite(rep_q).all(axis=1)
    metrics["valid_frames"]["arm"] = int(valid.sum())
    if valid.any():
        diff = np.abs(orig_q[valid] - rep_q[valid])
        mae = np.rad2deg(np.mean(diff, axis=0))
        rmse = np.rad2deg(np.sqrt(np.mean(diff**2, axis=0)))
        metrics["arm_joint"].update(
            mae_per_joint_deg=np.round(mae, 4).tolist(),
            rmse_per_joint_deg=np.round(rmse, 4).tolist(),
            mae_overall_deg=round(float(np.mean(mae)), 4),
            mean_joint_rmse_deg=round(float(np.mean(rmse)), 4),
        )

    orig_pos, rep_pos = trajectory.arm_ee[:count, :3], replay_data["eef_pos"][:count]
    valid = np.isfinite(orig_pos).all(axis=1) & np.isfinite(rep_pos).all(axis=1)
    metrics["valid_frames"]["eef_position"] = int(valid.sum())
    if valid.any():
        error = np.linalg.norm(orig_pos[valid] - rep_pos[valid], axis=1)
        metrics["eef_position"].update(
            mean_error_mm=round(float(np.mean(error) * 1000), 2),
            max_error_mm=round(float(np.max(error) * 1000), 2),
            rmse_error_mm=round(float(np.sqrt(np.mean(error**2)) * 1000), 2),
        )

    orig_rot, rep_rot = trajectory.arm_ee[:count, 3:9], replay_data["eef_rot6d"][:count]
    valid = np.isfinite(orig_rot).all(axis=1) & np.isfinite(rep_rot).all(axis=1)
    errors = []
    for index in np.flatnonzero(valid):
        try:
            errors.append(
                geodesic_distance_deg(
                    rot6d_to_rotmat(orig_rot[index]),
                    rot6d_to_rotmat(rep_rot[index]),
                )
            )
        except ValueError:
            pass
    finite_errors = np.asarray(errors)[np.isfinite(errors)]
    metrics["valid_frames"]["eef_rotation"] = len(finite_errors)
    if len(finite_errors):
        metrics["eef_orientation"].update(
            mean_error_deg=round(float(np.mean(finite_errors)), 2),
            max_error_deg=round(float(np.max(finite_errors)), 2),
        )

    hand_count = min(len(trajectory.hand_qpos), len(replay_data["hand_qpos"]))
    orig_hand, rep_hand = trajectory.hand_qpos[:hand_count], replay_data["hand_qpos"][:hand_count]
    valid = np.isfinite(orig_hand).all(axis=1) & np.isfinite(rep_hand).all(axis=1)
    metrics["valid_frames"]["hand"] = int(valid.sum())
    if valid.any():
        diff = np.abs(orig_hand[valid] - rep_hand[valid])
        metrics["hand_joint"] = {
            "mae_overall_deg": round(float(np.rad2deg(np.mean(diff))), 4),
            "pooled_rmse_deg": round(float(np.rad2deg(np.sqrt(np.mean(diff**2)))), 4),
        }

    offsets = trajectory_alignment_offsets(orig_q, rep_q, fps)
    metrics["trajectory_alignment"]["per_active_joint_offset_frames"] = offsets
    if offsets:
        metrics["trajectory_alignment"]["median_offset_frames"] = int(
            np.median(list(offsets.values()))
        )

    gap_count = min(len(replay_data["arm_cmd"]), replayed_frames)
    commands, measured = replay_data["arm_cmd"][:gap_count], replay_data["arm_qpos"][:gap_count]
    valid = np.isfinite(commands).all(axis=1) & np.isfinite(measured).all(axis=1)
    if valid.any():
        gaps = np.max(np.abs(commands[valid] - measured[valid]), axis=1)
        metrics["pre_submit_arm_target_gap"] = {
            "mean_deg": round(float(np.rad2deg(np.mean(gaps))), 2),
            "p95_deg": round(float(np.rad2deg(np.percentile(gaps, TARGET_GAP_PERCENTILE))), 2),
            "max_deg": round(float(np.rad2deg(np.max(gaps))), 2),
        }

    def json_defined(value):
        if isinstance(value, dict):
            return {k: json_defined(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_defined(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value

    return json_defined(metrics)


def report_consistency(metrics: dict) -> None:
    print("\nCross-run Consistency by Frame Index")
    print(
        f"  Frames: {metrics['replayed_frames']} replayed / {metrics['original_frames']} original"
    )
    print(f"  Valid frames: {metrics['valid_frames']}")
    print("  Joint errors use absolute SDK angles; EEF errors use geometric pose equivalence.")
    print("  Frame alignment compares separate runs; it does not measure hardware response delay.")
    print("  None means unavailable, not zero error.")
    arm = metrics["arm_joint"]
    print(f"  Arm MAE (deg): {arm['mae_per_joint_deg']}; overall={arm['mae_overall_deg']}")
    print(
        f"  Arm RMSE (deg): {arm['rmse_per_joint_deg']}; mean_joint_rmse={arm['mean_joint_rmse_deg']}"
    )
    print(f"  EEF position: {metrics['eef_position']}")
    print(f"  EEF orientation: {metrics['eef_orientation']}")
    print(f"  Trajectory alignment: {metrics['trajectory_alignment']}")
    if "hand_joint" in metrics:
        print(f"  Hand: {metrics['hand_joint']}")
    if "pre_submit_arm_target_gap" in metrics:
        print(
            f"  Arm submitted target vs consumed pre-submit feedback: {metrics['pre_submit_arm_target_gap']}"
        )


def evaluate_replay(
    trajectory: TrajectoryData,
    replay_data: dict[str, np.ndarray] | None,
    *,
    output_dir: str,
    hand_start_duration_s: float = 0.0,
) -> None:
    if replay_data is None:
        print("\nNo replay data collected (replay interrupted before any frames captured).")
        return
    # Preserve execution/termination evidence even when preparation produced zero rows.
    save_replay_data(replay_data, output_dir)
    if replay_data["arm_qpos"].shape[0] == 0:
        print("\nSkipping metrics: no valid reference or replay data available")
        return
    try:
        fk = make_arm_fk()
        count = len(replay_data["arm_qpos"])
        eef_pos = np.empty((count, 3), dtype=np.float64)
        eef_rot6d = np.empty((count, 6), dtype=np.float64)
        eef_quat_wxyz = np.full((count, 4), np.nan, dtype=np.float64)
        for index, qpos in enumerate(replay_data["arm_qpos"]):
            eef_pos[index], eef_rot6d[index] = fk.compute(qpos)
            try:
                quat_xyzw = Rotation.from_matrix(rot6d_to_rotmat(eef_rot6d[index])).as_quat()
                eef_quat_wxyz[index] = quat_xyzw[[3, 0, 1, 2]]
            except ValueError:
                pass
    except Exception as exc:
        raise RuntimeError(
            f"replay EEF derivation failed after physical capture was saved: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    replay_data.update(eef_pos=eef_pos, eef_rot6d=eef_rot6d, eef_quat_wxyz=eef_quat_wxyz)
    save_replay_data(replay_data, output_dir)
    metrics = compute_metrics(trajectory, replay_data, hand_start_duration_s=hand_start_duration_s)
    report_consistency(metrics)
    metrics_path = atomic_json_dump(metrics, Path(output_dir) / "metrics.json", ensure_ascii=False)
    print(f"\nMetrics saved: {metrics_path}")


def save_replay_data(replay_data: dict[str, np.ndarray], output_dir: str) -> Path:
    """Persist captured replay samples even when consistency metrics are unavailable."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_path = output_path / "replay_data.npz"
    # NumPy appends .npz; match the temporary suffix before atomic publication.
    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **replay_data)
    os.replace(tmp_path, npz_path)
    print(f"Replay data saved: {npz_path}  ({replay_data['arm_qpos'].shape[0]} frames)")
    return npz_path


def trajectory_alignment_offsets(orig_q, rep_q, fps):
    """RMSE alignment within nominal-fps bounds; positive lags match higher replay indices."""
    frame_count = len(orig_q)
    lags = {}
    if frame_count >= MIN_ALIGNMENT_SEQUENCE_FRAMES:
        max_lag = max(int(np.ceil(fps * ALIGNMENT_WINDOW_S)), MIN_ALIGNMENT_OFFSET_FRAMES)
        for joint_index in range(ARM_JOINT_SHAPE[0]):
            original_finite = orig_q[np.isfinite(orig_q[:, joint_index]), joint_index]
            replay_finite = rep_q[np.isfinite(rep_q[:, joint_index]), joint_index]
            if min(len(original_finite), len(replay_finite)) < MIN_ALIGNMENT_OVERLAP_FRAMES:
                continue
            if min(np.ptp(original_finite), np.ptp(replay_finite)) < np.deg2rad(1):
                continue  # A stationary axis cannot identify temporal lag.
            best_lag = None
            best_key = (float("inf"), float("inf"), 0)
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    original = orig_q[-lag:, joint_index]
                    replayed = rep_q[:lag, joint_index]
                elif lag > 0:
                    original = orig_q[:-lag, joint_index]
                    replayed = rep_q[lag:, joint_index]
                else:
                    original = orig_q[:, joint_index]
                    replayed = rep_q[:, joint_index]
                finite = np.isfinite(original) & np.isfinite(replayed)
                if int(np.count_nonzero(finite)) < MIN_ALIGNMENT_OVERLAP_FRAMES:
                    continue
                rmse = float(np.sqrt(np.mean((original[finite] - replayed[finite]) ** 2)))
                candidate_key = (rmse, abs(lag), lag)
                if candidate_key < best_key:
                    best_key = candidate_key
                    best_lag = lag
            if best_lag is not None:
                lags[joint_index] = best_lag
    return lags
