"""Replay capture buffers, consistency metrics, and result persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dexmani_real.planning.pose_utils import rot6d_to_rotmat
from dexmani_real.recording.transaction import atomic_json_dump
from dexmani_real.robot.replay_trajectory import TrajectoryData
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import ARM_JOINT_SHAPE

logger = get_logger(__name__)

_TRACKING_LAG_WINDOW_S = 0.4
_MIN_TRACKING_LAG_FRAMES = 6
_MIN_TRACKING_OVERLAP_FRAMES = 10
_MIN_TRACKING_SEQUENCE_FRAMES = 20
_TRACKING_ERROR_PERCENTILE = 95.0


@dataclass
class ReplayMetrics:
    """Evaluated consistency between replayed and original trajectory."""

    episode_path: str = ""
    task_label: str = ""
    speed_factor: float = 1.0
    original_frames: int = 0
    replayed_frames: int = 0
    matching_frames: int = 0

    arm_joint_mae_deg: np.ndarray = field(
        default_factory=lambda: np.zeros(ARM_JOINT_SHAPE)
    )
    arm_joint_rmse_deg: np.ndarray = field(
        default_factory=lambda: np.zeros(ARM_JOINT_SHAPE)
    )
    arm_joint_mae_overall_deg: float = 0.0
    arm_joint_rmse_overall_deg: float = 0.0

    eef_pos_error_mean_mm: float = 0.0
    eef_pos_error_max_mm: float = 0.0
    eef_pos_error_rmse_mm: float = 0.0

    eef_rot_error_mean_deg: float = 0.0
    eef_rot_error_max_deg: float = 0.0

    eef_pos_error_per_frame_mm: np.ndarray | None = None
    eef_rot_error_per_frame_deg: np.ndarray | None = None

    hand_joint_mae_overall_deg: float | None = None
    hand_joint_rmse_overall_deg: float | None = None

    tracking_lag_frames: int = 0
    tracking_lag_seconds: float = 0.0

    arm_tracking_error_mean_deg: float = 0.0
    arm_tracking_error_p95_deg: float = 0.0
    arm_tracking_error_max_deg: float = 0.0


def _geodesic_distance_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Geodesic angular distance between two rotation matrices in degrees."""
    cos_angle = 0.5 * (np.trace(rotation_a @ rotation_b.T) - 1.0)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def compute_metrics(
    original_arm_qpos: np.ndarray,
    replay_arm_qpos: np.ndarray,
    original_arm_ee: np.ndarray | None,
    replay_arm_ee_pos: np.ndarray,
    replay_arm_ee_rot6d: np.ndarray,
    fps: float,
    original_hand_qpos: np.ndarray | None = None,
    replay_hand_qpos: np.ndarray | None = None,
    episode_path: str = "",
    task_label: str = "",
    speed_factor: float = 1.0,
    arm_tracking_error: np.ndarray | None = None,
) -> ReplayMetrics:
    """Compare matching frame indices from the recorded and replayed streams."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    original_frames = original_arm_qpos.shape[0]
    replayed_frames = replay_arm_qpos.shape[0]
    frame_count = min(original_frames, replayed_frames)

    metrics = ReplayMetrics(
        episode_path=episode_path,
        task_label=task_label,
        speed_factor=speed_factor,
        original_frames=original_frames,
        replayed_frames=replayed_frames,
        matching_frames=frame_count,
    )

    if frame_count == 0:
        logger.warning("No frames to compare")
        return metrics

    orig_q = original_arm_qpos[:frame_count]
    rep_q = replay_arm_qpos[:frame_count]
    valid = np.all(np.isfinite(orig_q), axis=1) & np.all(np.isfinite(rep_q), axis=1)
    if valid.sum() > 0:
        diff = np.abs(orig_q[valid] - rep_q[valid])
        metrics.arm_joint_mae_deg = np.rad2deg(np.mean(diff, axis=0))
        metrics.arm_joint_rmse_deg = np.rad2deg(np.sqrt(np.mean(diff**2, axis=0)))
        metrics.arm_joint_mae_overall_deg = float(np.mean(metrics.arm_joint_mae_deg))
        metrics.arm_joint_rmse_overall_deg = float(np.mean(metrics.arm_joint_rmse_deg))

    if original_arm_ee is not None and original_arm_ee.shape[0] >= frame_count:
        orig_ee_pos = original_arm_ee[:frame_count, :3]
        rep_ee_pos = replay_arm_ee_pos[:frame_count]
        valid_ee = np.all(np.isfinite(orig_ee_pos), axis=1) & np.all(
            np.isfinite(rep_ee_pos), axis=1
        )
        if valid_ee.sum() > 0:
            pos_err = np.linalg.norm(
                orig_ee_pos[valid_ee] - rep_ee_pos[valid_ee], axis=1
            )
            metrics.eef_pos_error_per_frame_mm = pos_err * 1000.0
            metrics.eef_pos_error_mean_mm = float(np.mean(pos_err) * 1000.0)
            metrics.eef_pos_error_max_mm = float(np.max(pos_err) * 1000.0)
            metrics.eef_pos_error_rmse_mm = float(np.sqrt(np.mean(pos_err**2)) * 1000.0)

    if (
        original_arm_ee is not None
        and original_arm_ee.shape[0] >= frame_count
        and replay_arm_ee_rot6d.shape[0] >= frame_count
    ):
        orig_rot6d = original_arm_ee[:frame_count, 3:9]
        rep_rot6d = replay_arm_ee_rot6d[:frame_count]
        valid_rot = np.all(np.isfinite(orig_rot6d), axis=1) & np.all(
            np.isfinite(rep_rot6d), axis=1
        )
        if valid_rot.sum() > 0:
            rot_errs = []
            for i in np.where(valid_rot)[0]:
                try:
                    rotation_a = rot6d_to_rotmat(orig_rot6d[i])
                    rotation_b = rot6d_to_rotmat(rep_rot6d[i])
                    rot_errs.append(_geodesic_distance_deg(rotation_a, rotation_b))
                except ValueError:
                    rot_errs.append(np.nan)
            rot_errs_arr = np.array(rot_errs)
            finite = np.isfinite(rot_errs_arr)
            if finite.sum() > 0:
                metrics.eef_rot_error_per_frame_deg = rot_errs_arr
                metrics.eef_rot_error_mean_deg = float(np.mean(rot_errs_arr[finite]))
                metrics.eef_rot_error_max_deg = float(np.max(rot_errs_arr[finite]))

    if original_hand_qpos is not None and replay_hand_qpos is not None:
        hand_frame_count = min(original_hand_qpos.shape[0], replay_hand_qpos.shape[0])
        if hand_frame_count > 0:
            orig_h = original_hand_qpos[:hand_frame_count]
            rep_h = replay_hand_qpos[:hand_frame_count]
            valid_h = np.all(np.isfinite(orig_h), axis=1) & np.all(
                np.isfinite(rep_h), axis=1
            )
            if valid_h.sum() > 0:
                diff_h = np.abs(orig_h[valid_h] - rep_h[valid_h])
                metrics.hand_joint_mae_overall_deg = float(np.rad2deg(np.mean(diff_h)))
                metrics.hand_joint_rmse_overall_deg = float(
                    np.rad2deg(np.sqrt(np.mean(diff_h**2)))
                )

    if frame_count >= _MIN_TRACKING_SEQUENCE_FRAMES:
        max_lag = max(
            int(np.ceil(fps * _TRACKING_LAG_WINDOW_S)), _MIN_TRACKING_LAG_FRAMES
        )
        joint_lags: list[int] = []
        for joint_index in range(ARM_JOINT_SHAPE[0]):
            best_lag, best_rmse = 0, float("inf")
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
                if int(np.count_nonzero(finite)) < _MIN_TRACKING_OVERLAP_FRAMES:
                    continue
                rmse = float(
                    np.sqrt(np.mean((original[finite] - replayed[finite]) ** 2))
                )
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_lag = lag
            joint_lags.append(best_lag)
        peak_lag = int(np.median(joint_lags))
        metrics.tracking_lag_frames = peak_lag
        metrics.tracking_lag_seconds = float(peak_lag) / fps

    if arm_tracking_error is not None:
        finite_tracking_error = arm_tracking_error[np.isfinite(arm_tracking_error)]
        if finite_tracking_error.size:
            metrics.arm_tracking_error_mean_deg = float(
                np.rad2deg(np.mean(finite_tracking_error))
            )
            metrics.arm_tracking_error_p95_deg = float(
                np.rad2deg(
                    np.percentile(finite_tracking_error, _TRACKING_ERROR_PERCENTILE)
                )
            )
            metrics.arm_tracking_error_max_deg = float(
                np.rad2deg(np.max(finite_tracking_error))
            )

    return metrics


def report_consistency(metrics: ReplayMetrics) -> None:
    """Print a human-readable replay-vs-original consistency summary."""
    print("\n" + "=" * 60)
    print("Consistency Evaluation")
    print("=" * 60)
    print(
        f"  Frames: {metrics.replayed_frames} replayed / {metrics.original_frames} original"
    )
    print(
        f"  Arm joint MAE:  {np.round(metrics.arm_joint_mae_deg, 2)} deg  "
        f"(overall: {metrics.arm_joint_mae_overall_deg:.3f} deg)"
    )
    print(
        f"  Arm joint RMSE: {np.round(metrics.arm_joint_rmse_deg, 2)} deg  "
        f"(overall: {metrics.arm_joint_rmse_overall_deg:.3f} deg)"
    )
    if metrics.eef_pos_error_per_frame_mm is not None:
        print(
            f"  EEF pos error:  mean={metrics.eef_pos_error_mean_mm:.1f}mm  "
            f"max={metrics.eef_pos_error_max_mm:.1f}mm  "
            f"rmse={metrics.eef_pos_error_rmse_mm:.1f}mm"
        )
    if metrics.eef_rot_error_per_frame_deg is not None:
        print(
            f"  EEF rot error:  mean={metrics.eef_rot_error_mean_deg:.2f}°  "
            f"max={metrics.eef_rot_error_max_deg:.2f}°"
        )
    if metrics.hand_joint_mae_overall_deg is not None:
        print(f"  Hand joint MAE: {metrics.hand_joint_mae_overall_deg:.3f} deg")
    print(
        f"  Tracking lag:  {metrics.tracking_lag_frames} frames "
        f"({metrics.tracking_lag_seconds:.3f}s)"
    )
    if metrics.arm_tracking_error_mean_deg > 0:
        print(
            "  Replay tracking error (cmd vs actual): "
            f"mean={metrics.arm_tracking_error_mean_deg:.2f}°  "
            f"p95={metrics.arm_tracking_error_p95_deg:.2f}°  "
            f"max={metrics.arm_tracking_error_max_deg:.2f}°"
        )
    print("=" * 60)


def evaluate_replay(
    trajectory: TrajectoryData,
    replay_data: dict[str, np.ndarray] | None,
    *,
    evaluate_consistency: bool,
    output_dir: str,
) -> None:
    """Evaluate and persist samples captured by one physical replay."""
    if replay_data is None:
        print(
            "\nNo replay data collected (replay interrupted before any frames captured)."
        )
        return
    if replay_data["arm_qpos"].shape[0] == 0:
        print("\nSkipping metrics: no valid reference or replay data available")
        return
    if not evaluate_consistency:
        print("\nSkipping consistency metrics; saving captured replay data.")
        save_replay_data(replay_data, output_dir)
        return

    print("\nComputing consistency metrics...")
    try:
        metrics = compute_metrics(
            original_arm_qpos=trajectory.arm_qpos,
            replay_arm_qpos=replay_data["arm_qpos"],
            original_arm_ee=trajectory.arm_ee,
            replay_arm_ee_pos=replay_data["eef_pos"],
            replay_arm_ee_rot6d=replay_data["eef_rot6d"],
            fps=trajectory.fps,
            original_hand_qpos=trajectory.hand_qpos,
            replay_hand_qpos=replay_data.get("hand_qpos"),
            episode_path=trajectory.episode_path,
            task_label=trajectory.task_label,
            speed_factor=1.0,
            arm_tracking_error=replay_data.get("arm_tracking_error"),
        )
    except Exception:
        logger.error(
            "replay consistency evaluation failed; saving raw replay data",
            exc_info=True,
        )
        save_replay_data(replay_data, output_dir)
        raise

    report_consistency(metrics)
    save_results(metrics, replay_data, output_dir)


def save_replay_data(replay_data: dict[str, np.ndarray], output_dir: str) -> Path:
    """Persist captured replay samples even when consistency metrics are unavailable."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_path = output_path / "replay_data.npz"
    # Write to a sibling temp file that already ends in ".npz" (numpy appends
    # ".npz" to any name that does not), then atomically move it into place.
    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **replay_data)
    os.replace(tmp_path, npz_path)
    print(f"Replay data saved: {npz_path}  ({replay_data['arm_qpos'].shape[0]} frames)")
    return npz_path


def save_results(
    metrics: ReplayMetrics, replay_data: dict[str, np.ndarray], output_dir: str
) -> None:
    """Save replay data and consistency metrics to output directory.

    Produces:
        <output_dir>/metrics.json   — human-readable scalar metrics
        <output_dir>/replay_data.npz — full time-series arrays
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_dict: dict[str, object] = {
        "episode_path": metrics.episode_path,
        "task_label": metrics.task_label,
        "speed_factor": metrics.speed_factor,
        "original_frames": metrics.original_frames,
        "replayed_frames": metrics.replayed_frames,
        "matching_frames": metrics.matching_frames,
        "arm_joint": {
            "mae_per_joint_deg": np.round(metrics.arm_joint_mae_deg, 4).tolist(),
            "rmse_per_joint_deg": np.round(metrics.arm_joint_rmse_deg, 4).tolist(),
            "mae_overall_deg": round(metrics.arm_joint_mae_overall_deg, 4),
            "rmse_overall_deg": round(metrics.arm_joint_rmse_overall_deg, 4),
        },
        "eef_position": {
            "mean_error_mm": round(metrics.eef_pos_error_mean_mm, 2),
            "max_error_mm": round(metrics.eef_pos_error_max_mm, 2),
            "rmse_error_mm": round(metrics.eef_pos_error_rmse_mm, 2),
        },
        "eef_orientation": {
            "mean_error_deg": round(metrics.eef_rot_error_mean_deg, 2),
            "max_error_deg": round(metrics.eef_rot_error_max_deg, 2),
        },
        "tracking_lag": {
            "peak_lag_frames": metrics.tracking_lag_frames,
            "peak_lag_seconds": round(metrics.tracking_lag_seconds, 3),
        },
    }
    if metrics.arm_tracking_error_mean_deg > 0:
        metrics_dict["tracking_error"] = {
            "mean_deg": round(metrics.arm_tracking_error_mean_deg, 2),
            "p95_deg": round(metrics.arm_tracking_error_p95_deg, 2),
            "max_deg": round(metrics.arm_tracking_error_max_deg, 2),
        }
    if (
        metrics.hand_joint_mae_overall_deg is not None
        and metrics.hand_joint_rmse_overall_deg is not None
    ):
        metrics_dict["hand_joint"] = {
            "mae_overall_deg": round(metrics.hand_joint_mae_overall_deg, 4),
            "rmse_overall_deg": round(metrics.hand_joint_rmse_overall_deg, 4),
        }

    metrics_path = atomic_json_dump(
        metrics_dict, output_path / "metrics.json", ensure_ascii=False
    )
    print(f"\nMetrics saved: {metrics_path}")

    save_replay_data(replay_data, output_dir)
