"""Replay-state capture and consistency metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.ipc.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.planning.pose_utils import rot6d_to_quat_wxyz
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_SAFETY_REASON_BYTES = 256
_TRACKING_LAG_WINDOW_S = 0.4
_MIN_TRACKING_LAG_FRAMES = 6
_MIN_TRACKING_OVERLAP_FRAMES = 10
_MIN_TRACKING_SEQUENCE_FRAMES = 20


class ReplayRecorder:
    """Pre-allocated buffer capturing robot state during replay.

    Single-threaded (main loop), so no locking is needed.
    """

    def __init__(self, max_frames: int, has_hand: bool = False) -> None:
        self.max_frames = max_frames
        self.has_hand = has_hand
        self._count = 0

        self.arm_qpos = np.full((max_frames, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.eef_pos = np.full((max_frames, 3), np.nan, dtype=np.float64)
        self.eef_quat_wxyz = np.full((max_frames, 4), np.nan, dtype=np.float64)
        self.eef_rot6d = np.full((max_frames, 6), np.nan, dtype=np.float64)
        self.arm_cmd = np.full((max_frames, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.arm_sent_cmd = np.full((max_frames, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.arm_tracking_error = np.full((max_frames,), np.nan, dtype=np.float64)
        self.timestamps = np.full((max_frames,), np.nan, dtype=np.float64)

        # A rejected frame keeps state/candidate data but leaves arm_sent_cmd NaN.
        self.flag_safety_reject = np.zeros(max_frames, dtype=bool)
        self.safety_reject_reason: list[str | None] = [None] * max_frames

        self.hand_qpos: np.ndarray | None = None
        self.hand_cmd: np.ndarray | None = None
        if has_hand:
            self.hand_qpos = np.full((max_frames, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64)
            self.hand_cmd = np.full((max_frames, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64)

    def record(
        self,
        idx: int,
        arm_qpos: np.ndarray,
        eef_pos: np.ndarray,
        eef_rot6d: np.ndarray,
        arm_cmd: np.ndarray,
        hand_cmd: np.ndarray | None,
        ts: float,
        arm_sent_cmd: np.ndarray | None = None,
        arm_tracking_error: float | None = None,
        safety_reject_reason: str | None = None,
        hand_qpos: np.ndarray | None = None,
    ) -> None:
        """Record one frame of replay state.

        When *safety_reject_reason* is set the frame is marked as a safety
        gate rejection: observables + cmd are preserved (the action that was
        *attempted*), but ``arm_sent_cmd`` is left at its pre-allocated NaN
        (nothing was sent to the robot).  Downstream scripts can filter on
        ``flag_safety_reject`` or ``safety_reject_reason``.
        """
        if idx < 0:
            raise ValueError("replay frame index must be non-negative")
        if idx >= self.max_frames:
            return
        self.arm_qpos[idx] = arm_qpos
        self.eef_pos[idx] = eef_pos
        self.eef_rot6d[idx] = eef_rot6d
        try:
            rotation_matrix = _rot6d_to_matrix(eef_rot6d)
            quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
            self.eef_quat_wxyz[idx] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        except ValueError:
            self.eef_quat_wxyz[idx] = np.full(4, np.nan)
        self.arm_cmd[idx] = arm_cmd
        self.timestamps[idx] = ts
        if arm_sent_cmd is not None:
            self.arm_sent_cmd[idx] = arm_sent_cmd
        if arm_tracking_error is not None:
            self.arm_tracking_error[idx] = arm_tracking_error
        if self.has_hand and self.hand_qpos is not None and hand_qpos is not None:
            self.hand_qpos[idx] = hand_qpos
        if self.has_hand and self.hand_cmd is not None and hand_cmd is not None:
            self.hand_cmd[idx] = hand_cmd
        if safety_reject_reason is not None:
            self.flag_safety_reject[idx] = True
            self.safety_reject_reason[idx] = safety_reject_reason
        self._count = idx + 1

    @property
    def count(self) -> int:
        return self._count

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return truncated arrays as a dict."""
        n = self._count
        # Fixed bytes keep the NPZ readable with allow_pickle=False.
        reasons = np.array(
            [r.encode() if r else b"" for r in self.safety_reject_reason[:n]],
            dtype=f"S{_SAFETY_REASON_BYTES}",
        )
        result: dict[str, np.ndarray] = {
            "arm_qpos": self.arm_qpos[:n].copy(),
            "eef_pos": self.eef_pos[:n].copy(),
            "eef_quat_wxyz": self.eef_quat_wxyz[:n].copy(),
            "eef_rot6d": self.eef_rot6d[:n].copy(),
            "arm_cmd": self.arm_cmd[:n].copy(),
            "arm_sent_cmd": self.arm_sent_cmd[:n].copy(),
            "arm_tracking_error": self.arm_tracking_error[:n].copy(),
            "timestamp": self.timestamps[:n].copy(),
            "flag_safety_reject": self.flag_safety_reject[:n].copy(),
            "safety_reject_reason": reasons,
        }
        if self.hand_qpos is not None:
            result["hand_qpos"] = self.hand_qpos[:n].copy()
        if self.hand_cmd is not None:
            result["hand_cmd"] = self.hand_cmd[:n].copy()
        return result


@dataclass
class ReplayMetrics:
    """Evaluated consistency between replayed and original trajectory."""

    # Metadata
    episode_path: str = ""
    task_label: str = ""
    speed_factor: float = 1.0
    original_frames: int = 0
    replayed_frames: int = 0
    matching_frames: int = 0

    # Arm joint error (rad, converted to deg for reporting)
    arm_joint_mae_deg: np.ndarray = field(default_factory=lambda: np.zeros(ARM_JOINT_SHAPE))
    arm_joint_rmse_deg: np.ndarray = field(default_factory=lambda: np.zeros(ARM_JOINT_SHAPE))
    arm_joint_mae_overall_deg: float = 0.0
    arm_joint_rmse_overall_deg: float = 0.0

    # EEF position error (m → mm for reporting)
    eef_pos_error_mean_mm: float = 0.0
    eef_pos_error_max_mm: float = 0.0
    eef_pos_error_rmse_mm: float = 0.0

    # EEF orientation error (deg)
    eef_rot_error_mean_deg: float = 0.0
    eef_rot_error_max_deg: float = 0.0

    # Per-frame error arrays (1-D)
    eef_pos_error_per_frame_mm: np.ndarray | None = None
    eef_rot_error_per_frame_deg: np.ndarray | None = None

    # Hand joint error (deg, optional)
    hand_joint_mae_overall_deg: float | None = None
    hand_joint_rmse_overall_deg: float | None = None

    # Tracking lag
    tracking_lag_frames: int = 0
    tracking_lag_seconds: float = 0.0

    # Tracking error during replay (replay cmd vs actual, from inner-loop monitor)
    arm_tracking_error_mean_deg: float = 0.0
    arm_tracking_error_p95_deg: float = 0.0
    arm_tracking_error_max_deg: float = 0.0


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert (6,) rot6d to (3,3) rotation matrix via rot6d→quat→matrix."""
    q_wxyz = rot6d_to_quat_wxyz(rot6d)
    return Rotation.from_quat(np.roll(q_wxyz, -1)).as_matrix()  # wxyz→xyzw


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
        valid_ee = np.all(np.isfinite(orig_ee_pos), axis=1) & np.all(np.isfinite(rep_ee_pos), axis=1)
        if valid_ee.sum() > 0:
            pos_err = np.linalg.norm(orig_ee_pos[valid_ee] - rep_ee_pos[valid_ee], axis=1)
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
        valid_rot = np.all(np.isfinite(orig_rot6d), axis=1) & np.all(np.isfinite(rep_rot6d), axis=1)
        if valid_rot.sum() > 0:
            rot_errs = []
            for i in np.where(valid_rot)[0]:
                try:
                    rotation_a = _rot6d_to_matrix(orig_rot6d[i])
                    rotation_b = _rot6d_to_matrix(rep_rot6d[i])
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
            valid_h = np.all(np.isfinite(orig_h), axis=1) & np.all(np.isfinite(rep_h), axis=1)
            if valid_h.sum() > 0:
                diff_h = np.abs(orig_h[valid_h] - rep_h[valid_h])
                metrics.hand_joint_mae_overall_deg = float(np.rad2deg(np.mean(diff_h)))
                metrics.hand_joint_rmse_overall_deg = float(np.rad2deg(np.sqrt(np.mean(diff_h**2))))

    # Per-joint position RMSE gives physical following lag; median rejects a noisy joint.
    if frame_count >= _MIN_TRACKING_SEQUENCE_FRAMES:
        max_lag = max(int(np.ceil(fps * _TRACKING_LAG_WINDOW_S)), _MIN_TRACKING_LAG_FRAMES)
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
                rmse = float(np.sqrt(np.mean((original[finite] - replayed[finite]) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_lag = lag
            joint_lags.append(best_lag)
        peak_lag = int(np.median(joint_lags))
        metrics.tracking_lag_frames = peak_lag
        metrics.tracking_lag_seconds = float(peak_lag) / fps

    return metrics


def save_results(metrics: ReplayMetrics, replay_data: dict[str, np.ndarray], output_dir: str) -> None:
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
    if metrics.hand_joint_mae_overall_deg is not None and metrics.hand_joint_rmse_overall_deg is not None:
        metrics_dict["hand_joint"] = {
            "mae_overall_deg": round(metrics.hand_joint_mae_overall_deg, 4),
            "rmse_overall_deg": round(metrics.hand_joint_rmse_overall_deg, 4),
        }

    metrics_path = output_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics_dict, stream, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved: {metrics_path}")

    save_replay_data(replay_data, output_dir)


def save_replay_data(replay_data: dict[str, np.ndarray], output_dir: str) -> Path:
    """Persist captured replay samples even when consistency metrics are unavailable."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    npz_path = output_path / "replay_data.npz"
    np.savez_compressed(npz_path, **replay_data)
    print(f"Replay data saved: {npz_path}  ({replay_data['arm_qpos'].shape[0]} frames)")
    return npz_path
