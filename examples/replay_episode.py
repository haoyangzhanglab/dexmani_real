#!/usr/bin/env python3
"""Inspect or replay one DexMani HDF5 episode.

Spawn-only architecture: arm_loop + hand_loop processes with SharedStorage
and the SafetyState machine. Commands flow through arm_action_q / hand_cmd_ring;
state is read from arm_state_ring / hand_state_ring. No direct SDK access from
the main process.

Two modes, both performed by this single self-contained script:

- **Dry-run (default)**: offline validation — checks array shapes, finite values,
  and performs a fast shape-only pass. No hardware, no workers.
- **Live** (``--live --source sent``): reruns dense geometry and provenance
  preflight, spawns arm/hand workers, replays the recorded joint commands on the
  real robot, captures actual robot state, and evaluates consistency metrics
  (joint MAE/RMSE, EEF position/orientation error, tracking lag).

Usage:
    python examples/replay_episode.py episodes/episode_20260729_213332
    python examples/replay_episode.py episodes/episode_20260729_213332 --max-frames 200
    python examples/replay_episode.py episodes/episode_20260729_213332 --source sent --live
    python examples/replay_episode.py episodes/episode_20260729_213332 --source sent --live --output results/my_replay/

Live replay controls:
    Q     clean exit (save partial results)
    H     return arm to home (post-replay prompt)
    ESC   emergency stop
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real import ASSET_DIR
from dexmani_real.config.runtime import (ResolvedRuntimeConfig,
                                         resolve_runtime_config)
from dexmani_real.planning import (Pose, TeleopProfile, XArm7MotionPlanner,
                                   XArm7PlannerConfig)
from dexmani_real.planning.pose_utils import rot6d_to_quat_wxyz
from dexmani_real.policy.safety import (SafetyGate, advance_run_generation,
                                        planner_action_safety_gate,
                                        publish_hand_home_and_wait_applied,
                                        publish_joint_targets)
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.recording.transaction import atomic_json_dump
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.arm_sdk import ArmLoopConfig
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.homing import send_arm_home
from dexmani_real.robot.safety import SafetyState, require_transition
from dexmani_real.runtime.processes import (
    ShutdownReport,
    WorkerSpec,
    build_processes,
    start_processes,
)
from dexmani_real.runtime.supervisor import (shutdown_processes,
                                             wait_subsystem_ready)
from dexmani_real.shm.shared_storage import (SharedStorage,
                                             SharedStorageConfig,
                                             read_arm_state_dict,
                                             read_hand_state_dict)
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.hand_health import (validate_arm_feedback,
                                            validate_hand_feedback)
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import (ARM_EE_SHAPE, ARM_JOINT_SHAPE,
                                       HAND_JOINT_SHAPE)

logger = get_logger(__name__)

# ========================================================================
# Constants & helpers (shared across all sections)
# ========================================================================

DEFAULT_OUTPUT_DIR = "replay_results"

_MIN_EPISODE_RATE_HZ = 1.0
_MAX_EPISODE_RATE_HZ = 100.0
_OFFLINE_FALLBACK_RATE_HZ = 16.0

_MODEL_PROVENANCE_KEYS = (
    "arm_hand_collision_urdf_sha256",
    "arm_hand_urdf_sha256",
    "arm_hand_srdf_sha256",
)
_PREFLIGHT_MAX_JOINT_STEP_RAD = 0.02

_STATUS_INTERVAL_FRAMES = 50
_WAIT_POLL_INTERVAL_S = 0.01

_TRACKING_LAG_WINDOW_S = 0.4
_MIN_TRACKING_LAG_FRAMES = 6
_MIN_TRACKING_OVERLAP_FRAMES = 10
_MIN_TRACKING_SEQUENCE_FRAMES = 20
_SAFETY_REASON_BYTES = 256

_TRACKING_ERROR_PERCENTILE = 95.0


def _positive_finite_float(value: str) -> float:
    """Validate and return a finite positive float for argparse."""
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {value!r}")
    return parsed


def _positive_int(value: str) -> int:
    """Validate and return a positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value!r}")
    return parsed


def _is_sha256(value: str | None) -> bool:
    """Return True if *value* is a 64-character hex SHA-256 string."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def preflight_model_paths() -> tuple[Path, ...]:
    """Return the three URDF/SRDF model paths used for geometry provenance checks."""
    model_dir = ASSET_DIR / "robots" / "xhand"
    return (
        model_dir / "xarm7_xhand_collision.urdf",
        model_dir / "xarm7_xhand_right.urdf",
        model_dir / "xarm7_xhand.srdf",
    )


# ========================================================================
# Section 1: Trajectory data loading
# ========================================================================


@dataclass
class TrajectoryData:
    """Preloaded state and command streams used by replay and evaluation."""

    episode_path: str
    num_frames: int
    fps: float
    task_label: str
    action_arm_joint: np.ndarray
    action_hand_joint: np.ndarray | None
    arm_qpos: np.ndarray
    hand_qpos: np.ndarray | None
    arm_ee: np.ndarray | None
    action_source: str | None = None
    resolved_config_sha256: str | None = None
    model_provenance: tuple[tuple[str, str], ...] = ()
    send_mask: np.ndarray | None = None

    @property
    def has_hand(self) -> bool:
        """Whether a fixed-shape hand action stream is present."""
        return self.has_hand_actions

    @property
    def has_hand_actions(self) -> bool:
        """Whether a fixed-shape hand action dataset is present."""
        return self.action_hand_joint is not None


def resolve_episode_path(raw_path: str) -> tuple[str, str]:
    """Validate and name one schema-v16 episode directory."""
    path = Path(raw_path)
    if not path.is_dir():
        raise ValueError(f"episode must be a schema-v16 directory: {path}")
    return str(path), path.name


def load_trajectory(
    episode_path: str,
    max_frames: int | None = None,
    source: str = "cmd",
    *,
    require_live_validity: bool = False,
    require_exact_source: bool = False,
) -> TrajectoryData:
    """Load one command trajectory from a schema-v16 episode."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    if source not in {"cmd", "sent"}:
        raise ValueError("source must be 'cmd' or 'sent'")

    resolved_path, _episode_name = resolve_episode_path(episode_path)
    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"Episode not found: {episode_path}")

    with EpisodeReader(resolved_path) as reader:
        if not reader.meets_min_duration:
            logger.warning(
                "Episode %s is internally readable but below the configured minimum recording duration",
                resolved_path,
            )
        if require_live_validity:
            reader.require_valid(purpose="live replay")
        h5 = reader.h5f
        meta = h5.get("meta")
        num_frames_attr = int(meta.attrs.get("num_frames", 0)) if meta is not None else 0
        fps = float(reader.timing.rate_hz)
        if not _MIN_EPISODE_RATE_HZ <= fps <= _MAX_EPISODE_RATE_HZ:
            if require_live_validity:
                raise ValueError(f"live replay requires a valid episode rate, got {fps!r} Hz")
            logger.warning(
                "Implausible episode rate %.3f Hz; using %.1f Hz for offline replay",
                fps,
                _OFFLINE_FALLBACK_RATE_HZ,
            )
            fps = _OFFLINE_FALLBACK_RATE_HZ
        task_label = str(meta.attrs.get("task_label", "")) if meta is not None else ""
        resolved_config_sha256 = None
        model_provenance: tuple[tuple[str, str], ...] = ()
        if meta is not None:
            raw_hash = meta.attrs.get("resolved_config_sha256")
            resolved_config_sha256 = None if raw_hash is None else str(raw_hash)
            model_provenance = tuple(
                sorted(
                    (name.removeprefix("provenance_"), str(meta.attrs[name]))
                    for name in meta.attrs
                    if name.startswith("provenance_arm_hand_")
                )
            )

        arm_action_key = "action_arm_joint_sent" if source == "sent" else "action_arm_joint"
        action_source = source
        if arm_action_key not in h5 and source == "sent":
            if require_live_validity or require_exact_source:
                raise ValueError("the requested replay operation requires /action_arm_joint_sent")
            logger.warning("/action_arm_joint_sent is absent; using /action_arm_joint for offline replay")
            arm_action_key = "action_arm_joint"
            action_source = "cmd"
        for key in (arm_action_key, "arm_qpos"):
            if key not in h5:
                raise ValueError(f"episode missing required dataset: /{key}")

        source_frames = int(h5[arm_action_key].shape[0])
        total_frames = source_frames if num_frames_attr == 0 else min(source_frames, num_frames_attr)
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        action_arm_joint = np.asarray(h5[arm_action_key][:total_frames], dtype=np.float64)
        arm_qpos = np.asarray(h5["arm_qpos"][:total_frames], dtype=np.float64)
        action_hand_joint = (
            np.asarray(h5["action_hand_joint"][:total_frames], dtype=np.float64) if "action_hand_joint" in h5 else None
        )
        hand_qpos = np.asarray(h5["hand_qpos"][:total_frames], dtype=np.float64) if "hand_qpos" in h5 else None
        arm_ee = np.asarray(h5["arm_ee"][:total_frames], dtype=np.float64) if "arm_ee" in h5 else None
        send_mask = (
            np.asarray(h5["flag_action_queued"][:total_frames], dtype=bool)
            if "flag_action_queued" in h5
            else None
        )

    arrays: dict[str, tuple[np.ndarray, tuple[int, ...]]] = {
        "arm action": (action_arm_joint, (total_frames, *ARM_JOINT_SHAPE)),
        "arm state": (arm_qpos, (total_frames, *ARM_JOINT_SHAPE)),
    }
    if action_hand_joint is not None:
        arrays["hand action"] = (action_hand_joint, (total_frames, *HAND_JOINT_SHAPE))
    if hand_qpos is not None:
        arrays["hand state"] = (hand_qpos, (total_frames, *HAND_JOINT_SHAPE))
    if arm_ee is not None:
        arrays["arm EEF"] = (arm_ee, (total_frames, *ARM_EE_SHAPE))
    if send_mask is not None:
        arrays["send mask"] = (send_mask, (total_frames,))
    for name, (array, expected_shape) in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(f"episode {name} has shape {array.shape}, expected {expected_shape}")

    trajectory = TrajectoryData(
        episode_path=resolved_path,
        num_frames=total_frames,
        fps=fps,
        task_label=task_label,
        action_arm_joint=action_arm_joint,
        action_hand_joint=action_hand_joint,
        arm_qpos=arm_qpos,
        hand_qpos=hand_qpos,
        arm_ee=arm_ee,
        action_source=action_source,
        resolved_config_sha256=resolved_config_sha256,
        model_provenance=model_provenance,
        send_mask=send_mask,
    )
    logger.info(
        "Loaded trajectory: %d frames, fps=%.1f, task=%s, hand=%s, ee=%s",
        trajectory.num_frames,
        trajectory.fps,
        trajectory.task_label or "(none)",
        ("yes" if trajectory.has_hand_actions else "no"),
        "yes" if trajectory.arm_ee is not None else "no",
    )
    return trajectory


# ========================================================================
# Section 2: Replay metrics
# ========================================================================


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

    metrics_path = atomic_json_dump(metrics_dict, output_path / "metrics.json", ensure_ascii=False)
    print(f"\nMetrics saved: {metrics_path}")

    save_replay_data(replay_data, output_dir)


# ========================================================================
# Section 3: Preflight validation
# ========================================================================


def modeled_hand_actions(
    trajectory: TrajectoryData,
    *,
    no_hand: bool,
    home_qpos_rad: np.ndarray,
) -> np.ndarray:
    """Return the hand action stream used for geometry preflight.

    When a recorded hand action stream is available and *no_hand* is false,
    returns the recorded actions. Otherwise tiles the configured home pose
    across all frames so collision/table checks degrade safely.
    """
    if not no_hand and trajectory.has_hand:
        assert trajectory.action_hand_joint is not None
        return np.asarray(trajectory.action_hand_joint, dtype=np.float64)
    home = np.asarray(home_qpos_rad, dtype=np.float64)
    if home.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(home)):
        raise ValueError(f"configured hand home must be finite shape {HAND_JOINT_SHAPE}")
    return np.repeat(home[None, :], trajectory.num_frames, axis=0)


def require_explicit_hand_mode(trajectory: TrajectoryData, *, no_hand: bool) -> None:
    """Fail closed when an episode cannot prove that live hand data was recorded."""
    if no_hand:
        return
    if not trajectory.has_hand_actions:
        raise ValueError("episode has no hand action stream; pass --no-hand only after securing the physical hand")


def _verify_trajectory_provenance(trajectory: TrajectoryData, *, provenance_sha256: str) -> None:
    """Fail-closed provenance gate: source stream, config hash, and model hashes."""
    if trajectory.action_source != "sent":
        raise ValueError("live replay requires the exact submitted action stream ('sent')")
    if trajectory.num_frames <= 0:
        raise ValueError("live replay trajectory is empty")
    arm_actions = np.asarray(trajectory.action_arm_joint)
    expected_arm_shape = (trajectory.num_frames, *ARM_JOINT_SHAPE)
    if arm_actions.shape != expected_arm_shape or not np.all(np.isfinite(arm_actions)):
        raise ValueError(f"live replay arm actions must be finite shape {expected_arm_shape}")
    if not _is_sha256(trajectory.resolved_config_sha256):
        raise ValueError("live replay recording provenance lacks a valid resolved_config_sha256")
    if trajectory.resolved_config_sha256 != provenance_sha256:
        raise ValueError("live replay config provenance mismatch: recorded resolved config differs from replay config")

    recorded_models = dict(trajectory.model_provenance)
    missing_models = [name for name in _MODEL_PROVENANCE_KEYS if not _is_sha256(recorded_models.get(name))]
    if missing_models:
        raise ValueError(f"live replay model provenance is incomplete: {missing_models}")
    current_models = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(_MODEL_PROVENANCE_KEYS, preflight_model_paths())
    }
    mismatched_models = [name for name in _MODEL_PROVENANCE_KEYS if recorded_models[name] != current_models[name]]
    if mismatched_models:
        raise ValueError(f"live replay model provenance mismatch: {mismatched_models}")


def verify_live_replay_preflight(
    trajectory: TrajectoryData,
    runtime: ResolvedRuntimeConfig,
    *,
    no_hand: bool,
    provenance_sha256: str,
) -> None:
    """Fail-closed validation immediately before spawning hardware workers.

    Checks: hand-mode attestation, provenance (source/config/models), every
    adjacent-frame pair via dense sub-step interpolation for workspace,
    self-collision, and table penetration. Called once before worker startup;
    any rejection prevents hardware access entirely.
    """
    require_explicit_hand_mode(trajectory, no_hand=no_hand)
    if not no_hand and not bool(runtime.policy.hand_enabled):
        raise ValueError("runtime policy.hand_enabled=false requires explicit no-hand acknowledgement")
    _verify_trajectory_provenance(trajectory, provenance_sha256=provenance_sha256)
    modeled_hand = modeled_hand_actions(
        trajectory,
        no_hand=no_hand,
        home_qpos_rad=np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)),
    )
    workspace = np.array(
        [
            [runtime.policy.workspace.x_min, runtime.policy.workspace.x_max],
            [runtime.policy.workspace.y_min, runtime.policy.workspace.y_max],
            [runtime.policy.workspace.z_min, runtime.policy.workspace.z_max],
        ],
        dtype=np.float64,
    )
    model_dir = ASSET_DIR / "robots" / "xhand"
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(model_dir / "xarm7_xhand_collision.urdf"),
            srdf_path=str(model_dir / "xarm7_xhand.srdf"),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=workspace,
        ),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
        table=runtime.environment.table,
    )
    arm_actions = np.asarray(trajectory.action_arm_joint, dtype=np.float64)
    for index in range(max(1, trajectory.num_frames - 1)):
        start = min(index, trajectory.num_frames - 1)
        end = min(index + 1, trajectory.num_frames - 1)
        arm_start, arm_end = arm_actions[start], arm_actions[end]
        hand_start, hand_end = modeled_hand[start], modeled_hand[end]
        if not planner.is_workspace_segment_safe(arm_start, arm_end):
            raise ValueError(f"live replay workspace rejection at transition {start}->{end}")
        if not planner.collision_model.check_transition_collision_free(arm_start, arm_end, hand_start, hand_end):
            raise ValueError(f"live replay collision rejection at transition {start}->{end}")
        max_delta = max(
            float(np.max(np.abs(arm_end - arm_start))),
            float(np.max(np.abs(hand_end - hand_start))),
        )
        steps = max(1, int(np.ceil(max_delta / _PREFLIGHT_MAX_JOINT_STEP_RAD)))
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            arm_qpos = arm_start + alpha * (arm_end - arm_start)
            hand_qpos = hand_start + alpha * (hand_end - hand_start)
            planner.collision_model.set_hand_qpos(hand_qpos)
            table_distance_m = planner.collision_model.minimum_table_distance(arm_qpos)
            if (
                not np.isfinite(table_distance_m)
                or table_distance_m < planner.collision_model.table_soft_clearance_m
            ):
                raise ValueError(f"live replay table rejection at transition {start}->{end}")


# ========================================================================
# Section 4: Replay runner
# ========================================================================


class ReplayStatus(str, Enum):
    """Terminal state of one replay attempt."""

    COMPLETED = "completed"
    USER_QUIT = "user_quit"
    REJECTED = "rejected"
    ESTOP = "estop"
    FAULT = "fault"


@dataclass(frozen=True)
class ReplayOutcome:
    """Replay result, including any samples captured before it stopped."""

    status: ReplayStatus
    replay_data: dict[str, np.ndarray] | None = None
    reason: str = ""

    @property
    def successful(self) -> bool:
        return self.status in (ReplayStatus.COMPLETED, ReplayStatus.USER_QUIT)


def arm_error_requires_stop(error_code: int) -> bool:
    """Any non-zero arm controller error is a stop condition (no worker recovery)."""
    return error_code != 0


def arm_feedback_issue(
    state: dict[str, Any] | None,
    max_age_s: float,
    *,
    now_ns: int | None = None,
) -> str | None:
    """Return why xArm feedback is unusable for replay, excluding controller error codes."""
    if state is None:
        return "arm feedback unavailable"
    try:
        return validate_arm_feedback(
            connected=bool(state.get("connected", False)),
            state_valid=bool(state.get("state_valid", False)),
            source_monotonic_ns=int(state.get("source_monotonic_ns", 0)),
            now_monotonic_ns=time.monotonic_ns() if now_ns is None else now_ns,
            max_age_s=max_age_s,
            qpos=np.asarray(state.get("qpos")),
            qvel=np.asarray(state.get("qvel")),
            eef_pos=np.asarray(state.get("eef_pos")),
            eef_rot6d=np.asarray(state.get("eef_rot6d")),
        )
    except (TypeError, ValueError) as exc:
        return f"invalid arm feedback: {exc}"


def hand_feedback_is_healthy(
    state: dict[str, Any] | None,
    max_age_s: float,
    *,
    now_ns: int | None = None,
) -> bool:
    """Validate every hand feedback health field used by live replay."""
    if state is None:
        return False
    try:
        issue = validate_hand_feedback(
            connected=bool(state.get("connected", False)),
            error_state=bool(state.get("error_state", True)),
            state_valid=bool(state.get("state_valid", False)),
            send_healthy=bool(state.get("send_healthy", False)),
            read_healthy=bool(state.get("read_healthy", False)),
            source_monotonic_ns=int(state.get("source_monotonic_ns", 0)),
            now_monotonic_ns=time.monotonic_ns() if now_ns is None else now_ns,
            max_age_s=max_age_s,
            qpos=np.asarray(state.get("qpos")),
        )
    except (TypeError, ValueError):
        return False
    return issue is None


class TrajectoryReplayer:
    """Replay one preflight-validated command stream through arm and hand workers."""

    START_POSE_TOLERANCE_DEG = 5.0

    def __init__(
        self,
        trajectory: TrajectoryData,
        shared: SharedStorage | None,
        *,
        speed: float = 1.0,
        dry_run: bool = False,
        no_hand: bool = False,
        max_frames: int | None = None,
        runtime: ResolvedRuntimeConfig | None = None,
        health_check: Callable[[], str | None] | None = None,
    ) -> None:
        self.traj = trajectory
        self.shared = shared
        self.speed = speed
        self.dry_run = dry_run
        self.no_hand = no_hand
        self.runtime = runtime
        self.replay_hz = trajectory.fps * speed
        if not np.isfinite(self.replay_hz) or self.replay_hz <= 0:
            raise ValueError("replay rate must be finite and positive")

        self._health_check = health_check
        self._planner: XArm7MotionPlanner | None = None
        self._action_safety_gate: SafetyGate | None = None
        self._recorder: ReplayRecorder | None = None
        self._running = False
        self._estopped = False
        self._live_motion_started = False
        self._status = ReplayStatus.COMPLETED
        self._reason = ""
        self._hand_available = trajectory.has_hand and not no_hand
        self._frame_count = trajectory.num_frames if max_frames is None else min(trajectory.num_frames, max_frames)

    def setup(self) -> None:
        """Create the action safety gate without connecting to hardware."""
        if self.dry_run:
            return
        if self.runtime is None or self.shared is None:
            raise RuntimeError("live replay requires runtime configuration and SharedStorage")
        runtime_arm = self.runtime.arm
        runtime_hand = self.runtime.hand
        runtime_policy = self.runtime.policy
        workspace = np.array(
            [
                [runtime_policy.workspace.x_min, runtime_policy.workspace.x_max],
                [runtime_policy.workspace.y_min, runtime_policy.workspace.y_max],
                [runtime_policy.workspace.z_min, runtime_policy.workspace.z_max],
            ],
            dtype=np.float64,
        )
        model_dir = ASSET_DIR / "robots" / "xhand"
        self._planner = XArm7MotionPlanner(
            XArm7PlannerConfig(
                urdf_path=str(model_dir / "xarm7_xhand_collision.urdf"),
                srdf_path=str(model_dir / "xarm7_xhand.srdf"),
                base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
                workspace_bounds=workspace,
            ),
            teleop_profile=TeleopProfile(
                max_pose_error_pos_m=float(runtime_policy.ik_max_pose_error_pos_m),
                max_pose_error_rot_rad=float(runtime_policy.ik_max_pose_error_rot_rad),
            ),
            hand_dof=True,
            static_boxes=tuple(self.runtime.environment.static_boxes),
            table=self.runtime.environment.table,
        )
        self._action_safety_gate = planner_action_safety_gate(
            planner=self._planner,
            arm_joint_lower_rad=tuple(runtime_arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime_arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime_hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime_hand.qpos_max_rad),
        )
        print("Replay safety gate ready")

    def _align_to_start(
        self,
        first_arm_cmd: np.ndarray,
        arm_qpos: np.ndarray,
        first_hand_cmd: np.ndarray | None = None,
        hand_qpos: np.ndarray | None = None,
    ) -> bool:
        """Require measured joints to already be close to the validated start."""
        max_dev = float(np.max(np.rad2deg(np.abs(arm_qpos - first_arm_cmd))))
        if first_hand_cmd is not None:
            if hand_qpos is None:
                print("Cannot verify the hand start pose from fresh connected feedback.")
                return False
            max_dev = max(max_dev, float(np.max(np.rad2deg(np.abs(hand_qpos - first_hand_cmd)))))
        if np.isfinite(max_dev) and max_dev <= self.START_POSE_TOLERANCE_DEG:
            return True
        print(f"\nRobot is {max_dev:.1f}° from the trajectory start (limit {self.START_POSE_TOLERANCE_DEG:.1f}°).")
        print("Move to the validated start pose in a separate supervised procedure, then retry replay.")
        return False

    def _outcome(self) -> ReplayOutcome:
        replay_data = self._recorder.to_dict() if self._recorder is not None else None
        return ReplayOutcome(self._status, replay_data, self._reason)

    def _fault(self, reason: str, *, estop: bool = False) -> None:
        """Latch a replay fault without clearing it during cleanup."""
        self._running = False
        self._estopped = self._estopped or estop
        self._status = ReplayStatus.ESTOP if estop else ReplayStatus.FAULT
        self._reason = reason
        if self.shared is None:
            return
        if estop:
            self.shared.estop_request.value = True
        self.shared.error_state.value = True
        self.shared.is_running.value = False
        require_transition(self.shared, SafetyState.FAULT)

    def _runtime_issue(self) -> tuple[ReplayStatus, str] | None:
        """Return (status, reason) if shared state signals a fault, estop, or health issue; None otherwise."""
        assert self.shared is not None
        if self.shared.estop_request.value:
            return ReplayStatus.ESTOP, "e-stop requested"
        if self.shared.error_state.value:
            return ReplayStatus.FAULT, "sticky error_state set"
        if int(self.shared.safety_state.value) == int(SafetyState.FAULT):
            return ReplayStatus.FAULT, "safety state is FAULT"
        if not self.shared.is_running.value:
            return ReplayStatus.FAULT, "runtime stop requested unexpectedly"
        if self._health_check is not None:
            issue = self._health_check()
            if issue:
                return ReplayStatus.FAULT, issue
        return None

    def _enter_terminal_quiescence(self) -> None:
        """Invalidate queued replay endpoints and publish nothing further.

        An endpoint already accepted by firmware is not retractable; verified
        shutdown later places the controller in State 4.
        """
        assert self.shared is not None
        run_generation = advance_run_generation(self.shared)
        logger.info(
            "replay entered terminal command quiescence (run=%d)",
            run_generation,
        )

    def _poll_control(self, keyboard: KeyboardHandler, timeout_s: float) -> bool:
        """Poll keyboard and runtime health for up to *timeout_s* seconds.

        Returns:
            True if the replay loop should continue, False if a stop/fault/quit
            signal was handled.
        """
        assert self.shared is not None
        signals = set(keyboard.poll(timeout=max(0.0, timeout_s)))
        if ControlSignal.EMERGENCY_STOP in signals:
            print("\nESC: emergency stop")
            self._fault("operator emergency stop", estop=True)
            return False
        issue = self._runtime_issue()
        if issue is not None:
            status, reason = issue
            self._fault(reason, estop=status is ReplayStatus.ESTOP)
            return False
        if ControlSignal.QUIT in signals:
            print("\nQ: stopping command publication")
            self._enter_terminal_quiescence()
            self._running = False
            self._status = ReplayStatus.USER_QUIT
            self._reason = "operator quit after entering command quiescence"
            return False
        return True

    def _wait_until_deadline(self, keyboard: KeyboardHandler, deadline_s: float) -> bool:
        """Wait in short slices so controls and worker health remain responsive."""
        while self._running:
            remaining_s = deadline_s - time.perf_counter()
            if remaining_s <= 0:
                return self._poll_control(keyboard, 0.0)
            if not self._poll_control(keyboard, min(_WAIT_POLL_INTERVAL_S, remaining_s)):
                return False
        return False

    def run(self) -> ReplayOutcome:
        """Execute the replay loop and report its explicit terminal outcome."""
        frame_count = self._frame_count
        print(f"\nReplay: {frame_count} frames @ {self.replay_hz:.1f} Hz (speed={self.speed}x)")
        print(f"  Source: {self.traj.episode_path}")
        if self.traj.task_label:
            print(f"  Task:   {self.traj.task_label}")
        print(f"  Hand:   {'ON' if (self._hand_available and not self.dry_run) else 'OFF'}")
        print(f"  Mode:   {'DRY RUN (no robot)' if self.dry_run else 'LIVE'}")
        print("\nControl: Q=quit  ESC=emergency_stop\n")

        if self.dry_run:
            self.validate_offline()
            return self._outcome()
        if self.shared is None or self.runtime is None:
            raise RuntimeError("live replay requires runtime configuration and SharedStorage")

        has_hand = self._hand_available
        self._recorder = ReplayRecorder(frame_count, has_hand=has_hand)
        arm_state = read_arm_state_dict(self.shared)
        initial_arm_issue = arm_feedback_issue(
            arm_state,
            float(self.runtime.policy.arm_state_stale_threshold_s),
        )
        if arm_state is None or initial_arm_issue is not None or arm_state["error_code"] != 0:
            self._fault(f"initial arm feedback is unavailable or unhealthy: {initial_arm_issue or 'controller error'}")
            return self._outcome()

        first_arm_cmd = self.traj.action_arm_joint[0].copy()
        first_hand_cmd: np.ndarray | None = None
        initial_hand_qpos: np.ndarray | None = None
        if has_hand and self.traj.action_hand_joint is not None:
            first_hand_cmd = self.traj.action_hand_joint[0].copy()
            hand_state = read_hand_state_dict(self.shared)
            if hand_feedback_is_healthy(
                hand_state,
                float(self.runtime.safety.heartbeat_timeouts["hand"]),
            ):
                assert hand_state is not None
                initial_hand_qpos = np.asarray(hand_state["qpos"], dtype=np.float64)
        if not self._align_to_start(first_arm_cmd, arm_state["qpos"], first_hand_cmd, initial_hand_qpos):
            self._status = ReplayStatus.REJECTED
            self._reason = "robot is not at the validated trajectory start"
            return self._outcome()

        shared = self.shared
        keyboard = KeyboardHandler(estop_callback=lambda: setattr(shared.estop_request, "value", True))
        keyboard.start()
        self._running = True
        error_count = 0
        max_consecutive_errors = int(self.runtime.policy.max_consecutive_errors)
        period_s = 1.0 / self.replay_hz
        next_deadline_s = time.perf_counter()
        start_time = next_deadline_s
        frame_idx = 0

        try:
            require_transition(self.shared, SafetyState.RUNNING)
            self._live_motion_started = True
            while frame_idx < frame_count and self._wait_until_deadline(keyboard, next_deadline_s):
                arm_cmd = self.traj.action_arm_joint[frame_idx].copy()
                hand_cmd = None
                if has_hand and self.traj.action_hand_joint is not None:
                    hand_cmd = self.traj.action_hand_joint[frame_idx].copy()
                # ``flag_action_queued`` marks whether a command was actually
                # queued on this grid slot during recording.  Synthetic hold
                # slots recorded no send event; reproduce that quiescence below
                # rather than republishing the inherited effective target.
                send_this = self.traj.send_mask is None or bool(self.traj.send_mask[frame_idx])
                if send_this and (
                    not np.all(np.isfinite(arm_cmd))
                    or (hand_cmd is not None and not np.all(np.isfinite(hand_cmd)))
                ):
                    self._fault(f"frame {frame_idx} contains a non-finite replay action")
                    break

                arm_state = read_arm_state_dict(self.shared)
                if arm_feedback_issue(arm_state, float(self.runtime.policy.arm_state_stale_threshold_s)) is not None:
                    error_count += 1
                    if error_count >= max_consecutive_errors:
                        self._fault("too many consecutive arm state read failures")
                        break
                    next_deadline_s = time.perf_counter() + min(period_s, _WAIT_POLL_INTERVAL_S)
                    continue
                assert arm_state is not None

                error_code = int(arm_state["error_code"])
                if arm_error_requires_stop(error_code):
                    self._fault(f"fatal arm controller error C{error_code}")
                    break

                error_count = 0
                hand_qpos: np.ndarray | None = None
                if has_hand:
                    hand_state = read_hand_state_dict(self.shared)
                    if not hand_feedback_is_healthy(
                        hand_state,
                        float(self.runtime.safety.heartbeat_timeouts["hand"]),
                    ):
                        self._fault(f"frame {frame_idx}: hand feedback is unavailable or unhealthy")
                        break
                    assert hand_state is not None
                    hand_qpos = hand_state["qpos"]

                if not send_this:
                    # Recorded command quiescence: observe and record the slot
                    # but send nothing (Mode 6 finishes the last accepted
                    # endpoint).  ``arm_sent_cmd`` stays NaN in the recorder.
                    self._recorder.record(
                        frame_idx,
                        arm_state["qpos"],
                        arm_state["eef_pos"],
                        arm_state["eef_rot6d"],
                        arm_cmd,
                        hand_cmd,
                        time.perf_counter(),
                        arm_tracking_error=arm_state["tracking_err"],
                        hand_qpos=hand_qpos,
                    )
                    frame_idx += 1
                    next_deadline_s += period_s
                    now_s = time.perf_counter()
                    if next_deadline_s < now_s:
                        next_deadline_s = now_s + period_s
                    continue

                is_final_frame = frame_idx == frame_count - 1
                published = publish_joint_targets(
                    self.shared,
                    arm_cmd,
                    hand_cmd,
                    prepare_timeout_s=float(self.runtime.policy.action_prepare_timeout_s),
                    safety_gate=self._action_safety_gate,
                    wait_applied=is_final_frame,
                    apply_timeout_s=float(self.runtime.policy.action_apply_timeout_s),
                    hand_mechanical_lower_rad=np.asarray(
                        self.runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
                    ),
                    hand_mechanical_upper_rad=np.asarray(
                        self.runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
                    ),
                    hand_feedback_max_age_s=float(
                        self.runtime.safety.heartbeat_timeouts["hand"]
                    ),
                )
                if not published.succeeded or published.candidate is None:
                    boundary = "publish/APPLIED" if is_final_frame else "publish"
                    self._fault(f"frame {frame_idx}: joint {boundary} boundary rejected: {published.reason}")
                    break
                candidate = published.candidate
                assert candidate.arm_qpos is not None
                sent_arm_cmd = np.asarray(candidate.arm_qpos, dtype=np.float64)
                if candidate.hand_qpos is not None:
                    hand_cmd = np.asarray(candidate.hand_qpos, dtype=np.float64)

                self._recorder.record(
                    frame_idx,
                    arm_state["qpos"],
                    arm_state["eef_pos"],
                    arm_state["eef_rot6d"],
                    sent_arm_cmd,
                    hand_cmd,
                    time.perf_counter(),
                    arm_sent_cmd=sent_arm_cmd,
                    arm_tracking_error=arm_state["tracking_err"],
                    hand_qpos=hand_qpos,
                )
                frame_idx += 1
                if frame_idx % _STATUS_INTERVAL_FRAMES == 0 or frame_idx == 1:
                    elapsed_s = time.perf_counter() - start_time
                    print(
                        f"[T+{elapsed_s:.1f}s f={frame_idx}/{frame_count}] "
                        f"eef={np.round(arm_state['eef_pos'], 3)}m  err={error_count}",
                        flush=True,
                    )
                next_deadline_s += period_s
                now_s = time.perf_counter()
                if next_deadline_s < now_s:
                    next_deadline_s = now_s + period_s
        except KeyboardInterrupt:
            print("\nInterrupted by user; stopping command publication")
            self._enter_terminal_quiescence()
            self._running = False
            self._status = ReplayStatus.USER_QUIT
            self._reason = "operator interrupt after entering command quiescence"
        except Exception as exc:
            logger.error("Unexpected replay failure", exc_info=True)
            self._fault(f"unexpected replay failure: {exc}")
        finally:
            keyboard.stop()
            if not self._estopped and int(self.shared.safety_state.value) == int(SafetyState.RUNNING):
                require_transition(self.shared, SafetyState.ARMED)

        if self._recorder.count < frame_count:
            print(f"\nReplay stopped at frame {self._recorder.count}/{frame_count}")
        return self._outcome()

    def validate_offline(self) -> None:
        """Validate trajectory arrays without simulating wall-clock playback."""
        frame_count = self._frame_count
        arm = self.traj.action_arm_joint[:frame_count]
        if arm.shape != (frame_count, *ARM_JOINT_SHAPE) or not np.all(np.isfinite(arm)):
            raise ValueError("arm replay actions must be finite with shape (T, 7)")
        if not self.no_hand and self.traj.action_hand_joint is not None:
            hand = self.traj.action_hand_joint[:frame_count]
            if hand.shape != (frame_count, *HAND_JOINT_SHAPE) or not np.all(np.isfinite(hand)):
                raise ValueError("hand replay actions must be finite with shape (T, 12)")
        print(f"Dry-run complete: validated {frame_count} frames at nominal {self.replay_hz:.1f} Hz")

    @property
    def can_offer_home(self) -> bool:
        return self._live_motion_started and not self._estopped

    @property
    def partial_data(self) -> dict[str, np.ndarray] | None:
        return self._recorder.to_dict() if self._recorder is not None else None

    @property
    def planner(self) -> XArm7MotionPlanner | None:
        return self._planner

    def shutdown(self) -> None:
        """Signal processes to stop; the session owns SharedStorage cleanup."""
        if not self.dry_run and self.shared is not None:
            self.shared.is_running.value = False


# ========================================================================
# Section 5: Live replay session
# ========================================================================


@dataclass(frozen=True)
class LiveReplayConfig:
    """Bundles CLI-level live-replay overrides (speed, joint limits, hand mode, dry-run flag)."""
    speed: float
    no_hand: bool
    max_frames: int | None
    output_dir: str
    evaluate_consistency: bool
    config_sha256: str


def _latched_fault_status(shared: SharedStorage) -> ReplayStatus | None:
    """Classify sticky runtime state before any transition can mask it."""
    if shared.estop_request.value:
        return ReplayStatus.ESTOP
    if shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
        return ReplayStatus.FAULT
    return None


def _post_shutdown_outcome(outcome: ReplayOutcome, report: ShutdownReport) -> ReplayOutcome:
    """Apply faults observed only after workers have reached a terminal state."""
    if report.estop_requested:
        shutdown_reason = "e-stop latched during replay shutdown"
        reason = f"{outcome.reason}; {shutdown_reason}" if outcome.reason else shutdown_reason
        return ReplayOutcome(
            ReplayStatus.ESTOP,
            outcome.replay_data,
            reason,
        )
    if report.faulted:
        failed = ", ".join(f"{item.name}={item.escalation}:{item.exitcode}" for item in report.abnormal_exits)
        shutdown_reason = (
            f"worker failed during replay shutdown: {failed}" if failed else "fault latched during replay shutdown"
        )
        reason = f"{outcome.reason}; {shutdown_reason}" if outcome.reason else shutdown_reason
        return ReplayOutcome(ReplayStatus.FAULT, outcome.replay_data, reason)
    return outcome


def _live_worker_health_issue(
    shared: SharedStorage,
    processes: list[Any],
    heartbeat_timeouts_s: dict[str, float],
    *,
    now_s: float | None = None,
) -> str | None:
    """Return the first arm/hand worker-health failure, if any."""
    for process in processes:
        if not process.is_alive():
            return f"worker {process.name!r} exited with code {process.exitcode}"

    heartbeat_by_name = {"arm", "hand"}
    for process in processes:
        if process.name not in heartbeat_by_name:
            continue
        last_s = shared.get_heartbeat(process.name)
        now = time.monotonic() if now_s is None else now_s
        timeout_s = float(heartbeat_timeouts_s[process.name])
        if not np.isfinite(last_s) or last_s <= 0 or last_s > now or now - last_s > timeout_s:
            return f"worker {process.name!r} heartbeat timed out"
    return None


def _report_consistency(metrics: ReplayMetrics) -> None:
    """Print human-readable replay-vs-original consistency summary."""
    print("\n" + "=" * 60)
    print("Consistency Evaluation")
    print("=" * 60)
    print(f"  Frames: {metrics.replayed_frames} replayed / {metrics.original_frames} original")
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
            f"max={metrics.eef_pos_error_max_mm:.1f}mm  rmse={metrics.eef_pos_error_rmse_mm:.1f}mm"
        )
    if metrics.eef_rot_error_per_frame_deg is not None:
        print(
            f"  EEF rot error:  mean={metrics.eef_rot_error_mean_deg:.2f}°  "
            f"max={metrics.eef_rot_error_max_deg:.2f}°"
        )
    if metrics.hand_joint_mae_overall_deg is not None:
        print(f"  Hand joint MAE: {metrics.hand_joint_mae_overall_deg:.3f} deg")
    print(f"  Tracking lag:  {metrics.tracking_lag_frames} frames ({metrics.tracking_lag_seconds:.3f}s)")
    if metrics.arm_tracking_error_mean_deg > 0:
        print(
            "  Replay tracking error (cmd vs actual): "
            f"mean={metrics.arm_tracking_error_mean_deg:.2f}°  "
            f"p95={metrics.arm_tracking_error_p95_deg:.2f}°  "
            f"max={metrics.arm_tracking_error_max_deg:.2f}°"
        )
    print("=" * 60)


def _evaluate_replay(
    trajectory: TrajectoryData,
    replay_data: dict[str, np.ndarray] | None,
    config: LiveReplayConfig,
) -> None:
    """Compute consistency metrics from captured replay data, report, and persist results."""
    if replay_data is None:
        print("\nNo replay data collected (replay interrupted before any frames captured).")
        return
    if replay_data["arm_qpos"].shape[0] == 0:
        print("\nSkipping metrics: no valid reference or replay data available")
        return
    if not config.evaluate_consistency:
        print("\nSkipping consistency metrics; saving captured replay data.")
        save_replay_data(replay_data, config.output_dir)
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
            speed_factor=config.speed,
        )
    except Exception:
        logger.error("replay consistency evaluation failed; saving raw replay data", exc_info=True)
        save_replay_data(replay_data, config.output_dir)
        raise
    tracking_error = replay_data.get("arm_tracking_error")
    if tracking_error is not None:
        finite = tracking_error[np.isfinite(tracking_error)]
        if finite.size:
            metrics.arm_tracking_error_mean_deg = float(np.rad2deg(np.mean(finite)))
            metrics.arm_tracking_error_p95_deg = float(np.rad2deg(np.percentile(finite, _TRACKING_ERROR_PERCENTILE)))
            metrics.arm_tracking_error_max_deg = float(np.rad2deg(np.max(finite)))

    _report_consistency(metrics)
    save_results(metrics, replay_data, config.output_dir)


def _offer_return_home(
    shared: SharedStorage,
    replayer: TrajectoryReplayer,
    runtime: ResolvedRuntimeConfig,
    arm_config: ArmLoopConfig,
    *,
    hand_available: bool,
    health_check: Callable[[], str | None],
) -> tuple[ReplayStatus, str] | None:
    """Post-replay prompt: press H to return arm/hand to home, Q to exit."""
    print("\nPress H to return_home, or Q to exit...")
    keyboard = KeyboardHandler(estop_callback=lambda: setattr(shared.estop_request, "value", True))
    keyboard.start()
    try:
        deadline = time.perf_counter() + float(runtime.policy.post_teleop_timeout_s)
        while time.perf_counter() < deadline:
            signals = set(keyboard.poll(timeout=0.1))
            if ControlSignal.EMERGENCY_STOP in signals or shared.estop_request.value:
                shared.estop_request.value = True
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                return ReplayStatus.ESTOP, "operator emergency stop during return-home prompt"
            if shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
                require_transition(shared, SafetyState.FAULT)
                return ReplayStatus.FAULT, "runtime fault during return-home prompt"
            health_issue = health_check()
            if health_issue:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                return ReplayStatus.FAULT, health_issue
            if ControlSignal.QUIT in signals:
                return None
            if ControlSignal.HOME not in signals:
                continue

            if hand_available:
                hand_home = np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64))
                hand_accepted = publish_hand_home_and_wait_applied(
                    shared,
                    hand_home,
                    command_lower_rad=np.asarray(runtime.hand.qpos_min_rad, dtype=np.float64),
                    command_upper_rad=np.asarray(runtime.hand.qpos_max_rad, dtype=np.float64),
                    mechanical_lower_rad=np.asarray(runtime.hand.mechanical_qpos_min_rad, dtype=np.float64),
                    mechanical_upper_rad=np.asarray(runtime.hand.mechanical_qpos_max_rad, dtype=np.float64),
                    hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
                    timeout_s=float(runtime.hand.home_command_ack_timeout_s),
                    heartbeat=False,
                    check_is_running=False,
                    verbose=True,
                    abort_requested=lambda: keyboard.estop_latched or not keyboard.healthy,
                )
                if not hand_accepted:
                    logger.warning("arm home cancelled because hand-home command was not accepted")
                    continue
                assert replayer.planner is not None
                replayer.planner.set_hand_qpos(hand_home)

            arm_reached = send_arm_home(
                shared,
                np.asarray(arm_config.home_qpos, dtype=np.float64),
                planner=replayer.planner,
                table_z_surface_m=float(runtime.arm.table_z_surface_m),
                queue_timeout=float(runtime.arm.homing.request_queue_timeout_s),
                converge_timeout_s=float(runtime.arm.homing.convergence_timeout_s),
                state_max_age_s=float(runtime.arm.homing.state_max_age_s),
                homing_max_speed_rad_s=float(np.deg2rad(runtime.arm.homing.max_speed_deg_s)),
                homing_target_timeout_s=float(runtime.arm.homing.target_timeout_s),
                preplan_velocity_rad_s=float(runtime.arm.homing.velocity_convergence_rad_s),
                result_tolerance_rad=float(runtime.arm.homing.convergence_rad),
                arm_heartbeat_max_age_s=float(runtime.safety.heartbeat_timeouts["arm"]),
                estop_requested=lambda: keyboard.estop_latched or not keyboard.healthy,
                heartbeat=False,
                verbose=True,
            )
            if not arm_reached:
                if shared.estop_request.value:
                    shared.error_state.value = True
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.ESTOP, "e-stop requested during return home"
                if shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.FAULT, "runtime fault during return home"
                health_issue = health_check()
                if health_issue:
                    shared.error_state.value = True
                    require_transition(shared, SafetyState.FAULT)
                    return ReplayStatus.FAULT, health_issue
                return ReplayStatus.REJECTED, "return-home request was not completed"
            print("Press Q to exit...")
    finally:
        keyboard.stop()
    if shared.estop_request.value:
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.ESTOP, "e-stop requested when return-home prompt expired"
    if shared.error_state.value or int(shared.safety_state.value) == int(SafetyState.FAULT):
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.FAULT, "runtime fault when return-home prompt expired"
    health_issue = health_check()
    if health_issue:
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        return ReplayStatus.FAULT, health_issue
    return None


def run_live_replay(
    trajectory: TrajectoryData,
    runtime: ResolvedRuntimeConfig,
    config: LiveReplayConfig,
) -> ReplayOutcome:
    """Start arm/hand workers, run one replay, then shut the session down."""
    try:
        verify_live_replay_preflight(
            trajectory,
            runtime,
            no_hand=config.no_hand,
            provenance_sha256=config.config_sha256,
        )
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        return ReplayOutcome(
            ReplayStatus.REJECTED,
            reason=f"live replay preflight rejected: {exc}",
        )

    print("\n" + "=" * 60)
    print("Replay — SharedStorage architecture (arm_loop + hand_loop)")
    print("=" * 60)

    context = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix=f"dexmani_replay_{os.getpid()}",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=context,
    )
    processes: list[Any] = []
    replayer: TrajectoryReplayer | None = None
    outcome = ReplayOutcome(ReplayStatus.REJECTED, reason="replay did not start")
    try:
        arm_config = ArmLoopConfig.from_runtime(runtime)
        hand_available = trajectory.has_hand and not config.no_hand
        specs = [WorkerSpec("arm", _arm_loop, (shared, arm_config), ready_name="arm")]
        if hand_available:
            from dexmani_real.robot.hand_process import HandProcessConfig

            hand_config = HandProcessConfig.from_runtime(runtime)
            specs.append(WorkerSpec("hand", _hand_loop, (shared, hand_config), ready_name="hand"))

        require_transition(shared, SafetyState.DISARMED)
        processes = build_processes(context, specs)
        start_processes(processes)

        timeouts = runtime.safety.readiness_timeouts_s
        ready_checks = [
            (spec.ready_name, float(timeouts[spec.ready_name]))
            for spec in specs
            if spec.ready_name
        ]
        workers_ready = wait_subsystem_ready(shared, ready_checks, processes)
        if not workers_ready:
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
            outcome = ReplayOutcome(ReplayStatus.FAULT, reason="worker readiness failed")
        else:
            heartbeat_timeouts = dict(runtime.safety.heartbeat_timeouts)

            def health_check() -> str | None:
                return _live_worker_health_issue(shared, processes, heartbeat_timeouts)

            replayer = TrajectoryReplayer(
                trajectory,
                shared,
                speed=config.speed,
                no_hand=config.no_hand,
                max_frames=config.max_frames,
                runtime=runtime,
                health_check=health_check,
            )
            replayer.setup()
            latched_status = _latched_fault_status(shared)
            health_issue = None if latched_status is not None else health_check()
            if latched_status is not None:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(latched_status, reason="fault latched before replay could arm")
            elif health_issue:
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(ReplayStatus.FAULT, reason=health_issue)
            else:
                require_transition(shared, SafetyState.ARMED)
                try:
                    outcome = replayer.run()
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    if replayer.can_offer_home:
                        shared.error_state.value = True
                        require_transition(shared, SafetyState.FAULT)
                        outcome = ReplayOutcome(
                            ReplayStatus.FAULT,
                            replayer.partial_data,
                            "replay interrupted before command quiescence could be established",
                        )
                    else:
                        outcome = ReplayOutcome(ReplayStatus.USER_QUIT, replayer.partial_data, "KeyboardInterrupt")

                if (
                    outcome.status is ReplayStatus.COMPLETED
                    and not shared.error_state.value
                    and replayer.can_offer_home
                ):
                    home_outcome = _offer_return_home(
                        shared,
                        replayer,
                        runtime,
                        arm_config,
                        hand_available=hand_available,
                        health_check=health_check,
                    )
                    if home_outcome is not None:
                        status, reason = home_outcome
                        outcome = ReplayOutcome(status, outcome.replay_data, reason)
    except Exception:
        logger.error("live replay session failed", exc_info=True)
        shared.error_state.value = True
        require_transition(shared, SafetyState.FAULT)
        raise
    finally:
        if replayer is not None:
            try:
                replayer.shutdown()
            except Exception:
                logger.error("replay controller shutdown failed", exc_info=True)
                shared.error_state.value = True
                require_transition(shared, SafetyState.FAULT)
                outcome = ReplayOutcome(
                    ReplayStatus.FAULT,
                    outcome.replay_data,
                    outcome.reason or "replay controller shutdown failed",
                )

        if outcome.status is ReplayStatus.ESTOP:
            shared.estop_request.value = True
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)
        elif outcome.status is ReplayStatus.FAULT:
            shared.error_state.value = True
            require_transition(shared, SafetyState.FAULT)

        started = [process for process in processes if process.pid is not None]
        try:
            shutdown_report = shutdown_processes(
                shared,
                started,
                graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
                disarm_if_clean=outcome.status not in (ReplayStatus.ESTOP, ReplayStatus.FAULT),
            )
        except RuntimeError:
            logger.critical("child process remains alive; leaving SharedStorage linked", exc_info=True)
            raise
        outcome = _post_shutdown_outcome(outcome, shutdown_report)

    if replayer is not None:
        _evaluate_replay(trajectory, outcome.replay_data, config)
    return outcome


# ========================================================================
# Section 6: CLI & main
# ========================================================================


@dataclass(frozen=True)
class ReplayRuntimeSelection:
    """Resolved runtime config with per-session derived values and provenance hashes."""
    runtime: ResolvedRuntimeConfig
    acceleration_deg_s2: float
    joint_speed_deg_s: float
    acceleration_source: str
    config_sha256: str


def _resolve_replay_runtime(args: argparse.Namespace) -> ReplayRuntimeSelection:
    """Resolve runtime config with CLI overrides and the recording-equivalent config provenance hash."""
    base_runtime = resolve_runtime_config(yaml_path=args.config)
    if args.acc is not None:
        acceleration = float(args.acc)
        acceleration_source = " (--acc override)"
    else:
        acceleration = float(base_runtime.arm.max_joint_acceleration_deg_per_s2)
        acceleration_source = " (from runtime config)"

    joint_speed = float(
        args.joint_speed
        if args.joint_speed is not None
        else base_runtime.arm.max_joint_velocity_deg_per_s
    )
    runtime = resolve_runtime_config(
        yaml_path=args.config,
        cli_overrides={
            "arm.ip": args.arm_ip,
            "arm.max_joint_acceleration_deg_per_s2": acceleration,
            "arm.max_joint_velocity_deg_per_s": joint_speed,
            "policy.hand_enabled": False if args.no_hand else None,
        },
    )
    if not args.no_hand and not bool(runtime.policy.hand_enabled):
        raise ValueError("policy.hand_enabled=false requires explicit --no-hand confirmation")
    # Provenance is the resolved config the replay actually runs (all CLI
    # overrides included), so it matches the recorder's runtime.sha256 exactly
    # when --acc/--joint-speed/--no-hand/--arm-ip agree with recording.
    config_sha256 = runtime.sha256
    return ReplayRuntimeSelection(runtime, acceleration, joint_speed, acceleration_source, config_sha256)


def _run_offline(
    args: argparse.Namespace,
    trajectory: TrajectoryData,
    selection: ReplayRuntimeSelection,
) -> None:
    replayer = TrajectoryReplayer(
        trajectory,
        None,
        speed=args.speed,
        dry_run=True,
        no_hand=args.no_hand,
        max_frames=args.max_frames,
        runtime=selection.runtime,
    )
    replayer.validate_offline()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a DexMani trajectory offline or run a fail-closed live replay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline inspection is the default; --live is the hardware boundary.
  python examples/replay_episode.py episodes/episode_20260729_213332

  # Validate only a bounded prefix
  python examples/replay_episode.py episodes/episode_20260729_213332 --max-frames 200

  # Validate trajectory without hardware (dry-run)
  python examples/replay_episode.py episodes/episode_20260729_213332 --dry-run

  # Arm-only (skip hand even if episode has hand data)
  python examples/replay_episode.py episodes/episode_20260729_213332 --no-hand

  # Live replay re-runs dense geometry and provenance checks before workers start
  python examples/replay_episode.py episodes/episode_20260729_213332 --source sent --live

  # Live replay with a custom capture/metrics directory
  python examples/replay_episode.py episodes/episode_20260729_213332 --source sent --live \
    --output results/my_replay/

Live replay controls:
  Q     clean exit (save partial results)
  H     return arm to home (post-replay prompt)
  ESC   emergency stop
        """,
    )
    parser.add_argument(
        "episode",
        nargs="?",
        type=str,
        help="Published schema-v16 episode directory.",
    )
    parser.add_argument(
        "--h5",
        dest="episode_h5",
        metavar="PATH",
        help="Legacy alias for the episode path; use the positional path in new commands.",
    )
    parser.add_argument(
        "--speed",
        type=_positive_finite_float,
        default=1.0,
        help="Speed factor used by live replay (1.0=recorded speed).",
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        default=None,
        help="Maximum number of frames to replay (default: all).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Offline validation only (default; retained for explicit scripts).",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Run dense preflight checks, then cross the hardware boundary.",
    )
    parser.add_argument("--config", type=str, default=None, help="Validated experiment YAML overrides.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Live-only directory for captured replay data and optional consistency metrics.",
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Skip hand commands; live use requires the physical hand to be absent or secured at its configured home.",
    )
    parser.add_argument(
        "--arm-ip",
        type=str,
        default=None,
        help="XArm controller IP (experiment YAML/defaults when omitted).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="cmd",
        choices=["cmd", "sent"],
        help="Arm stream: policy target (cmd) or exact submitted target (sent).",
    )
    parser.add_argument(
        "--joint-speed",
        type=_positive_finite_float,
        default=None,
        metavar="DEG_S",
        help="Mode-6 joint speed in °/s (defaults to the runtime config value).",
    )
    parser.add_argument(
        "--acc",
        type=_positive_finite_float,
        default=None,
        metavar="DEG_S2",
        help="Joint max acceleration in °/s² (defaults to the runtime config value).",
    )
    args = parser.parse_args(argv)
    args.dry_run = not args.live

    if args.episode is not None and args.episode_h5 is not None:
        parser.error("provide the episode path either positionally or with --h5, not both")
    args.episode = args.episode if args.episode is not None else args.episode_h5
    if args.episode is None:
        parser.error("an episode path is required (positional path or --h5 PATH)")
    if not args.live and args.output is not None:
        parser.error("--output is only used with --live; offline validation does not produce replay metrics")
    if args.live and args.source != "sent":
        parser.error("--live requires --source sent; raw policy candidates are never replayed on hardware")
    if args.live and args.speed > 1.0:
        print("Warning: speed-up increases commanded joint velocities; live safety gates may reject the replay")

    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        traj = load_trajectory(
            args.episode,
            max_frames=args.max_frames,
            source=args.source,
            require_live_validity=args.live,
            require_exact_source=args.live,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error loading episode: {exc}")
        return 1

    if traj.num_frames == 0:
        print("Error: trajectory has 0 frames")
        return 1

    try:
        selection = _resolve_replay_runtime(args)
    except (FileNotFoundError, TypeError, ValueError, OSError) as exc:
        print(f"Error resolving replay config: {exc}")
        return 1

    print(f"Trajectory: {traj.episode_path}")
    print(f"  Frames: {traj.num_frames}  FPS: {traj.fps:.1f}  Duration: {traj.num_frames/traj.fps:.1f}s")
    print(f"  Task: {traj.task_label or '(none)'}")
    hand_metadata = "yes" if traj.has_hand_actions else "no"
    print(f"  Hand available: {hand_metadata}  action dataset: {'yes' if traj.has_hand_actions else 'no'}")
    print(f"  EE data: {'yes' if traj.arm_ee is not None else 'no'}")
    print(f"  Acc: {selection.acceleration_deg_s2:.0f}°/s²{selection.acceleration_source}")
    print(f"  Joint speed: {selection.joint_speed_deg_s:.0f}°/s")
    print(f"  Replay config: {selection.config_sha256[:12]}")

    if args.dry_run:
        try:
            _run_offline(args, traj, selection)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Error: dry-run validation failed: {exc}")
            return 1
        print("Done.")
        return 0

    eval_available = bool(np.all(np.isfinite(traj.arm_qpos)))
    if not eval_available:
        print("Warning: arm_qpos missing/invalid in episode, cannot evaluate consistency.")
    if args.output is not None:
        output_dir = args.output
    else:
        _, episode_name = resolve_episode_path(args.episode)
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"{episode_name}_replay")
    print(f"Output: {output_dir}")

    try:
        outcome = run_live_replay(
            traj,
            selection.runtime,
            LiveReplayConfig(
                speed=args.speed,
                no_hand=args.no_hand,
                max_frames=args.max_frames,
                output_dir=output_dir,
                evaluate_consistency=bool(eval_available),
                config_sha256=selection.config_sha256,
            ),
        )
    except Exception as exc:
        logger.error("live replay failed", exc_info=True)
        print(f"\nLive replay failed: {exc}")
        return 1

    if not outcome.successful:
        print(f"Live replay stopped: {outcome.status.value}: {outcome.reason}")
        return 1
    if outcome.status is ReplayStatus.COMPLETED:
        print("Replay completed.")
    else:
        print("Replay exited cleanly; partial results were retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
