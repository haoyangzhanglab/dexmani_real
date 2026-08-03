#!/usr/bin/env python3
"""Replay a recorded robot trajectory and evaluate consistency.

.. attention::
   **SharedStorage architecture** — uses arm_loop + hand_loop processes
   with SafetyState machine and heartbeat monitoring.  Commands go through
   ``arm_action_q`` / ``hand_cmd_ring``; state is read from ``arm_state_ring``
   / ``hand_state_ring``.  No direct SDK access from the main process.

   Data collection: use **vr_teleop_hand_record.py** (canonical entry point).

Reads a DexMani episode (schema v8+) — either a **directory** (``data.h5``,
``depth.h5``, ``rgb.mp4``) or a **legacy single ``.h5`` file** — replays
the recorded joint commands on the real robot, captures the actual robot state
during replay, and evaluates how closely the replayed motion matches the
original recording.

Architecture:
    Episode dir / legacy .h5 → load_trajectory() → TrajectoryReplayer (robot control loop)
                                                        │
                                                  ReplayRecorder (capture replay state)
                                                        │
                                                  compute_metrics() → metrics.json + replay_data.npz

Usage:
    # Basic replay (new directory-format episode)
    python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332

    # Legacy single-file episode
    python examples/real/replay_traj.py --h5 episodes/episode_20260716_120000.h5

    # Slow motion (half-speed) with frame limit
    python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --speed 0.5 --max-frames 200

    # Dry-run: validate trajectory without connecting to hardware
    python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --dry-run

    # Arm-only replay (ignore hand data even if present)
    python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --no-hand

    # Replay the delta-clamped "sent" stream (schema v9+) instead of raw cmd
    python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --source sent

    # Override acceleration from HDF5 meta (e.g. match the collection condition)
    python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --acc 900

    # Custom output directory (default: replay_results/<episode>_replay/)
    python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --output results/my_replay/

Control:
    Q     clean exit (stop replay, save partial results)
    H     return arm to home position (post-replay prompt)
    ESC   emergency stop
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Ensure repo root is on sys.path (belt-and-suspenders for runs without PYTHONPATH=.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real import ASSET_DIR
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.pose_utils import rot6d_to_quat_wxyz
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.robot.inner_loop import ArmInnerLoopConfig, arm_loop as _arm_loop
from dexmani_real.robot.hand_process import hand_loop as _hand_loop
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import (
    HOME_SENTINEL,
    SharedStorage,
    SharedStorageConfig,
    read_arm_state,
    read_hand_state,
    write_hand_cmd,
)
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

# ═══════════════════════════════════════════════ Constants

# Home waypoint interval for alignment (s) — must be >= arm_loop tick (1/30 ≈ 0.033s)
# to avoid overflowing arm_action_q (maxsize=2).
HOME_DT = 0.04

# Default arm acceleration for replay (°/s²). When the HDF5 /meta carries a
# joint_max_acc attribute that value is preferred; this is the fallback.
ARM_MAX_ACC_DEG_S2 = 500.0

DEFAULT_OUTPUT_DIR = "replay_results"

# ═══════════════════════════════════════════════ HDF5 Loading


@dataclass
class TrajectoryData:
    """Preloaded HDF5 trajectory ready for replay."""

    h5_path: str
    num_frames: int
    fps: float
    task_label: str

    # Command trajectory (what we replay)
    action_arm_joint: np.ndarray  # (T, 7) float64
    action_hand_joint: np.ndarray | None  # (T, 12) float64 or None

    # Original recorded state (ground truth for evaluation)
    arm_qpos: np.ndarray  # (T, 7) float64
    hand_qpos: np.ndarray | None  # (T, 12) float64 or None
    arm_ee: np.ndarray | None  # (T, 9) float64 — [pos(3), rot6d(6)]

    # Recording parameters (from /meta, for matching replay conditions)
    joint_max_acc: float | None = None  # °/s², read from HDF5 meta; None if absent (legacy)

    @property
    def has_hand(self) -> bool:
        return self.action_hand_joint is not None


def _resolve_episode_path(raw_path: str) -> tuple[str, str]:
    """Normalise an episode path and extract the episode name.

    Accepts three forms::

        episodes/episode_20260729_213332/          → (dir,      "episode_20260729_213332")
        episodes/episode_20260729_213332/data.h5   → (dir,      "episode_20260729_213332")
        episodes/episode_20260716_120000.h5         → (file.h5,  "episode_20260716_120000")

    When the path points at a ``data.h5`` inside a new-format episode
    directory the parent directory is returned so :class:`EpisodeReader`
    can open it in merged mode (with ``depth.h5`` + ``rgb.mp4`` sidecars
    available).  All other paths are passed through unchanged.

    Returns:
        (resolved_path, episode_name) — *resolved_path* is the path to
        pass to :class:`EpisodeReader`; *episode_name* is a human-readable
        label derived from the directory/file name (no extension).
    """
    p = Path(raw_path)
    # data.h5 inside a new-format episode dir → resolve to the parent dir
    if p.is_file() and p.name == "data.h5":
        parent = p.parent
        # Heuristic: parent looks like an episode dir (contains depth.h5 or rgb.mp4)
        if (parent / "depth.h5").exists() or (parent / "rgb.mp4").exists():
            return (str(parent), parent.name)
    # Episode directory (new format)
    if p.is_dir():
        return (str(p), p.name)
    # Legacy single .h5 file (or unknown)
    return (str(p), p.stem)


def load_trajectory(h5_path: str, max_frames: int | None = None, source: str = "cmd") -> TrajectoryData:
    """Load and validate an episode for replay.

    Supports both new directory-format episodes (``data.h5`` + ``depth.h5``
    + ``rgb.mp4``) and legacy single ``.h5`` files.  If *h5_path* points at
    a ``data.h5`` inside a new-format episode directory it is automatically
    resolved to the parent directory so the merged view is available.

    Args:
        h5_path: Path to the episode — directory, ``data.h5``, or legacy
            ``.h5`` file (all three accepted).
        max_frames: Truncate to first N frames (None = all).
        source: Which arm action stream to replay: "cmd" (default) reads
            ``/action_arm_joint`` (policy output); "sent" reads
            ``/action_arm_joint_sent`` (schema v9+, the delta-clamped value
            actually sent). Falls back to cmd with a warning if sent is absent.

    Returns:
        TrajectoryData with all arrays preloaded.

    Raises:
        FileNotFoundError: Episode does not exist.
        ValueError: Missing required datasets or schema version mismatch.
    """
    resolved_path, _episode_name = _resolve_episode_path(h5_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Episode not found: {h5_path}")

    with EpisodeReader(resolved_path) as reader:
        f = reader.h5f
        # ── Validate schema ──
        # f.get("/meta") returns None when the group is missing (safe h5py idiom).
        # Using a dict default (f.get("/meta", {})) causes an AttributeError crash
        # on meta.attrs because the returned dict has no .attrs member.
        meta = f.get("/meta")
        schema = meta.attrs.get("schema_version", None) if meta else None
        if schema is not None and schema < 3:
            logger.warning("HDF5 schema v%d < 3 — some datasets may be missing", schema)

        num_frames_orig = meta.attrs.get("num_frames", 0) if meta else 0
        # Nominal grid rate: schema v7 stores control_hz; fps is the achieved
        # rate recomputed at stop (frame_count/duration) — prefer the nominal.
        fps = meta.attrs.get("control_hz", meta.attrs.get("fps", 16.0)) if meta else 16.0
        # Clamp: replay_hz drives the real arm — a diluted fps from an old
        # paused episode (or fps=0) must not set the physical replay rate.
        if not (1.0 <= float(fps) <= 100.0):
            logger.warning("Implausible meta rate %.3f Hz — falling back to 16 Hz for replay", float(fps))
            fps = 16.0
        task_label = meta.attrs.get("task_label", "") if meta else ""

        # ── Recording parameters for matching replay conditions ──
        _joint_max_acc = None
        if meta is not None:
            _raw = meta.attrs.get("joint_max_acc")
            if _raw is not None:
                try:
                    _joint_max_acc = float(_raw)
                except (TypeError, ValueError):
                    pass

        # ── Source: which arm action stream to replay ──
        arm_action_key = "action_arm_joint_sent" if source == "sent" else "action_arm_joint"
        if arm_action_key not in f:
            if source == "sent":
                logger.warning("/action_arm_joint_sent not found (pre-v9 HDF5) — falling back to /action_arm_joint")
                arm_action_key = "action_arm_joint"

        # ── Required datasets ──
        for key in [arm_action_key, "arm_qpos"]:
            if key not in f:
                raise ValueError(f"HDF5 missing required dataset: /{key}")

        # ── Determine frame count from the source dataset ──
        T_raw = f[arm_action_key].shape[0]
        if num_frames_orig == 0:
            num_frames_orig = T_raw
        T = min(T_raw, num_frames_orig)
        if max_frames is not None:
            T = min(T, max_frames)

        # ── Load command trajectory ──
        action_arm_joint = np.asarray(f[arm_action_key][:T], dtype=np.float64)
        action_hand_joint = None
        if "action_hand_joint" in f:
            action_hand_joint = np.asarray(f["action_hand_joint"][:T], dtype=np.float64)

        # ── Load original state (ground truth) ──
        arm_qpos = np.asarray(f["arm_qpos"][:T], dtype=np.float64)
        hand_qpos = None
        if "hand_qpos" in f:
            hand_qpos = np.asarray(f["hand_qpos"][:T], dtype=np.float64)

        arm_ee = None
        if "arm_ee" in f:
            arm_ee = np.asarray(f["arm_ee"][:T], dtype=np.float64)

    traj = TrajectoryData(
        h5_path=resolved_path,
        num_frames=T,
        fps=float(fps),
        task_label=str(task_label),
        action_arm_joint=action_arm_joint,
        action_hand_joint=action_hand_joint,
        arm_qpos=arm_qpos,
        hand_qpos=hand_qpos,
        arm_ee=arm_ee,
        joint_max_acc=_joint_max_acc,
    )

    logger.info(
        "Loaded trajectory: %d frames, fps=%.1f, task=%s, hand=%s, ee=%s",
        traj.num_frames,
        traj.fps,
        traj.task_label or "(none)",
        "yes" if traj.has_hand else "no",
        "yes" if traj.arm_ee is not None else "no",
    )
    return traj


# ═══════════════════════════════════════════════ Replay Recorder


class ReplayRecorder:
    """Pre-allocated buffer capturing robot state during replay.

    Single-threaded (main loop), so no locking is needed.
    """

    def __init__(self, max_frames: int, has_hand: bool = False) -> None:
        self.max_frames = max_frames
        self.has_hand = has_hand
        self._count = 0

        # Pre-allocate
        self.arm_qpos = np.full((max_frames, 7), np.nan, dtype=np.float64)
        self.eef_pos = np.full((max_frames, 3), np.nan, dtype=np.float64)
        self.eef_quat_wxyz = np.full((max_frames, 4), np.nan, dtype=np.float64)
        self.eef_rot6d = np.full((max_frames, 6), np.nan, dtype=np.float64)
        self.arm_cmd = np.full((max_frames, 7), np.nan, dtype=np.float64)
        self.arm_sent_cmd = np.full((max_frames, 7), np.nan, dtype=np.float64)
        self.arm_tracking_error = np.full((max_frames,), np.nan, dtype=np.float64)
        self.timestamps = np.full((max_frames,), np.nan, dtype=np.float64)

        # Safety-reject diagnostics: preserves observability on skipped frames
        # so downstream analysis can distinguish "safety gate fired" (valid
        # state + cmd, sent_cmd=NaN, flag_safety_reject=True) from "data
        # missing" (all NaN).
        self.flag_safety_reject = np.zeros(max_frames, dtype=bool)
        self.safety_reject_reason: list[str | None] = [None] * max_frames

        self.hand_qpos: np.ndarray | None = None
        self.hand_cmd: np.ndarray | None = None
        if has_hand:
            self.hand_qpos = np.full((max_frames, 12), np.nan, dtype=np.float64)
            self.hand_cmd = np.full((max_frames, 12), np.nan, dtype=np.float64)

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
        if idx >= self.max_frames:
            return
        self.arm_qpos[idx] = arm_qpos
        self.eef_pos[idx] = eef_pos
        self.eef_rot6d[idx] = eef_rot6d
        # Convert rot6d to quat wxyz for backward compat with downstream tools.
        try:
            _r = _rot6d_to_matrix(eef_rot6d)
            _q_xyzw = Rotation.from_matrix(_r).as_quat()
            self.eef_quat_wxyz[idx] = np.array([_q_xyzw[3], _q_xyzw[0], _q_xyzw[1], _q_xyzw[2]])
        except Exception:
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
        # Encode as fixed-length bytes so the NPZ can be loaded without
        # allow_pickle=True (object arrays require pickle, which np.load
        # blocks by default for security).
        reasons = np.array(
            [r.encode() if r else b"" for r in self.safety_reject_reason[:n]],
            dtype="S256",
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


# ═══════════════════════════════════════════════ Consistency Metrics


@dataclass
class ReplayMetrics:
    """Evaluated consistency between replayed and original trajectory."""

    # Metadata
    h5_path: str = ""
    task_label: str = ""
    speed_factor: float = 1.0
    original_frames: int = 0
    replayed_frames: int = 0
    matching_frames: int = 0

    # Arm joint error (rad, converted to deg for reporting)
    arm_joint_mae_deg: np.ndarray = field(default_factory=lambda: np.zeros(7))
    arm_joint_rmse_deg: np.ndarray = field(default_factory=lambda: np.zeros(7))
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


def _geodesic_distance_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angular distance between two rotation matrices in degrees."""
    # arccos(0.5 * (trace(R1 @ R2.T) - 1))
    cos_angle = 0.5 * (np.trace(R1 @ R2.T) - 1.0)
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
    h5_path: str = "",
    task_label: str = "",
    speed_factor: float = 1.0,
) -> ReplayMetrics:
    """Compare replayed trajectory against original recording.

    Aligns to min(original, replay) frames.
    """
    T_orig = original_arm_qpos.shape[0]
    T_replay = replay_arm_qpos.shape[0]
    T = min(T_orig, T_replay)

    metrics = ReplayMetrics(
        h5_path=h5_path,
        task_label=task_label,
        speed_factor=speed_factor,
        original_frames=T_orig,
        replayed_frames=T_replay,
        matching_frames=T,
    )

    if T == 0:
        logger.warning("No frames to compare")
        return metrics

    # ── Arm joint error ──
    orig_q = original_arm_qpos[:T]
    rep_q = replay_arm_qpos[:T]
    valid = np.all(np.isfinite(orig_q), axis=1) & np.all(np.isfinite(rep_q), axis=1)
    if valid.sum() > 0:
        diff = np.abs(orig_q[valid] - rep_q[valid])
        metrics.arm_joint_mae_deg = np.rad2deg(np.mean(diff, axis=0))
        metrics.arm_joint_rmse_deg = np.rad2deg(np.sqrt(np.mean(diff**2, axis=0)))
        metrics.arm_joint_mae_overall_deg = float(np.mean(metrics.arm_joint_mae_deg))
        metrics.arm_joint_rmse_overall_deg = float(np.mean(metrics.arm_joint_rmse_deg))

    # ── EEF position error ──
    if original_arm_ee is not None and original_arm_ee.shape[0] >= T:
        orig_ee_pos = original_arm_ee[:T, :3]
        rep_ee_pos = replay_arm_ee_pos[:T]
        valid_ee = np.all(np.isfinite(orig_ee_pos), axis=1) & np.all(np.isfinite(rep_ee_pos), axis=1)
        if valid_ee.sum() > 0:
            pos_err = np.linalg.norm(orig_ee_pos[valid_ee] - rep_ee_pos[valid_ee], axis=1)
            metrics.eef_pos_error_per_frame_mm = pos_err * 1000.0
            metrics.eef_pos_error_mean_mm = float(np.mean(pos_err) * 1000.0)
            metrics.eef_pos_error_max_mm = float(np.max(pos_err) * 1000.0)
            metrics.eef_pos_error_rmse_mm = float(np.sqrt(np.mean(pos_err**2)) * 1000.0)

    # ── EEF orientation error ──
    if original_arm_ee is not None and original_arm_ee.shape[0] >= T and replay_arm_ee_rot6d.shape[0] >= T:
        orig_rot6d = original_arm_ee[:T, 3:9]
        rep_rot6d = replay_arm_ee_rot6d[:T]
        valid_rot = np.all(np.isfinite(orig_rot6d), axis=1) & np.all(np.isfinite(rep_rot6d), axis=1)
        if valid_rot.sum() > 0:
            rot_errs = []
            for i in np.where(valid_rot)[0]:
                try:
                    R1 = _rot6d_to_matrix(orig_rot6d[i])
                    R2 = _rot6d_to_matrix(rep_rot6d[i])
                    rot_errs.append(_geodesic_distance_deg(R1, R2))
                except Exception:
                    rot_errs.append(np.nan)
            rot_errs_arr = np.array(rot_errs)
            finite = np.isfinite(rot_errs_arr)
            if finite.sum() > 0:
                metrics.eef_rot_error_per_frame_deg = rot_errs_arr
                metrics.eef_rot_error_mean_deg = float(np.mean(rot_errs_arr[finite]))
                metrics.eef_rot_error_max_deg = float(np.max(rot_errs_arr[finite]))

    # ── Hand joint error (if available) ──
    if original_hand_qpos is not None and replay_hand_qpos is not None:
        T_h = min(original_hand_qpos.shape[0], replay_hand_qpos.shape[0])
        if T_h > 0:
            orig_h = original_hand_qpos[:T_h]
            rep_h = replay_hand_qpos[:T_h]
            valid_h = np.all(np.isfinite(orig_h), axis=1) & np.all(np.isfinite(rep_h), axis=1)
            if valid_h.sum() > 0:
                diff_h = np.abs(orig_h[valid_h] - rep_h[valid_h])
                metrics.hand_joint_mae_overall_deg = float(np.rad2deg(np.mean(diff_h)))
                metrics.hand_joint_rmse_overall_deg = float(np.rad2deg(np.sqrt(np.mean(diff_h**2))))

    # ── Tracking lag: per-joint RMSE-minimizing shift → median consensus ──
    # Cross-correlating *velocity* diff-norms (the old approach) only captures
    # motion-pattern similarity, which is always aligned when replaying the same
    # commands.  To detect actual physical following lag, we shift the replay
    # position signal relative to the original and find the lag that minimises
    # per-joint RMSE, then take the median across joints (robust to outliers).
    if T >= 20:
        try:
            MAX_LAG = int(np.ceil(fps * 0.4))  # ±400 ms search window
            MAX_LAG = max(MAX_LAG, 6)          # at least ±6 frames
            joint_lags: list[int] = []
            for j in range(7):
                best_lag, best_rmse = 0, float("inf")
                for lag in range(-MAX_LAG, MAX_LAG + 1):
                    if lag < 0:
                        a = orig_q[-lag:, j]
                        b = rep_q[:lag, j]
                    elif lag > 0:
                        a = orig_q[:-lag, j]
                        b = rep_q[lag:, j]
                    else:
                        a = orig_q[:, j]
                        b = rep_q[:, j]
                    if len(a) < 10:
                        continue
                    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_lag = lag
                joint_lags.append(best_lag)
            # Median consensus — robust even if one joint gives a spurious value
            peak_lag = int(np.median(joint_lags))
            metrics.tracking_lag_frames = peak_lag
            metrics.tracking_lag_seconds = float(peak_lag) / fps
        except Exception:
            metrics.tracking_lag_frames = 0
            metrics.tracking_lag_seconds = 0.0

    return metrics


# ═══════════════════════════════════════════════ Save Results


def save_results(metrics: ReplayMetrics, replay_data: dict[str, np.ndarray], output_dir: str) -> None:
    """Save replay data and consistency metrics to output directory.

    Produces:
        <output_dir>/metrics.json   — human-readable scalar metrics
        <output_dir>/replay_data.npz — full time-series arrays
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Metrics JSON ──
    metrics_dict: dict = {
        "h5_path": metrics.h5_path,
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

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as fp:
        json.dump(metrics_dict, fp, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved: {metrics_path}")

    # ── Replay data NPZ ──
    npz_path = os.path.join(output_dir, "replay_data.npz")
    np.savez_compressed(npz_path, **replay_data)
    print(f"Replay data saved: {npz_path}  ({replay_data['arm_qpos'].shape[0]} frames)")


# ═══════════════════════════════════════════════ Helpers


def _make_planner() -> XArm7MotionPlanner:
    """Create XArm7MotionPlanner with standard teleop config.

    Under the SharedStorage architecture the planner is only used for
    start-alignment (collision-checked approach to trajectory start);
    the arm IP and servo loop are managed by ``arm_loop``.
    """
    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf")
    return XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            base_pose_world=Pose(
                p=np.array([0.0, 0.0, 0.0]),
                q=np.array([np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12)]),
            ),
        ),
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )


def _read_arm_state_dict(shared: SharedStorage) -> dict | None:
    """Read latest arm state from ring, return as dict of numpy arrays or None."""
    data = read_arm_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "qvel": np.asarray(data["qvel"][0], dtype=np.float64),
        "tau": np.asarray(data["tau"][0], dtype=np.float64),
        "eef_pos": np.asarray(data["eef_pos"][0], dtype=np.float64),
        "eef_rot6d": np.asarray(data["eef_rot6d"][0], dtype=np.float64),
        "error_code": int(data["error_code"][0]),
        "connected": bool(data["connected"][0]),
        "tracking_err": float(data["tracking_err"][0]),
    }


def _read_hand_state_dict(shared: SharedStorage) -> dict | None:
    """Read latest hand state from ring, return as dict of numpy arrays or None."""
    data = read_hand_state(shared)
    if data is None:
        return None
    return {
        "qpos": np.asarray(data["qpos"][0], dtype=np.float64),
        "connected": bool(data["connected"][0]),
        "error_state": bool(data["error_state"][0]),
    }


# ═══════════════════════════════════════════════ Main Replayer


class TrajectoryReplayer:
    """Orchestrates the replay of a recorded trajectory on the real robot.

    Under the SharedStorage architecture the replay script communicates with
    arm_loop / hand_loop processes exclusively through rings and queues.  There
    is no direct SDK access from the main process.
    """

    JOINT_ALIGN_MAX_DEG = 5.0  # max per-joint deviation before requiring planned approach

    def __init__(
        self,
        trajectory: TrajectoryData,
        shared: SharedStorage,
        *,
        speed: float = 1.0,
        dry_run: bool = False,
        no_hand: bool = False,
        max_frames: int | None = None,
    ) -> None:
        self.traj = trajectory
        self.shared = shared
        self.speed = speed
        self.dry_run = dry_run
        self.no_hand = no_hand

        self.replay_hz = trajectory.fps * speed
        self._rate_mgr = RateManager(self.replay_hz)

        self._planner: XArm7MotionPlanner | None = None
        self._recorder: ReplayRecorder | None = None
        self._running = False
        self._estopped = False
        self._hand_available = False

        effective_T = trajectory.num_frames
        if max_frames is not None:
            effective_T = min(effective_T, max_frames)
        self._T = effective_T

    # ── Setup ──

    def setup(self) -> None:
        """Create planner for start-alignment (no hardware connect — arm_loop owns the SDK)."""
        if self.dry_run:
            return
        self._planner = _make_planner()
        self._hand_available = self.traj.has_hand and not self.no_hand
        print("Planner ready for start-alignment (arm_loop/hand_loop already running)")

    # ── Start alignment ──

    def _align_to_start(self, first_cmd: np.ndarray, arm_qpos: np.ndarray) -> np.ndarray | None:
        """Check proximity to trajectory start and plan a safe approach if needed.

        Sends waypoints through ``arm_action_q``; arm_loop servos them at its
        configured rate (30 Hz, Mode 6).  This method sleeps between waypoints
        to avoid overflowing the bounded queue (maxsize=2).

        Returns:
            Updated arm_qpos after approach, or None if alignment failed.
        """
        joint_diff_deg = np.rad2deg(np.abs(arm_qpos - first_cmd))
        max_dev = float(np.max(joint_diff_deg))

        if max_dev <= self.JOINT_ALIGN_MAX_DEG:
            return arm_qpos

        assert self._planner is not None

        print(f"\nArm is {max_dev:.1f}° from trajectory start (threshold: {self.JOINT_ALIGN_MAX_DEG}°)")
        print("Planning collision-checked approach to trajectory start ...")

        # ── Target EEF pose ──
        if self.traj.arm_ee is not None and self.traj.arm_ee.shape[0] > 0:
            ee = self.traj.arm_ee[0]
            target_pos = ee[:3].copy()
            target_quat = rot6d_to_quat_wxyz(ee[3:9])
        else:
            pose = self._planner.compute_eef_pose_world(first_cmd)
            target_pos = pose.p.copy()
            target_quat = pose.q.copy()

        target_pose = Pose(p=target_pos, q=target_quat)

        # ── Sync hand qpos for collision checks ──
        if self._hand_available:
            try:
                hs = _read_hand_state_dict(self.shared)
                if hs is not None and hs["connected"] and np.all(np.isfinite(hs["qpos"])):
                    self._planner.set_hand_qpos(hs["qpos"])
            except Exception:
                pass

        # ── Plan ──
        try:
            path_result = self._planner.plan_path(
                target_eef_pose_world=target_pose,
                current_qpos=arm_qpos.copy(),
            )
        except Exception as e:
            logger.error("plan_path raised exception during start alignment: %s", e)
            print(f"\nApproach planning failed: {e}")
            print("Aborting replay for safety.")
            return None

        if not path_result.success or path_result.qpos_path is None:
            print(f"\nApproach planning failed: {path_result.reason}")
            print("Aborting replay for safety.")
            return None

        qpos_path = path_result.qpos_path
        print(f"Approach path planned: {qpos_path.shape[0]} waypoints (source={path_result.source})")
        print(f"Executing approach at ~{np.rad2deg(0.035) / HOME_DT:.0f}°/s ...")

        # ── Execute approach waypoints through arm_action_q ──
        for _i, waypoint in enumerate(qpos_path):
            if self.shared.error_state.value:
                print("  Approach aborted: error_state detected")
                return None
            self.shared.arm_action_q.put({"qpos": waypoint})
            time.sleep(HOME_DT)

        # Settle
        time.sleep(0.15)

        # ── Read final position from ring ──
        as_dict = _read_arm_state_dict(self.shared)
        if as_dict is None or not np.all(np.isfinite(as_dict["qpos"])):
            print("  WARNING: cannot read arm state after approach")
            return arm_qpos  # best-effort: return the input
        arm_qpos = as_dict["qpos"]

        final_dev = np.rad2deg(np.max(np.abs(arm_qpos - first_cmd)))
        print(f"Approach complete. Remaining deviation: {final_dev:.2f}°")

        # ── Joint-space fallback ──
        if final_dev > self.JOINT_ALIGN_MAX_DEG:
            n_steps = max(1, int(np.ceil(final_dev / 5.0)))
            print(f"Joint-space interpolation: {n_steps} step(s) ...")
            for alpha in np.linspace(0, 1, n_steps + 1)[1:]:
                if self.shared.error_state.value:
                    print("  Joint-space fallback aborted: error_state detected")
                    return None
                self.shared.arm_action_q.put({"qpos": arm_qpos + alpha * (first_cmd - arm_qpos)})
                time.sleep(HOME_DT)
            time.sleep(0.15)

            as_dict = _read_arm_state_dict(self.shared)
            if as_dict is not None and np.all(np.isfinite(as_dict["qpos"])):
                arm_qpos = as_dict["qpos"]
                final_dev = np.rad2deg(np.max(np.abs(arm_qpos - first_cmd)))
                print(f"Joint-space approach done. Remaining deviation: {final_dev:.2f}°")

        return arm_qpos

    # ── Run ──

    def run(self) -> dict[str, np.ndarray] | None:
        """Execute the replay loop. Returns replay data dict or None on dry-run/failure."""
        T = self._T
        print(f"\nReplay: {T} frames @ {self.replay_hz:.1f} Hz (speed={self.speed}x)")
        print(f"  Source: {self.traj.h5_path}")
        if self.traj.task_label:
            print(f"  Task:   {self.traj.task_label}")
        print(f"  Hand:   {'ON' if (self._hand_available and not self.dry_run) else 'OFF'}")
        print(f"  Mode:   {'DRY RUN (no robot)' if self.dry_run else 'LIVE'}")
        print("\nControl: Q=quit  ESC=emergency_stop\n")

        if self.dry_run:
            self._dry_run_loop(T)
            return None

        has_hand = self._hand_available
        self._recorder = ReplayRecorder(T, has_hand=has_hand)

        # ── Read initial state from rings ──
        as_dict = _read_arm_state_dict(self.shared)
        if as_dict is None or not np.all(np.isfinite(as_dict["qpos"])):
            print("ERROR: cannot read initial arm state from ring — aborting")
            return None
        arm_qpos = as_dict["qpos"]

        # ── Start alignment ──
        first_cmd = self.traj.action_arm_joint[0].copy()
        aligned_qpos = self._align_to_start(first_cmd, arm_qpos.copy())
        if aligned_qpos is None:
            print("Aborting replay: failed to align to trajectory start.")
            return self._recorder.to_dict() if self._recorder is not None else None

        # ── Keyboard ──
        kb = KeyboardHandler()
        kb.start()
        atexit.register(kb.stop)

        # ── Pre-warm: send first target before rate limiter ──
        self.shared.arm_action_q.put({"qpos": first_cmd})
        if has_hand and self.traj.action_hand_joint is not None:
            write_hand_cmd(self.shared, self.traj.action_hand_joint[0])
        self._rate_mgr = RateManager(self.replay_hz)

        self._running = True
        error_count = 0
        max_consecutive_errors = 10
        start_time = time.perf_counter()

        try:
            for frame_idx in range(T):
                self._rate_mgr.wait()

                # ── Keyboard ──
                for sig in kb.poll(timeout=0.0):
                    if sig == ControlSignal.EMERGENCY_STOP:
                        print("\nESC: emergency_stop")
                        self._emergency_stop()
                        self._running = False
                        break
                    elif sig == ControlSignal.QUIT:
                        print("\nQ: clean exit")
                        self._running = False
                        break
                if not self._running:
                    break

                # ── Parse command ──
                arm_cmd = self.traj.action_arm_joint[frame_idx].copy()
                hand_cmd = None
                if has_hand and self.traj.action_hand_joint is not None:
                    hand_cmd = self.traj.action_hand_joint[frame_idx].copy()

                # NaN guard
                if not np.all(np.isfinite(arm_cmd)):
                    logger.warning("Frame %d arm_cmd has NaN, using current state", frame_idx)
                    as_cur = _read_arm_state_dict(self.shared)
                    if as_cur is not None and np.all(np.isfinite(as_cur["qpos"])):
                        arm_cmd = as_cur["qpos"].copy()
                    else:
                        error_count += 1
                        continue

                # ── Read arm state from ring ──
                as_dict = _read_arm_state_dict(self.shared)
                if as_dict is None:
                    error_count += 1
                    logger.warning("arm_state_ring read returned None at frame %d", frame_idx)
                    if error_count > max_consecutive_errors:
                        print("Too many consecutive ring read failures, emergency_stop")
                        self._emergency_stop()
                        break
                    continue

                # ── Check for fatal error state ──
                if self.shared.error_state.value:
                    print("error_state set — aborting replay")
                    self._emergency_stop()
                    break

                # ── Arm error code (diagnostic — arm_loop auto-recovers C22/C24/C31) ──
                if as_dict["error_code"] != 0:
                    err = as_dict["error_code"]
                    logger.warning("Frame %d: arm error_code=%d (arm_loop handles internally)", frame_idx, err)
                    # Non-recoverable errors (>31 or unexpected) — abort
                    if err not in (22, 24, 31):
                        print(f"Arm non-recoverable error C{err} — aborting replay")
                        self._emergency_stop()
                        break

                if not np.all(np.isfinite(as_dict["qpos"])):
                    error_count += 1
                    continue

                error_count = 0

                # ── Pre-send safety gate (lightweight — ring-based) ──
                if not as_dict["connected"]:
                    fail_reason = "arm not connected"
                    print(f"  [SAFETY] {fail_reason} — skip frame", flush=True)
                    ts = time.perf_counter()
                    if self._recorder is not None:
                        self._recorder.record(
                            frame_idx, as_dict["qpos"], as_dict["eef_pos"], as_dict["eef_rot6d"],
                            arm_cmd, hand_cmd, ts,
                            arm_sent_cmd=None,
                            arm_tracking_error=as_dict["tracking_err"],
                            safety_reject_reason=fail_reason,
                        )
                    continue

                # ── Read hand state ──
                hand_qpos: np.ndarray | None = None
                if has_hand:
                    hs = _read_hand_state_dict(self.shared)
                    if hs is not None and hs["connected"]:
                        hand_qpos = hs["qpos"]

                # ── Send arm command ──
                sent_cmd = arm_cmd.copy()
                if self.shared.error_state.value:
                    logger.warning("Frame %d: error_state set — stopping replay", frame_idx)
                    break
                self.shared.arm_action_q.put({"qpos": arm_cmd})

                # ── Send hand command ──
                if has_hand and hand_cmd is not None:
                    write_hand_cmd(self.shared, hand_cmd)

                # ── Record ──
                ts = time.perf_counter()
                if self._recorder is not None:
                    self._recorder.record(
                        frame_idx,
                        as_dict["qpos"], as_dict["eef_pos"], as_dict["eef_rot6d"],
                        arm_cmd, hand_cmd, ts,
                        arm_sent_cmd=sent_cmd,
                        arm_tracking_error=as_dict["tracking_err"],
                        hand_qpos=hand_qpos,
                    )

                # ── Periodic log ──
                if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[T+{elapsed:.1f}s f={frame_idx+1}/{T}] "
                        f"eef={np.round(as_dict['eef_pos'], 3)}m  "
                        f"err={error_count}",
                        flush=True,
                    )

        finally:
            kb.stop()

            if self._recorder is not None:
                actual = self._recorder.count
                if actual < T:
                    print(f"\nReplay stopped at frame {actual}/{T}")
                return self._recorder.to_dict()

        return None

    def _emergency_stop(self) -> None:
        """Signal all processes to stop via flags (no direct SDK access)."""
        self._running = False
        self._estopped = True
        self.shared.estop_request.value = True
        self.shared.is_running.value = False
        transition(self.shared, SafetyState.FAULT)

    def _dry_run_loop(self, T: int) -> None:
        """Iterate through frames without connecting to robot."""
        print("Dry-run: iterating through trajectory without robot...\n")
        for frame_idx in range(T):
            self._rate_mgr.wait()
            arm_cmd = self.traj.action_arm_joint[frame_idx]
            hand_cmd = (
                self.traj.action_hand_joint[frame_idx]
                if self.traj.has_hand and self.traj.action_hand_joint is not None
                else None
            )
            if frame_idx < 5:
                print(f"  f={frame_idx}: arm={np.round(arm_cmd, 3)}  hand={'yes' if hand_cmd is not None else 'no'}")
            if frame_idx % 50 == 49:
                print(f"  ... f={frame_idx+1}/{T}")
        print(f"\nDry-run complete: {T} frames @ {self.replay_hz:.1f} Hz")

    # ── Shutdown ──

    def shutdown(self) -> None:
        """Signal processes to stop (SharedStorage cleanup is done by main)."""
        if self.dry_run:
            return
        self.shared.is_running.value = False
        print("Replay processes signaled to stop")


# ═══════════════════════════════════════════════ CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded robot trajectory from HDF5 and evaluate consistency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Directory-format episode (data.h5 + depth.h5 + rgb.mp4)
  python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332

  # Legacy single-file episode
  python examples/real/replay_traj.py --h5 episodes/episode_20260716_120000.h5

  # Slow motion with frame limit
  python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --speed 0.5 --max-frames 200

  # Validate trajectory without hardware (dry-run)
  python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --dry-run

  # Arm-only (skip hand even if episode has hand data)
  python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --no-hand

  # Replay "sent" stream (delta-clamped, schema v9+)
  python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --source sent

  # Override acceleration (e.g. match collection conditions)
  python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --acc 900

  # Custom output directory (default: replay_results/<episode>_replay/)
  python examples/real/replay_traj.py --h5 episodes/episode_20260729_213332 --output results/my_replay/

Control keys:
  Q     clean exit (save partial results)
  H     return arm to home (post-replay prompt)
  ESC   emergency stop
        """,
    )
    parser.add_argument(
        "--h5",
        required=True,
        type=str,
        help="Path to episode: directory (episode_XXX/), data.h5, or legacy .h5 file.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed factor (1.0=original recording speed, 0.5=half-speed).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to replay (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate trajectory without connecting to robot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for replay data + metrics. Default: replay_results/<episode>_replay/",
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Skip hand commands even if hand data is present in HDF5.",
    )
    parser.add_argument(
        "--arm-ip",
        type=str,
        default="192.168.1.215",
        help="XArm controller IP address (passed to ArmInnerLoopConfig).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="cmd",
        choices=["cmd", "sent"],
        help="Which arm action stream to replay: cmd (action_arm_joint, default) or sent (action_arm_joint_sent, schema v9+).",
    )
    parser.add_argument(
        "--acc",
        type=float,
        default=None,
        metavar="DEG_S2",
        help=f"Joint max acceleration in °/s². "
        "Default: auto-detect from HDF5 /meta/joint_max_acc, "
        f"fallback {ARM_MAX_ACC_DEG_S2:.0f} for legacy files. "
        "Use this flag to override (e.g. --acc 900).",
    )
    args = parser.parse_args()

    # ── Validate args ──
    if args.speed <= 0:
        print("Error: --speed must be positive")
        sys.exit(1)
    if args.speed > 1.5:
        print(f"Warning: --speed={args.speed}x may exceed hardware limits (inner loop runs at 30Hz)")

    # ── Load trajectory ──
    try:
        traj = load_trajectory(args.h5, max_frames=args.max_frames, source=args.source)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Error loading episode: {e}")
        sys.exit(1)

    if traj.num_frames == 0:
        print("Error: trajectory has 0 frames")
        sys.exit(1)

    # ── Resolve joint_max_acc: CLI override > HDF5 metadata > default ──
    _auto_acc = traj.joint_max_acc
    _acc_source = ""
    if args.acc is not None:
        _replay_acc = args.acc
        _acc_source = " (--acc override)"
    elif _auto_acc is not None:
        _replay_acc = _auto_acc
        _acc_source = " (from HDF5 meta)"
    else:
        _replay_acc = ARM_MAX_ACC_DEG_S2
        _acc_source = " (default, no HDF5 meta)"

    # ── Print trajectory info ──
    print(f"Trajectory: {traj.h5_path}")
    print(f"  Frames: {traj.num_frames}  FPS: {traj.fps:.1f}  Duration: {traj.num_frames/traj.fps:.1f}s")
    print(f"  Task: {traj.task_label or '(none)'}")
    print(f"  Hand data: {'yes' if traj.has_hand else 'no'}")
    print(f"  EE data: {'yes' if traj.arm_ee is not None else 'no'}")
    print(f"  Acc: {_replay_acc:.0f}°/s²{_acc_source}")

    eval_available = traj.arm_qpos is not None and np.all(np.isfinite(traj.arm_qpos))
    if not eval_available:
        print("Warning: arm_qpos missing/invalid in HDF5, cannot evaluate consistency.")

    # ── Setup output dir ──
    if args.output is not None:
        output_dir = args.output
    else:
        _, episode_name = _resolve_episode_path(args.h5)
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"{episode_name}_replay")
    print(f"Output: {output_dir}")

    # ── Dry-run: no hardware, just validate ──
    if args.dry_run:
        replayer = TrajectoryReplayer(traj, None, speed=args.speed, dry_run=True, no_hand=args.no_hand, max_frames=args.max_frames)  # type: ignore[arg-type]
        replayer._dry_run_loop(replayer._T)
        print("Done.")
        return

    # ═══════════════════════════════════════════════════════════════
    # SharedStorage architecture — spawn arm_loop (+ optional hand_loop)
    # ═══════════════════════════════════════════════════════════════
    import multiprocessing as mp

    print("\n" + "=" * 60)
    print("Replay — SharedStorage architecture (arm_loop + hand_loop)")
    print("=" * 60)

    # ── SharedStorage ──
    shm_cfg = SharedStorageConfig()
    shared = SharedStorage.create(prefix="dexmani_replay", config=shm_cfg)

    # ── ArmInnerLoop config: match replay acceleration ──
    _joint_max_acc_rad = float(np.deg2rad(_replay_acc))
    arm_cfg = ArmInnerLoopConfig(
        arm_ip=args.arm_ip,
        joint_max_acc=_joint_max_acc_rad,
    )
    hand_available = traj.has_hand and not args.no_hand

    # ── Spawn processes ──
    procs: list[mp.Process] = [
        mp.Process(target=_arm_loop, args=(shared, arm_cfg), name="arm", daemon=True),
    ]
    if hand_available:
        from dexmani_real.robot.hand_process import HandProcessConfig
        hand_cfg = HandProcessConfig()
        procs.append(mp.Process(target=_hand_loop, args=(shared, hand_cfg), name="hand", daemon=True))

    for p in procs:
        p.start()

    # ── Wait for ready ──
    transition(shared, SafetyState.DISARMED)

    ready_checks: list[tuple[str, Any, float]] = [("arm", shared.arm_ready, 30)]
    if hand_available:
        ready_checks.append(("hand", shared.hand_ready, 30))

    for name, ev, timeout in ready_checks:
        if not ev.wait(timeout=timeout):
            logger.error("%s startup failed: ready-event timeout after %ds", name, timeout)
            shared.is_running.value = False
            _shutdown_replay(procs, shared)
            return

    print("  arm_loop: ready")
    if hand_available:
        print("  hand_loop: ready")

    # All subsystems ready — transition to ARMED
    transition(shared, SafetyState.ARMED)
    print(f"Safety state: ARMED ({int(SafetyState.ARMED)})\n")

    # ── Replay ──
    replayer = TrajectoryReplayer(
        traj, shared,
        speed=args.speed,
        dry_run=False,
        no_hand=args.no_hand,
        max_frames=args.max_frames,
    )

    try:
        replayer.setup()
        replay_data = replayer.run()

        # ── Compute metrics ──
        if replay_data is not None and replay_data["arm_qpos"].shape[0] > 0 and eval_available:
            print("\nComputing consistency metrics...")
            metrics = compute_metrics(
                original_arm_qpos=traj.arm_qpos,
                replay_arm_qpos=replay_data["arm_qpos"],
                original_arm_ee=traj.arm_ee,
                replay_arm_ee_pos=replay_data["eef_pos"],
                replay_arm_ee_rot6d=replay_data["eef_rot6d"],
                fps=traj.fps,
                original_hand_qpos=traj.hand_qpos,
                replay_hand_qpos=replay_data.get("hand_qpos"),
                h5_path=traj.h5_path,
                task_label=traj.task_label,
                speed_factor=args.speed,
            )

            if "arm_tracking_error" in replay_data:
                track_err = replay_data["arm_tracking_error"]
                valid = track_err[np.isfinite(track_err)]
                if len(valid) > 0:
                    metrics.arm_tracking_error_mean_deg = float(np.rad2deg(np.mean(valid)))
                    metrics.arm_tracking_error_p95_deg = float(np.rad2deg(np.percentile(valid, 95)))
                    metrics.arm_tracking_error_max_deg = float(np.rad2deg(np.max(valid)))

            print("\n" + "=" * 60)
            print("Consistency Evaluation")
            print("=" * 60)
            print(f"  Frames: {metrics.replayed_frames} replayed / {metrics.original_frames} original")
            print(
                f"  Arm joint MAE:  {np.round(metrics.arm_joint_mae_deg, 2)} deg  (overall: {metrics.arm_joint_mae_overall_deg:.3f} deg)"
            )
            print(
                f"  Arm joint RMSE: {np.round(metrics.arm_joint_rmse_deg, 2)} deg  (overall: {metrics.arm_joint_rmse_overall_deg:.3f} deg)"
            )
            if metrics.eef_pos_error_mean_mm > 0:
                print(
                    f"  EEF pos error:  mean={metrics.eef_pos_error_mean_mm:.1f}mm  max={metrics.eef_pos_error_max_mm:.1f}mm  rmse={metrics.eef_pos_error_rmse_mm:.1f}mm"
                )
            if metrics.eef_rot_error_mean_deg > 0:
                print(
                    f"  EEF rot error:  mean={metrics.eef_rot_error_mean_deg:.2f}°  max={metrics.eef_rot_error_max_deg:.2f}°"
                )
            if metrics.hand_joint_mae_overall_deg is not None:
                print(f"  Hand joint MAE: {metrics.hand_joint_mae_overall_deg:.3f} deg")
            print(f"  Tracking lag:  {metrics.tracking_lag_frames} frames ({metrics.tracking_lag_seconds:.3f}s)")
            if metrics.arm_tracking_error_mean_deg > 0:
                print(
                    f"  Replay tracking error (cmd vs actual): "
                    f"mean={metrics.arm_tracking_error_mean_deg:.2f}°  "
                    f"p95={metrics.arm_tracking_error_p95_deg:.2f}°  "
                    f"max={metrics.arm_tracking_error_max_deg:.2f}°"
                )
            print("=" * 60)

            save_results(metrics, replay_data, output_dir)
        elif replay_data is None:
            print("\nNo replay data collected (replay interrupted before any frames captured).")
        else:
            print("\nSkipping metrics: no replay data available")

    except (ConnectionError, RuntimeError) as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # ── Post-loop: offer return-to-home via HOME_SENTINEL ──
        # Skip home prompt if emergency-stopped (arm_loop already dead)
        if not shared.error_state.value and not replayer._estopped:
            print("\nPress H to return_home, or Q to exit...")
            kb = KeyboardHandler()
            kb.start()
            try:
                _deadline = time.perf_counter() + 60.0
                while time.perf_counter() < _deadline:
                    sigs = {s for s in kb.poll(timeout=0.1)}
                    if ControlSignal.QUIT in sigs or ControlSignal.EMERGENCY_STOP in sigs:
                        break
                    if ControlSignal.HOME in sigs:
                        print("\nH: return_home (HOME_SENTINEL → arm_action_q)")
                        shared.arm_action_q.put(HOME_SENTINEL)
                        # Wait for arm to execute homing — detect heartbeat stall.
                        _home_wait = time.perf_counter() + 20.0
                        _last_hb = shared.arm_heartbeat_s.value
                        while time.perf_counter() < _home_wait:
                            time.sleep(0.5)
                            _cur_hb = shared.arm_heartbeat_s.value
                            if _cur_hb == _last_hb:
                                break  # heartbeat stalled → arm_loop likely dead
                            _last_hb = _cur_hb
                        print("Press Q to exit...")
            finally:
                kb.stop()

        transition(shared, SafetyState.DISARMED)
        replayer.shutdown()
        _shutdown_replay(procs, shared)

    print("Done.")


def _shutdown_replay(procs: list, shared: SharedStorage) -> None:
    """Graceful shutdown for replay processes."""
    shared.is_running.value = False
    for p in procs:
        p.join(timeout=5)
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
    shared.close()
    print("Shutdown complete")


if __name__ == "__main__":
    main()
