#!/usr/bin/env python3
"""Replay a recorded robot trajectory from HDF5 and evaluate consistency.

Reads an HDF5 episode (schema v5, 50Hz aligned grid) collected by VR teleop,
replays the recorded joint commands on the real robot, records the actual robot
state during replay, and evaluates how closely the replayed motion matches the
original recording.

Architecture:
    HDF5 file → load_trajectory() → TrajectoryReplayer (robot control loop)
                                         │
                                   ReplayRecorder (capture replay state)
                                         │
                                   compute_metrics() → metrics.json + replay_data.npz

Usage:
    python examples/real/replay_traj.py --h5 episodes/episode_20260716_120000.h5
    python examples/real/replay_traj.py --h5 episode.h5 --speed 0.5 --max-frames 200
    python examples/real/replay_traj.py --h5 episode.h5 --dry-run
    python examples/real/replay_traj.py --h5 episode.h5 --no-hand --output results/

Control:
    Q     clean exit (stop replay, save partial results)
    ESC   emergency stop
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real import ASSET_DIR
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.collision_config import CollisionConfig
from dexmani_real.planning.pose_utils import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.robot.arm_process import ArmServo, make_arm_servo
from dexmani_real.robot.inner_loop import ArmInnerLoopConfig
from dexmani_real.robot.interface import RobotAction, RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.preflight import preflight_check, print_preflight
from dexmani_real.robot.types import RobotState
from dexmani_real.robot.validate import validate_action
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.teleop.control.keyboard import ControlSignal, KeyboardHandler
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

# ═══════════════════════════════════════════════ Constants

HOME_DT = 0.04  # homing waypoint interval (s)

WORKSPACE_BOUNDS = np.array([[0.24, 0.72], [-0.50, 0.50], [0.05, 0.5]], dtype=np.float64)

COLLISION_CONFIG = CollisionConfig(
    table_z_world=0.0,
    hand_extension_below_eef=0.076,
    hand_safe_margin=0.03,
)

ARM_MAX_SPEED_DEG_S = 120.0  # 对齐采集入口 (vr_teleop_arm_only_record*.py) — 回放与采集同速
_INNER_CFG = ArmInnerLoopConfig(joint_max_speed=float(np.deg2rad(ARM_MAX_SPEED_DEG_S)))

DEFAULT_OUTPUT_DIR = "replay_results"

# ═══════════════════════════════════════════════ Keyboard (pynput, 全局捕获)


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

    @property
    def has_hand(self) -> bool:
        return self.action_hand_joint is not None


def load_trajectory(h5_path: str, max_frames: int | None = None, source: str = "cmd") -> TrajectoryData:
    """Load and validate an HDF5 episode for replay.

    Args:
        h5_path: Path to the HDF5 episode file.
        max_frames: Truncate to first N frames (None = all).
        source: Which arm action stream to replay: "cmd" (default) reads
            ``/action_arm_joint`` (policy output); "sent" reads
            ``/action_arm_joint_sent`` (schema v9, the delta-clamped value
            actually sent). Falls back to cmd with a warning if sent is absent.

    Returns:
        TrajectoryData with all arrays preloaded.

    Raises:
        FileNotFoundError: h5_path does not exist.
        ValueError: Missing required datasets or schema version mismatch.
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # ── Validate schema ──
        meta = f.get("/meta", {})
        schema = meta.attrs.get("schema_version", None) if meta else None
        if schema is not None and schema < 3:
            logger.warning("HDF5 schema v%d < 3 — some datasets may be missing", schema)

        num_frames_orig = meta.attrs.get("num_frames", 0) if meta else 0
        # Nominal grid rate: schema v7 stores control_hz; fps is the achieved
        # rate recomputed at stop (frame_count/duration) — prefer the nominal.
        fps = meta.attrs.get("control_hz", meta.attrs.get("fps", 50.0)) if meta else 50.0
        # Clamp: replay_hz drives the real arm — a diluted fps from an old
        # paused episode (or fps=0) must not set the physical replay rate.
        if not (1.0 <= float(fps) <= 100.0):
            logger.warning("Implausible meta rate %.3f Hz — falling back to 50 Hz for replay", float(fps))
            fps = 50.0
        task_label = meta.attrs.get("task_label", "") if meta else ""

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
        h5_path=h5_path,
        num_frames=T,
        fps=float(fps),
        task_label=str(task_label),
        action_arm_joint=action_arm_joint,
        action_hand_joint=action_hand_joint,
        arm_qpos=arm_qpos,
        hand_qpos=hand_qpos,
        arm_ee=arm_ee,
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
        self.timestamps = np.full((max_frames,), np.nan, dtype=np.float64)

        if has_hand:
            self.hand_qpos = np.full((max_frames, 12), np.nan, dtype=np.float64)
            self.hand_cmd = np.full((max_frames, 12), np.nan, dtype=np.float64)
        else:
            self.hand_qpos = None
            self.hand_cmd = None

    def record(
        self,
        idx: int,
        state: RobotState,
        arm_cmd: np.ndarray,
        hand_cmd: np.ndarray | None,
        ts: float,
    ) -> None:
        """Record one frame of replay state."""
        if idx >= self.max_frames:
            return
        self.arm_qpos[idx] = state.arm_qpos
        self.eef_pos[idx] = state.eef_pos
        self.eef_quat_wxyz[idx] = state.eef_quat_wxyz
        self.eef_rot6d[idx] = state.eef_rot6d
        self.arm_cmd[idx] = arm_cmd
        self.timestamps[idx] = ts
        if self.has_hand and self.hand_qpos is not None:
            self.hand_qpos[idx] = state.hand_qpos
        if self.has_hand and self.hand_cmd is not None and hand_cmd is not None:
            self.hand_cmd[idx] = hand_cmd
        self._count = idx + 1

    @property
    def count(self) -> int:
        return self._count

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return truncated arrays as a dict."""
        n = self._count
        result: dict[str, np.ndarray] = {
            "arm_qpos": self.arm_qpos[:n].copy(),
            "eef_pos": self.eef_pos[:n].copy(),
            "eef_quat_wxyz": self.eef_quat_wxyz[:n].copy(),
            "eef_rot6d": self.eef_rot6d[:n].copy(),
            "arm_cmd": self.arm_cmd[:n].copy(),
            "timestamp": self.timestamps[:n].copy(),
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


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert (6,) rot6d to (3,3) rotation matrix via Gram-Schmidt."""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


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

    # ── Tracking lag via cross-correlation on joint L2 distances ──
    if T >= 20:
        try:
            dist_orig = np.linalg.norm(np.diff(orig_q, axis=0), axis=1)
            dist_rep = np.linalg.norm(np.diff(rep_q, axis=0), axis=1)
            # Trim to equal length
            L = min(len(dist_orig), len(dist_rep))
            if L >= 10:
                xcorr = np.correlate(
                    dist_orig[:L] - np.mean(dist_orig[:L]), dist_rep[:L] - np.mean(dist_rep[:L]), mode="full"
                )
                peak_lag = int(np.argmax(xcorr)) - (L - 1)
                # Clamp to reasonable range (±5 frames = ±100ms)
                peak_lag = max(-5, min(5, peak_lag))
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
    if metrics.hand_joint_mae_overall_deg is not None:
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


def _make_planner(arm_ip: str) -> XArm7MotionPlanner:
    """Create XArm7MotionPlanner with standard teleop config."""
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
            collision=COLLISION_CONFIG,
        ),
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(
            use_position_ik=True,
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )


def _make_robot(
    planner: XArm7MotionPlanner, arm_ip: str, *, use_hand_process_isolation: bool = False
) -> RobotInterface:
    """Create and return RobotInterface (not connected)."""
    arm_cfg = XArm7Config(ip=arm_ip)
    return RobotInterface(
        RobotInterfaceConfig(
            arm=arm_cfg,
            collision=COLLISION_CONFIG,
            hand_urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf"),
            use_hand_process_isolation=use_hand_process_isolation,
        ),
        kinematics=planner.kin,
        planner=planner,
    )


def _do_return_home(
    robot: RobotInterface,
    planner: XArm7MotionPlanner,
    arm_inner: ArmServo,
    arm_ip: str = "192.168.1.111",
    *,
    inner_cfg: ArmInnerLoopConfig = _INNER_CFG,
    use_arm_isolation: bool = False,
) -> ArmServo:
    """Return arm to home position: stop inner loop → plan+execute → restart.

    Args:
        inner_cfg: Per-run inner-loop config (e.g. with extended target_timeout_s
                   for slow replays). Defaults to the module-level _INNER_CFG but
                   the caller MUST pass the replayer's per-run cfg to avoid
                   timeout loss (plan §6 P1 slow-replay risk).
    """
    print("return_home ...", flush=True)
    try:
        arm_inner.set_target(None)
        arm_inner.stop()
        print("  Arm inner loop stopped")

        ok = robot.return_to_home(home_dt=HOME_DT)
        print(f"  {'OK' if ok else 'FAIL'}")

        new_inner = make_arm_servo(cfg=inner_cfg, ip=arm_ip, use_arm_isolation=use_arm_isolation)
        new_inner.start()
        print("  Arm inner loop restarted")
        return new_inner
    except Exception:
        traceback.print_exc()
        print("  return_to_home failed, attempting emergency_stop")
        arm_inner.set_target(None)
        arm_inner.stop()
        robot.emergency_stop()
        raise


# ═══════════════════════════════════════════════ Main Replayer


class TrajectoryReplayer:
    """Orchestrates the replay of a recorded trajectory on the real robot."""

    def __init__(
        self,
        trajectory: TrajectoryData,
        *,
        speed: float = 1.0,
        dry_run: bool = False,
        no_hand: bool = False,
        arm_ip: str = "192.168.1.111",
        max_frames: int | None = None,
    ) -> None:
        self.traj = trajectory
        self.speed = speed
        self.dry_run = dry_run
        self.no_hand = no_hand
        self.arm_ip = arm_ip

        self.replay_hz = trajectory.fps * speed  # follow the episode's recorded grid rate
        self._rate_mgr = RateManager(self.replay_hz)

        self.planner: XArm7MotionPlanner | None = None
        self.robot: RobotInterface | None = None
        self._arm_inner: ArmServo | None = None
        self._recorder: ReplayRecorder | None = None
        self._running = False

        # Effective frame count
        effective_T = trajectory.num_frames
        if max_frames is not None:
            effective_T = min(effective_T, max_frames)
        self._T = effective_T

        # ArmInnerLoop: extend timeout for slow replay speeds; speed aligned with capture (120°/s)
        self._inner_cfg = ArmInnerLoopConfig(
            joint_max_speed=float(np.deg2rad(ARM_MAX_SPEED_DEG_S)),
            target_timeout_s=max(0.2, 1.0 / max(self.replay_hz, 1.0) + 0.1),
        )

    # ── Setup ──

    def setup(self) -> None:
        """Connect to robot, preflight, start ArmInnerLoop."""
        if self.dry_run:
            return

        self.planner = _make_planner(self.arm_ip)
        self.robot = _make_robot(self.planner, self.arm_ip)

        print("\nConnecting to hardware...")
        result = self.robot.connect()
        print(f"  arm:  {'OK' if result.get('arm') else 'FAIL'}")
        print(f"  hand: {'OK' if result.get('hand') else 'FAIL (degraded — arm only)'}")

        if not result.get("arm"):
            raise ConnectionError("Arm connection failed")

        self.robot.arm.clear_error()
        self._hand_available = bool(result.get("hand")) and not self.no_hand

        # Preflight
        report = preflight_check(self.robot)
        print_preflight(report)
        if not report.passed:
            raise RuntimeError("Pre-flight check failed")

        # ArmInnerLoop
        self._arm_inner = make_arm_servo(cfg=self._inner_cfg, ip=self.arm_ip)
        self._arm_inner.start()
        print("Arm inner loop starting...", flush=True)

        if not self._arm_inner.wait_ready(timeout=30.0):
            print("Arm inner loop start timed out, falling back to direct read")
        else:
            print("Arm inner loop ready (50Hz online trajectory planning)")

    # ── Start alignment ──

    JOINT_ALIGN_MAX_DEG = 5.0  # max per-joint deviation before requiring planned approach

    def _align_to_start(self, first_cmd: np.ndarray, current_qpos: np.ndarray) -> np.ndarray | None:
        """Check proximity to trajectory start and plan a safe approach if needed.

        Args:
            first_cmd: Trajectory first-frame arm joint command (7,).
            current_qpos: Current arm joint positions from inner loop / SDK read (7,).

        Returns:
            Updated arm_qpos after approach, or None if alignment failed
            (caller should abort replay).
        """
        joint_diff_deg = np.rad2deg(np.abs(current_qpos - first_cmd))
        max_dev = float(np.max(joint_diff_deg))

        if max_dev <= self.JOINT_ALIGN_MAX_DEG:
            return current_qpos  # close enough — inner loop delta clamp handles the rest

        assert self.planner is not None
        assert self._arm_inner is not None

        print(f"\nArm is {max_dev:.1f}° from trajectory start (threshold: {self.JOINT_ALIGN_MAX_DEG}°)")
        print("Planning collision-checked approach to trajectory start ...")

        # ── Target EEF pose: prefer recorded arm_ee, fallback to FK ──
        if self.traj.arm_ee is not None and self.traj.arm_ee.shape[0] > 0:
            ee = self.traj.arm_ee[0]
            target_pos = ee[:3].copy()
            target_quat = rot6d_to_quat_wxyz(ee[3:9])
        else:
            pose = self.planner.compute_eef_pose_world(first_cmd)
            target_pos = pose.p.copy()
            target_quat = pose.q.copy()

        target_pose = Pose(p=target_pos, q=target_quat)

        # ── Plan ──
        try:
            path_result = self.planner.plan_path(
                target_eef_pose_world=target_pose,
                current_qpos=current_qpos.copy(),
            )
        except Exception as e:
            logger.error("plan_path raised exception during start alignment: %s", e)
            print(f"\nApproach planning failed: {e}")
            print("Aborting replay for safety. Manually return the arm near the trajectory start and retry.")
            return None

        if not path_result.success or path_result.qpos_path is None:
            print(f"\nApproach planning failed: {path_result.reason}")
            print("Aborting replay for safety. Manually return the arm near the trajectory start and retry.")
            return None

        qpos_path = path_result.qpos_path
        print(f"Approach path planned: {qpos_path.shape[0]} waypoints " f"(source={path_result.source})")
        print(f"Executing approach at ~{np.rad2deg(0.035) / HOME_DT:.0f}°/s ...")

        # ── Execute approach waypoints through inner loop ──
        for i, waypoint in enumerate(qpos_path):
            self._arm_inner.set_target(waypoint)
            time.sleep(HOME_DT)

        # Settle briefly
        time.sleep(0.1)

        # ── Read final position ──
        arm_qpos, error_state, _ = self._arm_inner.get_state()
        if arm_qpos is None or not np.all(np.isfinite(arm_qpos)) or np.all(arm_qpos == 0):
            assert self.robot is not None
            arm_qpos = self.robot.get_state().arm_qpos

        if arm_qpos is not None and np.all(np.isfinite(arm_qpos)):
            final_dev = np.rad2deg(np.max(np.abs(arm_qpos - first_cmd)))
            print(f"Approach complete. Remaining deviation: {final_dev:.2f}°")

        return arm_qpos

    # ── Run ──

    def run(self) -> dict[str, np.ndarray] | None:
        """Execute the replay loop. Returns replay data dict or None on dry-run/failure."""
        T = self._T
        print(f"\nReplay: {T} frames @ {self.replay_hz:.1f} Hz (speed={self.speed}x)")
        print(f"  Source: {self.traj.h5_path}")
        if self.traj.task_label:
            print(f"  Task:   {self.traj.task_label}")
        print(f"  Hand:   {'ON' if (self.traj.has_hand and not self.no_hand and not self.dry_run) else 'OFF'}")
        print(f"  Mode:   {'DRY RUN (no robot)' if self.dry_run else 'LIVE'}")
        print("\nControl: Q=quit  ESC=emergency_stop\n")

        if self.dry_run:
            self._dry_run_loop(T)
            return None

        assert self._arm_inner is not None
        assert self.robot is not None
        assert self.planner is not None

        has_hand = self.traj.has_hand and self._hand_available
        self._recorder = ReplayRecorder(T, has_hand=has_hand)

        # Read initial state
        arm_qpos, error_state, _ = self._arm_inner.get_state()
        if arm_qpos is None or not np.all(np.isfinite(arm_qpos)) or np.all(arm_qpos == 0):
            arm_qpos = self.robot.get_state().arm_qpos
        state = self.robot.get_state(arm_qpos=arm_qpos)

        # ── Start alignment: plan + execute collision-checked approach if
        #     the arm is not already near the trajectory start ──
        first_cmd = self.traj.action_arm_joint[0].copy()
        aligned_qpos = self._align_to_start(first_cmd, state.arm_qpos.copy())
        if aligned_qpos is None:
            print("Aborting replay: failed to align to trajectory start.")
            if self._recorder is not None:
                return self._recorder.to_dict()
            return None

        # Re-read state after alignment (inner loop may have settled)
        arm_qpos, error_state, _ = self._arm_inner.get_state()
        if arm_qpos is None or not np.all(np.isfinite(arm_qpos)) or np.all(arm_qpos == 0):
            arm_qpos = self.robot.get_state().arm_qpos
        state = self.robot.get_state(arm_qpos=arm_qpos)

        # Keyboard
        kb = KeyboardHandler()
        kb.start()
        atexit.register(kb.stop)

        # Send the first target BEFORE the rate limiter to pre-warm the inner loop
        self._arm_inner.set_target(first_cmd)
        self._rate_mgr = RateManager(self.replay_hz)  # reset timer after first send

        self._running = True
        error_count = 0
        validate_fail_count = 0
        max_consecutive_errors = 10
        start_time = time.perf_counter()
        last_arm_cmd: np.ndarray | None = None

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
                if self._hand_available and self.traj.action_hand_joint is not None:
                    hand_cmd = self.traj.action_hand_joint[frame_idx].copy()

                # NaN guard: if command has NaN, use current state
                if not np.all(np.isfinite(arm_cmd)):
                    logger.warning("Frame %d arm_cmd has NaN, using current state", frame_idx)
                    arm_cmd = state.arm_qpos.copy()

                # ── Read ArmInnerLoop state ──
                try:
                    arm_qpos, error_state, _inner_ts = self._arm_inner.get_state()
                    if error_state:
                        logger.warning("Arm inner loop error state at frame %d", frame_idx)
                        error_count += 1
                        if error_count > 3:
                            print("Arm inner loop consecutive errors, emergency_stop")
                            self._emergency_stop()
                            break
                        continue
                    # Dynamics from inner-loop 50Hz readback → torque/temp gates
                    arm_qvel, arm_tau, arm_temps = self._arm_inner.get_dynamics()
                    state = self.robot.get_state(arm_qpos=arm_qpos, arm_qvel=arm_qvel, arm_tau=arm_tau)
                except Exception as e:
                    error_count += 1
                    logger.warning("get_state exception at frame %d: %s", frame_idx, e)
                    if error_count > max_consecutive_errors:
                        print("Too many consecutive errors, emergency_stop")
                        self._emergency_stop()
                        break
                    continue

                # ── Robot error check ──
                if self.robot.arm.is_error():
                    arm_code = self.robot.arm.arm.error_code if self.robot.arm.arm else 0
                    sdk_code = self.robot.arm.last_sdk_error_code
                    if arm_code == 22 or sdk_code == 22:
                        print("  ⚠ ControllerError 22 (C31/C32), clearing and continuing", flush=True)
                        self.robot.arm.clear_error()
                        state = self.robot.get_state()
                        error_count = 0
                        continue
                    print(f"Arm error: C{arm_code}")
                    self._emergency_stop()
                    break

                if not np.all(np.isfinite(state.arm_qpos)):
                    error_count += 1
                    continue

                error_count = 0

                # ── Build action ──
                if self._hand_available and hand_cmd is not None:
                    action = RobotAction(
                        arm_qpos_cmd=arm_cmd,
                        hand_qpos_cmd=hand_cmd,
                    )
                else:
                    action = RobotAction(
                        arm_qpos_cmd=arm_cmd,
                        hand_qpos_cmd=np.zeros(12, dtype=np.float64),
                    )

                # ── Pre-send gate: torque/temp/soft-limit (autonomous replay →
                #     repeated failures abort instead of pressing on) ──
                action_valid, fail_reason = validate_action(
                    self.robot,
                    action,
                    actual_arm_qpos=arm_qpos,
                    actual_arm_tau=state.arm_tau,
                    actual_arm_temps=arm_temps,
                )
                if not action_valid:
                    validate_fail_count += 1
                    print(f"  [SAFETY] Pre-send gate: {fail_reason} — skip frame ({validate_fail_count}/3)", flush=True)
                    if validate_fail_count > 3:
                        print("Pre-send gate failed repeatedly, emergency_stop")
                        self._emergency_stop()
                        break
                    continue
                validate_fail_count = 0

                # ── Send arm command ──
                self._arm_inner.set_target(action.arm_qpos_cmd)
                last_arm_cmd = arm_cmd

                # ── Send hand command ──
                self.robot.send_action(action)

                # ── Record ──
                ts = time.perf_counter()
                if self._recorder is not None:
                    self._recorder.record(frame_idx, state, arm_cmd, hand_cmd, ts)

                # ── Periodic log ──
                if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[T+{elapsed:.1f}s f={frame_idx+1}/{T}] "
                        f"eef={np.round(state.eef_pos, 3)}m  "
                        f"err={error_count}",
                        flush=True,
                    )

        finally:
            kb.stop()

            # Hold position on clean exit
            if self._arm_inner is not None and self._arm_inner.is_alive:
                self._arm_inner.set_target(None)

            if self._recorder is not None:
                actual = self._recorder.count
                if actual < T:
                    print(f"\nReplay stopped at frame {actual}/{T}")
                return self._recorder.to_dict()

        return None

    def _emergency_stop(self) -> None:
        """Stop inner loop + emergency stop robot."""
        self._running = False
        if self._arm_inner is not None and self._arm_inner.is_alive:
            self._arm_inner.emergency_stop()  # plan §8 A5: hard-stop child within ≤1 tick
            self._arm_inner.set_target(None)
            self._arm_inner.stop()
        if self.robot is not None:
            self.robot.emergency_stop()
            self.robot.arm.clear_error()

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
        """Clean up robot resources."""
        if self.dry_run:
            return
        if self._arm_inner is not None and self._arm_inner.is_alive:
            self._arm_inner.set_target(None)
            self._arm_inner.stop()
            print("Arm inner loop stopped")
        if self.robot is not None:
            self.robot.disconnect()
            print("Robot disconnected")


# ═══════════════════════════════════════════════ CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded robot trajectory from HDF5 and evaluate consistency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/real/replay_traj.py --h5 episodes/episode_20260716_120000.h5
  python examples/real/replay_traj.py --h5 episode.h5 --speed 0.5 --max-frames 200
  python examples/real/replay_traj.py --h5 episode.h5 --dry-run
  python examples/real/replay_traj.py --h5 episode.h5 --no-hand --output results/
        """,
    )
    parser.add_argument("--h5", required=True, type=str, help="Path to HDF5 episode file (.h5).")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed factor (1.0=50Hz real-time, 0.5=half-speed).",
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
        default="192.168.1.111",
        help="XArm controller IP address.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="cmd",
        choices=["cmd", "sent"],
        help="Which arm action stream to replay: cmd (action_arm_joint, default) or sent (action_arm_joint_sent, schema v9+).",
    )
    args = parser.parse_args()

    # ── Validate args ──
    if args.speed <= 0:
        print("Error: --speed must be positive")
        sys.exit(1)
    if args.speed > 1.5:
        print(f"Warning: --speed={args.speed}x may exceed hardware limits (inner loop runs at 50Hz)")

    # ── Load trajectory ──
    try:
        traj = load_trajectory(args.h5, max_frames=args.max_frames, source=args.source)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Error loading HDF5: {e}")
        sys.exit(1)

    if traj.num_frames == 0:
        print("Error: trajectory has 0 frames")
        sys.exit(1)

    # ── Print trajectory info ──
    print(f"Trajectory: {traj.h5_path}")
    print(f"  Frames: {traj.num_frames}  FPS: {traj.fps:.1f}  Duration: {traj.num_frames/traj.fps:.1f}s")
    print(f"  Task: {traj.task_label or '(none)'}")
    print(f"  Hand data: {'yes' if traj.has_hand else 'no'}")
    print(f"  EE data: {'yes' if traj.arm_ee is not None else 'no'}")

    # ── Warn about missing evaluation data ──
    eval_available = traj.arm_qpos is not None and np.all(np.isfinite(traj.arm_qpos))
    if not eval_available:
        print("Warning: arm_qpos missing/invalid in HDF5, cannot evaluate consistency.")

    # ── Setup output dir ──
    if args.output is not None:
        output_dir = args.output
    else:
        episode_name = Path(args.h5).stem
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"{episode_name}_replay")
    print(f"Output: {output_dir}")

    # ── Replay ──
    replayer = TrajectoryReplayer(
        traj,
        speed=args.speed,
        dry_run=args.dry_run,
        no_hand=args.no_hand,
        arm_ip=args.arm_ip,
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

            # ── Print summary ──
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
            print("=" * 60)

            # ── Save ──
            save_results(metrics, replay_data, output_dir)
        elif replay_data is None and not args.dry_run:
            print("\nNo replay data collected (replay interrupted before any frames captured).")
        elif args.dry_run:
            pass  # no metrics on dry-run
        else:
            print("\nSkipping metrics: no replay data available")

    except (ConnectionError, RuntimeError) as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        replayer.shutdown()

        # Post-loop: offer return-to-home
        if not args.dry_run and replayer._arm_inner is not None:
            print("\nPress H to return_home, or Q to exit...")
            kb = KeyboardHandler()
            kb.start()
            try:
                while True:
                    sigs = {s for s in kb.poll(timeout=0.1)}
                    if ControlSignal.QUIT in sigs or ControlSignal.EMERGENCY_STOP in sigs:
                        break
                    if ControlSignal.HOME in sigs:
                        print("\nH: return_home")
                        # Need to re-create planner+robot since they may have been disconnected
                        try:
                            p = _make_planner(args.arm_ip)
                            r = _make_robot(p, args.arm_ip)
                            if r.connect().get("arm"):
                                new_inner = _do_return_home(
                                    r, p, replayer._arm_inner, arm_ip=args.arm_ip, inner_cfg=replayer._inner_cfg
                                )
                                replayer._arm_inner = new_inner
                            r.disconnect()
                        except Exception as exc:
                            print(f"return_home failed: {exc}")
                        print("Press Q to exit...")
            finally:
                kb.stop()

    print("Done.")


if __name__ == "__main__":
    main()
