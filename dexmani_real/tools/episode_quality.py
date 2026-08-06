#!/usr/bin/env python3
"""Episode quality toolkit — assess, health-check, filter, and validate recorded episodes.

API: ``EpisodeQuality(episode_dir)`` context manager with ``.assess()``, ``.health()``,
``.validate()``, ``.filter()`` methods.  Also provides convenience functions
``assess_episode()``, ``check_episode_health()``, ``batch_assess()``, ``batch_health()``.

CLI: ``python -m dexmani_real.tools.episode_quality {assess,health,filter,validate} ...``
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared constants
# ═══════════════════════════════════════════════════════════════════════════════

_TRACKING_NOISE_RAD: float = 0.07
_DEFAULT_JOINT_MAX_ACC_DEG_S2: float = 500.0
_DEFAULT_JOINT_MAX_SPEED_DEG_S: float = 120.0
_ARM_RATE: float = 30.0
_ADAPTIVE_MIN_RAD: float = 0.18
_ADAPTIVE_MAX_RAD: float = 0.60
_DEFAULT_ANOMALY_CAP_RAD: float = 0.50

_FRAME_OK = 0
_FRAME_HELD = 1
_FRAME_IK_FAIL = 2
_FRAME_SAFETY_REJECT = 3
_FRAME_RETARGET_FAIL = 4

ANOMALY_CLEAN_MAX: float = 0.01
ANOMALY_MARGINAL_MAX: float = 0.05

ASSUMED_CAMERA_FPS = 30.0
FILL_WARN_PCT = 10.0
CAM_DUP_WARN_MARGIN_PCT = 15.0
TRACK_P95_WARN_DEG = 20.0
FREEZE_REPORT_MIN_S = 0.2
HAND_TRACK_P95_WARN_DEG = 20.0
TACTILE_ALLZERO_WARN_PCT = 90.0
TACTILE_TIPBOARD_ERR_WARN = 1

SENSOR_NAMES = ["thumb", "index", "middle", "ring", "little"]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start_index, length) of each run of consecutive True in *mask*."""
    runs: list[tuple[int, int]] = []
    n = 0
    for i, v in enumerate(mask):
        if v:
            n += 1
        elif n:
            runs.append((i - n, n))
            n = 0
    if n:
        runs.append((len(mask) - n, n))
    return runs


def _compute_adaptive_threshold(
    cmd_vel_rad_s: float,
    joint_max_acc_rad_s2: float,
    arm_loop_hz: float = 30.0,
    adaptive_max_rad: float = 0.60,
) -> float:
    """Replicate arm_loop adaptive tracking error threshold formula."""
    if joint_max_acc_rad_s2 <= 0:
        return adaptive_max_rad
    steady = cmd_vel_rad_s / arm_loop_hz
    accel = cmd_vel_rad_s * cmd_vel_rad_s / joint_max_acc_rad_s2
    expected = steady + accel + _TRACKING_NOISE_RAD
    return float(np.clip(expected, _ADAPTIVE_MIN_RAD, adaptive_max_rad))


def _read_meta_defaults(f: h5py.File) -> dict:
    """Read tracking parameters from /meta with safe defaults."""
    meta = f.get("/meta")
    if meta is None:
        return {
            "joint_max_acc": float(np.deg2rad(_DEFAULT_JOINT_MAX_ACC_DEG_S2)),
            "joint_max_speed": float(np.deg2rad(_DEFAULT_JOINT_MAX_SPEED_DEG_S)),
            "control_hz": 16.0,
            "arm_loop_hz": _ARM_RATE,
            "adaptive_max_rad": _ADAPTIVE_MAX_RAD,
        }
    return {
        "joint_max_acc": float(np.deg2rad(meta.attrs.get("joint_max_acc", _DEFAULT_JOINT_MAX_ACC_DEG_S2))),
        "joint_max_speed": float(np.deg2rad(meta.attrs.get("joint_max_speed", _DEFAULT_JOINT_MAX_SPEED_DEG_S))),
        "control_hz": float(meta.attrs.get("control_hz", 16.0)),
        "arm_loop_hz": float(meta.attrs.get("arm_loop_hz", _ARM_RATE)),
        "adaptive_max_rad": float(meta.attrs.get("tracking_error_adaptive_max_rad", _ADAPTIVE_MAX_RAD)),
    }


def _is_episode_dir(path: Path) -> bool:
    return path.is_dir() and (path / "data.h5").exists()


def _is_legacy_episode(path: Path) -> bool:
    return path.is_file() and path.suffix == ".h5" and not path.name.startswith("depth")


def _find_episodes(data_dir: Path) -> list[Path]:
    episodes: list[Path] = []
    for entry in sorted(data_dir.iterdir()):
        if _is_episode_dir(entry) or _is_legacy_episode(entry):
            episodes.append(entry)
    return episodes


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses (API return types)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualityReport:
    """Per-episode trajectory quality assessment result."""

    episode_path: str
    num_frames: int
    classification: str  # "CLEAN", "MARGINAL", "DEGRADED"
    anomaly_ratio: float
    elevated_ratio: float
    per_joint_mean_deg: np.ndarray = field(default_factory=lambda: np.zeros(7))
    per_joint_p95_deg: np.ndarray = field(default_factory=lambda: np.zeros(7))
    per_joint_rmse_deg: np.ndarray = field(default_factory=lambda: np.zeros(7))
    overall_mean_deg: float = 0.0
    overall_p95_deg: float = 0.0
    overall_max_deg: float = 0.0
    worst_frame: int = 0
    worst_joint: int = 0
    joint_max_acc_deg_s2: float = _DEFAULT_JOINT_MAX_ACC_DEG_S2
    per_frame_anomalous: np.ndarray | None = None
    per_frame_error_rad: np.ndarray | None = None
    per_frame_adaptive_rad: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "episode_path": self.episode_path,
            "num_frames": self.num_frames,
            "classification": self.classification,
            "anomaly_ratio": round(self.anomaly_ratio, 4),
            "elevated_ratio": round(self.elevated_ratio, 4),
            "joint_max_acc_deg_s2": self.joint_max_acc_deg_s2,
            "overall_mean_deg": round(self.overall_mean_deg, 2),
            "overall_p95_deg": round(self.overall_p95_deg, 2),
            "overall_max_deg": round(self.overall_max_deg, 2),
            "worst_frame": self.worst_frame,
            "worst_joint": self.worst_joint,
            "per_joint_mean_deg": np.round(self.per_joint_mean_deg, 2).tolist(),
            "per_joint_p95_deg": np.round(self.per_joint_p95_deg, 2).tolist(),
            "per_joint_rmse_deg": np.round(self.per_joint_rmse_deg, 2).tolist(),
        }


@dataclass
class FreezeRun:
    """A single hand freeze event."""

    start_frame: int
    length: int
    max_cmd_gap_deg: float


@dataclass
class HealthReport:
    """Comprehensive per-episode data-quality report."""

    episode_path: str
    num_frames: int
    # Grid
    grid_fill_pct: float = 0.0
    # Camera
    camera_dup_pct: dict[str, float] = field(default_factory=dict)
    camera_expected_baseline_pct: float = 0.0
    cam_frames_dropped: int = 0
    # Arm tracking
    tracking_p95_deg: float | None = None
    tracking_over_pct: float = 0.0
    # Hand tracking
    hand_tracking_p95_deg: float | None = None
    hand_tracking_over_pct: float = 0.0
    # Hand freeze
    hand_freeze_runs: list[FreezeRun] = field(default_factory=list)
    hand_freeze_total_frames: int = 0
    # Tactile
    tactile_zero_pcts: dict[str, float] = field(default_factory=dict)
    tactile_nan_pct: float = 0.0
    tipboard_errors: int = 0
    # Meta
    is_truncated: bool = False
    stop_reason: str = ""
    flags: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    # Warnings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_path": self.episode_path,
            "num_frames": self.num_frames,
            "grid_fill_pct": round(self.grid_fill_pct, 2),
            "camera_dup_pct": {k: round(v, 2) for k, v in self.camera_dup_pct.items()},
            "cam_frames_dropped": self.cam_frames_dropped,
            "tracking_p95_deg": round(self.tracking_p95_deg, 2) if self.tracking_p95_deg is not None else None,
            "hand_tracking_p95_deg": (
                round(self.hand_tracking_p95_deg, 2) if self.hand_tracking_p95_deg is not None else None
            ),
            "hand_freeze_runs": len(self.hand_freeze_runs),
            "hand_freeze_total_frames": self.hand_freeze_total_frames,
            "tactile_zero_pcts": {k: round(v, 2) for k, v in self.tactile_zero_pcts.items()},
            "is_truncated": self.is_truncated,
            "warnings": self.warnings,
        }


@dataclass
class ValidationReport:
    """Aggregate pre-training validation report for one episode."""

    episode_path: str
    checks: list[dict] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.passed_count == len(self.checks) > 0

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c["passed"])

    @property
    def total_count(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict:
        return {
            "episode_path": self.episode_path,
            "is_valid": self.is_valid,
            "passed_checks": self.passed_count,
            "total_checks": self.total_count,
            "checks": self.checks,
        }


@dataclass
class FilterResult:
    """Result of a filter operation on one episode."""

    input_path: str
    output_path: str | None
    total_frames: int
    kept_frames: int
    dropped_held: int = 0
    dropped_ik_fail: int = 0
    dropped_safety_reject: int = 0
    dropped_retarget_fail: int = 0
    dropped_tracking_error: int = 0

    @property
    def keep_rate(self) -> float:
        return self.kept_frames / self.total_frames if self.total_frames > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# EpisodeQuality — context manager (primary API)
# ═══════════════════════════════════════════════════════════════════════════════


class EpisodeQuality:
    """Context manager for single-pass episode quality analysis.

    Usage::

        with EpisodeQuality("episode_001") as eq:
            report = eq.assess()       # → QualityReport
            health = eq.health()       # → HealthReport
            result = eq.filter(output_dir="filtered/", drop_held=True)

    All methods share one HDF5 open.  Intermediate arrays (arm_qpos,
    tracking error, etc.) are lazily cached so repeated access costs
    nothing.
    """

    def __init__(
        self,
        path: str | Path,
        roi_stride: int = 8,
        track_thresh_rad: float = 0.35,
        anomaly_cap_rad: float = _DEFAULT_ANOMALY_CAP_RAD,
    ) -> None:
        self._path = str(path)
        self._roi_stride = roi_stride
        self._track_thresh_rad = track_thresh_rad
        self._anomaly_cap_rad = anomaly_cap_rad
        self._reader: EpisodeReader | None = None
        self._cache: dict[str, Any] = {}

    # ── context manager ──

    def __enter__(self) -> EpisodeQuality:
        self._reader = EpisodeReader(self._path)
        self._h5f = self._reader.h5f
        self._meta = dict(self._h5f["meta"].attrs) if "meta" in self._h5f else {}
        self._params = _read_meta_defaults(self._h5f)
        self._n_frames = int(self._h5f["arm_qpos"].shape[0]) if "arm_qpos" in self._h5f else 0
        self._control_hz = float(self._meta.get("control_hz", 0.0) or 0.0)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._reader is not None:
            self._reader.close()
        self._cache.clear()

    # ── cached data access ──

    def _cached(self, key: str, loader: Any) -> Any:
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    @property
    def _arm_qpos(self) -> np.ndarray:
        return self._cached("arm_qpos", lambda: np.asarray(self._h5f["arm_qpos"][:], dtype=np.float64))

    @property
    def _action_arm_joint(self) -> np.ndarray:
        return self._cached("action_arm_joint", lambda: np.asarray(self._h5f["action_arm_joint"][:], dtype=np.float64))

    @property
    def _hand_qpos(self) -> np.ndarray:
        return self._cached("hand_qpos", lambda: np.asarray(self._h5f["hand_qpos"][:], dtype=np.float64))

    @property
    def _action_hand_joint(self) -> np.ndarray:
        return self._cached(
            "action_hand_joint", lambda: np.asarray(self._h5f["action_hand_joint"][:], dtype=np.float64)
        )

    # ── assess ──

    def assess(self) -> QualityReport | None:
        """Velocity-adaptive trajectory quality classification.

        Returns QualityReport with CLEAN/MARGINAL/DEGRADED classification,
        or None if required datasets are missing.
        """
        if "action_arm_joint" not in self._h5f or "arm_qpos" not in self._h5f:
            logger.error("%s: missing action_arm_joint or arm_qpos", self._path)
            return None

        action = self._action_arm_joint
        arm_qpos = self._arm_qpos
        T = min(action.shape[0], arm_qpos.shape[0])
        if T < 2:
            logger.warning("%s: too few frames (%d)", self._path, T)
            return None

        dt = 1.0 / self._params["control_hz"]
        per_frame_error = np.max(np.abs(action[:T] - arm_qpos[:T]), axis=1)

        cmd_deltas = np.abs(np.diff(action[:T], axis=0))
        cmd_vels = np.max(cmd_deltas, axis=1) / dt
        cmd_vels = np.clip(cmd_vels, 0.0, self._params["joint_max_speed"])
        cmd_vels = np.concatenate([[cmd_vels[0]], cmd_vels])

        steady = cmd_vels / self._params["arm_loop_hz"]
        accel = cmd_vels * cmd_vels / self._params["joint_max_acc"]
        expected = steady + accel + _TRACKING_NOISE_RAD
        per_frame_adaptive = np.clip(expected, _ADAPTIVE_MIN_RAD, self._params["adaptive_max_rad"])

        anomalous_mask = (per_frame_error > 2.0 * per_frame_adaptive) | (per_frame_error >= self._anomaly_cap_rad)
        elevated_mask = (per_frame_error > per_frame_adaptive) & ~anomalous_mask

        anomaly_ratio = int(np.sum(anomalous_mask)) / T
        elevated_ratio = int(np.sum(elevated_mask)) / T

        if anomaly_ratio < ANOMALY_CLEAN_MAX:
            classification = "CLEAN"
        elif anomaly_ratio < ANOMALY_MARGINAL_MAX:
            classification = "MARGINAL"
        else:
            classification = "DEGRADED"

        diff_per_joint = np.abs(np.rad2deg(action[:T] - arm_qpos[:T]))
        overall_mean = float(np.mean(per_frame_error))
        overall_max = float(np.max(per_frame_error))
        worst_frame = int(np.argmax(per_frame_error))
        worst_joint = int(np.argmax(np.abs(action[worst_frame] - arm_qpos[worst_frame])))

        return QualityReport(
            episode_path=self._path,
            num_frames=T,
            classification=classification,
            anomaly_ratio=anomaly_ratio,
            elevated_ratio=elevated_ratio,
            per_joint_mean_deg=np.mean(diff_per_joint, axis=0),
            per_joint_p95_deg=np.percentile(diff_per_joint, 95, axis=0),
            per_joint_rmse_deg=np.sqrt(np.mean(diff_per_joint**2, axis=0)),
            overall_mean_deg=float(np.rad2deg(overall_mean)),
            overall_p95_deg=float(np.rad2deg(np.percentile(per_frame_error, 95))),
            overall_max_deg=float(np.rad2deg(overall_max)),
            worst_frame=worst_frame,
            worst_joint=worst_joint + 1,
            joint_max_acc_deg_s2=float(np.rad2deg(self._params["joint_max_acc"])),
            per_frame_anomalous=anomalous_mask,
            per_frame_error_rad=per_frame_error,
            per_frame_adaptive_rad=per_frame_adaptive,
        )

    # ── health ──

    def health(self) -> HealthReport:
        """Comprehensive data-quality report.

        Returns a HealthReport with structured fields for grid fill,
        camera duplication, tracking error, hand freeze, tactile health,
        and flags/meta.
        """
        report = HealthReport(
            episode_path=self._path,
            num_frames=self._n_frames,
            camera_expected_baseline_pct=(
                100.0 * max(0.0, 1.0 - ASSUMED_CAMERA_FPS / self._control_hz) if self._control_hz > 0 else 0.0
            ),
            meta=self._meta,
        )

        # ── Grid fill ──
        if "timestamp" in self._h5f:
            ts = self._h5f["timestamp"][:]
            if len(ts) >= 2:
                dup = np.diff(ts) == 0.0
                report.grid_fill_pct = 100.0 * dup.sum() / len(ts)
                if report.grid_fill_pct > FILL_WARN_PCT:
                    report.warnings.append(
                        f"grid fill {report.grid_fill_pct:.1f}% > {FILL_WARN_PCT:.0f}% — decision loop overrun"
                    )

        # ── Camera content duplication ──
        cam_keys = [k for k in self._h5f.keys() if k == "rgb" or k.endswith("_rgb")]
        rgb_data = None
        if not cam_keys:
            try:
                assert self._reader is not None
                rgb_data = self._reader.read_camera_all("rgb")
                cam_keys.append("rgb")
            except KeyError:
                pass
        for key in cam_keys:
            data = rgb_data if (key == "rgb" and "rgb" not in self._h5f) else self._h5f[key]
            dup_pct = self._compute_cam_dup(data)
            report.camera_dup_pct[key] = dup_pct
            if dup_pct > report.camera_expected_baseline_pct + CAM_DUP_WARN_MARGIN_PCT:
                report.warnings.append(
                    f"{key} content dup {dup_pct:.1f}% exceeds baseline "
                    f"{report.camera_expected_baseline_pct:.0f}% — camera backlog drops"
                )

        report.cam_frames_dropped = int(self._meta.get("cam_frames_dropped", 0) or 0)
        if report.cam_frames_dropped > 0:
            report.warnings.append(f"cam_frames_dropped={report.cam_frames_dropped} — writer thread queue drops")

        # ── Arm tracking error ──
        if "action_arm_joint" in self._h5f and "arm_qpos" in self._h5f:
            err = np.max(np.abs(self._action_arm_joint - self._arm_qpos), axis=1)
            err = err[np.isfinite(err)]
            if err.size > 0:
                report.tracking_p95_deg = float(np.degrees(np.percentile(err, 95)))
                report.tracking_over_pct = 100.0 * float(np.mean(err > self._track_thresh_rad))
                if report.tracking_p95_deg > TRACK_P95_WARN_DEG:
                    report.warnings.append(
                        f"tracking error p95 {report.tracking_p95_deg:.1f}° > {TRACK_P95_WARN_DEG:.0f}°"
                    )

        # ── Hand tracking error ──
        if "action_hand_joint" in self._h5f and "hand_qpos" in self._h5f:
            err = np.max(np.abs(self._action_hand_joint - self._hand_qpos), axis=1)
            err = err[np.isfinite(err)]
            if err.size > 0:
                report.hand_tracking_p95_deg = float(np.degrees(np.percentile(err, 95)))
                report.hand_tracking_over_pct = 100.0 * float(np.mean(err > self._track_thresh_rad))
                if report.hand_tracking_p95_deg > HAND_TRACK_P95_WARN_DEG:
                    report.warnings.append(
                        f"hand tracking error p95 {report.hand_tracking_p95_deg:.1f}° > {HAND_TRACK_P95_WARN_DEG:.0f}°"
                    )

        # ── Hand freeze ──
        if "hand_qpos" in self._h5f and "action_hand_joint" in self._h5f:
            freeze_runs = self._detect_hand_freeze()
            report.hand_freeze_runs = [FreezeRun(s, l, g) for s, l, g in freeze_runs]
            report.hand_freeze_total_frames = sum(r.length for r in report.hand_freeze_runs)
            if report.hand_freeze_runs:
                report.warnings.append(
                    f"hand freeze {len(report.hand_freeze_runs)} runs / "
                    f"{100.0 * report.hand_freeze_total_frames / max(1, self._n_frames):.1f}% frames"
                )

        # ── Tactile ──
        report.tactile_zero_pcts = self._check_tactile_health(report)

        # ── Tipboard errors ──
        if "hand_tipboard_err" in self._h5f:
            report.tipboard_errors = int(np.sum(self._h5f["hand_tipboard_err"][:] != 0))
            if report.tipboard_errors > TACTILE_TIPBOARD_ERR_WARN:
                report.warnings.append("hand_tipboard_err has non-zero entries — fingertip PCB sensor fault")

        # ── Flags ──
        for k in ("flag_ik_ok", "flag_held", "flag_retarget_ok"):
            if k in self._h5f:
                report.flags[k] = 100.0 * float(np.mean(self._h5f[k][:]))

        # ── Truncation ──
        report.is_truncated = bool(self._meta.get("truncated", False))
        if report.is_truncated:
            report.stop_reason = str(self._meta.get("stop_reason", "-"))
            report.warnings.append(f"truncated=True (stop_reason={report.stop_reason})")

        return report

    def _compute_cam_dup(self, data: np.ndarray | h5py.Dataset) -> float:
        """Return content-duplication percentage for camera data."""
        T = data.shape[0]
        if T <= 1:
            return 0.0
        dup = np.zeros(T, dtype=bool)
        prev: int | None = None
        for t in range(T):
            h = hash(data[t, :: self._roi_stride, :: self._roi_stride].tobytes())
            dup[t] = prev is not None and h == prev
            prev = h
        return 100.0 * dup[1:].sum() / (T - 1)

    def _detect_hand_freeze(self) -> list[tuple[int, int, float]]:
        """Detect hand qpos freeze runs."""
        hq = self._hand_qpos
        cmd = self._action_hand_joint
        min_frames = max(8, int(self._control_hz * 0.5))
        cmd_active_thresh_rad = 0.05

        hq_step = np.max(np.abs(np.diff(hq, axis=0)), axis=1)
        cmd_step = np.max(np.abs(np.diff(cmd, axis=0)), axis=1)
        frozen_mask = np.concatenate([[False], hq_step < 1e-4])

        freeze_runs: list[tuple[int, int, float]] = []
        for start, length in _runs_of(frozen_mask):
            if length < min_frames:
                continue
            end = start + length
            cmd_slice = cmd_step[max(0, start - 1) : min(len(cmd_step), end)]
            if not np.any(cmd_slice >= cmd_active_thresh_rad):
                continue
            gap = np.max(np.abs(cmd[start:end] - hq[start:end]))
            freeze_runs.append((start, length, float(np.rad2deg(gap))))
        return freeze_runs

    def _check_tactile_health(self, report: HealthReport) -> dict[str, float]:
        """Compute tactile health stats; populate warnings on report."""
        result: dict[str, float] = {}

        if "hand_contact" not in self._h5f:
            return result

        force_sum = self._h5f["hand_contact"][:]
        if force_sum.size == 0:
            return result

        has_nan = not np.all(np.isfinite(force_sum))
        if has_nan:
            report.tactile_nan_pct = 100.0 * float(np.mean(~np.isfinite(force_sum)))

        force_mag = np.linalg.norm(force_sum, axis=2)
        for i, name in enumerate(SENSOR_NAMES):
            zero_pct = 100.0 * float(np.mean(force_mag[:, i] == 0.0))
            result[name] = zero_pct
            if zero_pct > TACTILE_ALLZERO_WARN_PCT:
                report.warnings.append(f"tactile {name} zero rate {zero_pct:.1f}% > {TACTILE_ALLZERO_WARN_PCT:.0f}%")

        return result

    # ── validate ──

    def validate(
        self,
        min_frames: int = 50,
        variance_epsilon: float = 1e-8,
    ) -> ValidationReport:
        """Pre-training automated quality checks (NaN, variance, camera, etc.)."""
        report = ValidationReport(episode_path=self._path)

        # NaN checks
        for key in ("arm_qpos", "arm_ee", "hand_qpos", "action_arm_joint", "action_arm_ee", "action_hand_joint"):
            if key not in self._h5f:
                continue
            data = np.asarray(self._h5f[key][:], dtype=np.float64)
            ok = not np.any(~np.isfinite(data))
            report.checks.append(
                {"name": "no_nan", "passed": ok, "detail": f"{key}: {'OK' if ok else 'CONTAINS NaN/Inf'}"}
            )

        # Variance checks
        for key in ("arm_qpos", "arm_ee", "hand_qpos", "action_arm_joint", "action_hand_joint"):
            if key not in self._h5f:
                continue
            data = np.asarray(self._h5f[key][:], dtype=np.float64)
            if data.ndim == 1:
                data = data[:, np.newaxis]
            var = np.var(data, axis=0)
            zero_var = int(np.sum(var < variance_epsilon))
            report.checks.append(
                {
                    "name": "non_zero_variance",
                    "passed": zero_var == 0,
                    "detail": f"{key}: OK" if zero_var == 0 else f"{key}: {zero_var}/{var.shape[0]} dims zero variance",
                }
            )

        # Min frames
        report.checks.append(
            {
                "name": "min_frames",
                "passed": self._n_frames >= min_frames,
                "detail": f"{self._n_frames} frames (min={min_frames})",
            }
        )

        # Duplicate frames
        if "arm_qpos" in self._h5f and "action_arm_joint" in self._h5f:
            obs = self._arm_qpos
            act = self._action_arm_joint
            if len(obs) >= 2:
                if "flag_held" in self._h5f:
                    active = ~np.asarray(self._h5f["flag_held"][:], dtype=bool)
                elif "flag_frame_status" in self._h5f:
                    active = np.asarray(self._h5f["flag_frame_status"][:], dtype=np.int32) != 1
                else:
                    active = np.ones(len(obs), dtype=bool)
                if active.sum() >= 2:
                    obs_diff = np.abs(np.diff(obs[active], axis=0)).sum(axis=1)
                    act_diff = np.abs(np.diff(act[active], axis=0)).sum(axis=1)
                    n_dup = max(int((obs_diff < 1e-4).sum()), int((act_diff < 1e-4).sum()))
                    report.checks.append(
                        {
                            "name": "no_duplicate_frames",
                            "passed": n_dup == 0,
                            "detail": f"{n_dup} duplicate frames" if n_dup else "No duplicate frames.",
                        }
                    )

        # Timestamp monotonicity
        if "timestamp" in self._h5f:
            ts = np.asarray(self._h5f["timestamp"][:], dtype=np.float64)
            if len(ts) >= 2:
                diffs = np.diff(ts)
                n_regressions = int(np.sum(diffs < 0))
                report.checks.append(
                    {
                        "name": "timestamp_monotonic",
                        "passed": n_regressions == 0,
                        "detail": (
                            f"{n_regressions} backwards timestamps" if n_regressions else "Timestamps non-decreasing"
                        ),
                    }
                )

        # Camera freshness
        try:
            assert self._reader is not None
            rgb = self._reader.read_camera_all("rgb")
            sample = rgb[: min(10, rgb.shape[0])]
            all_zero = all(np.count_nonzero(frame) == 0 for frame in sample)
            report.checks.append(
                {
                    "name": "camera_fresh",
                    "passed": not all_zero,
                    "detail": "Camera frames OK" if not all_zero else "All camera frames zero",
                }
            )
        except (KeyError, AttributeError):
            pass

        return report

    # ── filter ──

    def build_filter_mask(
        self,
        drop_held: bool = True,
        drop_ik_fail: bool = False,
        drop_safety_reject: bool = False,
        drop_retarget_fail: bool = False,
        max_tracking_error: float | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Build boolean mask of frames to KEEP.

        Returns (mask, counts) where mask is shape (T,) bool and counts
        is a dict of per-category dropped frame counts.
        """
        mask = np.ones(self._n_frames, dtype=bool)
        counts: dict[str, int] = {}

        if drop_held and "flag_held" in self._h5f:
            held = np.asarray(self._h5f["flag_held"][: self._n_frames], dtype=bool)
            counts["held"] = int(np.sum(held))
            mask &= ~held

        status_drops = drop_ik_fail or drop_safety_reject or drop_retarget_fail
        if status_drops and "flag_frame_status" in self._h5f:
            status = np.asarray(self._h5f["flag_frame_status"][: self._n_frames], dtype=np.int32)
            if drop_ik_fail:
                counts["ik_fail"] = int(np.sum(status == _FRAME_IK_FAIL))
                mask &= status != _FRAME_IK_FAIL
            if drop_safety_reject:
                counts["safety_reject"] = int(np.sum(status == _FRAME_SAFETY_REJECT))
                mask &= status != _FRAME_SAFETY_REJECT
            if drop_retarget_fail:
                counts["retarget_fail"] = int(np.sum(status == _FRAME_RETARGET_FAIL))
                mask &= status != _FRAME_RETARGET_FAIL

        if max_tracking_error is not None and "tracking_error" in self._h5f:
            te = np.asarray(self._h5f["tracking_error"][: self._n_frames], dtype=np.float64)
            valid = np.isfinite(te)
            counts["tracking_error"] = int(np.sum(valid & (te > max_tracking_error)))
            mask &= ~(valid & (te > max_tracking_error))

        return mask, counts

    def filter(
        self,
        output_dir: str | Path | None = None,
        *,
        mask: np.ndarray | None = None,
        drop_held: bool = True,
        drop_ik_fail: bool = False,
        drop_safety_reject: bool = False,
        drop_retarget_fail: bool = False,
        max_tracking_error: float | None = None,
    ) -> FilterResult:
        """Filter frames and optionally write cleaned HDF5.

        If *mask* is provided it is used directly.  Otherwise a mask is
        built from the keyword arguments.

        Returns FilterResult with per-category drop counts.
        """
        if mask is None:
            mask, counts = self.build_filter_mask(
                drop_held=drop_held,
                drop_ik_fail=drop_ik_fail,
                drop_safety_reject=drop_safety_reject,
                drop_retarget_fail=drop_retarget_fail,
                max_tracking_error=max_tracking_error,
            )
        else:
            counts = {}

        kept = int(np.sum(mask))
        total = self._n_frames

        result = FilterResult(
            input_path=self._path,
            output_path=None,
            total_frames=total,
            kept_frames=kept,
            dropped_held=counts.get("held", 0),
            dropped_ik_fail=counts.get("ik_fail", 0),
            dropped_safety_reject=counts.get("safety_reject", 0),
            dropped_retarget_fail=counts.get("retarget_fail", 0),
            dropped_tracking_error=counts.get("tracking_error", 0),
        )

        if output_dir is not None and kept > 0:
            input_path = Path(self._path)
            out_dir = Path(output_dir)
            out_name = input_path.name
            out_path = out_dir / out_name

            h5_path = input_path / "data.h5" if input_path.is_dir() else input_path
            out_h5_path = out_path / "data.h5" if input_path.is_dir() else out_path.with_suffix(".h5")
            out_h5_path.parent.mkdir(parents=True, exist_ok=True)

            # Collect time-series keys
            time_series_keys: list[str] = []
            for key in sorted(self._h5f.keys()):
                if key == "meta":
                    continue
                ds = self._h5f[key]
                if isinstance(ds, h5py.Dataset) and ds.ndim >= 1 and ds.shape[0] == total:
                    time_series_keys.append(key)

            with h5py.File(out_h5_path, "w") as out_f:
                if "meta" in self._h5f:
                    # MergedH5File.copy doesn't exist — copy attrs manually
                    meta_src = self._h5f["meta"]
                    out_meta = out_f.create_group("meta")
                    for k, v in meta_src.attrs.items():
                        out_meta.attrs[k] = v
                for key in time_series_keys:
                    data = np.asarray(self._h5f[key][:total])
                    out_f.create_dataset(key, data=data[mask], compression="gzip", compression_opts=4)
                if "meta" in out_f:
                    out_f["meta"].attrs["num_frames"] = kept
                    out_f["meta"].attrs["filter_original_frames"] = total
                    out_f["meta"].attrs["filter_kept_frames"] = kept

            if input_path.is_dir():
                for sidecar in ("depth.h5", "rgb.mp4"):
                    src = input_path / sidecar
                    if src.exists():
                        shutil.copy2(src, out_path / sidecar)

            result.output_path = str(out_h5_path)

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level convenience functions
# ═══════════════════════════════════════════════════════════════════════════════


def assess_episode(
    h5_path: str,
    anomaly_cap_rad: float = _DEFAULT_ANOMALY_CAP_RAD,
) -> QualityReport | None:
    """Convenience wrapper: open+assess+close one episode."""
    if not os.path.exists(h5_path):
        logger.error("Episode not found: %s", h5_path)
        return None
    try:
        with EpisodeQuality(h5_path, anomaly_cap_rad=anomaly_cap_rad) as eq:
            return eq.assess()
    except (OSError, RuntimeError) as e:
        logger.error("%s: failed to open: %s", h5_path, e)
        return None


def check_episode_health(
    h5_path: str,
    roi_stride: int = 8,
    track_thresh_rad: float = 0.35,
) -> HealthReport | None:
    """Convenience wrapper: open+health+close one episode."""
    if not os.path.exists(h5_path):
        logger.error("Episode not found: %s", h5_path)
        return None
    try:
        with EpisodeQuality(h5_path, roi_stride=roi_stride, track_thresh_rad=track_thresh_rad) as eq:
            return eq.health()
    except (OSError, RuntimeError) as e:
        logger.error("%s: failed to open: %s", h5_path, e)
        return None


def validate_episode(
    h5_path: str | Path,
    min_frames: int = 50,
    variance_epsilon: float = 1e-8,
) -> ValidationReport | None:
    """Convenience wrapper: open+validate+close one episode."""
    try:
        with EpisodeQuality(h5_path) as eq:
            return eq.validate(min_frames=min_frames, variance_epsilon=variance_epsilon)
    except (OSError, RuntimeError) as e:
        logger.error("%s: failed to open: %s", h5_path, e)
        return None


# ── Batch helpers ──


def batch_assess(
    paths: list[str],
    anomaly_cap_rad: float = _DEFAULT_ANOMALY_CAP_RAD,
    verbose: bool = True,
) -> list[QualityReport]:
    """Assess multiple episodes; returns list of QualityReport (failed skipped)."""
    reports: list[QualityReport] = []
    for p in paths:
        r = assess_episode(p, anomaly_cap_rad=anomaly_cap_rad)
        if r is not None:
            reports.append(r)
            if verbose:
                print(
                    f"  {r.classification:>9s}  {r.anomaly_ratio*100:5.1f}% anomalous  p95={r.overall_p95_deg:5.1f}°  {p}"
                )
        else:
            if verbose:
                print(f"  SKIP      {p}")
    return reports


def batch_health(
    paths: list[str],
    roi_stride: int = 8,
    track_thresh_rad: float = 0.35,
) -> list[HealthReport]:
    """Run health checks on multiple episodes."""
    reports: list[HealthReport] = []
    for p in paths:
        r = check_episode_health(p, roi_stride=roi_stride, track_thresh_rad=track_thresh_rad)
        if r is not None:
            reports.append(r)
    return reports


def batch_validate(
    paths: list[str | Path],
    min_frames: int = 50,
    variance_epsilon: float = 1e-8,
    verbose: bool = True,
) -> list[ValidationReport]:
    """Validate multiple episodes."""
    reports: list[ValidationReport] = []
    for p in paths:
        r = validate_episode(p, min_frames=min_frames, variance_epsilon=variance_epsilon)
        if r is not None:
            reports.append(r)
            if verbose:
                status = "PASS" if r.is_valid else "FAIL"
                print(f"  [{status}] {Path(p).name}: {r.passed_count}/{r.total_count} checks")
    return reports


def batch_filter(
    paths: list[str],
    output_dir: str | Path | None = None,
    verbose: bool = True,
    **filter_kwargs: Any,
) -> list[FilterResult]:
    """Filter multiple episodes to output_dir."""
    results: list[FilterResult] = []
    for p in paths:
        try:
            with EpisodeQuality(p) as eq:
                r = eq.filter(output_dir=output_dir, **filter_kwargs)
            results.append(r)
            if verbose:
                print(f"  {Path(p).name}: {r.kept_frames}/{r.total_frames} frames kept ({r.keep_rate:.1%})")
        except (OSError, RuntimeError) as e:
            logger.error("%s: failed: %s", p, e)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Print helpers (only print() calls in the file — used by CLI and Jupyter)
# ═══════════════════════════════════════════════════════════════════════════════


def print_quality_report(report: QualityReport) -> None:
    """Human-readable output for a QualityReport."""
    print(f"\n{'=' * 60}")
    print("Trajectory Quality Assessment")
    print(f"{'=' * 60}")
    print(f"  Episode:        {report.episode_path}")
    print(f"  Frames:         {report.num_frames}")
    print(f"  Joint max acc:  {report.joint_max_acc_deg_s2:.0f} °/s²")
    print(f"  Classification: {report.classification}")
    print(f"  Anomalous:      {report.anomaly_ratio*100:.1f}% ({int(report.anomaly_ratio*report.num_frames)} frames)")
    print(f"  Elevated:       {report.elevated_ratio*100:.1f}% ({int(report.elevated_ratio*report.num_frames)} frames)")
    print("  Tracking error (L∞ across joints):")
    print(
        f"    Overall:      mean={report.overall_mean_deg:.2f}°  p95={report.overall_p95_deg:.2f}°  max={report.overall_max_deg:.2f}°"
    )
    print(f"    Per-joint mean:  {np.array2string(report.per_joint_mean_deg, precision=2, separator=', ')}")
    print(f"    Per-joint p95:   {np.array2string(report.per_joint_p95_deg, precision=2, separator=', ')}")
    print(f"    Per-joint rmse:  {np.array2string(report.per_joint_rmse_deg, precision=2, separator=', ')}")
    print(f"  Worst frame:    {report.worst_frame} (J{report.worst_joint} = {report.overall_max_deg:.2f}°)")
    print(f"{'=' * 60}")

    if report.classification == "DEGRADED":
        print("\n  ⚠  DEGRADED: >5% of frames have physically impossible commands.")
    elif report.classification == "MARGINAL":
        print("\n  ⚡ MARGINAL: 1-5% of frames show tracking degradation.")


def print_health_report(report: HealthReport) -> None:
    """Human-readable output for a HealthReport."""
    print(f"\n=== {report.episode_path} ===")
    for k in sorted(report.meta):
        v = report.meta[k]
        if isinstance(v, np.ndarray):
            continue
        print(f"  meta  {k}={v:.1f}" if isinstance(v, float) else f"  meta  {k}={v}")

    dt = report.meta.get("control_hz", 0)
    dt = 1.0 / dt if dt and dt > 0 else float("nan")
    longest_grid = 0  # best-effort
    print(f"  grid fill: {report.grid_fill_pct:.1f}%  " f"(longest run ≈ {longest_grid * dt * 1000:.0f}ms)")

    for key, dup_pct in report.camera_dup_pct.items():
        print(f"  {key} content dup: {dup_pct:.1f}% " f"(expected baseline {report.camera_expected_baseline_pct:.0f}%)")

    if report.tracking_p95_deg is not None:
        print(f"  tracking error p95: {report.tracking_p95_deg:.1f}°  >threshold: {report.tracking_over_pct:.1f}%")

    if report.hand_tracking_p95_deg is not None:
        print(f"  hand tracking error p95: {report.hand_tracking_p95_deg:.1f}°")

    if report.hand_freeze_runs:
        freeze_desc = "  ".join(
            f"{r.length * dt:.1f}s@t={r.start_frame * dt:.1f}s(gap={r.max_cmd_gap_deg:.0f}°)"
            for r in report.hand_freeze_runs[:3]
        )
        print(
            f"  hand freeze: {len(report.hand_freeze_runs)} runs, "
            f"{report.hand_freeze_total_frames}/{report.num_frames} frames  {freeze_desc}"
        )
    else:
        print("  hand freeze: none")

    if report.tactile_zero_pcts:
        desc = "  ".join(f"{name}={v:.1f}%" for name, v in report.tactile_zero_pcts.items())
        print(f"  tactile force zero rate: {desc}")

    if report.flags:
        print("  flags: " + "  ".join(f"{k}={v:.1f}%" for k, v in report.flags.items()))

    for w in report.warnings:
        print(f"  ⚠ WARN: {w}")
    if not report.warnings:
        print("  ✓ no warnings")


def print_validation_report(report: ValidationReport) -> None:
    """Human-readable output for a ValidationReport."""
    status = "PASS" if report.is_valid else "FAIL"
    name = Path(report.episode_path).name
    print(f"  [{status}] {name}: {report.passed_count}/{report.total_count} checks")
    for c in report.checks:
        flag = "✓" if c["passed"] else "✗"
        print(f"    {flag} {c['name']}: {c['detail']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Thin CLI (delegates to the API above)
# ═══════════════════════════════════════════════════════════════════════════════


def _cli_filter(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    if _is_episode_dir(input_path) or _is_legacy_episode(input_path):
        episodes = [str(input_path)]
    elif input_path.is_dir():
        episodes = [str(p) for p in _find_episodes(input_path)]
        if not episodes:
            print(f"No episodes found in {input_path}")
            sys.exit(1)
        print(f"Found {len(episodes)} episode(s) in {input_path}")
    else:
        print(f"ERROR: {input_path} is not an episode or directory")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else None
    results = batch_filter(
        episodes,
        output_dir=output_dir,
        drop_held=args.drop_held,
        drop_ik_fail=args.drop_ik_fail,
        drop_safety_reject=args.drop_safety_reject,
        drop_retarget_fail=args.drop_retarget_fail,
        max_tracking_error=args.max_tracking_error,
    )
    total_frames = sum(r.total_frames for r in results)
    total_kept = sum(r.kept_frames for r in results)
    print(f"\n{'=' * 50}")
    print(f"Total: {total_frames} frames → {total_kept} kept ({total_frames - total_kept} dropped)")
    if total_frames > 0:
        print(f"Keep rate: {100.0 * total_kept / total_frames:.1f}%")
    print(f"{'=' * 50}")


def _cli_assess(args: argparse.Namespace) -> None:
    reports = batch_assess(args.h5_files, anomaly_cap_rad=args.anomaly_cap, verbose=False)
    for r in reports:
        if args.summary:
            print(
                f"{r.classification:>9s}  {r.anomaly_ratio*100:5.1f}% anomalous  "
                f"p95={r.overall_p95_deg:5.1f}°  max={r.overall_max_deg:5.1f}°  {r.episode_path}"
            )
        else:
            print_quality_report(r)

    if args.json and len(reports) == 1:
        r = reports[0]
        d = r.to_dict()
        if r.per_frame_anomalous is not None:
            d["per_frame_anomalous"] = r.per_frame_anomalous.tolist()
        if r.per_frame_error_rad is not None:
            d["per_frame_error_deg"] = np.rad2deg(r.per_frame_error_rad).tolist()
        if r.per_frame_adaptive_rad is not None:
            d["per_frame_adaptive_deg"] = np.rad2deg(r.per_frame_adaptive_rad).tolist()
        with open(args.json, "w") as fp:
            json.dump(d, fp, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved: {args.json}")

    if len(reports) > 1:
        clean = sum(1 for r in reports if r.classification == "CLEAN")
        marginal = sum(1 for r in reports if r.classification == "MARGINAL")
        degraded = sum(1 for r in reports if r.classification == "DEGRADED")
        print(f"\nBatch summary ({len(reports)} episodes):  CLEAN={clean}  MARGINAL={marginal}  DEGRADED={degraded}")

    if any(r.classification == "DEGRADED" for r in reports):
        sys.exit(1)


def _cli_health(args: argparse.Namespace) -> None:
    total_warns = 0
    for ep in args.episodes:
        path = Path(ep).expanduser().resolve()
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            total_warns += 1
            continue
        try:
            report = check_episode_health(str(path), roi_stride=args.roi_stride, track_thresh_rad=args.track_thresh_rad)
            if report is not None:
                print_health_report(report)
                total_warns += len(report.warnings)
            else:
                total_warns += 1
        except (OSError, KeyError, ValueError) as e:
            logger.error("Failed to check %s: %s", path, e)
            total_warns += 1

        if args.quality:
            qr = assess_episode(str(path))
            if qr is not None:
                print(
                    f"\n  [quality] {qr.classification:>9s}  {qr.anomaly_ratio*100:.1f}% anomalous  "
                    f"p95={qr.overall_p95_deg:.1f}°  max={qr.overall_max_deg:.1f}°  J{qr.worst_joint} worst"
                )
                if qr.classification == "DEGRADED":
                    total_warns += 1
            else:
                print("\n  [quality] SKIP — could not assess")

    if len(args.episodes) > 1:
        print(f"\n{len(args.episodes)} files total, {total_warns} warning(s)")
    sys.exit(1 if total_warns else 0)


def _cli_validate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    h5_paths = sorted(data_dir.glob("episode_*.h5"))
    if not h5_paths:
        print(f"No episode_*.h5 files found in {data_dir}")
        sys.exit(1)

    paths: list[str | Path] = [str(p) for p in h5_paths]
    reports = batch_validate(paths, min_frames=args.min_frames, variance_epsilon=args.variance_epsilon, verbose=True)

    total_pass = sum(1 for r in reports if r.is_valid)
    print(f"\n{total_pass}/{len(reports)} episodes passed validation")

    if args.output_json:
        with open(args.output_json, "w") as fp:
            json.dump([r.to_dict() for r in reports], fp, indent=2, default=str)
        print(f"Report saved: {args.output_json}")

    sys.exit(0 if total_pass == len(reports) else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode quality toolkit — API-first with thin CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fp = sub.add_parser("filter", help="Drop low-quality frames, write cleaned HDF5 copy")
    fp.add_argument("input", help="Episode directory, legacy .h5 file, or directory of episodes")
    fp.add_argument("--output", type=str, default=None, help="Output directory for filtered episodes")
    fp.add_argument("--drop-held", action="store_true", default=True)
    fp.add_argument("--keep-held", action="store_true", help="Keep held frames (overrides --drop-held)")
    fp.add_argument("--drop-ik-fail", action="store_true")
    fp.add_argument("--drop-safety-reject", action="store_true")
    fp.add_argument("--drop-retarget-fail", action="store_true")
    fp.add_argument(
        "--max-tracking-error", type=float, default=None, help="Drop frames with tracking_error > THRESHOLD (rad)"
    )

    ap = sub.add_parser("assess", help="Velocity-adaptive trajectory quality classification")
    ap.add_argument("h5_files", nargs="+", help="HDF5 episode files to assess")
    ap.add_argument("--json", type=str, default=None, metavar="PATH")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--anomaly-cap", type=float, default=_DEFAULT_ANOMALY_CAP_RAD)

    hp = sub.add_parser("health", help="Comprehensive data-quality health report")
    hp.add_argument("episodes", nargs="+", help="Episode path(s)")
    hp.add_argument("--roi-stride", type=int, default=8)
    hp.add_argument("--track-thresh-rad", type=float, default=0.35)
    hp.add_argument("--quality", action="store_true", help="Also run trajectory quality assessment")

    vp = sub.add_parser("validate", help="Pre-training automated quality checks")
    vp.add_argument("data_dir", help="Directory containing episode_*.h5 files")
    vp.add_argument("--min-frames", type=int, default=50)
    vp.add_argument("--variance-epsilon", type=float, default=1e-8)
    vp.add_argument("--output-json", type=str, default=None, help="Save validation report as JSON")

    args = parser.parse_args()

    if args.command == "filter":
        if args.keep_held:
            args.drop_held = False
        _cli_filter(args)
    elif args.command == "assess":
        _cli_assess(args)
    elif args.command == "health":
        _cli_health(args)
    elif args.command == "validate":
        _cli_validate(args)


if __name__ == "__main__":
    main()
