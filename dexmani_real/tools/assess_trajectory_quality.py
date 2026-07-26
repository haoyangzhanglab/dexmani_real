#!/usr/bin/env python3
"""Assess recorded trajectory quality by classifying physical feasibility.

Replicates the velocity-adaptive tracking error threshold from
``ArmInnerLoop._monitor()`` (inner_loop.py:659-675) to classify each frame
as normal, elevated, or anomalous based on how well the arm tracked the
commanded joint positions during recording.

Episode classification:
    CLEAN      — < 1% anomalous frames (arm tracked commands well)
    MARGINAL   — 1-5% anomalous frames (occasional tracking degradation)
    DEGRADED   — > 5% anomalous frames (frequent physically impossible commands)

Usage:
    python -m dexmani_real.tools.assess_trajectory_quality episode.h5
    python -m dexmani_real.tools.assess_trajectory_quality episode.h5 --json report.json
    python -m dexmani_real.tools.assess_trajectory_quality episodes/*.h5 --summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# ── Constants (mirror inner_loop.py) ──
_TRACKING_NOISE_RAD: float = 0.07
_DEFAULT_JOINT_MAX_ACC_DEG_S2: float = 500.0
_DEFAULT_JOINT_MAX_SPEED_DEG_S: float = 120.0
_INNER_RATE: float = 30.0
_DEFAULT_ANOMALY_CAP_RAD: float = 0.50
_ADAPTIVE_MIN_RAD: float = 0.18
_ADAPTIVE_MAX_RAD: float = 0.60

# ── Classification thresholds ──
ANOMALY_CLEAN_MAX: float = 0.01  # 1%
ANOMALY_MARGINAL_MAX: float = 0.05  # 5%


@dataclass
class QualityReport:
    """Per-episode quality assessment result."""

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
        """Serialize to JSON-compatible dict (excludes per-frame arrays)."""
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


def _compute_adaptive_threshold(
    cmd_vel_rad_s: float,
    joint_max_acc_rad_s2: float,
    inner_loop_hz: float = 30.0,
    adaptive_max_rad: float = 0.60,
) -> float:
    """Replicate ArmInnerLoop._monitor adaptive threshold formula.

    Args:
        cmd_vel_rad_s: L-infinity commanded joint velocity (rad/s),
            clamped to joint_max_speed.
        joint_max_acc_rad_s2: Configured joint max acceleration (rad/s²).
        inner_loop_hz: Inner loop rate in Hz (from recording meta, default 30).
        adaptive_max_rad: Upper clamp for the adaptive threshold (from recording
            meta, default 0.60).

    Returns:
        Adaptive tracking error threshold in radians, clamped to
        [_ADAPTIVE_MIN_RAD, adaptive_max_rad].
    """
    if joint_max_acc_rad_s2 <= 0:
        return adaptive_max_rad
    steady = cmd_vel_rad_s / inner_loop_hz
    accel = cmd_vel_rad_s * cmd_vel_rad_s / joint_max_acc_rad_s2
    expected = steady + accel + _TRACKING_NOISE_RAD
    return float(np.clip(expected, _ADAPTIVE_MIN_RAD, adaptive_max_rad))


def assess_episode(
    h5_path: str,
    anomaly_cap_rad: float = _DEFAULT_ANOMALY_CAP_RAD,
) -> QualityReport | None:
    """Run quality assessment on one HDF5 episode.

    Args:
        h5_path: Path to the HDF5 episode file.
        anomaly_cap_rad: Absolute ceiling for anomalous classification (rad).

    Returns:
        QualityReport, or None if the file cannot be read or is missing
        required datasets.
    """
    if not os.path.isfile(h5_path):
        logger.error("File not found: %s", h5_path)
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            # ── Validate required datasets ──
            if "action_arm_joint" not in f or "arm_qpos" not in f:
                logger.error("%s: missing action_arm_joint or arm_qpos", h5_path)
                return None

            action = np.asarray(f["action_arm_joint"][:], dtype=np.float64)
            arm_qpos = np.asarray(f["arm_qpos"][:], dtype=np.float64)

            # ── Read parameters from meta ──
            # f.get("/meta") returns None when the group is missing (safe h5py idiom).
            # Using a dict default (f.get("/meta", {})) causes an AttributeError crash
            # on meta.attrs because the returned dict has no .attrs member.
            meta = f.get("/meta")
            if meta is None:
                joint_max_acc = float(np.deg2rad(_DEFAULT_JOINT_MAX_ACC_DEG_S2))
                joint_max_speed = float(np.deg2rad(_DEFAULT_JOINT_MAX_SPEED_DEG_S))
                control_hz = 16.0
                inner_loop_hz = _INNER_RATE
                adaptive_max_rad = _ADAPTIVE_MAX_RAD
            else:
                joint_max_acc = float(
                    np.deg2rad(meta.attrs.get("joint_max_acc", _DEFAULT_JOINT_MAX_ACC_DEG_S2))
                )
                joint_max_speed = float(
                    np.deg2rad(meta.attrs.get("joint_max_speed", _DEFAULT_JOINT_MAX_SPEED_DEG_S))
                )
                control_hz = float(meta.attrs.get("control_hz", 16.0))
                if not (1.0 <= control_hz <= 100.0):
                    control_hz = 16.0
                inner_loop_hz = float(meta.attrs.get("inner_loop_hz", _INNER_RATE))
                adaptive_max_rad = float(
                    meta.attrs.get("tracking_error_adaptive_max_rad", _ADAPTIVE_MAX_RAD)
                )
    except (OSError, RuntimeError) as e:
        logger.error("%s: failed to open/read HDF5: %s", h5_path, e)
        return None

    T = min(action.shape[0], arm_qpos.shape[0])
    if T < 2:
        logger.warning("%s: too few frames (%d)", h5_path, T)
        return None

    dt = 1.0 / control_hz

    # ── Per-frame tracking error (L-infinity across 7 joints) ──
    per_frame_error = np.max(np.abs(action[:T] - arm_qpos[:T]), axis=1)  # (T,) rad

    # ── Per-frame commanded velocity (L-infinity, from action deltas) ──
    cmd_deltas = np.abs(np.diff(action[:T], axis=0))  # (T-1, 7)
    cmd_vels = np.max(cmd_deltas, axis=1) / dt  # (T-1,) rad/s
    cmd_vels = np.clip(cmd_vels, 0.0, joint_max_speed)
    # Pad: frame 0 gets velocity from frames 0→1
    cmd_vels = np.concatenate([[cmd_vels[0]], cmd_vels])  # (T,) rad/s

    # ── Per-frame adaptive threshold ──
    per_frame_adaptive = np.array(
        [_compute_adaptive_threshold(v, joint_max_acc, inner_loop_hz, adaptive_max_rad)
         for v in cmd_vels],
        dtype=np.float64,
    )

    # ── Classify each frame ──
    anomalous_mask = (per_frame_error > 2.0 * per_frame_adaptive) | (
        per_frame_error >= anomaly_cap_rad
    )
    elevated_mask = (per_frame_error > per_frame_adaptive) & ~anomalous_mask

    n_anomalous = int(np.sum(anomalous_mask))
    n_elevated = int(np.sum(elevated_mask))
    anomaly_ratio = n_anomalous / T if T > 0 else 0.0
    elevated_ratio = n_elevated / T if T > 0 else 0.0

    # ── Classify episode ──
    if anomaly_ratio < ANOMALY_CLEAN_MAX:
        classification = "CLEAN"
    elif anomaly_ratio < ANOMALY_MARGINAL_MAX:
        classification = "MARGINAL"
    else:
        classification = "DEGRADED"

    # ── Per-joint statistics ──
    diff_per_joint = np.abs(np.rad2deg(action[:T] - arm_qpos[:T]))  # (T, 7) deg
    per_joint_mean = np.mean(diff_per_joint, axis=0)
    per_joint_p95 = np.percentile(diff_per_joint, 95, axis=0)
    per_joint_rmse = np.sqrt(np.mean(diff_per_joint**2, axis=0))

    overall_mean = float(np.mean(per_frame_error))
    overall_p95 = float(np.percentile(per_frame_error, 95))
    overall_max = float(np.max(per_frame_error))
    worst_frame = int(np.argmax(per_frame_error))
    worst_joint = int(np.argmax(np.abs(action[worst_frame] - arm_qpos[worst_frame])))

    report = QualityReport(
        episode_path=h5_path,
        num_frames=T,
        classification=classification,
        anomaly_ratio=anomaly_ratio,
        elevated_ratio=elevated_ratio,
        per_joint_mean_deg=per_joint_mean,
        per_joint_p95_deg=per_joint_p95,
        per_joint_rmse_deg=per_joint_rmse,
        overall_mean_deg=float(np.rad2deg(overall_mean)),
        overall_p95_deg=float(np.rad2deg(overall_p95)),
        overall_max_deg=float(np.rad2deg(overall_max)),
        worst_frame=worst_frame,
        worst_joint=worst_joint + 1,  # 1-indexed for display
        joint_max_acc_deg_s2=float(np.rad2deg(joint_max_acc)),
        per_frame_anomalous=anomalous_mask,
        per_frame_error_rad=per_frame_error,
        per_frame_adaptive_rad=per_frame_adaptive,
    )

    return report


def print_report(report: QualityReport) -> None:
    """Print a human-readable quality report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Trajectory Quality Assessment")
    print(f"{'=' * 60}")
    print(f"  Episode:        {report.episode_path}")
    print(f"  Frames:         {report.num_frames}")
    print(f"  Joint max acc:  {report.joint_max_acc_deg_s2:.0f} °/s²")
    print(f"  Classification: {report.classification}")
    print(f"  Anomalous:      {report.anomaly_ratio*100:.1f}% ({int(report.anomaly_ratio*report.num_frames)} frames)")
    print(f"  Elevated:       {report.elevated_ratio*100:.1f}% ({int(report.elevated_ratio*report.num_frames)} frames)")
    print(f"  Tracking error (L∞ across joints):")
    print(f"    Overall:      mean={report.overall_mean_deg:.2f}°  p95={report.overall_p95_deg:.2f}°  max={report.overall_max_deg:.2f}°")
    print(f"    Per-joint mean:  {np.array2string(report.per_joint_mean_deg, precision=2, separator=', ')}")
    print(f"    Per-joint p95:   {np.array2string(report.per_joint_p95_deg, precision=2, separator=', ')}")
    print(f"    Per-joint rmse:  {np.array2string(report.per_joint_rmse_deg, precision=2, separator=', ')}")
    print(f"  Worst frame:    {report.worst_frame} (J{report.worst_joint} = {report.overall_max_deg:.2f}°)")
    print(f"{'=' * 60}")

    if report.classification == "DEGRADED":
        print(
            "\n  ⚠  DEGRADED: >5% of frames have physically impossible commands.\n"
            "     The arm could not track these commands during recording.\n"
            "     Policy training on this episode may learn infeasible actions."
        )
    elif report.classification == "MARGINAL":
        print(
            "\n  ⚡ MARGINAL: 1-5% of frames show tracking degradation.\n"
            "     Consider reviewing the worst frames before using for training."
        )


# ── CLI ──


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess recorded trajectory quality by classifying physical feasibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dexmani_real.tools.assess_trajectory_quality episode.h5
  python -m dexmani_real.tools.assess_trajectory_quality episode.h5 --json report.json
  python -m dexmani_real.tools.assess_trajectory_quality episodes/*.h5 --summary
        """,
    )
    parser.add_argument(
        "h5_files",
        nargs="+",
        type=str,
        help="One or more HDF5 episode files to assess.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Save JSON report to this file (single episode only).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a one-line summary per episode (useful for batch runs).",
    )
    parser.add_argument(
        "--anomaly-cap",
        type=float,
        default=_DEFAULT_ANOMALY_CAP_RAD,
        metavar="RAD",
        help=f"Absolute anomalous ceiling in rad (default: {_DEFAULT_ANOMALY_CAP_RAD}).",
    )
    args = parser.parse_args()

    reports: list[QualityReport] = []
    for h5_path in args.h5_files:
        report = assess_episode(h5_path, anomaly_cap_rad=args.anomaly_cap)
        if report is None:
            print(f"SKIP: {h5_path} (could not assess)", file=sys.stderr)
            continue
        reports.append(report)

        if args.summary:
            print(
                f"{report.classification:>9s}  {report.anomaly_ratio*100:5.1f}% anomalous  "
                f"p95={report.overall_p95_deg:5.1f}°  max={report.overall_max_deg:5.1f}°  "
                f"{report.episode_path}"
            )
        else:
            print_report(report)

    # ── JSON output (single episode only) ──
    if args.json and len(reports) == 1:
        report = reports[0]
        json_dict = report.to_dict()
        # Include per-frame data in JSON output
        if report.per_frame_anomalous is not None:
            json_dict["per_frame_anomalous"] = report.per_frame_anomalous.tolist()
        if report.per_frame_error_rad is not None:
            json_dict["per_frame_error_deg"] = np.rad2deg(report.per_frame_error_rad).tolist()
        if report.per_frame_adaptive_rad is not None:
            json_dict["per_frame_adaptive_deg"] = np.rad2deg(report.per_frame_adaptive_rad).tolist()
        with open(args.json, "w") as fp:
            json.dump(json_dict, fp, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved: {args.json}")

    # ── Batch summary ──
    if len(reports) > 1:
        clean = sum(1 for r in reports if r.classification == "CLEAN")
        marginal = sum(1 for r in reports if r.classification == "MARGINAL")
        degraded = sum(1 for r in reports if r.classification == "DEGRADED")
        print(f"\nBatch summary ({len(reports)} episodes):")
        print(f"  CLEAN:    {clean}")
        print(f"  MARGINAL: {marginal}")
        print(f"  DEGRADED: {degraded}")

    # Exit code: non-zero if any DEGRADED episodes found
    if any(r.classification == "DEGRADED" for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
