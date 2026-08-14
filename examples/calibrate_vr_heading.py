#!/usr/bin/env python3
"""One-shot VR heading calibration: compute the fixed T_vr_to_robot matrix.

Starts a VR receiver, collects head (or wrist) orientation data, computes
the mean forward direction, and writes ``vr_transform.json`` to
``dexmani_real/config/``.

Usage::

    conda activate real_robot
    python examples/calibrate_vr_heading.py [--duration 10] [--ref head]

By default, uses **head** orientation — the operator faces the robot +X
direction.  Pass ``--ref wrist`` to use wrist orientation instead
(extend arm, fingers pointing toward robot +X).

Features:

- Head-based (default): simple — face the robot.
- Countdown (3-2-1): allows operator to settle into a stable pose.
- Outlier rejection: frames > 3σ from the circular mean are discarded.
- Duplicate-frame skipping: identical VR sequence numbers do not inflate the
  quality estimate.
- Quality grading: excellent (σ < 2°), good (σ < 5°), poor otherwise.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dexmani_real import ASSET_DIR, PACKAGE_DIR
from dexmani_real.planning.pose_utils import forward_from_quat_wxyz, normalize_quat_wxyz
from dexmani_real.recording.transaction import atomic_json_dump
from dexmani_real.sensor.vr_receiver_process import VRReceiverConfig, vr_loop
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig
from dexmani_real.teleop.vr_transform import (
    VR_TRANSFORM_CONVENTION,
    VR_TRANSFORM_MIN_FRAMES,
    VR_TRANSFORM_SCHEMA_VERSION,
)

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════

_OUTPUT_PATH = PACKAGE_DIR / "config" / "vr_transform.json"
_AUDIO_PATH = ASSET_DIR / "audio" / "轴向已标定.wav"

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_MIN_FORWARD_NORM = 1e-6
_POLL_INTERVAL_S = 0.01
_PRINT_INTERVAL_S = 5.0
_JOIN_TIMEOUT_S = 5.0
_TERMINATE_TIMEOUT_S = 1.0
_COUNTDOWN_DWELL_S = 1.0

# ═══════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HeadingCalibrationConfig:
    """Tunable parameters for VR heading calibration."""

    duration_s: float = 10.0
    port: int = 8000
    vr_ready_timeout_s: float = 120.0
    tracking_data_timeout_s: float = 15.0
    settle_s: float = 3.0
    min_frames: int = VR_TRANSFORM_MIN_FRAMES
    outlier_sigma: float = 3.0
    excellent_std_deg: float = 2.0
    good_std_deg: float = 5.0

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.min_frames < 5:
            raise ValueError("min_frames must be at least 5")
        if self.excellent_std_deg >= self.good_std_deg:
            raise ValueError("excellent_std_deg must be less than good_std_deg")


# ═══════════════════════════════════════════════════════════════════════
# Heading estimation
# ═══════════════════════════════════════════════════════════════════════


def _circular_mean(
    forwards: np.ndarray, *, outlier_sigma: float = 3.0
) -> tuple[float, np.ndarray, np.ndarray]:
    """Circular mean of 2D forward directions with outlier rejection.

    Args:
        forwards: (N, 3) array of forward direction vectors.
        outlier_sigma: rejection threshold in standard deviations.

    Returns:
        theta_rad: Mean heading angle in radians.
        mean_fwd: Mean unit 2D vector.
        inlier_mask: Boolean mask of kept frames (same length as input).
    """
    fwd_2d = forwards[:, :2]
    norms = np.linalg.norm(fwd_2d, axis=1)
    valid = norms >= _MIN_FORWARD_NORM
    n_bad = int(np.sum(~valid))
    if n_bad:
        print(f"  WARNING: {n_bad} frames with near-vertical forward (skipped)")
    fwd_2d = fwd_2d[valid]
    norms = norms[valid]
    fwd_unit = fwd_2d / norms[:, None]

    # First pass: unweighted circular mean.
    mean_0 = np.mean(fwd_unit, axis=0)
    mean_0 /= np.linalg.norm(mean_0)

    # Outlier rejection: |sin(Δθ)| ≈ angular distance from mean.
    dists = np.abs(np.cross(fwd_unit, mean_0))
    threshold = outlier_sigma * float(np.std(dists))
    inlier = dists <= threshold
    n_out = int(np.sum(~inlier))
    if n_out > 0:
        print(f"  INFO: rejected {n_out} outlier frames (> {outlier_sigma}σ)")

    # Second pass: mean of inliers only.
    fwd_inlier = fwd_unit[inlier]
    mean_fwd = np.mean(fwd_inlier, axis=0)
    mean_fwd /= np.linalg.norm(mean_fwd)
    theta = float(np.arctan2(mean_fwd[1], mean_fwd[0]))

    # Remap M-length inlier back to N-length array for caller indexing.
    full_inlier = np.zeros(len(forwards), dtype=bool)
    full_inlier[valid] = inlier
    return theta, mean_fwd, full_inlier


def _quality_grade(
    forwards: np.ndarray,
    theta_mean: float,
    inlier: np.ndarray,
    *,
    excellent_std_deg: float = 2.0,
    good_std_deg: float = 5.0,
) -> dict[str, float | str]:
    """Grade calibration quality from per-frame theta scatter.

    Returns machine-readable metrics consumed by the runtime preflight.
    """
    fwd_2d = forwards[:, :2]
    norms = np.linalg.norm(fwd_2d, axis=1)
    mask = (norms >= _MIN_FORWARD_NORM) & inlier
    fwd_unit = fwd_2d[mask] / norms[mask, None]
    thetas = np.arctan2(fwd_unit[:, 1], fwd_unit[:, 0])
    dtheta = np.angle(np.exp(1j * (thetas - theta_mean)))
    std_deg = float(np.rad2deg(np.std(dtheta)))
    max_dev = float(np.rad2deg(float(np.max(np.abs(dtheta)))))

    if std_deg < excellent_std_deg:
        grade = "excellent"
    elif std_deg < good_std_deg:
        grade = "good"
    else:
        grade = "poor"
    return {
        "grade": grade,
        "std_deg": std_deg,
        "max_deviation_deg": max_dev,
    }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _wait_for_vr_tracking(shared: SharedStorage, timeout_s: float) -> bool:
    """Block until VR tracking data is observed, or return False on timeout."""
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    while time.monotonic() < deadline:
        result = shared.vr_ring.read_latest()
        if result is not None:
            data, _ts, _seq = result
            hp = np.asarray(data["head_pos"][0], dtype=np.float64)
            if np.any(hp != 0):
                print("  VR tracking active\n", flush=True)
                return True
        now = time.monotonic()
        if now - last_print >= _PRINT_INTERVAL_S:
            elapsed = int(now - (deadline - timeout_s))
            print(f"    waiting... ({elapsed}s elapsed)", flush=True)
            last_print = now
        time.sleep(0.1)
    return False


def _fatal_exit(shared: SharedStorage, vr_proc: mp.Process, message: str) -> None:
    """Clean up and exit with an error message."""
    print(f"  ERROR: {message}", flush=True)
    shared.is_running.value = False
    vr_proc.join(timeout=_JOIN_TIMEOUT_S)
    shared.close()
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    cfg = HeadingCalibrationConfig()
    parser = argparse.ArgumentParser(description="VR heading calibration")
    parser.add_argument(
        "--duration", type=float, default=cfg.duration_s,
        help=f"collection duration (s), default: {cfg.duration_s}",
    )
    parser.add_argument("--port", type=int, default=cfg.port, help=f"VR TCP port, default: {cfg.port}")
    parser.add_argument(
        "--ref", choices=["wrist", "head"], default="head",
        help="reference: head=face robot +X, wrist=extend arm pointing at robot +X",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="write the transform even when quality grade is 'poor'",
    )
    args = parser.parse_args()

    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a positive finite number of seconds")

    ref_label = "head (face robot +X)" if args.ref == "head" else "wrist (extend arm, point at robot +X)"

    print("=" * 55)
    print("VR Heading Calibration")
    print(f"  reference:  {ref_label}")
    print(f"  duration:   {args.duration}s")
    print(f"  port:       {args.port}")
    print("=" * 55)

    # ── Start VR receiver ──
    ctx = mp.get_context("spawn")
    shared = SharedStorage.create(prefix="dexmani_vr_calib", config=SharedStorageConfig(), mp_context=ctx)
    vr_proc = ctx.Process(
        target=vr_loop, args=(shared, VRReceiverConfig(port=args.port)), name="vr-calib", daemon=True,
    )
    vr_proc.start()

    # Wait for headset-on event (not just TCP connect).
    print("\n  Waiting for VR connection (up to 120 s) — put on Quest headset...", flush=True)
    if not shared.wait_ready("vr", cfg.vr_ready_timeout_s):
        _fatal_exit(shared, vr_proc, "VR receiver startup timeout")
    print("  VR connected", flush=True)

    # Wait for the first tracking data (headset is on, data should arrive quickly).
    print("  Waiting for VR tracking data (up to 15 s)...", flush=True)
    if not _wait_for_vr_tracking(shared, cfg.tracking_data_timeout_s):
        _fatal_exit(shared, vr_proc, "no VR tracking data received")

    # ── Settle / countdown / collect (guarded so SharedStorage is always cleaned) ──
    try:
        # ── Settle period ──
        print(f"  Settling ({cfg.settle_s:.0f} s) — fine-tune your pose...", flush=True)
        time.sleep(cfg.settle_s)

        # ── Countdown ──
        if args.ref == "head":
            print("  Face the robot +X direction, hold your head still...")
        else:
            print("  Extend your right arm, point fingers toward robot +X, hold steady...")
        for i in [3, 2, 1]:
            print(f"  {i}...")
            time.sleep(_COUNTDOWN_DWELL_S)

        # ── Collect ──
        quat_field = "wrist_quat_wxyz" if args.ref == "wrist" else "head_quat_wxyz"
        forwards: list[np.ndarray] = []
        deadline = time.monotonic() + args.duration
        last_print = 0.0
        prev_seq: int | None = None
        stale_count = 0

        print(f"  Collecting {args.duration}s (hold still)...")
        while time.monotonic() < deadline:
            result = shared.vr_ring.read_latest()
            if result is None:
                time.sleep(_POLL_INTERVAL_S)
                continue

            data, _ts, _seq = result

            # Skip duplicate frames (stale VR data → falsely low std).
            if prev_seq is not None and _seq == prev_seq:
                stale_count += 1
                time.sleep(_POLL_INTERVAL_S)
                continue
            prev_seq = _seq

            q = np.asarray(data[quat_field][0], dtype=np.float64)
            if not np.all(np.isfinite(q)):
                continue

            # For head mode: skip frames without a valid head position.
            if args.ref == "head":
                hp = np.asarray(data["head_pos"][0], dtype=np.float64)
                if not np.any(hp != 0):
                    continue

            q = normalize_quat_wxyz(q)
            forwards.append(forward_from_quat_wxyz(q))

            now = time.monotonic()
            if now - last_print >= 1.0:
                print(f"    collected {len(forwards)} frames...")
                last_print = now
            time.sleep(_POLL_INTERVAL_S)
    finally:
        # ── Shutdown VR (always run, even on KeyboardInterrupt) ──
        shared.is_running.value = False
        vr_proc.join(timeout=_JOIN_TIMEOUT_S)
        if vr_proc.is_alive():
            vr_proc.terminate()
            vr_proc.join(timeout=_TERMINATE_TIMEOUT_S)
        shared.close()

    if len(forwards) < cfg.min_frames:
        print(f"ERROR: only {len(forwards)} frames collected (< {cfg.min_frames} required)")
        sys.exit(1)

    # ── Compute ──
    forwards_arr = np.array(forwards, dtype=np.float64)
    theta_rad, mean_fwd, inlier = _circular_mean(forwards_arr, outlier_sigma=cfg.outlier_sigma)
    inlier_frames = int(np.sum(inlier))
    if inlier_frames < cfg.min_frames:
        print(f"ERROR: only {inlier_frames} inlier frames remain (< {cfg.min_frames} required)")
        sys.exit(1)
    theta_deg = float(np.rad2deg(theta_rad))
    quality = _quality_grade(
        forwards_arr, theta_rad, inlier,
        excellent_std_deg=cfg.excellent_std_deg, good_std_deg=cfg.good_std_deg,
    )

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    T = np.array(
        [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Sanity check.
    corrected = T @ np.array([mean_fwd[0], mean_fwd[1], 0.0])

    print(f"\n{'=' * 55}")
    print("Calibration result")
    print(f"  valid frames:  {inlier_frames} "
          f"(rejected {int(np.sum(~inlier))} outliers, skipped {stale_count} dupes)")
    print(f"  forward:       [{mean_fwd[0]:.4f}, {mean_fwd[1]:.4f}]")
    print(f"  theta:         {theta_deg:.1f}°")
    print(
        "  quality:       "
        f"{quality['grade']} (σ={float(quality['std_deg']):.1f}°, "
        f"max={float(quality['max_deviation_deg']):.1f}°)"
    )
    print(f"  verification:  T·forward = [{corrected[0]:.4f}, {corrected[1]:.4f}] (expect [1, 0])")
    print(f"  T = R_z(-{theta_deg:.1f}°):")
    print(f"    [{T[0, 0]:.4f}, {T[0, 1]:.4f}, {T[0, 2]:.4f}],")
    print(f"    [{T[1, 0]:.4f}, {T[1, 1]:.4f}, {T[1, 2]:.4f}],")
    print(f"    [{T[2, 0]:.4f}, {T[2, 1]:.4f}, {T[2, 2]:.4f}]")
    print(f"{'=' * 55}")

    # ── Write config ──
    if quality["grade"] == "poor" and not args.force:
        print(
            f"\n  NOT written: quality grade is 'poor' "
            f"(σ={float(quality['std_deg']):.1f}°). Re-collect a steadier sample, "
            f"or pass --force to write anyway."
        )
        sys.exit(1)

    if _OUTPUT_PATH.exists():
        backup = _OUTPUT_PATH.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(_OUTPUT_PATH, backup)
        print(f"  backed up previous transform → {backup.name}")

    config = {
        "schema_version": VR_TRANSFORM_SCHEMA_VERSION,
        "description": "Fixed VR-to-robot transform (FLU→robot frame)",
        "T_vr_to_robot": T.tolist(),
        "theta_deg": theta_deg,
        "convention": VR_TRANSFORM_CONVENTION,
        "ref": args.ref,
        "quality": {**quality, "frames": inlier_frames},
    }
    atomic_json_dump(config, _OUTPUT_PATH)
    print(f"\nSaved to: {_OUTPUT_PATH}")

    # ── Audio feedback ──
    if _AUDIO_PATH.exists():
        player = "aplay" if sys.platform == "linux" else "afplay"
        subprocess.run([player, str(_AUDIO_PATH)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  Audio feedback played")


if __name__ == "__main__":
    main()
