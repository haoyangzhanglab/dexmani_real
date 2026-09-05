#!/usr/bin/env python3
"""Calibrate ``T_vr_to_robot`` from live VR orientation samples.

This starts the VR receiver, writes ``config/vr_transform.json`` after quality
checks, and never connects to or commands the robot.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexmani_real import PACKAGE_DIR
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.planning.poses import forward_from_quat_wxyz, normalize_quat_wxyz
from dexmani_real.runtime.supervisor import wait_subsystem_ready
from dexmani_real.runtime.workers import (
    WorkerSpec,
    build_processes,
    shutdown_processes_verified,
    start_processes,
)
from dexmani_real.sensor.vr_worker import VRReceiverConfig, vr_loop
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.vr_transform import (
    VR_TRANSFORM_CONVENTION,
    VR_TRANSFORM_MIN_FRAMES,
    VR_TRANSFORM_SCHEMA_VERSION,
)
from dexmani_real.utils.atomic_io import atomic_json_dump

_OUTPUT_PATH = PACKAGE_DIR / "config" / "vr_transform.json"

_MIN_FORWARD_NORM = 1e-6
_POLL_INTERVAL_S = 0.01
_PRINT_INTERVAL_S = 5.0
_JOIN_TIMEOUT_S = 5.0
_TERMINATE_TIMEOUT_S = 1.0
_KILL_TIMEOUT_S = 1.0
_COUNTDOWN_DWELL_S = 1.0
_AUDIO_IDLE_TIMEOUT_S = 5.0


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


def _circular_mean(
    forwards: np.ndarray, *, outlier_sigma: float = 3.0
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the circular mean heading and an inlier mask."""
    fwd_2d = forwards[:, :2]
    norms = np.linalg.norm(fwd_2d, axis=1)
    valid = norms >= _MIN_FORWARD_NORM
    n_bad = int(np.sum(~valid))
    if n_bad:
        print(f"  WARNING: {n_bad} frames with near-vertical forward (skipped)")
    fwd_2d = fwd_2d[valid]
    norms = norms[valid]
    if fwd_2d.shape[0] == 0:
        raise ValueError("all VR forward samples are near-vertical")
    fwd_unit = fwd_2d / norms[:, None]

    mean_0 = np.mean(fwd_unit, axis=0)
    mean_0_norm = float(np.linalg.norm(mean_0))
    if mean_0_norm < _MIN_FORWARD_NORM:
        raise ValueError("VR forward samples have no stable mean heading")
    mean_0 /= mean_0_norm

    # Outlier rejection: |sin(Δθ)| ≈ angular distance from mean.
    dists = np.abs(np.cross(fwd_unit, mean_0))
    threshold = outlier_sigma * float(np.std(dists))
    inlier = dists <= threshold
    n_out = int(np.sum(~inlier))
    if n_out > 0:
        print(f"  INFO: rejected {n_out} outlier frames (> {outlier_sigma}σ)")

    fwd_inlier = fwd_unit[inlier]
    mean_fwd = np.mean(fwd_inlier, axis=0)
    mean_fwd_norm = float(np.linalg.norm(mean_fwd))
    if mean_fwd_norm < _MIN_FORWARD_NORM:
        raise ValueError("VR heading inliers have no stable mean direction")
    mean_fwd /= mean_fwd_norm
    theta = float(np.arctan2(mean_fwd[1], mean_fwd[0]))

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


def _wait_for_vr_tracking(shared: RuntimeChannels, timeout_s: float) -> bool:
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


def _shutdown_vr_receiver(shared: RuntimeChannels, vr_proc: mp.Process) -> bool:
    """Return whether the receiver and every IPC resource stopped cleanly."""
    if vr_proc.pid is None:
        shared.is_running.value = False
        return shared.close()
    try:
        report = shutdown_processes_verified(
            shared,
            [vr_proc],
            graceful_timeout_s=_JOIN_TIMEOUT_S,
            terminate_timeout_s=_TERMINATE_TIMEOUT_S,
            kill_timeout_s=_KILL_TIMEOUT_S,
        )
    except RuntimeError as exc:
        print(f"  ERROR: VR receiver shutdown could not be verified: {exc}")
        return False
    if not report.clean:
        print(f"  ERROR: VR receiver shutdown was not clean: {report.exits}")
    return report.clean


def _play_completion_audio() -> None:
    """Request the canonical completion cue without affecting calibration success."""
    audio: AudioFeedback | None = None
    try:
        audio = AudioFeedback()
        audio.play("calibrated")
        if not audio.wait_until_idle(timeout_s=_AUDIO_IDLE_TIMEOUT_S):
            print("  WARNING: audio feedback timed out")
    except Exception as exc:
        print(f"  WARNING: audio feedback unavailable: {exc}")
    finally:
        if audio is not None:
            try:
                audio.close()
            except Exception as exc:
                print(f"  WARNING: audio feedback cleanup failed: {exc}")


def main(argv: list[str] | None = None) -> int:
    cfg = HeadingCalibrationConfig()
    parser = argparse.ArgumentParser(description="VR heading calibration")
    parser.add_argument(
        "--duration",
        type=float,
        default=cfg.duration_s,
        help=f"collection duration (s), default: {cfg.duration_s}",
    )
    parser.add_argument(
        "--port", type=int, default=cfg.port, help=f"VR TCP port, default: {cfg.port}"
    )
    parser.add_argument(
        "--ref",
        choices=["wrist", "head"],
        default="head",
        help="reference: head=face robot +X, wrist=extend arm pointing at robot +X",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write the transform even when quality grade is 'poor'",
    )
    args = parser.parse_args(argv)

    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a positive finite number of seconds")

    ref_label = (
        "head (face robot +X)"
        if args.ref == "head"
        else "wrist (extend arm, point at robot +X)"
    )

    print("=" * 55)
    print("VR Heading Calibration")
    print(f"  reference:  {ref_label}")
    print(f"  duration:   {args.duration}s")
    print(f"  port:       {args.port}")
    print("=" * 55)

    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix="dexmani_vr_calib", config=RuntimeChannelsConfig(), mp_context=ctx
    )
    specs = [
        WorkerSpec(
            "vr-calib",
            vr_loop,
            (shared, VRReceiverConfig(port=args.port)),
            ready_name="vr",
            daemon=True,
        )
    ]
    vr_proc = build_processes(ctx, specs)[0]
    forwards: list[np.ndarray] = []
    stale_count = 0
    shutdown_clean = False
    try:
        start_processes([vr_proc])
        print(
            "\n  Waiting for VR connection (up to 120 s) — put on Quest headset...",
            flush=True,
        )
        if not wait_subsystem_ready(
            shared,
            [(specs[0], vr_proc)],
            {"vr": cfg.vr_ready_timeout_s},
        ):
            print("  ERROR: VR receiver startup timeout", flush=True)
            return 1
        print("  VR connected", flush=True)

        print("  Waiting for VR tracking data (up to 15 s)...", flush=True)
        if not _wait_for_vr_tracking(shared, cfg.tracking_data_timeout_s):
            print("  ERROR: no VR tracking data received", flush=True)
            return 1

        print(f"  Settling ({cfg.settle_s:.0f} s) — fine-tune your pose...", flush=True)
        time.sleep(cfg.settle_s)

        if args.ref == "head":
            print("  Face the robot +X direction, hold your head still...")
        else:
            print(
                "  Extend your right arm, point fingers toward robot +X, hold steady..."
            )
        for i in [3, 2, 1]:
            print(f"  {i}...")
            time.sleep(_COUNTDOWN_DWELL_S)

        quat_field = "wrist_quat_wxyz" if args.ref == "wrist" else "head_quat_wxyz"
        deadline = time.monotonic() + args.duration
        last_print = 0.0
        prev_seq: int | None = None

        print(f"  Collecting {args.duration}s (hold still)...")
        while time.monotonic() < deadline:
            result = shared.vr_ring.read_latest()
            if result is None:
                time.sleep(_POLL_INTERVAL_S)
                continue

            data, _ts, _seq = result

            # Ignore duplicate sequence numbers in the quality estimate.
            if prev_seq is not None and _seq == prev_seq:
                stale_count += 1
                time.sleep(_POLL_INTERVAL_S)
                continue
            prev_seq = _seq

            q = np.asarray(data[quat_field][0], dtype=np.float64)
            if not np.all(np.isfinite(q)):
                continue

            # Head calibration requires a valid head position.
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
        shutdown_clean = _shutdown_vr_receiver(shared, vr_proc)

    if not shutdown_clean:
        print("ERROR: VR receiver cleanup failed; calibration was not published")
        return 1

    if len(forwards) < cfg.min_frames:
        print(
            f"ERROR: only {len(forwards)} frames collected (< {cfg.min_frames} required)"
        )
        return 1

    forwards_arr = np.array(forwards, dtype=np.float64)
    try:
        theta_rad, mean_fwd, inlier = _circular_mean(
            forwards_arr, outlier_sigma=cfg.outlier_sigma
        )
    except ValueError as exc:
        print(f"ERROR: invalid heading sample: {exc}")
        return 1
    inlier_frames = int(np.sum(inlier))
    if inlier_frames < cfg.min_frames:
        print(
            f"ERROR: only {inlier_frames} inlier frames remain (< {cfg.min_frames} required)"
        )
        return 1
    theta_deg = float(np.rad2deg(theta_rad))
    quality = _quality_grade(
        forwards_arr,
        theta_rad,
        inlier,
        excellent_std_deg=cfg.excellent_std_deg,
        good_std_deg=cfg.good_std_deg,
    )

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    T = np.array(
        [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    corrected = T @ np.array([mean_fwd[0], mean_fwd[1], 0.0])

    print(f"\n{'=' * 55}")
    print("Calibration result")
    print(
        f"  valid frames:  {inlier_frames} "
        f"(rejected {int(np.sum(~inlier))} outliers, skipped {stale_count} dupes)"
    )
    print(f"  forward:       [{mean_fwd[0]:.4f}, {mean_fwd[1]:.4f}]")
    print(f"  theta:         {theta_deg:.1f}°")
    print(
        "  quality:       "
        f"{quality['grade']} (σ={float(quality['std_deg']):.1f}°, "
        f"max={float(quality['max_deviation_deg']):.1f}°)"
    )
    print(
        f"  verification:  T·forward = [{corrected[0]:.4f}, {corrected[1]:.4f}] (expect [1, 0])"
    )
    print(f"  T = R_z(-{theta_deg:.1f}°):")
    print(f"    [{T[0, 0]:.4f}, {T[0, 1]:.4f}, {T[0, 2]:.4f}],")
    print(f"    [{T[1, 0]:.4f}, {T[1, 1]:.4f}, {T[1, 2]:.4f}],")
    print(f"    [{T[2, 0]:.4f}, {T[2, 1]:.4f}, {T[2, 2]:.4f}]")
    print(f"{'=' * 55}")

    if quality["grade"] == "poor" and not args.force:
        print(
            f"\n  NOT written: quality grade is 'poor' "
            f"(σ={float(quality['std_deg']):.1f}°). Re-collect a steadier sample, "
            f"or pass --force to write anyway."
        )
        return 1

    if _OUTPUT_PATH.exists():
        backup = _OUTPUT_PATH.with_suffix(
            f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
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

    _play_completion_audio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
