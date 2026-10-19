#!/usr/bin/env python3
"""采集实时 VR 数据标定朝向，通过质量检查后保存对齐变换。

python examples/calibrate_vr_heading.py
python examples/calibrate_vr_heading.py --ref wrist --duration 5 --output vr_transform.json

参数：--ref 选择 head/wrist 参考，--duration 为采样秒数，--output 指定变换文件。
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.sensor.vr.config import VRParams
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.runtime.state import RuntimeState
from dexmani_real.planning.kinematics.pose import forward_from_quat_wxyz
from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
from dexmani_real.sensor.vr.worker import run_vr_worker
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.vr_transform import (
    VR_TRANSFORM_MIN_FRAMES,
    parse_vr_transform,
)
from dexmani_real.utils.atomic_io import atomic_json_dump

_MIN_FORWARD_NORM = 1e-6
_POLL_INTERVAL_S = 0.01
_PRINT_INTERVAL_S = 5.0
_JOIN_TIMEOUT_S = 5.0
_COUNTDOWN_DWELL_S = 1.0
_AUDIO_IDLE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class HeadingCalibrationConfig:
    duration_s: float = 10.0
    port: int = 8000
    vr_ready_timeout_s: float = 120.0
    tracking_data_timeout_s: float = 15.0
    settle_s: float = 3.0
    min_frames: int = VR_TRANSFORM_MIN_FRAMES
    outlier_sigma: float = 3.0


def circular_mean(
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

    angles = np.arctan2(fwd_unit[:, 1], fwd_unit[:, 0])
    center = np.arctan2(mean_0[1], mean_0[0])
    delta = np.angle(np.exp(1j * (angles - center)))
    dists = np.abs(delta)
    threshold = max(1e-12, outlier_sigma * float(np.sqrt(np.mean(delta**2))))
    inlier = dists <= threshold
    n_out = int(np.sum(~inlier))
    if n_out > 0:
        print(f"  INFO: rejected {n_out} outlier frames (> {outlier_sigma}σ)")

    if not inlier.any():
        raise ValueError("no VR heading inliers")
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


def quality_measurements(
    forwards: np.ndarray,
    theta_mean: float,
    inlier: np.ndarray,
) -> dict[str, float]:
    """Measure heading scatter for runtime preflight."""
    fwd_2d = forwards[:, :2]
    norms = np.linalg.norm(fwd_2d, axis=1)
    mask = (norms >= _MIN_FORWARD_NORM) & inlier
    fwd_unit = fwd_2d[mask] / norms[mask, None]
    thetas = np.arctan2(fwd_unit[:, 1], fwd_unit[:, 0])
    dtheta = np.angle(np.exp(1j * (thetas - theta_mean)))
    std_deg = float(np.rad2deg(np.std(dtheta)))
    max_dev = float(np.rad2deg(float(np.max(np.abs(dtheta)))))

    return {
        "std_deg": std_deg,
        "max_deviation_deg": max_dev,
    }


def read_heading(runtime_state: RuntimeState, reference: str):
    result = runtime_state.sensors.vr_ring.read_latest(use_last_good=True)
    if result is None:
        return None
    frame = result[0][0]
    # Hand publications cache head pose; only its own receive time identifies a head sample.
    if reference == "head":
        stamp = int(frame["head_recv_ts_ns"])
        return (stamp, frame["head_quat_wxyz"]) if stamp > 0 else None
    return int(frame["sequence_id"]), frame["wrist_quat_wxyz"]


def wait_for_vr_tracking(
    runtime_state: RuntimeState,
    sensors: SensorProcesses,
    reference: str,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    while time.monotonic() < deadline:
        if not sensors.check():
            return False
        if read_heading(runtime_state, reference) is not None:
            print("  VR tracking active\n", flush=True)
            return True
        now = time.monotonic()
        if now - last_print >= _PRINT_INTERVAL_S:
            elapsed = int(now - (deadline - timeout_s))
            print(f"    waiting... ({elapsed}s elapsed)", flush=True)
            last_print = now
        time.sleep(0.1)
    return False


def play_completion_audio() -> None:
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
    parser.add_argument(
        "--output", type=Path, default=VR_TRANSFORM_PATH, help="VR alignment JSON destination"
    )
    args = parser.parse_args(argv)

    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a positive finite number of seconds")

    ref_label = (
        "head (face robot +X)" if args.ref == "head" else "wrist (extend arm, point at robot +X)"
    )

    print("=" * 55)
    print("VR Heading Calibration")
    print(f"  reference:  {ref_label}")
    print(f"  duration:   {args.duration}s")
    print(f"  port:       {args.port}")
    print("=" * 55)

    vr_config = VRParams(port=args.port)
    vr_config.validate()
    from dexmani_real.utils.log import configure_logging

    configure_logging()
    ctx = mp.get_context("spawn")
    runtime_state = RuntimeState(SensorChannels(ctx, vr=True))
    sensors = SensorProcesses(runtime_state)
    vr_proc = ctx.Process(
        name="vr", target=run_vr_worker, args=(runtime_state.sensors, vr_config), daemon=True
    )
    forwards: list[np.ndarray] = []
    stale_count = 0
    try:
        print(
            "\n  Waiting for VR connection (up to 120 s) — put on Quest headset...",
            flush=True,
        )
        try:
            sensors.start(
                vr_proc,
                ready=runtime_state.sensors.vr_ready,
                timeout_s=cfg.vr_ready_timeout_s,
            )
        except RuntimeError as exc:
            print(f"  ERROR: VR receiver startup failed: {exc}", flush=True)
            return 1
        print("  VR connected", flush=True)

        print("  Waiting for VR tracking data (up to 15 s)...", flush=True)
        if not wait_for_vr_tracking(runtime_state, sensors, args.ref, cfg.tracking_data_timeout_s):
            print("  ERROR: no VR tracking data received", flush=True)
            return 1

        print(f"  Settling ({cfg.settle_s:.0f} s) — fine-tune your pose...", flush=True)
        time.sleep(cfg.settle_s)

        if args.ref == "head":
            print("  Face the robot +X direction, hold your head still...")
        else:
            print("  Extend your right arm, point fingers toward robot +X, hold steady...")
        for i in [3, 2, 1]:
            print(f"  {i}...")
            time.sleep(_COUNTDOWN_DWELL_S)

        deadline = time.monotonic() + args.duration
        last_print = 0.0
        prev_seq: int | None = None

        print(f"  Collecting {args.duration}s (hold still)...")
        while time.monotonic() < deadline:
            if not sensors.check():
                print("  ERROR: VR receiver exited during collection", flush=True)
                return 1
            sample = read_heading(runtime_state, args.ref)
            if sample is None:
                time.sleep(_POLL_INTERVAL_S)
                continue
            sequence, quaternion = sample
            # Ignore duplicate sequence numbers in the quality estimate.
            if sequence == prev_seq:
                stale_count += 1
                time.sleep(_POLL_INTERVAL_S)
                continue
            prev_seq = sequence
            forwards.append(forward_from_quat_wxyz(quaternion))

            now = time.monotonic()
            if now - last_print >= 1.0:
                print(f"    collected {len(forwards)} frames...")
                last_print = now
            time.sleep(_POLL_INTERVAL_S)
    finally:
        shutdown_clean = shutdown_local_runtime(
            runtime_state,
            sensors=sensors,
            timeout_s=_JOIN_TIMEOUT_S,
        )

    if not shutdown_clean:
        print("ERROR: VR receiver cleanup failed; calibration was not published")
        return 1

    if len(forwards) < cfg.min_frames:
        print(f"ERROR: only {len(forwards)} frames collected (< {cfg.min_frames} required)")
        return 1

    forwards_arr = np.array(forwards, dtype=np.float64)
    try:
        theta_rad, mean_fwd, inlier = circular_mean(forwards_arr, outlier_sigma=cfg.outlier_sigma)
    except ValueError as exc:
        print(f"ERROR: invalid heading sample: {exc}")
        return 1
    inlier_frames = int(np.sum(inlier))
    if inlier_frames < cfg.min_frames:
        print(f"ERROR: only {inlier_frames} inlier frames remain (< {cfg.min_frames} required)")
        return 1
    theta_deg = float(np.rad2deg(theta_rad))
    quality = quality_measurements(
        forwards_arr,
        theta_rad,
        inlier,
    )

    config = {
        "theta_deg": theta_deg,
        "ref": args.ref,
        "quality": {**quality, "frames": inlier_frames},
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    calibration = parse_vr_transform(config)
    T = calibration.transform

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
        f"{calibration.quality.grade} (σ={float(quality['std_deg']):.1f}°, "
        f"max={float(quality['max_deviation_deg']):.1f}°)"
    )
    print(f"  verification:  T·forward = [{corrected[0]:.4f}, {corrected[1]:.4f}] (expect [1, 0])")
    print(f"  T = R_z(-{theta_deg:.1f}°):")
    print(f"    [{T[0, 0]:.4f}, {T[0, 1]:.4f}, {T[0, 2]:.4f}],")
    print(f"    [{T[1, 0]:.4f}, {T[1, 1]:.4f}, {T[1, 2]:.4f}],")
    print(f"    [{T[2, 0]:.4f}, {T[2, 1]:.4f}, {T[2, 2]:.4f}]")
    print(f"{'=' * 55}")

    if calibration.quality.grade == "poor" and not args.force:
        print(
            f"\n  NOT written: quality grade is 'poor' "
            f"(σ={float(quality['std_deg']):.1f}°). Re-collect a steadier sample, "
            f"or pass --force to write anyway."
        )
        return 1

    atomic_json_dump(config, args.output)
    print(f"\nSaved to: {args.output}")

    play_completion_audio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
