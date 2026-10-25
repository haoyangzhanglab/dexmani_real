#!/usr/bin/env python3
"""采集实时 VR 数据标定朝向，通过质量检查后保存对齐变换。"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dexmani_real.calibration.paths import VR_TRANSFORM_PATH
from dexmani_real.calibration.vr_alignment import (
    VR_TRANSFORM_MIN_FRAMES,
    VRTransformCalibration,
    circular_mean,
    parse_vr_transform,
    quality_measurements,
)
from dexmani_real.geometry.pose import forward_from_quat_wxyz
from dexmani_real.ipc.channels import SensorChannels
from dexmani_real.runtime.config import RuntimeTimeouts
from dexmani_real.runtime.sensor_processes import SensorProcesses
from dexmani_real.runtime.shutdown import shutdown_local_runtime
from dexmani_real.runtime.state import RuntimeState
from dexmani_real.sensor.vr.config import VRParams
from dexmani_real.sensor.vr.worker import run_vr_worker
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.utils.atomic_io import atomic_json_dump

POLL_INTERVAL_S = 0.01
PRINT_INTERVAL_S = 5.0
JOIN_TIMEOUT_S = 5.0
COUNTDOWN_DWELL_S = 1.0
AUDIO_IDLE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class HeadingCalibrationConfig:
    duration_s: float = 10.0
    tracking_data_timeout_s: float = 15.0
    settle_s: float = 3.0
    min_frames: int = VR_TRANSFORM_MIN_FRAMES


def publish_vr_transform(
    calibration: VRTransformCalibration,
    output: Path | None,
    *,
    force: bool = False,
) -> Path:
    """合格结果可更新当前标定；低质量结果只能显式导出到未占用的新文件。"""
    path = VR_TRANSFORM_PATH if output is None else output
    poor = calibration.quality.grade == "poor"
    if poor:
        if not force:
            raise ValueError(
                "quality grade is 'poor'; re-collect a steadier sample, "
                "or use --force --output NEW_FILE for diagnostic export"
            )
        if output is None or path.resolve() == VR_TRANSFORM_PATH.resolve():
            raise ValueError("poor calibration requires --output pointing to a new non-default file")
    payload = {
        "theta_deg": calibration.theta_deg,
        "ref": calibration.reference,
        "quality": asdict(calibration.quality),
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return atomic_json_dump(payload, path, overwrite=not poor)


def read_heading(channels: SensorChannels, reference: str):
    result = channels.vr_ring.read_latest(use_last_good=True)
    if result is None:
        return None
    frame = result[0][0]
    # Hand publications cache head pose; only its own receive time identifies a head sample.
    if reference == "head":
        stamp = int(frame["head_recv_ts_ns"])
        return (stamp, frame["head_quat_wxyz"]) if stamp > 0 else None
    return int(frame["sequence_id"]), frame["wrist_quat_wxyz"]


def wait_for_vr_tracking(
    channels: SensorChannels,
    sensors: SensorProcesses,
    reference: str,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    while time.monotonic() < deadline:
        if not sensors.check():
            return False
        if read_heading(channels, reference) is not None:
            print("  VR tracking active\n", flush=True)
            return True
        now = time.monotonic()
        if now - last_print >= PRINT_INTERVAL_S:
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
        if not audio.wait_until_idle(timeout_s=AUDIO_IDLE_TIMEOUT_S):
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
    safety = RuntimeTimeouts()
    parser = argparse.ArgumentParser(description="VR heading calibration")
    parser.add_argument(
        "--duration",
        type=float,
        default=cfg.duration_s,
        help=f"collection duration (s), default: {cfg.duration_s}",
    )
    parser.add_argument(
        "--port", type=int, default=VRParams.port, help=f"VR TCP port, default: {VRParams.port}"
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
        help="export poor-quality results only with --output pointing to a new non-default file",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"VR alignment JSON destination (default: {VR_TRANSFORM_PATH})",
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
    runtime_state = RuntimeState()
    channels = SensorChannels(ctx, vr=True)
    sensors = SensorProcesses(channels)
    vr_proc = ctx.Process(
        name="vr", target=run_vr_worker, args=(channels, vr_config), daemon=True
    )
    forwards: list[np.ndarray] = []
    stale_count = 0
    try:
        print(
            f"\n  Waiting for VR connection (up to {safety.vr_ready_timeout_s:g} s) "
            "— put on Quest headset...",
            flush=True,
        )
        try:
            sensors.start(
                vr_proc,
                ready=channels.vr_ready,
                timeout_s=safety.vr_ready_timeout_s,
                may_start=runtime_state.requests_allow_motion,
            )
        except RuntimeError as exc:
            print(f"  ERROR: VR receiver startup failed: {exc}", flush=True)
            return 1
        print("  VR connected", flush=True)

        print(
            f"  Waiting for VR tracking data (up to {cfg.tracking_data_timeout_s:g} s)...",
            flush=True,
        )
        if not wait_for_vr_tracking(channels, sensors, args.ref, cfg.tracking_data_timeout_s):
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
            time.sleep(COUNTDOWN_DWELL_S)

        deadline = time.monotonic() + args.duration
        last_print = 0.0
        prev_seq: int | None = None

        print(f"  Collecting {args.duration}s (hold still)...")
        while time.monotonic() < deadline:
            if not sensors.check():
                print("  ERROR: VR receiver exited during collection", flush=True)
                return 1
            sample = read_heading(channels, args.ref)
            if sample is None:
                time.sleep(POLL_INTERVAL_S)
                continue
            sequence, quaternion = sample
            # Ignore duplicate sequence numbers in the quality estimate.
            if sequence == prev_seq:
                stale_count += 1
                time.sleep(POLL_INTERVAL_S)
                continue
            prev_seq = sequence
            forwards.append(forward_from_quat_wxyz(quaternion))

            now = time.monotonic()
            if now - last_print >= 1.0:
                print(f"    collected {len(forwards)} frames...")
                last_print = now
            time.sleep(POLL_INTERVAL_S)
    finally:
        shutdown_clean = shutdown_local_runtime(
            runtime_state,
            sensors=sensors,
            timeout_s=JOIN_TIMEOUT_S,
        )

    if not shutdown_clean:
        print("ERROR: VR receiver cleanup failed; calibration was not published")
        return 1

    if len(forwards) < cfg.min_frames:
        print(f"ERROR: only {len(forwards)} frames collected (< {cfg.min_frames} required)")
        return 1

    forwards_arr = np.array(forwards, dtype=np.float64)
    try:
        theta_rad, mean_fwd, inlier = circular_mean(forwards_arr)
    except ValueError as exc:
        print(f"ERROR: invalid heading sample: {exc}")
        return 1
    inlier_frames = int(np.sum(inlier))
    if inlier_frames < cfg.min_frames:
        print(f"ERROR: only {inlier_frames} inlier frames remain (< {cfg.min_frames} required)")
        return 1
    theta_deg = float(np.rad2deg(theta_rad))
    calibration = parse_vr_transform({
        "theta_deg": theta_deg,
        "ref": args.ref,
        "quality": {
            **quality_measurements(forwards_arr, theta_rad, inlier),
            "frames": inlier_frames,
        },
    })
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
        f"{calibration.quality.grade} (σ={calibration.quality.std_deg:.1f}°, "
        f"max={calibration.quality.max_deviation_deg:.1f}°)"
    )
    print(f"  verification:  T·forward = [{corrected[0]:.4f}, {corrected[1]:.4f}] (expect [1, 0])")
    print(f"  T = R_z(-{theta_deg:.1f}°):")
    print(f"    [{T[0, 0]:.4f}, {T[0, 1]:.4f}, {T[0, 2]:.4f}],")
    print(f"    [{T[1, 0]:.4f}, {T[1, 1]:.4f}, {T[1, 2]:.4f}],")
    print(f"    [{T[2, 0]:.4f}, {T[2, 1]:.4f}, {T[2, 2]:.4f}]")
    print(f"{'=' * 55}")

    try:
        output_path = publish_vr_transform(calibration, args.output, force=args.force)
    except (ValueError, FileExistsError) as exc:
        print(f"\n  NOT written: {exc}")
        return 1

    print(f"\nSaved to: {output_path}")

    play_completion_audio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
