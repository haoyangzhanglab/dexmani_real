#!/usr/bin/env python3
"""采集实时 VR 数据标定朝向，通过质量检查后保存对齐变换。"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from pathlib import Path

from dexmani_real.calibration import VR_TRANSFORM_PATH
from dexmani_real.calibration.vr_heading import (
    collect_headings,
    estimate_heading,
    read_heading,
    save_vr_transform,
)
from dexmani_real.ipc.sensors import SensorChannels
from dexmani_real.runtime.lifecycle import (
    VR_READY_TIMEOUT_S,
    SensorWorkers,
    close_runtime,
)
from dexmani_real.runtime.state import SessionState
from dexmani_real.sensor.config import VRConfig
from dexmani_real.sensor.vr import run_vr
from dexmani_real.teleop.audio import AudioFeedback

PRINT_INTERVAL_S = 5.0
JOIN_TIMEOUT_S = 5.0
COUNTDOWN_DWELL_S = 1.0
AUDIO_IDLE_TIMEOUT_S = 5.0
DEFAULT_DURATION_S = 10.0
TRACKING_DATA_TIMEOUT_S = 15.0
SETTLE_S = 3.0


def wait_for_vr_tracking(
    channels: SensorChannels,
    sensors: SensorWorkers,
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
    audio = AudioFeedback()
    try:
        audio.play("calibrated")
        if not audio.wait_until_idle(timeout_s=AUDIO_IDLE_TIMEOUT_S):
            print("  WARNING: audio feedback timed out")
    finally:
        audio.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VR heading calibration")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"collection duration (s), default: {DEFAULT_DURATION_S}",
    )
    parser.add_argument(
        "--port", type=int, default=VRConfig.port, help=f"VR TCP port, default: {VRConfig.port}"
    )
    parser.add_argument(
        "--ref",
        choices=["wrist", "head"],
        default="head",
        help="reference: head=face robot +X, wrist=extend arm pointing at robot +X",
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

    vr_config = VRConfig(port=args.port)
    vr_config.validate()
    from dexmani_real.utils.log import configure_logging

    configure_logging()
    ctx = mp.get_context("spawn")
    runtime_state = SessionState()
    channels = SensorChannels(ctx, vr=True)
    sensors = SensorWorkers(channels)
    vr_proc = ctx.Process(
        name="vr", target=run_vr, args=(channels, vr_config), daemon=True
    )
    try:
        print(
            f"\n  Waiting for VR connection (up to {VR_READY_TIMEOUT_S:g} s) "
            "— put on Quest headset...",
            flush=True,
        )
        try:
            sensors.start(
                vr_proc,
                ready=channels.vr_ready,
                timeout_s=VR_READY_TIMEOUT_S,
                may_start=runtime_state.requests_allow_motion,
            )
        except RuntimeError as exc:
            print(f"  ERROR: VR receiver startup failed: {exc}", flush=True)
            return 1
        print("  VR connected", flush=True)

        print(
            f"  Waiting for VR tracking data (up to {TRACKING_DATA_TIMEOUT_S:g} s)...",
            flush=True,
        )
        if not wait_for_vr_tracking(channels, sensors, args.ref, TRACKING_DATA_TIMEOUT_S):
            print("  ERROR: no VR tracking data received", flush=True)
            return 1

        print(f"  Settling ({SETTLE_S:.0f} s) — fine-tune your pose...", flush=True)
        time.sleep(SETTLE_S)

        if args.ref == "head":
            print("  Face the robot +X direction, hold your head still...")
        else:
            print("  Extend your right arm, point fingers toward robot +X, hold steady...")
        for i in [3, 2, 1]:
            print(f"  {i}...")
            time.sleep(COUNTDOWN_DWELL_S)

        print(f"  Collecting {args.duration}s (hold still)...")
        forwards, stale_count = collect_headings(channels, sensors, args.ref, args.duration)
    finally:
        shutdown_clean = close_runtime(
            runtime_state,
            sensors=sensors,
            deadline=time.monotonic() + JOIN_TIMEOUT_S,
        )

    if not shutdown_clean:
        print("ERROR: VR receiver cleanup failed; calibration was not published")
        return 1

    try:
        calibration = estimate_heading(forwards, args.ref)
    except ValueError as exc:
        print(f"ERROR: invalid heading sample: {exc}")
        return 1
    print(
        f"VR heading: {calibration.theta_deg:.2f}°, {calibration.frames} inliers, "
        f"{len(forwards) - calibration.frames} rejected, {stale_count} duplicate frames"
    )
    print(
        f"Quality: {calibration.grade}, σ={calibration.std_deg:.2f}°, "
        f"max={calibration.max_deviation_deg:.2f}°"
    )

    try:
        output_path = save_vr_transform(calibration, args.output or VR_TRANSFORM_PATH)
    except ValueError as exc:
        print(f"\n  NOT written: {exc}")
        return 1

    print(f"\nSaved to: {output_path}")

    play_completion_audio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
