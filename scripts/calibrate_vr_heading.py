#!/usr/bin/env python3
"""采集实时 VR 数据标定朝向，通过质量检查后保存对齐变换。"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import closing
from pathlib import Path

from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
from dexmani_real.config.sensor import VRConfig
from dexmani_real.sensor.vr import iter_vr_events
from dexmani_real.utils.vr_heading import (
    collect_headings,
    estimate_heading,
    read_heading,
    save_vr_transform,
)

PRINT_INTERVAL_S = 5.0
COUNTDOWN_DWELL_S = 1.0
DEFAULT_DURATION_S = 10.0
TRACKING_DATA_TIMEOUT_S = 120.0
SETTLE_S = 3.0


def wait_for_vr_tracking(
    events,
    reference: str,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    for event in events:
        if time.monotonic() >= deadline:
            break
        if read_heading(event, reference) is not None:
            print("  VR tracking active\n", flush=True)
            return True
        now = time.monotonic()
        if now - last_print >= PRINT_INTERVAL_S:
            elapsed = int(now - (deadline - timeout_s))
            print(f"    waiting... ({elapsed}s elapsed)", flush=True)
            last_print = now
    return False


def drain_events(events, duration_s):
    """摆位和倒计时期间仍读取网络，避免采样开始时消费积压旧帧。"""
    deadline = time.monotonic() + duration_s
    for _ in events:
        if time.monotonic() >= deadline:
            return
    raise RuntimeError("VR receiver stopped before collection")


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
    with closing(iter_vr_events(
        vr_config, running=lambda: True, include_head_frames=args.ref == "head",
    )) as events:
        print(
            f"\n  Waiting for VR tracking (up to {TRACKING_DATA_TIMEOUT_S:g} s) "
            "— put on Quest headset...",
            flush=True,
        )
        if not wait_for_vr_tracking(events, args.ref, TRACKING_DATA_TIMEOUT_S):
            print("  ERROR: no VR tracking data received", flush=True)
            return 1

        print(f"  Settling ({SETTLE_S:.0f} s) — fine-tune your pose...", flush=True)
        drain_events(events, SETTLE_S)

        if args.ref == "head":
            print("  Face the robot +X direction, hold your head still...")
        else:
            print("  Extend your right arm, point fingers toward robot +X, hold steady...")
        for i in [3, 2, 1]:
            print(f"  {i}...")
            drain_events(events, COUNTDOWN_DWELL_S)

        print(f"  Collecting {args.duration}s (hold still)...")
        forwards, stale_count = collect_headings(events, args.ref, args.duration)

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
