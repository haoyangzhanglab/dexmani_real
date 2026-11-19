"""采集实时 VR 数据标定朝向，显示统计并保存对齐角度。"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from dexmani_real import ASSET_DIR
from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
from dexmani_real.config.real import load_config
from dexmani_real.sensor.vr import VRReceiveStatus, iter_vr_events
from dexmani_real.teleop.audio import play_wav
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


def tracking_status(status: VRReceiveStatus, reference: str) -> str:
    if status.lines_received == 0:
        return status.transport_message or "No telemetry received."
    if status.parse_errors == status.lines_received:
        return f"Received {status.lines_received} lines that could not be parsed as HTS telemetry."
    pose = "head pose" if reference == "head" else "right wrist pose"
    return f"Received HTS telemetry. Still waiting for a valid {pose}."


def wait_for_vr_tracking(
    events,
    reference: str,
    timeout_s: float,
    status: VRReceiveStatus,
) -> bool:
    started_at = time.monotonic()
    deadline = started_at + timeout_s
    last_print = started_at
    for event in events:
        now = time.monotonic()
        if now >= deadline:
            break
        if read_heading(event, reference) is not None:
            print("  VR tracking active\n", flush=True)
            return True
        if now - last_print >= PRINT_INTERVAL_S:
            elapsed = int(now - started_at)
            print(f"    {tracking_status(status, reference)} ({elapsed}s elapsed)", flush=True)
            last_print = now
    return False


def drain_events(events, duration_s):
    """摆位和倒计时期间仍读取网络，避免采样开始时消费积压旧帧。"""
    from hand_tracking_sdk import TransportDisconnectedError

    deadline = time.monotonic() + duration_s
    for event in events:
        if isinstance(event, TransportDisconnectedError):
            raise event
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
        "--port", type=int, default=None, help="Override CONFIG.vr.port"
    )
    parser.add_argument("--config", type=Path, help="Trusted Python experiment file exporting CONFIG: RealConfig")
    parser.add_argument("--print-config", action="store_true", help="Print declared VR config without connecting")
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

    vr_config = load_config(args.config).vr
    if args.port is not None:
        vr_config = replace(vr_config, port=args.port)
    if args.print_config:
        print(vr_config)
        return 0

    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a positive finite number of seconds")

    ref_label = (
        "head (face robot +X)" if args.ref == "head" else "wrist (extend arm, point at robot +X)"
    )

    print("=" * 55)
    print("VR Heading Calibration")
    print(f"  reference:  {ref_label}")
    print(f"  duration:   {args.duration}s")
    print(f"  transport:  {vr_config.transport} {vr_config.host}:{vr_config.port}")
    print("=" * 55)

    vr_config.validate()
    from hand_tracking_sdk import TransportDisconnectedError

    from dexmani_real.utils.log import configure_logging

    configure_logging()
    status = VRReceiveStatus()
    with closing(iter_vr_events(
        vr_config, source=args.ref, running=lambda: True, status=status,
    )) as events:
        print(
            f"\n  Waiting for VR tracking (up to {TRACKING_DATA_TIMEOUT_S:g} s).",
            flush=True,
        )
        if vr_config.transport == "tcp_server":
            print(
                f"  HTS TCP Wired uses ADB reverse on port {vr_config.port}. "
                f"For TCP Wireless, use the PC's reachable IP and port {vr_config.port}.",
                flush=True,
            )
        tracking = "enable Head Pose" if args.ref == "head" else "enable right-hand tracking"
        print(f"  Put on the Quest, {tracking} in Hand Tracking Streamer, and tap Start.", flush=True)
        if not wait_for_vr_tracking(events, args.ref, TRACKING_DATA_TIMEOUT_S, status):
            print(f"  ERROR: tracking wait timed out. {tracking_status(status, args.ref)}", file=sys.stderr)
            return 1

        try:
            print(f"  Settling ({SETTLE_S:.0f} s) — fine-tune your pose...", flush=True)
            drain_events(events, SETTLE_S)
            if args.ref == "head":
                print("  Face the robot +X direction, hold your head still...")
            else:
                print("  Extend your right arm, point fingers toward robot +X, hold steady...")
            for i in (3, 2, 1):
                print(f"  {i}...")
                drain_events(events, COUNTDOWN_DWELL_S)
            print(f"  Collecting {args.duration}s (hold still)...")
            forwards = collect_headings(events, args.ref, args.duration)
        except TransportDisconnectedError as exc:
            print(f"ERROR: VR disconnected during calibration. {exc}", file=sys.stderr)
            return 1

    try:
        calibration = estimate_heading(forwards)
    except ValueError as exc:
        print(f"ERROR: invalid heading sample: {exc}")
        return 1
    print(
        f"VR heading: {calibration.theta_deg:.2f}°, {calibration.frames} inliers, "
        f"{len(forwards) - calibration.frames} rejected"
    )
    print(
        f"Heading spread: σ={calibration.std_deg:.2f}°, "
        f"max={calibration.max_deviation_deg:.2f}°"
    )

    output_path = save_vr_transform(calibration, args.output or VR_TRANSFORM_PATH)
    print(f"\nSaved to: {output_path}", flush=True)
    play_wav(ASSET_DIR / "audio" / "轴向已标定.wav", timeout_s=5.0)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nVR heading calibration interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
