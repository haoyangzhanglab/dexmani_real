#!/usr/bin/env python3
"""通过 VR 控制 xArm7/XHand，并将示教录制为 Raw episode。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from dexmani_real.config.real import load_config
from dexmani_real.config.teleop import DEFAULT_TASK_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VR Teleop xArm7 + XHand with recording")
    parser.add_argument(
        "task_name",
        metavar="TASK",
        nargs="?",
        default=DEFAULT_TASK_NAME,
        help=(
            "Task name used for recording metadata and episodes/<task_name>/; "
            f"default: {DEFAULT_TASK_NAME!r}."
        ),
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help=(
            "Confirm hand.enabled=False in the experiment config; hand must be absent or secured at home. "
            "Arm debug only; also pass --no-record (or disable recording in config)."
        ),
    )
    parser.add_argument(
        "--record", action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable capture and camera; overrides collection.enabled",
    )
    parser.add_argument("--episodes-dir", default=None, help="Capture root directory")
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="Maximum duration of each capture in seconds",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Trusted Python experiment file exporting CONFIG: RealConfig"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print declared Real config without loading hardware or calibration",
    )
    parser.add_argument("--vr-transform", default=None, help="Explicit VR alignment JSON")
    parser.add_argument(
        "--camera-calibration", default=None, help="Explicit camera extrinsics JSON"
    )
    args = parser.parse_args(argv)

    runtime = load_config(args.config)
    runtime = replace(runtime, collection=replace(
        runtime.collection,
        enabled=runtime.collection.enabled if args.record is None else args.record,
        episodes_dir=runtime.collection.episodes_dir if args.episodes_dir is None else args.episodes_dir,
        max_duration_s=runtime.collection.max_duration_s if args.max_duration is None else args.max_duration,
    ))
    if args.no_hand and runtime.hand.enabled:
        parser.error("--no-hand requires hand.enabled=False in the experiment config")
    if args.print_config:
        print(json.dumps(asdict(runtime), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if not runtime.hand.enabled and not args.no_hand:
        parser.error("hand.enabled=False requires explicit --no-hand confirmation")

    from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
    from dexmani_real.teleop.run import run_teleop

    return run_teleop(
        runtime,
        task_name=args.task_name,
        allow_no_hand=args.no_hand,
        vr_transform_path=args.vr_transform or VR_TRANSFORM_PATH,
        camera_calibration_path=args.camera_calibration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
