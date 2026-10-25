#!/usr/bin/env python3
"""通过 VR 控制 xArm7/XHand，并将示教录制为 Raw episode。"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import yaml

from dexmani_real.config import load_experiment_config
from dexmani_real.teleop.config import DEFAULT_TASK_NAME


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
            "Do not start XHand; it must be absent or secured at configured home. "
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
        "--config", type=str, default=None, help="Optional experiment YAML overrides"
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
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Teleop/action/recording frequency; xArm servo frequency is independent",
    )
    args = parser.parse_args(argv)

    runtime = load_experiment_config(
        yaml_path=args.config,
        cli_overrides={
            "collection.enabled": args.record,
            "collection.episodes_dir": args.episodes_dir,
            "collection.max_duration_s": args.max_duration,
            "teleop.control_hz": args.control_hz,
            "hand.enabled": False if args.no_hand else None,
        },
    )
    if args.print_config:
        runtime.collection.validate()
        sections = ["arm", "hand", "motion", "teleop", "collection", "vr", "safety", "environment"]
        if runtime.collection.enabled:
            sections.append("camera")
        if runtime.hand.enabled:
            sections.append(runtime.teleop.hand_retargeting_type + "_retargeting")
        print("# 声明配置；CLI 专属参数仍需在命令行指定。")
        for name in ("task_name", "vr_transform", "camera_calibration", "no_hand"):
            print(f"# {name}: {getattr(args, name)}")
        print(yaml.safe_dump(
            {name: asdict(getattr(runtime, name)) for name in sections},
            allow_unicode=True, sort_keys=False,
        ), end="")
        return 0
    if not runtime.hand.enabled and not args.no_hand:
        parser.error("hand.enabled=false requires explicit --no-hand confirmation")

    from dexmani_real.state import VR_TRANSFORM_PATH
    from dexmani_real.teleop.session import TeleopSession

    return TeleopSession(
        runtime,
        task_name=args.task_name,
        allow_no_hand=args.no_hand,
        vr_transform_path=args.vr_transform or VR_TRANSFORM_PATH,
        camera_calibration_path=args.camera_calibration,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
