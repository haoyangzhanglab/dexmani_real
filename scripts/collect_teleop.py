#!/usr/bin/env python3
"""通过 VR 控制 xArm7/XHand，并将示教录制为 Raw episode。"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import yaml

from dexmani_real.config.control import CollectionOptions
from dexmani_real.config.experiment import (
    DEFAULT_TASK_NAME, apply_config_overrides, load_experiment_config, read_config_yaml,
)


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
    parser.add_argument("--max-duration", type=float, default=None, help="Maximum duration of each capture in seconds")
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

    config_data = read_config_yaml(args.config)
    collection = apply_config_overrides(CollectionOptions(), config_data.pop("collection", {}))
    collection = apply_config_overrides(collection, {
        key: value for key, value in dict(
            enabled=args.record, episodes_dir=args.episodes_dir, max_duration_s=args.max_duration
        ).items() if value is not None
    })
    runtime = load_experiment_config(
        data=config_data,
        cli_overrides={
            "teleop.control_hz": args.control_hz,
            "hand.enabled": False if args.no_hand else None,
        },
    )
    if args.print_config:
        collection.validate()
        print(
            yaml.safe_dump(dict(asdict(runtime), collection=asdict(collection)), allow_unicode=True, sort_keys=False),
            end="",
        )
        return 0
    if not runtime.hand.enabled and not args.no_hand:
        parser.error("hand.enabled=false requires explicit --no-hand confirmation")

    from dexmani_real.calibration import VR_TRANSFORM_PATH
    from dexmani_real.teleop.session import TeleopSession

    return TeleopSession(
        runtime,
        collection=collection,
        task_name=args.task_name,
        allow_no_hand=args.no_hand,
        vr_transform_path=args.vr_transform or VR_TRANSFORM_PATH,
        camera_calibration_path=args.camera_calibration,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
