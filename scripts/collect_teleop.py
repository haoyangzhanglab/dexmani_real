"""
    通过 VR 控制 xArm7/XHand，并将示教录制为 Raw episode。
    python scripts/collect_teleop.py pnp_apple_banana
"""

from __future__ import annotations

import argparse
from pprint import pprint

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
            "Task label and recording subdirectory <episodes-dir>/<task_name>/ (unused with --no-record); "
            f"default: {DEFAULT_TASK_NAME!r}."
        ),
    )
    parser.add_argument(
        "--record", action=argparse.BooleanOptionalAction, default=True,
        help="Enable/disable capture and camera",
    )
    parser.add_argument(
        "--episodes-dir", default="episodes",
        help="Capture root directory, relative to cwd; supports ~ (default: %(default)s)",
    )
    parser.add_argument(
        "--max-duration", type=float, default=60.0,
        help="Maximum duration of each capture in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Trusted Python experiment file exporting CONFIG: RealConfig"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print Real config and recording options without loading hardware or calibration",
    )
    parser.add_argument("--vr-transform", default=None, help="Explicit VR alignment JSON")
    parser.add_argument(
        "--camera-calibration", default=None, help="Explicit camera extrinsics JSON"
    )
    args = parser.parse_args(argv)

    runtime = load_config(args.config)
    if args.print_config:
        pprint({
            "runtime": runtime,
            "collection": {
                "enabled": args.record,
                "episodes_dir": args.episodes_dir,
                "max_duration_s": args.max_duration,
            },
        }, sort_dicts=False)
        return 0

    from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
    from dexmani_real.teleop.run import run_teleop

    return run_teleop(
        runtime,
        record=args.record,
        episodes_dir=args.episodes_dir,
        max_duration_s=args.max_duration,
        task_name=args.task_name,
        vr_transform_path=VR_TRANSFORM_PATH if args.vr_transform is None else args.vr_transform,
        camera_calibration_path=args.camera_calibration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
