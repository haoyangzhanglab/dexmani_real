#!/usr/bin/env python3
"""通过 VR 控制 xArm7/XHand，并将示教录制为 Raw episode。"""

from __future__ import annotations

import logging
import argparse
from dataclasses import asdict

import yaml

from dexmani_real.config.experiment import load_experiment_config
from dexmani_real.config.experiment import DEFAULT_TASK_NAME

logger = logging.getLogger("dexmani_real.cli.collect_teleop")


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
        "--no-record",
        action="store_true",
        help="Run VR teleoperation without recording; the camera worker and episode writer are not started.",
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

    try:
        runtime = load_experiment_config(
            yaml_path=args.config,
            cli_overrides={
                "teleop.control_hz": args.control_hz,
                "control.hand_enabled": False if args.no_hand else None,
                "control.recording_enabled": False if args.no_record else None,
            },
        )
    except (
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        parser.error(f"invalid experiment config: {exc}")
    if args.print_config:
        print(
            yaml.safe_dump(asdict(runtime), allow_unicode=True, sort_keys=True),
            end="",
        )
        return 0
    if not runtime.control.hand_enabled and not args.no_hand:
        parser.error("control.hand_enabled=false requires explicit --no-hand confirmation")

    try:
        from dexmani_real.calibration import VR_TRANSFORM_PATH
        from dexmani_real.teleop.session import TeleopSession

        return TeleopSession(
            runtime,
            task_name=args.task_name,
            allow_no_hand=args.no_hand,
            vr_transform_path=args.vr_transform or VR_TRANSFORM_PATH,
            camera_calibration_path=args.camera_calibration,
        ).run()
    except Exception:
        logger.error("teleoperation failed", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
