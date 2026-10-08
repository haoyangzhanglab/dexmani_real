#!/usr/bin/env python3
"""通过 VR 控制 xArm7/XHand，并将示教录制为 Raw episode。

python examples/collect_teleop.py test
python examples/collect_teleop.py test --config local.yaml
python examples/collect_teleop.py --print-config

参数：TASK 指定任务和录制目录；--config 覆盖配置，--print-config 仅打印配置、不连接设备。
"""

from __future__ import annotations

import argparse

import yaml

from dexmani_real.config.experiment import config_as_dict, load_experiment_config
from dexmani_real.teleop.config import (
    DEFAULT_TASK_NAME,
    validate_task_dir_name,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def _task_name_arg(value: str) -> str:
    try:
        return validate_task_dir_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VR Teleop xArm7 + XHand with recording")
    parser.add_argument(
        "task_name",
        metavar="TASK",
        nargs="?",
        type=_task_name_arg,
        default=DEFAULT_TASK_NAME,
        help=(
            "Task name used for recording metadata and episodes/<task_name>/; "
            f"default: {DEFAULT_TASK_NAME!r}."
        ),
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not start XHand; it must be absent or secured at configured home. Arm debug only; also pass --no-record (or disable recording in config).",
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
    args = parser.parse_args(argv)

    try:
        runtime = load_experiment_config(
            yaml_path=args.config,
            cli_overrides={
                "policy.hand_enabled": False if args.no_hand else None,
                "policy.recording_enabled": False if args.no_record else None,
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
            yaml.safe_dump(config_as_dict(runtime), allow_unicode=True, sort_keys=True),
            end="",
        )
        return 0
    if not runtime.policy.hand_enabled and not args.no_hand:
        parser.error("policy.hand_enabled=false requires explicit --no-hand confirmation")

    try:
        from dexmani_real.calibration import VR_TRANSFORM_PATH
        from dexmani_real.teleop.session import run_teleop_experiment

        return run_teleop_experiment(
            runtime,
            task_name=args.task_name,
            allow_no_hand=args.no_hand,
            vr_transform_path=args.vr_transform or VR_TRANSFORM_PATH,
            camera_calibration_path=args.camera_calibration,
        )
    except Exception:
        logger.error(
            "teleoperation startup failed before lifecycle ownership was established",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
