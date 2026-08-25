#!/usr/bin/env python3
"""Run one hardware-affecting VR teleoperation and collection session.

The session can command xArm7/XHand and, unless disabled, records a raw episode.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.teleop.session import (
    DEFAULT_TASK_NAME,
    run_teleop_experiment,
    validate_operator,
    validate_task_name,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def _task_name_arg(value: str) -> str:
    try:
        return validate_task_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_float(value: str) -> float:
    """Argparse type: positive finite float."""
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {value}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VR Teleop xArm7 + XHand with recording"
    )
    parser.add_argument(
        "--task-name",
        "--task",
        dest="task_name",
        type=_task_name_arg,
        default=DEFAULT_TASK_NAME,
        help=(
            "Task name used for recording metadata and episodes/<task_name>/; "
            f"default: {DEFAULT_TASK_NAME!r}. --task is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--operator", type=str, default="", help="Operator name for recording metadata"
    )
    parser.add_argument(
        "--acc",
        type=_positive_float,
        default=None,
        help="Joint max acceleration (°/s²; defaults to YAML/defaults)",
    )
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=None,
        help="Joint max speed (°/s; YAML/defaults if omitted)",
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not start XHand; use only when the physical hand is absent or secured.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Run VR teleoperation without recording; camera and RecorderIO are not started.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="YAML file with experiment overrides"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print all config values and exit"
    )
    args = parser.parse_args(argv)

    try:
        runtime = resolve_runtime_config(
            yaml_path=args.config,
            cli_overrides={
                "arm.max_joint_acceleration_deg_per_s2": args.acc,
                "arm.max_joint_velocity_deg_per_s": args.speed,
                "policy.hand_enabled": False if args.no_hand else None,
                "policy.recording_enabled": False if args.no_record else None,
            },
        )
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        parser.error(f"invalid experiment config: {exc}")
    if args.print_config:
        print(runtime.canonical_yaml, end="")
        print(f"sha256={runtime.sha256}")
        return 0
    if not bool(runtime.policy.hand_enabled) and not args.no_hand:
        parser.error(
            "policy.hand_enabled=false requires explicit --no-hand confirmation"
        )
    operator = args.operator
    if bool(runtime.policy.recording_enabled):
        try:
            operator = validate_operator(operator)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        return run_teleop_experiment(
            runtime,
            task_name=args.task_name,
            operator=operator,
            allow_no_hand=args.no_hand,
        )
    except Exception:
        logger.error(
            "teleoperation startup failed before lifecycle ownership was established",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
