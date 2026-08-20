#!/usr/bin/env python3
"""CLI for a keyboard teleoperation session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.teleop.keyboard_session import run_keyboard_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for xArm7")
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not connect XHand; it must be absent or secured at configured home",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Validated experiment YAML"
    )
    args = parser.parse_args(argv)
    try:
        runtime = resolve_runtime_config(
            yaml_path=args.config,
            cli_overrides={"policy.hand_enabled": False if args.no_hand else None},
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
    return run_keyboard_experiment(runtime, no_hand=args.no_hand)


if __name__ == "__main__":
    raise SystemExit(main())
