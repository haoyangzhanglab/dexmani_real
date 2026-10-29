#!/usr/bin/env python3
"""通过键盘控制 xArm7/XHand，支持回零与点动。"""

from __future__ import annotations

import argparse
from pathlib import Path

from dexmani_real.config.real import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for xArm7")
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Do not connect XHand; it must be absent or secured at configured home",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional experiment YAML overrides"
    )
    parser.add_argument("--control-hz", type=float, default=None, help="Keyboard target frequency")
    args = parser.parse_args(argv)

    from dexmani_real.teleop.keyboard import KeyboardTeleop

    runtime = load_config(
        path=args.config,
        overrides={
            "hand": {"enabled": False if args.no_hand else None},
            "keyboard_teleop": {"control_hz": args.control_hz},
        },
    )
    return KeyboardTeleop(runtime, no_hand=args.no_hand).run()


if __name__ == "__main__":
    raise SystemExit(main())
