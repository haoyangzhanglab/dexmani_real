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
        help="Confirm hand.enabled=False in config; hand must be absent or secured at home",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Trusted Python experiment file exporting CONFIG: RealConfig"
    )
    args = parser.parse_args(argv)

    runtime = load_config(args.config)
    if args.no_hand and runtime.hand.enabled:
        parser.error("--no-hand requires hand.enabled=False in the experiment config")
    if not runtime.hand.enabled and not args.no_hand:
        parser.error("hand.enabled=False requires explicit --no-hand confirmation")

    from dexmani_real.teleop.keyboard import KeyboardTeleop
    return KeyboardTeleop(runtime, no_hand=args.no_hand).run()


if __name__ == "__main__":
    raise SystemExit(main())
