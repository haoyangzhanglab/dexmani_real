#!/usr/bin/env python3
"""通过键盘控制 xArm7/XHand，支持回零与点动。"""

from __future__ import annotations

import argparse
from pathlib import Path

from dexmani_real.config.real import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for xArm7")
    parser.add_argument(
        "--config", type=Path, default=None, help="Trusted Python experiment file exporting CONFIG: RealConfig"
    )
    args = parser.parse_args(argv)

    runtime = load_config(args.config)

    from dexmani_real.teleop.keyboard import KeyboardTeleop
    return KeyboardTeleop(runtime).run()


if __name__ == "__main__":
    raise SystemExit(main())
