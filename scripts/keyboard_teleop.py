#!/usr/bin/env python3
"""通过键盘控制 xArm7/XHand；--no-hand 保留固定手姿态的几何检查。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dexmani_real.config.real import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for xArm7")
    parser.add_argument(
        "--config", type=Path, default=None, help="Trusted Python experiment file exporting CONFIG: RealConfig"
    )
    parser.add_argument("--no-hand", action="store_true", help="不连接灵巧手；几何检查仍使用配置的固定手姿态")
    args = parser.parse_args(argv)

    runtime = load_config(args.config)
    if args.no_hand:
        runtime = replace(runtime, hand=replace(runtime.hand, enabled=False))

    from dexmani_real.teleop.keyboard import KeyboardTeleop
    return KeyboardTeleop(runtime).run()


if __name__ == "__main__":
    raise SystemExit(main())
