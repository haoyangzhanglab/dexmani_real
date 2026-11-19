"""键盘点动与 R 键回零，机械臂目标由 arm.home_qpos 配置。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dexmani_real.config.real import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None, help="Trusted Python experiment file exporting CONFIG: RealConfig"
    )
    parser.add_argument("--no-hand", action="store_true", help="不连接灵巧手，按 R 只回机械臂 HOME")
    parser.add_argument("--timing-output", type=Path, metavar="PATH", help="退出后保存遥操作时序诊断 JSON")
    args = parser.parse_args(argv)

    runtime = load_config(args.config)
    if args.no_hand:
        runtime = replace(runtime, hand=replace(runtime.hand, enabled=False))

    from dexmani_real.teleop.keyboard import KeyboardTeleop
    return KeyboardTeleop(runtime, timing_output=args.timing_output).run()


if __name__ == "__main__":
    raise SystemExit(main())
