#!/usr/bin/env python3
"""相机标定：只读采样与离线求解分开执行。"""

from __future__ import annotations

import argparse
from pathlib import Path

from dexmani_real.config.calibration_data import CAMERAS_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="只读采集；先用厂商示教或独立工具摆位并退出控制连接")
    collect.add_argument("--config", type=Path, help="导出 CONFIG: RealConfig 的可信 Python 实验文件")
    collect.add_argument("--output", type=Path, required=True, help="尚不存在的采样目录")
    solve = commands.add_parser("solve", help="从已保存样本离线求解，不连接设备")
    solve.add_argument("samples", type=Path, help="collect 输出目录")
    solve.add_argument("--output", type=Path, default=CAMERAS_PATH, help="按 serial 保存外参的 JSON")
    solve.add_argument("--exclude", type=int, action="append", default=[], metavar="SAMPLE_ID",
                       help="剔除原始 sample ID，可重复指定不同 ID")
    args = parser.parse_args(argv)
    from dexmani_real.utils.camera_calibrate.calibrate import collect_samples, solve_samples

    if args.command == "collect":
        from dexmani_real.config.real import load_config
        from dexmani_real.utils.log import configure_logging

        configure_logging()
        collect_samples(load_config(args.config), args.output)
    else:
        solve_samples(args.samples, args.output, exclude=args.exclude)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
