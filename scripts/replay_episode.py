#!/usr/bin/env python3
"""顺序播放绝对关节 demo；S/Q 停止退出，Esc 急停。"""

from __future__ import annotations

import argparse
import logging

from dexmani_real.config.real import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="包含绝对关节目标 data.h5 的目录")
    parser.add_argument("--config", help="导出 CONFIG 的可信 Python 配置")
    parser.add_argument("--segment", type=int, help="连续区间编号，从 0 开始；多区间 Raw 必须指定。按标称频率回放。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from dexmani_real.replay.run import run_replay
    from dexmani_real.utils.log import configure_logging

    configure_logging()
    try:
        run_replay(args.episode, load_config(args.config), segment=args.segment)
    except (Exception, KeyboardInterrupt):
        logging.getLogger(__name__).exception("Replay failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
