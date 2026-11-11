"""为已有 Episode 目录标注人工任务结果。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dexmani_real.data.raw import EPISODE_OUTCOMES, write_episode_outcome


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="已有的 Episode 目录")
    parser.add_argument("outcome", choices=EPISODE_OUTCOMES)
    parser.add_argument("--note", help="可选任务结果备注")
    parser.add_argument("--overwrite", action="store_true", help="原子替换已有标注（不保留旧备注）")
    args = parser.parse_args(argv)
    path = write_episode_outcome(args.episode, args.outcome, note=args.note, overwrite=args.overwrite)
    print(f"Task outcome: {args.outcome} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
