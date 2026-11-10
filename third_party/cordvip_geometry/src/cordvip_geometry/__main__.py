"""物体资产准备、模型下载和依赖检查。"""

import argparse
import sys
from importlib import import_module
from pathlib import Path

COMMANDS = {
    "object": "分割、单图生成和资产准备",
    "check": "静态检查追踪依赖，不连接设备",
    "download": "从明确的官方 revision 下载模型权重",
}


def main(argv=None, *, workspace=None):
    parser = argparse.ArgumentParser(
        prog="python -m cordvip_geometry"
        if Path(sys.argv[0]).name == "__main__.py"
        else None,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name:10} {description}" for name, description in COMMANDS.items()
        )
        + "\n\n使用 <command> --help 查看对应参数。",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Load only the selected workflow; root help imports no optional dependencies.
    entry = import_module(f".cli.{args.command}", package="cordvip_geometry").main
    original_program = sys.argv[0]
    try:
        sys.argv[0] = f"{parser.prog} {args.command}"
        return entry(args.args, workspace=workspace)
    finally:
        sys.argv[0] = original_program


if __name__ == "__main__":
    raise SystemExit(main())
