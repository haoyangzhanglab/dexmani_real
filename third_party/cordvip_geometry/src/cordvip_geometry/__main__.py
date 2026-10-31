"""物体资产准备、Raw 离线追踪和结果查看。"""

import argparse
from importlib import import_module
from pathlib import Path
import sys

COMMANDS = {
    "object": ("build_object", "分割、单图生成和资产准备"),
    "prepare": ("prepare_object", "将已有 mesh 校准为米制资产"),
    "track": ("scene_tracking", "确认首帧实例后顺序处理 Raw 全部行"),
    "view": ("object_tracking_demo", "查看已保存的离线场景结果，不加载模型"),
    "check": ("check_tracking", "静态检查追踪依赖，不连接设备"),
    "download": ("download_models", "从明确的官方 revision 下载模型权重"),
}


def main(argv=None, *, workspace=None):
    parser = argparse.ArgumentParser(
        prog="python -m cordvip_geometry" if Path(sys.argv[0]).name == "__main__.py" else None,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {name:10} {spec[1]}" for name, spec in COMMANDS.items())
        + "\n\n使用 <command> --help 查看对应参数。",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    module, _ = COMMANDS[args.command]
    # Load only the selected workflow; root help imports no optional dependencies.
    entry = import_module(f".cli.{module}", package="cordvip_geometry").main
    original_program = sys.argv[0]
    try:
        sys.argv[0] = f"{parser.prog} {args.command}"
        return entry(args.args, workspace=workspace)
    finally:
        sys.argv[0] = original_program


if __name__ == "__main__":
    raise SystemExit(main())
