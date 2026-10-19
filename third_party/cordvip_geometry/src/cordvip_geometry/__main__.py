"""物体感知工具：拍照、资产准备、追踪和离线回放。"""

import argparse
from importlib import import_module
from pathlib import Path
import sys

COMMANDS = {
    "capture": ("capture_object", "冻结并保存 RGB-D；默认连接相机"),
    "object": ("build_object", "分割、单图生成和资产准备"),
    "prepare": ("prepare_object", "将已有 mesh 校准为米制资产"),
    "track": ("scene_tracking", "实时位姿与点云；默认连接相机"),
    "replay": ("object_tracking_demo", "只读 Raw 或 RGB-D 离线回放"),
    "check": ("check_tracking", "静态检查追踪依赖，不连接设备"),
    "download": ("download_models", "下载或校验固定版本的模型权重"),
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
