#!/usr/bin/env python3
"""显式检查离线几何的本地源码、解释器和权重路径；不导入模型。"""

import argparse
import json
from pathlib import Path

from ..io import existing_file
from ..sam2_config import (
    add_sam_arguments,
    check_sam_environment,
    resolve_sam_arguments,
)

STATIC_CHECK_LIMITATION = "路径检查不证明 CUDA、扩展 ABI 或模型推理可用；不连接设备。"


def check_tracking_environment(root):
    root = Path(root).expanduser().resolve()
    checks = []
    required = ["estimater.py", "Utils.py"]
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
        required.extend((f"weights/{run}/config.yml", f"weights/{run}/model_best.pth"))
    for relative in required:
        try:
            path = existing_file(root / relative)
            if path.suffix == ".pth":
                with path.open("rb") as stream:
                    if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                        raise ValueError(f"只有 Git LFS 指针，尚未下载权重：{path}")
            checks.append(dict(name=relative, ok=True, detail=str(path)))
        except (OSError, ValueError) as error:
            checks.append(dict(name=relative, ok=False, detail=str(error)))
    extensions = list((root / "mycpp/build").glob("mycpp*.so"))
    checks.append(dict(
        name="FoundationPose mycpp", ok=any(path.is_file() and path.stat().st_size for path in extensions),
        detail=str(root / "mycpp/build/mycpp*.so"),
    ))
    return checks


def main(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-root", type=Path, default=workspace / "third_party/FoundationPose")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出简短的路径检查结果")
    add_sam_arguments(parser, workspace)
    args = resolve_sam_arguments(parser.parse_args(argv), workspace)
    checks = check_tracking_environment(args.foundationpose_root) + check_sam_environment(args)
    report = dict(ok=all(item["ok"] for item in checks), checks=checks, limitation=STATIC_CHECK_LIMITATION)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            print(f"[{'ok' if item['ok'] else '缺失'}] {item['name']}：{item['detail']}")
        print(STATIC_CHECK_LIMITATION)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
