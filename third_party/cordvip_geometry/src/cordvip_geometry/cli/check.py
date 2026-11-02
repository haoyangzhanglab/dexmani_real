"""显式检查离线几何的本地源码、解释器和权重路径；不导入模型。"""

import argparse
import json
import os
from pathlib import Path

from ..io import existing_file
from ..models import GROUNDING_FILES, GROUNDING_MODEL
from .arguments import add_sam_arguments, resolve_sam_arguments

STATIC_CHECK_LIMITATION = "路径检查不证明 CUDA、扩展 ABI 或模型推理可用；不连接设备。"


def check_sam_environment(args):
    root = args.workspace / "third_party/Grounded-SAM-2"
    groups = (
        ("SAM Python", [args.sam_python]),
        (
            "Grounded-SAM2 source",
            [
                root / "sam2" / name
                for name in (
                    "build_sam.py",
                    "sam2_video_predictor.py",
                    "sam2_image_predictor.py",
                )
            ],
        ),
        (
            "SAM2 checkpoint/config",
            [root / "sam2" / args.sam_config, args.sam_checkpoint],
        ),
        (
            "Grounding DINO local files",
            [
                args.models_dir / GROUNDING_MODEL["directory"] / name
                for name in GROUNDING_FILES
            ],
        ),
    )
    checks = []
    for name, paths in groups:
        try:
            files = [existing_file(path) for path in paths]
            if name == "SAM Python" and not os.access(files[0], os.X_OK):
                raise ValueError(f"Python is not executable: {files[0]}")
            checks.append(
                {"name": name, "ok": True, "detail": ", ".join(map(str, files))}
            )
        except (OSError, ValueError) as error:
            checks.append({"name": name, "ok": False, "detail": str(error)})
    return checks


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
                    if stream.read(80).startswith(
                        b"version https://git-lfs.github.com/spec/"
                    ):
                        raise ValueError(f"只有 Git LFS 指针，尚未下载权重：{path}")
            checks.append({"name": relative, "ok": True, "detail": str(path)})
        except (OSError, ValueError) as error:
            checks.append({"name": relative, "ok": False, "detail": str(error)})
    extensions = list((root / "mycpp/build").glob("mycpp*.so"))
    checks.append(
        {
            "name": "FoundationPose mycpp",
            "ok": any(path.is_file() and path.stat().st_size for path in extensions),
            "detail": str(root / "mycpp/build/mycpp*.so"),
        }
    )
    return checks


def main(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument(
        "--json", action="store_true", help="以 JSON 输出简短的路径检查结果"
    )
    add_sam_arguments(parser, workspace)
    args = parser.parse_args(argv)
    try:
        args = resolve_sam_arguments(args, workspace)
    except ValueError as error:
        parser.error(str(error))
    checks = check_tracking_environment(
        workspace / "third_party/FoundationPose"
    ) + check_sam_environment(args)
    report = {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "limitation": STATIC_CHECK_LIMITATION,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            print(
                f"[{'ok' if item['ok'] else '缺失'}] {item['name']}：{item['detail']}"
            )
        print(STATIC_CHECK_LIMITATION)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
