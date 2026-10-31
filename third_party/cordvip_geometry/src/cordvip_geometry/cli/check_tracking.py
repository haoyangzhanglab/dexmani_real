#!/usr/bin/env python3
"""显式静态检查离线追踪的本地依赖；不导入模型。"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

STATIC_CHECK_LIMITATION = "静态检查不证明 CUDA 可用、扩展 ABI 兼容或推理成功；未枚举或连接相机。"


def check_tracking_environment(root, *, need_gui=True, need_pose=True) -> dict:
    """Inspect top-level module specs only; importing dotted parents could initialize CUDA or SDKs."""
    root = Path(root).expanduser().resolve()
    checks = []

    required_files = (
        "estimater.py",
        "Utils.py",
        "datareader.py",
        "learning/training/predict_pose_refine.py",
        "learning/training/predict_score.py",
        "learning/models/refine_network.py",
        "learning/models/score_network.py",
        "learning/models/network_modules.py",
        "learning/datasets/h5_dataset.py",
        "learning/datasets/pose_dataset.py",
        "weights/2023-10-28-18-33-37/config.yml",
        "weights/2023-10-28-18-33-37/model_best.pth",
        "weights/2024-01-11-20-02-45/config.yml",
        "weights/2024-01-11-20-02-45/model_best.pth",
    )
    for relative in required_files if need_pose else ():
        path = root / relative
        try:
            ok = path.is_file() and path.stat().st_size > 0
            detail = str(path) if ok else f"缺失或为空：{path}"
            if ok and path.suffix == ".pth":
                with path.open("rb") as stream:
                    is_lfs_pointer = stream.read(80).startswith(
                        b"version https://git-lfs.github.com/spec/"
                    )
                if is_lfs_pointer:
                    ok, detail = False, f"只有 Git LFS 指针，尚未下载权重：{path}"
        except OSError as error:
            ok, detail = False, f"无法读取 {path}：{error}"
        checks.append({"name": f"文件 {relative}", "ok": ok, "detail": detail})

    modules = ["numpy", "cv2", "PIL", "h5py", "yaml", "tqdm"]
    if need_pose:
        modules.extend(
            [
                "torch",
                "torchvision",
                "scipy",
                "trimesh",
                "pytorch3d",
                "nvdiffrast",
                "kornia",
                "warp",
                "ruamel",
                "omegaconf",
                "transformations",
                "imageio",
                "pandas",
                "joblib",
                "psutil",
                "matplotlib",
            ]
        )
    if need_pose:
        modules.append("open3d")
    for name in modules:
        try:
            spec = importlib.util.find_spec(name)
            ok = spec is not None
            detail = (spec.origin or "namespace package") if spec is not None else "当前 Python 环境找不到该模块"
        except (ImportError, ValueError, OSError) as error:
            ok, detail = False, str(error)
        if ok and name in {"torch", "torchvision", "open3d"}:
            try:
                detail += f"；version={importlib.metadata.version(name)}"
            except importlib.metadata.PackageNotFoundError:
                pass
        checks.append({"name": f"模块 {name}", "ok": ok, "detail": detail})

    if need_pose:
        # Check cluster_poses now; upstream hides its import failure until estimator construction.
        try:
            candidates = [
                path
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
                for path in (root / "mycpp" / "build").glob(f"mycpp*{suffix}")
                if path.is_file() and path.stat().st_size > 0
            ]
            if not candidates:
                spec = importlib.util.find_spec("mycpp")
                if (
                    spec is not None
                    and spec.origin
                    and any(
                        spec.origin.endswith(suffix)
                        for suffix in importlib.machinery.EXTENSION_SUFFIXES
                    )
                ):
                    path = Path(spec.origin)
                    if path.is_file() and path.stat().st_size > 0:
                        candidates.append(path)
            detail = (
                str(candidates[0])
                if candidates
                else f"缺少编译扩展：{root / 'mycpp/build/mycpp*.so'}"
            )
            ok = bool(candidates)
        except (ImportError, ValueError, OSError) as error:
            ok, detail = False, str(error)
        checks.append({"name": "mycpp 编译扩展", "ok": ok, "detail": detail})

    if need_gui:
        available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        checks.append(
            {
                "name": "图形会话",
                "ok": available,
                "detail": (
                    "发现 DISPLAY/WAYLAND_DISPLAY；窗口是否可用仍需实际启动验证"
                    if available
                    else "没有 DISPLAY/WAYLAND_DISPLAY；请在桌面会话启动，离线检查可用 --headless"
                ),
            }
        )
    return {"ok": all(item["ok"] for item in checks), "checks": checks}


def print_tracking_report(report, *, as_json=False) -> None:
    if as_json:
        print(
            json.dumps(
                {**report, "limitation": STATIC_CHECK_LIMITATION}, ensure_ascii=False, indent=2
            )
        )
        return
    failed = [item for item in report["checks"] if not item["ok"]]
    print(f"追踪环境静态检查：{len(report['checks']) - len(failed)}/{len(report['checks'])} 项通过")
    for item in failed:
        print(f"[缺失] {item['name']}：{item['detail']}")
    print(STATIC_CHECK_LIMITATION)
    if failed:
        print("请在当前示例使用的环境中补齐以上项目；本命令不安装依赖或下载权重。")


def parse_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=workspace / "third_party/FoundationPose",
        help="本地 FoundationPose 源码及 weights 目录",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不要求图形会话；仍检查 FoundationPose 使用的 Open3D",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出全部检查项目")
    from ..sam2_config import add_sam_arguments, resolve_sam_arguments

    add_sam_arguments(parser, workspace)
    return resolve_sam_arguments(parser.parse_args(argv), workspace)


def main(argv=None, *, workspace=None) -> int:
    args = parse_args(argv, workspace=workspace)
    report = check_tracking_environment(
        args.foundationpose_root, need_gui=not args.headless
    )
    from ..sam2_config import check_sam_environment

    report["checks"].extend(check_sam_environment(args))
    report["ok"] = all(item["ok"] for item in report["checks"])
    print_tracking_report(report, as_json=args.json)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
