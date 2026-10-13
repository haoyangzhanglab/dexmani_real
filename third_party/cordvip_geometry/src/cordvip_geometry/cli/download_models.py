#!/usr/bin/env python3
"""显式下载单图示例的固定版本权重；不加载模型或连接设备。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import _sha256


MODELS = {
    "grounding": {
        "repo_id": "IDEA-Research/grounding-dino-base",
        "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "directory": "grounding-dino-base",
        "patterns": ["*.json", "model.safetensors", "vocab.txt", "README.md", "LICENSE*"],
        "weight": "model.safetensors",
        "size": 933400872,
        "sha256": "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21",
    },
    "sam": {
        "repo_id": "facebook/sam2.1-hiera-large",
        "revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
        "directory": ".",
        "patterns": ["sam2.1_hiera_large.pt"],
        "weight": "sam2.1_hiera_large.pt",
        "size": 898083611,
        "sha256": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    },
    "shape": {
        "repo_id": "tencent/Hunyuan3D-2.1",
        "revision": "0b94677654c57bb9a6b6845cd7b704ccf551d327",
        "directory": "Hunyuan3D-2.1",
        "patterns": ["hunyuan3d-dit-v2-1/config.yaml", "hunyuan3d-dit-v2-1/model.fp16.ckpt", "README.md", "LICENSE*"],
        "weight": "hunyuan3d-dit-v2-1/model.fp16.ckpt",
        "size": 7366389768,
        "sha256": "6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e",
    },
}


def build_parser(*, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output", type=Path, default=workspace / "reports/cordvip_models")
    parser.add_argument("--verify-only", action="store_true", help="只校验已下载权重并保存版本记录，不联网")
    return parser


def main(argv=None, *, workspace=None):
    parser = build_parser(workspace=workspace)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if not output.is_relative_to((args.workspace / "reports").resolve()):
        parser.error("权重目录必须位于 reports/ 下")
    output.mkdir(parents=True, exist_ok=True)
    # 固定官方 Hub revision 和 LFS SHA；每个完整模型成功后才写版本记录。
    for name in args.models:
        spec = MODELS[name]
        destination = output / spec["directory"]
        if not args.verify_only:
            from huggingface_hub import snapshot_download

            print(f"Downloading {spec['repo_id']} @ {spec['revision']}", flush=True)
            snapshot_download(
                repo_id=spec["repo_id"], revision=spec["revision"],
                allow_patterns=spec["patterns"], local_dir=destination,
                cache_dir=output / ".hub_cache", max_workers=3,
            )
        weight = destination / spec["weight"]
        if any(Path(str(weight) + suffix).exists() for suffix in (".aria2", ".part", ".incomplete")):
            raise RuntimeError(f"权重仍在下载：{weight}")
        if not weight.is_file() or weight.stat().st_size != spec["size"]:
            raise RuntimeError(f"权重缺失或大小不符：{weight}")
        if _sha256(weight) != spec["sha256"]:
            raise RuntimeError(f"权重 SHA256 与官方版本不符：{weight}")
        files = [str(p.relative_to(destination)) for pattern in spec["patterns"]
                 for p in destination.glob(pattern) if p.is_file()]
        record = {**spec, "files": sorted(set(files))}
        target = output / f"{name}_download.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(target)
        print(f"Ready: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
