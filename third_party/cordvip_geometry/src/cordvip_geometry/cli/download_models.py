#!/usr/bin/env python3
"""下载或校验分割与单图建模使用的固定版本权重；不加载模型或连接设备。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import sha256
from ..sam2_config import GROUNDING_MODEL, SAM_MODELS, validate_grounding_weights

MODELS = {
    "grounding": GROUNDING_MODEL,
    "sam": None,  # Resolved from the explicit SAM2.1 variant below.
    "shape": {
        "repo_id": "tencent/Hunyuan3D-2.1",
        "revision": "0b94677654c57bb9a6b6845cd7b704ccf551d327",
        "directory": "Hunyuan3D-2.1",
        "patterns": [
            "hunyuan3d-dit-v2-1/config.yaml",
            "hunyuan3d-dit-v2-1/model.fp16.ckpt",
            "README.md",
            "LICENSE*",
        ],
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
    parser.add_argument(
        "--sam-variant", choices=SAM_MODELS, default="small", help="SAM2.1 权重型号"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="只校验已下载权重并保存版本记录，不联网"
    )
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
        if name == "sam":
            model = SAM_MODELS[args.sam_variant]
            spec = dict(**model, directory=".", patterns=[model["weight"]])
        else:
            spec = MODELS[name]
        destination = output / spec["directory"]
        if not args.verify_only:
            from huggingface_hub import snapshot_download

            print(f"Downloading {spec['repo_id']} @ {spec['revision']}", flush=True)
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                allow_patterns=spec["patterns"],
                local_dir=destination,
                cache_dir=output / ".hub_cache",
                max_workers=3,
            )
        weight = destination / spec["weight"]
        if any(
            Path(str(weight) + suffix).exists() for suffix in (".aria2", ".part", ".incomplete")
        ):
            raise RuntimeError(f"权重仍在下载：{weight}")
        if not weight.is_file() or weight.stat().st_size != spec["size"]:
            raise RuntimeError(f"权重缺失或大小不符：{weight}")
        if sha256(weight) != spec["sha256"]:
            raise RuntimeError(f"权重 SHA256 与官方版本不符：{weight}")
        if name == "grounding":
            validate_grounding_weights(destination, verify_hash=False)
        files = [
            str(p.relative_to(destination))
            for pattern in spec["patterns"]
            for p in destination.glob(pattern)
            if p.is_file()
        ]
        record = {
            **spec,
            "files": sorted(set(files)),
            "download_source": "verification_only" if args.verify_only else "huggingface",
        }
        record_name = f"sam2_{args.sam_variant}" if name == "sam" else name
        target = output / f"{record_name}_download.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(target)
        print(f"Ready: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
