#!/usr/bin/env python3
"""从明确的官方 revision 下载分割与单图建模权重；不加载模型或连接设备。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io import existing_file, output_path
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
    return parser


def main(argv=None, *, workspace=None):
    parser = build_parser(workspace=workspace)
    args = parser.parse_args(argv)
    output = output_path(args.output, workspace=args.workspace, allow_existing=True)
    output.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    for name in args.models:
        if name == "sam":
            model = SAM_MODELS[args.sam_variant]
            spec = dict(**model, directory=".", patterns=[model["weight"]])
        else:
            spec = MODELS[name]
        destination = output / spec["directory"]
        print(f"Downloading {spec['repo_id']} @ {spec['revision']}", flush=True)
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            allow_patterns=spec["patterns"],
            local_dir=destination,
            cache_dir=output / ".hub_cache",
            max_workers=3,
        )
        if name == "grounding":
            validate_grounding_weights(destination)
        else:
            existing_file(destination / spec["weight"])
        files = [
            str(p.relative_to(destination))
            for pattern in spec["patterns"]
            for p in destination.glob(pattern)
            if p.is_file()
        ]
        record = {
            **spec,
            "files": sorted(set(files)),
            "download_source": "huggingface",
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
