"""从明确的官方 revision 下载分割与单图建模权重；不加载模型或连接设备。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..io import existing_file, output_path
from ..models import (
    DEFAULT_MODELS_DIR,
    GROUNDING_FILES,
    GROUNDING_MODEL,
    SAM_MODELS,
    SHAPE_MODEL,
)


def build_parser(*, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("grounding", "sam", "shape"),
        default=["grounding", "sam", "shape"],
    )
    parser.add_argument(
        "--models-dir", type=Path, default=workspace / DEFAULT_MODELS_DIR
    )
    parser.add_argument(
        "--sam-variant", choices=SAM_MODELS, default="small", help="SAM2.1 权重型号"
    )
    return parser


def main(argv=None, *, workspace=None):
    parser = build_parser(workspace=workspace)
    args = parser.parse_args(argv)
    output = output_path(args.models_dir, allow_existing=True)
    output.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        spec: dict[str, Any]
        if name == "sam":
            model = SAM_MODELS[args.sam_variant]
            spec = dict(**model, directory=".", patterns=[model["weight"]])
        else:
            spec = {"grounding": GROUNDING_MODEL, "shape": SHAPE_MODEL}[name]
        destination = output / spec["directory"]
        if name == "grounding":
            required = [destination / filename for filename in GROUNDING_FILES]
        else:
            required = [destination / spec["weight"]]
            if name == "shape":
                required.append(destination / "hunyuan3d-dit-v2-1/config.yaml")
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            print(f"复用本地模型，跳过下载：{destination / spec['weight']}", flush=True)
            continue
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
        for path in required:
            existing_file(path)
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
