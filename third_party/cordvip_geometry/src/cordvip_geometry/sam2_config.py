"""Local Grounded-SAM2 inputs; checks never import models or open devices."""

import json
from pathlib import Path
import subprocess
import sys

from .io import existing_file

SAM_MODELS = {
    "small": dict(
        config="configs/sam2.1/sam2.1_hiera_s.yaml",
        repo_id="facebook/sam2.1-hiera-small",
        revision="ee5bba1d82bb8749febdf90f45e84b687142ba03",
        weight="sam2.1_hiera_small.pt",
    ),
    "large": dict(
        config="configs/sam2.1/sam2.1_hiera_l.yaml",
        repo_id="facebook/sam2.1-hiera-large",
        revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
        weight="sam2.1_hiera_large.pt",
    ),
}
GROUNDING_MODEL = dict(
    repo_id="IDEA-Research/grounding-dino-base",
    revision="12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    directory="grounding-dino-base",
    weight="model.safetensors",
    patterns=["*.json", "model.safetensors", "vocab.txt", "README.md", "LICENSE*"],
)
GROUNDING_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


def add_sam_arguments(parser, workspace):
    parser.add_argument(
        "--sam-python",
        type=Path,
        default=Path(sys.executable),
        help="分割子进程解释器；默认当前解释器，需具备 SAM2 推理依赖",
    )
    parser.add_argument("--sam-root", type=Path, default=workspace / "third_party/Grounded-SAM-2")
    parser.add_argument(
        "--sam-variant", choices=SAM_MODELS,
        help="官方 SAM2.1 型号；省略时使用 small，同时选择匹配的配置与本地权重",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        help="自训模型本地权重；须与 --sam-config 完整配对，不能混用 --sam-variant",
    )
    parser.add_argument(
        "--sam-config",
        help="自训模型在 SAM2 包内的配置；须与 --sam-checkpoint 完整配对",
    )
    parser.add_argument(
        "--dino-weights",
        type=Path,
        default=workspace / "reports/cordvip_models" / GROUNDING_MODEL["directory"],
    )
    parser.add_argument(
        "--box-threshold", type=float, default=0.3, help="Grounding DINO 候选框阈值"
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.25, help="Grounding DINO 文本 token 阈值"
    )


def resolve_sam_arguments(args, workspace):
    """在 CLI 边界一次确定官方配对或显式自训配对，不访问模型。"""
    custom = args.sam_checkpoint is not None or args.sam_config is not None
    if custom:
        if args.sam_checkpoint is None or args.sam_config is None:
            raise ValueError("--sam-checkpoint 与 --sam-config 必须完整配对")
        if args.sam_variant is not None:
            raise ValueError("自训配对不能同时指定 --sam-variant")
    else:
        args.sam_variant = args.sam_variant or "small"
        model = SAM_MODELS[args.sam_variant]
        args.sam_config = model["config"]
        args.sam_checkpoint = Path(workspace) / "reports/cordvip_models" / model["weight"]
    return args


def validate_sam_source(root):
    root = Path(root).expanduser().resolve()
    for relative in (
        "sam2/build_sam.py",
        "sam2/sam2_video_predictor.py",
        "sam2/sam2_image_predictor.py",
    ):
        existing_file(root / relative)
    return root


def validate_sam_checkpoint(root, checkpoint, config):
    existing_file(Path(root).expanduser() / "sam2" / config)
    return existing_file(checkpoint)


def validate_grounding_weights(directory):
    directory = Path(directory).expanduser().resolve()
    for name in GROUNDING_FILES:
        existing_file(directory / name)
    existing_file(directory / GROUNDING_MODEL["weight"])
    return directory


def check_sam_environment(args, *, require_grounding=True):
    checks = []
    for name, check in (
        ("Grounded-SAM2 source", lambda: validate_sam_source(args.sam_root)),
        (
            "SAM2.1 checkpoint/config",
            lambda: validate_sam_checkpoint(args.sam_root, args.sam_checkpoint, args.sam_config),
        ),
    ):
        try:
            detail = check()
            checks.append(dict(name=name, ok=True, detail=str(detail)))
        except (OSError, ValueError) as error:
            checks.append(dict(name=name, ok=False, detail=str(error)))
    if require_grounding:
        try:
            directory = validate_grounding_weights(args.dino_weights)
            checks.append(dict(name="Grounding DINO local files", ok=True, detail=str(directory)))
        except (OSError, ValueError) as error:
            checks.append(dict(name="Grounding DINO local files", ok=False, detail=str(error)))
    # Metadata only. The worker adds the local source directory to its import path.
    script = """
import importlib.metadata
import importlib.util
import json
import sys

packages = ["torch", "torchvision", "numpy"]
modules = ["hydra", "omegaconf", "iopath", "PIL", "tqdm"]
if sys.argv[1] == "grounding":
    packages.append("transformers")
    modules.append("safetensors")
versions = {}
for name in packages:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps(dict(
    python=list(sys.version_info[:2]),
    packages=versions,
    modules={name: importlib.util.find_spec(name) is not None for name in modules},
)))
"""
    try:
        result = subprocess.run(
            [
                str(args.sam_python.expanduser().absolute()),
                "-I",
                "-c",
                script,
                "grounding" if require_grounding else "tracking",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        data = json.loads(result.stdout)
        ok = all(data["packages"].values()) and all(data["modules"].values())
        checks.append(dict(name="Grounded-SAM2 environment", ok=ok, detail=json.dumps(data)))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        checks.append(
            dict(
                name="Grounded-SAM2 environment",
                ok=False,
                detail=getattr(error, "stderr", None) or str(error),
            )
        )
    return checks
