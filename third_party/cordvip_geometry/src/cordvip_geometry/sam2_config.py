"""Pinned local Grounded-SAM2 inputs; checks never import models or open devices."""

import json
from pathlib import Path
import subprocess
import sys

from .io import sha256, existing_file

SOURCE_REVISION = "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28"
CORE_VERSIONS = dict(torch="2.4.1+cu124", torchvision="0.19.1+cu124", numpy="1.26.4")
TRANSFORMERS_VERSION = "4.48.0"
SAM_MODELS = {
    "small": dict(
        config="configs/sam2.1/sam2.1_hiera_s.yaml",
        repo_id="facebook/sam2.1-hiera-small",
        revision="ee5bba1d82bb8749febdf90f45e84b687142ba03",
        weight="sam2.1_hiera_small.pt",
        size=184416285,
        sha256="6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    ),
    "large": dict(
        config="configs/sam2.1/sam2.1_hiera_l.yaml",
        repo_id="facebook/sam2.1-hiera-large",
        revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
        weight="sam2.1_hiera_large.pt",
        size=898083611,
        sha256="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    ),
}
GROUNDING_MODEL = dict(
    repo_id="IDEA-Research/grounding-dino-base",
    revision="12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    directory="grounding-dino-base",
    weight="model.safetensors",
    size=933400872,
    sha256="5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21",
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
        help="分割子进程解释器；默认当前解释器，须满足固定 Python/依赖版本",
    )
    parser.add_argument("--sam-root", type=Path, default=workspace / "third_party/Grounded-SAM-2")
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=workspace / "reports/cordvip_models" / SAM_MODELS["small"]["weight"],
    )
    parser.add_argument(
        "--sam-config",
        default=SAM_MODELS["small"]["config"],
        help="SAM2 包内配置；切换 large 时须同时指定匹配 checkpoint",
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


def validate_sam_source(root):
    root = Path(root).expanduser().resolve()
    for relative in (
        "sam2/build_sam.py",
        "sam2/sam2_video_predictor.py",
        "sam2/sam2_image_predictor.py",
    ):
        existing_file(root / relative)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    top = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if Path(top).resolve() != root:
        raise ValueError("Grounded-SAM2 root must be a checkout/worktree top level")
    if revision != SOURCE_REVISION:
        raise ValueError(f"Grounded-SAM2 requires source {SOURCE_REVISION}; found {revision}")
    # Hash untracked SAM/Hydra code too: it can shadow pinned executable inputs.
    protected = (
        "setup.py",
        "pyproject.toml",
        "setup.cfg",
        *(
            f":(glob)sam2/**/*.{suffix}"
            for suffix in ("py", "yaml", "yml", "json", "cu", "cpp", "h", "cuh")
        ),
    )
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *protected,
        ],
        text=True,
    )
    if dirty:
        raise ValueError("Grounded-SAM2 executable source/config contains edits")
    return root


def validate_sam_checkpoint(root, checkpoint, config, *, verify_hash=True):
    spec = next((spec for spec in SAM_MODELS.values() if spec["config"] == config), None)
    if spec is None:
        raise ValueError("Use the pinned SAM2.1 small or large configuration")
    existing_file(Path(root) / "sam2" / config)
    checkpoint = existing_file(checkpoint)
    if checkpoint.stat().st_size != spec["size"] or (
        verify_hash and sha256(checkpoint) != spec["sha256"]
    ):
        raise ValueError(f"SAM2.1 checkpoint does not match {config}: {checkpoint}")
    return spec


def validate_grounding_weights(directory, *, verify_hash=True):
    directory = Path(directory).expanduser().resolve()
    for name in GROUNDING_FILES:
        existing_file(directory / name)
    weight = existing_file(directory / GROUNDING_MODEL["weight"])
    if weight.stat().st_size != GROUNDING_MODEL["size"] or (
        verify_hash and sha256(weight) != GROUNDING_MODEL["sha256"]
    ):
        raise ValueError(
            f"Grounding DINO weight does not match the pinned official SHA256: {weight}"
        )
    return directory


def check_sam_environment(args, *, require_grounding=True):
    checks = []
    for name, check in (
        ("Grounded-SAM2 source", lambda: validate_sam_source(args.sam_root)),
        (
            "SAM2.1 checkpoint/config",
            lambda: validate_sam_checkpoint(
                args.sam_root, args.sam_checkpoint, args.sam_config, verify_hash=False
            ),
        ),
    ):
        try:
            detail = check()
            checks.append(dict(name=name, ok=True, detail=str(detail)))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            checks.append(dict(name=name, ok=False, detail=str(error)))
    if require_grounding:
        try:
            directory = validate_grounding_weights(args.dino_weights, verify_hash=False)
            checks.append(dict(name="Grounding DINO local files", ok=True, detail=str(directory)))
        except (OSError, ValueError) as error:
            checks.append(dict(name="Grounding DINO local files", ok=False, detail=str(error)))
    # Metadata only. The source checkout is added by the worker, not installed via pip.
    script = """import sys,json,importlib.metadata as m,importlib.util as u
packages = {}
for p in ['torch', 'torchvision', 'numpy', 'transformers']:
    try: packages[p] = m.version(p)
    except m.PackageNotFoundError: packages[p] = None
modules = ['hydra','omegaconf','iopath','PIL','tqdm']
if sys.argv[1] == 'grounding': modules.append('safetensors')
print(json.dumps(dict(python=list(sys.version_info[:2]), packages=packages,
    modules={p:u.find_spec(p) is not None for p in
             modules})))
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
        expected = CORE_VERSIONS.copy()
        if require_grounding:
            expected["transformers"] = TRANSFORMERS_VERSION
        ok = (
            data["python"] == [3, 10]
            and all(data["packages"][k] == v for k, v in expected.items())
            and all(data["modules"].values())
        )
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
