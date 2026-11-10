"""冻结任务资产及原始 Mesh 坐标到 xArm base 的唯一变换。"""

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np


def template_digest(template_path, objects, mesh_paths):
    """绑定固定点、物体顺序、跟踪 Mesh 和原始检测 Prompt 的 v2 任务身份。"""
    identity = [(obj["name"], obj["prompt"]) for obj in objects]
    digest = hashlib.sha256(b"dexmani-hoi-task-v2\0")
    digest.update(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    for path in (template_path, *mesh_paths):
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def task_objects(task):
    """读取并校验有序任务对象，在内存中补齐默认 Prompt。"""
    task = Path(task)
    config = json.loads((task / "task.json").read_text()) if (task / "task.json").exists() else {}
    objects = config.get("objects")
    if objects is None:
        objects = [
            {"name": p.name}
            for p in sorted(task.iterdir())
            if p.is_dir() and ((p / "asset.json").is_file() or (p / "reference.jpg").is_file())
        ]
    if not objects:
        raise ValueError("任务必须包含非空且不重复的对象")
    names_by_prompt: dict[str, str] = {}
    for obj in objects:
        name = obj["name"]
        if not isinstance(name, str) or name in ("", ".", "..") or "\0" in name or Path(name).name != name:
            raise ValueError(f"非法对象名：{name!r}")
        if name in names_by_prompt.values():
            raise ValueError(f"重复对象名：{name!r}")
        prompt = obj.setdefault("prompt", name)
        if not isinstance(prompt, str) or all(
            char.isspace() or unicodedata.category(char).startswith("P") for char in prompt
        ):
            raise ValueError(f"{name}: Prompt 必须是非空字符串，且不能仅含空白或标点")
        # 与 SceneTracker.detect_masks 的标签匹配一致；保留原始 Prompt 文本。
        normalized = prompt.strip().lower().rstrip(".")
        if normalized in names_by_prompt:
            raise ValueError(f"对象 {names_by_prompt[normalized]!r} 与 {name!r} 的 Prompt 冲突：{prompt!r}")
        names_by_prompt[normalized] = name
    return objects


def load_task_template(task):
    task = Path(task)
    manifest = json.loads((task / "hoi/manifest.json").read_text())
    if manifest["task_name"] != task.name:
        raise ValueError(f"{task}: Manifest task_name 与任务目录名不匹配")
    names = manifest["object_names"]
    objects = task_objects(task)
    if [obj["name"] for obj in objects] != names:
        raise ValueError("任务对象顺序已改变")
    meshes = []
    for name in names:
        metadata = json.loads((task / name / "asset.json").read_text())
        meshes.append(task / name / metadata["mesh_file"])
    if template_digest(task / "hoi/template.npz", objects, meshes) != manifest["template_sha256"]:
        raise ValueError(f"{task}: HOI 任务身份不匹配：模板、Mesh 或检测 Prompt 与冻结任务不一致")
    with np.load(task / "hoi/template.npz", allow_pickle=False) as saved:
        template = {key: saved[key] for key in ("points_local", "offsets")}
    points, offsets = template["points_local"], template["offsets"]
    if points.shape != (1024, 3) or points.dtype != np.float32 or not np.isfinite(points).all():
        raise ValueError("非法 HOI 模板点")
    # 直接比较顺序，避免整数差分溢出后掩盖倒序。
    if (
        offsets.dtype.kind not in "iu"
        or offsets.shape != (len(names) + 1,)
        or offsets[0] != 0
        or offsets[-1] != 1024
        or np.any(offsets[1:] <= offsets[:-1])
        or np.any(np.diff(offsets) < 64)
    ):
        raise ValueError("非法 HOI 点分段")
    return template, manifest


def transform_task_points(template, poses_camera, base_from_camera):
    """位姿和外参已由跟踪器、相机读取入口校验；这里只做固定点变换。"""
    offsets = template["offsets"]
    # Raw 位姿存储为 float32，在线使用同一精度，避免两条路径舍入不同。
    poses_camera = np.asarray(poses_camera, dtype=np.float32)
    points = np.empty((1024, 3), dtype=np.float32)
    for index in range(len(offsets) - 1):
        transform = base_from_camera @ poses_camera[index]
        segment = slice(offsets[index], offsets[index + 1])
        points[segment] = template["points_local"][segment] @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(points).all():
        raise ValueError("HOI 变换产生非有限坐标")
    return points
