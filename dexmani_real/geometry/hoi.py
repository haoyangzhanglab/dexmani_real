"""HOI 任务资产、固定表面采样与坐标变换。"""

import json
import unicodedata
from pathlib import Path

import numpy as np


def task_objects(task):
    """读取有序任务对象，补齐默认 Prompt。"""
    task = Path(task)
    objects = json.loads((task / "task.json").read_text())["objects"]
    if not objects:
        raise ValueError("任务必须包含非空且不重复的对象")
    names_by_prompt: dict[str, str] = {}
    for obj in objects:
        name = obj["name"]
        if (
            not isinstance(name, str)
            or name in ("", ".", "..")
            or "\0" in name
            or Path(name).name != name
        ):
            raise ValueError(f"非法对象名：{name!r}")
        if name in names_by_prompt.values():
            raise ValueError(f"重复对象名：{name!r}")
        prompt = obj.setdefault("prompt", name)
        if not isinstance(prompt, str) or all(
            char.isspace() or unicodedata.category(char).startswith("P")
            for char in prompt
        ):
            raise ValueError(f"{name}: Prompt 必须是非空字符串，且不能仅含空白或标点")
        # 与跟踪器的标签匹配规则一致。
        normalized = prompt.strip().lower().rstrip(".")
        if normalized in names_by_prompt:
            raise ValueError(
                f"对象 {names_by_prompt[normalized]!r} 与 {name!r} 的 Prompt 冲突：{prompt!r}"
            )
        names_by_prompt[normalized] = name
    return objects


def load_task_template(task):
    task = Path(task)
    with np.load(task / "hoi/template.npz", allow_pickle=False) as saved:
        template = {key: saved[key] for key in ("points_local", "offsets", "object_names")}
    names = template["object_names"]
    if (
        names.ndim != 1
        or names.dtype.kind != "U"
        or not len(names)
        or len(set(names)) != len(names)
        or set(names) != {obj["name"] for obj in task_objects(task)}
    ):
        raise ValueError("模板对象与任务对象不对应")
    points, offsets = template["points_local"], template["offsets"]
    if (
        points.shape != (1024, 3)
        or points.dtype != np.float32
        or not np.isfinite(points).all()
    ):
        raise ValueError("非法 HOI 模板点")
    # 直接比较顺序，避免整数差分溢出后掩盖倒序。
    if (
        offsets.dtype.kind not in "iu"
        or offsets.shape != (len(names) + 1,)
        or offsets[0] != 0
        or offsets[-1] != 1024
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise ValueError("非法 HOI 点分段")
    return template


def transform_task_points(template, poses_camera, base_from_camera):
    """将 [..., objects, 4, 4] 位姿变换为 [..., 1024, 3] 基座系点云。"""
    offsets = template["offsets"]
    # 在线与离线采用相同位姿精度。
    poses_camera = np.asarray(poses_camera, dtype=np.float32)
    points = np.empty((*poses_camera.shape[:-3], 1024, 3), dtype=np.float32)
    for index in range(len(offsets) - 1):
        transform = base_from_camera @ poses_camera[..., index, :, :]
        segment = slice(offsets[index], offsets[index + 1])
        points[..., segment, :] = (
            template["points_local"][segment] @ transform[..., :3, :3].swapaxes(-1, -2)
            + transform[..., None, :3, 3]
        )
    if not np.isfinite(points).all():
        raise ValueError("HOI 变换产生非有限坐标")
    return points


def sample_mesh_surface(mesh, num_points=1024, seed=0):
    """按面积采样固定表面 XYZ。"""
    areas = mesh.area_faces
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Mesh has no finite, nonzero surface area")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(mesh.faces), num_points, p=areas / total)
    triangles = mesh.vertices[mesh.faces[indices]]
    # sqrt(u) 保证三角面内均匀采样。
    uv = rng.random((num_points, 2))
    root_u = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root_u, root_u * (1 - uv[:, 1]), root_u * uv[:, 1]))
    points = np.sum(triangles * weights[:, :, None], axis=1)
    return points.astype(np.float32)


def sample_surface_points(mesh, num_points, seed=0):
    """固定面积候选与 FPS；只在生成新资产时调用。"""
    from fpsample import bucket_fps_kdline_sampling

    if type(num_points) is not int or num_points < 1:
        raise ValueError("num_points must be a positive integer")
    dense = sample_mesh_surface(mesh, 8 * num_points, seed)
    ids = bucket_fps_kdline_sampling(
        dense,
        num_points,
        h=min(5, len(dense).bit_length() - 1),
        start_idx=0,
    )
    return dense[ids]
