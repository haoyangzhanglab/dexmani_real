"""HOI 任务资产、固定表面采样与坐标变换。"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from dexmani_real.config.perception import HOI_NUM_POINTS
from dexmani_real.geometry.sampling import (
    farthest_point_indices,
    validate_point_segments,
)

TASK_MESH_PATH = Path("mesh/mesh_m.obj")


def task_objects(task):
    """读取有序任务对象，补齐默认 Prompt。"""
    task = Path(task)
    objects = json.loads((task / "task.json").read_text())["objects"]
    if not objects:
        raise ValueError("任务必须包含非空且不重复的对象")
    names = set()
    for obj in objects:
        name = obj["name"]
        if (
            not isinstance(name, str)
            or name in ("", ".", "..")
            or "\0" in name
            or Path(name).name != name
        ):
            raise ValueError(f"非法对象名：{name!r}")
        if name in names:
            raise ValueError(f"重复对象名：{name!r}")
        names.add(name)
        obj.setdefault("prompt", name)
    return objects


def load_hoi_task(
    task: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Path], dict[str, str]]:
    """加载已校验的模板及按模板顺序排列的跟踪输入。"""
    task = Path(task)
    with np.load(task / "hoi/template.npz", allow_pickle=False) as saved:
        template = {key: saved[key] for key in ("points_local", "offsets", "object_names")}
    validate_task_template(template)
    names = template["object_names"].tolist()
    prompts_by_name = {obj["name"]: obj["prompt"] for obj in task_objects(task)}
    if set(names) != prompts_by_name.keys():
        raise ValueError("模板对象与任务对象不对应")
    mesh_paths = {name: task / name / TASK_MESH_PATH for name in names}
    prompts = {name: prompts_by_name[name] for name in names}
    return template, mesh_paths, prompts


def validate_task_template(template: Mapping[str, np.ndarray]) -> None:
    """资产文件和 Raw 标注共用同一模板结构。"""
    validate_point_segments(
        template["points_local"], template["object_names"], template["offsets"],
        num_points=HOI_NUM_POINTS, label="HOI template",
    )


def transform_task_points(template, poses_camera, base_from_camera):
    """将 [..., objects, 4, 4] 位姿变换为 [..., 1024, 3] 基座系点云。"""
    offsets = template["offsets"]
    # 在线与离线采用相同位姿精度。
    poses_camera = np.asarray(poses_camera, dtype=np.float32)
    points = np.empty((*poses_camera.shape[:-3], HOI_NUM_POINTS, 3), dtype=np.float32)
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


def sample_surface_points(mesh, num_points, seed=0):
    """从已校验的网格按面积采样，再用 FPS 选出固定点序。"""
    candidate_count = 8 * num_points
    areas = mesh.area_faces
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(mesh.faces), candidate_count, p=areas / areas.sum())
    triangles = mesh.vertices[mesh.faces[indices]]
    # sqrt(u) 保证三角面内均匀采样。
    uv = rng.random((candidate_count, 2))
    root_u = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root_u, root_u * (1 - uv[:, 1]), root_u * uv[:, 1]))
    dense = np.sum(triangles * weights[:, :, None], axis=1).astype(np.float32)
    return dense[farthest_point_indices(dense, num_points)]
