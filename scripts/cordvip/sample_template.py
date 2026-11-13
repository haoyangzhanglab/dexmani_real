"""按表面积分配配额，用 FPS 生成固定的任务级 1024 点模板。"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry.mesh import load_mesh
from dexmani_real.geometry.hoi import (
    load_task_template,
    sample_surface_points,
    task_objects,
)


def allocate_counts(areas):
    areas = np.asarray(areas, dtype=np.float64)
    if (
        areas.ndim != 1
        or not len(areas)
        or not np.isfinite(areas).all()
        or np.any(areas <= 0)
    ):
        raise ValueError("表面积必须为非空有限正数数组")
    remaining = 1024 - 64 * len(areas)
    if remaining < 0:
        raise ValueError("1024 点、每物体至少 64 点，最多支持 16 个物体")
    quota = remaining * (areas / areas.sum())
    counts = np.floor(quota).astype(int)
    order = np.argsort(-(quota - counts), kind="stable")
    counts += 64
    counts[order[: 1024 - counts.sum()]] += 1
    return counts


def sample_template(task):
    task = Path(task).resolve()
    output = task / "hoi"
    if (output / "template.npz").exists():
        load_task_template(task)
        return output
    names = [obj["name"] for obj in task_objects(task)]
    meshes = [load_mesh(task / name / "mesh/mesh_m.obj") for name in names]
    counts = allocate_counts([mesh.area for mesh in meshes])
    points = [
        sample_surface_points(mesh, int(count), seed=0) for mesh, count in zip(meshes, counts)
    ]
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "template.npz",
        points_local=np.concatenate(points),
        object_names=np.asarray(names),
        offsets=np.r_[0, np.cumsum(counts)].astype(np.int64),
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    args = parser.parse_args(argv)
    if Path(args.task).name != args.task or args.task in (".", ".."):
        parser.error("非法任务名")
    print(sample_template(ROOT / "assets/pointcloud_template/object" / args.task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
