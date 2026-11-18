"""按表面积分配配额，用 FPS 生成固定的任务级 1024 点模板。"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry.mesh import load_mesh

from dexmani_real.config.perception import (
    HOI_MAX_OBJECTS,
    HOI_MIN_POINTS_PER_OBJECT,
    HOI_NUM_POINTS,
)
from dexmani_real.geometry.hoi import (
    load_hoi_task,
    sample_surface_points,
    task_objects,
    validate_task_template,
)
from dexmani_real.utils.io import publish


def allocate_counts(areas):
    areas = np.asarray(areas, dtype=np.float64)
    remaining = HOI_NUM_POINTS - HOI_MIN_POINTS_PER_OBJECT * len(areas)
    if remaining < 0:
        raise ValueError(
            f"{HOI_NUM_POINTS} 点、每物体至少 {HOI_MIN_POINTS_PER_OBJECT} 点，"
            f"最多支持 {HOI_MAX_OBJECTS} 个物体"
        )
    quota = remaining * (areas / areas.sum())
    counts = np.floor(quota).astype(int)
    order = np.argsort(-(quota - counts), kind="stable")
    counts += HOI_MIN_POINTS_PER_OBJECT
    counts[order[: HOI_NUM_POINTS - counts.sum()]] += 1
    return counts


def sample_template(task, *, overwrite=False):
    task = Path(task).expanduser().resolve()
    output = task / "hoi"
    path = output / "template.npz"
    if path.exists() and not overwrite:
        load_hoi_task(task)
        return output
    names = [obj["name"] for obj in task_objects(task)]
    meshes = [load_mesh(task / name / "mesh/mesh_m.obj") for name in names]
    counts = allocate_counts([mesh.area for mesh in meshes])
    points = [
        sample_surface_points(mesh, int(count), seed=0) for mesh, count in zip(meshes, counts)
    ]
    template = {
        "points_local": np.concatenate(points),
        "object_names": np.asarray(names),
        "offsets": np.r_[0, np.cumsum(counts)].astype(np.int64),
    }
    validate_task_template(template)
    output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".template.npz.", dir=output)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **template)
        if overwrite:
            os.replace(temporary, path)
        else:
            publish(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task-dir", type=Path, required=True, help="任务资产目录；相对 cwd，支持 ~")
    parser.add_argument("--overwrite", action="store_true", help="显式重建模板，可能改变点序")
    args = parser.parse_args(argv)
    print(
        sample_template(
            args.task_dir,
            overwrite=args.overwrite,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
