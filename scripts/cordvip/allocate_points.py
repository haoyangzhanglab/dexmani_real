"""按表面积配额直接采样并冻结任务级 1024 点模板。"""

# Usage: python scripts/cordvip/allocate_points.py assets/pointcloud_template/object/pnp_apple_banana
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.assets import load_asset
from cordvip_geometry.geometry import sample_mesh_surface

from dexmani_real.geometry.hoi import load_task_template, task_objects
from dexmani_real.utils.io import publish


def allocate_counts(areas):
    areas = np.asarray(areas, dtype=np.float64)
    if areas.ndim != 1 or not len(areas) or not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("表面积必须为非空有限正数数组")
    remaining = 1024 - 64 * len(areas)
    if remaining < 0:
        raise ValueError("1024 点、每物体至少 64 点，最多支持 16 个物体")
    quota = remaining * (areas / areas.sum())
    counts = np.floor(quota).astype(int) + 64
    order = np.argsort(-(quota - np.floor(quota)), kind="stable")
    counts[order[: 1024 - counts.sum()]] += 1
    return counts


def allocate_task(task):
    task = Path(task).resolve()
    output = task / "hoi"
    if output.exists():
        load_task_template(task)
        return output
    objects = task_objects(task)
    names = [obj["name"] for obj in objects]
    loaded = [load_asset(task / name) for name in names]
    counts = allocate_counts([mesh.area for mesh, _ in loaded])
    points = [sample_mesh_surface(mesh, int(count), seed=0)[0]
              for (mesh, _), count in zip(loaded, counts)]
    with tempfile.TemporaryDirectory(prefix=".hoi-", dir=task) as temporary:
        staging = Path(temporary)
        np.savez(
            staging / "template.npz",
            points_local=np.concatenate(points),
            object_names=np.asarray(names),
            offsets=np.r_[0, np.cumsum(counts)].astype(np.int64),
        )
        publish(staging, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    args = parser.parse_args(argv)
    print(allocate_task(args.task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
