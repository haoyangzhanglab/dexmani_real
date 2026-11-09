"""按表面积冻结任务级 1024 点模板，不复制原始资产。"""

# Usage: python scripts/cordvip/allocate_points.py assets/pointcloud_template/object/pnp_apple_banana
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.assets import load_asset
from dexmani_real.geometry.hoi import template_digest, task_objects, load_task_template


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


def select_indices(points, count):
    points = np.asarray(points, dtype=np.float64)
    selected = np.empty(count, dtype=np.int64)
    distances = np.full(len(points), np.inf)
    index = 0
    for i in range(count):
        selected[i] = index
        distances = np.minimum(distances, np.sum((points - points[index]) ** 2, axis=1))
        distances[selected[: i + 1]] = -1
        index = int(np.argmax(distances))
    return selected


def allocate_task(task):
    task = Path(task).resolve()
    output = task / "hoi"
    if output.exists():
        load_task_template(task)
        return output
    names = [obj["name"] for obj in task_objects(task)]
    loaded = [load_asset(task / name) for name in names]
    counts = allocate_counts([mesh.area for mesh, _, _, _ in loaded])
    points, meshes = [], []
    for name, (_, p, _, metadata), count in zip(names, loaded, counts):
        if len(p) != 1024:
            raise ValueError(f"{name}: 单物体必须有 1024 个点")
        indices = select_indices(p, int(count))
        points.append(p[indices])
        meshes.append(task / name / metadata["mesh_file"])
    with tempfile.TemporaryDirectory(prefix=".hoi-", dir=task) as temporary:
        staging = Path(temporary)
        np.savez(
            staging / "template.npz",
            points_local=np.concatenate(points),
            offsets=np.r_[0, np.cumsum(counts)].astype(np.int64),
        )
        manifest = dict(
            task_name=task.name,
            object_names=names,
            template_sha256=template_digest(staging / "template.npz", names, meshes),
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )
        staging.rename(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    args = parser.parse_args(argv)
    print(allocate_task(args.task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
