"""按 task 内各闭合物体的实体体积分配总计 1024 个表面点。"""

import argparse
import shutil
import sys
from numbers import Integral
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry.assets import load_asset, write_asset_samples
from cordvip_geometry.io import output_path, write_json

TOTAL_POINTS = 1024
SEED = 0


def allocate_counts(volumes, total=TOTAL_POINTS):
    volumes = np.asarray(volumes, dtype=np.float64)
    if (
        volumes.ndim != 1
        or not len(volumes)
        or not np.isfinite(volumes).all()
        or np.any(volumes <= 0)
    ):
        raise ValueError("每个物体的体积必须为有限正数")
    if (
        isinstance(total, bool)
        or not isinstance(total, Integral)
        or total < len(volumes)
    ):
        raise ValueError("总点数必须是足以给每个物体至少一个点的正整数")
    quota = (total - len(volumes)) * volumes / volumes.sum()
    counts = np.floor(quota).astype(int) + 1
    remainder = total - int(counts.sum())
    order = np.argsort(-(quota - np.floor(quota)), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def write_ply(path, points, normals):
    with path.open("x", encoding="ascii") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        for name in ("x", "y", "z", "nx", "ny", "nz"):
            stream.write(f"property float {name}\n")
        stream.write("end_header\n")
        np.savetxt(stream, np.column_stack((points, normals)), fmt="%.9g")


def allocate_task(task, output=None):
    task = Path(task).expanduser().resolve()
    directories = sorted(path.parent for path in task.glob("*/asset.json"))
    if not directories:
        raise ValueError(f"{task}: 没有包含 asset.json 的物体子目录")
    output = output_path(
        output or task / f"sampled_{TOTAL_POINTS}",
        protected=directories,
    )
    objects = []
    for directory in directories:
        mesh, _, _, metadata = load_asset(directory)
        volume = float(mesh.volume)
        if not mesh.is_volume or not np.isfinite(volume):
            raise ValueError(
                f"{directory.name}: mesh 必须是朝向一致、体积为正的闭合实体"
            )
        objects.append((directory, mesh, metadata, volume))
    counts = allocate_counts([volume for _, _, _, volume in objects])
    output.mkdir(parents=True, exist_ok=False)
    allocation = {}
    for (source, mesh, metadata, volume), count in zip(objects, counts):
        count = int(count)
        destination = output / source.name
        excluded = {"asset.json", metadata["points_file"], metadata["normals_file"]}
        if metadata.get("pointcloud_file"):
            excluded.add(metadata["pointcloud_file"])
        excluded = {(source / relative).resolve() for relative in excluded}
        destination.mkdir()
        for path in source.rglob("*"):
            copied = destination / path.relative_to(source)
            if path.is_dir():
                copied.mkdir(parents=True, exist_ok=True)
            elif path.resolve() not in excluded:
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copied)
        points, normals = write_asset_samples(
            destination,
            mesh,
            metadata,
            num_points=count,
            seed=SEED,
        )
        write_ply(destination / "pointcloud.ply", points, normals)
        if "source_mesh" in metadata:
            metadata["source_mesh"] = str((source / metadata["source_mesh"]).resolve())
        metadata["pointcloud_file"] = "pointcloud.ply"
        write_json(destination / "asset.json", metadata)
        allocation[source.name] = {"volume_m3": volume, "num_points": count}
        print(f"{source.name:20} {volume * 1e6:12.3f} cm³  {count:4d} points")
    write_json(
        output / "allocation.json",
        {
            "source_task": str(task),
            "total_points": TOTAL_POINTS,
            "volume_method": "mesh",
            "objects": allocation,
        },
    )
    print(f"总计 {int(counts.sum())} points；资产目录：{output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, help="包含各物体资产子目录的 task 目录")
    parser.add_argument(
        "--output", type=Path, help="新输出目录，默认 task/sampled_1024"
    )
    args = parser.parse_args(argv)
    allocate_task(args.task, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
