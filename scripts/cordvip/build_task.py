"""构建任务物体资产；已有完整资产优先复用，不覆盖冻结资产。"""

# Usage: python scripts/cordvip/build_task.py --task pnp_apple_banana
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.cli.arguments import parse_object_args
from cordvip_geometry.cli.object import build
from cordvip_geometry.assets import load_asset, validate_asset_parameters
from dexmani_real.geometry.hoi import task_objects


def build_task(task):
    task = Path(task).resolve()
    for obj in task_objects(task):
        directory = task / obj["name"]
        image = directory / obj.get("image", "reference.jpg")
        existing = directory / "asset.json"
        if existing.exists():
            _, points, _, _ = load_asset(directory)
            if len(points) != 1024:
                raise ValueError(f"{directory}: 单物体必须有 1024 个点")
            print(f"复用已发布资产：{directory}")
            continue
        validate_asset_parameters(
            obj.get("scale_to_meters"), obj.get("size_axis"), obj.get("size_meters")
        )
        for name in ("mesh", "points.npy", "normals.npy", "reconstruction"):
            if (directory / name).exists():
                raise FileExistsError(f"拒绝覆盖未发布的资产文件：{directory / name}")
        # 失败时保留候选和生成结果，供明确诊断，不发布 asset.json。
        work = Path(tempfile.mkdtemp(prefix=".build-", dir=directory))
        argv = [
            "build",
            "--image",
            str(image),
            "--prompt",
            obj.get("prompt", obj["name"]),
            "--output",
            str(work / "result"),
            "--headless",
            "--num-points",
            "1024",
        ]
        for name in (
            "scale_to_meters",
            "size_axis",
            "size_meters",
            "instance",
            "seed",
            "steps",
            "octree_resolution",
            "num_chunks",
        ):
            if name in obj:
                argv.extend(["--" + name.replace("_", "-"), str(obj[name])])
        args = parse_object_args(argv, workspace=ROOT)
        build(args)
        asset = work / "result/asset"
        _, _, _, metadata = load_asset(asset)
        metadata.update(
            source_image=image.name, source_mesh="reconstruction/generation/mesh_unscaled.obj"
        )
        for name in ("mesh", "points.npy", "normals.npy"):
            (asset / name).rename(directory / name)
        (asset / "asset.json").unlink()
        asset.rmdir()
        (work / "result").rename(directory / "reconstruction")
        work.rmdir()
        with (directory / "asset.json").open("x") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"发布资产：{directory}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    args = parser.parse_args(argv)
    if Path(args.task).name != args.task or args.task in (".", ".."):
        parser.error("--task 必须是任务目录名")
    build_task(ROOT / "assets/pointcloud_template/object" / args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
