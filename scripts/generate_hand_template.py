#!/usr/bin/env python3
"""离线生成手部 link 局部 XYZRGB 模板，运行时仅按 FK 变换组装。"""

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.config.perception import HandCloudConfig
from dexmani_real.perception.hand_cloud import HAND_TEMPLATE_DIR
from dexmani_real.robot.model import XARM7_XHAND_RIGHT_URDF_PATH
from dexmani_real.utils.io import publish


def generate_hand_template(
    num_points: int,
    *,
    urdf_path: Path = XARM7_XHAND_RIGHT_URDF_PATH,
    output_dir: Path = HAND_TEMPLATE_DIR,
    overwrite: bool = False,
) -> Path:
    import trimesh

    HandCloudConfig(num_points=num_points)
    path = output_dir / f"hand_{num_points}.npz"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Hand template already exists: {path}; use --overwrite to regenerate")
    meshes, colors, names = [], [], []
    for link in ET.parse(urdf_path).getroot().findall("link"):
        if not link.attrib["name"].startswith("right_hand_"):
            continue
        visuals = link.findall("visual")
        if not visuals:
            continue
        (visual,) = visuals
        mesh_tag = visual.find("geometry/mesh")
        color = visual.find("material/color")
        if mesh_tag is None or color is None:
            raise ValueError(f"Hand visual requires a mesh and inline color: {link.attrib['name']}")
        mesh = trimesh.load_mesh(urdf_path.parent / mesh_tag.attrib["filename"], process=False)
        mesh.apply_scale(np.fromstring(mesh_tag.get("scale", "1 1 1"), sep=" "))
        origin = visual.find("origin")
        if origin is not None:
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "),
            ).as_matrix()
            mesh.apply_transform(transform)
        if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area) or mesh.area <= 0:
            raise ValueError(f"Invalid hand visual mesh: {link.attrib['name']}")
        # 与仿真一致，剔除小于 1 cm² 的辅助标记，保留 bend links。
        if mesh.area < 1e-4:
            continue
        rgb = np.fromstring(color.attrib["rgba"], sep=" ")[:3]
        if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError(f"Invalid hand visual color: {link.attrib['name']}")
        meshes.append(mesh)
        colors.append(rgb)
        names.append(link.attrib["name"])
    if not meshes:
        raise ValueError(f"No hand visual surfaces in {urdf_path}")

    areas = np.array([mesh.area for mesh in meshes])
    quota = num_points * areas / areas.sum()
    counts = np.floor(quota).astype(np.int64)
    order = np.argsort(-(quota - counts), kind="stable")
    counts[order[: num_points - int(counts.sum())]] += 1
    if np.any(counts == 0):
        raise ValueError("Hand point budget leaves a retained link without samples; increase num_points")

    clouds = []
    for index, (mesh, rgb, count) in enumerate(zip(meshes, colors, counts)):
        rng = np.random.default_rng(index)
        selected = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
        uv = rng.random((count, 2))
        root_u = np.sqrt(uv[:, 0])
        weights = np.column_stack((1 - root_u, root_u * (1 - uv[:, 1]), root_u * uv[:, 1]))
        xyz = np.sum(mesh.triangles[selected] * weights[:, :, None], axis=1)
        clouds.append(np.column_stack((xyz, np.broadcast_to(rgb, (count, 3)))))

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                points=np.concatenate(clouds).astype(np.float32),
                part_names=np.asarray(names),
                part_offsets=np.r_[0, np.cumsum(counts)],
                part_areas=areas,
            )
        if overwrite:
            os.replace(temporary, path)
        else:
            publish(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-points", type=int, default=HandCloudConfig().num_points)
    parser.add_argument("--output-dir", type=Path, default=HAND_TEMPLATE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = generate_hand_template(
        args.num_points, output_dir=args.output_dir, overwrite=args.overwrite,
    )
    print(f"Saved {path}: {args.num_points} points")


if __name__ == "__main__":
    main()
