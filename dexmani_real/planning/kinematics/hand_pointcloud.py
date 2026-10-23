"""Fixed XHand visual-surface XYZRGB samples; only measured FK moves them.

Surface sampling follows cordvip_geometry/hand_cloud.py; FK stays in HandKinematics.
The small URDF marker meshes are excluded as in Sim's hand-template generator.
"""

import xml.etree.ElementTree as ET

import numpy as np

from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.sensor.pointcloud.hand_config import (
    HAND_CLOUD_COLOR,
    HAND_CLOUD_LINK_SELECTION,
    HAND_CLOUD_SAMPLING,
    HandPointCloudConfig,
)


def _vector(text, default):
    value = np.asarray(default if text is None else text.split(), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("URDF visual scale/origin must contain three finite numbers")
    return value


class HandPointCloudTemplate:
    """One immutable template per consumer, using its existing hand FK workspace."""

    def __init__(self, config: HandPointCloudConfig, hand_fk):
        config.validate()
        import trimesh

        path = XHAND_RIGHT_URDF_PATH
        root = ET.parse(path).getroot()
        materials = {
            node.get("name"): node.find("color").get("rgba")
            for node in root.findall("material") if node.find("color") is not None
        }
        parts, names = [], []
        for link in root.findall("link"):
            name = link.get("name", "")
            if not name.startswith("right_hand_"):
                continue
            triangles, colors = [], []
            for visual in link.findall("visual"):
                mesh_tag = visual.find("geometry/mesh")
                if mesh_tag is None:
                    raise ValueError(f"{name}: hand visual must be an explicit triangle mesh")
                filename = mesh_tag.get("filename", "")
                if not filename or "://" in filename:
                    raise ValueError(f"{name}: hand mesh must use a local path")
                mesh_path = (path.parent / filename).resolve()
                mesh = trimesh.load(mesh_path, process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(f"{mesh_path}: expected one triangle mesh")
                scale = _vector(mesh_tag.get("scale"), (1, 1, 1))
                if np.any(scale <= 0):
                    raise ValueError("hand visual scale must be positive")
                mesh.apply_scale(scale)
                origin = visual.find("origin")
                if origin is not None:
                    transform = np.eye(4)
                    transform[:3, 3] = _vector(origin.get("xyz"), (0, 0, 0))
                    transform[:3, :3] = hand_fk.pin.rpy.rpyToMatrix(
                        _vector(origin.get("rpy"), (0, 0, 0))
                    )
                    mesh.apply_transform(transform)
                if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.area) or mesh.area <= 0:
                    raise ValueError(f"{mesh_path}: mesh must have positive finite area")
                material = visual.find("material")
                rgba = None
                if material is not None:
                    color = material.find("color")
                    rgba = color.get("rgba") if color is not None else materials.get(material.get("name"))
                rgb = np.full(3, 0.5) if rgba is None else np.asarray(rgba.split(), dtype=np.float64)[:3]
                if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
                    raise ValueError(f"{name}: invalid URDF material RGB")
                triangles.append(np.asarray(mesh.triangles, dtype=np.float64))
                colors.append(np.broadcast_to(rgb, (len(mesh.faces), 3)))
            if not triangles:
                continue
            triangles = np.concatenate(triangles)
            colors = np.concatenate(colors)
            area = np.linalg.norm(np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ), axis=1) / 2
            if area.sum() < 1e-4:
                continue
            parts.append((triangles, colors, area))
            names.append(name)
        if not parts:
            raise ValueError("hand URDF contains no retained visual surfaces")
        areas = np.asarray([part[2].sum() for part in parts])
        quota = config.num_points * areas / areas.sum()
        counts = np.floor(quota).astype(np.int64)
        order = np.argsort(-(quota - counts), kind="stable")
        counts[order[:config.num_points - int(counts.sum())]] += 1
        if np.any(counts == 0):
            raise ValueError("hand point budget must cover every retained visual link")
        self.frame_ids = tuple(hand_fk.model.getFrameId(name) for name in names)
        if any(fid >= hand_fk.model.nframes for fid in self.frame_ids):
            raise ValueError("hand visual link is missing from the FK model")
        points, point_colors, link_ids = [], [], []
        for index, ((triangles, colors, area), count) in enumerate(zip(parts, counts)):
            rng = np.random.default_rng(config.seed + index)
            selected = rng.choice(len(area), int(count), p=area / area.sum())
            uv = rng.random((int(count), 2))
            root_u = np.sqrt(uv[:, 0])
            weights = np.column_stack((1 - root_u, root_u * (1 - uv[:, 1]), root_u * uv[:, 1]))
            points.append(np.sum(triangles[selected] * weights[:, :, None], axis=1))
            point_colors.append(colors[selected])
            link_ids.append(np.full(int(count), index, dtype=np.intp))
        self.points = np.concatenate(points)
        self.colors = np.concatenate(point_colors).astype(np.float32)
        self.link_ids = np.concatenate(link_ids)
        for array in (self.points, self.colors, self.link_ids):
            array.setflags(write=False)
        self.recipe = dict(
            num_points=config.num_points, seed=config.seed,
            sampling_method=HAND_CLOUD_SAMPLING, link_selection=HAND_CLOUD_LINK_SELECTION,
            color_rule=HAND_CLOUD_COLOR,
        )

    def transform(self, handbase_from_links, base_from_handbase) -> np.ndarray:
        """Return base-frame XYZ metres and fixed RGB in [0,1], in template point order."""
        transforms = (base_from_handbase @ handbase_from_links)[self.link_ids]
        xyz = np.einsum("nij,nj->ni", transforms[:, :3, :3], self.points) + transforms[:, :3, 3]
        cloud = np.empty((len(self.points), 6), dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            cloud[:, :3] = xyz
        cloud[:, 3:] = self.colors
        if not np.isfinite(cloud[:, :3]).all():
            raise ValueError("hand point-cloud geometry is not finite float32")
        return cloud
