"""按 XHand URDF 表面面积分配固定 XYZRGB 样本，再由实测关节 FK 变换。"""

import xml.etree.ElementTree as ET

import numpy as np

from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.geometry.pose import rot6d_to_rotmat
from dexmani_real.perception.config import HandCloudConfig
from dexmani_real.robot.model import HAND_FINGERTIP_SHAPE, XHAND_RIGHT_URDF_PATH


def _urdf_vector(text, default):
    value = np.asarray(default if text is None else text.split(), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("URDF visual scale/origin must contain three finite numbers")
    return value


class HandSurface:
    def __init__(self, config: HandCloudConfig, hand_fk: HandFK):
        import trimesh

        path = XHAND_RIGHT_URDF_PATH
        root = ET.parse(path).getroot()
        materials = {
            node.get("name"): color.get("rgba")
            for node in root.findall("material")
            if (color := node.find("color")) is not None
        }
        parts, names = [], []
        for link in root.findall("link"):
            name = link.get("name", "")
            if not name.startswith("right_hand_"):
                continue
            triangle_parts, color_parts = [], []
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
                scale = _urdf_vector(mesh_tag.get("scale"), (1, 1, 1))
                if np.any(scale <= 0):
                    raise ValueError("hand visual scale must be positive")
                mesh.apply_scale(scale)
                origin = visual.find("origin")
                if origin is not None:
                    transform = np.eye(4)
                    transform[:3, 3] = _urdf_vector(origin.get("xyz"), (0, 0, 0))
                    transform[:3, :3] = hand_fk.pin.rpy.rpyToMatrix(
                        _urdf_vector(origin.get("rpy"), (0, 0, 0))
                    )
                    mesh.apply_transform(transform)
                if (
                    not np.isfinite(mesh.vertices).all()
                    or not np.isfinite(mesh.area)
                    or mesh.area <= 0
                ):
                    raise ValueError(f"{mesh_path}: mesh must have positive finite area")
                material = visual.find("material")
                rgba = None
                if material is not None:
                    color = material.find("color")
                    rgba = (
                        color.get("rgba")
                        if color is not None
                        else materials.get(material.get("name"))
                    )
                rgb = (
                    np.full(3, 0.5)
                    if rgba is None
                    else np.asarray(rgba.split(), dtype=np.float64)[:3]
                )
                if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
                    raise ValueError(f"{name}: invalid URDF material RGB")
                triangle_parts.append(np.asarray(mesh.triangles, dtype=np.float64))
                color_parts.append(np.broadcast_to(rgb, (len(mesh.faces), 3)))
            if not triangle_parts:
                continue
            triangles = np.concatenate(triangle_parts)
            colors = np.concatenate(color_parts)
            area = (
                np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                    axis=1,
                )
                / 2
            )
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
        counts[order[: config.num_points - int(counts.sum())]] += 1
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

    def transform(self, handbase_from_links, base_from_handbase) -> np.ndarray:
        transforms = base_from_handbase @ handbase_from_links
        rotations = transforms[self.link_ids, :3, :3]
        translations = transforms[self.link_ids, :3, 3]
        cloud = np.empty((len(self.points), 6), dtype=np.float32)
        cloud[:, :3] = np.einsum("nij,nj->ni", rotations, self.points) + translations
        cloud[:, 3:] = self.colors
        return cloud


def hand_geometry_sequence(
    eef_pose_history: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    hand_fk: HandFK,
    handbase_position_eef_m: np.ndarray | tuple[float, ...],
    handbase_rotation_eef: np.ndarray,
    hand_cloud: HandSurface | None = None,
    include_fingertips: bool = True,
) -> dict[str, np.ndarray]:
    """由已对齐的 FK 位姿和手关节推导指尖或手点云，每行仅计算一次手部 FK。"""
    poses = np.asarray(eef_pose_history, dtype=np.float64)
    hand = np.asarray(hand_qpos, dtype=np.float64)
    mount_p = np.asarray(handbase_position_eef_m, dtype=np.float64)
    frame_ids = list(hand_fk.fingertip_frame_ids) if include_fingertips else []
    tip_count = len(frame_ids)
    if hand_cloud is not None:
        frame_ids.extend(hand_cloud.frame_ids)
    arrays = {}
    if include_fingertips:
        arrays["fingertip_xyz"] = np.empty((len(poses), *HAND_FINGERTIP_SHAPE), dtype=np.float32)
    if hand_cloud is not None:
        arrays["hand_pointcloud"] = np.empty(
            (len(poses), len(hand_cloud.points), 6), dtype=np.float32
        )
    for index, pose in enumerate(poses):
        rotation = rot6d_to_rotmat(pose[3:])
        base_from_hand = np.eye(4)
        base_from_hand[:3, :3] = rotation @ handbase_rotation_eef
        base_from_hand[:3, 3] = rotation @ mount_p + pose[:3]
        placements = hand_fk.forward(hand[index], frame_ids)
        if include_fingertips:
            tips_hand = placements[:tip_count, :3, 3]
            arrays["fingertip_xyz"][index] = (
                tips_hand @ base_from_hand[:3, :3].T + base_from_hand[:3, 3]
            )
        if hand_cloud is not None:
            arrays["hand_pointcloud"][index] = hand_cloud.transform(
                placements[tip_count:], base_from_hand
            )
    return arrays
