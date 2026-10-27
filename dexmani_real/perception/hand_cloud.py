"""在 XHand 的 link 局部网格上固定采样，再由关节 FK 变换到基座系。"""

import xml.etree.ElementTree as ET

import numpy as np

from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.geometry.pose import rot6d_to_rotmat
from dexmani_real.robot.model import HAND_FINGERTIP_SHAPE, XHAND_RIGHT_URDF_PATH


class HandPointCloud:
    def __init__(self, hand_fk: HandFK, num_points: int):
        import trimesh

        path = XHAND_RIGHT_URDF_PATH
        meshes, colors, names = [], [], []
        for link in ET.parse(path).getroot().findall("link"):
            visual = link.find("visual")
            if visual is None:
                continue
            mesh = trimesh.load_mesh(
                path.parent / visual.findall("geometry/mesh")[0].attrib["filename"], process=False
            )
            if mesh.area < 1e-4:
                continue
            rgb = np.fromstring(visual.findall("material/color")[0].attrib["rgba"], sep=" ")[:3]
            meshes.append(mesh)
            colors.append(rgb)
            names.append(link.attrib["name"])

        areas = np.array([mesh.area for mesh in meshes])
        quota = num_points * areas / areas.sum()
        counts = np.floor(quota).astype(np.int64)
        order = np.argsort(-(quota - counts), kind="stable")
        counts[order[: num_points - int(counts.sum())]] += 1
        self.frame_ids = tuple(hand_fk.model.getFrameId(name) for name in names)
        if any(fid >= hand_fk.model.nframes for fid in self.frame_ids):
            raise ValueError("hand visual link is missing from the FK model")

        points, point_colors, link_ids = [], [], []
        for index, (mesh, rgb, count) in enumerate(zip(meshes, colors, counts)):
            rng = np.random.default_rng(index)
            selected = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
            uv = rng.random((count, 2))
            root_u = np.sqrt(uv[:, 0])
            weights = np.column_stack((1 - root_u, root_u * (1 - uv[:, 1]), root_u * uv[:, 1]))
            points.append(np.sum(mesh.triangles[selected] * weights[:, :, None], axis=1))
            point_colors.append(np.broadcast_to(rgb, (count, 3)))
            link_ids.append(np.full(count, index, dtype=np.intp))
        self.points = np.concatenate(points)
        self.colors = np.concatenate(point_colors).astype(np.float32)
        self.link_ids = np.concatenate(link_ids)

    def transform(self, handbase_from_links, base_from_handbase) -> np.ndarray:
        transforms = base_from_handbase @ handbase_from_links
        rotations = transforms[self.link_ids, :3, :3]
        translations = transforms[self.link_ids, :3, 3]
        cloud = np.empty((len(self.points), 6), dtype=np.float32)
        cloud[:, :3] = np.einsum("nij,nj->ni", rotations, self.points) + translations
        cloud[:, 3:] = self.colors
        return cloud


def hand_geometry(
    eef_pose_history: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    hand_fk: HandFK,
    handbase_position_eef_m: np.ndarray | tuple[float, ...],
    handbase_rotation_eef: np.ndarray,
    hand_cloud: HandPointCloud | None = None,
    include_fingertips: bool = True,
) -> dict[str, np.ndarray]:
    """由已对齐的 EEF 位姿和手关节生成指尖与手点云，每行共用一次 FK。"""
    poses = np.asarray(eef_pose_history, dtype=np.float64)
    rotations = rot6d_to_rotmat(poses[:, 3:])
    base_from_hand = np.tile(np.eye(4), (len(poses), 1, 1))
    base_from_hand[:, :3, :3] = rotations @ handbase_rotation_eef
    base_from_hand[:, :3, 3] = rotations @ np.asarray(handbase_position_eef_m) + poses[:, :3]

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

    for index, transform in enumerate(base_from_hand):
        placements = hand_fk.forward(hand_qpos[index], frame_ids)
        if include_fingertips:
            arrays["fingertip_xyz"][index] = (
                placements[:tip_count, :3, 3] @ transform[:3, :3].T + transform[:3, 3]
            )
        if hand_cloud is not None:
            arrays["hand_pointcloud"][index] = hand_cloud.transform(placements[tip_count:], transform)
    return arrays
