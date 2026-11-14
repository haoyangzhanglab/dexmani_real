"""加载 XHand 的 link 局部点云模板，由关节 FK 变换到基座系。"""

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.geometry.kinematics import ArmFK, HandFK
from dexmani_real.robot.model import HAND_FINGERTIP_SHAPE

HAND_TEMPLATE_DIR = ASSET_DIR / "pointcloud_template" / "hand"


class HandPointCloud:
    def __init__(self, hand_fk: HandFK, num_points: int):
        path = HAND_TEMPLATE_DIR / f"hand_{num_points}.npz"
        with np.load(path, allow_pickle=False) as template:
            points = template["points"][:, :3]
            names = template["part_names"]
            offsets = template["part_offsets"]
        if (
            points.shape != (num_points, 3)
            or points.dtype != np.float32
            or not np.isfinite(points).all()
            or names.ndim != 1
            or names.dtype.kind != "U"
            or not len(names)
            or len(set(names)) != len(names)
            or offsets.shape != (len(names) + 1,)
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != num_points
            or np.any(offsets[1:] <= offsets[:-1])
        ):
            raise ValueError(f"Invalid hand point-cloud template: {path}")
        self.frame_ids = tuple(hand_fk.model.getFrameId(name) for name in names.tolist())
        if any(frame_id >= hand_fk.model.nframes for frame_id in self.frame_ids):
            raise ValueError(f"Hand point-cloud template names a link missing from the FK model: {path}")
        self.local_points = points
        self.point_frame_indices = np.repeat(
            np.arange(len(self.frame_ids), dtype=np.intp),
            np.diff(offsets),
        )

    def transform_batch(self, base_from_links: np.ndarray) -> np.ndarray:
        """批量将 [T, links, 4, 4] 变换应用于固定点序的 link 局部点云。"""
        point_transforms = np.take(base_from_links, self.point_frame_indices, axis=1)
        xyz = np.matmul(
            point_transforms[..., :3, :3],
            self.local_points[None, :, :, None],
        )[..., 0]
        xyz += point_transforms[..., :3, 3]
        return xyz.astype(np.float32, copy=False)

    def transform(
        self, handbase_from_links: np.ndarray, base_from_handbase: np.ndarray,
    ) -> np.ndarray:
        """单帧接口；与批量路径共用同一变换实现。"""
        transforms = base_from_handbase @ handbase_from_links
        return self.transform_batch(transforms[None])[0]


def hand_geometry(
    eef_transforms: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    arm_fk: ArmFK,
    hand_fk: HandFK,
    hand_cloud: HandPointCloud | None = None,
    include_fingertips: bool = True,
) -> dict[str, np.ndarray]:
    """一次收集批次 FK，再批量变换指尖与手部表面点。"""
    base_from_hand = eef_transforms @ arm_fk.eef_from_handbase

    frame_ids = list(hand_fk.fingertip_frame_ids) if include_fingertips else []
    tip_count = len(frame_ids)
    if hand_cloud is not None:
        frame_ids.extend(hand_cloud.frame_ids)
    arrays = {}
    if include_fingertips:
        arrays["fingertip_xyz"] = np.empty((len(eef_transforms), *HAND_FINGERTIP_SHAPE), dtype=np.float32)
    if hand_cloud is not None:
        arrays["hand_pointcloud"] = np.empty(
            (len(eef_transforms), *hand_cloud.local_points.shape), dtype=np.float32
        )

    placements = np.empty((len(base_from_hand), len(frame_ids), 4, 4), dtype=np.float64)
    for index, qpos in enumerate(hand_qpos):
        placements[index] = hand_fk.forward(qpos, frame_ids)

    if include_fingertips:
        tip_positions = placements[:, :tip_count, :3, 3]
        arrays["fingertip_xyz"] = (
            np.matmul(base_from_hand[:, None, :3, :3], tip_positions[..., None])[..., 0]
            + base_from_hand[:, None, :3, 3]
        ).astype(np.float32, copy=False)
    if hand_cloud is not None:
        base_from_links = base_from_hand[:, None] @ placements[:, tip_count:]
        arrays["hand_pointcloud"] = hand_cloud.transform_batch(base_from_links)
    return arrays
