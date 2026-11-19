"""加载 XHand 的 link 局部点云模板，由关节 FK 变换到基座系。"""

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.perception import HandCloudConfig
from dexmani_real.geometry.kinematics import ArmFK, HandFK
from dexmani_real.geometry.sampling import validate_point_segments

HAND_TEMPLATE_DIR = ASSET_DIR / "pointcloud_template" / "hand"


class HandPointCloud:
    """读取 link 局部 XYZ 模板，保留分段及点序。"""

    def __init__(self, hand_fk: HandFK, num_points: int):
        HandCloudConfig(num_points=num_points).validate()
        path = HAND_TEMPLATE_DIR / f"hand_{num_points}.npz"
        with np.load(path, allow_pickle=False) as template:
            points = template["points"]
            names = template["part_names"]
            offsets = template["part_offsets"]
        validate_point_segments(
            points, names, offsets, num_points=num_points, label=f"Hand point-cloud template {path}",
        )
        self.frame_ids = tuple(hand_fk.model.getFrameId(name) for name in names.tolist())
        if any(frame_id >= hand_fk.model.nframes for frame_id in self.frame_ids):
            raise ValueError(
                f"Hand point-cloud template names a link missing from the FK model: {path}"
            )
        self.local_points = points
        self.part_offsets = offsets

    def transform_batch(self, base_from_links: np.ndarray) -> np.ndarray:
        """批量将 [T, links, 4, 4] 变换应用于固定点序的 link 局部点云。"""
        xyz = np.empty((len(base_from_links), len(self.local_points), 3), dtype=np.float32)
        # 模板已按 link 连续分段，无需为每个点复制一份 4×4 变换。
        for index, (start, end) in enumerate(zip(self.part_offsets[:-1], self.part_offsets[1:])):
            transform = base_from_links[:, index]
            segment = self.local_points[start:end] @ transform[:, :3, :3].swapaxes(1, 2)
            segment += transform[:, None, :3, 3]
            xyz[:, start:end] = segment
        return xyz


def hand_geometry(
    eef_transforms: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    arm_fk: ArmFK,
    hand_fk: HandFK,
    hand_cloud: HandPointCloud | None = None,
    include_fingertips: bool = True,
) -> dict[str, np.ndarray]:
    """由 [T,4,4] EEF 与 [T,12] SDK 关节批量生成基座系几何。

    按请求返回 float32 指尖 [T,5,3] 和手部表面点 [T,N,3]；允许 T=0。
    """
    if len(eef_transforms) != len(hand_qpos):
        raise ValueError("eef_transforms and hand_qpos must have the same batch length")

    frame_ids = list(hand_fk.fingertip_frame_ids) if include_fingertips else []
    tip_count = len(frame_ids)
    if hand_cloud is not None:
        frame_ids.extend(hand_cloud.frame_ids)
    arrays: dict[str, np.ndarray] = {}
    if not frame_ids:
        return arrays

    base_from_hand = eef_transforms @ arm_fk.eef_from_handbase
    placements = np.empty((len(base_from_hand), len(frame_ids), 4, 4), dtype=np.float64)
    for index, qpos in enumerate(hand_qpos):
        hand_fk.forward(qpos, frame_ids, out=placements[index])

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
