"""加载 XHand 的 link 局部点云模板，由关节 FK 变换到基座系。"""

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.perception import HandCloudConfig
from dexmani_real.geometry.kinematics import ArmFK, HandFK
from dexmani_real.robot.model import HAND_FINGERTIP_SHAPE

HAND_TEMPLATE_DIR = ASSET_DIR / "pointcloud_template" / "hand"


class HandPointCloud:
    def __init__(self, hand_fk: HandFK, num_points: int):
        HandCloudConfig(num_points=num_points).validate()
        path = HAND_TEMPLATE_DIR / f"hand_{num_points}.npz"
        try:
            with np.load(path, allow_pickle=False) as template:
                points = template["points"]
                names = template["part_names"]
                offsets = template["part_offsets"]
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Hand point-cloud template missing: {path}. Generate it with "
                f"python scripts/generate_hand_template.py --num-points {num_points}"
            ) from exc
        if (
            points.shape != (num_points, 6)
            or points.dtype != np.float32
            or not np.isfinite(points).all()
            or np.any((points[:, 3:] < 0) | (points[:, 3:] > 1))
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
        self.part_slices = tuple(slice(start, end) for start, end in zip(offsets[:-1], offsets[1:]))

    def transform(
        self, handbase_from_links: np.ndarray, base_from_handbase: np.ndarray,
    ) -> np.ndarray:
        transforms = base_from_handbase @ handbase_from_links
        cloud = self.local_points.copy()
        for part, transform in zip(self.part_slices, transforms):
            cloud[part, :3] = self.local_points[part, :3] @ transform[:3, :3].T + transform[:3, 3]
        return cloud


def hand_geometry(
    eef_transforms: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    arm_fk: ArmFK,
    hand_fk: HandFK,
    hand_cloud: HandPointCloud | None = None,
    include_fingertips: bool = True,
) -> dict[str, np.ndarray]:
    """使用组合 URDF 的安装变换生成指尖与手点云，每行共用一次手部 FK。"""
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

    for index, transform in enumerate(base_from_hand):
        placements = hand_fk.forward(hand_qpos[index], frame_ids)
        if include_fingertips:
            arrays["fingertip_xyz"][index] = (
                placements[:tip_count, :3, 3] @ transform[:3, :3].T + transform[:3, 3]
            )
        if hand_cloud is not None:
            arrays["hand_pointcloud"][index] = hand_cloud.transform(placements[tip_count:], transform)
    return arrays
