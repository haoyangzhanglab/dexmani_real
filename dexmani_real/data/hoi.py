"""Raw HOI 的结构校验和分块发布。"""

import numpy as np

from dexmani_real.geometry.hoi import transform_task_points

HOI_FIELDS = ("object_pointcloud", "object_pointcloud_valid")


def validate_hoi(group, frames):
    for name, shape, dtype in (
        ("object_pointcloud", (frames, 1024, 3), np.float32),
        ("object_pointcloud_valid", (frames,), np.bool_),
    ):
        if group[name].shape != shape or group[name].dtype != dtype:
            raise ValueError(f"Raw /hoi/{name}: shape/dtype 错误")


def write_hoi(path, template, poses, base_from_camera):
    import h5py

    frames = len(poses)
    with h5py.File(path, "r+") as data:
        if len(data["timing/rows"]) != frames:
            raise ValueError("HOI 与 Raw 帧数不一致")
        if "hoi" in data:
            raise FileExistsError(f"拒绝覆盖已有 /hoi：{path}")
        if "_hoi_staging" in data:
            del data["_hoi_staging"]
        group = data.create_group("_hoi_staging")
        cloud = group.create_dataset("object_pointcloud", (frames, 1024, 3), dtype="f4", chunks=(1, 1024, 3))
        for start in range(0, frames, 128):
            cloud[start:start + 128] = np.stack([
                transform_task_points(template, pose, base_from_camera)
                for pose in poses[start:start + 128]
            ])
        group.create_dataset("object_pointcloud_valid", data=np.ones(frames, dtype=bool))
        data.flush()
        data.move("_hoi_staging", "hoi")
        data.flush()
