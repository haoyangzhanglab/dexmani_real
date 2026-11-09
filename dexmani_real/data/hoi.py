"""Raw 内 HOI 分组的校验和分块发布；不改变采集格式。"""

import numpy as np

from dexmani_real.geometry.hoi import transform_task_points

HOI_FIELDS = (
    "object_pointcloud",
    "object_pointcloud_valid",
    "object_pose_camera",
    "object_pose_valid",
    "model_supported_fraction",
)


def validate_hoi(group, frames):
    """打开文件时检查完成状态和数组结构；数值在写入、显示或导出时检查。"""
    if not group.attrs["complete"]:
        raise ValueError("Raw /hoi 未完成，不能作为正式标注读取")
    names = group.attrs["object_names"].tolist()
    if not names or len(set(names)) != len(names):
        raise ValueError("Raw /hoi 对象顺序非法")
    offsets = np.asarray(group.attrs["offsets"])
    if (
        offsets.dtype.kind not in "iu"
        or offsets.shape != (len(names) + 1,)
        or offsets[0] != 0
        or offsets[-1] != 1024
        or np.any(np.diff(offsets) < 64)
    ):
        raise ValueError("Raw /hoi 点分段非法")
    count = len(names)
    specs = [
        ((frames, 1024, 3), np.float32),
        ((frames,), np.bool_),
        ((frames, count, 4, 4), np.float32),
        ((frames, count), np.bool_),
        ((frames, count), np.float32),
    ]
    for name, (shape, dtype) in zip(HOI_FIELDS, specs):
        if group[name].shape != shape or group[name].dtype != dtype:
            raise ValueError(f"Raw /hoi/{name}: shape/dtype 错误")
    return {
        "task_name": group.attrs["task_name"],
        "object_names": names,
        "template_sha256": group.attrs["template_sha256"],
    }


def write_hoi(path, template, manifest, poses, valid, quality, base_from_camera):
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
        group.attrs.update(manifest, offsets=template["offsets"], complete=False)
        cloud = group.create_dataset(
            "object_pointcloud", (frames, 1024, 3), dtype="f4", chunks=(1, 1024, 3)
        )
        frame_valid = valid.all(axis=1)
        group.create_dataset("object_pointcloud_valid", data=frame_valid)
        poses = np.array(poses, dtype=np.float32, copy=True)
        poses[~valid] = np.nan
        group.create_dataset("object_pose_camera", data=poses)
        group.create_dataset("object_pose_valid", data=valid)
        group.create_dataset("model_supported_fraction", data=np.asarray(quality, np.float32))
        for start in range(0, frames, 128):
            block = np.zeros((min(128, frames - start), 1024, 3), np.float32)
            for offset in range(len(block)):
                index = start + offset
                if frame_valid[index]:
                    block[offset] = transform_task_points(template, poses[index], base_from_camera)
            cloud[start : start + len(block)] = block
        group.attrs["complete"] = True
        validate_hoi(group, frames)
        data.flush()
        data.move("_hoi_staging", "hoi")
        data.flush()
