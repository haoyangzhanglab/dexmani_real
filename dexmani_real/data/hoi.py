"""将物体点云分块写入 Raw。"""

from dexmani_real.geometry.hoi import transform_task_points


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
        cloud = group.create_dataset(
            "object_pointcloud", (frames, 1024, 3), dtype="f4", chunks=(1, 1024, 3)
        )
        for start in range(0, frames, 128):
            cloud[start : start + 128] = transform_task_points(
                template, poses[start : start + 128], base_from_camera
            )
        data.flush()
        data.move("_hoi_staging", "hoi")
        data.flush()
