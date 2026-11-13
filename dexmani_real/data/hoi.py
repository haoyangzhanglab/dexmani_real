"""将物体点云分块写入 Raw。"""

from dexmani_real.geometry.hoi import transform_task_points, validate_template


def write_hoi(path, template, poses, base_from_camera):
    import h5py

    validate_template(template)
    frames = len(poses)
    with h5py.File(path, "r+") as data:
        if len(data["timing/rows"]) != frames:
            raise ValueError("HOI 与 Raw 帧数不一致")
        if "hoi" in data:
            raise FileExistsError(f"拒绝覆盖已有 /hoi：{path}")
        if "_hoi_staging" in data:
            del data["_hoi_staging"]
        group = data.create_group("_hoi_staging")
        # 保存实际参与标注的模板，不能在导出时从可能已重采样的资产反推点身份。
        saved = group.create_group("template")
        saved.create_dataset("points_local", data=template["points_local"])
        saved.create_dataset("offsets", data=template["offsets"])
        saved.create_dataset(
            "object_names", data=template["object_names"].tolist(),
            dtype=h5py.string_dtype("utf-8"),
        )
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
