"""发布 HOI 标注，并按 Raw 连续区间对齐关节。"""

import json

import numpy as np

from dexmani_real.geometry.hoi import transform_task_points
from dexmani_real.utils.timing import interpolate_joints, validate_joint_gap


def write_hoi(path, template, poses, base_from_camera, *, quality, tracking_config):
    import h5py

    poses = np.asarray(poses, dtype=np.float32)
    frames = len(poses)
    if (
        frames == 0
        or len(quality) != frames
        or any(len(row) != len(template["object_names"]) for row in quality)
    ):
        raise ValueError("HOI quality shape does not match poses/template")
    with h5py.File(path, "r+") as data:
        if len(data["timing/rows"]) != frames:
            raise ValueError("HOI 与 Raw 帧数不一致")
        if "hoi" in data:
            raise FileExistsError(f"拒绝覆盖已有 /hoi：{path}")
        if "_hoi_staging" in data:
            del data["_hoi_staging"]
        group = data.create_group("_hoi_staging")
        group.attrs["tracking_config"] = json.dumps(tracking_config)
        group.create_dataset("object_pose_camera", data=poses)
        group.create_dataset("base_from_camera", data=base_from_camera)
        saved_template = group.create_group("template")
        for key, value in template.items():
            value = np.asarray(value)
            if value.dtype.kind == "U":
                value = value.astype(h5py.string_dtype("utf-8"))
            saved_template.create_dataset(key, data=value)
        quality_group = group.create_group("quality")
        for key in quality[0][0]:
            values = np.array([[obj[key] for obj in row] for row in quality])
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite HOI quality: {key}")
            quality_group.create_dataset(key, data=values)
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


def align_hoi_episode(reader, max_gap_s, *, devices=("arm", "hand")):
    """逐连续区间对齐；只裁首尾，内部缺口拒绝导出。"""
    validate_joint_gap(max_gap_s)
    timing = reader.read_timing()
    count = reader.num_frames
    for key in ("camera_ns", *(f"{device}_ns" for device in devices)):
        if np.any(timing[key] <= 0) or np.any(timing[key] > timing["observation_ns"]):
            raise ValueError(f"Invalid HOI source timestamp: {key}")
    keep = np.zeros(count, bool)
    ranges, starts = [], []
    joints = {
        f"{device}_qpos": np.full(reader[f"{device}_qpos"].shape, np.nan) for device in devices
    }
    boundaries = np.r_[reader.segment_starts, count]
    offset = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        valid = np.ones(end - start, bool)
        for device in devices:
            qpos_key = f"{device}_qpos"
            positions, device_valid = interpolate_joints(
                timing[f"{device}_ns"][start:end],
                reader[qpos_key][start:end],
                timing["camera_ns"][start:end],
                max_gap_s,
            )
            joints[qpos_key][start:end] = positions
            valid &= device_valid
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError(f"HOI segment {start}:{end} has no alignable rows")
        lo, hi = int(indices[0]), int(indices[-1]) + 1
        if not valid[lo:hi].all():
            raise ValueError(f"HOI segment {start}:{end} has an internal joint gap")
        keep[start + lo : start + hi] = True
        ranges.append([int(start + lo), int(start + hi)])
        starts.append(offset)
        offset += hi - lo
    return {"keep": keep, "ranges": ranges, "starts": starts, **joints}
