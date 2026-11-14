"""发布 HOI 标注，并按 Raw 连续区间对齐关节。"""

import json
from itertools import pairwise

import numpy as np

from dexmani_real.geometry.hoi import transform_task_points
from dexmani_real.utils.timing import interpolate_joints, validate_joint_gap


def write_hoi(
    path,
    template,
    poses,
    base_from_camera,
    *,
    quality,
    tracking_config,
    object_valid,
):
    import h5py

    poses = np.asarray(poses, dtype=np.float32)
    frames = len(poses)
    expected = (frames, len(template["object_names"]))
    if poses.shape != (*expected, 4, 4):
        raise ValueError("HOI pose shape mismatch")
    object_valid = np.asarray(object_valid)
    if object_valid.shape != expected or object_valid.dtype != bool:
        raise ValueError("HOI validity shape/dtype mismatch")
    if not np.isfinite(poses[object_valid]).all():
        raise ValueError("Nonfinite valid HOI pose")
    poses = poses.copy()
    poses[~object_valid] = np.nan
    valid = object_valid.all(axis=1)
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
        group.create_dataset("object_valid", data=object_valid)
        group.create_dataset("valid", data=valid)
        group.create_dataset("base_from_camera", data=base_from_camera)
        saved_template = group.create_group("template")
        for key, value in template.items():
            value = np.asarray(value)
            if value.dtype.kind == "U":
                value = value.astype(h5py.string_dtype("utf-8"))
            saved_template.create_dataset(key, data=value)
        quality_group = group.create_group("quality")
        for key in sorted({key for row in quality for obj in row for key in obj}):
            values = np.array(
                [[obj.get(key, np.nan) for obj in row] for row in quality], dtype=float
            )
            if not np.isfinite(values[object_valid]).all():
                raise ValueError(f"Nonfinite HOI quality: {key}")
            quality_group.create_dataset(key, data=values)
        cloud = group.create_dataset(
            "object_pointcloud", (frames, 1024, 3), dtype="f4", chunks=(1, 1024, 3)
        )
        for start in range(0, frames, 128):
            selected = valid[start : start + 128]
            points = np.zeros((len(selected), 1024, 3), np.float32)
            if selected.any():
                points[selected] = transform_task_points(
                    template, poses[start : start + 128][selected], base_from_camera
                )
            cloud[start : start + 128] = points
        data.flush()
        data.move("_hoi_staging", "hoi")
        data.flush()


def align_hoi_episode(reader, max_gap_s, *, devices=("arm", "hand")):
    """逐连续区间对齐关节，再按感知有效性拆分；关节内部缺口仍报错。"""
    validate_joint_gap(max_gap_s)
    timing = reader.read_timing()
    count = reader.num_frames
    for key in ("camera_ns", *(f"{device}_ns" for device in devices)):
        if np.any(timing[key] <= 0) or np.any(timing[key] > timing["observation_ns"]):
            raise ValueError(f"Invalid HOI source timestamp: {key}")
    keep = np.zeros(count, bool)
    joints = {
        f"{device}_qpos": np.full(reader[f"{device}_qpos"].shape, np.nan) for device in devices
    }
    boundaries = np.r_[reader.segment_starts, count]
    for start, end in pairwise(boundaries):
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
    keep &= read_hoi_validity(reader)
    ranges, starts = [], []
    offset = 0
    for start, end in pairwise(boundaries):
        transitions = np.diff(np.r_[False, keep[start:end], False].astype(int))
        for lo, hi in zip(
            np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        ):
            ranges.append([int(start + lo), int(start + hi)])
            starts.append(offset)
            offset += int(hi - lo)
    return {"keep": keep, "ranges": ranges, "starts": starts, **joints}


def read_hoi_validity(reader):
    """读取当前 HOI 契约的显式有效性，缺失字段由读取边界报错。"""
    data = reader.data_file
    valid = np.asarray(data["hoi/valid"])
    objects = np.asarray(data["hoi/object_valid"])
    count = len(data["hoi/template/object_names"])
    if (
        valid.dtype != bool
        or objects.dtype != bool
        or valid.shape != (reader.num_frames,)
        or objects.shape != (reader.num_frames, count)
    ):
        raise ValueError("Invalid HOI validity shape/dtype")
    if not np.array_equal(valid, objects.all(axis=1)):
        raise ValueError("Inconsistent HOI validity")
    return valid
