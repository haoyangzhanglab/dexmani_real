"""发布 HOI 标注，并按 Raw 连续区间对齐关节。"""

import json
import logging
import os
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np

from dexmani_real.config.perception import HOI_NUM_POINTS
from dexmani_real.geometry.hoi import transform_task_points
from dexmani_real.utils.io import publish
from dexmani_real.utils.timing import interpolate_joints, validate_joint_gap

logger = logging.getLogger(__name__)


def write_hoi(
    path,
    template,
    poses,
    base_from_camera,
    *,
    quality,
    tracking_config,
    object_valid,
    overwrite=False,
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
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        os.close(fd)
        with h5py.File(temporary, "w") as data:
            data.attrs["tracking_config"] = json.dumps(tracking_config)
            data.create_dataset("object_pose_camera", data=poses)
            data.create_dataset("object_valid", data=object_valid)
            data.create_dataset("valid", data=valid)
            data.create_dataset("base_from_camera", data=base_from_camera)
            saved_template = data.create_group("template")
            for key, value in template.items():
                value = np.asarray(value)
                if value.dtype.kind == "U":
                    value = value.astype(h5py.string_dtype("utf-8"))
                saved_template.create_dataset(key, data=value)
            quality_group = data.create_group("quality")
            for key in sorted({key for row in quality for obj in row for key in obj}):
                values = np.array(
                    [[obj.get(key, np.nan) for obj in row] for row in quality], dtype=float
                )
                quality_group.create_dataset(key, data=values)
            cloud = data.create_dataset(
                "object_pointcloud", (frames, HOI_NUM_POINTS, 3), dtype="f4",
                chunks=(1, HOI_NUM_POINTS, 3),
            )
            for start in range(0, frames, 128):
                selected = valid[start : start + 128]
                points = np.zeros((len(selected), HOI_NUM_POINTS, 3), np.float32)
                if selected.any():
                    points[selected] = transform_task_points(
                        template, poses[start : start + 128][selected], base_from_camera
                    )
                cloud[start : start + 128] = points
        if overwrite:
            os.replace(temporary, path)
        else:
            publish(temporary, path)
    except BaseException:
        try:
            Path(temporary).unlink(missing_ok=True)
        except BaseException:
            logger.exception("HOI temporary file cleanup also failed: %s", temporary)
        raise
    return path


def align_hoi_episode(reader, max_gap_s, *, devices=("arm", "hand")):
    """为可视化对齐各原始区间；无法对齐的行保留 NaN，不筛行或生成 episode。"""
    validate_joint_gap(max_gap_s)
    timing = reader.read_timing()
    for key in ("camera_ns", *(f"{device}_ns" for device in devices)):
        if np.any(timing[key] <= 0) or np.any(timing[key] > timing["observation_ns"]):
            raise ValueError(f"Invalid HOI source timestamp: {key}")
    joints = {
        f"{device}_qpos": np.empty(reader[f"{device}_qpos"].shape) for device in devices
    }
    boundaries = np.r_[reader.segment_starts, reader.num_frames]
    for start, end in pairwise(boundaries):
        for device in devices:
            name = f"{device}_qpos"
            positions, _ = interpolate_joints(
                timing[f"{device}_ns"][start:end], reader[name][start:end],
                timing["camera_ns"][start:end], max_gap_s,
            )
            joints[name][start:end] = positions
    return joints


def read_hoi_validity(reader):
    """读取显式有效性，缺失字段由读取边界报错。"""
    data = reader.hoi_file
    valid = np.asarray(data["valid"])
    objects = np.asarray(data["object_valid"])
    count = len(data["template/object_names"])
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
