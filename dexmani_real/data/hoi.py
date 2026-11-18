"""发布 HOI 标注，并按 Raw 连续区间对齐关节。"""

import json
import logging
import os
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np

from dexmani_real.config.perception import HOI_NUM_POINTS
from dexmani_real.data.raw import RawDataError, RawReader
from dexmani_real.geometry.hoi import transform_task_points
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.utils.io import publish
from dexmani_real.utils.timing import interpolate_joints

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

    base_from_camera = validate_rigid_transform(base_from_camera, label="HOI base_from_camera")
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


def align_hoi_episode(
    reader: RawReader, timing: np.ndarray, boundaries: np.ndarray, max_gap_s: float,
    *, devices: tuple[str, ...] = ("arm", "hand"), require_complete: bool = False,
) -> dict[str, np.ndarray]:
    """对齐已校验的时间前缀；只在同一原生段内读取所需邻点。"""
    count = len(timing)
    joints = {
        f"{device}_qpos": np.empty((count, *reader[f"{device}_qpos"].shape[1:]))
        for device in devices
    }
    for segment, (start, end) in enumerate(pairwise(boundaries)):
        if start >= count:
            break
        selected_end = min(end, count)
        targets = timing["camera_ns"][start:selected_end]
        gaps = []
        for device in devices:
            name, clock = f"{device}_qpos", f"{device}_ns"
            stamps = timing[clock][start:selected_end]
            values = reader[name][start:selected_end]
            neighbors: list[tuple[int, int]] = []
            index, stamp = int(selected_end) - 1, int(stamps[-1])
            # 顺序查找尾部时钟，只保留各目标真正需要的邻点；长时间重复
            # 反馈或缺少右邻点时，也不加载整段尾部 qpos。
            for target in np.unique(targets[targets > stamp]):
                while index + 1 < end and stamp < target:
                    left_index, left_stamp = index, stamp
                    index += 1
                    row = reader["timing/rows"][index]
                    stamp = int(row[clock])
                    if stamp <= 0 or stamp < left_stamp or stamp > row["observation_ns"]:
                        raise ValueError(f"Invalid HOI source timestamp: {clock}")
                if stamp < target:
                    break
                needed = (
                    ((index, stamp),) if stamp == target
                    else ((left_index, left_stamp), (index, stamp))
                )
                for neighbor in needed:
                    if neighbor[0] >= selected_end and (not neighbors or neighbor[0] > neighbors[-1][0]):
                        neighbors.append(neighbor)
            if neighbors:
                indices, extra_stamps = zip(*neighbors)
                stamps = np.r_[stamps, extra_stamps]
                values = np.concatenate((values, reader[name][list(indices)]))
            positions, valid = interpolate_joints(stamps, values, targets, max_gap_s)
            joints[name][start:selected_end] = positions
            if require_complete and not valid.all():
                gaps.append((start + np.flatnonzero(~valid)[0], device))
        if gaps:
            row, device = min(gaps)
            raise RawDataError(f"{device}: unalignable row {row} in segment {segment}")
    return joints


def read_hoi_validity(reader: RawReader, stop: int | None = None) -> np.ndarray:
    """整帧有效性唯一来自所选范围内的逐物体有效性。"""
    import h5py

    data = reader.hoi_file
    objects = data["object_valid"]
    names = data["template/object_names"]
    if not isinstance(names, h5py.Dataset) or names.ndim != 1 or not len(names):
        raise RawDataError("HOI template/object_names must be a nonempty one-dimensional dataset")
    if (
        not isinstance(objects, h5py.Dataset)
        or objects.dtype != bool
        or objects.shape != (reader.num_frames, names.shape[0])
    ):
        raise RawDataError("Invalid HOI object_valid shape/dtype")
    return objects[:stop].all(axis=1)
