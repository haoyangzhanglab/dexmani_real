"""将 Raw 观测与绝对动作转换为训练用 Zarr。"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr

from dexmani_real.config.perception import (
    HOI_NUM_POINTS,
    HandCloudConfig,
    PointCloudConfig,
)
from dexmani_real.data.hoi import read_hoi_validity
from dexmani_real.data.raw import (
    RawDataError,
    RawReader,
    require_published_raw,
    validate_task_label,
)
from dexmani_real.geometry.camera import (
    camera_extrinsic_cv,
    depth_scales_match,
    init_undistort_maps,
    parse_camera_matrix,
    remap_image,
)
from dexmani_real.geometry.hoi import geometry_digest, validate_task_template
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_poses
from dexmani_real.geometry.pose import encode_eef_pose, validate_rigid_transform
from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
from dexmani_real.perception.pointcloud import PointCloudProcessor, processing_metadata
from dexmani_real.robot.model import (
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_TACTILE_SCALE_DIVISOR,
)
from dexmani_real.utils.io import publish
from dexmani_real.utils.timing import interpolate_joints, validate_joint_gap

CANONICAL_FORMAT = "dexmani.real.canonical"
logger = logging.getLogger(__name__)
PROCESSING_BYTES = 64 * 1024 * 1024
MAX_BATCH_FRAMES = 128
STORAGE_CHUNK_BYTES = 1024 * 1024
STORAGE_CHUNK_MAX_ROWS = 128


@dataclass(frozen=True)
class ExportConfig:
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    hand_pointcloud: HandCloudConfig = field(default_factory=HandCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = None
    hoi_max_joint_gap_s: float = 0.10


# 输出字段直接对应 Raw 输入；FK 和相机计算由下方普通条件启用。
FIELD_INPUTS = {
    "joint_state": ("arm_qpos", "hand_qpos"),
    "action": ("action_arm_joint_target", "action_hand_joint_target"),
    "action_ee": ("action_arm_joint_target", "action_hand_joint_target"),
    "eef_pose": ("arm_qpos",),
    "fingertip_xyz": ("arm_qpos", "hand_qpos"),
    "hand_pointcloud": ("arm_qpos", "hand_qpos"),
    "finger_tactile_force": ("hand_contact",),
    "taxel_tactile_force": ("hand_tactile_force",),
    "rgb": ("rgb",),
    "depth": ("depth",),
    "camera_intrinsic": (),
    "camera_extrinsic": (),
    "point_cloud": ("rgb", "depth"),
    "object_pointcloud": ("object_pointcloud",),
}

ARM_FK_FIELDS = {"action_ee", "eef_pose", "fingertip_xyz", "hand_pointcloud"}
HAND_FK_FIELDS = {"fingertip_xyz", "hand_pointcloud"}


def resolve_fields(fields):
    selected = set(fields)
    if not selected or selected - FIELD_INPUTS.keys():
        raise ValueError(f"fields must be nonempty and selected from {tuple(FIELD_INPUTS)}")
    return tuple(name for name in FIELD_INPUTS if name in selected)


def export_camera(reader, fields):
    """读取所选字段需要的相机参数，并为图像输出初始化映射。"""
    names = set(fields)
    camera: dict[str, Any] = {"parameters": {}}
    snapshot = None
    if names & {"rgb", "depth", "point_cloud"}:
        snapshot = reader.read_camera(
            include_depth=bool(names & {"depth", "point_cloud"}),
            include_extrinsics=bool(names & {"camera_extrinsic", "point_cloud"}),
        )
    if snapshot is not None:
        camera["snapshot"] = snapshot
        camera["shape"] = (snapshot.color.height, snapshot.color.width)
        if names & {"point_cloud", "camera_extrinsic"} and snapshot.T_xarm_base_from_color is None:
            raise ValueError("point_cloud/camera_extrinsic require camera extrinsics")
        if names & {"depth", "point_cloud"}:
            if snapshot.depth_scale_m is None:
                raise ValueError("depth and point_cloud require a camera depth scale")
            camera["depth_scale_m"] = snapshot.depth_scale_m
        if names & {"rgb", "depth"}:
            camera["image_maps"] = init_undistort_maps(snapshot.color)
    if "camera_intrinsic" in names:
        if snapshot is not None:
            matrix = snapshot.color.matrix(dtype=np.float32)
        else:
            matrix = parse_camera_matrix(reader.meta["camera_color_intrinsics"]).astype(np.float32)
        camera["parameters"]["camera_intrinsic"] = matrix.ravel()
    if "camera_extrinsic" in names:
        transform = snapshot.T_xarm_base_from_color if snapshot is not None else validate_rigid_transform(
            np.asarray(reader.meta["camera_T_xarm_base_from_color"]).reshape(4, 4),
            label="camera_T_xarm_base_from_color",
        )
        camera["parameters"]["camera_extrinsic"] = camera_extrinsic_cv(transform)
    return camera


def canonical_array_specs(fields, processing, camera):
    shapes = {
        "object_pointcloud": (HOI_NUM_POINTS, 3),
        "joint_state": (19,), "action": (19,), "action_ee": (21,),
        "finger_tactile_force": (5, 3), "taxel_tactile_force": (5, 120, 3),
        "fingertip_xyz": (5, 3), "eef_pose": (9,),
        "camera_intrinsic": (9,), "camera_extrinsic": (12,),
        "point_cloud": (processing.pointcloud.num_points, 6),
        "hand_pointcloud": (processing.hand_pointcloud.num_points, 3),
    }
    if "rgb" in fields:
        shapes["rgb"] = (*camera["shape"], 3)
    if "depth" in fields:
        shapes["depth"] = camera["shape"]
    return {name: (shapes[name], np.dtype({"rgb": np.uint8, "depth": np.uint16}.get(name, np.float32)))
            for name in fields}


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """枚举所有 episode 目录，损坏文件由读取边界直接报错。"""
    requested_path = Path(input_root).expanduser()
    root = requested_path.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (root / "data.h5").is_file():
        require_published_raw(requested_path)
        return (root,)
    episodes = tuple(
        sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith("episode_")
        )
    )
    if not episodes:
        raise FileNotFoundError(f"no episode directories found in {root}")
    return episodes


def iter_canonical_blocks(
    reader: RawReader,
    config: ExportConfig,
    *,
    fields,
    arm_fk,
    hand_fk,
    hand_cloud,
    camera,
    alignment,
):
    frames = reader.num_frames
    inputs = {name for field in fields for name in FIELD_INPUTS[field]}
    specs = canonical_array_specs(fields, config, camera)
    row_bytes = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in specs.values())
    if "point_cloud" in fields and "depth" not in fields:
        # 点云逐帧计算，但原始深度仍按块读取；额外计入读取预算。
        row_bytes += int(np.prod(camera["shape"])) * np.dtype(np.uint16).itemsize
    chunk_frames = min(MAX_BATCH_FRAMES, max(1, PROCESSING_BYTES // row_bytes))
    processor = None
    if "point_cloud" in fields:
        snapshot = camera["snapshot"]
        processor = PointCloudProcessor(
            depth_scale_m=snapshot.depth_scale_m, intrinsics=snapshot.color,
            T_xarm_base_from_color=snapshot.T_xarm_base_from_color,
            table_plane_abcd=config.table_plane_abcd, config=config.pointcloud,
        )
    # 视频容器归 RawReader 所有，退出 reader 上下文时统一关闭。
    images = reader.iter_rgb_frames() if "rgb" in inputs else None
    # 联合物体点云输出使用对齐观测，其余输出仍读取原始关节。
    sources = {
        name: alignment[name] if name in alignment else reader[name]
        for name in inputs - {"rgb"}
    }
    for start in range(0, frames, chunk_frames):
        end = min(frames, start + chunk_frames)
        count = end - start
        with raw_input():
            raw = {name: source[start:end] for name, source in sources.items()}
            for name, values in raw.items():
                if values.dtype.kind == "f":
                    valid = np.isfinite(values).reshape(count, -1).all(axis=1)
                    if not valid.all():
                        row = start + int(np.flatnonzero(~valid)[0])
                        raise RawDataError(f"{name}: nonfinite values at row {row}")
        if raw:
            del values
        block = {}
        if images is not None:
            height, width = camera["shape"]
            if "rgb" in fields:
                block["rgb"] = np.empty((count, height, width, 3), dtype=np.uint8)
            if processor is not None:
                block["point_cloud"] = np.empty(
                    (count, config.pointcloud.num_points, 6), dtype=np.float32
                )
            for offset, index in enumerate(range(start, end)):
                with raw_input():
                    raw_rgb = next(images, None)
                if raw_rgb is None:
                    raise RawDataError(f"{reader.path / 'rgb.mp4'}: RGB missing at row {index}")
                if "rgb" in fields:
                    block["rgb"][offset] = remap_image(
                        raw_rgb, camera["image_maps"], interpolation=cv2.INTER_LINEAR,
                    )
                if processor is not None:
                    cloud = processor.process(raw["depth"][offset], raw_rgb)
                    if not len(cloud):
                        raise RawDataError(f"point_cloud unavailable at row {index}")
                    block["point_cloud"][offset] = cloud
                    del cloud
                del raw_rgb
        if "object_pointcloud" in fields:
            block["object_pointcloud"] = raw["object_pointcloud"]
        if "joint_state" in fields:
            block["joint_state"] = np.concatenate((raw["arm_qpos"], raw["hand_qpos"]), axis=1, dtype=np.float32)
        if "action" in fields:
            block["action"] = np.concatenate((raw["action_arm_joint_target"], raw["action_hand_joint_target"]), axis=1, dtype=np.float32)
        if "action_ee" in fields:
            poses = eef_poses(raw["action_arm_joint_target"], arm_fk=arm_fk)
            block["action_ee"] = np.concatenate((encode_eef_pose(poses), raw["action_hand_joint_target"]), axis=1, dtype=np.float32)
        if {"eef_pose", "fingertip_xyz", "hand_pointcloud"}.intersection(fields):
            poses = eef_poses(raw["arm_qpos"], arm_fk=arm_fk)
            if "eef_pose" in fields:
                block["eef_pose"] = encode_eef_pose(poses).astype(np.float32)
            if hand_fk is not None:
                block.update(hand_geometry(
                    poses, raw["hand_qpos"], arm_fk=arm_fk, hand_fk=hand_fk,
                    hand_cloud=hand_cloud, include_fingertips="fingertip_xyz" in fields,
                ))
        for name, source in (("finger_tactile_force", "hand_contact"), ("taxel_tactile_force", "hand_tactile_force")):
            if name in fields:
                # Raw 已为 SDK / 255；导出只复制，不再次缩放。
                block[name] = raw[source].astype(np.float32, copy=False)
        for name, values in camera["parameters"].items():
            block[name] = np.broadcast_to(values, (count, values.size))
        if "depth" in fields:
            maps = camera["image_maps"]
            if maps is None:
                block["depth"] = raw["depth"]
            else:
                for index in range(count):
                    raw["depth"][index] = remap_image(raw["depth"][index], maps, interpolation=cv2.INTER_NEAREST)
                block["depth"] = raw["depth"]
        yield block
        del raw, block
    with raw_input():
        if images is not None and next(images, None) is not None:
            raise RawDataError(f"{reader.path / 'rgb.mp4'}: unexpected RGB frame at row {frames}")


@contextmanager
def raw_input():
    """仅包围输入读取和输入校验，不包含输出写入或模型计算。"""
    from av.error import EOFError as VideoEOFError
    from av.error import InvalidDataError

    try:
        yield
    except RawDataError:
        raise
    except (KeyError, ValueError, FileNotFoundError, InvalidDataError, VideoEOFError) as exc:
        raise RawDataError(str(exc)) from exc
    except OSError as exc:
        # HDF5 内容损坏；权限、磁盘等系统错误仍由调用方处理。
        if exc.errno is None and any(
            detail in str(exc).lower()
            for detail in (
                "file signature not found",
                "truncated file",
                "inflate() failed",
                "filter returned failure",
                "checksum",
            )
        ):
            raise RawDataError(str(exc)) from exc
        raise


def prepare_raw(reader, processing, fields):
    """验证完整 Raw，并准备低维关节对齐；RGB-D 仍在写入时单遍读取。"""
    with raw_input():
        validate_task_label(reader.meta["task_label"])
        if reader.num_frames <= 0:
            raise RawDataError("episode contains no rows")
        starts = reader.segment_starts
        timing = reader.read_timing()
        has_object = "object_pointcloud" in fields
        template = None
        if has_object:
            reader["object_pointcloud"]
            import h5py

            saved = reader.hoi_file["template"]
            if not isinstance(saved, h5py.Group) or any(
                not isinstance(saved.get(key), h5py.Dataset)
                for key in ("points_local", "offsets", "object_names")
            ):
                raise RawDataError("invalid HOI template datasets")
            if h5py.check_string_dtype(saved["object_names"].dtype) is None:
                raise RawDataError("HOI object_names must be strings")
            template = {key: saved[key][:] for key in ("points_local", "offsets")}
            template["object_names"] = saved["object_names"].asstr()[:].astype(str)
            validate_task_template(template)
            valid = read_hoi_validity(reader)
            if not valid.all():
                raise RawDataError(f"HOI contains invalid rows: first at {int(np.flatnonzero(~valid)[0])}")
        inputs = {name for field in fields for name in FIELD_INPUTS[field]}
        source_clocks = {
            "arm_qpos": "arm_ns",
            "hand_qpos": "hand_ns",
            "hand_contact": "tactile_aggregate_ns",
            "hand_tactile_force": "tactile_dense_ns",
            "rgb": "camera_ns",
            "depth": "camera_ns",
            "object_pointcloud": "camera_ns",
        }
        clocks = {source_clocks[name] for name in inputs if name in source_clocks}
        if has_object:
            clocks.add("camera_ns")
        for clock in clocks:
            stamps = timing[clock]
            if (
                np.any(stamps <= 0)
                or np.any(stamps > timing["observation_ns"])
                or np.any(np.diff(stamps) < 0)
            ):
                raise RawDataError(f"invalid source timestamp: {clock}")
        for name in inputs - {"rgb"}:
            reader[name]
        alignment = {}
        if has_object:
            for device in ("arm", "hand"):
                name = f"{device}_qpos"
                if name not in inputs:
                    continue
                source = reader[name][:]
                values = np.empty_like(source, dtype=np.float64)
                boundaries = np.r_[starts, reader.num_frames]
                for segment, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                    aligned, valid = interpolate_joints(
                        timing[f"{device}_ns"][start:stop], source[start:stop],
                        timing["camera_ns"][start:stop], processing.hoi_max_joint_gap_s,
                    )
                    if not valid.all():
                        row = int(start + np.flatnonzero(~valid)[0])
                        raise RawDataError(f"{device}: unalignable row {row} in segment {segment}")
                    values[start:stop] = aligned
                alignment[name] = values
        return template, alignment, export_camera(reader, fields), np.r_[starts, reader.num_frames]


def export_zarr(
    input_root, output_path, *, processing, fields, exclude=(), progress_callback=None
):
    selected_fields = resolve_fields(fields)
    if "object_pointcloud" in selected_fields and set(selected_fields) & {
        "joint_state", "eef_pose", "fingertip_xyz", "hand_pointcloud",
    }:
        validate_joint_gap(processing.hoi_max_joint_gap_s)
    if "point_cloud" in selected_fields:
        processing.pointcloud.validate()
        if processing.pointcloud.remove_table:
            plane = np.asarray(processing.table_plane_abcd, dtype=float)
            if (
                plane.shape != (4,)
                or not np.isfinite(plane).all()
                or np.linalg.norm(plane[:3]) == 0
            ):
                raise ValueError("remove_table=True requires a finite calibrated table plane")
    if "hand_pointcloud" in selected_fields:
        processing.hand_pointcloud.validate()
    target = Path(output_path).expanduser().absolute()
    if os.path.lexists(target):
        raise FileExistsError(target)
    selected, excluded = select_episodes(input_root, exclude)
    arm_fk = ArmFK() if ARM_FK_FIELDS.intersection(selected_fields) else None
    hand_fk = HandFK() if HAND_FK_FIELDS.intersection(selected_fields) else None
    hand_cloud = (
        HandPointCloud(hand_fk, processing.hand_pointcloud.num_points)
        if "hand_pointcloud" in selected_fields
        else None
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    root = arrays = None
    ends, sources, ranges, rejected = [], [], [], []
    source_count = 0
    shared_geometry = None
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            offset = ends[-1] if ends else 0
            try:
                with raw_input():
                    require_published_raw(episode)
                    reader = RawReader(episode)
                with reader:
                    template, alignment, camera, boundaries = prepare_raw(reader, processing, selected_fields)
                    names = selected_fields
                    with raw_input():
                        hz = reader.control_hz
                    if root is None:
                        root, arrays = create_store(
                            zarr.DirectoryStore(str(staging)), processing, camera,
                            fields=names, control_hz=hz, task_name=reader.meta["task_label"],
                            hand_cloud=hand_cloud,
                        )
                    else:
                        if reader.meta["task_label"] != root.attrs["task_name"] or not np.isclose(1 / hz, root.attrs["dt"], rtol=1e-6, atol=0):
                            raise RawDataError("task_label or dt mismatch")
                        if "depth" in names and not depth_scales_match(camera["depth_scale_m"], root.attrs["depth_scale_m_per_unit"]):
                            raise RawDataError("depth_scale_m_per_unit must be uniform")
                        specs = canonical_array_specs(names, processing, camera)
                        if any(arrays[name].shape[1:] != shape for name, (shape, _) in specs.items()):
                            raise RawDataError("inconsistent field shape")
                    blocks = iter_canonical_blocks(
                        reader, processing, fields=names, arm_fk=arm_fk, hand_fk=hand_fk,
                        hand_cloud=hand_cloud, camera=camera, alignment=alignment,
                    )
                    append_rows(arrays, offset, blocks, reader.num_frames)
                    digest = geometry_digest(template["points_local"], template["offsets"]) if template is not None else None
                # 整条输入与关闭成功后才提交来源、段边界及摘要。
                ends.extend((offset + boundaries[1:]).tolist())
                sources.extend([episode.name] * (len(boundaries) - 1))
                ranges.extend([[int(a), int(b)] for a, b in zip(boundaries[:-1], boundaries[1:])])
                if source_count == 0:
                    shared_geometry = digest
                elif shared_geometry != digest:
                    shared_geometry = None
                source_count += 1
            except RawDataError as exc:
                if arrays is not None:
                    for array in arrays.values():
                        array.resize((offset, *array.shape[1:]))
                    if not ends:
                        shutil.rmtree(staging)
                        staging.mkdir()
                        root = arrays = None
                rejected.append({"episode": episode.name, "reason": str(exc)})
                logger.warning("Rejected Raw %s: %s", episode, exc)
        if not ends:
            raise ValueError(f"all Raw episodes rejected: {rejected}")
        root.attrs["source_episodes"] = sources
        root.attrs["source_row_ranges"] = ranges
        if "object_pointcloud" in selected_fields:
            metadata = root.attrs["data_processing"]
            metadata["object_pointcloud"]["geometry_sha256"] = shared_geometry
            root.attrs["data_processing"] = metadata
        root.create_group("meta").create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
        publish(staging, target)
        if progress_callback:
            progress_callback(len(selected), len(selected))
        return {
            "path": str(target), "source_count": source_count, "episode_count": len(ends),
            "total_frames": ends[-1], "rejected_episodes": rejected, "excluded_episodes": sorted(excluded),
        }
    finally:
        failure = sys.exc_info()[1]
        try:
            if staging.exists():
                shutil.rmtree(staging)
        except OSError:
            if failure is None:
                raise
            logger.exception("export staging cleanup also failed: %s", staging)


def create_store(store, processing, camera, *, fields, control_hz, task_name, hand_cloud):
    specs = canonical_array_specs(fields, processing, camera)
    root = zarr.group(store=store, overwrite=True)
    root.attrs.update(
        format=CANONICAL_FORMAT,
        dt=1.0 / control_hz,
        task_name=task_name,
        action_space="joint",
    )
    if {"finger_tactile_force", "taxel_tactile_force"}.intersection(fields):
        root.attrs["tactile_scale_divisor"] = XHAND_TACTILE_SCALE_DIVISOR
    if {"action", "action_ee"}.intersection(fields):
        root.attrs["action_semantics"] = "absolute_joint_goal_not_next_step_arrival"
    if "point_cloud" in fields and processing.pointcloud.remove_table:
        root.attrs["table_plane_abcd"] = processing.table_plane_abcd
    if "fingertip_xyz" in fields:
        root.attrs["fingertip_link_names"] = list(XHAND_FINGERTIP_LINK_NAMES)
    if {"rgb", "depth"}.intersection(fields):
        root.attrs["image_geometry"] = "pinhole"
    if "depth" in fields:
        root.attrs["depth_scale_m_per_unit"] = camera["depth_scale_m"]
    metadata = {}
    if "point_cloud" in fields:
        metadata["point_cloud"] = processing_metadata(processing.pointcloud)
    if "hand_pointcloud" in fields:
        metadata["hand_pointcloud"] = {
            "num_points": processing.hand_pointcloud.num_points,
            "geometry_sha256": hand_cloud.geometry_sha256,
        }
    if "object_pointcloud" in fields:
        metadata["object_pointcloud"] = {"geometry_sha256": None}
        if set(fields) & {"joint_state", "eef_pose", "fingertip_xyz", "hand_pointcloud"}:
            metadata["object_pointcloud"]["max_joint_gap_s"] = processing.hoi_max_joint_gap_s
    root.attrs["data_processing"] = metadata
    data = root.create_group("data")
    arrays = {}
    for name, (shape, dtype) in specs.items():
        row_bytes = int(np.prod(shape)) * dtype.itemsize
        storage_rows = max(1, min(STORAGE_CHUNK_MAX_ROWS, STORAGE_CHUNK_BYTES // row_bytes))
        arrays[name] = data.create_dataset(
            name,
            shape=(0, *shape),
            chunks=(storage_rows, *shape),
            dtype=dtype,
            compressor=zarr.get_codec({"id": "zstd", "level": 3}),
        )
    return root, arrays


def append_rows(arrays, offset, blocks, frames):
    for array in arrays.values():
        array.resize((offset + frames, *array.shape[1:]))
    count = 0
    for block in blocks:
        if not block or block.keys() != arrays.keys():
            raise ValueError("canonical block must contain exactly the selected fields")
        rows = len(next(iter(block.values())))
        if any(len(values) != rows for values in block.values()):
            raise ValueError("canonical fields must have equal row counts")
        for name, values in block.items():
            if np.issubdtype(values.dtype, np.floating):
                valid = np.isfinite(values).reshape(rows, -1).all(axis=1)
                if not valid.all():
                    row = count + int(np.flatnonzero(~valid)[0])
                    raise RawDataError(f"canonical {name}: nonfinite values at row {row}")
            arrays[name][offset + count : offset + count + rows] = values
        count += rows
        # 同步写入完成后释放消费者引用，避免与下一块分配重叠。
        del block, values
    if count != frames:
        raise ValueError(f"canonical row count mismatch: expected {frames}, wrote {count}")


def select_episodes(source, exclude):
    episodes = discover_episode_dirs(source)
    excluded = set(exclude)
    unknown = excluded - {episode.name for episode in episodes}
    if unknown:
        raise ValueError(f"unknown excluded episode(s): {sorted(unknown)}")
    selected = [episode for episode in episodes if episode.name not in excluded]
    if not selected:
        raise ValueError("export has no selected episodes")
    return selected, excluded
