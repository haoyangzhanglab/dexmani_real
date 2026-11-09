"""将 Raw 观测与绝对动作转换为训练用 Zarr。"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr

from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.data.raw import (
    TACTILE_AGE_WARNING_NS,
    RawReader,
    require_continuous_recording,
    require_published_raw,
)
from dexmani_real.geometry.camera import (
    camera_extrinsic_cv,
    depth_scales_match,
    parse_camera_matrix,
)
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_poses
from dexmani_real.geometry.pose import encode_eef_pose, validate_rigid_transform
from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
from dexmani_real.perception.pointcloud import PointCloudProcessor
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.utils.io import publish

CANONICAL_FORMAT = "dexmani.real.canonical"
logger = logging.getLogger(__name__)
PROCESSING_BYTES = 64 * 1024 * 1024
MAX_BATCH_FRAMES = 100
STORAGE_CHUNK_BYTES = 1024 * 1024
STORAGE_CHUNK_MAX_ROWS = 128


@dataclass(frozen=True)
class ExportConfig:
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    hand_pointcloud: HandCloudConfig = field(default_factory=HandCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = None


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
}
DEFAULT_FIELDS = tuple(FIELD_INPUTS)
FIELD_INPUTS["object_pointcloud"] = ("object_pointcloud",)

SOURCE_FIELDS = {
    "arm_ns": {"joint_state", "eef_pose", "fingertip_xyz", "hand_pointcloud"},
    "hand_ns": {"joint_state", "fingertip_xyz", "hand_pointcloud"},
    "tactile_aggregate_ns": {"finger_tactile_force"},
    "tactile_dense_ns": {"taxel_tactile_force"},
    "camera_ns": {"rgb", "depth", "point_cloud", "object_pointcloud"},
}
ARM_FK_FIELDS = {"action_ee", "eef_pose", "fingertip_xyz", "hand_pointcloud"}
HAND_FK_FIELDS = {"fingertip_xyz", "hand_pointcloud"}


def resolve_fields(fields):
    selected = set(DEFAULT_FIELDS if fields is None else fields)
    if not selected or selected - FIELD_INPUTS.keys():
        raise ValueError(f"fields must be nonempty and selected from {tuple(FIELD_INPUTS)}")
    return tuple(name for name in FIELD_INPUTS if name in selected)


def export_camera(reader, fields):
    """只读取所选输出实际使用的相机参数，不补造完整相机快照。"""
    names = set(fields)
    camera: dict[str, Any] = {"parameters": {}}
    snapshot = reader.read_camera() if "point_cloud" in names else None
    if snapshot is not None:
        if snapshot.T_xarm_base_from_color is None:
            raise ValueError("point_cloud requires camera extrinsics")
        camera["snapshot"] = snapshot
    if names & {"rgb", "depth", "point_cloud"}:
        if snapshot is not None:
            shape = (snapshot.color.height, snapshot.color.width)
        else:
            shape = tuple(
                np.asarray(reader.meta[f"camera_color_{axis}"]).item()
                for axis in ("height", "width")
            )
            if any(type(value) is not int or value <= 0 for value in shape):
                raise ValueError("camera image dimensions must be positive integers")
        camera["shape"] = shape
    if names & {"depth", "point_cloud"}:
        if snapshot is not None:
            scale = snapshot.depth_scale_m
        else:
            scale = float(reader.meta["depth_scale"])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("depth_scale must be finite and positive")
        camera["depth_scale_m"] = scale
        expected = (reader.num_frames, *camera["shape"])
        depth = reader["depth"]
        if depth.shape != expected or depth.dtype != np.uint16:
            raise ValueError(f"depth must be uint16 {expected}")
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
        "object_pointcloud": (1024, 3),
        "joint_state": (19,), "action": (19,), "action_ee": (21,),
        "finger_tactile_force": (5, 3), "taxel_tactile_force": (5, 120, 3),
        "fingertip_xyz": (5, 3), "eef_pose": (9,),
        "camera_intrinsic": (9,), "camera_extrinsic": (12,),
        "point_cloud": (processing.pointcloud.num_points, 6),
        "hand_pointcloud": (processing.hand_pointcloud.num_points, 6),
    }
    if "rgb" in fields:
        shapes["rgb"] = (*camera["shape"], 3)
    if "depth" in fields:
        shapes["depth"] = camera["shape"]
    return {name: (shapes[name], np.dtype({"rgb": np.uint8, "depth": np.uint16}.get(name, np.float32)))
            for name in fields}


def validate_export_episode(reader: RawReader, fields):
    """校验所用输入与基本控制时间，来源时钟只作定位诊断。"""
    frames = reader.num_frames
    if frames <= 0:
        raise ValueError("episode contains no rows")
    inputs = {name for field in fields for name in FIELD_INPUTS[field]}
    reader.require_fields(*inputs)
    for name in inputs - {"rgb"}:
        reader[name]
    camera = export_camera(reader, fields)
    rows = reader["timing/rows"][:]
    require_continuous_recording(reader.meta, rows)
    period = round(1e9 / reader.control_hz)

    def first_bad(bad):
        indices = np.flatnonzero(bad)
        return int(indices[0]) if len(indices) else None

    control, observation = rows["control_ns"], rows["observation_ns"]
    if frames == 1:
        logger.warning("%s: single row, no sampling interval to verify", reader.path)
    if np.any(np.abs(np.diff(control) - period) > period // 4):
        logger.warning("%s: short control intervals; 不重采样，实际时间保留于 Raw", reader.path)
    if np.any(observation[:-1] >= control[1:]):
        logger.warning("%s: observation readout overlaps the next control sample", reader.path)

    diagnostics = []
    for name, used_by in SOURCE_FIELDS.items():
        if not used_by.intersection(fields):
            continue
        values = rows[name]
        bad = (values <= 0) | (values > observation)
        bad[1:] |= np.diff(values) < 0
        row = first_bad(bad)
        if row is not None:
            diagnostics.append(f"{name} invalid/regressed at row {row}")
        if name.startswith("tactile_"):
            row = first_bad((values > 0) & (observation - values > TACTILE_AGE_WARNING_NS))
            if row is not None:
                diagnostics.append(f"{name} held beyond {TACTILE_AGE_WARNING_NS / 1e9:g}s at row {row}")
    if SOURCE_FIELDS["camera_ns"].intersection(fields):
        sequence = rows["camera_sequence"]
        bad = sequence <= 0
        bad[1:] |= np.diff(sequence) < 0
        row = first_bad(bad)
        if row is not None:
            diagnostics.append(f"camera_sequence invalid/regressed at row {row}")
    if diagnostics:
        logger.warning("%s: %s; 保留全部原始行，不修补来源时间", reader.path, "; ".join(diagnostics))
    return camera


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
    reader: RawReader, config: ExportConfig, *, fields, arm_fk, hand_fk,
    hand_cloud, camera,
):
    frames = reader.num_frames
    inputs = {name for field in fields for name in FIELD_INPUTS[field]}
    specs = canonical_array_specs(fields, config, camera)
    row_bytes = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in specs.values())
    if "point_cloud" in fields and "depth" not in fields:
        # 点云逐帧计算，但原始深度仍按块读取；图像尺寸可以逐段变化。
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
    for start in range(0, frames, chunk_frames):
        end = min(frames, start + chunk_frames)
        count = end - start
        raw = {name: reader[name][start:end] for name in inputs - {"rgb"}}
        block = {}
        if "object_pointcloud" in fields:
            valid = reader["object_pointcloud_valid"][start:end]
            valid &= np.isfinite(raw["object_pointcloud"]).all(axis=(1, 2))
            if not valid.all():
                row = start + int(np.flatnonzero(~valid)[0])
                raise ValueError(f"object_pointcloud invalid at row {row}; use --exclude {reader.path.name}")
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
                block[name] = raw[source].astype(np.float32, copy=False)
        for name, values in camera["parameters"].items():
            block[name] = np.broadcast_to(values, (count, values.size))
        if "depth" in fields:
            block["depth"] = raw["depth"]
        if images is not None:
            height, width = camera["shape"]
            if "rgb" in fields:
                block["rgb"] = np.empty((count, height, width, 3), dtype=np.uint8)
            if processor is not None:
                block["point_cloud"] = np.empty((count, config.pointcloud.num_points, 6), dtype=np.float32)
            for offset, index in enumerate(range(start, end)):
                image = next(images, None)
                if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
                    raise ValueError(f"{reader.path / 'rgb.mp4'}: RGB missing or invalid at row {index}")
                if "rgb" in fields:
                    block["rgb"][offset] = image
                if processor is not None:
                    cloud = processor.process(raw["depth"][offset], image)
                    if not len(cloud):
                        raise ValueError(f"point_cloud unavailable at row {index}")
                    block["point_cloud"][offset] = cloud
                    del cloud
                del image
        yield block
        del raw, block
    if images is not None and next(images, None) is not None:
        raise ValueError(f"{reader.path / 'rgb.mp4'}: unexpected RGB frame at row {frames}")


def export_zarr(
    input_root: str | Path,
    output_path: str | Path,
    *,
    processing: ExportConfig,
    fields=None,
    exclude: tuple[str, ...] = (),
    progress_callback=None,
) -> dict:
    fields = resolve_fields(fields)
    if "point_cloud" in fields:
        processing.pointcloud.validate()
    if "hand_pointcloud" in fields:
        processing.hand_pointcloud.validate()
    requested_source = Path(input_root).expanduser()
    output = Path(output_path).expanduser()
    target = output.resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {output}")
    selected, excluded = select_episodes(requested_source, exclude)
    staging = None
    store = None
    ends: list[int] = []
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            offset = ends[-1] if ends else 0
            try:
                require_published_raw(episode)
                with RawReader(episode) as reader:
                    camera = validate_export_episode(reader, fields)
                    frames = reader.num_frames
                    logger.info("%s: exporting full episode (%d frames)", episode.name, frames)
                    if index == 0:
                        arm_fk = ArmFK() if ARM_FK_FIELDS.intersection(fields) else None
                        hand_fk = HandFK() if HAND_FK_FIELDS.intersection(fields) else None
                        hand_cloud = (HandPointCloud(cast(HandFK, hand_fk), processing.hand_pointcloud.num_points)
                                      if "hand_pointcloud" in fields else None)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        staging = Path(
                            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                        )
                        store = zarr.DirectoryStore(str(staging))
                        root, arrays = create_store(
                            store,
                            processing,
                            camera,
                            fields=fields,
                            control_hz=reader.control_hz,
                        )
                    else:
                        if "object_pointcloud" in fields and reader.hoi_metadata != root.attrs["hoi_pcd"]:
                            raise ValueError("HOI task/template mismatch; use --exclude")
                        if not np.isclose(reader.dt, root.attrs["dt"], rtol=1e-6, atol=0):
                            raise ValueError("one canonical store requires uniform dt")
                        if "depth" in fields and not depth_scales_match(camera["depth_scale_m"], root.attrs["depth_scale_m_per_unit"]):
                            raise ValueError("depth_scale_m_per_unit must be uniform")
                        for name in ("rgb", "depth"):
                            if name in fields:
                                shape = (*camera["shape"], 3) if name == "rgb" else camera["shape"]
                                if shape != arrays[name].shape[1:]:
                                    raise ValueError(f"inconsistent {name} shape")

                    if "object_pointcloud" in fields and index == 0:
                        root.attrs["hoi_pcd"] = reader.hoi_metadata
                    blocks = iter_canonical_blocks(
                        reader,
                        processing,
                        fields=fields,
                        arm_fk=arm_fk,
                        hand_fk=hand_fk,
                        hand_cloud=hand_cloud,
                        camera=camera,
                    )
                    append_episode(arrays, offset, blocks, frames)
                    ends.append(offset + frames)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Raw {episode}: {exc}") from exc
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.asarray(ends, dtype=np.int64)
        )
        root.store.close()
        store = None
        publish(cast(Path, staging), target)
        staging = None
        if progress_callback:
            progress_callback(len(selected), len(selected))
        return {
            "episode_count": len(ends), "total_frames": ends[-1], "excluded_episodes": sorted(excluded),
        }
    finally:
        if staging is not None:
            try:
                if store is not None:
                    store.close()
            except BaseException:
                logger.exception("export store close also failed")
            try:
                shutil.rmtree(staging)
            except BaseException:
                logger.exception("export staging cleanup also failed: %s", staging)


def create_store(store, processing, camera, *, fields, control_hz):
    specs = canonical_array_specs(fields, processing, camera)
    root = zarr.group(store=store, overwrite=True)
    root.attrs.update(format=CANONICAL_FORMAT, dt=1.0 / control_hz)
    if {"action", "action_ee"}.intersection(fields):
        root.attrs["action_semantics"] = "absolute_joint_goal_not_next_step_arrival"
    if "point_cloud" in fields:
        root.attrs["pointcloud_config"] = asdict(processing.pointcloud)
        if processing.pointcloud.remove_table:
            root.attrs["table_plane_abcd"] = processing.table_plane_abcd
    if "hand_pointcloud" in fields:
        root.attrs["hand_pointcloud_config"] = asdict(processing.hand_pointcloud)
    if "fingertip_xyz" in fields:
        root.attrs["fingertip_link_names"] = list(XHAND_FINGERTIP_LINK_NAMES)
    if "depth" in fields:
        root.attrs["depth_scale_m_per_unit"] = camera["depth_scale_m"]
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


def append_episode(arrays, offset, blocks, frames):
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
                    raise ValueError(f"canonical {name}: nonfinite values at row {row}")
            arrays[name][offset + count : offset + count + rows] = values
        count += rows
        # 同步写入完成后释放消费者引用，避免与下一块分配重叠。
        del block, values
    if count != frames:
        raise ValueError(f"canonical row count mismatch: expected {frames}, wrote {count}")


def select_episodes(source, exclude):
    episodes = discover_episode_dirs(source)
    excluded = set(exclude) & {episode.name for episode in episodes}
    selected = [episode for episode in episodes if episode.name not in excluded]
    if not selected:
        raise ValueError("export has no selected episodes")
    return selected, excluded
