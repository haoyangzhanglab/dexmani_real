"""将 Raw 观测与绝对动作转换为训练用 Zarr。"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import zarr

from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.data.raw import (
    ROBOT_DATASET_SPECS,
    RawReader,
    require_published_raw,
    validate_task_label,
)
from dexmani_real.geometry.camera import (
    CameraSnapshot,
    camera_parameters_cv,
    depth_scales_match,
    validate_aligned_depth_distortion,
)
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_poses
from dexmani_real.geometry.pose import encode_eef_pose
from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
from dexmani_real.perception.pointcloud import PointCloudProcessor
from dexmani_real.robot.model import (
    XARM7_XHAND_RIGHT_URDF_PATH,
    XHAND_FINGERTIP_LINK_NAMES,
)
from dexmani_real.utils.io import publish
from dexmani_real.utils.timing import sampling_clock_ns

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


def canonical_array_specs(
    num_points: int,
    rgb_height: int,
    rgb_width: int,
    hand_num_points: int,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """公共导出的固定 13 字段及逐行形状、类型。"""
    return {
        "joint_state": ((19,), np.dtype(np.float32)),
        "action": ((19,), np.dtype(np.float32)),
        "action_ee": ((21,), np.dtype(np.float32)),
        "finger_tactile_force": ((5, 3), np.dtype(np.float32)),
        "taxel_tactile_force": ((5, 120, 3), np.dtype(np.float32)),
        "fingertip_xyz": ((5, 3), np.dtype(np.float32)),
        "eef_pose": ((9,), np.dtype(np.float32)),
        "camera_intrinsic": ((9,), np.dtype(np.float32)),
        "camera_extrinsic": ((12,), np.dtype(np.float32)),
        "rgb": ((rgb_height, rgb_width, 3), np.dtype(np.uint8)),
        "depth": ((rgb_height, rgb_width), np.dtype(np.uint16)),
        "point_cloud": ((num_points, 6), np.dtype(np.float32)),
        "hand_pointcloud": ((hand_num_points, 6), np.dtype(np.float32)),
    }


def validate_export_episode(reader: RawReader) -> tuple[CameraSnapshot, dict]:
    """前置验证元数据、结构和时间；数值及 RGB 在流式转换时验证。"""
    validate_task_label(reader.meta["task_label"])
    if reader.meta["collection_source"] != "teleop":
        raise ValueError(
            "canonical demonstration export currently supports only teleop Raw; "
            f"got collection_source={reader.meta['collection_source']!r}"
        )
    frames = reader.num_frames
    if frames <= 0:
        raise ValueError("episode contains no rows")
    reader.require_fields(*ROBOT_DATASET_SPECS, "rgb", "depth")
    camera = reader.read_camera()
    validate_aligned_depth_distortion(camera.color.distortion_model)
    if camera.T_xarm_base_from_color is None:
        raise ValueError("canonical export requires camera extrinsics")
    color = camera.color
    expected = (frames, color.height, color.width)
    if reader["depth"].shape != expected or reader["depth"].dtype != np.uint16:
        raise ValueError(f"depth must be uint16 {expected}")
    if frames == 1:
        logger.warning(
            "%s: single row, no sampling interval to verify", reader.path
        )
    rows = reader["timing/rows"][:]
    period = sampling_clock_ns(reader.control_hz)
    tolerance = period // 4

    def reject_invalid(bad, detail, offset=0):
        indices = np.flatnonzero(bad)
        if len(indices):
            row = int(indices[0]) + offset
            raise ValueError(f"{detail} at row {row}")

    control, observation = rows["control_ns"], rows["observation_ns"]
    reject_invalid((control <= 0) | (observation < control), "invalid_control_or_observation_clock")
    grid_error = control - int(control[0]) - np.arange(frames) * period
    reject_invalid(np.diff(control) <= 0, "control_clock_not_increasing", 1)
    reject_invalid(np.diff(observation) <= 0, "observation_clock_not_increasing", 1)
    if np.any(np.abs(np.diff(control) - period) > tolerance):
        logger.warning("%s: control interval jitter/gaps; 不重采样，实际时间保留于 Raw", reader.path)
    mean_period = None
    if frames >= 2:
        duration = int(control[-1]) - int(control[0])
        mean_period = duration / (frames - 1)
    readout = observation - control
    if np.any(observation[:-1] >= control[1:]):
        logger.warning("%s: observation readout overlaps the next control sample", reader.path)

    for name in ("arm_ns", "hand_ns", "tactile_aggregate_ns", "tactile_dense_ns", "camera_ns"):
        values = rows[name]
        reject_invalid((values <= 0) | (values > observation), f"invalid_source_clock:{name}")
        reject_invalid(np.diff(values) < 0, f"source_clock_regressed:{name}", 1)
    sequence = rows["camera_sequence"]
    reject_invalid(sequence < 0, "invalid_camera_sequence")
    reject_invalid(np.diff(sequence) < 0, "camera_sequence_regressed", 1)
    # 访问 dataset 触发 RawReader 的结构/dtype 校验，不读取数值切片。
    for name in ROBOT_DATASET_SPECS:
        reader[name]
    return camera, dict(
        max_control_grid_deviation_ns=int(np.abs(grid_error).max()),
        max_readout_ns=int(readout.max()),
        max_tactile_aggregate_age_ns=int((observation - rows["tactile_aggregate_ns"]).max()),
        max_tactile_dense_age_ns=int((observation - rows["tactile_dense_ns"]).max()),
        mean_control_period_ns=mean_period,
        max_control_interval_ns=int(np.diff(control).max()) if frames > 1 else None,
        termination_reason=reader.meta.get("termination_reason"),
        recording_truncated_reason=reader.meta.get("recording_truncated_reason"),
    )


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
    arm_fk,
    hand_fk,
    hand_cloud,
    chunk_frames: int,
    camera: CameraSnapshot,
):
    """每块原始数值只读一次，全部通过 finite 检查后转换；坏块拒绝整段。"""
    frames = reader.num_frames
    height, width = camera.color.height, camera.color.width
    processor = PointCloudProcessor(
        depth_scale_m=camera.depth_scale_m,
        intrinsics=camera.color,
        T_xarm_base_from_color=camera.T_xarm_base_from_color,
        table_plane_abcd=config.table_plane_abcd,
        config=config.pointcloud,
    )
    camera_parameters = camera_parameters_cv(camera.color, camera.T_xarm_base_from_color)
    with closing(reader.iter_rgb_frames()) as images:
        for start in range(0, frames, chunk_frames):
            end = min(frames, start + chunk_frames)
            raw = {name: reader[name][start:end] for name in ROBOT_DATASET_SPECS}
            for name, values in raw.items():
                valid = np.isfinite(values).reshape(end - start, -1).all(axis=1)
                if not valid.all():
                    row = start + int(np.flatnonzero(~valid)[0])
                    raise ValueError(f"nonfinite:{name} at row {row}")
            arm, hand = raw["arm_qpos"], raw["hand_qpos"]
            arm_action, hand_action = (
                raw["action_arm_joint_target"],
                raw["action_hand_joint_target"],
            )
            poses = eef_poses(arm, arm_fk=arm_fk)
            action_poses = eef_poses(arm_action, arm_fk=arm_fk)
            hand_features = hand_geometry(
                poses,
                hand,
                arm_fk=arm_fk,
                hand_fk=hand_fk,
                hand_cloud=hand_cloud,
            )
            block = {
                "joint_state": np.concatenate((arm, hand), axis=1, dtype=np.float32),
                "action": np.concatenate((arm_action, hand_action), axis=1, dtype=np.float32),
                "action_ee": np.concatenate((encode_eef_pose(action_poses), hand_action), axis=1, dtype=np.float32),
                "eef_pose": encode_eef_pose(poses).astype(np.float32),
                "finger_tactile_force": raw["hand_contact"],
                "taxel_tactile_force": raw["hand_tactile_force"],
                **hand_features,
            }
            for name, values in camera_parameters.items():
                block[name] = np.broadcast_to(values, (end - start, values.size))
            # 深度按块读取，RGB 逐帧解码到输出数组。
            rgb = np.empty((end - start, height, width, 3), dtype=np.uint8)
            depth = reader["depth"][start:end]
            clouds = np.empty((end - start, config.pointcloud.num_points, 6), dtype=np.float32)
            for offset, index in enumerate(range(start, end)):
                image = next(images, None)
                if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
                    raise ValueError(
                        f"{reader.path / 'rgb.mp4'}: RGB missing or invalid at row {index}"
                    )
                cloud = processor.process(depth[offset], image)
                if not len(cloud):
                    raise ValueError(f"point_cloud unavailable at row {index}")
                rgb[offset], clouds[offset] = image, cloud
            block.update(rgb=rgb, depth=depth, point_cloud=clouds)
            yield block
        if next(images, None) is not None:
            raise ValueError(f"{reader.path / 'rgb.mp4'}: unexpected RGB frame at row {frames}")


def export_zarr(
    input_root: str | Path,
    output_path: str | Path,
    *,
    processing: ExportConfig,
    exclude: tuple[str, ...] = (),
    progress_callback=None,
) -> dict:
    # Canonical 始终导出两种点云，须在按点数创建存储前接纳实际启用的配置。
    processing.pointcloud.validate()
    processing.hand_pointcloud.validate()
    requested_source = Path(input_root).expanduser()
    source = requested_source.resolve()
    output = Path(output_path).expanduser()
    target = output.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("output and Raw input must not contain one another")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {output}")
    selected, excluded = select_episodes(requested_source, exclude)
    staging = None
    store = None
    ends: list[int] = []
    episode_ids: list[str] = []
    episode_quality = []
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            offset = ends[-1] if ends else 0
            try:
                require_published_raw(episode)
                with RawReader(episode) as reader:
                    camera, quality = validate_export_episode(reader)
                    frames = reader.num_frames
                    logger.info("%s: exporting full episode (%d frames)", episode.name, frames)
                    task = reader.meta["task_label"]
                    height, width = camera.color.height, camera.color.width
                    if index == 0:
                        arm_fk, hand_fk = ArmFK(), HandFK()
                        hand_cloud = HandPointCloud(hand_fk, processing.hand_pointcloud.num_points)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        staging = Path(
                            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                        )
                        store = zarr.DirectoryStore(str(staging))
                        root, arrays, chunk = create_store(
                            store,
                            processing,
                            camera,
                            task=task,
                            control_hz=reader.control_hz,
                        )
                    else:
                        if task != root.attrs["task_name"] or not np.isclose(
                            reader.dt, root.attrs["dt"], rtol=0, atol=1e-12
                        ):
                            raise ValueError("one canonical store requires uniform task and dt")
                        if not depth_scales_match(camera.depth_scale_m, root.attrs["depth_scale_m_per_unit"]):
                            raise ValueError("depth_scale_m_per_unit must be uniform")
                        if (height, width, 3) != arrays["rgb"].shape[1:]:
                            raise ValueError("inconsistent RGB shape")

                    blocks = iter_canonical_blocks(
                        reader,
                        processing,
                        arm_fk=arm_fk,
                        hand_fk=hand_fk,
                        hand_cloud=hand_cloud,
                        chunk_frames=chunk,
                        camera=camera,
                    )
                    append_episode(arrays, offset, blocks, frames)
                    ends.append(offset + frames)
                    episode_quality.append(quality)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Raw {episode}: {exc}") from exc
            except Exception:
                logger.exception("Export failed at %s", episode)
                raise
            episode_ids.append(
                episode.name if episode == source else episode.relative_to(source).as_posix()
            )
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.asarray(ends, dtype=np.int64)
        )
        root.attrs["episode_ids"] = episode_ids
        root.attrs["episode_quality"] = episode_quality
        root.attrs["source_id"] = str(uuid4())
        root.store.close()
        store = None
        publish(cast(Path, staging), target)
        staging = None
        if progress_callback:
            progress_callback(len(selected), len(selected))
        return dict(
            episode_count=len(ends), total_frames=ends[-1], excluded_episodes=sorted(excluded),
            episode_quality=episode_quality,
        )
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


def create_store(store, processing, camera, *, task, control_hz):
    height, width = camera.color.height, camera.color.width
    specs = canonical_array_specs(
        processing.pointcloud.num_points,
        height,
        width,
        processing.hand_pointcloud.num_points,
    )
    row_bytes = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in specs.values())
    chunk = min(MAX_BATCH_FRAMES, max(1, PROCESSING_BYTES // row_bytes))
    root = zarr.group(store=store, overwrite=True)
    root.attrs.update(
        format=CANONICAL_FORMAT,
        task_name=task,
        dt=1.0 / control_hz,
        action_semantics="absolute_joint_goal_not_next_step_arrival",
        pointcloud_config=asdict(processing.pointcloud),
        hand_pointcloud_config=asdict(processing.hand_pointcloud),
        fingertip_link_names=list(XHAND_FINGERTIP_LINK_NAMES),
        kinematics_urdf=XARM7_XHAND_RIGHT_URDF_PATH.read_text(encoding="utf-8"),
        table_plane_abcd=processing.table_plane_abcd,
        depth_scale_m_per_unit=camera.depth_scale_m,
    )
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
    return root, arrays, chunk


def append_episode(arrays, offset, blocks, frames):
    for array in arrays.values():
        array.resize((offset + frames, *array.shape[1:]))
    count = 0
    for block in blocks:
        rows = len(block["joint_state"])
        for name, values in block.items():
            # 两份触觉 float32 原样拷贝，已由原始块检查；转换和新计算仍检查。
            if np.issubdtype(values.dtype, np.floating) and name not in {
                "finger_tactile_force", "taxel_tactile_force",
            }:
                valid = np.isfinite(values).reshape(rows, -1).all(axis=1)
                if not valid.all():
                    row = count + int(np.flatnonzero(~valid)[0])
                    raise ValueError(f"canonical {name}: nonfinite values at row {row}")
            arrays[name][offset + count : offset + count + rows] = values
        count += rows
    if count != frames:
        raise ValueError(f"canonical row count mismatch: expected {frames}, wrote {count}")


def select_episodes(source, exclude):
    episodes = discover_episode_dirs(source)
    excluded = set(exclude)
    unknown = excluded - {episode.name for episode in episodes}
    if unknown:
        raise ValueError(f"exclusions reference unknown episodes: {sorted(unknown)}")
    selected = [episode for episode in episodes if episode.name not in excluded]
    if not selected:
        raise ValueError("export has no selected episodes")
    return selected, excluded
