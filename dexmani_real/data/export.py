"""将 Raw 观测与绝对动作转换为训练用 Zarr。"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import zarr

from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.data.raw import ROBOT_DATASET_SPECS, RawReader
from dexmani_real.geometry.camera import (
    CameraSnapshot,
    camera_parameters_cv,
    validate_aligned_depth_distortion,
)
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_poses
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
    for name in ROBOT_DATASET_SPECS:
        reader[name]
    return camera, training_prefix(reader)


def training_prefix(reader: RawReader) -> dict:
    """只接纳从第零行开始的真实连续片段，不重采样或改写来源时钟。"""
    status = reader.meta["technical_status"]
    if status not in {"normal", "failed"}:
        raise ValueError(f"unknown Raw technical_status: {status}")
    rows = reader["timing/rows"][:]
    count = len(rows)
    period, tolerance = sampling_clock_ns(reader.control_hz)
    # Raw 未逐行记录提交成功；技术失败的末行不能推断为已成功提交。
    end = count - 1 if status == "failed" else count
    reason = "technical_failure_last_row" if status == "failed" else None

    def cut_at(bad, detail, offset=0):
        nonlocal end, reason
        indices = np.flatnonzero(bad)
        if len(indices) and int(indices[0]) + offset <= end:
            end, reason = int(indices[0]) + offset, detail

    control, observation = rows["control_ns"], rows["observation_ns"]
    cut_at((control <= 0) | (observation < control), "invalid_control_or_observation_clock")
    cut_at(np.diff(observation) <= 0, "observation_clock_not_increasing", 1)
    grid_error = control - int(control[0]) - np.arange(count) * period
    cut_at(np.abs(np.diff(control) - period) > tolerance, "control_interval_gap_or_drift", 1)
    cut_at(np.abs(grid_error) > tolerance, "control_grid_drift")
    readout = observation - control
    cut_at(readout >= period, "observation_crossed_control_period")
    cut_at(observation[:-1] >= control[1:], "overlapping_control_observations")

    for name in ("arm_ns", "hand_ns", "tactile_aggregate_ns", "tactile_dense_ns", "camera_ns"):
        values = rows[name]
        cut_at((values <= 0) | (values > observation), f"invalid_source_clock:{name}")
        cut_at(np.diff(values) < 0, f"source_clock_regressed:{name}", 1)
    sequence = rows["camera_sequence"]
    cut_at(sequence < 0, "invalid_camera_sequence")
    cut_at(np.diff(sequence) < 0, "camera_sequence_regressed", 1)
    tactile_age = int(reader.validate_tactile_metadata() * 1e9)
    for name in ("tactile_aggregate_ns", "tactile_dense_ns"):
        cut_at(observation - rows[name] >= tactile_age, f"stale_source:{name}")

    # 结构已整段验证；数值只需扫描可能保留的前缀，不读整段 RGB-D 到内存。
    for start in range(0, end, 4096):
        if start >= end:
            break
        for name in ROBOT_DATASET_SPECS:
            values = reader[name][start:min(end, start + 4096)]
            valid = np.isfinite(values).reshape(len(values), -1).all(axis=1)
            cut_at(~valid, f"nonfinite:{name}", start)
            if end == start:
                break
    if end <= 0:
        raise ValueError(f"Raw {reader.path}: no valid training prefix ({reason})")
    return dict(
        raw_frames=count, row_start=0, row_stop=end,
        source_technical_status=status, trim_reason=reason,
        max_control_grid_deviation_ns=int(np.abs(grid_error[:end]).max()),
        max_readout_ns=int(readout[:end].max()),
        readout_over_tolerance_rows=int(np.count_nonzero(readout[:end] > tolerance)),
    )


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """枚举所有 episode 目录，损坏文件由读取边界直接报错。"""
    requested_path = Path(input_root)
    root = requested_path.resolve()
    if requested_path.name.startswith(".tmp_") or root.name.startswith(".tmp_"):
        raise ValueError(f"unpublished Raw episode is not readable: {root}")
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (root / "data.h5").is_file():
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
    frames: int,
):
    """分块转换，不修补缺失观测或重采样行。"""
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
                "action_ee": np.concatenate((action_poses, hand_action), axis=1, dtype=np.float32),
                "eef_pose": poses.astype(np.float32),
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


def export_zarr(
    input_root: str | Path,
    output_path: str | Path,
    *,
    processing: ExportConfig,
    exclude: tuple[str, ...] = (),
    progress_callback=None,
) -> dict:
    source = Path(input_root).expanduser().resolve()
    output = Path(output_path).expanduser()
    target = output.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("output and Raw input must not contain one another")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {output}")
    selected, excluded = select_episodes(source, exclude)
    arm_fk, hand_fk = ArmFK(), HandFK()
    hand_cloud = HandPointCloud(hand_fk, processing.hand_pointcloud.num_points)
    staging = None
    ends: list[int] = []
    episode_ids: list[str] = []
    episode_quality = []
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            offset = ends[-1] if ends else 0
            with RawReader(episode) as reader:
                camera, quality = validate_export_episode(reader)
                frames = quality["row_stop"]
                logger.info("%s: training rows [0, %d)/%d; discarded=%d; status=%s; trim=%s; readout over dt/4 rows=%d",
                            episode.name, frames, reader.num_frames, reader.num_frames - frames,
                            quality["source_technical_status"], quality["trim_reason"],
                            quality["readout_over_tolerance_rows"])
                task = reader.meta["task_label"]
                if not task.strip() or any(ord(char) < 32 or ord(char) == 127 for char in task):
                    raise ValueError("task_label must be nonempty without control characters")
                height, width = camera.color.height, camera.color.width
                if index == 0:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staging = Path(
                        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                    )
                    root, arrays, chunk = create_store(
                        staging,
                        processing,
                        camera,
                        task=task,
                        control_hz=reader.control_hz,
                    )
                if task != root.attrs["task_name"] or not np.isclose(
                    reader.dt, root.attrs["dt"], rtol=0, atol=1e-12
                ):
                    raise ValueError("one canonical store requires uniform task and dt")
                if camera.depth_scale_m != root.attrs["depth_scale_m_per_unit"]:
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
                    frames=frames,
                )
                append_episode(arrays, offset, blocks, frames)
                ends.append(offset + frames)
                episode_quality.append(quality)
            episode_ids.append(
                episode.name if episode == source else episode.relative_to(source).as_posix()
            )
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.asarray(ends, dtype=np.int64)
        )
        root.attrs["episode_ids"] = episode_ids
        root.attrs["episode_quality"] = episode_quality
        root.attrs["source_id"] = str(uuid4())
        publish(cast(Path, staging), target)
        staging = None
        if progress_callback:
            progress_callback(len(selected), len(selected))
        return dict(
            episode_count=len(ends), total_frames=ends[-1], excluded_episodes=sorted(excluded),
            episode_quality=episode_quality,
        )
    except Exception:
        logger.exception("Export failed at %s", episode)
        raise
    finally:
        if staging is not None:
            logger.warning("Export staging retained: %s", staging)


def create_store(path, processing, camera, *, task, control_hz):
    height, width = camera.color.height, camera.color.width
    specs = canonical_array_specs(
        processing.pointcloud.num_points,
        height,
        width,
        processing.hand_pointcloud.num_points,
    )
    row_bytes = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in specs.values())
    chunk = min(MAX_BATCH_FRAMES, max(1, PROCESSING_BYTES // row_bytes))
    period_ns, tolerance_ns = sampling_clock_ns(control_hz)
    root = zarr.open_group(str(path), mode="w")
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
        timing_admission=dict(
            rule="continuous_control_prefix_with_bounded_readout",
            period_ns=period_ns,
            control_tolerance_ns=tolerance_ns,
            control_clock="control_ns",
            observation_clock="observation_ns",
            readout_limit_ns=period_ns,
            resampled=False,
            failed_capture_tail="exclude_last_row",
        ),
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
            # 两份触觉 float32 原样拷贝，已由 training_prefix 扫描；转换和新计算仍检查。
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
